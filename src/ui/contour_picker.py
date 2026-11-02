from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.tri as mtri
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QVBoxLayout
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import polygonize
import geopandas as gpd

from src.config import AppConfig
from src.core.field_interpolation import gaussian_knn_smooth
from src.core.pipeline_mapping import save_selected_lines_from_contour
from src.core.workflow import prepare_step3_contours
from src.ui.plotting import load_line_segments, safe_add_basemap


LogFn = Callable[[str], None]


def run_contour_picker(
    config: AppConfig,
    pipeline_state: Dict,
    *,
    log: Optional[LogFn] = None,
    overlay_cache: Optional[Dict] = None,
    parent=None,
) -> bool:
    """
    Interactive tricontour picker. Left-click saves contour GIS immediately.

    Mutates pipeline_state with selected_polygon_override and shapefile paths.
    Returns True if GIS was saved.
    """
    def _log(msg: str):
        if log:
            log(msg)

    original_save_keys = list(matplotlib.rcParams.get("keymap.save", []))
    try:
        matplotlib.rcParams["keymap.save"] = [k for k in original_save_keys if str(k).lower() != "s"]

        prep = prepare_step3_contours(config, pipeline_state)
        step2 = pipeline_state.get("step2")
        if step2 is None:
            raise RuntimeError("Step2 result is missing.")
        day_idx = int(prep["day_idx"])
        dir_idx = int(prep["dir_idx"])
        vals = np.asarray(prep["values"], dtype=float)
        xs = np.asarray(prep["coords"][:, 0], dtype=float)
        ys = np.asarray(prep["coords"][:, 1], dtype=float)
        if bool(getattr(config, "field_smooth_enable", True)) and vals.size >= 3:
            vals = gaussian_knn_smooth(
                xs,
                ys,
                vals,
                k_neighbors=int(config.k_neighbors),
                bandwidth_m=float(getattr(config, "gauss_bandwidth_m", 20.0)),
            )
        if vals.size < 3:
            raise RuntimeError("Insufficient points for contour selection.")

        mode = str(config.filter_mode).lower()
        threshold = float(config.contour_threshold)
        zmin = float(np.nanmin(vals))
        zmax = float(np.nanmax(vals))
        lo, hi = (threshold, zmax) if mode == "above" else (zmin, threshold)
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or abs(hi - lo) < 1e-9:
            raise RuntimeError("No contour polygon candidates found.")
        level_values = np.linspace(lo, hi, 30)
        cmap = "coolwarm" if mode == "above" else "coolwarm_r"
        norm = mcolors.Normalize(vmin=float(np.min(level_values)), vmax=float(np.max(level_values)))

        fig = Figure(figsize=(8, 6), tight_layout=True)
        ax = fig.add_subplot(111)
        tri = mtri.Triangulation(xs, ys)
        ax.tricontourf(tri, vals, levels=level_values, cmap=cmap, norm=norm, zorder=5, alpha=0.7)
        cs = ax.tricontour(tri, vals, levels=level_values, colors="none", linewidths=0.0, zorder=6)
        mask = vals >= threshold if mode == "above" else vals <= threshold
        sc = ax.scatter(xs[mask], ys[mask], c=vals[mask], s=2, cmap=cmap, norm=norm, linewidths=0, alpha=0.6, zorder=7)

        cache = overlay_cache if overlay_cache is not None else {}
        pipe_segments = load_line_segments(config, config.pipe_axis_path, "pipe", cache)
        if pipe_segments:
            ax.add_collection(LineCollection(pipe_segments, colors="gray", linewidths=1.2, zorder=8))
        fault_segments = load_line_segments(config, config.fault_line_shp_path, "fault", cache)
        if fault_segments:
            ax.add_collection(LineCollection(fault_segments, colors="red", linewidths=1.8, alpha=0.8, zorder=9))
        cbar = fig.colorbar(sc, ax=ax)
        dir_names = ["North", "East", "Vertical"]
        cbar.set_label(f"Displacement-{dir_names[dir_idx]} (mm)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Left click: select contour and save | Right click: deselect | 'r': reset | close window to continue")
        safe_add_basemap(ax, config)

        level_polygons = []
        for segs_at_level in cs.allsegs:
            line_strings = []
            for seg in segs_at_level:
                if seg is None or len(seg) < 2:
                    continue
                try:
                    line_strings.append(LineString(seg))
                except Exception:
                    continue
            polygons = list(polygonize(line_strings)) if line_strings else []
            level_polygons.append(polygons)

        gdf_lines = gpd.read_file(config.resolve_path(config.pipe_axis_path))
        if gdf_lines.crs is None:
            gdf_lines = gdf_lines.set_crs(epsg=int(config.dst_epsg))
        gdf_lines = gdf_lines.to_crs(epsg=int(config.dst_epsg))
        gdf_lines = gdf_lines[gdf_lines.geometry.notnull()]
        gdf_lines = gdf_lines[gdf_lines.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()

        selected = {"poly": None}
        saved = {"ok": False, "info": None}
        selected_artist = {"line": None}
        inside_lc = {"line": None}
        intersect_lc = {"line": None}
        poly_patches = []
        dialog_holder = {"dlg": None}

        def _pipe_seg_builder(geoms):
            segs = []
            for g in geoms:
                if isinstance(g, LineString):
                    xg, yg = g.xy
                    segs.append(np.column_stack([xg, yg]))
                elif isinstance(g, MultiLineString):
                    for part in g.geoms:
                        xg, yg = part.xy
                        segs.append(np.column_stack([xg, yg]))
            return segs

        def _build_segments(idx_list):
            if gdf_lines.empty or len(idx_list) == 0:
                return []
            geoms = gdf_lines.loc[idx_list, "geometry"].values
            return _pipe_seg_builder(geoms)

        def _classify_lines(poly):
            if poly is None or poly.is_empty or gdf_lines.empty:
                return [], []
            poly_buf = poly.buffer(1e-6)
            try:
                cand_idx = list(gdf_lines.sindex.query(poly_buf, predicate="intersects"))
                cand = gdf_lines.iloc[cand_idx]
            except Exception:
                cand = gdf_lines[gdf_lines.intersects(poly_buf)]
            inside_mask = cand.covered_by(poly_buf)
            intersect_mask = cand.intersects(poly_buf) & (~inside_mask)
            return cand.index[inside_mask].tolist(), cand.index[intersect_mask].tolist()

        def _find_nearest_level(pt: Point):
            best = (None, float("inf"))
            for lvl, segs_at_level in enumerate(cs.allsegs):
                for seg in segs_at_level:
                    try:
                        d = LineString(seg).distance(pt)
                        if d < best[1]:
                            best = (lvl, d)
                    except Exception:
                        continue
            return best[0]

        def _choose_polygon(level_idx: int, pt: Point):
            polys = level_polygons[level_idx] if 0 <= level_idx < len(level_polygons) else []
            for poly in polys:
                if poly.is_valid and poly.contains(pt):
                    return poly
            dmin, chosen = float("inf"), None
            for poly in polys:
                try:
                    d = poly.boundary.distance(pt)
                    if d < dmin:
                        dmin, chosen = d, poly
                except Exception:
                    continue
            return chosen

        def _clear_selected_draw():
            if selected_artist["line"] is not None:
                try:
                    selected_artist["line"].remove()
                except Exception:
                    pass
                selected_artist["line"] = None
            if inside_lc["line"] is not None:
                try:
                    inside_lc["line"].remove()
                except Exception:
                    pass
                inside_lc["line"] = None
            if intersect_lc["line"] is not None:
                try:
                    intersect_lc["line"].remove()
                except Exception:
                    pass
                intersect_lc["line"] = None
            for patch in poly_patches:
                try:
                    patch.remove()
                except Exception:
                    pass
            poly_patches.clear()

        def _draw_polygon_outline(poly):
            if poly is None or poly.is_empty:
                return

            def _add(exterior):
                patch = MplPolygon(np.asarray(exterior.coords), facecolor="none", edgecolor="yellow", linewidth=2.0, zorder=10)
                ax.add_patch(patch)
                poly_patches.append(patch)

            if poly.geom_type == "Polygon":
                _add(poly.exterior)
            else:
                for p in poly.geoms:
                    _add(p.exterior)

        def _on_click(event):
            if event.inaxes is not ax or event.xdata is None or event.ydata is None:
                return
            if event.button == 3:
                selected["poly"] = None
                _clear_selected_draw()
                fig.canvas.draw_idle()
                return
            if event.button != 1:
                return
            click_point = Point(float(event.xdata), float(event.ydata))
            lvl = _find_nearest_level(click_point)
            if lvl is None:
                _log("No contour level found near click.")
                return
            selected["poly"] = _choose_polygon(lvl, click_point)
            if selected["poly"] is None or selected["poly"].is_empty:
                _log(f"No polygon at level index {lvl}.")
                return
            inside_idx, intersect_idx = _classify_lines(selected["poly"])
            _log(f"[Pick] level={lvl}, inside={len(inside_idx)}, intersect={len(intersect_idx)}")
            _clear_selected_draw()
            _draw_polygon_outline(selected["poly"])
            if inside_idx:
                inside_lc["line"] = LineCollection(_build_segments(inside_idx), colors="purple", linewidths=3.0, zorder=11)
                ax.add_collection(inside_lc["line"])
            if intersect_idx:
                intersect_lc["line"] = LineCollection(_build_segments(intersect_idx), colors="purple", linewidths=3.0, zorder=11)
                ax.add_collection(intersect_lc["line"])
            try:
                out_info = save_selected_lines_from_contour(
                    config=config,
                    selected_polygon=selected["poly"],
                    output_dir=Path(config.resolve_path(config.output_dir)),
                    out_name=str(config.selected_lines_shp_name),
                )
                pipeline_state["selected_polygon_override"] = selected["poly"]
                pipeline_state["step3_selected_lines_path"] = out_info["selected_lines_path"]
                pipeline_state["step3_selected_contour_path"] = out_info["selected_contour_path"]
                saved["ok"] = True
                saved["info"] = out_info
                _log(
                    "Contour saved on click: "
                    f"level={int(lvl)}, "
                    f"inside={int(out_info['inside_count'])}, "
                    f"intersect={int(out_info['intersect_count'])}, "
                    f"union={int(out_info['union_count'])}, "
                    f"path={out_info['selected_lines_path']}"
                )
            except Exception as e:
                saved["ok"] = False
                saved["info"] = None
                _log(f"Contour selected but failed to save GIS: {e}")
            fig.canvas.draw_idle()

        def _on_key(event):
            key = getattr(event, "key", None)
            if key == "r":
                selected["poly"] = None
                _clear_selected_draw()
                fig.canvas.draw_idle()
            elif key in ("d", "escape"):
                dlg = dialog_holder["dlg"]
                if dlg is not None:
                    dlg.close()

        host = parent if parent is not None else QApplication.activeWindow()
        dialog = QDialog(host)
        dialog_holder["dlg"] = dialog
        dialog.setWindowTitle("Step 3-1: Select Landslide Area")
        dialog.resize(980, 760)
        dialog.setWindowModality(Qt.ApplicationModal)
        layout = QVBoxLayout(dialog)
        hint = QLabel(
            "Left click: select and save contour GIS. "
            "Right click: deselect. Press 'r' to reset. "
            "Close this window after saving to continue Step 3-2 / 3-3."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        canvas = FigureCanvas(fig)
        canvas.setFocusPolicy(Qt.StrongFocus)
        layout.addWidget(NavigationToolbar(canvas, dialog))
        layout.addWidget(canvas)
        canvas.mpl_connect("button_press_event", _on_click)
        canvas.mpl_connect("key_press_event", _on_key)
        canvas.setFocus()
        dialog.exec()
        if not saved["ok"]:
            _log("No contour GIS was saved. Please left-click a contour to save it first.")
            return False
        poly = pipeline_state.get("selected_polygon_override")
        c = poly.centroid if poly is not None else None
        if c is not None:
            _log(f"Contour ready for Step3-2/3: centroid=({c.x:.2f}, {c.y:.2f})")
        return True
    finally:
        matplotlib.rcParams["keymap.save"] = original_save_keys
