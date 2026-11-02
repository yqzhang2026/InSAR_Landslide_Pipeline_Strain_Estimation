from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.tri as mtri
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PySide6.QtWidgets import QVBoxLayout, QWidget
from scipy.spatial import cKDTree
import geopandas as gpd

from src.core.pipeline_strain import compute_bending_strain_3d

try:
    import contextily as cx
except Exception:
    cx = None


class MplTab(QWidget):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(5, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar(self.canvas, self)
        lay = QVBoxLayout(self)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)
        self.setObjectName(title)


def load_line_segments(config, shp_path: str, cache_key: str, overlay_cache: Dict) -> list:
    if overlay_cache.get(cache_key) is not None:
        return overlay_cache[cache_key]
    segments = []
    try:
        gdf = gpd.read_file(config.resolve_path(shp_path))
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=int(config.dst_epsg))
        gdf = gdf.to_crs(epsg=int(config.dst_epsg))
        for g in gdf.geometry:
            if g is None:
                continue
            gt = getattr(g, "geom_type", "")
            if gt == "LineString":
                x, y = g.xy
                segments.append(np.column_stack([x, y]))
            elif gt == "MultiLineString":
                for part in g.geoms:
                    x, y = part.xy
                    segments.append(np.column_stack([x, y]))
    except Exception:
        segments = []
    overlay_cache[cache_key] = segments
    return segments


_OSM_DE_URL = "https://tile.openstreetmap.de/{z}/{x}/{y}.png"
_OSM_ATTRIBUTION = "© OpenStreetMap contributors"
_ESRI_ATTRIBUTION = "© Esri"
_ESRI_TILE_URLS = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
)


def _basemap_sources():
    sources = []
    if cx is not None:
        try:
            sources.append((getattr(cx.providers.OpenStreetMap, "DE"), _OSM_ATTRIBUTION))
        except Exception:
            pass
    sources.append((_OSM_DE_URL, _OSM_ATTRIBUTION))
    if cx is not None:
        for name in ("WorldStreetMap", "WorldTopoMap", "WorldImagery"):
            try:
                sources.append((getattr(cx.providers.Esri, name), _ESRI_ATTRIBUTION))
            except Exception:
                continue
    for url in _ESRI_TILE_URLS:
        sources.append((url, _ESRI_ATTRIBUTION))
    return sources


def safe_add_basemap(ax, config, alpha: float = 0.55) -> None:
    if cx is None:
        return
    crs = f"EPSG:{int(config.dst_epsg)}"
    for source, attribution in _basemap_sources():
        try:
            cx.add_basemap(ax, crs=crs, source=source, alpha=alpha, attribution=attribution)
            return
        except Exception:
            continue


def _set_or_update_cbar(holder: Dict, key: str, figure, ax, mappable, label: str):
    cbar = holder.get(key)
    if cbar is None:
        holder[key] = figure.colorbar(mappable, ax=ax, shrink=0.8)
    else:
        try:
            cbar.update_normal(mappable)
        except Exception:
            holder[key] = figure.colorbar(mappable, ax=ax, shrink=0.8)
    holder[key].set_label(label)


def _geometry_line_segments(gdf) -> list:
    segments = []
    if gdf is None:
        return segments
    try:
        for g in gdf.geometry:
            if g is None:
                continue
            gt = getattr(g, "geom_type", "")
            if gt == "LineString":
                x, y = g.xy
                segments.append(np.column_stack([x, y]))
            elif gt == "MultiLineString":
                for part in g.geoms:
                    x, y = part.xy
                    segments.append(np.column_stack([x, y]))
    except Exception:
        return []
    return segments


def _apply_axis_limits(ax, x_min, x_max, y_min, y_max) -> None:
    if x_min is not None and x_max is not None:
        ax.set_xlim(x_min, x_max)
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_min, y_max)


def _fit_xy_limits(ax, xs, ys, pad_frac: float = 0.05) -> None:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size == 0 or ys.size == 0:
        return
    x0, x1 = float(np.nanmin(xs)), float(np.nanmax(xs))
    y0, y1 = float(np.nanmin(ys)), float(np.nanmax(ys))
    if not np.isfinite([x0, x1, y0, y1]).all():
        return
    dx = max(x1 - x0, 1.0)
    dy = max(y1 - y0, 1.0)
    ax.set_xlim(x0 - pad_frac * dx, x1 + pad_frac * dx)
    ax.set_ylim(y0 - pad_frac * dy, y1 + pad_frac * dy)


_MATERIAL_COLORS = {
    "Cast Iron": "#1e3a8a",
    "Concrete": "#e67e22",
    "Non-metallic": "#ee6b8a",
    "Steel": "#f4d03f",
}
_MATERIAL_CODE_MAP = {
    "s": "Steel",
    "steel": "Steel",
    "t": "Steel",
    "c": "Cast Iron",
    "ci": "Cast Iron",
    "cast iron": "Cast Iron",
    "d": "Cast Iron",
    "w": "Cast Iron",
    "n": "Non-metallic",
    "h": "Non-metallic",
    "p": "Non-metallic",
    "plastic": "Non-metallic",
    "k": "Non-metallic",
    "l": "Non-metallic",
    "a": "Concrete",
    "ac": "Concrete",
    "concrete": "Concrete",
    "f": "Concrete",
}
_DIAMETER_INCHES = (2, 4, 6, 8, 12, 16)
_DIAMETER_COLORS = {
    2: "#6a1b9a",
    4: "#0d47a1",
    6: "#00838f",
    8: "#43a047",
    12: "#c0ca33",
    16: "#d4e157",
}


def _placeholder(ax, title: str, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def _pick_column(gdf, candidates) -> Optional[str]:
    cols = {str(c).lower(): c for c in gdf.columns}
    for name in candidates:
        if name.lower() in cols:
            return cols[name.lower()]
    for c in gdf.columns:
        cl = str(c).lower()
        for name in candidates:
            if name.lower() in cl:
                return c
    return None


def _load_selected_pipe_gdf(window, config):
    path = None
    if window.pipeline_state:
        path = window.pipeline_state.get("step3_selected_lines_path")
    cache = window._overlay_cache
    if not path:
        cache["selected_pipe_gdf"] = None
        cache["selected_pipe_gdf_path"] = None
        return None
    path_s = str(path)
    if cache.get("selected_pipe_gdf_path") == path_s and cache.get("selected_pipe_gdf") is not None:
        return cache["selected_pipe_gdf"]
    try:
        gdf = gpd.read_file(path_s)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=int(config.dst_epsg))
        else:
            gdf = gdf.to_crs(epsg=int(config.dst_epsg))
        gdf = gdf[gdf.geometry.notnull()].copy()
        gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    except Exception:
        gdf = None
    cache["selected_pipe_gdf"] = gdf
    cache["selected_pipe_gdf_path"] = path_s
    return gdf


def _iter_line_parts(gdf):
    for _, row in gdf.iterrows():
        g = row.geometry
        if g is None:
            continue
        gt = getattr(g, "geom_type", "")
        if gt == "LineString":
            yield row, np.column_stack(g.xy), g
        elif gt == "MultiLineString":
            for part in g.geoms:
                yield row, np.column_stack(part.xy), part


def _normalize_material(raw) -> str:
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        return "Non-metallic"
    key = str(raw).strip().lower()
    return _MATERIAL_CODE_MAP.get(key, "Non-metallic")


def _normalize_diameter_in(raw) -> Optional[int]:
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val) or val <= 0:
        return None
    if val > 80:
        val = val / 25.4
    nearest = min(_DIAMETER_INCHES, key=lambda d: abs(d - val))
    return int(nearest)


def _normalize_age(raw) -> Optional[float]:
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(val):
        return None
    if val > 1800:
        val = 2020.0 - val
    if val < 0:
        return None
    return float(val)


def _parse_insar_date(raw) -> Optional[datetime]:
    text = str(raw).strip()
    if text.lower().startswith("d"):
        text = text[1:]
    digits = "".join(ch for ch in text if ch.isdigit())[:8]
    if len(digits) < 8:
        return None
    try:
        return datetime.strptime(digits, "%Y%m%d")
    except ValueError:
        return None


def _coverage_n_years(window, config) -> float:
    dates = []
    step1 = (window.pipeline_state or {}).get("step1")
    pairs = getattr(step1, "matched_dates", None) if step1 is not None else None
    if pairs:
        for pair in pairs:
            items = pair[:2] if isinstance(pair, (list, tuple)) else [pair]
            for item in items:
                parsed = _parse_insar_date(item)
                if parsed is not None:
                    dates.append(parsed)
    if len(dates) >= 2:
        span_days = float((max(dates) - min(dates)).days)
        if span_days > 0:
            return span_days / 365.0
    span = max(float(getattr(config, "annualize_span_days", 3040.0)), 1e-9)
    year = float(getattr(config, "annualize_days_per_year", 365.0))
    return span / year


def _diameter_m_from_raw(raw, default_m: float) -> float:
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return float(default_m)
    if not np.isfinite(val) or val <= 0:
        return float(default_m)
    if val > 80:
        return float(val) / 1000.0
    if val >= 1.5:
        return float(val) * 0.0254
    return float(val)


def _map_pipe_attrs_to_nodes(gdf, mesh_points: np.ndarray, default_diam_m: float) -> Tuple[np.ndarray, np.ndarray]:
    n = int(mesh_points.shape[0])
    node_age = np.full(n, np.nan, dtype=float)
    node_diam = np.full(n, float(default_diam_m), dtype=float)
    if gdf is None or getattr(gdf, "empty", True):
        return node_age, node_diam
    age_col = _pick_column(gdf, ("Age", "age", "INSTALLATI", "install", "year", "instyear"))
    diam_col = _pick_column(gdf, ("NOMINALDIA", "diameter", "diam", "nom_diam"))
    verts = []
    ages = []
    diams = []
    for row, coords, _geom in _iter_line_parts(gdf):
        if coords.shape[0] < 1:
            continue
        age = _normalize_age(row.get(age_col)) if age_col else None
        diam_m = _diameter_m_from_raw(row.get(diam_col), default_diam_m) if diam_col else float(default_diam_m)
        for pt in coords:
            verts.append(pt)
            ages.append(np.nan if age is None else float(age))
            diams.append(float(diam_m))
    if not verts:
        return node_age, node_diam
    _, idx = cKDTree(np.asarray(verts, dtype=float)).query(np.asarray(mesh_points, dtype=float)[:, :2])
    idx = np.asarray(idx, dtype=int)
    node_age = np.asarray(ages, dtype=float)[idx]
    node_diam = np.asarray(diams, dtype=float)[idx]
    return node_age, node_diam


def _element_ages(node_age: np.ndarray, conn: np.ndarray) -> np.ndarray:
    a0 = node_age[conn[:, 0]]
    a1 = node_age[conn[:, 1]]
    out = np.nanmean(np.column_stack([a0, a1]), axis=1)
    missing = ~np.isfinite(out)
    out[missing] = 1.0
    return out


def _apply_strain_period(values: np.ndarray, period: str, n_years: float, ages: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    key = str(period).strip().lower()
    if key.startswith("data"):
        return values
    yearly = values / max(float(n_years), 1e-9)
    if key.startswith("year"):
        return yearly
    age = np.asarray(ages, dtype=float)
    age = np.where(np.isfinite(age) & (age > 0), age, 1.0)
    return yearly * age


def _period_title_suffix(period: str) -> str:
    key = str(period).strip().lower()
    if key.startswith("year"):
        return "yearly"
    if key.startswith("entire"):
        return "entire life"
    return "data period"


def _render_strain_tab(window, config, x_min, x_max, y_min, y_max) -> None:
    ax = window.tab_strain.ax
    ax.clear()
    step3 = (window.pipeline_state or {}).get("step3")
    strain_type = str(getattr(window, "cmb_strain_type", None).currentText() if getattr(window, "cmb_strain_type", None) else "Axial")
    period = str(getattr(window, "cmb_strain_period", None).currentText() if getattr(window, "cmb_strain_period", None) else "Data period")
    use_diam_field = str(
        getattr(window, "cmb_use_diam_field", None).currentText() if getattr(window, "cmb_use_diam_field", None) else "False"
    ).strip().lower() == "true"
    has_mesh = (
        step3 is not None
        and getattr(step3, "mesh_connectivity", None) is not None
        and np.asarray(step3.mesh_connectivity).size > 0
        and getattr(step3, "selected_pipe_coords", None) is not None
        and np.asarray(step3.selected_pipe_coords).shape[0] > 0
    )
    title_kind = "axial" if strain_type.strip().lower().startswith("axial") else "bending"
    extra = ""
    if title_kind == "bending":
        extra = "; GIS diameter" if use_diam_field else "; default diameter"
    title = f"Pipeline {title_kind} strain ({_period_title_suffix(period)}{extra}; display only)"
    if not has_mesh:
        _placeholder(ax, title, "Run Step 3-3 Map Pipeline first.")
        window.tab_strain.canvas.draw_idle()
        return

    coords = np.asarray(step3.selected_pipe_coords, dtype=float)
    conn = np.asarray(step3.mesh_connectivity, dtype=int)
    valid = (conn[:, 0] >= 0) & (conn[:, 0] < coords.shape[0]) & (conn[:, 1] >= 0) & (conn[:, 1] < coords.shape[0])
    conn = conn[valid]
    if conn.size == 0:
        _placeholder(ax, title, "Run Step 3-3 Map Pipeline first.")
        window.tab_strain.canvas.draw_idle()
        return

    segments = [[coords[int(i0)], coords[int(i1)]] for i0, i1 in conn]
    default_diam = float(getattr(config, "pipeline_diameter_m", 0.9144))
    gdf = _load_selected_pipe_gdf(window, config)
    node_age, node_diam = _map_pipe_attrs_to_nodes(gdf, coords, default_diam)
    n_years = _coverage_n_years(window, config)
    diam_col = _pick_column(gdf, ("NOMINALDIA", "diameter", "diam", "nom_diam")) if gdf is not None else None
    if (not use_diam_field) or diam_col is None:
        diameters = np.full(coords.shape[0], default_diam, dtype=float)
    else:
        diameters = np.asarray(node_diam, dtype=float)
        diameters[~np.isfinite(diameters) | (diameters <= 0)] = default_diam

    if title_kind == "axial":
        axial = np.asarray(getattr(step3, "axial_strain", np.array([])), dtype=float).reshape(-1)
        n_elem = min(int(axial.shape[0]), int(conn.shape[0]))
        if n_elem <= 0:
            _placeholder(ax, title, "Run Step 3-3 Map Pipeline first.")
            window.tab_strain.canvas.draw_idle()
            return
        conn = conn[:n_elem]
        segments = segments[:n_elem]
        elem_age = _element_ages(node_age, conn)
        values = _apply_strain_period(axial[:n_elem], period, n_years, elem_age)
        cbar_label = "Axial strain (microstrain)"
    else:
        pipe_disp = getattr(step3, "pipe_disp", None)
        if pipe_disp is not None and np.asarray(pipe_disp).shape[0] == coords.shape[0]:
            bending_node = compute_bending_strain_3d(coords, np.asarray(pipe_disp, dtype=float), conn, diameters)
        else:
            bending_node = np.asarray(getattr(step3, "bending_strain", np.zeros(coords.shape[0])), dtype=float).reshape(-1)
            if bending_node.shape[0] < coords.shape[0]:
                padded = np.zeros(coords.shape[0], dtype=float)
                padded[: bending_node.shape[0]] = bending_node
                bending_node = padded
        bending_node = _apply_strain_period(bending_node[: coords.shape[0]], period, n_years, node_age)
        values = 0.5 * (bending_node[conn[:, 0]] + bending_node[conn[:, 1]])
        cbar_label = "Bending strain (microstrain)"

    lc = LineCollection(segments, cmap="coolwarm", linewidths=3.0)
    lc.set_array(np.asarray(values, dtype=float) * 1e6)
    ax.add_collection(lc)
    ax.autoscale()
    _apply_axis_limits(ax, x_min, x_max, y_min, y_max)
    safe_add_basemap(ax, config)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    _set_or_update_cbar(window._cbars, "strain", window.tab_strain.figure, ax, lc, cbar_label)
    window.tab_strain.canvas.draw_idle()


def _coords_for_limits(segments) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for seg in segments:
        xs.extend(np.asarray(seg)[:, 0])
        ys.extend(np.asarray(seg)[:, 1])
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _finish_pipe_map(ax, config, segments, x_min, x_max, y_min, y_max) -> None:
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    xs, ys = _coords_for_limits(segments)
    _fit_xy_limits(ax, xs, ys)
    _apply_axis_limits(ax, x_min, x_max, y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    safe_add_basemap(ax, config)


def _render_pipe_attribute_tabs(window, config, x_min, x_max, y_min, y_max) -> None:
    tabs = (
        (window.tab_pipe_age, "Pipeline Age"),
        (window.tab_pipe_mat, "Pipeline Material Type"),
        (window.tab_pipe_diam, "Pipeline Diameter"),
    )
    gdf = _load_selected_pipe_gdf(window, config)
    if gdf is None or gdf.empty:
        for tab, title in tabs:
            tab.ax.clear()
            _placeholder(tab.ax, title, "Run Step 3-1 Pick Contour first.")
            tab.canvas.draw_idle()
        return

    parts = list(_iter_line_parts(gdf))
    if not parts:
        for tab, title in tabs:
            tab.ax.clear()
            _placeholder(tab.ax, title, "No pipeline lines in selected contour GIS.")
            tab.canvas.draw_idle()
        return

    age_col = _pick_column(gdf, ("Age", "age", "INSTALLATI", "install", "year", "instyear"))
    mat_col = _pick_column(gdf, ("MATERIALTY", "material", "matl", "mat"))
    diam_col = _pick_column(gdf, ("NOMINALDIA", "diameter", "diam", "nom_diam"))

    ax_age = window.tab_pipe_age.ax
    ax_age.clear()
    if age_col is None:
        _placeholder(ax_age, "Pipeline Age", "Age column not found in selected contour GIS.")
    else:
        segs, ages, mids = [], [], []
        for row, coords, geom in parts:
            age = _normalize_age(row.get(age_col))
            if age is None or coords.shape[0] < 2:
                continue
            segs.append(coords)
            ages.append(age)
            try:
                c = geom.interpolate(0.5, normalized=True)
                mids.append((float(c.x), float(c.y), int(round(age))))
            except Exception:
                mids.append((float(np.mean(coords[:, 0])), float(np.mean(coords[:, 1])), int(round(age))))
        if not segs:
            _placeholder(ax_age, "Pipeline Age", "No valid age values in selected contour GIS.")
        else:
            ages_arr = np.asarray(ages, dtype=float)
            lo, hi = float(np.nanmin(ages_arr)), float(np.nanmax(ages_arr))
            lc = LineCollection(segs, cmap="viridis", linewidths=3.5, zorder=3)
            lc.set_array(ages_arr)
            lc.set_clim(lo, hi)
            ax_age.add_collection(lc)
            for x, y, lab in mids:
                ax_age.text(x, y, str(lab), fontsize=6, ha="center", va="center", color="black", zorder=4)
            ax_age.set_title("Pipeline Age")
            ax_age.text(
                0.0,
                -0.14,
                f"Range:[{lo:.0f},{hi:.0f}]",
                transform=ax_age.transAxes,
                ha="left",
                va="top",
                fontsize=10,
            )
            _set_or_update_cbar(
                window._cbars,
                "pipe_age",
                window.tab_pipe_age.figure,
                ax_age,
                lc,
                "Pipeline Age (years)",
            )
            _finish_pipe_map(ax_age, config, segs, x_min, x_max, y_min, y_max)
    window.tab_pipe_age.canvas.draw_idle()

    ax_mat = window.tab_pipe_mat.ax
    ax_mat.clear()
    if mat_col is None:
        _placeholder(ax_mat, "Pipeline Material Type", "Material column not found in selected contour GIS.")
    else:
        grouped = {k: [] for k in _MATERIAL_COLORS}
        for row, coords, _geom in parts:
            if coords.shape[0] < 2:
                continue
            grouped[_normalize_material(row.get(mat_col))].append(coords)
        any_seg = False
        handles = []
        for name, color in _MATERIAL_COLORS.items():
            segs = grouped[name]
            if segs:
                any_seg = True
                ax_mat.add_collection(LineCollection(segs, colors=color, linewidths=3.5, zorder=3))
            handles.append(Line2D([0], [0], color=color, lw=3.5, label=name))
        if not any_seg:
            _placeholder(ax_mat, "Pipeline Material Type", "No valid material values in selected contour GIS.")
        else:
            all_segs = [s for segs in grouped.values() for s in segs]
            ax_mat.legend(handles=handles, loc="upper left", framealpha=0.9, title="Pipeline Material Type")
            ax_mat.set_title("Pipeline Material Type")
            _finish_pipe_map(ax_mat, config, all_segs, x_min, x_max, y_min, y_max)
    window.tab_pipe_mat.canvas.draw_idle()

    ax_diam = window.tab_pipe_diam.ax
    ax_diam.clear()
    if diam_col is None:
        _placeholder(ax_diam, "Pipeline Diameter", "Diameter column not found in selected contour GIS.")
    else:
        grouped = {d: [] for d in _DIAMETER_INCHES}
        for row, coords, _geom in parts:
            if coords.shape[0] < 2:
                continue
            din = _normalize_diameter_in(row.get(diam_col))
            if din is None:
                continue
            grouped[din].append(coords)
        any_seg = False
        handles = []
        for din in _DIAMETER_INCHES:
            segs = grouped[din]
            color = _DIAMETER_COLORS[din]
            if segs:
                any_seg = True
                ax_diam.add_collection(LineCollection(segs, colors=color, linewidths=3.5, zorder=3))
            handles.append(Line2D([0], [0], color=color, lw=3.5, label=f"{din} in"))
        if not any_seg:
            _placeholder(ax_diam, "Pipeline Diameter", "No valid diameter values in selected contour GIS.")
        else:
            all_segs = [s for segs in grouped.values() for s in segs]
            ax_diam.legend(handles=handles, loc="upper left", framealpha=0.9)
            ax_diam.set_title("Pipeline Diameter")
            _finish_pipe_map(ax_diam, config, all_segs, x_min, x_max, y_min, y_max)
    window.tab_pipe_diam.canvas.draw_idle()


def _render_mesh_tab(window, config, selected_polygon, x_min, x_max, y_min, y_max) -> None:
    ax = window.tab_mesh.ax
    ax.clear()
    mesh = window.pipeline_state.get("step3_mesh") if window.pipeline_state else None
    arrays = (window.pipeline_state or {}).get("arrays") or {}
    mesh_points = None
    mesh_conn = None
    merged_gdf = None
    if isinstance(mesh, dict):
        if mesh.get("mesh_points") is not None:
            mesh_points = np.asarray(mesh["mesh_points"], dtype=float)
        if mesh.get("mesh_connectivity") is not None:
            mesh_conn = np.asarray(mesh["mesh_connectivity"], dtype=int)
        merged_gdf = mesh.get("merged_networks_gdf")
    if mesh_points is None and arrays.get("mesh_points") is not None:
        mesh_points = np.asarray(arrays["mesh_points"], dtype=float)
        if arrays.get("mesh_connectivity") is not None:
            mesh_conn = np.asarray(arrays["mesh_connectivity"], dtype=int)

    has_mesh = mesh_points is not None and mesh_points.ndim == 2 and mesh_points.shape[0] > 0
    if has_mesh:
        if merged_gdf is not None:
            try:
                gdf = merged_gdf
                if getattr(gdf, "crs", None) is None:
                    gdf = gdf.set_crs(epsg=int(config.dst_epsg))
                else:
                    gdf = gdf.to_crs(epsg=int(config.dst_epsg))
                merged_segments = _geometry_line_segments(gdf)
                if merged_segments:
                    ax.add_collection(
                        LineCollection(merged_segments, colors="orange", linewidths=2.2, alpha=0.7, zorder=2)
                    )
            except Exception:
                pass
        if mesh_conn is not None and mesh_conn.size > 0:
            segs = [
                [mesh_points[int(i0)], mesh_points[int(i1)]]
                for i0, i1 in np.asarray(mesh_conn, dtype=int)
                if 0 <= int(i0) < mesh_points.shape[0] and 0 <= int(i1) < mesh_points.shape[0]
            ]
            if segs:
                ax.add_collection(LineCollection(segs, colors="steelblue", linewidths=1.2, alpha=0.9, zorder=3))
        ax.scatter(
            mesh_points[:, 0],
            mesh_points[:, 1],
            s=12,
            c="navy",
            edgecolors="white",
            linewidths=0.2,
            zorder=4,
        )
        fault_segments = load_line_segments(config, config.fault_line_shp_path, "fault", window._overlay_cache)
        if fault_segments:
            ax.add_collection(LineCollection(fault_segments, colors="red", linewidths=1.2, alpha=0.7, zorder=5))
        if selected_polygon is not None:
            try:
                x, y = selected_polygon.exterior.xy
                ax.plot(x, y, color="yellow", linewidth=2.2, zorder=6)
            except Exception:
                pass
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        n_nodes = int(mesh_points.shape[0])
        n_elems = int(mesh_conn.shape[0]) if mesh_conn is not None and mesh_conn.ndim == 2 else 0
        ax.set_title(f"Pipeline mesh ({n_nodes} nodes, {n_elems} elements)")
        _fit_xy_limits(ax, mesh_points[:, 0], mesh_points[:, 1])
        _apply_axis_limits(ax, x_min, x_max, y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        safe_add_basemap(ax, config)
    else:
        ax.text(
            0.5,
            0.5,
            "Run Step 3-2 Mesh Pipeline to display mesh.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Pipeline mesh")
        ax.set_xticks([])
        ax.set_yticks([])
    window.tab_mesh.canvas.draw_idle()


def render_plots(window) -> None:
    """Draw all visualization tabs from MainWindow widgets and pipeline_state."""
    if not window.pipeline_state:
        return
    step1 = window.pipeline_state.get("step1")
    step2 = window.pipeline_state.get("step2")
    step3 = window.pipeline_state.get("step3")
    selected_polygon = window.selected_polygon_override
    config = window.config
    cbars = window._cbars

    window.tab_dem.ax.clear()
    if step1 is not None:
        use_smooth = window.cmb_aspect_view.currentText().startswith("Smoothed")
        using_smoothed = bool(use_smooth and step1.aspect_smooth is not None)
        aspect = step1.aspect_smooth if using_smoothed else step1.aspect_raw
        if aspect is not None:
            im = window.tab_dem.ax.imshow(aspect, cmap="twilight", origin="upper")
            window.tab_dem.ax.set_title("DEM-derived Aspect (smoothed)" if using_smoothed else "DEM-derived Aspect (raw)")
            _set_or_update_cbar(
                cbars,
                "dem_aspect",
                window.tab_dem.figure,
                window.tab_dem.ax,
                im,
                "Aspect (deg)",
            )
            if selected_polygon is not None:
                try:
                    inv = ~step1.dem_transform
                    geom = selected_polygon
                    geoms = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
                    for poly in geoms:
                        xw, yw = poly.exterior.xy
                        cols = []
                        rows = []
                        for x, y in zip(xw, yw):
                            c, r = inv * (x, y)
                            cols.append(c)
                            rows.append(r)
                        window.tab_dem.ax.plot(cols, rows, color="yellow", linewidth=2.0, zorder=9)
                except Exception:
                    pass
    window.tab_dem.canvas.draw_idle()

    window.tab_insar.ax.clear()
    if step2 is not None and step2.coords_remain.shape[0] >= 3:
        epoch_idx = window.cmb_epoch.currentData()
        if epoch_idx is None or int(epoch_idx) < 0:
            epoch_idx = 0
        epoch_idx = int(np.clip(int(epoch_idx), 0, step2.nez.shape[0] - 1))
        dir_map = {"N": 0, "E": 1, "Z": 2}
        dir_idx = dir_map.get(window.cmb_nez_dir.currentText(), 1)
        vals = np.asarray(step2.nez[epoch_idx, step2.remain_idx, dir_idx], dtype=float)
        xs = np.asarray(step2.coords_remain[:, 0], dtype=float)
        ys = np.asarray(step2.coords_remain[:, 1], dtype=float)
        vmin = window._parse_optional_float(window.le_vmin.text())
        vmax = window._parse_optional_float(window.le_vmax.text())
        cmap = "coolwarm_r" if config.filter_mode.lower() == "below" else "coolwarm"
        if config.filter_mode.lower() == "below":
            mask = vals <= float(config.contour_threshold)
        else:
            mask = vals >= float(config.contour_threshold)
        field_label = "Complete"
        plot_xs, plot_ys, plot_vals = xs, ys, vals
        cmb_field = getattr(window, "cmb_insar_field", None)
        if cmb_field is not None and str(cmb_field.currentText()).startswith("Filtered"):
            field_label = "Filtered"
            plot_xs, plot_ys, plot_vals = xs[mask], ys[mask], vals[mask]

        mappable = None
        if plot_vals.size >= 3:
            tri = mtri.Triangulation(plot_xs, plot_ys)
            cf = window.tab_insar.ax.tricontourf(
                tri,
                plot_vals,
                levels=30,
                cmap=cmap,
                alpha=0.6,
                vmin=vmin,
                vmax=vmax,
            )
            mappable = cf
        if plot_vals.size > 0:
            sc = window.tab_insar.ax.scatter(
                plot_xs,
                plot_ys,
                c=plot_vals,
                s=2,
                cmap=cmap,
                linewidths=0,
                alpha=0.65,
                zorder=8,
                vmin=vmin,
                vmax=vmax,
            )
            mappable = sc
        pipe_segments = load_line_segments(config, config.pipe_axis_path, "pipe", window._overlay_cache)
        if pipe_segments:
            window.tab_insar.ax.add_collection(LineCollection(pipe_segments, colors="gray", linewidths=0.8, zorder=9))
        fault_segments = load_line_segments(config, config.fault_line_shp_path, "fault", window._overlay_cache)
        if fault_segments:
            window.tab_insar.ax.add_collection(LineCollection(fault_segments, colors="red", linewidths=1.2, alpha=0.7, zorder=10))
        if selected_polygon is not None:
            x, y = selected_polygon.exterior.xy
            window.tab_insar.ax.plot(x, y, color="yellow", linewidth=2.2, zorder=11)
        window.tab_insar.ax.set_title(
            f"InSAR Displacement | {field_label} | Day {epoch_idx + 1} | {window.cmb_nez_dir.currentText()}"
        )
        window.tab_insar.ax.set_xlabel("X (m)")
        window.tab_insar.ax.set_ylabel("Y (m)")
        window.tab_insar.ax.set_aspect("equal", adjustable="box")
        if mappable is not None:
            _set_or_update_cbar(
                cbars,
                "insar",
                window.tab_insar.figure,
                window.tab_insar.ax,
                mappable,
                f"Displacement-{window.cmb_nez_dir.currentText()} (mm)",
            )
    x_min = window._parse_optional_float(window.le_xlim_min.text())
    x_max = window._parse_optional_float(window.le_xlim_max.text())
    y_min = window._parse_optional_float(window.le_ylim_min.text())
    y_max = window._parse_optional_float(window.le_ylim_max.text())
    if step2 is not None and step2.coords_remain.shape[0] >= 3:
        _fit_xy_limits(
            window.tab_insar.ax,
            step2.coords_remain[:, 0],
            step2.coords_remain[:, 1],
        )
        _apply_axis_limits(window.tab_insar.ax, x_min, x_max, y_min, y_max)
        window.tab_insar.ax.set_aspect("equal", adjustable="box")
        safe_add_basemap(window.tab_insar.ax, config)
    window.tab_insar.canvas.draw_idle()

    _render_mesh_tab(window, config, selected_polygon, x_min, x_max, y_min, y_max)
    _render_pipe_attribute_tabs(window, config, x_min, x_max, y_min, y_max)

    window.tab_disp.ax.clear()
    if step3 is not None and step3.selected_pipe_coords.shape[0] > 0:
        dir_map = {"N": 0, "E": 1, "Z": 2}
        dir_idx = dir_map.get(window.cmb_nez_dir.currentText(), int(np.clip(step3.dir_idx, 0, 2)))
        dir_label = ["N", "E", "Z"][dir_idx]
        xs_pipe = step3.selected_pipe_coords[:, 0]
        ys_pipe = step3.selected_pipe_coords[:, 1]
        disp_vals = np.asarray(step3.pipe_disp[:, dir_idx], dtype=float)
        if step3.mesh_connectivity is not None and step3.mesh_connectivity.size > 0:
            segs = [
                [step3.selected_pipe_coords[int(i0)], step3.selected_pipe_coords[int(i1)]]
                for i0, i1 in np.asarray(step3.mesh_connectivity, dtype=int)
            ]
            window.tab_disp.ax.add_collection(LineCollection(segs, colors="black", linewidths=1.0, alpha=0.35, zorder=2))
        cmap = "coolwarm_r" if config.filter_mode.lower() == "below" else "coolwarm"
        norm = mcolors.Normalize(vmin=float(np.nanmin(disp_vals)), vmax=float(np.nanmax(disp_vals)))
        sc = window.tab_disp.ax.scatter(xs_pipe, ys_pipe, c=disp_vals, s=18, cmap=cmap, norm=norm, edgecolors="k", linewidths=0.15, zorder=3)
        safe_add_basemap(window.tab_disp.ax, config)
        window.tab_disp.ax.set_aspect("equal", adjustable="box")
        window.tab_disp.ax.set_xlabel("X (m)")
        window.tab_disp.ax.set_ylabel("Y (m)")
        window.tab_disp.ax.set_title(f"Displacement field ({dir_label})")
        _set_or_update_cbar(cbars, "disp", window.tab_disp.figure, window.tab_disp.ax, sc, f"Displacement-{dir_label} (mm)")
    window.tab_disp.canvas.draw_idle()

    _render_strain_tab(window, config, x_min, x_max, y_min, y_max)
