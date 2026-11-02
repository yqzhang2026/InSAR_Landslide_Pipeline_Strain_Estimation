from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import linemerge, polygonize, unary_union
from shapely.strtree import STRtree

from src.config import AppConfig
from src.core.field_interpolation import interpolate_nez_to_points
from src.core.pipeline_strain import (
    compute_axial_by_connectivity,
    compute_bending_strain_3d,
    resolve_diameter_at_nodes,
)


def _load_pipe_geometry(path: Path, dst_epsg: int) -> LineString:
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.asarray(np.load(path), dtype=float)
        return LineString(arr[:, :2])
    if ext in {".csv", ".txt"}:
        arr = np.loadtxt(path, delimiter="," if ext == ".csv" else None)
        return LineString(np.asarray(arr, dtype=float)[:, :2])
    if ext == ".shp":
        gdf = gpd.read_file(path).to_crs(epsg=dst_epsg)
        geom = gdf.geometry.unary_union
        if isinstance(geom, LineString):
            return geom
        if isinstance(geom, MultiLineString):
            return max(list(geom.geoms), key=lambda g: g.length)
    raise ValueError(f"Unsupported pipe geometry format: {path}")


def _build_interpolated_grid(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, n: int = 220) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax = np.nanmin(xs), np.nanmax(xs)
    ymin, ymax = np.nanmin(ys), np.nanmax(ys)
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n))
    gz = griddata((xs, ys), zs, (gx, gy), method="linear")
    nan_mask = ~np.isfinite(gz)
    if np.any(nan_mask):
        gz_nn = griddata((xs, ys), zs, (gx, gy), method="nearest")
        gz[nan_mask] = gz_nn[nan_mask]
    return gx, gy, gz


def extract_contour_polygons(
    gx: np.ndarray,
    gy: np.ndarray,
    gz: np.ndarray,
    threshold: float,
    mode: str,
    levels: int = 30,
) -> List[Polygon]:
    """
    Robust contour polygon extraction compatible with different Matplotlib versions.
    It avoids relying on QuadContourSet.collections and uses allsegs + polygonize.
    """
    zmin = float(np.nanmin(gz))
    zmax = float(np.nanmax(gz))
    if mode == "above":
        lo = threshold
        hi = zmax
    else:
        lo = zmin
        hi = threshold
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return []

    level_values = np.linspace(lo, hi, int(max(levels, 2)))
    fig, ax = plt.subplots()
    # Use contour to get isolines, then polygonize like notebook workflow.
    cs = ax.contour(gx, gy, gz, levels=level_values)
    polygons: List[Polygon] = []
    for segs_at_level in cs.allsegs:
        line_strings = []
        for seg in segs_at_level:
            if seg is None or len(seg) < 2:
                continue
            try:
                line_strings.append(LineString(seg))
            except Exception:
                continue
        if not line_strings:
            continue
        for poly in polygonize(line_strings):
            if poly.is_valid and poly.area > 0:
                polygons.append(poly)
    plt.close(fig)

    # Filter polygons by threshold side using representative point sampling.
    # (fallback to centroid index-based nearest sample if interpolation unavailable)
    filtered: List[Polygon] = []
    flat_points = np.column_stack([gx.ravel(), gy.ravel()])
    flat_vals = gz.ravel()
    for poly in polygons:
        rp = poly.representative_point()
        d2 = (flat_points[:, 0] - rp.x) ** 2 + (flat_points[:, 1] - rp.y) ** 2
        idx = int(np.argmin(d2))
        v = float(flat_vals[idx])
        if mode == "above":
            if v >= threshold:
                filtered.append(poly)
        else:
            if v <= threshold:
                filtered.append(poly)
    return filtered


def _coords_from_linestring(line: LineString) -> np.ndarray:
    arr = np.asarray(line.coords, dtype=float)
    if arr.shape[0] < 2:
        raise ValueError("Selected pipeline segment has insufficient points.")
    return arr[:, :2]


def _map_displacement_knn(src_xy: np.ndarray, src_disp: np.ndarray, dst_xy: np.ndarray, k: int) -> np.ndarray:
    tree = cKDTree(src_xy)
    d, idx = tree.query(dst_xy, k=min(k, len(src_xy)))
    if d.ndim == 1:
        d = d[:, None]
        idx = idx[:, None]
    w = 1.0 / np.maximum(d, 1e-6)
    w = w / np.sum(w, axis=1, keepdims=True)
    sampled = np.sum(src_disp[idx] * w[:, :, None], axis=1)
    return sampled.astype(np.float32)


def _compute_axial_strain(coords: np.ndarray, disp: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    eps = np.zeros(n - 1, dtype=np.float32)
    for i in range(n - 1):
        p1 = coords[i]
        p2 = coords[i + 1]
        d1 = disp[i]
        d2 = disp[i + 1]
        L0 = np.linalg.norm(p2 - p1)
        q1 = np.array([p1[0] + d1[1] * 1e-3, p1[1] + d1[0] * 1e-3, d1[2] * 1e-3], dtype=float)
        q2 = np.array([p2[0] + d2[1] * 1e-3, p2[1] + d2[0] * 1e-3, d2[2] * 1e-3], dtype=float)
        L1 = np.linalg.norm(q2 - q1)
        eps[i] = (L1 - L0) / max(L0, 1e-9)
    return eps


def _curvature_2d(p_prev: np.ndarray, p: np.ndarray, p_next: np.ndarray) -> float:
    x1, y1 = p_prev
    x2, y2 = p
    x3, y3 = p_next
    num = 2.0 * ((x3 - x2) * (y1 - y2) - (x1 - x2) * (y3 - y2))
    a = np.linalg.norm(p_prev - p)
    b = np.linalg.norm(p_next - p)
    c = np.linalg.norm(p_next - p_prev)
    den = a * b * c
    return num / den if den > 0 else 0.0


def _compute_bending_strain(coords: np.ndarray, disp: np.ndarray, diameter_m: float) -> np.ndarray:
    deformed = coords + np.column_stack([disp[:, 1], disp[:, 0]]) * 1e-3
    n = coords.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for i in range(1, n - 1):
        k0 = _curvature_2d(coords[i - 1], coords[i], coords[i + 1])
        k1 = _curvature_2d(deformed[i - 1], deformed[i], deformed[i + 1])
        out[i] = (k1 - k0) * (diameter_m / 2.0)
    if n > 2:
        out[0] = out[1]
        out[-1] = out[-2]
    return out


def build_step3_grid(
    config: AppConfig,
    coords_remain: np.ndarray,
    nez_remain: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    day_idx = int(np.clip(config.day_idx, 0, nez_remain.shape[0] - 1))
    dir_idx = int(np.clip(config.dir_idx, 0, 2))
    zs = np.asarray(nez_remain[day_idx, :, dir_idx], dtype=float)
    xs = np.asarray(coords_remain[:, 0], dtype=float)
    ys = np.asarray(coords_remain[:, 1], dtype=float)
    gx, gy, gz = _build_interpolated_grid(xs, ys, zs)
    return gx, gy, gz, day_idx, dir_idx


def run_step3_mapping(
    config: AppConfig,
    coords_remain: np.ndarray,
    nez_remain: np.ndarray,
    selected_polygon: Optional[Polygon] = None,
) -> Dict:
    gx, gy, gz, day_idx, dir_idx = build_step3_grid(config, coords_remain, nez_remain)
    poly = selected_polygon
    if poly is None:
        candidates = extract_contour_polygons(gx, gy, gz, float(config.contour_threshold), config.filter_mode.lower())
        poly = max(candidates, key=lambda p: p.area) if candidates else None
    if poly is None:
        raise RuntimeError("No contour polygon generated under current threshold settings.")

    pipe_line = _load_pipe_geometry(config.resolve_path(config.pipe_axis_path), int(config.dst_epsg))
    segment = pipe_line.intersection(poly)
    if segment.is_empty:
        raise RuntimeError("No pipeline segment falls inside selected contour polygon.")
    if isinstance(segment, MultiLineString):
        segment = max(list(segment.geoms), key=lambda g: g.length)
    coords_pipe = _coords_from_linestring(segment)

    nez_epoch = np.asarray(nez_remain[day_idx], dtype=float)
    disp_pipe = _map_displacement_knn(coords_remain, nez_epoch, coords_pipe, int(config.k_neighbors))
    axial = _compute_axial_strain(coords_pipe, disp_pipe)
    bending = _compute_bending_strain(coords_pipe, disp_pipe, float(config.pipeline_diameter_m))
    return {
        "day_idx": day_idx,
        "dir_idx": dir_idx,
        "threshold": float(config.contour_threshold),
        "selected_polygon": poly,
        "selected_pipe_coords": coords_pipe,
        "pipe_disp": disp_pipe,
        "axial_strain": axial,
        "bending_strain": bending,
        "grid_x": gx,
        "grid_y": gy,
        "grid_z": gz,
    }


def build_step3_exports(result: Dict, dst_epsg: int, output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    poly = result["selected_polygon"]
    coords = np.asarray(result["selected_pipe_coords"], dtype=float)
    disp = np.asarray(result["pipe_disp"], dtype=float)
    axial = np.asarray(result["axial_strain"], dtype=float)
    bending = np.asarray(result["bending_strain"], dtype=float)

    contour_path = output_dir / "step3_contour_polygon.shp"
    if poly is not None:
        gdf_poly = gpd.GeoDataFrame({"name": ["selected_contour"]}, geometry=[poly], crs=f"EPSG:{dst_epsg}")
        gdf_poly.to_file(contour_path)

    pipe_path = output_dir / "step3_pipeline_inside_contour.shp"
    gdf_pipe = gpd.GeoDataFrame({"name": ["pipeline_inside"]}, geometry=[LineString(coords)], crs=f"EPSG:{dst_epsg}")
    gdf_pipe.to_file(pipe_path)

    seg_geoms = []
    axial_vals = []
    bend_vals = []
    n_seg = min(len(coords) - 1, len(axial))
    for i in range(n_seg):
        seg_geoms.append(LineString([coords[i], coords[i + 1]]))
        axial_vals.append(float(axial[i]))
        bend_vals.append(float(np.nanmean(bending[i : i + 2])))
    strain_path = output_dir / "step3_pipeline_strain_segments.shp"
    gdf_strain = gpd.GeoDataFrame(
        {
            "axial_strain": axial_vals,
            "bending_strain": bend_vals,
        },
        geometry=seg_geoms,
        crs=f"EPSG:{dst_epsg}",
    )
    gdf_strain.to_file(strain_path)

    csv_path = output_dir / "step3_pipeline_node_displacement.csv"
    pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "disp_n_mm": disp[:, 0],
            "disp_e_mm": disp[:, 1],
            "disp_z_mm": disp[:, 2],
            "bending_strain": bending,
        }
    ).to_csv(csv_path, index=False)

    outputs = {
        "pipeline_inside_shp": str(pipe_path),
        "pipeline_strain_shp": str(strain_path),
        "pipeline_node_displacement_csv": str(csv_path),
    }
    if poly is not None:
        outputs["contour_polygon_shp"] = str(contour_path)
    return outputs


def _as_lines_list(geom) -> List[LineString]:
    if geom is None or geom.is_empty:
        return []
    gt = getattr(geom, "geom_type", "")
    if gt == "LineString":
        return [geom]
    if gt == "MultiLineString":
        return list(geom.geoms)
    if gt == "GeometryCollection":
        out = []
        for g in geom.geoms:
            out.extend(_as_lines_list(g))
        return out
    return []


def _pipe_segments_from_gdf(gdf: gpd.GeoDataFrame) -> List[np.ndarray]:
    segs: List[np.ndarray] = []
    for g in gdf.geometry:
        gt = getattr(g, "geom_type", "")
        if gt == "LineString":
            x, y = g.xy
            segs.append(np.column_stack([x, y]))
        elif gt == "MultiLineString":
            for part in g.geoms:
                x, y = part.xy
                segs.append(np.column_stack([x, y]))
    return segs


def save_selected_lines_from_contour(
    config: AppConfig,
    selected_polygon: Polygon,
    output_dir: Path,
    out_name: str = "landslide1.shp",
) -> Dict[str, object]:
    """
    Save selected pipeline lines consistent with notebook contour-pick step.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pipes = gpd.read_file(config.resolve_path(config.pipe_axis_path))
    if pipes.crs is None:
        pipes = pipes.set_crs(epsg=int(config.dst_epsg))
    pipes = pipes.to_crs(epsg=int(config.dst_epsg))
    pipes = pipes[pipes.geometry.notnull()]
    pipes = pipes[pipes.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    if pipes.empty:
        raise RuntimeError("No LineString geometries found in pipe source.")

    poly_buf = selected_polygon.buffer(1e-6)
    try:
        cand_idx = list(pipes.sindex.query(poly_buf, predicate="intersects"))
        cand = pipes.iloc[cand_idx].copy()
    except Exception:
        cand = pipes[pipes.intersects(poly_buf)].copy()
    inside_mask = cand.covered_by(poly_buf)
    inter_mask = cand.intersects(poly_buf) & (~inside_mask)
    inside_ids = cand.index[inside_mask].tolist()
    intersect_ids = cand.index[inter_mask].tolist()
    sel_ids = sorted(set(inside_ids) | set(intersect_ids))
    if len(sel_ids) == 0:
        raise RuntimeError("No pipeline lines selected by contour.")
    gdf_sel = pipes.loc[sel_ids].copy()
    if hasattr(gdf_sel, "explode"):
        gdf_sel = gdf_sel.explode(index_parts=False, ignore_index=True)
    out_path = output_dir / out_name
    gdf_sel.to_file(out_path)

    contour_path = output_dir / "selected_contour_polygon.shp"
    gpd.GeoDataFrame({"name": ["selected_contour"]}, geometry=[selected_polygon], crs=pipes.crs).to_file(contour_path)
    return {
        "selected_lines_path": str(out_path),
        "selected_contour_path": str(contour_path),
        "inside_count": len(inside_ids),
        "intersect_count": len(intersect_ids),
        "union_count": len(sel_ids),
    }


def build_mesh_from_selected_lines(
    selected_lines_path: Path,
    top_n: int,
    mesh_size: float,
    progress: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, object]:
    def emit(pct: int, msg: str) -> None:
        if progress:
            progress(int(pct), msg)

    emit(25, "Reading selected pipeline lines")
    gdf_sel = gpd.read_file(selected_lines_path)
    gdf_sel = gdf_sel[gdf_sel.geometry.notnull()].copy()
    if hasattr(gdf_sel, "explode"):
        gdf_sel = gdf_sel.explode(index_parts=False, ignore_index=True)
    if gdf_sel.empty:
        raise RuntimeError("Selected geometry is empty after cleaning.")

    U = unary_union(gdf_sel.geometry.values)
    lines_node = _as_lines_list(U)
    if len(lines_node) == 0:
        raise RuntimeError("No lines after unary_union.")

    tree = STRtree(lines_node)
    parent = list(range(len(lines_node)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    n_lines = len(lines_node)
    for i, gi in enumerate(lines_node):
        if n_lines > 0 and (i % max(1, n_lines // 8) == 0):
            emit(30 + int(25 * i / n_lines), f"Connecting pipeline segments {i + 1}/{n_lines}")
        candidate_idxs = tree.query(gi)
        for j in candidate_idxs:
            j = int(j)
            if j <= i:
                continue
            gj = lines_node[j]
            try:
                if gi.intersects(gj) or gi.touches(gj):
                    union(i, j)
            except Exception:
                continue

    components: Dict[int, List[int]] = {}
    for i in range(len(lines_node)):
        r = find(i)
        components.setdefault(r, []).append(i)

    def total_length(idxs: List[int]) -> float:
        return float(sum(lines_node[i].length for i in idxs))

    ranked = sorted(components.items(), key=lambda kv: total_length(kv[1]), reverse=True)
    top_keep = ranked[: max(1, min(int(top_n), len(ranked)))]

    def merge_lines_robust(lines_list: List[LineString]):
        U_local = unary_union(lines_list)
        Ls = _as_lines_list(U_local)
        if len(Ls) == 1:
            return Ls[0]
        return linemerge(MultiLineString(Ls))

    merged_per_component = []
    for _, idxs in top_keep:
        lines_list = [lines_node[i] for i in idxs]
        mg = merge_lines_robust(lines_list)
        merged_per_component.extend(_as_lines_list(mg))

    all_points = []
    all_connectivity = []
    line_ranges: List[Tuple[int, int]] = []
    point_offset = 0
    for line in merged_per_component:
        coords = np.asarray(line.coords)
        if coords.shape[0] < 2:
            continue
        seg_vecs = np.diff(coords, axis=0)
        seg_lengths = np.linalg.norm(seg_vecs, axis=1)
        cum_lengths = np.insert(np.cumsum(seg_lengths), 0, 0.0)
        total_len = cum_lengths[-1]
        if total_len <= 0:
            continue
        n_steps = max(2, int(np.ceil(total_len / float(mesh_size))) + 1)
        target_d = np.linspace(0.0, total_len, n_steps)
        sampled = []
        for td in target_d:
            si = int(np.clip(np.searchsorted(cum_lengths, td) - 1, 0, len(seg_lengths) - 1))
            seg_len = seg_lengths[si]
            t = (td - cum_lengths[si]) / seg_len if seg_len > 0 else 0.0
            p = (1 - t) * coords[si] + t * coords[si + 1]
            sampled.append(p)
        sampled = np.asarray(sampled)
        conn = np.column_stack(
            [
                np.arange(point_offset, point_offset + len(sampled) - 1),
                np.arange(point_offset + 1, point_offset + len(sampled)),
            ]
        )
        all_points.append(sampled)
        all_connectivity.append(conn)
        line_ranges.append((point_offset, point_offset + len(sampled)))
        point_offset += len(sampled)

    if len(all_points) == 0:
        raise RuntimeError("No mesh generated from selected lines.")
    points = np.vstack(all_points)
    conn = np.vstack(all_connectivity)
    merged_gdf = gpd.GeoDataFrame(geometry=merged_per_component, crs=gdf_sel.crs)
    return {
        "mesh_points": points,
        "mesh_connectivity": conn,
        "line_ranges": line_ranges,
        "merged_networks_gdf": merged_gdf,
        "component_count": len(components),
        "kept_component_count": len(top_keep),
    }


def _compute_axial_by_connectivity(points: np.ndarray, connectivity: np.ndarray, disp_nodes: np.ndarray) -> np.ndarray:
    eps = np.zeros(connectivity.shape[0], dtype=np.float32)
    for i, (i0, i1) in enumerate(connectivity):
        p0 = points[int(i0)]
        p1 = points[int(i1)]
        d0 = disp_nodes[int(i0)]
        d1 = disp_nodes[int(i1)]
        L0 = np.linalg.norm(p1 - p0)
        q0 = np.array([p0[0] + d0[1] * 1e-3, p0[1] + d0[0] * 1e-3, d0[2] * 1e-3], dtype=float)
        q1 = np.array([p1[0] + d1[1] * 1e-3, p1[1] + d1[0] * 1e-3, d1[2] * 1e-3], dtype=float)
        L1 = np.linalg.norm(q1 - q0)
        eps[i] = (L1 - L0) / max(L0, 1e-9)
    return eps


def run_step3_mesh_pipeline(
    config: AppConfig,
    coords_remain: np.ndarray,
    nez_remain: np.ndarray,
    selected_lines_path: Path,
) -> Dict[str, object]:
    mesh = build_mesh_from_selected_lines(
        selected_lines_path=selected_lines_path,
        top_n=int(config.top_n),
        mesh_size=max(float(config.mesh_size_m), 1.0),
    )
    return compute_step3_response_from_mesh(config, coords_remain, nez_remain, mesh)


def compute_step3_response_from_mesh(
    config: AppConfig,
    coords_remain: np.ndarray,
    nez_remain: np.ndarray,
    mesh: Dict[str, object],
) -> Dict[str, object]:
    day_idx = int(np.clip(config.day_idx, 0, nez_remain.shape[0] - 1))
    nez_epoch = np.asarray(nez_remain[day_idx], dtype=float)
    mesh_points = np.asarray(mesh["mesh_points"], dtype=float)
    mesh_connectivity = np.asarray(mesh["mesh_connectivity"], dtype=int)
    disp_nodes = interpolate_nez_to_points(
        src_xy=np.asarray(coords_remain, dtype=float),
        nez_epoch=nez_epoch,
        dst_xy=mesh_points,
        k_neighbors=int(config.k_neighbors),
        gauss_bandwidth_m=float(getattr(config, "gauss_bandwidth_m", 20.0)),
        k_fallback=int(config.k_fallback),
        smooth_enable=bool(getattr(config, "field_smooth_enable", True)),
    )
    disp_nodes = np.asarray(disp_nodes, dtype=np.float32)
    axial_elem = compute_axial_by_connectivity(mesh_points, mesh_connectivity, disp_nodes)
    diameter = resolve_diameter_at_nodes(config, mesh_points.shape[0])
    bending_nodes = compute_bending_strain_3d(
        mesh_points,
        disp_nodes,
        mesh_connectivity,
        diameter,
    )
    return {
        "day_idx": day_idx,
        "selected_pipe_coords": mesh_points,
        "mesh_connectivity": mesh_connectivity,
        "pipe_disp": disp_nodes,
        "axial_strain": axial_elem,
        "bending_strain": bending_nodes,
        "merged_networks_gdf": mesh["merged_networks_gdf"],
        "component_count": mesh["component_count"],
        "kept_component_count": mesh["kept_component_count"],
    }
