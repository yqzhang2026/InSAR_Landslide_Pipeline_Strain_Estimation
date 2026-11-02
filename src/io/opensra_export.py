from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from src.config import AppConfig
from src.core.contracts import Step1Result, Step3Result


_DIAMETER_COL_CANDIDATES = ("NOMINALDIA", "diameter", "diam", "nom_diam")


def _find_diameter_column(columns) -> Optional[str]:
    cols = {str(c).lower(): str(c) for c in columns}
    for name in _DIAMETER_COL_CANDIDATES:
        if name.lower() in cols:
            return cols[name.lower()]
    for c in columns:
        cl = str(c).lower()
        for name in _DIAMETER_COL_CANDIDATES:
            if name.lower() in cl:
                return str(c)
    return None


def _diam_mm_from_raw(raw: pd.Series) -> pd.Series:
    val = pd.to_numeric(raw, errors="coerce")
    out = pd.Series(np.nan, index=val.index, dtype=float)
    valid = val.notna() & np.isfinite(val.to_numpy(dtype=float, copy=False)) & (val > 0)
    mm_mask = valid & (val > 80)
    in_mask = valid & ~mm_mask
    out.loc[mm_mask] = val.loc[mm_mask].astype(float)
    out.loc[in_mask] = val.loc[in_mask].astype(float) * 25.4
    return out


def _sanitize_shp_field_names(columns: List[str]) -> Dict[str, str]:
    used = set()
    mapping: Dict[str, str] = {}
    for raw in columns:
        base = str(raw).replace("%", "_")[:10]
        if not base:
            base = "field"
        name = base
        n = 1
        while name.lower() in used:
            suffix = str(n)
            name = (base[: max(1, 10 - len(suffix))] + suffix)[:10]
            n += 1
        used.add(name.lower())
        mapping[raw] = name
    return mapping


def _years_for_epoch(step1: Optional[Step1Result], day_idx: int) -> float:
    if step1 is None or step1.day_diffs is None:
        return float("nan")
    diffs = np.asarray(step1.day_diffs, dtype=float).reshape(-1)
    if diffs.size == 0:
        return float("nan")
    idx = int(np.clip(int(day_idx), 0, diffs.size - 1))
    years = float(diffs[idx]) / 365.25
    if not np.isfinite(years) or years <= 0:
        return float("nan")
    return years


def _element_strain_rates(
    axial_elem: np.ndarray,
    bending_node: np.ndarray,
    connectivity: np.ndarray,
    years: float,
) -> tuple:
    conn = np.asarray(connectivity, dtype=int)
    axial = np.asarray(axial_elem, dtype=float).reshape(-1)
    bending = np.asarray(bending_node, dtype=float).reshape(-1)
    n_elem = int(conn.shape[0])
    if axial.size != n_elem:
        n = min(axial.size, n_elem)
        conn = conn[:n]
        axial = axial[:n]
        n_elem = n
    i0 = np.clip(conn[:, 0], 0, max(bending.size - 1, 0))
    i1 = np.clip(conn[:, 1], 0, max(bending.size - 1, 0))
    if bending.size == 0:
        eps_b = np.zeros(n_elem, dtype=float)
    else:
        eps_b = 0.5 * (bending[i0] + bending[i1])
    eps_long = axial + np.sign(axial) * np.abs(eps_b)
    micro = eps_long * 1.0e6
    if not np.isfinite(years) or years <= 0:
        rate = np.full(n_elem, np.nan, dtype=float)
    else:
        rate = micro / (1.0e4 * years)
    eps_c = np.maximum(-rate, 0.0)
    eps_t = np.maximum(rate, 0.0)
    return conn, eps_c, eps_t


def _load_pipe_gdf(config: AppConfig) -> Optional[gpd.GeoDataFrame]:
    path = config.resolve_path(config.pipe_axis_path)
    if not path.exists():
        return None
    gdf = gpd.read_file(path)
    if gdf.empty:
        return None
    src_epsg = int(config.dst_epsg)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=src_epsg)
    else:
        gdf = gdf.to_crs(epsg=src_epsg)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])].copy()
    if gdf.empty:
        return None
    gdf = gdf.reset_index(drop=True)
    return gdf


def _attach_pipe_attributes(elements: gpd.GeoDataFrame, pipes: Optional[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    if pipes is None or pipes.empty or elements.empty:
        elements["Diam_mm"] = np.nan
        return elements
    pipe_attr = pipes.copy()
    pipe_attr["_pipe_fid"] = np.arange(len(pipe_attr), dtype=int)
    mids = elements.copy()
    mids.geometry = mids.geometry.centroid
    joined = gpd.sjoin_nearest(mids[["geometry"]], pipe_attr, how="left", distance_col="_pipe_dist")
    joined = joined[~joined.index.duplicated(keep="first")]
    joined = joined.drop(columns=["geometry", "_pipe_dist", "index_right", "_pipe_fid"], errors="ignore")
    diam_col = _find_diameter_column(joined.columns)
    if diam_col is not None:
        joined["Diam_mm"] = _diam_mm_from_raw(joined[diam_col])
    else:
        joined["Diam_mm"] = np.nan
    drop_geom = [c for c in joined.columns if c == "geometry"]
    joined = joined.drop(columns=drop_geom, errors="ignore")
    out = elements.join(joined, how="left", rsuffix="_pipe")
    if "Diam_mm" not in out.columns:
        out["Diam_mm"] = np.nan
    return out


def export_opensra_layers(config: AppConfig, state: Dict, run_dir: Path) -> Optional[Path]:
    step3 = state.get("step3") if state else None
    if step3 is None or not isinstance(step3, Step3Result):
        return None
    if step3.mesh_connectivity is None or step3.selected_pipe_coords is None:
        return None
    coords = np.asarray(step3.selected_pipe_coords, dtype=float)
    conn = np.asarray(step3.mesh_connectivity, dtype=int)
    if coords.ndim != 2 or coords.shape[0] < 2 or conn.ndim != 2 or conn.shape[1] < 2 or conn.shape[0] == 0:
        return None
    step1 = state.get("step1")
    years = _years_for_epoch(step1 if isinstance(step1, Step1Result) else None, int(step3.day_idx))
    conn, eps_c, eps_t = _element_strain_rates(step3.axial_strain, step3.bending_strain, conn, years)
    geoms = []
    keep = []
    for i, (i0, i1) in enumerate(conn):
        i0 = int(i0)
        i1 = int(i1)
        if i0 < 0 or i1 < 0 or i0 >= coords.shape[0] or i1 >= coords.shape[0]:
            continue
        p0 = coords[i0]
        p1 = coords[i1]
        if not np.isfinite([p0[0], p0[1], p1[0], p1[1]]).all():
            continue
        geoms.append(LineString([(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))]))
        keep.append(i)
    if not geoms:
        return None
    keep = np.asarray(keep, dtype=int)
    src_epsg = int(config.dst_epsg)
    gdf = gpd.GeoDataFrame(
        {
            "epsrate%c": np.asarray(eps_c, dtype=float)[keep],
            "epsrate%t": np.asarray(eps_t, dtype=float)[keep],
        },
        geometry=geoms,
        crs=f"EPSG:{src_epsg}",
    )
    pipes = _load_pipe_gdf(config)
    gdf = _attach_pipe_attributes(gdf, pipes)
    gdf = gdf.to_crs(epsg=4326)
    out_dir = Path(run_dir) / "opensra"
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg_path = out_dir / "pipeline_elements.gpkg"
    shp_path = out_dir / "pipeline_elements.shp"
    gdf.to_file(gpkg_path, layer="pipeline_elements", driver="GPKG")
    shp_cols = [c for c in gdf.columns if c != "geometry"]
    rename = _sanitize_shp_field_names(shp_cols)
    gdf_shp = gdf.rename(columns=rename)
    gdf_shp.to_file(shp_path, driver="ESRI Shapefile")
    return out_dir
