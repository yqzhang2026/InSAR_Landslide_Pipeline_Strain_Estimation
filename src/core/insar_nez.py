from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

from src.config import AppConfig


def _date_columns(df, prefix: str) -> List[str]:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise KeyError(f"No date columns with prefix '{prefix}' found.")
    return cols


def _parse_date_col(col: str) -> datetime:
    return datetime.strptime(col[1:], "%Y%m%d")


def load_insar_data(config: AppConfig) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    asc_path = config.resolve_path(config.asc_shp_path)
    desc_path = config.resolve_path(config.desc_shp_path)
    if not asc_path.exists() or not desc_path.exists():
        raise FileNotFoundError("asc_shp_path and desc_shp_path must exist.")
    asc = gpd.read_file(asc_path).to_crs(epsg=int(config.dst_epsg))
    desc = gpd.read_file(desc_path).to_crs(epsg=int(config.dst_epsg))
    return asc, desc


def match_insar_observations(config: AppConfig, asc_data, desc_data) -> Dict:
    prefix = config.date_column_prefix
    asc_dates = _date_columns(asc_data, prefix)
    desc_dates = _date_columns(desc_data, prefix)
    matched_pairs: List[Tuple[str, str]] = []
    for asc_col in asc_dates:
        asc_dt = _parse_date_col(asc_col)
        closest = min(desc_dates, key=lambda x: abs(_parse_date_col(x) - asc_dt))
        if abs((_parse_date_col(closest) - asc_dt).days) <= int(config.date_match_max_days):
            matched_pairs.append((asc_col, closest))
    if not matched_pairs:
        raise ValueError("No date pairs satisfy date_match_max_days.")

    asc_coords = np.array([(g.x, g.y) for g in asc_data.geometry], dtype=float)
    desc_coords = np.array([(g.x, g.y) for g in desc_data.geometry], dtype=float)
    tree = cKDTree(desc_coords)
    dists, idx = tree.query(asc_coords, distance_upper_bound=float(config.spatial_match_max_m))
    valid = np.isfinite(dists) & (dists <= float(config.spatial_match_max_m))
    if valid.sum() == 0:
        raise ValueError("No point pairs satisfy spatial_match_max_m.")

    matched_asc = asc_data.loc[valid].reset_index(drop=True)
    matched_desc = desc_data.iloc[idx[valid]].reset_index(drop=True)
    matched_coords = 0.5 * (asc_coords[valid] + desc_coords[idx[valid]])
    base_date = _parse_date_col(matched_pairs[0][0])
    day_diffs = np.array([(_parse_date_col(d1) - base_date).days for d1, _ in matched_pairs], dtype=int)
    return {
        "matched_pairs": matched_pairs,
        "matched_asc_data": matched_asc,
        "matched_desc_data": matched_desc,
        "matched_coords": matched_coords,
        "day_diffs": day_diffs,
    }


def _los_vector(heading_deg: float, inc_deg: float) -> np.ndarray:
    return np.array(
        [
            -np.sin(np.radians(inc_deg)) * np.sin(np.radians(heading_deg)),
            np.sin(np.radians(inc_deg)) * np.cos(np.radians(heading_deg)),
            -np.cos(np.radians(inc_deg)),
        ],
        dtype=float,
    )


def _constraint_row(aspect_deg: float, flip_sign: bool) -> np.ndarray:
    sign = -1.0 if flip_sign else 1.0
    return sign * np.array(
        [
            -np.sin(np.radians(aspect_deg)),
            np.cos(np.radians(aspect_deg)),
            0.0,
        ],
        dtype=float,
    )


def solve_nez_with_aspect(
    config: AppConfig,
    matched: Dict,
    insar_aspect: np.ndarray,
    valid_idx: np.ndarray,
    progress: Optional[Callable[[int, str], None]] = None,
) -> Dict:
    def emit(pct: int, msg: str):
        if progress:
            progress(int(pct), msg)

    l_asc = _los_vector(config.asc_heading_deg, config.asc_inc_deg)
    l_desc = _los_vector(config.desc_heading_deg, config.desc_inc_deg)
    asc_df = matched["matched_asc_data"]
    desc_df = matched["matched_desc_data"]
    pairs = matched["matched_pairs"]

    valid_idx = np.asarray(valid_idx, dtype=int)
    aspect_valid = np.asarray(insar_aspect, dtype=float)[valid_idx]
    n_times = len(pairs)
    n_valid = len(valid_idx)
    nez = np.zeros((n_times, n_valid, 3), dtype=np.float32)
    emit(5, f"Preparing NEZ solver matrices for {n_valid} valid points")

    # Build per-point A matrices once: [l_asc; l_desc; constraint(aspect_i)]
    asp_rad = np.deg2rad(aspect_valid.astype(np.float64, copy=False))
    sin_asp = np.sin(asp_rad)
    cos_asp = np.cos(asp_rad)
    sign = -1.0 if bool(config.flip_constraint_sign) else 1.0
    A = np.zeros((n_valid, 3, 3), dtype=np.float64)
    A[:, 0, :] = l_asc
    A[:, 1, :] = l_desc
    A[:, 2, 0] = -sign * sin_asp
    A[:, 2, 1] = sign * cos_asp
    # A[:, 2, 2] already zero

    cond = np.linalg.cond(A).astype(np.float32)
    emit(15, "Condition numbers computed")

    # Invert non-singular A in batch; fallback to pinv only where needed.
    det = np.linalg.det(A)
    good = np.isfinite(det) & (np.abs(det) > 1e-12)
    A_inv = np.zeros_like(A)
    if np.any(good):
        A_inv[good] = np.linalg.inv(A[good])
    bad_idx = np.where(~good)[0]
    for i in bad_idx:
        A_inv[i] = np.linalg.pinv(A[i])
    emit(25, f"Matrix inversion done (fallback for {len(bad_idx)} points)")

    # Per epoch: build RHS vector for all valid points, then batch multiply.
    for j, (asc_col, desc_col) in enumerate(pairs):
        asc = asc_df[asc_col].to_numpy(dtype=np.float64, copy=False)[valid_idx]
        desc = desc_df[desc_col].to_numpy(dtype=np.float64, copy=False)[valid_idx]
        L = np.column_stack([-asc, -desc, np.zeros(n_valid, dtype=np.float64)])
        nez[j] = np.einsum("nij,nj->ni", A_inv, L).astype(np.float32)
        if n_times > 0:
            pct = 30 + int((j + 1) * 65 / n_times)
            emit(pct, f"Solving NEZ epoch {j + 1}/{n_times}: {asc_col} vs {desc_col}")

    remain_idx = np.where(cond < float(config.condition_threshold))[0]
    coords_valid = np.asarray(matched["matched_coords"], dtype=float)[valid_idx]
    coords_remain = coords_valid[remain_idx]
    emit(100, "NEZ solve completed")
    return {
        "nez": nez,
        "condition_numbers": cond,
        "remain_idx": remain_idx.astype(int),
        "coords_valid": coords_valid,
        "coords_remain": coords_remain,
        "valid_idx": valid_idx,
    }
