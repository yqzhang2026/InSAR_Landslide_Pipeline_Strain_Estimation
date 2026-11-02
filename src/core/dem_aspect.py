from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.warp import reproject, transform_bounds
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from src.config import AppConfig


@dataclass
class DemAspectResult:
    merged_dem: np.ndarray
    transform: Affine
    dst_epsg: int


def _resampling_from_name(name: str) -> Resampling:
    key = (name or "bilinear").strip().lower()
    if key == "nearest":
        return Resampling.nearest
    if key == "cubic":
        return Resampling.cubic
    return Resampling.bilinear


def _target_grid(paths: Iterable[str], dst_crs: CRS, resolution_m: float) -> Tuple[Affine, int, int]:
    minx = np.inf
    miny = np.inf
    maxx = -np.inf
    maxy = -np.inf
    for p in paths:
        with rasterio.open(p) as src:
            b = transform_bounds(src.crs, dst_crs, *src.bounds, densify_pts=21)
            minx = min(minx, b[0])
            miny = min(miny, b[1])
            maxx = max(maxx, b[2])
            maxy = max(maxy, b[3])
    width = int(np.ceil((maxx - minx) / resolution_m))
    height = int(np.ceil((maxy - miny) / resolution_m))
    transform = from_origin(minx, maxy, resolution_m, resolution_m)
    return transform, width, height


def _reproject_dem(path: str, dst_crs: CRS, dst_transform: Affine, width: int, height: int, method: Resampling) -> np.ndarray:
    out = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        src_arr = src.read(1).astype(np.float32, copy=False)
        if src.nodata is not None:
            src_arr = np.where(src_arr == src.nodata, np.nan, src_arr)
        reproject(
            source=src_arr,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=method,
        )
    return out


def _merge_average(dems: Iterable[np.ndarray]) -> np.ndarray:
    stack = np.stack([np.asarray(a, dtype=np.float32) for a in dems], axis=0)
    valid = np.isfinite(stack)
    num = np.sum(np.where(valid, stack, 0.0), axis=0)
    den = np.sum(valid, axis=0)
    return np.where(den > 0, num / den, np.nan).astype(np.float32)


def _gaussian_smooth_nan(arr: np.ndarray, sigma: float) -> np.ndarray:
    valid = np.isfinite(arr)
    work = np.where(valid, arr, 0.0)
    weight = valid.astype(np.float32)
    smooth_num = gaussian_filter(work, sigma=sigma, mode="nearest")
    smooth_den = gaussian_filter(weight, sigma=sigma, mode="nearest")
    out = np.divide(
        smooth_num,
        smooth_den,
        out=np.full_like(smooth_num, np.nan, dtype=np.float32),
        where=smooth_den > 1e-6,
    )
    return out.astype(np.float32)


def aspect_from_dem(dem: np.ndarray, transform: Affine) -> np.ndarray:
    dx = float(abs(transform.a))
    dy = float(abs(transform.e))
    dz_dy, dz_dx = np.gradient(dem, dy, dx)
    aspect = np.degrees(np.arctan2(-dz_dx, -dz_dy))
    return np.mod(aspect + 360.0, 360.0).astype(np.float32)


def build_dem_aspect(config: AppConfig, progress: Optional[Callable[[int, str], None]] = None) -> DemAspectResult:
    def emit(pct: int, msg: str) -> None:
        if progress:
            progress(int(pct), msg)

    dem_paths = config.parse_dem_paths()
    resolved = [str(config.resolve_path(p)) for p in dem_paths]
    dst_crs = CRS.from_epsg(int(config.dst_epsg))
    emit(15, "Building DEM target grid")
    transform, width, height = _target_grid(
        resolved,
        dst_crs=dst_crs,
        resolution_m=float(config.dem_resolution_m),
    )
    method = _resampling_from_name(config.resampling_method)
    emit(30, "Reprojecting first DEM")
    d1 = _reproject_dem(
        resolved[0],
        dst_crs=dst_crs,
        dst_transform=transform,
        width=width,
        height=height,
        method=method,
    )
    emit(60, "Reprojecting second DEM")
    d2 = _reproject_dem(
        resolved[1],
        dst_crs=dst_crs,
        dst_transform=transform,
        width=width,
        height=height,
        method=method,
    )
    emit(85, "Averaging overlapping DEM pixels")
    merged = _merge_average([d1, d2])
    return DemAspectResult(
        merged_dem=merged,
        transform=transform,
        dst_epsg=int(config.dst_epsg),
    )


def smooth_dem(dem: np.ndarray, sigma_px: float) -> np.ndarray:
    return _gaussian_smooth_nan(dem, sigma=float(sigma_px))


def sample_aspect_at_points(aspect_deg: np.ndarray, transform: Affine, points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = aspect_deg.shape
    xs = transform.c + (np.arange(w, dtype=float) + 0.5) * transform.a
    ys_desc = transform.f + (np.arange(h, dtype=float) + 0.5) * transform.e
    ys = ys_desc[::-1]
    asp = aspect_deg[::-1, :]
    interp = RegularGridInterpolator(
        (ys, xs),
        asp,
        bounds_error=False,
        fill_value=np.nan,
    )
    q = np.column_stack([points_xy[:, 1], points_xy[:, 0]])
    sampled = interp(q).astype(np.float32)
    valid_idx = np.where(np.isfinite(sampled))[0]
    return sampled, valid_idx
