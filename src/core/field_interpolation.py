from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.tri as mtri
from matplotlib.tri import LinearTriInterpolator
from scipy.spatial import cKDTree


def gaussian_knn_smooth(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    k_neighbors: int,
    bandwidth_m: float,
) -> np.ndarray:
    """Notebook-style KNN Gaussian smoothing of a scalar field on scattered points."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    zs = np.asarray(zs, dtype=float).reshape(-1)
    n = xs.size
    if n == 0:
        return zs.copy()
    k = max(1, min(int(k_neighbors), n))
    tree = cKDTree(np.column_stack([xs, ys]))
    dists, idxs = tree.query(np.column_stack([xs, ys]), k=k)
    if dists.ndim == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]
    sigma = max(float(bandwidth_m), 1e-6)
    w = np.exp(-0.5 * (dists / sigma) ** 2)
    wsum = w.sum(axis=1, keepdims=True)
    wsum[wsum == 0] = 1.0
    return (w * zs[idxs]).sum(axis=1) / wsum.ravel()


def interpolate_scalar_to_points(
    src_xy: np.ndarray,
    src_vals: np.ndarray,
    dst_xy: np.ndarray,
    k_neighbors: int = 16,
    gauss_bandwidth_m: float = 20.0,
    k_fallback: int = 8,
    smooth_enable: bool = True,
    triangulation: Optional[mtri.Triangulation] = None,
) -> Tuple[np.ndarray, mtri.Triangulation, np.ndarray]:
    """
    Map a scalar field from InSAR points to destination points.

    Matches Berkeley_Hill_L3 notebook:
    1) optional KNN Gaussian smooth on the source field
    2) LinearTriInterpolator on the global triangulation
    3) IDW (p=2) fallback with k_fallback for nodes outside the hull
    """
    src_xy = np.asarray(src_xy, dtype=float)
    dst_xy = np.asarray(dst_xy, dtype=float)
    zs_raw = np.asarray(src_vals, dtype=float).reshape(-1)
    xs = src_xy[:, 0]
    ys = src_xy[:, 1]
    zs = zs_raw.copy()
    if bool(smooth_enable) and xs.size > 0:
        zs = gaussian_knn_smooth(xs, ys, zs, k_neighbors=k_neighbors, bandwidth_m=gauss_bandwidth_m)

    if triangulation is None:
        triangulation = mtri.Triangulation(xs, ys)
    lin_itp = LinearTriInterpolator(triangulation, zs)
    zi = np.asarray(lin_itp(dst_xy[:, 0], dst_xy[:, 1]).filled(np.nan), dtype=float)

    nan_mask = ~np.isfinite(zi)
    if np.any(nan_mask) and xs.size > 0:
        k = max(1, min(int(k_fallback), xs.size))
        tree = cKDTree(np.column_stack([xs, ys]))
        d, idx = tree.query(dst_xy[nan_mask], k=k)
        if d.ndim == 1:
            d = d[:, None]
            idx = idx[:, None]
        w = 1.0 / np.maximum(d, 1e-9) ** 2.0
        wsum = w.sum(axis=1, keepdims=True)
        wsum[wsum == 0] = 1.0
        zi[nan_mask] = (w * zs[idx]).sum(axis=1) / wsum.ravel()
    return zi.astype(np.float32), triangulation, zs.astype(np.float32)


def interpolate_nez_to_points(
    src_xy: np.ndarray,
    nez_epoch: np.ndarray,
    dst_xy: np.ndarray,
    k_neighbors: int = 16,
    gauss_bandwidth_m: float = 20.0,
    k_fallback: int = 8,
    smooth_enable: bool = True,
) -> np.ndarray:
    """Interpolate NEZ (N, E, Z) independently onto destination XY points."""
    nez_epoch = np.asarray(nez_epoch, dtype=float)
    dst_xy = np.asarray(dst_xy, dtype=float)
    out = np.zeros((dst_xy.shape[0], 3), dtype=np.float32)
    tri = None
    for c in range(3):
        vals, tri, _ = interpolate_scalar_to_points(
            src_xy,
            nez_epoch[:, c],
            dst_xy,
            k_neighbors=k_neighbors,
            gauss_bandwidth_m=gauss_bandwidth_m,
            k_fallback=k_fallback,
            smooth_enable=smooth_enable,
            triangulation=tri,
        )
        out[:, c] = vals
    return out
