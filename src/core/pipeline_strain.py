from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.config import AppConfig


def nez_mm_to_xyz_disp_m(disp_nez_mm: np.ndarray) -> np.ndarray:
    """
    Convert NEZ displacements (mm) to map-frame meters.

    EPSG:3857 / projected XY: x += East, y += North, z += Vertical.
    NEZ storage order is (N, E, Z).
    """
    d = np.asarray(disp_nez_mm, dtype=float)
    return np.column_stack([d[:, 1], d[:, 0], d[:, 2]]) * 1e-3


def compute_axial_by_connectivity(
    points: np.ndarray,
    connectivity: np.ndarray,
    disp_nez_mm: np.ndarray,
) -> np.ndarray:
    """Engineering axial strain (L1 - L0) / L0 on 2-node elements."""
    points = np.asarray(points, dtype=float)
    conn = np.asarray(connectivity, dtype=int)
    disp_xyz = nez_mm_to_xyz_disp_m(disp_nez_mm)
    n0 = np.column_stack([points[:, 0], points[:, 1], np.zeros(len(points))])
    n1 = n0 + disp_xyz
    i0 = conn[:, 0]
    i1 = conn[:, 1]
    l0 = np.linalg.norm(n0[i1] - n0[i0], axis=1)
    l1 = np.linalg.norm(n1[i1] - n1[i0], axis=1)
    eps = np.zeros(conn.shape[0], dtype=np.float32)
    valid = l0 > 1e-12
    eps[valid] = ((l1 - l0)[valid] / l0[valid]).astype(np.float32)
    return eps


def curvature_3d(p_prev: np.ndarray, p: np.ndarray, p_next: np.ndarray) -> float:
    v1 = np.asarray(p, dtype=float) - np.asarray(p_prev, dtype=float)
    v2 = np.asarray(p_next, dtype=float) - np.asarray(p, dtype=float)
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    c = np.linalg.norm(np.asarray(p_next, dtype=float) - np.asarray(p_prev, dtype=float))
    if a < 1e-12 or b < 1e-12 or c < 1e-12:
        return 0.0
    cross = np.linalg.norm(np.cross(v1, v2))
    return float(2.0 * cross / (a * b * c))


def _build_adjacency(connectivity: np.ndarray) -> Dict[int, list]:
    adj: Dict[int, list] = defaultdict(list)
    for i, j in np.asarray(connectivity, dtype=int):
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    return adj


def compute_bending_strain_3d(
    points_xy: np.ndarray,
    disp_nez_mm: np.ndarray,
    connectivity: np.ndarray,
    diameter_at_nodes: np.ndarray,
) -> np.ndarray:
    """
    Node-based 3D bending strain using graph adjacency (handles junctions).

    strain_i = (D_i / 2) * max |kappa_deformed - kappa_original| over neighbor pairs.
    """
    pts = np.asarray(points_xy, dtype=float)
    n = pts.shape[0]
    nodes0 = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(n)])
    nodes1 = nodes0 + nez_mm_to_xyz_disp_m(disp_nez_mm)
    diameters = np.asarray(diameter_at_nodes, dtype=float).reshape(-1)
    if diameters.size == 1:
        diameters = np.full(n, float(diameters[0]), dtype=float)
    adj = _build_adjacency(connectivity)
    node_strain = np.zeros(n, dtype=np.float32)
    for i in range(n):
        neigh = adj.get(i, [])
        if len(neigh) < 2:
            continue
        di = diameters[i]
        if not np.isfinite(di):
            continue
        pi0 = nodes0[i]
        pi1 = nodes1[i]
        delta_kappas = []
        for a in range(len(neigh)):
            j = neigh[a]
            for b in range(a + 1, len(neigh)):
                k = neigh[b]
                k0 = curvature_3d(nodes0[j], pi0, nodes0[k])
                k1 = curvature_3d(nodes1[j], pi1, nodes1[k])
                delta_kappas.append(k1 - k0)
        if delta_kappas:
            node_strain[i] = (di / 2.0) * float(np.max(np.abs(delta_kappas)))
    return node_strain


def resolve_diameter_at_nodes(config: AppConfig, n_nodes: int) -> np.ndarray:
    raw = str(getattr(config, "diameter_npy_path", "") or "").strip()
    if raw:
        path = config.resolve_path(raw)
        if path.exists():
            arr = np.asarray(np.load(path), dtype=float).reshape(-1)
            if arr.size == n_nodes:
                return arr.astype(float)
            if arr.size == 1:
                return np.full(n_nodes, float(arr[0]), dtype=float)
    return np.full(n_nodes, float(config.pipeline_diameter_m), dtype=float)


def maybe_annualize(values: np.ndarray, config: AppConfig) -> np.ndarray:
    if not bool(getattr(config, "annualize", False)):
        return values
    span = max(float(getattr(config, "annualize_span_days", 3040.0)), 1e-9)
    year = float(getattr(config, "annualize_days_per_year", 365.0))
    return np.asarray(values, dtype=float) * (year / span)
