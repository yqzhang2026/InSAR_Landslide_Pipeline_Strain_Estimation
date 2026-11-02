from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def _safe_metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    p = np.asarray(pred, dtype=float)
    r = np.asarray(ref, dtype=float)
    if p.shape != r.shape:
        return {
            "sample_count": 0.0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "shape_match": 0.0,
        }
    p_flat = p.reshape(-1)
    r_flat = r.reshape(-1)
    if p_flat.size == 0:
        return {"sample_count": 0.0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "shape_match": 1.0}
    err = p_flat - r_flat
    return {
        "sample_count": float(p_flat.size),
        "mae": float(np.nanmean(np.abs(err))),
        "rmse": float(np.sqrt(np.nanmean(err**2))),
        "bias": float(np.nanmean(err)),
        "shape_match": 1.0,
    }


def compare_if_exists(pred: np.ndarray, ref_path: str) -> Dict[str, float]:
    p = Path(ref_path).expanduser() if ref_path else None
    if not p or not p.exists():
        return {"sample_count": 0.0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "shape_match": np.nan}
    ref = np.load(p)
    return _safe_metrics(pred, ref)
