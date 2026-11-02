from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rasterio.transform import Affine
from shapely.geometry import Polygon


@dataclass
class Step1Result:
    merged_dem: np.ndarray
    dem_transform: Affine
    dst_epsg: int
    aspect_raw: Optional[np.ndarray] = None
    dem_smoothed: Optional[np.ndarray] = None
    aspect_smooth: Optional[np.ndarray] = None
    matched_dates: Optional[List[Tuple[str, str]]] = None
    matched_coords: Optional[np.ndarray] = None
    matched_asc_data: Optional[object] = None
    matched_desc_data: Optional[object] = None
    day_diffs: Optional[np.ndarray] = None
    insar_aspect: Optional[np.ndarray] = None
    valid_idx: Optional[np.ndarray] = None


@dataclass
class Step2Result:
    nez: np.ndarray
    condition_numbers: np.ndarray
    remain_idx: np.ndarray
    coords_valid: np.ndarray
    coords_remain: np.ndarray
    valid_idx: np.ndarray


@dataclass
class Step3Result:
    day_idx: int
    dir_idx: int
    threshold: float
    selected_polygon: Optional[Polygon]
    selected_lines_path: Optional[str]
    selected_pipe_coords: np.ndarray
    mesh_connectivity: Optional[np.ndarray]
    pipe_disp: np.ndarray
    axial_strain: np.ndarray
    bending_strain: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    grid_z: np.ndarray
    export_paths: Dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineState:
    """Workflow state. Stored as a dict in UI/workers via dataclasses.asdict-incompatible numpy blobs.

    Extra Step3 keys (also present when using ``__dict__``):
    - selected_polygon_override
    - step3_selected_lines_path
    - step3_selected_contour_path
    - step3_mesh
    - needs_contour_pick
    """

    config: object
    step1: Optional[Step1Result] = None
    step2: Optional[Step2Result] = None
    step3: Optional[Step3Result] = None
    step3_mesh: Optional[Dict[str, Any]] = None
    step3_selected_lines_path: Optional[str] = None
    step3_selected_contour_path: Optional[str] = None
    selected_polygon_override: Optional[Polygon] = None
    needs_contour_pick: bool = False
    arrays: Dict[str, np.ndarray] = field(default_factory=dict)
    tables: Dict[str, object] = field(default_factory=dict)
