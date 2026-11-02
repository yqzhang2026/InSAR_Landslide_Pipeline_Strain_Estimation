from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from src.paths import program_dir


@dataclass
class AppConfig:
    # File inputs (relative to program_dir / data)
    dem_paths: str = "data/dem/ca2018_ql2_wldfires_dem_J1314375tR0_C0.tif;data/dem/ca2018_ql2_wldfires_dem_J1314516tR0_C0.tif"
    asc_shp_path: str = "data/insar/AOI_2_TSX_A76.shp"
    desc_shp_path: str = "data/insar/AOI_2_TSX_D38.shp"
    fault_line_shp_path: str = "data/fault/all_hayward.shp"
    pipe_axis_path: str = "data/pipe/PRP_Pipe_2020RMIDs.shp"
    output_dir: str = "data/results"

    # Optional references (for validation)
    ref_nez_path: str = "data/ref/nez_landslide_dem_2026_3_16.npy"
    ref_condition_path: str = "data/ref/condition_numbers.npy"
    ref_axial_strain_path: str = "data/ref/pipeline_axial_strain_reference.npy"

    # Matching
    date_match_max_days: int = 4
    spatial_match_max_m: float = 5.0
    dst_epsg: int = 3857

    # DEM
    sigma_px: float = 20.0
    dem_resolution_m: float = 1.0
    resampling_method: str = "bilinear"  # nearest | bilinear | cubic

    # LOS + constraint
    asc_heading_deg: float = 349.623233
    asc_inc_deg: float = 33.153514
    desc_heading_deg: float = 191.303877
    desc_inc_deg: float = 21.43727
    flip_constraint_sign: bool = False

    # Step2
    condition_threshold: float = 10.0

    # Step3
    day_idx: int = 141
    dir_idx: int = 1
    contour_threshold: float = -20.0
    filter_mode: str = "below"  # below | above
    k_neighbors: int = 16
    top_n: int = 5
    k_fallback: int = 8
    field_smooth_enable: bool = True
    gauss_bandwidth_m: float = 20.0
    mesh_size_m: float = 8.0
    pipeline_diameter_m: float = 0.9144
    diameter_npy_path: str = ""
    selected_lines_shp_name: str = "landslide1.shp"
    annualize: bool = False
    annualize_span_days: float = 3040.0
    annualize_days_per_year: float = 365.0

    # General
    save_intermediate: bool = True
    date_column_prefix: str = "D"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        valid = cls().__dict__.keys()
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)

    def resolve_path(self, raw_path: str) -> Path:
        p = Path(raw_path).expanduser()
        if p.is_absolute():
            return p
        return program_dir() / p

    def parse_dem_paths(self) -> List[str]:
        parts = [p.strip() for p in str(self.dem_paths).split(";") if p.strip()]
        if len(parts) < 2:
            raise ValueError("dem_paths 需要至少两个 tiff 路径，用分号 ';' 分隔。")
        return parts[:2]
