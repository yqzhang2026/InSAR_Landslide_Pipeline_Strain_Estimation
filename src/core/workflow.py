from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from src.config import AppConfig
from src.core.contracts import PipelineState, Step1Result, Step2Result, Step3Result
from src.core.dem_aspect import aspect_from_dem, build_dem_aspect, sample_aspect_at_points, smooth_dem
from src.core.insar_nez import load_insar_data, match_insar_observations, solve_nez_with_aspect
from src.core.pipeline_mapping import (
    build_step3_exports,
    build_step3_grid,
    build_mesh_from_selected_lines,
    compute_step3_response_from_mesh,
    save_selected_lines_from_contour,
)
from src.core.validation import compare_if_exists
from src.io.opensra_export import export_opensra_layers
from src.io.writers import export_pipeline_results


def _emit(progress: Optional[Callable[[int, str], None]], pct: int, msg: str):
    if progress:
        progress(int(pct), msg)


def run_step1_1(config: AppConfig, state: Optional[Dict] = None, progress: Optional[Callable[[int, str], None]] = None) -> Dict:
    _emit(progress, 10, "Loading and merging DEM")
    dem = build_dem_aspect(config, progress=progress)
    step1 = Step1Result(
        merged_dem=dem.merged_dem,
        dem_transform=dem.transform,
        dst_epsg=int(config.dst_epsg),
    )
    new_state = PipelineState(config=config, step1=step1).__dict__
    _emit(progress, 100, "Step1-1 completed")
    return new_state


def run_step1_2(config: AppConfig, state: Dict, progress: Optional[Callable[[int, str], None]] = None) -> Dict:
    if not state or state.get("step1") is None:
        raise ValueError("Step1-2 requires Step1-1 results.")
    step1: Step1Result = state["step1"]
    _emit(progress, 30, "Computing raw aspect from merged DEM")
    step1.aspect_raw = aspect_from_dem(step1.merged_dem, step1.dem_transform)
    _emit(progress, 100, "Step1-2 completed")
    return state


def run_step1_3(config: AppConfig, state: Dict, progress: Optional[Callable[[int, str], None]] = None) -> Dict:
    if not state or state.get("step1") is None or state["step1"].aspect_raw is None:
        raise ValueError("Step1-3 requires Step1-2 results.")
    step1: Step1Result = state["step1"]
    _emit(progress, 20, "Smoothing DEM and recomputing aspect")
    step1.dem_smoothed = smooth_dem(step1.merged_dem, sigma_px=float(config.sigma_px))
    step1.aspect_smooth = aspect_from_dem(step1.dem_smoothed, step1.dem_transform)
    if bool(config.save_intermediate):
        out = Path(config.resolve_path(config.output_dir))
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "dem_merged.npy", step1.merged_dem)
        np.save(out / "aspect_raw.npy", step1.aspect_raw)
        np.save(out / "aspect_smooth.npy", step1.aspect_smooth)
    _emit(progress, 100, "Step1-3 completed")
    return state


def run_step1(config: AppConfig, state: Optional[Dict] = None, progress: Optional[Callable[[int, str], None]] = None) -> Dict:
    state = run_step1_1(config, state=state, progress=lambda p, m: _emit(progress, int(p * 0.4), m))
    state = run_step1_2(config, state, progress=lambda p, m: _emit(progress, 40 + int(p * 0.3), m))
    state = run_step1_3(config, state, progress=lambda p, m: _emit(progress, 70 + int(p * 0.3), m))
    _emit(progress, 100, "Step1 completed")
    return state


def run_step2_1(config: AppConfig, state: Dict, progress: Optional[Callable[[int, str], None]] = None) -> Dict:
    if not state or state.get("step1") is None or state["step1"].aspect_raw is None:
        raise ValueError("Step2 requires Step1-2/1-3 results.")
    step1: Step1Result = state["step1"]
    _emit(progress, 10, "Loading and matching InSAR data")
    asc, desc = load_insar_data(config)
    matched = match_insar_observations(config, asc, desc)
    aspect_used = step1.aspect_smooth if step1.aspect_smooth is not None else step1.aspect_raw
    if aspect_used is None:
        raise ValueError("No aspect available. Run Step1-2 at least once.")
    _emit(progress, 15, "Sampling aspect at matched InSAR points")
    insar_aspect, valid_idx = sample_aspect_at_points(aspect_used, step1.dem_transform, matched["matched_coords"])
    step1.matched_dates = matched["matched_pairs"]
    step1.matched_coords = matched["matched_coords"]
    step1.matched_asc_data = matched["matched_asc_data"]
    step1.matched_desc_data = matched["matched_desc_data"]
    step1.day_diffs = matched["day_diffs"]
    step1.insar_aspect = insar_aspect
    step1.valid_idx = valid_idx
    _emit(progress, 100, "Step2-1 completed")
    return state


def run_step2_2(config: AppConfig, state: Dict, progress: Optional[Callable[[int, str], None]] = None) -> Dict:
    if not state or state.get("step1") is None:
        raise ValueError("Step2-2 requires Step2-1 results.")
    step1: Step1Result = state["step1"]
    if step1.insar_aspect is None or step1.valid_idx is None:
        raise ValueError("Step2-2 requires sampled aspect and valid_idx from Step2-1.")
    _emit(progress, 5, "Solving NEZ displacement with DEM aspect constraint")
    solved = solve_nez_with_aspect(
        config=config,
        matched={
            "matched_pairs": step1.matched_dates,
            "matched_asc_data": step1.matched_asc_data,
            "matched_desc_data": step1.matched_desc_data,
            "matched_coords": step1.matched_coords,
        },
        insar_aspect=step1.insar_aspect,
        valid_idx=step1.valid_idx,
        progress=lambda p, m: _emit(progress, 5 + int(p * 0.9), m),
    )
    step2 = Step2Result(
        nez=solved["nez"],
        condition_numbers=solved["condition_numbers"],
        remain_idx=solved["remain_idx"],
        coords_valid=solved["coords_valid"],
        coords_remain=solved["coords_remain"],
        valid_idx=solved["valid_idx"],
    )
    state["step2"] = step2
    _emit(progress, 100, "Step2-2 completed")
    return state


def run_step2_3(config: AppConfig, state: Dict, progress: Optional[Callable[[int, str], None]] = None) -> Dict:
    if not state or state.get("step1") is None or state.get("step2") is None:
        raise ValueError("Step2-3 requires Step2-2 results.")
    step1: Step1Result = state["step1"]
    step2: Step2Result = state["step2"]
    _emit(progress, 10, "Building Step2 outputs and validation tables")
    state["arrays"] = {
        "nez": step2.nez,
        "condition_numbers": step2.condition_numbers,
        "remain_idx": step2.remain_idx,
        "valid_idx": step2.valid_idx,
        "coords_valid": step2.coords_valid,
        "coords_remain": step2.coords_remain,
        "day_diffs": step1.day_diffs,
    }
    state["tables"] = {
        "matched_dates": pd.DataFrame(step1.matched_dates, columns=["asc_date", "desc_date"]),
        "condition_summary": pd.DataFrame(
            {
                "condition_number": step2.condition_numbers,
                "pass_threshold": step2.condition_numbers < float(config.condition_threshold),
            }
        ),
    }
    nez_metrics = compare_if_exists(step2.nez, str(config.resolve_path(config.ref_nez_path)))
    cond_metrics = compare_if_exists(step2.condition_numbers, str(config.resolve_path(config.ref_condition_path)))
    state["tables"]["validation_step2"] = pd.DataFrame(
        [
            {"target": "NEZ", **nez_metrics},
            {"target": "condition_numbers", **cond_metrics},
        ]
    )
    if bool(config.save_intermediate):
        out = Path(config.resolve_path(config.output_dir))
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "nez.npy", step2.nez)
        np.save(out / "condition_numbers.npy", step2.condition_numbers)
        np.save(out / "remain_idx.npy", step2.remain_idx)
    _emit(progress, 100, "Step2-3 completed")
    return state


def run_step2(config: AppConfig, state: Dict, progress: Optional[Callable[[int, str], None]] = None) -> Dict:
    state = run_step2_1(config, state, progress=lambda p, m: _emit(progress, int(p * 0.2), m))
    state = run_step2_2(config, state, progress=lambda p, m: _emit(progress, 20 + int(p * 0.6), m))
    state = run_step2_3(config, state, progress=lambda p, m: _emit(progress, 80 + int(p * 0.2), m))
    _emit(progress, 100, "Step2 completed")
    return state


def run_step3_pick(config: AppConfig, state: Dict, progress: Optional[Callable[[int, str], None]] = None) -> Dict:
    """Commit contour GIS after interactive pick. Clears stale mesh/response."""
    if not state or state.get("step2") is None:
        raise ValueError("Step3-1 Pick requires Step2 results.")
    _emit(progress, 20, "Validating contour selection output")
    selected_lines_path = state.get("step3_selected_lines_path")
    if not selected_lines_path:
        selected_polygon = state.get("selected_polygon_override")
        if selected_polygon is None:
            raise ValueError("Step3-1 requires contour selection first.")
        out_dir = Path(config.resolve_path(config.output_dir))
        out_info = save_selected_lines_from_contour(
            config=config,
            selected_polygon=selected_polygon,
            output_dir=out_dir,
            out_name=str(config.selected_lines_shp_name),
        )
        state["step3_selected_lines_path"] = out_info["selected_lines_path"]
        state["step3_selected_contour_path"] = out_info["selected_contour_path"]
    state.pop("step3_mesh", None)
    state.pop("step3", None)
    state["needs_contour_pick"] = False
    _emit(progress, 100, "Step3-1 pick committed")
    return state


def run_step3_mesh(config: AppConfig, state: Dict, progress: Optional[Callable[[int, str], None]] = None) -> Dict:
    if not state or state.get("step2") is None:
        raise ValueError("Step3-2 Mesh requires Step2 results.")
    selected_lines_path = state.get("step3_selected_lines_path")
    if not selected_lines_path:
        raise ValueError("Step3-2 requires Step3-1 contour GIS shapefile.")
    _emit(progress, 20, "Building pipeline mesh from selected contour GIS")
    mesh = build_mesh_from_selected_lines(
        selected_lines_path=Path(selected_lines_path),
        top_n=int(config.top_n),
        mesh_size=max(float(config.mesh_size_m), 1.0),
        progress=progress,
    )
    state.pop("step3", None)
    state["step3_mesh"] = mesh
    state.setdefault("arrays", {}).update(
        {
            "mesh_points": np.asarray(mesh["mesh_points"], dtype=float),
            "mesh_connectivity": np.asarray(mesh["mesh_connectivity"], dtype=int),
        }
    )
    state.setdefault("tables", {})["step3_mesh_summary"] = pd.DataFrame(
        [
            {
                "component_count": int(mesh["component_count"]),
                "kept_component_count": int(mesh["kept_component_count"]),
                "mesh_nodes": int(np.asarray(mesh["mesh_points"]).shape[0]),
                "mesh_elements": int(np.asarray(mesh["mesh_connectivity"]).shape[0]),
                "selected_lines_path": str(selected_lines_path),
            }
        ]
    )
    _emit(progress, 100, "Step3-2 mesh completed")
    return state


def run_step3_response(config: AppConfig, state: Dict, progress: Optional[Callable[[int, str], None]] = None) -> Dict:
    if not state or state.get("step2") is None:
        raise ValueError("Step3-3 Response requires Step2 results.")
    selected_lines_path = state.get("step3_selected_lines_path")
    if not selected_lines_path:
        raise ValueError("Step3-3 requires Step3-1 contour GIS shapefile.")
    mesh = state.get("step3_mesh")
    if mesh is None:
        raise ValueError("Step3-3 requires Step3-2 mesh results. Please run Step3-2 first.")
    step2: Step2Result = state["step2"]
    _emit(progress, 20, "Mapping displacement and computing strain on mesh")
    mapping = compute_step3_response_from_mesh(
        config=config,
        coords_remain=step2.coords_remain,
        nez_remain=step2.nez[:, step2.remain_idx, :],
        mesh=mesh,
    )
    gx, gy, gz, _, dir_idx = build_step3_grid(config, step2.coords_remain, step2.nez[:, step2.remain_idx, :])
    step3 = Step3Result(
        day_idx=int(mapping["day_idx"]),
        dir_idx=int(dir_idx),
        threshold=float(config.contour_threshold),
        selected_polygon=state.get("selected_polygon_override"),
        selected_lines_path=str(selected_lines_path),
        selected_pipe_coords=np.asarray(mapping["selected_pipe_coords"], dtype=float),
        mesh_connectivity=np.asarray(mapping["mesh_connectivity"], dtype=int),
        pipe_disp=np.asarray(mapping["pipe_disp"], dtype=float),
        axial_strain=np.asarray(mapping["axial_strain"], dtype=float),
        bending_strain=np.asarray(mapping["bending_strain"], dtype=float),
        grid_x=gx,
        grid_y=gy,
        grid_z=gz,
        export_paths={},
    )
    if bool(config.save_intermediate):
        export_paths = build_step3_exports(
            result={
                "selected_polygon": step3.selected_polygon,
                "selected_pipe_coords": step3.selected_pipe_coords,
                "pipe_disp": step3.pipe_disp,
                "axial_strain": step3.axial_strain,
                "bending_strain": step3.bending_strain,
            },
            dst_epsg=int(config.dst_epsg),
            output_dir=Path(config.resolve_path(config.output_dir)),
        )
        merged_path = Path(config.resolve_path(config.output_dir)) / f"top{int(config.top_n)}_networks_merged.shp"
        mesh["merged_networks_gdf"].to_file(merged_path)
        export_paths["top_networks_merged_shp"] = str(merged_path)
        step3.export_paths = export_paths
    state["step3"] = step3
    _emit(progress, 60, "Building Step3 tables and validation outputs")
    state.setdefault("arrays", {}).update(
        {
            "pipe_disp": step3.pipe_disp,
            "axial_strain": step3.axial_strain,
            "bending_strain": step3.bending_strain,
            "contour_grid_z": step3.grid_z,
            "mesh_points": step3.selected_pipe_coords,
            "mesh_connectivity": step3.mesh_connectivity,
        }
    )
    state.setdefault("tables", {}).update(
        {
            "pipeline_nodes": pd.DataFrame(
                {
                    "x": step3.selected_pipe_coords[:, 0],
                    "y": step3.selected_pipe_coords[:, 1],
                    "disp_n_mm": step3.pipe_disp[:, 0],
                    "disp_e_mm": step3.pipe_disp[:, 1],
                    "disp_z_mm": step3.pipe_disp[:, 2],
                    "bending_strain": step3.bending_strain,
                }
            ),
            "pipeline_axial_segments": pd.DataFrame(
                {
                    "segment_id": np.arange(step3.axial_strain.shape[0]),
                    "axial_strain": step3.axial_strain,
                }
            ),
        }
    )
    axial_metrics = compare_if_exists(step3.axial_strain, str(config.resolve_path(config.ref_axial_strain_path)))
    state["tables"]["validation_step3"] = pd.DataFrame([{"target": "axial_strain", **axial_metrics}])
    _emit(progress, 100, "Step3-3 response completed")
    return state


def run_step3_mesh_and_response(config: AppConfig, state: Dict, progress: Optional[Callable[[int, str], None]] = None) -> Dict:
    if state.get("step3_mesh") is None:
        state = run_step3_mesh(config, state, progress=lambda p, m: _emit(progress, int(p * 0.75), m))
    else:
        _emit(progress, 75, "Step3-2 mesh found; reusing existing mesh")
    state = run_step3_response(config, state, progress=lambda p, m: _emit(progress, 75 + int(p * 0.25), m))
    _emit(progress, 100, "Step3 mesh + response completed")
    return state


def run_step3(config: AppConfig, state: Dict, progress: Optional[Callable[[int, str], None]] = None) -> Dict:
    state = run_step3_pick(config, state, progress=lambda p, m: _emit(progress, int(p * 0.2), m))
    state = run_step3_mesh(config, state, progress=lambda p, m: _emit(progress, 20 + int(p * 0.4), m))
    state = run_step3_response(config, state, progress=lambda p, m: _emit(progress, 60 + int(p * 0.4), m))
    _emit(progress, 100, "Step3 completed")
    return state


def prepare_step3_contours(config: AppConfig, state: Dict) -> Dict:
    """Return epoch field used by the interactive tricontour picker (no griddata)."""
    if not state or state.get("step2") is None:
        raise ValueError("Contour selection requires Step2 results.")
    step2: Step2Result = state["step2"]
    nez_remain = np.asarray(step2.nez[:, step2.remain_idx, :], dtype=float)
    day_idx = int(np.clip(config.day_idx, 0, nez_remain.shape[0] - 1))
    dir_idx = int(np.clip(config.dir_idx, 0, 2))
    return {
        "day_idx": day_idx,
        "dir_idx": dir_idx,
        "coords": np.asarray(step2.coords_remain, dtype=float),
        "values": np.asarray(nez_remain[day_idx, :, dir_idx], dtype=float),
    }


def run_steps_1_and_2(config: AppConfig, state: Optional[Dict] = None, progress: Optional[Callable[[int, str], None]] = None) -> Dict:
    """Run All prefix: DEM + NEZ, then wait for interactive contour pick."""
    state = run_step1_1(config, state=state, progress=lambda p, m: _emit(progress, int(p * 0.25), m))
    state = run_step1_2(config, state=state, progress=lambda p, m: _emit(progress, 25 + int(p * 0.1), m))
    state = run_step1_3(config, state=state, progress=lambda p, m: _emit(progress, 35 + int(p * 0.1), m))
    state = run_step2(config, state=state, progress=lambda p, m: _emit(progress, 45 + int(p * 0.55), m))
    state["needs_contour_pick"] = True
    _emit(progress, 100, "Step1/2 completed; contour pick required")
    return state


def run_all(config: AppConfig, progress: Optional[Callable[[int, str], None]] = None) -> Dict:
    """Stops after Step2 unless a contour GIS path is already in config/state."""
    return run_steps_1_and_2(config, state=None, progress=progress)


def export_state(config: AppConfig, state: Dict, figures: Optional[Dict[str, object]] = None) -> Path:
    arrays = state.get("arrays", {})
    tables = state.get("tables", {})
    run_dir = export_pipeline_results(config=config, arrays=arrays, tables=tables, figures=figures)
    try:
        export_opensra_layers(config, state or {}, run_dir)
    except Exception:
        pass
    return run_dir


# Backward-compatible aliases (old Step3 numbering).
run_step3_1 = run_step3_pick
run_step3_2 = run_step3_mesh
run_step3_3 = run_step3_response
run_step3_23 = run_step3_mesh_and_response
