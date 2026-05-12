import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.receptive_field_mapping.rf_baseline_deviation import compute_baseline_deviation
from analysis.receptive_field_mapping.rf_data_loader import load_forearm_vertices
from analysis.receptive_field_mapping.rf_extraction_io import (
    RF_CAMERA_SETTINGS_FILENAME,
    load_rf_camera_rotation,
)
from analysis.receptive_field_mapping.rf_grid_cell_metrics import compute_grid_cell_metrics
from utils.should_process_task import should_process_task

logger = logging.getLogger(__name__)

_REQUIRED_NPZ_KEYS = (
    "rf_maps",
    "grid_centers",
    "feature_names",
    "touch_counts",
    "neuron_mode",
    "gesture_type",
    "vertex_threshold_ratio",
)

_REQUIRED_BASELINE_NPZ_KEYS = (
    "rf_map",
    "touch_count",
    "baseline_type",
)

_DEVIATION_IDENTITY = {
    "deviation_area_ratio": 1.0,
    "deviation_hotspot_area_ratio": 1.0,
    "deviation_centroid_shift_mm": 0.0,
    "deviation_centroid_shift_normalized": 0.0,
    "deviation_mean_iff_ratio": 1.0,
    "deviation_peak_iff_ratio": 1.0,
    "deviation_vertex_overlap_jaccard": 1.0,
    "deviation_vertex_containment": 1.0,
}


@dataclass
class PopulationRFGridMetricsConfig:
    projection_method: str = "tangent_plane"


def _load_grid_npz(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)

    missing = [k for k in _REQUIRED_NPZ_KEYS if k not in data]
    if missing:
        raise ValueError(
            f"_load_grid_npz: NPZ file is missing required keys {missing}: {npz_path}"
        )

    raw_names = data["feature_names"].tolist()
    feature_names = [
        n.decode() if isinstance(n, bytes) else str(n) for n in raw_names
    ]

    return {
        "rf_maps": data["rf_maps"],
        "grid_centers": data["grid_centers"],
        "feature_names": feature_names,
        "touch_counts": data["touch_counts"],
        "neuron_mode": str(data["neuron_mode"].item()),
        "gesture_type": str(data["gesture_type"].item()),
        "vertex_threshold_ratio": float(data["vertex_threshold_ratio"].item()),
    }


def _load_baseline_npz(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)

    missing = [k for k in _REQUIRED_BASELINE_NPZ_KEYS if k not in data]
    if missing:
        raise ValueError(
            f"_load_baseline_npz: NPZ file is missing required keys {missing}: {npz_path}"
        )

    return {
        "rf_map": data["rf_map"],
        "touch_count": int(data["touch_count"].item()),
        "baseline_type": str(data["baseline_type"].item()),
    }


def _compute_baseline_metrics(
    rf_map: np.ndarray,
    forearm_vertices: np.ndarray,
    projection_method: str,
    rotation_matrix: np.ndarray = None,
) -> dict:
    return compute_grid_cell_metrics(rf_map, forearm_vertices, projection_method, rotation_matrix=rotation_matrix)


def _build_metrics_dataframe(
    rf_maps: np.ndarray,
    grid_centers: np.ndarray,
    feature_names: list,
    touch_counts: np.ndarray,
    gesture_type: str,
    neuron_mode: str,
    session_id: str,
    forearm_vertices: np.ndarray,
    projection_method: str,
    baseline_metrics: dict | None = None,
    baseline_rf_map: np.ndarray | None = None,
    rotation_matrix: np.ndarray = None,
) -> pd.DataFrame:
    G = rf_maps.shape[0]
    rows = []

    for g in range(G):
        if touch_counts[g] == 0:
            continue

        if (g + 1) % 100 == 0:
            logger.info("Processing cell %d/%d", g + 1, G)

        row: dict = {
            "grid_cell_index": g,
            "session_id": session_id,
            "gesture_type": gesture_type,
            "neuron_mode": neuron_mode,
            "touch_count": int(touch_counts[g]),
        }

        for f_idx, fname in enumerate(feature_names):
            row[f"{fname}_center"] = float(grid_centers[g, f_idx])

        metrics = compute_grid_cell_metrics(
            rf_maps[g],
            forearm_vertices,
            projection_method,
            rotation_matrix=rotation_matrix,
        )

        row.update(metrics)

        if baseline_metrics is not None and baseline_rf_map is not None:
            deviation = compute_baseline_deviation(
                condition_metrics=metrics,
                baseline_metrics=baseline_metrics,
                condition_rf_map=rf_maps[g],
                baseline_rf_map=baseline_rf_map,
            )
            row.update(deviation)

        rows.append(row)

    return pd.DataFrame(rows)


def _build_baseline_comparison_dataframe(
    global_baseline: dict,
    global_baseline_metrics: dict,
    gesture_baselines: dict[str, dict],
    gesture_baseline_metrics: dict[str, dict],
) -> pd.DataFrame:
    rows = []

    global_row = {
        "gesture_type": "global",
        "baseline_type": global_baseline["baseline_type"],
        "touch_count": global_baseline["touch_count"],
    }
    global_row.update(global_baseline_metrics)
    global_row.update(_DEVIATION_IDENTITY)
    rows.append(global_row)

    for gtype, baseline in gesture_baselines.items():
        b_metrics = gesture_baseline_metrics[gtype]
        deviation = compute_baseline_deviation(
            condition_metrics=b_metrics,
            baseline_metrics=global_baseline_metrics,
            condition_rf_map=baseline["rf_map"],
            baseline_rf_map=global_baseline["rf_map"],
        )
        row = {
            "gesture_type": gtype,
            "baseline_type": baseline["baseline_type"],
            "touch_count": baseline["touch_count"],
        }
        row.update(b_metrics)
        row.update(deviation)
        rows.append(row)

    return pd.DataFrame(rows)


def run_population_rf_grid_metrics(
    input_items: list,
    output_dir: Path,
    config: PopulationRFGridMetricsConfig,
    force: bool = False,
    group_name: str | None = None,
) -> list:
    produced = []

    for item in input_items:
        forearm_ply_path: Path = item["forearm_ply_path"]
        grid_dir: Path = item["grid_dir"]
        session_id: str = item["session_id"]

        npz_paths = sorted(grid_dir.glob("population_rf_grid_*.npz"))
        if not npz_paths:
            raise ValueError(
                f"run_population_rf_grid_metrics: no NPZ files found in {grid_dir}"
            )

        if group_name is not None:
            output_session_dir = output_dir / "population_rf_grid_metrics" / group_name / session_id
        else:
            output_session_dir = output_dir / "population_rf_grid_metrics" / session_id
        sentinel = output_session_dir / "population_rf_grid_metrics_summary.json"

        camera_settings_dir = output_dir / 'rf_camera_settings'
        camera_settings_json = camera_settings_dir / RF_CAMERA_SETTINGS_FILENAME
        extra_inputs = [camera_settings_json] if camera_settings_json.exists() else []

        if not should_process_task(
            input_paths=list(npz_paths) + extra_inputs,
            output_paths=[sentinel],
            force=force,
        ):
            produced.append(sentinel)
            continue

        forearm_vertices = load_forearm_vertices(forearm_ply_path)
        if forearm_vertices is None:
            raise ValueError(
                f"run_population_rf_grid_metrics: could not load forearm vertices "
                f"from {forearm_ply_path}"
            )

        session_R = load_rf_camera_rotation(output_dir / 'rf_camera_settings', session_id)

        baseline_npz_paths = sorted(grid_dir.glob("population_rf_grid_baseline_*.npz"))
        has_baseline = len(baseline_npz_paths) > 0

        global_baseline = None
        global_baseline_metrics = None
        gesture_baselines: dict[str, dict] = {}
        gesture_baseline_metrics: dict[str, dict] = {}

        if has_baseline:
            global_path = grid_dir / "population_rf_grid_baseline_global.npz"
            if not global_path.exists():
                raise ValueError(
                    f"run_population_rf_grid_metrics: baseline NPZ files found in "
                    f"{grid_dir} but global baseline is missing: {global_path}"
                )
            global_baseline = _load_baseline_npz(global_path)
            global_baseline_metrics = _compute_baseline_metrics(
                global_baseline["rf_map"],
                forearm_vertices,
                config.projection_method,
                rotation_matrix=session_R,
            )

            for npz_path in baseline_npz_paths:
                stem = npz_path.stem
                if stem == "population_rf_grid_baseline_global":
                    continue
                prefix = "population_rf_grid_baseline_"
                gtype = stem[len(prefix):]
                gesture_baselines[gtype] = _load_baseline_npz(npz_path)
                gesture_baseline_metrics[gtype] = _compute_baseline_metrics(
                    gesture_baselines[gtype]["rf_map"],
                    forearm_vertices,
                    config.projection_method,
                    rotation_matrix=session_R,
                )
        else:
            logging.warning(
                "No baseline NPZ files found in %s — deviation metrics skipped",
                grid_dir,
            )

        grid_npz_paths = [
            p for p in npz_paths
            if not p.stem.startswith("population_rf_grid_baseline_")
        ]

        for npz_path in grid_npz_paths:
            npz_data = _load_grid_npz(npz_path)
            gesture_type_str = npz_data["gesture_type"]

            if has_baseline and gesture_type_str == "all_gestures":
                b_metrics = global_baseline_metrics
                b_rf_map = global_baseline["rf_map"]
            elif has_baseline:
                b_metrics = gesture_baseline_metrics.get(gesture_type_str)
                b_rf_map = gesture_baselines[gesture_type_str]["rf_map"] if gesture_type_str in gesture_baselines else None
            else:
                b_metrics = None
                b_rf_map = None

            df = _build_metrics_dataframe(
                rf_maps=npz_data["rf_maps"],
                grid_centers=npz_data["grid_centers"],
                feature_names=npz_data["feature_names"],
                touch_counts=npz_data["touch_counts"],
                gesture_type=gesture_type_str,
                neuron_mode=npz_data["neuron_mode"],
                session_id=session_id,
                forearm_vertices=forearm_vertices,
                projection_method=config.projection_method,
                baseline_metrics=b_metrics,
                baseline_rf_map=b_rf_map,
                rotation_matrix=session_R,
            )

            csv_name = f"population_rf_grid_metrics_{gesture_type_str}.csv"

            output_session_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_session_dir / csv_name
            df.to_csv(csv_path, index=False)
            logger.info("Saved %s", csv_path)

        if has_baseline and global_baseline is not None and global_baseline_metrics is not None:
            comparison_df = _build_baseline_comparison_dataframe(
                global_baseline=global_baseline,
                global_baseline_metrics=global_baseline_metrics,
                gesture_baselines=gesture_baselines,
                gesture_baseline_metrics=gesture_baseline_metrics,
            )
            comparison_path = output_session_dir / "rf_baseline_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            logger.info("Saved %s", comparison_path)

        summary = {
            "session_id": session_id,
            "n_gesture_types": len(grid_npz_paths),
            "has_baseline": has_baseline,
        }
        with open(sentinel, "w") as fh:
            json.dump(summary, fh, indent=4)

        produced.append(sentinel)

    return produced
