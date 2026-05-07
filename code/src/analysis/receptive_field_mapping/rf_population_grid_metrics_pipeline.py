import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.receptive_field_mapping.rf_data_loader import load_forearm_vertices
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
        )

        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


def run_population_rf_grid_metrics(
    input_items: list,
    output_dir: Path,
    config: PopulationRFGridMetricsConfig,
    force: bool = False,
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

        output_session_dir = output_dir / "population_rf_grid_metrics" / session_id
        sentinel = output_session_dir / "population_rf_grid_metrics_summary.json"

        if not should_process_task(
            input_paths=npz_paths,
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

        for npz_path in npz_paths:
            npz_data = _load_grid_npz(npz_path)
            df = _build_metrics_dataframe(
                rf_maps=npz_data["rf_maps"],
                grid_centers=npz_data["grid_centers"],
                feature_names=npz_data["feature_names"],
                touch_counts=npz_data["touch_counts"],
                gesture_type=npz_data["gesture_type"],
                neuron_mode=npz_data["neuron_mode"],
                session_id=session_id,
                forearm_vertices=forearm_vertices,
                projection_method=config.projection_method,
            )

            gesture_type_str = npz_data["gesture_type"]
            csv_name = f"population_rf_grid_metrics_{gesture_type_str}.csv"

            output_session_dir.mkdir(parents=True, exist_ok=True)
            csv_path = output_session_dir / csv_name
            df.to_csv(csv_path, index=False)
            logger.info("Saved %s", csv_path)

        summary = {"session_id": session_id, "n_gesture_types": len(npz_paths)}
        with open(sentinel, "w") as fh:
            json.dump(summary, fh, indent=4)

        produced.append(sentinel)

    return produced
