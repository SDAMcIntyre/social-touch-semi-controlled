import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from analysis.receptive_field_mapping.rf_data_loader import load_forearm_vertices
from analysis.receptive_field_mapping.touch_population_data import (
    load_population_data,
    load_population_rf_data,
)
from utils.should_process_task import should_process_task

logger = logging.getLogger(__name__)

_VALID_NEURON_MODES = ("iff", "spike")


@dataclass
class PopulationRFGridConfig:
    features: dict[str, dict[str, float]]
    neuron_mode: str
    vertex_threshold_ratio: float
    per_gesture_type: bool
    compute_baseline: bool = True


def build_feature_grid(config: PopulationRFGridConfig) -> np.ndarray:
    if not config.features:
        raise ValueError("build_feature_grid: config.features is empty")

    axes = []
    for name, fdef in config.features.items():
        centers = np.arange(fdef["min"], fdef["max"], fdef["step"])
        if len(centers) == 0:
            raise ValueError(
                f"build_feature_grid: feature '{name}' produces zero grid points "
                f"(min={fdef['min']}, max={fdef['max']}, step={fdef['step']})"
            )
        axes.append(centers)

    grid_points = list(itertools.product(*axes))
    return np.array(grid_points, dtype=np.float64)


def filter_touches_for_cell(
    feature_matrix: np.ndarray,
    center: np.ndarray,
    spans: np.ndarray,
) -> np.ndarray:
    T, F = feature_matrix.shape
    nan_mask = np.any(np.isnan(feature_matrix), axis=1)

    half = spans / 2.0
    lo = center - half
    hi = center + half

    in_bounds = np.all((feature_matrix >= lo) & (feature_matrix <= hi), axis=1)
    return in_bounds & ~nan_mask


def compute_cell_rf(
    rf_vertex_indices: list,
    rf_values: list,
    touch_mask: np.ndarray,
    n_vertices: int,
    threshold_ratio: float,
) -> np.ndarray:
    selected = np.where(touch_mask)[0]
    n_cell_touches = len(selected)

    rf_map = np.full(n_vertices, np.nan, dtype=np.float64)

    if n_cell_touches == 0:
        return rf_map

    accumulator = np.zeros(n_vertices, dtype=np.float64)
    contact_count = np.zeros(n_vertices, dtype=np.int64)
    unique_touch_count = np.zeros(n_vertices, dtype=np.int64)

    for idx in selected:
        verts = rf_vertex_indices[idx]
        vals = rf_values[idx]
        if len(verts) == 0:
            continue
        np.add.at(accumulator, verts, vals)
        np.add.at(contact_count, verts, 1)

        touched_verts = np.unique(verts)
        np.add.at(unique_touch_count, touched_verts, 1)

    contacted = contact_count > 0
    rf_map[contacted] = accumulator[contacted] / contact_count[contacted]

    min_touches = max(1, round(threshold_ratio * n_cell_touches))
    below_threshold = contacted & (unique_touch_count < min_touches)
    rf_map[below_threshold] = np.nan

    return rf_map


def _save_grid_results(
    output_path: Path,
    summary_path: Path,
    session_id: str,
    grid_centers: np.ndarray,
    feature_names: list,
    feature_configs: dict,
    rf_maps: np.ndarray,
    touch_counts: np.ndarray,
    touch_ids_list: list,
    gesture_type: str,
    neuron_mode: str,
    vertex_threshold_ratio: float,
) -> None:
    G = len(grid_centers)
    touch_ids_obj = np.empty(G, dtype=object)
    for i, arr in enumerate(touch_ids_list):
        touch_ids_obj[i] = arr

    data_dict: dict[str, Any] = {
        "grid_centers": grid_centers,
        "feature_names": np.array(feature_names, dtype=object),
        "feature_configs": np.array(feature_configs, dtype=object),
        "rf_maps": rf_maps,
        "touch_counts": touch_counts,
        "touch_ids": touch_ids_obj,
        "gesture_type": np.array(gesture_type, dtype=object),
        "neuron_mode": np.array(neuron_mode, dtype=object),
        "vertex_threshold_ratio": np.array(vertex_threshold_ratio, dtype=np.float64),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **data_dict)
    logger.info("Saved grid RF NPZ: %s", output_path)

    feature_step_counts = [
        len(np.arange(feature_configs[n]["min"], feature_configs[n]["max"], feature_configs[n]["step"]))
        for n in feature_names
    ]
    non_empty = int(np.sum(touch_counts > 0))
    summary = {
        "session_id": session_id,
        "feature_names": feature_names,
        "grid_shape": feature_step_counts,
        "total_cells": G,
        "non_empty_cells": non_empty,
        "neuron_mode": neuron_mode,
        "vertex_threshold_ratio": vertex_threshold_ratio,
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=4)
    logger.info("Saved grid summary JSON: %s", summary_path)


def _sweep_grid(
    grid_centers: np.ndarray,
    feature_matrix: np.ndarray,
    spans: np.ndarray,
    rf_vertex_indices: list,
    rf_values: list,
    n_vertices: int,
    threshold_ratio: float,
    touch_triple_keys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list]:
    G = len(grid_centers)
    rf_maps = np.full((G, n_vertices), np.nan, dtype=np.float64)
    touch_counts = np.zeros(G, dtype=np.int64)
    touch_ids_list = []

    for g, center in enumerate(grid_centers):
        mask = filter_touches_for_cell(feature_matrix, center, spans)
        n_in_cell = int(mask.sum())
        touch_counts[g] = n_in_cell

        if n_in_cell > 0:
            rf_maps[g] = compute_cell_rf(
                rf_vertex_indices, rf_values, mask, n_vertices, threshold_ratio
            )
            selected_keys = touch_triple_keys[mask]
            touch_ids_list.append(selected_keys.astype(np.int64))
        else:
            touch_ids_list.append(np.empty((0, 3), dtype=np.int64))

    return rf_maps, touch_counts, touch_ids_list


def _compute_and_save_baseline(
    touch_mask: np.ndarray,
    rf_vertex_indices: list,
    rf_values: list,
    n_vertices: int,
    config: PopulationRFGridConfig,
    output_dir: Path,
    baseline_type: str,
    session_id: str,
) -> None:
    """Compute a baseline RF map for a given touch mask and save it as a single-map NPZ.

    Saves to ``output_dir / f"population_rf_grid_baseline_{safe_type}.npz"``.
    Skips saving when touch_count == 0 (no touches for this baseline type).
    """
    touch_count = int(touch_mask.sum())
    if touch_count == 0:
        logger.info(
            "_compute_and_save_baseline: no touches for baseline_type=%r in session %r — skipping",
            baseline_type,
            session_id,
        )
        return

    rf_map = compute_cell_rf(
        rf_vertex_indices,
        rf_values,
        touch_mask,
        n_vertices,
        config.vertex_threshold_ratio,
    )

    safe_type = baseline_type.replace(" ", "_").lower()
    filename = f"population_rf_grid_baseline_{safe_type}.npz"
    output_path = output_dir / filename

    np.savez(
        output_path,
        rf_map=rf_map,
        touch_count=np.int64(touch_count),
        neuron_mode=np.array(config.neuron_mode, dtype=object),
        vertex_threshold_ratio=np.float64(config.vertex_threshold_ratio),
        baseline_type=np.array(baseline_type, dtype=object),
        session_id=np.array(session_id, dtype=object),
    )
    logger.info("Saved baseline RF NPZ (%s): %s", baseline_type, output_path)


def run_population_rf_grid(
    input_items: list,
    output_dir: Path,
    config: PopulationRFGridConfig,
    force: bool = False,
) -> list[Path]:
    if config.neuron_mode not in _VALID_NEURON_MODES:
        raise ValueError(
            f"run_population_rf_grid: invalid neuron_mode {config.neuron_mode!r}. "
            f"Expected one of {_VALID_NEURON_MODES}."
        )

    if not config.features:
        raise ValueError("run_population_rf_grid: config.features is empty")

    feature_names = list(config.features.keys())
    spans = np.array([config.features[n]["span"] for n in feature_names], dtype=np.float64)

    for name, fdef in config.features.items():
        if fdef["span"] <= 0:
            raise ValueError(
                f"run_population_rf_grid: feature '{name}' has span={fdef['span']} — "
                f"must be positive"
            )

    grid_centers = build_feature_grid(config)
    G = len(grid_centers)

    if G > 10_000:
        logger.warning(
            "run_population_rf_grid: grid has %d cells (>10,000) — "
            "this may be slow and memory-intensive",
            G,
        )

    produced: list[Path] = []

    for item in input_items:
        series_csv_path = Path(item["series_csv_path"])
        npz_path = Path(item["npz_path"])
        forearm_ply_path = Path(item["forearm_ply_path"])
        session_id = item["session_id"]
        touch_features_dir = item.get("touch_features_dir")

        session_out_dir = output_dir / "population_rf_grid" / session_id

        if config.per_gesture_type:
            sentinel_paths = sorted(session_out_dir.glob("population_rf_grid_*_summary.json"))
        else:
            sentinel_paths = [session_out_dir / "population_rf_grid_summary.json"]

        if sentinel_paths and not should_process_task(
            input_paths=[series_csv_path, npz_path],
            output_paths=sentinel_paths,
            force=force,
        ):
            print(f"[Population RF Grid] {session_id}: up-to-date, skipping.")
            produced.extend(sentinel_paths)
            continue

        logger.info("Processing session: %s", session_id)

        pop_data = load_population_data(series_csv_path, forearm_ply_path, touch_features_dir=touch_features_dir)
        n_vertices = len(pop_data.forearm_vertices)
        rf_data = load_population_rf_data(npz_path, pop_data.touch_triple_keys, n_vertices)

        if rf_data.neuron_mode != config.neuron_mode:
            raise ValueError(
                f"run_population_rf_grid: config.neuron_mode={config.neuron_mode!r} "
                f"does not match RF data neuron_mode={rf_data.neuron_mode!r} "
                f"in {npz_path}"
            )

        feature_matrix_cols = []
        for name in feature_names:
            try:
                col = pop_data.get_feature_array(name)
            except KeyError as exc:
                raise ValueError(
                    f"run_population_rf_grid: feature '{name}' not found in session "
                    f"'{session_id}'. Available: {pop_data.feature_names}"
                ) from exc
            feature_matrix_cols.append(col)

        feature_matrix = np.column_stack(feature_matrix_cols)

        session_out_dir.mkdir(parents=True, exist_ok=True)

        if config.compute_baseline:
            # Global baseline: all touches pooled
            global_mask = np.ones(len(pop_data.touch_triple_keys), dtype=bool)
            _compute_and_save_baseline(
                touch_mask=global_mask,
                rf_vertex_indices=rf_data.rf_vertex_indices,
                rf_values=rf_data.rf_values,
                n_vertices=n_vertices,
                config=config,
                output_dir=session_out_dir,
                baseline_type="global",
                session_id=session_id,
            )

            # Per-gesture-type baselines
            unique_gesture_types = np.unique(pop_data.gesture_types)
            for gtype in unique_gesture_types:
                gtype_str = str(gtype)
                gtype_mask = pop_data.gesture_types == gtype
                _compute_and_save_baseline(
                    touch_mask=gtype_mask,
                    rf_vertex_indices=rf_data.rf_vertex_indices,
                    rf_values=rf_data.rf_values,
                    n_vertices=n_vertices,
                    config=config,
                    output_dir=session_out_dir,
                    baseline_type=gtype_str,
                    session_id=session_id,
                )

        if not config.per_gesture_type:
            rf_maps, touch_counts, touch_ids_list = _sweep_grid(
                grid_centers,
                feature_matrix,
                spans,
                rf_data.rf_vertex_indices,
                rf_data.rf_values,
                n_vertices,
                config.vertex_threshold_ratio,
                pop_data.touch_triple_keys,
            )
            _save_grid_results(
                output_path=session_out_dir / "population_rf_grid.npz",
                summary_path=session_out_dir / "population_rf_grid_summary.json",
                session_id=session_id,
                grid_centers=grid_centers,
                feature_names=feature_names,
                feature_configs=config.features,
                rf_maps=rf_maps,
                touch_counts=touch_counts,
                touch_ids_list=touch_ids_list,
                gesture_type="",
                neuron_mode=config.neuron_mode,
                vertex_threshold_ratio=config.vertex_threshold_ratio,
            )
            produced.append(session_out_dir / "population_rf_grid_summary.json")
        else:
            unique_gesture_types = np.unique(pop_data.gesture_types)
            for gtype in unique_gesture_types:
                gtype_str = str(gtype)
                gtype_mask = pop_data.gesture_types == gtype

                gtype_feature_matrix = feature_matrix[gtype_mask]
                gtype_rf_vertex_indices = [
                    rf_data.rf_vertex_indices[i]
                    for i in np.where(gtype_mask)[0]
                ]
                gtype_rf_values = [
                    rf_data.rf_values[i]
                    for i in np.where(gtype_mask)[0]
                ]
                gtype_triple_keys = pop_data.touch_triple_keys[gtype_mask]

                rf_maps, touch_counts, touch_ids_list = _sweep_grid(
                    grid_centers,
                    gtype_feature_matrix,
                    spans,
                    gtype_rf_vertex_indices,
                    gtype_rf_values,
                    n_vertices,
                    config.vertex_threshold_ratio,
                    gtype_triple_keys,
                )

                safe_gtype = gtype_str.replace(" ", "_")
                summary_path = session_out_dir / f"population_rf_grid_{safe_gtype}_summary.json"
                _save_grid_results(
                    output_path=session_out_dir / f"population_rf_grid_{safe_gtype}.npz",
                    summary_path=summary_path,
                    session_id=session_id,
                    grid_centers=grid_centers,
                    feature_names=feature_names,
                    feature_configs=config.features,
                    rf_maps=rf_maps,
                    touch_counts=touch_counts,
                    touch_ids_list=touch_ids_list,
                    gesture_type=gtype_str,
                    neuron_mode=config.neuron_mode,
                    vertex_threshold_ratio=config.vertex_threshold_ratio,
                )
                produced.append(summary_path)

    return produced
