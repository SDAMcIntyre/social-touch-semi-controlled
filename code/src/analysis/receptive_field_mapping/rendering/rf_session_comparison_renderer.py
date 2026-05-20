import json
import logging
import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage

matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from analysis.receptive_field_mapping.rf_population_grid_pipeline import (
    run_population_rf_grid,
    PopulationRFGridConfig,
)
from analysis.receptive_field_mapping.rf_population_grid_metrics_pipeline import (
    run_population_rf_grid_metrics,
    PopulationRFGridMetricsConfig,
)

logger = logging.getLogger(__name__)


def _build_session_feature_matrix(
    sessions_df: dict[str, pd.DataFrame],
    metric_name: str,
) -> pd.DataFrame:
    rows = {}
    for session_id, df in sessions_df.items():
        center_cols = [c for c in df.columns if c.endswith("_center")]
        if len(center_cols) != 1:
            raise ValueError(
                f"_build_session_feature_matrix: expected exactly 1 '_center' column "
                f"for session '{session_id}', found {len(center_cols)}: {center_cols}"
            )
        center_col = center_cols[0]
        rows[session_id] = df.set_index(center_col)[metric_name]

    matrix = pd.DataFrame(rows).T
    matrix.index = pd.Index(sorted(matrix.index), name="session_id")
    matrix = matrix[sorted(matrix.columns)]
    return matrix


def _nan_safe_correlation_distance(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            mask = np.isfinite(matrix[i]) & np.isfinite(matrix[j])
            if mask.sum() < 2:
                distances.append(1.0)
                continue
            a = matrix[i][mask]
            b = matrix[j][mask]
            if a.std() == 0.0 or b.std() == 0.0:
                distances.append(1.0)
                continue
            r = np.corrcoef(a, b)[0, 1]
            distances.append(float(np.clip(1.0 - r, 0.0, 2.0)))
    return np.array(distances, dtype=float)


def _cluster_session_rows(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    all_nan_mask = np.all(~np.isfinite(matrix), axis=1)
    clusterable = matrix[~all_nan_mask]
    nan_rows = matrix[all_nan_mask]

    if clusterable.shape[0] < 3:
        return matrix, None

    dist = _nan_safe_correlation_distance(clusterable)
    Z = linkage(dist, method='average', optimal_ordering=True)
    order = leaves_list(Z)
    reordered = np.concatenate([clusterable[order], nan_rows], axis=0)
    return reordered, Z


def render_session_comparison_heatmap(
    matrix: pd.DataFrame,
    metric_name: str,
    feature_name: str,
    gesture_type: str,
    output_path: Path,
    vmin: float | None,
    vmax: float | None,
    linkage_matrix: np.ndarray | None = None,
) -> None:
    n_sessions = len(matrix.index)
    n_bins = len(matrix.columns)

    fig_h = max(4.0, 0.6 * n_sessions + 1.5)
    fig_w = max(8.0, n_bins * 0.4 + 3.0)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="black")

    if linkage_matrix is not None:
        fig_w += 1.5
        fig = plt.figure(figsize=(fig_w, fig_h), layout="constrained")
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 5], figure=fig)
        ax_dendro = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1])

        dendrogram(
            linkage_matrix,
            orientation='left',
            color_threshold=0,
            above_threshold_color='#444444',
            no_labels=True,
            ax=ax_dendro,
        )
        ax_dendro.set_axis_off()
        ax_dendro.set_ylim(0, n_sessions * 10)
    else:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), layout="constrained")

    im = ax.pcolormesh(
        matrix.columns.values,
        np.arange(n_sessions),
        matrix.values,
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_yticks(np.arange(n_sessions) + 0.5)
    ax.set_yticklabels(list(matrix.index))
    ax.set_ylim(0, n_sessions)

    ax.set_xlabel(feature_name.replace("_", " "))
    ax.set_title(f"{gesture_type} / {metric_name}")

    plt.colorbar(im, ax=ax, label=metric_name.replace("_", " "))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Saved session comparison heatmap: %s", output_path)


def run_session_comparison_visualization(
    input_items: list,
    database_path: Path,
    features: dict,
    metric_name: str = "mean_iff",
    neuron_mode: str = "iff",
    vertex_threshold_ratio: float = 0.25,
    projection_method: str = "tangent_plane",
    force: bool = False,
) -> None:
    if len(features) != 1:
        raise ValueError(
            f"run_session_comparison_visualization: expected exactly 1 feature, "
            f"got {len(features)}: {list(features)}"
        )

    comparison_root = database_path / "4_analysed" / "session_comparison"
    output_heatmaps = comparison_root / "session_comparison_heatmaps" / metric_name
    sentinel = output_heatmaps / "summary.json"

    grid_config = PopulationRFGridConfig(
        features=features,
        neuron_mode=neuron_mode,
        vertex_threshold_ratio=vertex_threshold_ratio,
        per_gesture_type=True,
        compute_baseline=False,
    )

    run_population_rf_grid(input_items, comparison_root, grid_config, force)

    # load_rf_camera_rotation raises ValueError when settings are absent.
    # run_population_rf_grid_metrics reads from output_dir / 'rf_camera_settings'.
    # We create a symlink so the metrics pipeline finds the authoritative settings
    # that were set via the RF camera settings viewer.
    camera_settings_src = database_path / "4_analysed" / "rf_camera_settings"
    camera_settings_dst = comparison_root / "rf_camera_settings"
    if not camera_settings_dst.exists() and camera_settings_src.exists():
        try:
            os.symlink(str(camera_settings_src), str(camera_settings_dst),
                       target_is_directory=True)
            logger.info(
                "Created symlink %s -> %s", camera_settings_dst, camera_settings_src
            )
        except (OSError, NotImplementedError):
            # Symlinks may require elevated privileges on Windows; copy the file instead.
            import shutil
            camera_settings_dst.mkdir(parents=True, exist_ok=True)
            for entry in camera_settings_src.iterdir():
                shutil.copy2(str(entry), str(camera_settings_dst / entry.name))
            logger.info(
                "Copied camera settings from %s to %s",
                camera_settings_src, camera_settings_dst,
            )

    metrics_input_items = [
        {
            **item,
            "grid_dir": comparison_root / "population_rf_grid" / item["session_id"],
        }
        for item in input_items
    ]

    metrics_config = PopulationRFGridMetricsConfig(projection_method=projection_method)
    run_population_rf_grid_metrics(metrics_input_items, comparison_root, metrics_config, force)

    metrics_base = comparison_root / "population_rf_grid_metrics"
    all_csv_paths = [
        p
        for item in input_items
        for p in sorted(
            (metrics_base / item["session_id"]).glob("population_rf_grid_metrics_*.csv")
        )
    ]

    if not force and sentinel.exists():
        if all_csv_paths and all(
            sentinel.stat().st_mtime >= p.stat().st_mtime for p in all_csv_paths
        ):
            return

    # Group loaded CSVs by gesture type
    by_gesture: dict[str, dict[str, pd.DataFrame]] = {}

    for item in input_items:
        session_id: str = item["session_id"]
        session_dir = metrics_base / session_id
        for csv_path in sorted(session_dir.glob("population_rf_grid_metrics_*.csv")):
            stem = csv_path.stem
            prefix = "population_rf_grid_metrics_"
            gesture_type = stem[len(prefix):]

            df = pd.read_csv(csv_path)

            center_cols = [c for c in df.columns if c.endswith("_center")]
            if len(center_cols) != 1:
                raise ValueError(
                    f"run_session_comparison_visualization: expected exactly 1 '_center' "
                    f"column in {csv_path}, found {len(center_cols)}: {center_cols}"
                )

            if metric_name not in df.columns:
                logger.warning(
                    "Metric '%s' not found in %s — skipping session '%s' for gesture '%s'",
                    metric_name, csv_path, session_id, gesture_type,
                )
                continue

            by_gesture.setdefault(gesture_type, {})[session_id] = df

    if not by_gesture:
        logger.warning(
            "run_session_comparison_visualization: no usable CSVs found — skipping heatmap rendering"
        )
        return

    all_values = []
    for gesture_sessions in by_gesture.values():
        for df in gesture_sessions.values():
            all_values.append(df[metric_name].values)

    flat = np.concatenate(all_values)
    finite = flat[np.isfinite(flat)]
    global_vmin = float(np.nanmin(finite)) if len(finite) > 0 else None
    global_vmax = float(np.nanmax(finite)) if len(finite) > 0 else None

    feature_name = next(iter(features))
    rendered_gesture_types = []

    for gesture_type, gesture_sessions in by_gesture.items():
        if not gesture_sessions:
            logger.warning(
                "run_session_comparison_visualization: no sessions have data "
                "for gesture '%s' — skipping",
                gesture_type,
            )
            continue

        matrix = _build_session_feature_matrix(gesture_sessions, metric_name)
        arr = matrix.values
        all_nan_mask = np.all(~np.isfinite(arr), axis=1)
        clusterable_idx = np.where(~all_nan_mask)[0]
        nan_idx = np.where(all_nan_mask)[0]
        reordered_values, linkage_matrix = _cluster_session_rows(arr)
        if linkage_matrix is not None:
            order = leaves_list(linkage_matrix)
            row_order = np.concatenate([clusterable_idx[order], nan_idx])
        else:
            row_order = np.arange(len(matrix.index))
        matrix = pd.DataFrame(
            reordered_values,
            columns=matrix.columns,
            index=matrix.index[row_order],
        )
        output_path = output_heatmaps / f"{gesture_type}.png"

        render_session_comparison_heatmap(
            matrix=matrix,
            metric_name=metric_name,
            feature_name=feature_name,
            gesture_type=gesture_type,
            output_path=output_path,
            vmin=global_vmin,
            vmax=global_vmax,
            linkage_matrix=linkage_matrix,
        )
        rendered_gesture_types.append(gesture_type)

    output_heatmaps.mkdir(parents=True, exist_ok=True)
    with open(sentinel, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "status": "complete",
                "metric": metric_name,
                "gesture_types": sorted(rendered_gesture_types),
                "session_count": len(input_items),
            },
            fh,
            indent=4,
        )
    logger.info("Wrote sentinel: %s", sentinel)
