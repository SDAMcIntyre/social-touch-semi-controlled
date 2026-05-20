"""RF cluster metrics computation pipeline.

Reads extraction artifacts from output_dir and writes rf_metrics.json per cluster.
Never reads aggregated CSVs or the clustering directory.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from analysis.receptive_field_mapping.rf_extraction_io import (
    RF_CAMERA_SETTINGS_FILENAME,
    load_forearm_vertices_artifact,
    load_neuron_touches,
    load_rf_camera_rotation,
    metrics_computation_is_up_to_date,
    save_metrics_computation_summary,
)
from analysis.receptive_field_mapping.rf_metrics import (
    compute_rf_metrics,
    metrics_to_dict,
    metrics_to_row,
)
from analysis.pipeline.shared_constants import GESTURE_TYPES

from .rf_cluster_pipeline import _build_pairs

logger = logging.getLogger(__name__)


def run_cluster_rf_metrics_computation(
    output_dir: Path,
    cluster_groups: list = None,
    cluster_group_defs: dict = None,
    feature_combinations: dict = None,
    clustering_profiles: dict = None,
    projection_method: Optional[str] = None,
    force: bool = False,
    slim_uv_cache_dir: Optional[Path] = None,
) -> None:
    """Compute RF metrics from extraction artifacts and write rf_metrics.json per cluster.

    Reads only from output_dir intermediate artifacts — never from aggregated
    CSVs or the clustering directory.

    Parameters
    ----------
    output_dir:
        Root of the RF cluster artifact directory (same as extraction output_dir).
    cluster_groups:
        List of group names.
    cluster_group_defs:
        Dict mapping group_name -> group_spec.
    feature_combinations:
        **DEPRECATED.**
    clustering_profiles:
        **DEPRECATED.**
    projection_method:
        2D projection method used for metric computation; ``None`` defaults to
        ``"tangent_plane"``.
    force:
        If True, re-compute even if metrics are up-to-date for these params.
    slim_uv_cache_dir:
        Directory containing per-session SLIM UV cache files.  ``None`` skips
        SLIM cache look-up (non-SLIM projection methods).
    """
    if cluster_groups is None and feature_combinations is None:
        raise ValueError(
            "run_cluster_rf_metrics_computation: either 'cluster_groups' or 'feature_combinations' must be provided."
        )

    camera_settings_dir = output_dir.parent / 'rf_camera_settings'

    pairs = _build_pairs(
        cluster_groups, cluster_group_defs, feature_combinations, clustering_profiles,
        caller="run_cluster_rf_metrics_computation",
    )

    for combo_name, clusterer_name in pairs:
        group_spec = (cluster_group_defs or {}).get(combo_name, {})
        per_type = group_spec.get('per_type_clustering', False)
        if per_type:
            runs = [
                (gt, output_dir / combo_name / clusterer_name / gt)
                for gt in GESTURE_TYPES
            ]
        else:
            runs = [(None, output_dir / combo_name / clusterer_name)]

        for gesture_type, base_output in runs:
            _type_label = f"/{gesture_type}" if gesture_type else ""
            print(f"[RF Metrics] {combo_name}/{clusterer_name}{_type_label}...")
            extraction_json = base_output / 'extraction_summary.json'

            if not extraction_json.exists():
                raise ValueError(
                    f"run_cluster_rf_metrics_computation: extraction_summary.json missing for "
                    f"{combo_name}/{clusterer_name}{_type_label} — run extraction first: {extraction_json}"
                )

            camera_settings_json = camera_settings_dir / RF_CAMERA_SETTINGS_FILENAME
            if metrics_computation_is_up_to_date(
                base_output,
                projection_method,
                force=force,
                camera_settings_path=camera_settings_json,
            ):
                print(f"  Metrics up-to-date, skipping.")
                continue

            extraction_mtime = extraction_json.stat().st_mtime

            try:
                neuron_touches = load_neuron_touches(base_output)
            except ValueError as exc:
                raise ValueError(
                    f"run_cluster_rf_metrics_computation: cannot load neuron_touches "
                    f"for {combo_name}/{clusterer_name}{_type_label}: {exc}"
                ) from exc

            metrics_rows: List[dict] = []

            for cluster_dir in sorted(base_output.glob('cluster_*')):
                cluster_label_folder = cluster_dir.name
                spike_counts_csv = cluster_dir / 'spike_counts.csv'
                cluster_desc_json = cluster_dir / 'cluster_description.json'

                if not spike_counts_csv.exists():
                    logger.warning("spike_counts.csv missing for %s, skipping.", cluster_label_folder)
                    continue

                try:
                    pooled_df = pd.read_csv(spike_counts_csv)
                except Exception:
                    logger.exception("Failed to read spike_counts.csv: %s", spike_counts_csv)
                    continue

                cluster_description_data: dict = {}
                if cluster_desc_json.exists():
                    try:
                        with open(cluster_desc_json) as _f:
                            cluster_description_data = json.load(_f)
                    except Exception:
                        pass

                cluster_label = cluster_description_data.get('cluster_label', cluster_label_folder)

                metrics_sid = None
                metrics_forearm_vertices = None
                for sid in neuron_touches:
                    try:
                        metrics_forearm_vertices = load_forearm_vertices_artifact(base_output, sid)
                        metrics_sid = sid
                        break
                    except ValueError:
                        pass

                metrics_rotation = None
                if metrics_sid is not None:
                    metrics_rotation = load_rf_camera_rotation(camera_settings_dir, metrics_sid)

                metrics_slim_cache = (
                    slim_uv_cache_dir / metrics_sid / f"{metrics_sid}_slim_uv.npz"
                    if slim_uv_cache_dir is not None and metrics_sid is not None
                    else None
                )
                metrics = compute_rf_metrics(
                    pooled_df,
                    metrics_forearm_vertices,
                    projection_method=projection_method or "tangent_plane",
                    rotation_matrix=metrics_rotation,
                    slim_cache_path=metrics_slim_cache,
                )
                metrics_json_path = cluster_dir / 'rf_metrics.json'
                with open(metrics_json_path, 'w') as _f:
                    json.dump(metrics_to_dict(metrics), _f, indent=2)
                metrics_rows.append(
                    metrics_to_row(metrics, cluster_label, combo_name, clusterer_name)
                )

            if metrics_rows:
                metrics_summary_csv = base_output / 'rf_metrics_summary.csv'
                pd.DataFrame(metrics_rows).to_csv(metrics_summary_csv, index=False)

            save_metrics_computation_summary(
                base_output,
                projection_method,
                extraction_mtime,
                camera_settings_mtime=camera_settings_json.stat().st_mtime if camera_settings_json.exists() else None,
            )

            print(f"[RF Metrics] {combo_name}/{clusterer_name}{_type_label}: done.")
