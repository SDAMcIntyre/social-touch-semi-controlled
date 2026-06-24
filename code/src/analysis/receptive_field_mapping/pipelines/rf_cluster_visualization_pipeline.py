"""RF cluster visualization pipeline.

Reads extraction artifacts from output_dir and renders per-session heatmap PNGs.
Never reads aggregated CSVs or the clustering directory.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.receptive_field_mapping.rendering.rf_cluster_visualizer import render_forearm_heatmap, RFRenderContext
from analysis.receptive_field_mapping.data.rf_extraction_io import (
    RF_CAMERA_SETTINGS_FILENAME,
    description_summary_line,
    load_cluster_session_data,
    load_neuron_cluster_touches,
    load_neuron_contacts,
    load_neuron_touches,
    load_rf_camera_settings,
    load_sessions_metadata,
    save_visualization_summary,
    visualization_is_up_to_date,
)
from analysis.pipeline.shared_constants import GESTURE_TYPES

from .rf_cluster_pipeline import _build_pairs

logger = logging.getLogger(__name__)


def run_cluster_rf_visualization(
    output_dir: Path,
    cluster_groups: list = None,
    cluster_group_defs: dict = None,
    feature_combinations: dict = None,
    clustering_profiles: dict = None,
    projection_method: Optional[str] = None,
    disjoint_mask_distance_mm: float = 8.0,
    force: bool = False,
    gallery_viewer: bool = False,
    input_items: List[Tuple[Path, Path]] = None,
    slim_uv_cache_dir: Optional[Path] = None,
) -> List[Path]:
    """Render heatmaps from extraction artifacts.

    Reads only from output_dir intermediate artifacts — never from aggregated
    CSVs or the clustering directory.

    Parameters
    ----------
    output_dir:
        Root of RF cluster artifact directory (same as extraction output_dir).
    cluster_groups:
        List of group names.
    cluster_group_defs:
        Dict mapping group_name -> group_spec.
    feature_combinations:
        **DEPRECATED.**
    clustering_profiles:
        **DEPRECATED.**
    projection_method:
        2D projection method; ``None`` for 3D rendering.
    disjoint_mask_distance_mm:
        NaN-mask distance for 2D heatmaps. Default 8.0 mm.
    force:
        If True, re-render even if visualization is up-to-date for these params.
    gallery_viewer:
        If True, launch the interactive gallery viewer after rendering each
        combo/clusterer pair.  Blocks until the user closes the window.
        Default False (existing behaviour).
    input_items:
        List of ``(aggregated_csv_path, database_path)`` tuples.  Unused by
        this function; accepted for forward-compatibility.

    Returns
    -------
    List of paths to produced PNG files.
    """
    if cluster_groups is None and feature_combinations is None:
        raise ValueError(
            "run_cluster_rf_visualization: either 'cluster_groups' or 'feature_combinations' must be provided."
        )

    if gallery_viewer:
        import warnings
        warnings.warn(
            "run_cluster_rf_visualization: 'gallery_viewer=True' is deprecated. "
            "Use the standalone 'explore_rf_gallery' DAG task instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning(
            "run_cluster_rf_visualization: 'gallery_viewer=True' is deprecated — "
            "use the standalone 'explore_rf_gallery' DAG task instead."
        )

    from analysis.pipeline.output_dirs import SPATIAL_SET_CAMERA
    camera_settings_dir = output_dir.parent / SPATIAL_SET_CAMERA

    pairs = _build_pairs(
        cluster_groups, cluster_group_defs, feature_combinations, clustering_profiles,
        caller="run_cluster_rf_visualization",
    )
    produced: List[Path] = []

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
            print(f"[RF Visualization] {combo_name}/{clusterer_name}{_type_label}...")
            extraction_json = base_output / 'extraction_summary.json'

            if not extraction_json.exists():
                raise ValueError(
                    f"run_cluster_rf_visualization: extraction_summary.json missing for "
                    f"{combo_name}/{clusterer_name}{_type_label} — run extraction first: {extraction_json}"
                )

            if visualization_is_up_to_date(
                base_output, projection_method, disjoint_mask_distance_mm, force=force,
                camera_settings_path=camera_settings_dir / RF_CAMERA_SETTINGS_FILENAME,
            ):
                print(f"  Visualization up-to-date, skipping.")
                for cluster_dir in sorted(base_output.glob('cluster_*')):
                    produced.extend(sorted(cluster_dir.glob('*_rf_heatmap_*.png')))
                if gallery_viewer:
                    from .rf_cluster_gui_launchers import launch_gallery_viewer
                    print(f"[RF Gallery] Launching gallery viewer for {combo_name}/{clusterer_name}{_type_label}...")
                    launch_gallery_viewer(output_dir, combo_name, clusterer_name, gesture_type)
                continue

            extraction_mtime = extraction_json.stat().st_mtime

            try:
                neuron_touches = load_neuron_touches(base_output)
            except ValueError as exc:
                raise ValueError(
                    f"run_cluster_rf_visualization: cannot load neuron_touches "
                    f"for {combo_name}/{clusterer_name}{_type_label}: {exc}"
                ) from exc

            try:
                sessions_metadata = load_sessions_metadata(base_output)
            except ValueError as exc:
                raise ValueError(
                    f"run_cluster_rf_visualization: cannot load sessions_metadata "
                    f"for {combo_name}/{clusterer_name}{_type_label}: {exc}"
                ) from exc

            cluster_dirs = sorted(base_output.glob('cluster_*'))

            for cluster_dir in cluster_dirs:
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
                description_line = description_summary_line(cluster_description_data)

                try:
                    neuron_cluster_touches = load_neuron_cluster_touches(cluster_dir)
                except ValueError:
                    neuron_cluster_touches = {}

                total_spikes = int(pooled_df['spike_count'].sum()) if not pooled_df.empty else 0
                if pooled_df.empty:
                    print(f"  -> No spikes for {cluster_label_folder}, skipping render.")
                    logger.warning(
                        "No spikes for cluster %s (%s/%s%s), skipping render.",
                        cluster_label, combo_name, clusterer_name, _type_label,
                    )
                else:
                    print(
                        f"  -> {cluster_label_folder}: {total_spikes} total spikes,"
                        f" {len(pooled_df)} unique contact points."
                    )

                # Per-session rendering
                sessions_dir = cluster_dir / 'sessions'
                if not sessions_dir.exists():
                    continue
                all_cameras = load_rf_camera_settings(camera_settings_dir)
                for session_subdir in sorted(sessions_dir.iterdir()):
                    session_id = session_subdir.name
                    if session_id not in all_cameras:
                        raise ValueError(
                            f"run_cluster_rf_visualization: session '{session_id}' not found in camera settings. "
                            "Run 'spatial_set_camera' first."
                        )
                    session_cam = all_cameras[session_id]
                    try:
                        session_spike_df, cluster_contacts_xyz = load_cluster_session_data(
                            cluster_dir, session_id
                        )
                    except ValueError as exc:
                        logger.warning("Skipping session %s render: %s", session_id, exc)
                        continue

                    if session_spike_df.empty:
                        continue

                    try:
                        neuron_xyz = load_neuron_contacts(base_output, session_id)
                    except ValueError as exc:
                        logger.warning("Cannot load neuron contacts for %s: %s", session_id, exc)
                        continue

                    # Hull invariant check
                    if len(neuron_xyz) > 0 and len(cluster_contacts_xyz) > 0:
                        _spike_mm = np.round(session_spike_df[['x', 'y', 'z']].to_numpy())
                        _spike_set = set(map(tuple, _spike_mm))
                        _cluster_set = set(map(tuple, np.round(cluster_contacts_xyz)))
                        _neuron_set = set(map(tuple, np.round(neuron_xyz)))
                        if not (_spike_set <= _cluster_set <= _neuron_set):
                            logger.warning(
                                "RF hull invariant violated for session=%s, cluster=%s",
                                session_id, cluster_label,
                            )

                    _ply_str = sessions_metadata.get(session_id, {}).get('forearm_ply')
                    forearm_ply = Path(_ply_str) if _ply_str else None
                    session_slim_cache = (
                        slim_uv_cache_dir / session_id / f"{session_id}_slim_uv.npz"
                        if slim_uv_cache_dir is not None
                        else None
                    )

                    render_context = RFRenderContext(
                        neuron_touches=neuron_touches.get(session_id, 0),
                        neuron_cluster_touches=neuron_cluster_touches.get(session_id, 0),
                        neuron_contacts_xyz=neuron_xyz,
                        neuron_cluster_contacts_xyz=cluster_contacts_xyz,
                        feature_ranges=cluster_description_data.get('feature_ranges', {}),
                    )

                    suffix = f'_{projection_method}' if projection_method else ''

                    print(f"    Rendering count heatmap: {session_id}...")
                    count_png = cluster_dir / f'{session_id}_rf_heatmap_count{suffix}.png'
                    try:
                        render_forearm_heatmap(
                            forearm_ply_path=forearm_ply,
                            spike_counts_df=session_spike_df,
                            output_path=count_png,
                            session_id=session_id,
                            cluster_label=str(cluster_label),
                            projection_method=projection_method,
                            cluster_description=description_line,
                            display_metric="spike_count",
                            render_context=render_context,
                            disjoint_mask_distance_mm=disjoint_mask_distance_mm,
                            camera_settings=session_cam,
                            slim_cache_path=session_slim_cache,
                        )
                        produced.append(count_png)
                    except Exception:
                        logger.exception(
                            "Failed to render count heatmap for session %s, cluster %s",
                            session_id, cluster_label,
                        )

                    if render_context.neuron_cluster_touches > 0:
                        print(f"    Rendering ratio heatmap: {session_id}...")
                        ratio_png = cluster_dir / f'{session_id}_rf_heatmap_ratio{suffix}.png'
                        try:
                            render_forearm_heatmap(
                                forearm_ply_path=forearm_ply,
                                spike_counts_df=session_spike_df,
                                output_path=ratio_png,
                                session_id=session_id,
                                cluster_label=str(cluster_label),
                                projection_method=projection_method,
                                cluster_description=description_line,
                                display_metric="spike_ratio",
                                render_context=render_context,
                                disjoint_mask_distance_mm=disjoint_mask_distance_mm,
                                camera_settings=session_cam,
                                slim_cache_path=session_slim_cache,
                            )
                            produced.append(ratio_png)
                        except Exception:
                            logger.exception(
                                "Failed to render ratio heatmap for session %s, cluster %s",
                                session_id, cluster_label,
                            )
                    else:
                        logger.info(
                            "Skipping ratio heatmap for session %s, cluster %s: neuron_cluster_touches=0.",
                            session_id, cluster_label,
                        )

            camera_settings_json = camera_settings_dir / RF_CAMERA_SETTINGS_FILENAME
            save_visualization_summary(
                base_output, projection_method, disjoint_mask_distance_mm, extraction_mtime,
                camera_settings_mtime=camera_settings_json.stat().st_mtime if camera_settings_json.exists() else None,
            )

            print(f"[RF Visualization] {combo_name}/{clusterer_name}{_type_label}: done.")
            logger.info(
                "[%s/%s%s] RF visualization complete.",
                combo_name, clusterer_name, _type_label,
            )

            if gallery_viewer:
                from .rf_cluster_gui_launchers import launch_gallery_viewer
                print(f"[RF Gallery] Launching gallery viewer for {combo_name}/{clusterer_name}{_type_label}...")
                launch_gallery_viewer(output_dir, combo_name, clusterer_name, gesture_type)

    return produced
