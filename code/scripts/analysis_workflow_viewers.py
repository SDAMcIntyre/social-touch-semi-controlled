# analysis_workflow_viewers.py
# Self-contained entry point for viewer / viewer_support analysis tasks.
# Viewer flows and dispatch are defined here; no dependency on analysis_workflow_processing.py.
import argparse
import logging
from pathlib import Path
from multiprocessing import Queue, freeze_support
from typing import List, Tuple

from prefect import flow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from utils import path_tools, DagConfigHandler, PipelineMonitor
from utils.pipeline.session_config_resolver import resolve_session_configs
from analysis.receptive_field_mapping import (
    precompute_explorer_caches,
    launch_feature_space_explorer,
    launch_touch_playback_explorer,
    launch_touch_population_explorer,
    launch_single_touch_rf_explorer,
    launch_gallery_viewer,
    launch_rf_surface_viewer,
)
from analysis.touch_analytics.gui import launch_preparation_viewer
from analysis.pipeline import (
    collect_unique_session_dirs,
    discover_input_items,
    run_pipeline_stages,
)


@flow(name="precompute_explorer_caches")
def precompute_explorer_caches_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    max_workers: int = 4,
) -> None:
    """Pre-compute .npz sidecar caches for the RF Feature-Space Explorer.

    Runs ``load_explorer_data()`` for each session in a thread pool so that
    the subsequent ``explore_rf_feature_space`` GUI task opens instantly on a
    cache hit.  Cache invalidation is handled internally by
    ``load_explorer_data``; ``force_processing`` is accepted for interface
    consistency but is a no-op here.
    """
    print(f"[Batch Analysis] Pre-computing RF explorer caches for {len(input_items)} item(s)...")
    if not input_items:
        return

    precompute_explorer_caches(input_items, max_workers=max_workers)


@flow(name="explore_rf_feature_space")
def explore_rf_feature_space_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
) -> None:
    """
    Launch the RF Feature-Space Explorer GUI for the given sessions.
    Reads series-augmented CSVs produced by ``touch_series_transforms`` — no
    dependency on RF clustering or visualization.
    ``force_processing`` is accepted for interface consistency but is a no-op:
    the GUI is stateless and always launches fresh.
    """
    print(f"[Batch Analysis] Launching RF Feature-Space Explorer for {len(input_items)} item(s)...")
    if not input_items:
        return

    launch_feature_space_explorer(input_items)


@flow(name="explore_touch_playback")
def explore_touch_playback_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
) -> None:
    """
    Launch the Touch Playback Explorer GUI for the given sessions.
    Reads series-augmented CSVs produced by ``touch_series_transforms`` — no
    dependency on RF clustering or visualization.
    ``force_processing`` is accepted for interface consistency but is a no-op:
    the GUI is stateless and always launches fresh.
    """
    print(f"[Batch Analysis] Launching Touch Playback Explorer for {len(input_items)} item(s)...")
    if not input_items:
        return

    launch_touch_playback_explorer(input_items)


@flow(name="explore_touch_population")
def explore_touch_population_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    neuron_mode: str = "iff",
) -> None:
    """
    Launch the Touch Population Explorer GUI for the given sessions.
    Reads series-augmented CSVs produced by ``touch_series_transforms`` — no
    dependency on RF clustering or visualization.
    ``force_processing`` is accepted for interface consistency but is a no-op:
    the GUI is stateless and always launches fresh.

    ``neuron_mode`` is forwarded to ``launch_touch_population_explorer`` to
    determine which pre-computed RF maps to load (``"iff"`` or ``"spike"``).
    """
    print(f"[Batch Analysis] Launching Touch Population Explorer for {len(input_items)} item(s)...")
    if not input_items:
        return

    launch_touch_population_explorer(input_items, neuron_mode=neuron_mode)


@flow(name="explore_single_touch_rf")
def explore_single_touch_rf_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    neuron_mode: str = "iff",
) -> None:
    """
    Launch the Single-Touch RF Explorer GUI for the given sessions.
    Reads per-touch RF maps produced by ``map_single_touch_rf`` — no dependency
    on clustering or visualization.
    ``force_processing`` is accepted for interface consistency but is a no-op:
    the GUI is stateless and always launches fresh.
    """
    print(f"[Batch Analysis] Launching Single-Touch RF Explorer for {len(input_items)} item(s)...")
    if not input_items:
        return

    launch_single_touch_rf_explorer(input_items, neuron_mode=neuron_mode)


@flow(name="explore_rf_gallery")
def explore_rf_gallery_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    cluster_groups: list = None,
    cluster_group_defs: dict = None,
) -> None:
    """
    Launch the RF Cluster Gallery Viewer for each enabled cluster group / clusterer pair.

    Reads extraction artifacts produced by ``visualize_receptive_fields_clustered`` — one
    interactive PyQt5 window per combo/clusterer pair, opened sequentially.
    ``force_processing`` is accepted for interface consistency but is a no-op:
    the viewer is stateless and always launches fresh.
    """
    print(f"[Batch Analysis] Launching RF Gallery Viewer for {len(input_items)} item(s)...")
    if not input_items:
        return

    if cluster_groups is None or cluster_group_defs is None:
        raise ValueError(
            "explore_rf_gallery_flow: 'cluster_groups' and 'cluster_group_defs' are required."
        )

    database_path = input_items[0][1]
    output_dir = database_path / '4_analysed' / 'receptive_field_maps_clustered'

    for combo_name in cluster_groups:
        if combo_name not in cluster_group_defs:
            raise ValueError(
                f"explore_rf_gallery_flow: group '{combo_name}' not found in cluster_group_defs."
            )
        group_spec = cluster_group_defs[combo_name]
        per_type = group_spec.get('per_type_clustering', False)
        clustering_methods = group_spec.get('clustering_methods', {})
        for clusterer_name, clusterer_cfg in clustering_methods.items():
            if not clusterer_cfg.get('enabled', True):
                continue
            if per_type:
                from analysis.touch_analytics.clustering_pipeline import GESTURE_TYPES
                for gesture_type in GESTURE_TYPES:
                    print(
                        f"[RF Gallery] Launching viewer for "
                        f"{combo_name}/{clusterer_name}/{gesture_type}..."
                    )
                    launch_gallery_viewer(output_dir, combo_name, clusterer_name, gesture_type)
            else:
                print(f"[RF Gallery] Launching viewer for {combo_name}/{clusterer_name}...")
                launch_gallery_viewer(output_dir, combo_name, clusterer_name)


@flow(name="explore_preparation")
def explore_preparation_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
) -> None:
    """
    Launch the Touch Preparation Viewer GUI for the given sessions.
    Reads prepared CSVs produced by ``touch_preparation`` — no dependency on
    series transforms or feature extraction.
    ``force_processing`` is accepted for interface consistency but is a no-op:
    the GUI is stateless and always launches fresh.
    """
    print(f"[Batch Analysis] Launching Touch Preparation Viewer for {len(input_items)} item(s)...")
    if not input_items:
        return

    launch_preparation_viewer(input_items)


@flow(name="explore_rf_surface")
def explore_rf_surface_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    neuron_mode: str = "iff",
) -> None:
    """Interactive 3D RF surface viewer: Z = mean IFF, coloured by jet colormap.

    Loads the per-session ``_rf_population_vertex_data.npz`` produced by
    ``visualize_population_rf_maps`` and presents a rotatable 3D surface showing
    RF topography. Requires ``visualize_population_rf_maps`` to have run first.
    """
    print(f"[Batch Analysis] Launching RF surface viewer for {len(input_items)} item(s)...")
    if not input_items:
        return
    launch_rf_surface_viewer(input_items, neuron_mode=neuron_mode)


def main():
    freeze_support()
    parser = argparse.ArgumentParser(
        description=(
            "Analysis workflow — viewer tasks only. "
            "Runs tasks with category 'viewer' or 'viewer_support' from the given "
            "DAG config. Processing outputs (CSVs, heatmaps, etc.) are assumed to "
            "exist on disk; use analysis_workflow_processing.py to generate them."
        )
    )
    parser.add_argument(
        "--dag-config",
        type=Path,
        default=Path("configs/analyse_workflow_viewers_dag.yaml"),
        help="Path to the DAG config YAML (default: configs/analyse_workflow_viewers_dag.yaml)",
    )
    args = parser.parse_args()

    dag_config_path: Path = args.dag_config
    project_data_root = path_tools.get_project_data_root()
    configs_dir = Path("configs")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_file_path = reports_dir / "analysis_viewers_status.xlsx"

    if not dag_config_path.exists():
        raise FileNotFoundError(f"DAG config not found at {dag_config_path}")

    dag_handler = DagConfigHandler(dag_config_path)
    entries = dag_handler.get_parameter('kinect_configs')
    block_files = resolve_session_configs(entries, configs_dir / "kinect_configs")

    session_map = collect_unique_session_dirs(block_files, project_data_root)

    if not session_map:
        logging.warning("No valid session directories found. Exiting.")
        return

    items_to_process = discover_input_items(session_map)

    if not items_to_process:
        logging.warning("No input files found. Exiting.")
        return

    _clustering_options = dag_handler.get_task_options("touch_clustering") or {}
    _cluster_group_defs = _clustering_options.get("cluster_groups") or {}

    task_names = [
        "precompute_explorer_caches",
        "explore_preparation",
        "explore_rf_feature_space",
        "explore_touch_playback",
        "explore_single_touch_rf",
        "explore_touch_population",
        "explore_rf_gallery",
        "explore_rf_surface",
    ]
    monitor = PipelineMonitor(report_path=report_file_path, stages=task_names, data_queue=Queue())

    pipeline_stages = [
        {
            "name": "precompute_explorer_caches",
            "func": precompute_explorer_caches_flow,
            "params": lambda: (
                {"max_workers": int(dag_handler.get_task_options("precompute_explorer_caches").get("max_workers", 4))}
                if "max_workers" in (dag_handler.get_task_options("precompute_explorer_caches") or {})
                else {}
            ),
        },
        {
            "name": "explore_preparation",
            "func": explore_preparation_flow,
            "params": lambda: {},
        },
        {
            "name": "explore_rf_feature_space",
            "func": explore_rf_feature_space_flow,
            "params": lambda: {},
        },
        {
            "name": "explore_touch_playback",
            "func": explore_touch_playback_flow,
            "params": lambda: {},
        },
        {
            "name": "explore_single_touch_rf",
            "func": explore_single_touch_rf_flow,
            "params": lambda: (
                {"neuron_mode": dag_handler.get_task_options("explore_single_touch_rf").get("neuron_mode", "iff")}
                if "neuron_mode" in (dag_handler.get_task_options("explore_single_touch_rf") or {})
                else {}
            ),
        },
        {
            "name": "explore_touch_population",
            "func": explore_touch_population_flow,
            "params": lambda: {
                "neuron_mode": dag_handler.get_task_options("explore_touch_population").get("neuron_mode", "iff"),
            },
        },
        {
            "name": "explore_rf_gallery",
            "func": explore_rf_gallery_flow,
            "params": lambda: {
                "cluster_groups": dag_handler.get_task_options("explore_rf_gallery").get("cluster_groups"),
                "cluster_group_defs": _cluster_group_defs,
            },
        },
        {
            "name": "explore_rf_surface",
            "func": explore_rf_surface_flow,
            "params": lambda: {
                "neuron_mode": dag_handler.get_task_options("explore_rf_surface").get("neuron_mode", "iff"),
            },
        },
    ]

    run_pipeline_stages(pipeline_stages, dag_handler, monitor, items_to_process, "batch_run_viewers")

    logging.info("Batch viewer run finished.")


if __name__ == "__main__":
    main()
