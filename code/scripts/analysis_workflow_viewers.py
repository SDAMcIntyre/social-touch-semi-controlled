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


def _discover_processed_groups(output_dir: Path) -> dict:
    """Scan *output_dir* for completed RF cluster extraction artifacts.

    Returns ``{combo_name: {clusterer_name: {'per_type': bool, 'gesture_types': list}}}``.
    Only entries whose ``extraction_summary.json`` sentinel exists are included.
    """
    from analysis.pipeline import GESTURE_TYPES

    groups: dict = {}
    if not output_dir.is_dir():
        return groups

    for combo_dir in sorted(output_dir.iterdir()):
        if not combo_dir.is_dir():
            continue
        clusterers: dict = {}
        for clusterer_dir in sorted(combo_dir.iterdir()):
            if not clusterer_dir.is_dir():
                continue
            if (clusterer_dir / "extraction_summary.json").is_file():
                clusterers[clusterer_dir.name] = {
                    "per_type": False,
                    "gesture_types": [],
                }
                continue
            found_types = [
                gt for gt in GESTURE_TYPES
                if (clusterer_dir / gt).is_dir()
                and (clusterer_dir / gt / "extraction_summary.json").is_file()
            ]
            if found_types:
                clusterers[clusterer_dir.name] = {
                    "per_type": True,
                    "gesture_types": found_types,
                }
        if clusterers:
            groups[combo_dir.name] = clusterers

    return groups


@flow(name="explore_rf_gallery")
def explore_rf_gallery_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    cluster_groups: list = None,
) -> None:
    """
    Launch the RF Cluster Gallery Viewer for each enabled cluster group / clusterer pair.

    Discovers available groups and clusterers from disk by scanning the extraction
    output directory for ``extraction_summary.json`` sentinels.  The *cluster_groups*
    list (from the viewer DAG config) selects which groups to open.
    """
    print(f"[Batch Analysis] Launching RF Gallery Viewer for {len(input_items)} item(s)...")
    if not input_items:
        return

    if cluster_groups is None:
        raise ValueError(
            "explore_rf_gallery_flow: 'cluster_groups' is required."
        )

    database_path = input_items[0][1]
    output_dir = database_path / '4_analysed' / 'receptive_field_maps_clustered'
    available = _discover_processed_groups(output_dir)

    for combo_name in cluster_groups:
        if combo_name not in available:
            raise ValueError(
                f"explore_rf_gallery_flow: group '{combo_name}' not found on disk. "
                f"Scanned: {output_dir}\n"
                f"Available processed groups: {sorted(available.keys())}"
            )
        for clusterer_name, info in available[combo_name].items():
            if info["per_type"]:
                for gesture_type in info["gesture_types"]:
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

    Loads the per-session ``_population_response_fields.npz`` produced by
    ``extract_population_rf_response_field_boundaries`` and presents a rotatable 3D surface showing
    RF topography. Requires ``extract_population_rf_response_field_boundaries`` to have run first.
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
