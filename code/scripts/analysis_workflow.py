# analyse_workflow.py
import argparse
import os
import logging
from pathlib import Path
from multiprocessing import Queue, freeze_support
from typing import List, Optional, Set, Dict, Tuple
from collections import defaultdict

# Prefect for workflow management
from prefect import flow

# Setup a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Mock/External Imports ---
from utils import path_tools
from utils import (
    DagConfigHandler,
    PipelineMonitor,
    TaskExecutor
)
from utils.pipeline.session_config_resolver import resolve_session_configs
from primary_processing import (
    KinectConfigFileHandler,
    KinectConfig,
)

# Imported from the updated touch_analysis module (assuming path matches 'analysis/touch_analytics')
from analysis.touch_analytics import (
    generate_ap_efficacy_matrix,
    generate_session_summary
)
from analysis.touch_analytics.preparation_pipeline import run_preparation
from analysis.touch_analytics.series_pipeline import run_series_transforms
from analysis.touch_analytics.extraction_pipeline import run_feature_extraction
from analysis.touch_analytics.clustering_pipeline import run_clustering
from analysis.touch_analytics.comparing_pipeline import run_comparing
from analysis.receptive_field_mapping import (
    run_cluster_rf_extraction,
    run_cluster_rf_mapping,
    run_cluster_rf_visualization,
    run_simple_rf_mapping,
    pick_rf_camera_angle_batch,
    precompute_explorer_caches,
    launch_feature_space_explorer,
    launch_touch_playback_explorer,
)
from analysis.touch_analytics.pipeline_shared import session_id_from_path

# --- Analysis Flows ---

@flow(name="summarize_session_blocks")
def summarize_session_blocks_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
) -> List[Path]:
    """
    STEP 1: Session Block Summary.
    Reads all aggregated session CSVs and writes a single combined CSV
    (``4_analysed/session_block_summary.csv``) with one row per block,
    describing block IDs, trial counts, and trial ID lists.
    """
    print(f"[Batch Analysis] Generating session block summary for {len(input_items)} item(s)...")

    if not input_items:
        logging.warning("No input items provided for session block summary.")
        return []

    input_paths = [item[0] for item in input_items]
    anchor_db_path = input_items[0][1]

    output_dir = anchor_db_path / "4_analysed" / "session_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "session_block_summary.csv"

    try:
        result_path = generate_session_summary(
            input_paths=input_paths,
            output_path=output_path,
            force=force_processing,
        )
        return [result_path]
    except Exception as exc:
        logging.error(f"Failed to generate session block summary: {exc}")
        return []


@flow(name="map_receptive_fields_simple")
def map_receptive_fields_simple_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    show_interactive: bool = False,
    projection_method: str = None,
) -> List[Path]:
    """
    Simple RF mapping: raw spike-position CSV + forearm heatmap per session.
    Runs before feature extraction — no dependency on clustering or extraction.
    Output: ``4_analysed/receptive_field_maps_simple/<session_id>/``
    """
    print(f"[Batch Analysis] Running simple RF mapping for {len(input_items)} item(s)...")
    if not input_items:
        return []
    output_dir = input_items[0][1] / '4_analysed' / 'receptive_field_maps_simple'
    return run_simple_rf_mapping(
        input_items=input_items,
        output_dir=output_dir,
        force=force_processing,
        show_interactive=show_interactive,
        projection_method=projection_method,
    )


@flow(name="touch_preparation")
def touch_preparation_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    preparation: dict = None,
) -> List[Path]:
    """
    Stage 1: Per-session data preparation (block-ID synthesis + NaN-gap interpolation).
    Writes ``4_analysed/preparation/<session_id>_prepared.csv``.
    """
    print(f"[Batch Analysis] Running touch preparation for {len(input_items)} item(s)...")
    if not input_items:
        return []
    output_dir = input_items[0][1] / '4_analysed' / 'preparation'
    return run_preparation(
        input_items=input_items,
        preparation_cfg=preparation or {},
        output_dir=output_dir,
        force=force_processing,
    )


@flow(name="touch_series_transforms")
def touch_series_transforms_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    transforms: dict = None,
    preparation_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Stage 2a: Per-session series-level transforms (kinematics, pressure, MoS).
    Writes ``4_analysed/series_transforms/<session>_series_augmented.csv``.
    """
    print(f"[Batch Analysis] Running touch series transforms for {len(input_items)} item(s)...")
    if not input_items:
        return []
    output_dir = input_items[0][1] / '4_analysed' / 'series_transforms'
    return run_series_transforms(
        input_items=input_items,
        transforms=transforms or {},
        output_dir=output_dir,
        force=force_processing,
        preparation_dir=preparation_dir,
    )


@flow(name="touch_feature_extraction")
def touch_feature_extraction_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    features: dict = None,
    series_dir: Optional[Path] = None,
    preparation_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Stage 2b: Per-session feature extraction.
    Writes one CSV per (session, feature) to
    ``4_analysed/touch_features/<feature>/<session>_touch_summary.csv``.
    """
    print(f"[Batch Analysis] Running touch feature extraction for {len(input_items)} item(s)...")
    if not input_items:
        return []
    feature_dict = features or {'max': {'enabled': True}}
    output_dir = input_items[0][1] / '4_analysed' / 'touch_features'
    per_feature = run_feature_extraction(
        input_items=input_items,
        features=feature_dict,
        output_dir=output_dir,
        force=force_processing,
        series_dir=series_dir,
        preparation_dir=preparation_dir,
    )
    return [path for paths in per_feature.values() for path in paths]


@flow(name="touch_clustering")
def touch_clustering_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    cluster_groups: dict = None,
    feature_combinations: dict = None,
    clustering_profiles: dict = None,
    reduction: dict = None,
    evaluation: dict = None,
) -> List[Path]:
    """
    Stage 2: Global clustering on pooled feature CSVs.
    Discovers CSVs written by touch_feature_extraction, merges per group,
    and writes
    ``4_analysed/touch_clusters/<group>/<clusterer>/pooled_touch_summary_clustered.csv``.
    """
    print(f"[Batch Analysis] Running touch clustering for {len(input_items)} item(s)...")
    if not input_items:
        return []
    extraction_dir = input_items[0][1] / '4_analysed' / 'touch_features'
    output_dir = input_items[0][1] / '4_analysed' / 'touch_clusters'
    per_key = run_clustering(
        output_dir=output_dir,
        cluster_groups=cluster_groups,
        feature_combinations=feature_combinations,
        clustering_profiles=clustering_profiles,
        force=force_processing,
        extraction_dir=extraction_dir,
        reduction=reduction,
        evaluation=evaluation,
    )
    return [path for paths in per_key.values() for path in paths]


@flow(name="touch_comparing")
def touch_comparing_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    cluster_groups: list = None,
    cluster_group_defs: dict = None,
    feature_combinations: dict = None,
    clustering_profiles: dict = None,
    comparing_profiles: dict = None,
    min_instances_per_sensor: int = 5,
    min_sensor_types: int = 2,
) -> List[Path]:
    """
    Stage 3: Statistical comparison of sensor measurements across strata.
    Discovers clustered CSVs written by touch_clustering and writes
    per-strategy result JSONs and a dispersion-weighted synthesis report.
    """
    print(f"[Batch Analysis] Running touch comparing for {len(input_items)} item(s)...")
    if not input_items:
        return []
    clustering_dir = input_items[0][1] / '4_analysed' / 'touch_clusters'
    output_dir = input_items[0][1] / '4_analysed' / 'touch_comparisons'
    return run_comparing(
        output_dir=output_dir,
        cluster_groups=cluster_groups,
        cluster_group_defs=cluster_group_defs,
        feature_combinations=feature_combinations,
        clustering_profiles=clustering_profiles,
        comparing_profiles=comparing_profiles,
        min_instances_per_sensor=min_instances_per_sensor,
        min_sensor_types=min_sensor_types,
        force=force_processing,
        clustering_dir=clustering_dir,
    )


@flow(name="analyse_ap_efficacy")
def analyse_ap_efficacy_flow(
    input_items: List[Tuple[Path, Path]], 
    force_processing: bool = False
) -> List[Path]:
    """
    STEP 3: Matrix Generation (Efficacy).
    Generates a matrix of AP efficacy into '4_analysed/ap_efficacy'.
    """
    print(f"[Batch Analysis - AP Efficacy] Generating Efficacy Matrix...")
    
    unified_files = _collect_unified_files(input_items)
    
    if unified_files:
        try:
            anchor_db_path = input_items[0][1]
            
            # Define specific subfolder for this analysis flow
            output_dir = anchor_db_path / "4_analysed" / "ap_efficacy"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            matrix_output_path = output_dir / "batch_ap_efficacy_matrix.csv"
            
            logging.info(f"Calling batch AP efficacy matrix generation. Output: {matrix_output_path}")
            # Delegate processing (and checks) to the function
            generate_ap_efficacy_matrix(
                unified_files, 
                matrix_output_path, 
                show=True, 
                force=force_processing
            )
            return [matrix_output_path]
        except Exception as e:
            logging.error(f"Failed to generate AP matrix: {e}")
            return []
    else:
        logging.warning("No unified summary files found. Run 'unified_touch_analysis' first.")
        return []

@flow(name="map_receptive_fields_clustered")
def map_receptive_fields_clustered_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    cluster_groups: list = None,
    cluster_group_defs: dict = None,
    feature_combinations: dict = None,
    clustering_profiles: dict = None,
    camera_angle_mode: bool = False,
    projection_method: str = None,
    disjoint_mask_distance_mm: float = 8.0,
) -> List[Path]:
    """
    Cluster-based RF mapping: spike-count heatmaps per cluster from touch_clustering output.
    Reads pooled_touch_summary_clustered.csv, forward-fills contact_points (30Hz->1kHz),
    counts spikes per (x,y,z) point, and renders 3D forearm heatmap PNGs.
    Output: ``4_analysed/receptive_field_maps_clustered/<group>/<clusterer>/``

    After mapping, assigns camera angles per session automatically when
    ``camera_angle_mode`` is ``True``.
    """
    print(f"[Batch Analysis] Running cluster-based RF mapping for {len(input_items)} item(s)...")
    if not input_items:
        return []

    database_path = input_items[0][1]
    clustering_dir = database_path / '4_analysed' / 'touch_clusters'
    output_dir = database_path / '4_analysed' / 'receptive_field_maps_clustered'

    result = run_cluster_rf_mapping(
        clustering_dir=clustering_dir,
        input_items=input_items,
        output_dir=output_dir,
        cluster_groups=cluster_groups,
        cluster_group_defs=cluster_group_defs,
        feature_combinations=feature_combinations,
        clustering_profiles=clustering_profiles,
        force=force_processing,
        projection_method=projection_method,
        disjoint_mask_distance_mm=disjoint_mask_distance_mm,
    )

    session_output_dirs = {
        session_id_from_path(csv_path): csv_path.parent
        for csv_path, _ in input_items
    }
    pick_rf_camera_angle_batch(
        session_output_dirs,
        force_processing=force_processing,
        enabled=camera_angle_mode,
    )

    return result


@flow(name="extract_receptive_fields_clustered")
def extract_receptive_fields_clustered_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    cluster_groups: list = None,
    cluster_group_defs: dict = None,
    feature_combinations: dict = None,
    clustering_profiles: dict = None,
) -> List[Path]:
    """
    Extraction-only step: load aggregated CSVs, count spikes per contact point,
    and write intermediate artifacts to disk.
    Output: ``4_analysed/receptive_field_maps_clustered/<group>/<clusterer>/``
    """
    print(f"[Batch Analysis] Running RF extraction for {len(input_items)} item(s)...")
    if not input_items:
        return []

    database_path = input_items[0][1]
    clustering_dir = database_path / '4_analysed' / 'touch_clusters'
    output_dir = database_path / '4_analysed' / 'receptive_field_maps_clustered'

    return run_cluster_rf_extraction(
        clustering_dir=clustering_dir,
        input_items=input_items,
        output_dir=output_dir,
        cluster_groups=cluster_groups,
        cluster_group_defs=cluster_group_defs,
        feature_combinations=feature_combinations,
        clustering_profiles=clustering_profiles,
        force=force_processing,
    )


@flow(name="visualize_receptive_fields_clustered")
def visualize_receptive_fields_clustered_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    cluster_groups: list = None,
    cluster_group_defs: dict = None,
    feature_combinations: dict = None,
    clustering_profiles: dict = None,
    camera_angle_mode: bool = False,
    projection_method: str = None,
    disjoint_mask_distance_mm: float = 8.0,
    gallery_viewer: bool = False,
) -> List[Path]:
    """
    Visualization-only step: load extraction artifacts, compute RF metrics,
    and render per-session heatmap PNGs.
    Output: ``4_analysed/receptive_field_maps_clustered/<group>/<clusterer>/``

    After rendering, assigns camera angles per session automatically when
    ``camera_angle_mode`` is ``True``.
    """
    print(f"[Batch Analysis] Running RF visualization for {len(input_items)} item(s)...")
    if not input_items:
        return []

    database_path = input_items[0][1]
    output_dir = database_path / '4_analysed' / 'receptive_field_maps_clustered'

    result = run_cluster_rf_visualization(
        output_dir=output_dir,
        cluster_groups=cluster_groups,
        cluster_group_defs=cluster_group_defs,
        feature_combinations=feature_combinations,
        clustering_profiles=clustering_profiles,
        projection_method=projection_method,
        disjoint_mask_distance_mm=disjoint_mask_distance_mm,
        force=force_processing,
        gallery_viewer=gallery_viewer,
        input_items=input_items,
    )

    session_output_dirs = {
        session_id_from_path(csv_path): csv_path.parent
        for csv_path, _ in input_items
    }
    pick_rf_camera_angle_batch(
        session_output_dirs,
        force_processing=force_processing,
        enabled=camera_angle_mode,
    )

    return result


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


def _collect_unified_files(input_items: List[Tuple[Path, Path]]) -> List[Path]:
    """
    Reconstruct expected paths of touch-summary CSVs written by touch_feature_extraction.
    Scans all extraction profile subdirectories under '4_analysed/unified_touches/'.
    Returns deduplicated list of existing paths.
    """
    seen: set = set()
    unified_files = []
    for input_file, database_path in input_items:
        unified_root = database_path / "4_analysed" / "touch_features"
        filename = input_file.name

        if "_semicontrolled_" in filename:
            prefix = filename.split("_semicontrolled_")[0]
            new_filename = f"{prefix}_semicontrolled_touch_summary.csv"
        else:
            new_filename = f"{input_file.stem}_touch_summary.csv"

        # Scan all profile subdirectories
        for profile_dir in sorted(unified_root.glob("*")):
            if not profile_dir.is_dir():
                continue
            candidate = profile_dir / new_filename
            if candidate.exists() and candidate not in seen:
                seen.add(candidate)
                unified_files.append(candidate)

    return unified_files

# --- Batch Processing Logic (Main) ---

def collect_unique_session_dirs(
    block_files: List[Path],
    project_data_root: Path,
) -> Dict[Path, Path]:
    session_dir_map = {}
    for block_file in block_files:
        try:
            config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
            config = KinectConfig(config_data=config_data, database_path=project_data_root)

            if config.session_merged_output_dir and config.database_path:
                session_dir_map[config.session_merged_output_dir] = config.database_path
        except Exception as e:
            logging.debug(f"Skipping {block_file.name}: {e}")

    logging.info(f"Scanned {len(block_files)} config files.")
    logging.info(f"Identified {len(session_dir_map)} unique session contexts.")

    return session_dir_map

def run_batch_analysis(
    block_files: List[Path],
    project_data_root: Path,
    dag_handler: DagConfigHandler,
    report_file_path: Path
):
    session_map = collect_unique_session_dirs(block_files, project_data_root)

    if not session_map:
        logging.warning("No valid session directories found. Exiting.")
        return

    available_tasks = [
        ("summarize_session_blocks", summarize_session_blocks_flow),
        ("map_receptive_fields_simple", map_receptive_fields_simple_flow),
        ("touch_preparation", touch_preparation_flow),
        ("touch_series_transforms", touch_series_transforms_flow),
        ("touch_feature_extraction", touch_feature_extraction_flow),
        ("touch_clustering", touch_clustering_flow),
        ("touch_comparing", touch_comparing_flow),
        ("analyse_ap_efficacy", analyse_ap_efficacy_flow),
        ("map_receptive_fields_clustered", map_receptive_fields_clustered_flow),
        ("extract_receptive_fields_clustered", extract_receptive_fields_clustered_flow),
        ("visualize_receptive_fields_clustered", visualize_receptive_fields_clustered_flow),
        ("precompute_explorer_caches", precompute_explorer_caches_flow),
        ("explore_rf_feature_space", explore_rf_feature_space_flow),
        ("explore_touch_playback", explore_touch_playback_flow),
    ]
    
    task_names = [t[0] for t in available_tasks]
    monitor = PipelineMonitor(report_path=report_file_path, stages=task_names, data_queue=Queue())

    items_to_process: List[Tuple[Path, Path]] = []

    logging.info(f"Scanning {len(session_map)} sessions for data files...")
    for search_dir in sorted(session_map.keys()):
        database_path_context = session_map[search_dir]
        candidates = list(search_dir.glob("*_semicontrolled_aggregated_session.csv"))

        if candidates:
            target_file = candidates[0]
            items_to_process.append((target_file, database_path_context))

    if not items_to_process:
        logging.warning("No input files found. Exiting.")
        return

    logging.info(f"🚀 Starting analysis for {len(items_to_process)} collected items.")

    # Pre-compute inter-stage artifact dirs for forwarding to downstream flows
    _database = items_to_process[0][1]

    def _task_enabled(name: str) -> bool:
        return bool(
            name in dag_handler.tasks
            and dag_handler.tasks[name].get("enabled", True)
        )

    preparation_dir: Optional[Path] = (
        _database / '4_analysed' / 'preparation' if _task_enabled("touch_preparation") else None
    )
    series_dir: Optional[Path] = (
        _database / '4_analysed' / 'series_transforms' if _task_enabled("touch_series_transforms") else None
    )

    # Extract cluster group defs from touch_clustering to forward to downstream tasks.
    _clustering_options = dag_handler.get_task_options("touch_clustering") or {}
    _cluster_group_defs = _clustering_options.get("cluster_groups") or {}

    for task_name, flow_func in available_tasks:
        if task_name not in dag_handler.tasks or not dag_handler.tasks[task_name].get("enabled", True):
            logging.info(f"Task '{task_name}' is disabled in DAG. Skipping.")
            continue

        options = dag_handler.get_task_options(task_name)
        batch_id = f"batch_run_{task_name}"
        
        executor = TaskExecutor(task_name, batch_id, dag_handler, monitor)
        
        with executor:
            if executor.can_run:
                try:
                    kwargs = {
                        "input_items": items_to_process,
                        "force_processing": options.get("force_processing", False),
                    }
                    if "grouping_columns" in options:
                        kwargs["grouping_columns"] = options["grouping_columns"]
                    if "monitor" in options:
                        kwargs["monitor"] = options["monitor"]
                    if "preparation" in options:
                        kwargs["preparation"] = options["preparation"]
                    if "transforms" in options:
                        kwargs["transforms"] = options["transforms"]
                    if "features" in options:
                        kwargs["features"] = options["features"]
                    if task_name == "touch_series_transforms" and preparation_dir is not None:
                        kwargs["preparation_dir"] = preparation_dir
                    if task_name == "touch_feature_extraction":
                        if series_dir is not None:
                            kwargs["series_dir"] = series_dir
                        if preparation_dir is not None:
                            kwargs["preparation_dir"] = preparation_dir
                    if "cluster_groups" in options:
                        kwargs["cluster_groups"] = options["cluster_groups"]
                    if task_name in (
                        "touch_comparing",
                        "map_receptive_fields_clustered",
                        "extract_receptive_fields_clustered",
                        "visualize_receptive_fields_clustered",
                    ):
                        if _cluster_group_defs:
                            kwargs["cluster_group_defs"] = _cluster_group_defs
                    if "feature_combinations" in options:
                        logging.warning(
                            f"[{task_name}] 'feature_combinations' is deprecated — "
                            "migrate to 'cluster_groups'."
                        )
                        kwargs["feature_combinations"] = options["feature_combinations"]
                    if "clustering_profiles" in options:
                        kwargs["clustering_profiles"] = options["clustering_profiles"]
                    if "reduction" in options:
                        kwargs["reduction"] = options["reduction"]
                    if "evaluation" in options:
                        kwargs["evaluation"] = options["evaluation"]
                    if "comparing_profiles" in options:
                        kwargs["comparing_profiles"] = options["comparing_profiles"]
                    if "min_instances_per_sensor" in options:
                        kwargs["min_instances_per_sensor"] = options["min_instances_per_sensor"]
                    if "min_sensor_types" in options:
                        kwargs["min_sensor_types"] = options["min_sensor_types"]
                    if "camera_angle_mode" in options:
                        mode_cfg = options["camera_angle_mode"]
                        if not isinstance(mode_cfg, dict):
                            raise ValueError(
                                f"[{task_name}] 'camera_angle_mode' must be a mapping with "
                                f"an 'enabled' key, got: {mode_cfg!r}"
                            )
                        if "auto" in mode_cfg:
                            raise ValueError(
                                f"[{task_name}] Legacy 'camera_angle_mode.auto.enabled' "
                                "shape is no longer supported. "
                                "Use 'camera_angle_mode: {enabled: true|false}' instead."
                            )
                        if "enabled" not in mode_cfg:
                            raise ValueError(
                                f"[{task_name}] 'camera_angle_mode' must contain an 'enabled' "
                                f"key, got keys: {list(mode_cfg.keys())!r}"
                            )
                        kwargs["camera_angle_mode"] = bool(mode_cfg["enabled"])
                    if "max_workers" in options:
                        kwargs["max_workers"] = int(options["max_workers"])
                    if "show_interactive" in options:
                        kwargs["show_interactive"] = options["show_interactive"]
                    if options.get("projection_method"):
                        kwargs["projection_method"] = options["projection_method"]
                    if "disjoint_mask_distance_mm" in options:
                        kwargs["disjoint_mask_distance_mm"] = float(options["disjoint_mask_distance_mm"])
                    if "gallery_viewer" in options:
                        kwargs["gallery_viewer"] = bool(options["gallery_viewer"])
                    flow_func(**kwargs)
                except Exception as e:
                    executor.error_msg = f"Batch analysis failed: {str(e)}"
                    logging.error(f"Error during {task_name}: {e}")

    logging.info("✅ Batch analysis finished.")

def main():
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dag-config", type=Path, required=True)
    args = parser.parse_args()
    dag_config_path = args.dag_config
    project_data_root = path_tools.get_project_data_root()
    configs_dir = Path("configs")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_file_path = reports_dir / "analysis_status.xlsx"

    if not dag_config_path.exists():
        logging.error(f"DAG config not found at {dag_config_path}")
        exit(1)
        
    dag_handler = DagConfigHandler(dag_config_path)
    entries = dag_handler.get_parameter('kinect_configs')
    block_files = resolve_session_configs(entries, configs_dir / "kinect_configs")

    run_batch_analysis(
        block_files=block_files,
        project_data_root=project_data_root,
        dag_handler=dag_handler,
        report_file_path=report_file_path
    )

if __name__ == "__main__":
    main()