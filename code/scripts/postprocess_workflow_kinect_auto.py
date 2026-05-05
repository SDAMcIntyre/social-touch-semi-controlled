import argparse
import os
import logging
import tempfile
from pathlib import Path
from datetime import datetime
import shutil
import time
import traceback
from multiprocessing import Queue, freeze_support
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict

import pandas as pd
import numpy as np
import open3d as o3d
# sklearn is assumed to be present in the Anaconda environment
from sklearn.decomposition import PCA

from prefect import flow

# Setup a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mocking utils for template completeness - Replace with actual imports
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

from preprocessing.forearm_extraction import (
    ForearmFrameParametersFileHandler,
    ForearmCatalog,
    get_forearms_with_fallback,
)

from _5_postprocessing import (
    apply_icp_registration,
    set_xyz_reference_from_gestures,
    export_forearm_pca_calibrated,
    project_contacts_onto_forearm,
    center_on_receptive_field,
    deduplicate_forearm_ply,
    deduplicate_contact_points_csv,
)
from _4_merging.aggregate_blocks_session import aggregate_session_blocks

# --- Post-Processing Sub-Flows ---

@flow(name="apply_icp_registration")
def apply_icp_registration_flow(
    input_files: List[Path],
    session_configs: List[KinectConfig],
    output_dir: Path,
    force_processing: bool = False,
) -> List[Path]:
    """Apply ICP registration transforms to merged CSVs."""
    print(f"[{output_dir.name}] Applying ICP registration to {len(input_files)} files...")
    return apply_icp_registration(
        input_files, session_configs, output_dir,
        force_processing=force_processing,
    )

@flow(name="analyze_pca_components")
def set_xyz_reference_from_gestures_flow(input_files: List[Path], output_dir: Path, force_processing: bool = False) -> Tuple[List[Path], Path]:
    """
    Analyse the principal component of the XYZ position for stroke and tapping.
    Iterates over a list of files and produces a distinct output for each.
    """
    print(f"[{output_dir.name}] Performing PCA analysis on {len(input_files)} files...")

    output_files = set_xyz_reference_from_gestures(
        input_files, output_dir,
        monitor=False,
        monitor_segment=False,
        force_processing=force_processing
    )

    return output_files


@flow(name="export_forearm_pca_calibrated")
def export_forearm_pca_calibrated_flow(
    session_configs: List[KinectConfig],
    pca_output_dir: Path,
    output_dir: Path,
    force_processing: bool = False,
) -> Optional[Path]:
    """Export the forearm-of-reference PLY transformed into PCA-calibrated space."""
    print(f"[{output_dir.name}] Exporting PCA-calibrated forearm PLY...")
    return export_forearm_pca_calibrated(
        session_configs, pca_output_dir, output_dir,
        force_processing=force_processing,
    )


@flow(name="deduplicate_xy")
def deduplicate_xy_flow(
    input_files: List[Path],
    session_configs: List[KinectConfig],
    output_dir: Path,
    forearm_output_dir: Path,
    force_processing: bool = False,
) -> Tuple[List[Path], List[Optional[Path]]]:
    if len(input_files) != len(session_configs):
        raise ValueError(
            f"input_files length ({len(input_files)}) does not match "
            f"session_configs length ({len(session_configs)}) — cannot pair "
            "blocks to configs for per-block deduplication."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    forearm_output_dir.mkdir(parents=True, exist_ok=True)

    deduped_csv_paths: List[Path] = []
    deduped_forearm_ply_paths: List[Optional[Path]] = []

    with tempfile.TemporaryDirectory() as _tmp:
        tmp_dir = Path(_tmp)

        for config, input_csv in zip(session_configs, input_files):
            forearm_ply = _load_per_block_forearm_ply(config, tmp_dir)

            if forearm_ply is not None:
                forearm_out = forearm_output_dir / forearm_ply.name
                stats = deduplicate_forearm_ply(forearm_ply, forearm_out, epsilon=0.1)
                logging.info(
                    "Block %s forearm dedup: %d → %d (removed %d)",
                    config.block_id,
                    stats["n_original"],
                    stats["n_deduped"],
                    stats["n_removed"],
                )
                deduped_forearm_ply_paths.append(forearm_out)
            else:
                logging.warning(
                    "Block %s — forearm unavailable, skipping forearm deduplication.",
                    config.block_id,
                )
                deduped_forearm_ply_paths.append(None)

            csv_out = output_dir / input_csv.name
            stats = deduplicate_contact_points_csv(input_csv, csv_out, epsilon=0.1)
            logging.info(
                "Block %s CSV dedup: %d rows, %d → %d contact points",
                config.block_id,
                stats["n_rows_processed"],
                stats["total_points_before"],
                stats["total_points_after"],
            )
            deduped_csv_paths.append(csv_out)

    return deduped_csv_paths, deduped_forearm_ply_paths


@flow(name="project_contacts_onto_forearm_early")
def project_contacts_onto_merged_forearm_flow(
    input_files: List[Path],
    session_configs: List[KinectConfig],
    output_dir: Path,
    projection_stats_path: Path,
    force_processing: bool = False,
    deduped_forearm_plys: Optional[List[Optional[Path]]] = None,
) -> List[Path]:
    """Project contact points onto the per-block forearm surface before ICP.

    Iterates over each (config, input_file) pair.  For each block, the
    per-block reference forearm (frame 0) is resolved from the preprocessing
    forearm catalogue, saved to a temporary PLY, and passed to
    project_contacts_onto_forearm().

    Blocks whose forearm metadata is missing are skipped with a warning; their
    input CSV is forwarded unchanged so the ICP step still receives all files.
    All other blocks raise immediately if an unexpected error occurs (fail-fast).

    Args:
        input_files: Per-block merged CSVs from ``blocks_merged/``.
        session_configs: KinectConfig instances in the same order as
            *input_files* (one per block).
        output_dir: Destination directory (``blocks_merged_projected/``).
        projection_stats_path: Path for the combined projection-stats CSV.
        force_processing: Re-run even when outputs are already up-to-date.
        deduped_forearm_plys: Pre-deduped per-block forearm PLY paths from
            ``deduplicate_xy_flow``.  When provided, used directly instead of
            loading raw forearms via ``_load_per_block_forearm_ply``.

    Returns:
        List of output CSV paths (projected where forearm was available,
        otherwise the original input path copied through unchanged).
    """
    if len(input_files) != len(session_configs):
        raise ValueError(
            f"input_files length ({len(input_files)}) does not match "
            f"session_configs length ({len(session_configs)}) — cannot pair "
            "blocks to configs for per-block forearm projection."
        )

    print(f"[{output_dir.name}] Projecting {len(input_files)} blocks onto per-block forearm surfaces...")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_output_files: List[Path] = []

    with tempfile.TemporaryDirectory() as _tmp:
        tmp_dir = Path(_tmp)

        for i, (config, input_csv) in enumerate(zip(session_configs, input_files)):
            if deduped_forearm_plys is not None:
                forearm_ply = deduped_forearm_plys[i]
            else:
                forearm_ply = _load_per_block_forearm_ply(config, tmp_dir)

            if forearm_ply is None:
                # No forearm metadata for this block — forward input unchanged.
                logging.warning(
                    "Skipping projection for block %s — forearm unavailable. "
                    "Forwarding original CSV to ICP step.",
                    config.block_id,
                )
                all_output_files.append(input_csv)
                continue

            block_outputs = project_contacts_onto_forearm(
                input_files=[input_csv],
                forearm_ply_path=forearm_ply,
                output_dir=output_dir,
                projection_stats_path=projection_stats_path,
                force_processing=force_processing,
            )
            if not block_outputs:
                raise RuntimeError(
                    f"project_contacts_onto_forearm returned no output for block "
                    f"{config.block_id} — expected exactly one output CSV."
                )
            all_output_files.extend(block_outputs)

    return all_output_files


@flow(name="center_on_receptive_field")
def center_on_receptive_field_flow(
    input_files: List[Path],
    forearm_ply_path: Optional[Path],
    output_dir: Path,
    forearm_output_dir: Path,
    rf_origin_path: Path,
    force_processing: bool = False,
) -> List[Path]:
    """Center block CSVs and forearm PLY on the receptive field origin."""
    print(f"[{output_dir.name}] Centering spatial data on receptive field origin...")
    return center_on_receptive_field(
        input_files, forearm_ply_path, output_dir, forearm_output_dir, rf_origin_path,
        force_processing=force_processing,
    )


@flow(name="aggregate_session_blocks")
def aggregate_session_blocks_flow(
    input_files: List[Path],
    output_path: Path,
    forearm_ply_path: Optional[Path] = None,
    force_processing: bool = False,
) -> Path:
    """Aggregate all fully-processed block CSVs into one session-level CSV."""
    print(f"[{output_path.name}] Aggregating {len(input_files)} blocks...")
    return aggregate_session_blocks(
        input_paths=input_files,
        output_path=output_path,
        forearm_ply_path=forearm_ply_path,
        force_processing=force_processing,
    )


def _resolve_latest_forearm_ply(session_output_dir: Path) -> Optional[Path]:
    """Return the most recent forearm PLY: RF-centered if available, else PCA-calibrated."""
    for subdir in ("forearm_rf_centered", "forearm_pca_calibrated"):
        candidates = sorted((session_output_dir / subdir).glob("*.ply"))
        if candidates:
            return candidates[-1]
    return None


def _load_per_block_forearm_ply(config: KinectConfig, tmp_dir: Path) -> Optional[Path]:
    """Return a PLY path for the per-block reference forearm (frame 0), or None.

    Loads the forearm metadata for the current block via ForearmCatalog and
    retrieves the frame-0 reference point cloud using get_forearms_with_fallback().
    Saves it as a temporary PLY file inside *tmp_dir* so that the path-based
    projection API can consume it.

    Returns None when the metadata file is missing or when no forearm can be
    loaded for this block — the caller must handle the None case (fail-fast:
    raise if projection is required and forearm is missing).
    """
    forearm_dir = config.session_processed_output_dir / "forearm_pointclouds"
    metadata_path = forearm_dir / f"{config.session_id}_arm_roi_metadata.json"
    if not metadata_path.exists():
        logging.warning(
            "Forearm metadata not found at %s — per-block projection unavailable for block %s.",
            metadata_path,
            config.block_id,
        )
        return None
    try:
        forearm_params = ForearmFrameParametersFileHandler.load(metadata_path)
        catalog = ForearmCatalog(forearm_params, forearm_dir)
        forearms = get_forearms_with_fallback(catalog, config.source_video.name)
    except Exception as exc:
        logging.warning(
            "Could not load per-block forearm for block %s: %s",
            config.block_id,
            exc,
        )
        return None

    reference_cloud = forearms.get(0)
    if reference_cloud is None:
        logging.warning(
            "No frame-0 forearm found for block %s — per-block projection unavailable.",
            config.block_id,
        )
        return None

    tmp_ply = tmp_dir / f"{config.session_id}_{config.block_id}_forearm_ref.ply"
    o3d.io.write_point_cloud(str(tmp_ply), reference_cloud)
    return tmp_ply


# --- Worker Flow ---

# @flow(name="Run Single Session Postprocessing")
def run_single_session_postprocessing(
    session_id: str,
    session_configs: List[KinectConfig],
    dag_handler: DagConfigHandler,
    *,
    monitor_queue: Queue = None,
    report_file_path: Path = None
):
    print(f"🚀 Starting postprocessing for session: {session_id}")
    # Assuming all configs in a session share the same merged output dir root
    session_output_dir = session_configs[0].session_merged_output_dir

    # Resolve input files from Configs
    # We look for the specific file expected from the video processing stage
    session_input_files = []
    for config in session_configs:
        input_dir = config.session_merged_output_dir / "blocks_merged"
        input_path = input_dir / f"{config.session_id}_semicontrolled_{config.block_id}_merged_data.csv"
        # Only add if it vaguely looks like a path, validation happens in tasks
        session_input_files.append(input_path)

    if monitor_queue is not None:
        monitor = PipelineMonitor(
            report_path=report_file_path, stages=list(dag_handler.tasks.keys()), data_queue=monitor_queue
        )
    else:
        monitor = None

    # Initialize context with the raw list of files AND the session configs
    context = {
        "source_files": session_input_files,
        "session_configs": session_configs
    }

    # UPDATED: Pipeline stages using the architecture of function_of_reference
    pipeline_stages = [
        # Step 0: Deduplicate (x,y) in forearm PLYs and contact_points CSVs
        #   Input:  blocks_merged/ CSVs + per-block forearm PLYs
        #   Output: blocks_merged_deduped/ + forearm_deduped/
        {
            "name": "deduplicate_xy",
            "func": deduplicate_xy_flow,
            "params": lambda: {
                "input_files": context.get("source_files"),
                "session_configs": context.get("session_configs"),
                "output_dir": session_output_dir / "blocks_merged_deduped",
                "forearm_output_dir": session_output_dir / "forearm_deduped",
            },
            "outputs": ["deduped_source_files", "deduped_forearm_plys"]
        },
        # Step 1: Project contact points onto per-block forearm surface (before ICP)
        #   Input:  blocks_merged_deduped/ CSVs + forearm_deduped/ PLYs
        #   Output: blocks_merged_projected/ (contact points snapped to per-block forearm)
        {
            "name": "project_contacts_onto_forearm",
            "func": project_contacts_onto_merged_forearm_flow,
            "params": lambda: {
                "input_files": context.get("deduped_source_files"),
                "session_configs": context.get("session_configs"),
                "output_dir": session_output_dir / "blocks_merged_projected",
                "projection_stats_path": session_output_dir / "blocks_merged_projected" / "projection_stats.csv",
                "deduped_forearm_plys": context.get("deduped_forearm_plys"),
            },
            "outputs": ["projected_source_files"]
        },
        # Step 2: ICP Registration — consumes projected (on-surface) merged CSVs
        {
            "name": "apply_icp_registration",
            "func": apply_icp_registration_flow,
            "params": lambda: {
                "input_files": context.get("projected_source_files"),
                "session_configs": context.get("session_configs"),
                "output_dir": session_output_dir / "blocks_registered",
            },
            "outputs": ["registered_files"]
        },
        # Step 3: PCA XYZ Reference Calibration
        {
            "name": "set_xyz_reference_from_gestures",
            "func": set_xyz_reference_from_gestures_flow,
            "params": lambda: {
                "input_files": context.get("registered_files"),
                "output_dir": session_output_dir / "blocks_pca_calibrated",
            },
            "outputs": ["pca_data_files", "pca_report"]
        },
        # Step 4: Export forearm PLY in PCA-calibrated space
        {
            "name": "export_forearm_pca_calibrated",
            "func": export_forearm_pca_calibrated_flow,
            "params": lambda: {
                "session_configs": context.get("session_configs"),
                "pca_output_dir": context.get("pca_report"),
                "output_dir": session_output_dir / "forearm_pca_calibrated",
            },
            "outputs": ["forearm_pca_ply"]
        },
        # Step 5: Center spatial data on the receptive field origin
        #   Input: pca_data_files (PCA-calibrated; contact points are already on-surface
        #          since projection happened before ICP in step 1)
        {
            "name": "center_on_receptive_field",
            "func": center_on_receptive_field_flow,
            "params": lambda: {
                "input_files": context.get("pca_data_files"),
                "forearm_ply_path": context.get("forearm_pca_ply"),
                "output_dir": session_output_dir / "blocks_rf_centered",
                "forearm_output_dir": session_output_dir / "forearm_rf_centered",
                "rf_origin_path": session_output_dir / "rf_center_origin.json",
            },
            "outputs": ["rf_centered_files"]
        },
        # Step 6: Aggregate fully-processed blocks into one session-level CSV
        {
            "name": "aggregate_session",
            "func": aggregate_session_blocks_flow,
            "params": lambda: {
                "input_files": context.get("rf_centered_files"),
                "output_path": session_output_dir / f"{session_id}_semicontrolled_aggregated_session.csv",
                "forearm_ply_path": _resolve_latest_forearm_ply(session_output_dir),
            },
            "outputs": ["aggregated_file"]
        },
    ]

    for stage in pipeline_stages:
        task_name = stage["name"]
        # Use session_id as block_name equivalent here for the executor
        executor = TaskExecutor(task_name, session_id, dag_handler, monitor)

        with executor:
            if not executor.can_run: continue
            
            # Retrieve options from DAG handler (e.g. force_processing)
            options = dag_handler.get_task_options(task_name)
            
            # Resolve parameters lazily from context
            try:
                params = stage["params"]()
            except KeyError as e:
                 executor.error_msg = f"Missing dependency in context: {e}"
                 print(f"❌ {executor.error_msg}")
                 # Logic for failure inside executor context usually requires setting the error 
                 # and allowing the __exit__ to handle logging
                 continue

            # Inject options into params when supported by the flow
            if 'force_processing' in options:
                params['force_processing'] = options['force_processing']
            
            # Validation: Check if list inputs are empty
            # Note: We must exclude 'configs' from this check if configs are not lists of files, 
            # though in this architecture they are a List[KinectConfig], so len() check is valid.
            input_lists = [v for v in params.values() if isinstance(v, list)]
            if any(len(l) == 0 for l in input_lists):
                 print(f"⚠️ Warning: Empty input list for task {task_name}. Skipping execution.")
                 continue

            result = stage["func"](**params)

            if "outputs" in stage:
                outputs = stage["outputs"]
                # Ensure result is iterable/tuple for unpacking
                if not isinstance(result, tuple): 
                    result = (result,)
                
                for i, key in enumerate(outputs):
                    if key and i < len(result): 
                        context[key] = result[i]

        if executor.error_msg:
            print(f"🛑 Failure in {task_name}. Aborting session.")
            return {"status": "failed", "error": executor.error_msg}

    print(f"✅ Postprocessing finished for: {session_id}")
    return {"status": "success", "completed_tasks": list(dag_handler.completed_tasks)}

def run_batch_postprocessing(
    block_files: list[Path],
    project_data_root: Path,
    dag_config_path: Path,
    monitor_queue: Queue,
    report_file_path: Path,
    parallel: bool
):
    """
    Loads all block configurations, groups them by session_id, and triggers postprocessing per session.
    """
    # 1. Load Configs
    dag_handler_template = DagConfigHandler(dag_config_path)
    if not block_files:
        logging.warning("No config files found.")
        return

    # 2. Group by Session ID
    # Dictionary structure: { "session_id": [KinectConfig_Block1, KinectConfig_Block2, ...] }
    session_map: Dict[str, List[KinectConfig]] = defaultdict(list)
    
    logging.info(f"Scanning {len(block_files)} config files...")
    
    for block_file in block_files:
        try:
            config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
            # Initialize config object to resolve paths and IDs
            config = KinectConfig(config_data=config_data, database_path=project_data_root)
            session_map[config.session_id].append(config)
        except Exception as e:
            logging.error(f"Failed to load config {block_file}: {e}")

    logging.info(f"Found {len(session_map)} unique sessions to process.")

    # 3. Execute Pipeline per Session
    dag_handler_template = DagConfigHandler(dag_config_path)
    
    mode = "PARALLEL" if parallel else "SEQUENTIAL"
    logging.info(f"🚀 Starting postprocessing batch in {mode} mode.")
    
    for session_id, session_configs in session_map.items():
        dag_handler_instance = dag_handler_template.copy()
        
        if parallel:
            # Prefect Future submission logic would go here
            pass
        else:
            run_single_session_postprocessing(
                session_id=session_id,
                session_configs=session_configs,
                dag_handler=dag_handler_instance,
                monitor_queue=monitor_queue,
                report_file_path=report_file_path
            )

    logging.info("✅ All postprocessing tasks finished.")

# --- Main ---

def setup_environment():
    project_data_root = path_tools.get_project_data_root()
    configs_dir = Path("configs")

    print("🛠️  Setting up environment...")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_file_path = reports_dir / f"{timestamp}_postprocess_workflow_kinect_auto_status.xlsx"
    if report_file_path.exists():
        report_file_path.unlink()
        print("🧹 File with the same name found, removing it.")
    return project_data_root, configs_dir, report_file_path

def main():
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dag-config", type=Path, required=True)
    args = parser.parse_args()
    dag_config_path = args.dag_config
    project_data_root, configs_dir, report_file_path = setup_environment()

    try:
        main_dag_handler = DagConfigHandler(dag_config_path)
        is_parallel = main_dag_handler.get_parameter('parallel_execution', False)
        entries = main_dag_handler.get_parameter('kinect_configs')
        block_files = resolve_session_configs(entries, configs_dir / "kinect_configs")
    except FileNotFoundError:
        print(f"❌ Error: Configuration file '{dag_config_path}' not found.")
        exit(1)

    print("📊 Initializing pipeline monitor...")
    pipeline_stages = list(main_dag_handler.tasks.keys())
    main_monitor = PipelineMonitor(report_path=str(report_file_path), stages=pipeline_stages, live_plotting=True)
    main_monitor.show_dashboard()

    try:
        run_batch_postprocessing(
            block_files=block_files,
            project_data_root=project_data_root,
            dag_config_path=dag_config_path,
            monitor_queue=main_monitor.queue,
            report_file_path=report_file_path,
            parallel=is_parallel,
        )
        print("\n🏁 All pipeline tasks have completed.")
        print("✨ Dashboard will close automatically in 10 seconds...")
        time.sleep(10)
    finally:
        main_monitor.close_dashboard(block=True)
        print(f"👋 Processing finished. Final report saved to {report_file_path}")

if __name__ == "__main__":
    main()