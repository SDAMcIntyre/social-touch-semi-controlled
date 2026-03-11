import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from multiprocessing import freeze_support

from prefect import flow, get_run_logger
from prefect.futures import PrefectFuture

import utils.path_tools as path_tools
from utils import DagConfigHandler

from utils.pipeline.session_config_resolver import resolve_session_configs
from primary_processing import (
    KinectConfigFileHandler,
    KinectConfig,
)
from _4_merging import (
    align_and_merge_neural_and_kinect,
    aggregate_session_blocks,
    filter_block_by_neural_quality,
)

# --- Data Structures ---

@dataclass
class PipelineResult:
    """
    Structured return object for the session pipeline.
    Eliminates raw dictionary passing.
    """
    status: str
    session_id: str
    block_name: str
    session_merged_output_dir: Path
    block_output_path: Optional[Path] = None
    error: Optional[str] = None

# --- Helper Functions ---

def resolve_filenames(config: KinectConfig) -> Dict[str, Path]:
    """
    Centralizes all file naming conventions. 
    Decouples logic from string formatting.
    """
    # Input: Nerve Data
    nerve_name = f"{config.session_id}_semicontrolled_{config.block_id}_nerve.csv"
    
    # Input: Kinect Data — prefer registered version when available
    registered_name = f"{config.source_video.stem}_unified_registered.csv"
    registered_path = config.video_processed_output_dir / registered_name
    if registered_path.exists():
        kinect_path = registered_path
    else:
        kinect_name = f"{config.source_video.stem}_unified.csv"
        kinect_path = config.video_processed_output_dir / kinect_name

    # Output: Merged Block Data
    output_name = f"{config.session_id}_semicontrolled_{config.block_id}_merged_data.csv"

    return {
        "nerve_path": config.nerve_processed_dir / nerve_name,
        "kinect_path": kinect_path,
        "output_path": config.session_merged_output_dir / "sessions" / output_name
    }

# --- Individual Flows ---

@flow(name="9. Unify Dataset")
def unify_dataset(
    kinect_data_path: Path,
    nerve_data_path: Path,
    output_file_path: Path,
    *,
    force_processing: bool = False
) -> Path:
    """
    Merges Kinect contact data with Nerve data (Block-level Unification).
    """
    logger = get_run_logger()
    
    # Create parent directory if it doesn't exist
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[{output_file_path.parent.name}] Merging datasets...")
    logger.info(f"   - Kinect: {kinect_data_path.name}")
    logger.info(f"   - Nerve:  {nerve_data_path.name}")
    
    align_and_merge_neural_and_kinect(
        kinect_data_path,
        nerve_data_path,
        output_file_path,
        force_processing=force_processing
    )
    return output_file_path

@flow(name="10. Aggregate Session Blocks")
def aggregate_blocks(
    session_merged_output_dir: Path,
    session_id: str,
    glob_pattern: str = "*_merged_data.csv",
    force_processing: bool = False,
    input_subfolder: str = "sessions",
    output_suffix: str = "",
) -> Path:
    """
    Flow to aggregate all block-level merged CSV files within a session into one final CSV.
    Handles path resolution and invokes the aggregation logic.
    """
    logger = get_run_logger()

    input_dir = session_merged_output_dir / input_subfolder
    output_filename = f"{session_id}_semicontrolled_aggregated_session{output_suffix}.csv"
    output_path = session_merged_output_dir / output_filename

    logger.info(f"[{session_id}] Scanning for blocks in {input_dir}...")

    input_files = list(input_dir.glob(f"*{session_id}{glob_pattern}"))

    aggregate_session_blocks(
        input_paths=input_files,
        output_path=output_path,
        force_processing=force_processing
    )

    if input_files:
        logger.info(f"✅ Aggregation flow complete. Output: {output_filename}")
    else:
        logger.warning(f"⚠️ No files found matching pattern *{session_id}{glob_pattern}")

    return output_path


@flow(name="11. Filter by Neural Quality")
def filter_by_neural_quality_flow(
    merged_csv: Path,
    output_csv: Path,
    xlsx_path: Path,
    *,
    force_processing: bool = False,
    discard_from_first_not2use: bool = True,
) -> Path:
    """
    Flow to filter a single block's merged CSV by removing Not2Use trials.
    """
    logger = get_run_logger()
    logger.info(f"[{merged_csv.name}] Filtering by neural quality xlsx: {xlsx_path.name}")
    return filter_block_by_neural_quality(
        input_csv=merged_csv,
        output_csv=output_csv,
        xlsx_path=xlsx_path,
        force_processing=force_processing,
        discard_from_first_not2use=discard_from_first_not2use,
    )


@flow(name="Run Single Session Pipeline")
def run_single_session_pipeline(
    config: KinectConfig,
    dag_handler: DagConfigHandler
) -> PipelineResult:
    """
    Processes a single dataset block.
    Returns a PipelineResult object instead of a raw dict.
    """
    logger = get_run_logger()
    block_name = config.source_video.stem
    logger.info(f"🚀 Starting pipeline for block: {block_name}")
    
    paths = resolve_filenames(config)
    output_file_path = paths["output_path"]

    # --- Stage 5: Data Integration (Block Unification) + Neural Quality Filter ---
    try:
        task_name = 'unify_dataset'
        if dag_handler.can_run(task_name):
            logger.info(f"[{block_name}] ==> Running task: {task_name}")

            # Validation
            if not paths["kinect_path"].exists():
                raise FileNotFoundError(f"Kinect data not found: {paths['kinect_path']}")
            if not paths["nerve_path"].exists():
                raise FileNotFoundError(f"Nerve data not found: {paths['nerve_path']}")

            # Execution
            options = dag_handler.get_task_options(task_name)
            force = options.get('force_processing', False)

            unify_dataset(
                kinect_data_path=paths["kinect_path"],
                nerve_data_path=paths["nerve_path"],
                output_file_path=output_file_path,
                force_processing=force
            )

            dag_handler.mark_completed(task_name)

        # --- Filter by Neural Quality ---
        task_name = 'filter_by_neural_quality'
        if dag_handler.can_run(task_name):
            logger.info(f"[{block_name}] ==> Running task: {task_name}")

            xlsx_param = dag_handler.get_parameter('neural_quality_xlsx')
            if not xlsx_param:
                raise ValueError(f"neural_quality_xlsx parameter is not configured in DAG YAML")

            xlsx_path = Path(xlsx_param)
            if not xlsx_path.is_absolute():
                xlsx_path = config.session_merged_output_dir.parents[1] / xlsx_path
            if not xlsx_path.exists():
                raise FileNotFoundError(f"neural_quality_xlsx not found: {xlsx_path}")

            options = dag_handler.get_task_options(task_name)
            force = options.get('force_processing', False)
            discard_from_first = options.get('discard_from_first_not2use', True)

            filtered_output = (
                config.session_merged_output_dir / "sessions_filtered" / output_file_path.name
            )
            filter_by_neural_quality_flow(
                merged_csv=output_file_path,
                output_csv=filtered_output,
                xlsx_path=xlsx_path,
                force_processing=force,
                discard_from_first_not2use=discard_from_first,
            )

            dag_handler.mark_completed(task_name)

    except Exception as e:
        logger.error(f"❌ Pipeline failed for {block_name}: {e}")
        return PipelineResult(
            status="failed",
            session_id=config.session_id,
            block_name=block_name,
            session_merged_output_dir=config.session_merged_output_dir,
            error=str(e)
        )

    logger.info(f"✅ Pipeline finished successfully for block: {block_name}")
    
    return PipelineResult(
        status="success",
        session_id=config.session_id,
        block_name=block_name,
        session_merged_output_dir=config.session_merged_output_dir,
        block_output_path=output_file_path
    )

# --- Main Dispatcher ---

@flow(name="Batch Process All Sessions")
def run_batch_processing(
    block_files: list[Path],
    project_data_root: Path,
    dag_config_path: Path,
    parallel: bool
):
    """
    Dispatches pipeline runs for all session configs found in a directory.
    Handles both sequential and parallel execution uniformly.
    """
    logger = get_run_logger()
    dag_handler_template = DagConfigHandler(dag_config_path)

    mode = "PARALLEL" if parallel else "SEQUENTIAL"
    logger.info(f"🚀 Starting batch processing for {len(block_files)} sessions in {mode} mode.")

    # 1. Dispatch Runs
    futures_or_states = []
    
    for block_file in block_files:
        try:
            config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
            validated_config = KinectConfig(config_data=config_data, database_path=project_data_root)
            dag_handler_instance = dag_handler_template.copy()

            if parallel:
                # Submit returns a PrefectFuture
                run_future = run_single_session_pipeline.submit(
                    config=validated_config,
                    dag_handler=dag_handler_instance,
                    flow_run_name=f"block-{validated_config.source_video.stem}"
                )
                futures_or_states.append(run_future)
            else:
                # Direct call returns the result object immediately
                result = run_single_session_pipeline(
                    config=validated_config,
                    dag_handler=dag_handler_instance
                )
                futures_or_states.append(result)
                
        except Exception as e:
            logger.error(f"Failed to initialize config for {block_file}: {e}")

    # 2. Collect Results
    # We map session_id -> output_dir for aggregation
    session_map: Dict[str, Path] = {}

    if parallel:
        logger.info("Waiting for parallel runs to complete...")
        for future in futures_or_states:
            # Wait for completion and get the return value (PipelineResult)
            # Note: .result() behaves differently depending on Prefect version, 
            # assume standard behavior here.
            try:
                if isinstance(future, PrefectFuture):
                    # Safely extract result from future
                    state = future.wait()
                    if state.is_completed():
                        result: PipelineResult = state.result()
                        if result.status == "success":
                            session_map[result.session_id] = result.session_merged_output_dir
                    else:
                        logger.error(f"Flow run failed: {state}")
            except Exception as e:
                logger.error(f"Error retrieving future result: {e}")
    else:
        # In sequential mode, futures_or_states is just a list of PipelineResult objects
        for result in futures_or_states:
            if isinstance(result, PipelineResult) and result.status == "success":
                session_map[result.session_id] = result.session_merged_output_dir

    # 3. Aggregate Sessions
    if session_map:
        logger.info(f"\n--- Starting Aggregation for {len(session_map)} Sessions ---")
        for session_id, output_dir in session_map.items():
            aggregate_blocks(
                session_merged_output_dir=output_dir,
                session_id=session_id
            )

        # 3b. Aggregate filtered sessions (if sessions_filtered/ exists)
        logger.info(f"\n--- Starting Filtered Aggregation for {len(session_map)} Sessions ---")
        for session_id, output_dir in session_map.items():
            filtered_dir = output_dir / "sessions_filtered"
            if filtered_dir.exists():
                aggregate_blocks(
                    session_merged_output_dir=output_dir,
                    session_id=session_id,
                    input_subfolder="sessions_filtered",
                    output_suffix="_filtered",
                )
            else:
                logger.info(f"[{session_id}] No sessions_filtered/ directory; skipping filtered aggregation")

    logger.info("✅ All batch processing tasks have finished.")

# --- Entry Point ---

def setup_environment():
    """Handles filesystem setup and configuration path resolution."""
    project_data_root = path_tools.get_project_data_root()
    configs_dir = Path("configs")
    return project_data_root, configs_dir

def main():
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dag-config", type=Path, required=True)
    args = parser.parse_args()
    dag_config_path = args.dag_config
    project_data_root, configs_dir = setup_environment()

    print("🛠️  Initializing Merging Pipeline...")

    try:
        main_dag_handler = DagConfigHandler(dag_config_path)
        is_parallel = main_dag_handler.get_parameter('parallel_execution', False)
        entries = main_dag_handler.get_parameter('kinect_configs')
        block_files = resolve_session_configs(entries, configs_dir / "kinect_configs")

    except Exception as e:
        print(f"❌ Initialization Error: {e}")
        exit(1)

    run_batch_processing(
        block_files=block_files,
        project_data_root=project_data_root,
        dag_config_path=dag_config_path,
        parallel=is_parallel
    )

if __name__ == "__main__":
    main()