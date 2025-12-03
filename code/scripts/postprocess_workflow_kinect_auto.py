import os
import logging
from pathlib import Path
from datetime import datetime
import shutil
import time
import traceback
from multiprocessing import Queue, freeze_support
from typing import List, Dict, Tuple, Any
from collections import defaultdict

import pandas as pd
import numpy as np
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
from primary_processing import (
    KinectConfigFileHandler, 
    KinectConfig, 
    get_block_files
)

from _5_postprocessing import (
    determine_receptive_field,
    set_xyz_reference_from_gestures
)

# --- Post-Processing Sub-Flows ---

@flow(name="analyze_pca_components")
def set_xyz_reference_from_gestures_flow(input_files: List[Path], output_dir: Path, force_processing: bool = False) -> Tuple[List[Path], Path]:
    """
    Analyse the principal component of the XYZ position for stroke and tapping.
    Iterates over a list of files and produces a distinct output for each.
    """
    print(f"[{output_dir.name}] Performing PCA analysis on {len(input_files)} files...")
    
    output_dir = output_dir / "session_xyz_reference_from_gestures"
    output_files = set_xyz_reference_from_gestures(
        input_files, output_dir,
        monitor=False,
        monitor_segment=False,
        force_processing=force_processing
    )
    
    return output_files

@flow(name="determine_receptive_field")
def determine_receptive_field_flow(
    input_files: List[Path], 
    configs: List[KinectConfig], 
    output_dir: Path, 
    force_processing: bool = False
) -> Tuple[List[Path], List[Path]]:
    """
    Determine the receptive field based on touch locations for each file.
    Returns a list of metadata CSV paths corresponding to the input data files.
    
    Updated to accept configs.
    """
    print(f"[{output_dir.name}] Calculating receptive field for {len(input_files)} files...")
    
    kinect_config = configs[0]
    forearm_pointcloud_dir = kinect_config.session_processed_output_dir / "forearm_pointclouds"
    arm_roi_metadata_path = forearm_pointcloud_dir / (kinect_config.session_id + "_arm_roi_metadata.json")

    # Pass configs to the underlying function
    output_files, rf_pc_file = determine_receptive_field(
        input_files, 
        arm_roi_metadata_path, 
        output_dir, 
        force_processing=force_processing
    )

    return output_files, rf_pc_file

@flow(name="filter_by_receptive_field")
def filter_by_receptive_field(data_files: List[Path], rf_files: List[Path], output_dir: Path, force_processing: bool = False) -> List[Path]:
    """
    Filter the data based on the receptive field.
    Matches data files to RF files by index (assumes strictly ordered 1-to-1 flow).
    """
    print(f"[{output_dir.name}] Filtering data for {len(data_files)} files...")
    return


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
        input_dir = config.session_merged_output_dir / "sessions"
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
        # Receptive Field
        {
            "name": "determine_receptive_field",
            "func": determine_receptive_field_flow,
            "params": lambda: {
                "input_files": context.get("source_files"),
                "configs": context.get("session_configs"),  # Configs propagated here
                "output_dir": session_output_dir / "sessions_receptive-field"
            },
            "outputs": ["segmented_data_files", "rf_metadata_files"]
        },
        # Filtering
        {
            "name": "filter_by_receptive_field",
            "func": filter_by_receptive_field,
            "params": lambda: {
                "data_files": context.get("segmented_data_files"),
                "rf_files": context.get("rf_metadata_files"),
                "output_dir": session_output_dir
            },
            "outputs": ["final_data_files"]
        },
        
        # PCA Analysis
        {
            "name": "set_xyz_reference_from_gestures",
            "func": set_xyz_reference_from_gestures_flow,
            "params": lambda: {
                "input_files": context.get("source_files"),
                "output_dir": session_output_dir  / "session_xyz_reference_from_gestures"
            },
            "outputs": ["pca_data_files", "pca_report"]
        }
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

            # Inject force_processing if defined in options
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
    kinect_configs_dir: Path,
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
    block_files = get_block_files(kinect_configs_dir)
    if not block_files:
        logging.warning(f"No config files found in {kinect_configs_dir}")
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

def main():
    freeze_support()
    
    # Configuration
    project_data_root = path_tools.get_project_data_root() # Using path_tools as per reference script
    configs_dir = Path("configs")
    dag_config_path = configs_dir / "postprocess_workflow_kinect_auto_dag.yaml"
    
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_file_path = reports_dir / f"postprocess_status.xlsx"

    monitor_queue = Queue()
    
    # Load the Main DAG handler just to get directory settings if needed,
    # or hardcode if strictly following local paths.
    try:
        # Assuming the DAG config might contain the config directory name
        # If not, we default to 'kinect_configs' or similar
        main_dag_handler = DagConfigHandler(dag_config_path)
        kinect_dir_name = main_dag_handler.get_parameter('kinect_configs_directory')
        kinect_configs_dir: Path = configs_dir / kinect_dir_name
        
        if not kinect_configs_dir.exists():
             # Fallback for safety if parameter is missing/wrong
             kinect_configs_dir = configs_dir
             
    except Exception:
        kinect_configs_dir = configs_dir / "kinect_configs"

    if not kinect_configs_dir.exists():
        logging.error(f"Config directory {kinect_configs_dir} does not exist.")
        exit(1)

    run_batch_postprocessing(
        kinect_configs_dir=kinect_configs_dir,
        project_data_root=project_data_root,
        dag_config_path=dag_config_path,
        monitor_queue=monitor_queue,
        report_file_path=report_file_path,
        parallel=False 
    )

if __name__ == "__main__":
    main()