import os
import logging
from pathlib import Path
from datetime import datetime
import shutil
import time
import traceback
from multiprocessing import Queue, freeze_support
from typing import List, Dict, Tuple
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
from utils.pipeline.pipeline_config_manager import DagConfigHandler
from utils.pipeline_monitoring.pipeline_monitor import PipelineMonitor
from primary_processing import (
    KinectConfigFileHandler, 
    KinectConfig, 
    get_block_files
)

from _5_postprocessing import (
    set_xyz_reference_from_gestures
)


# --- Reused TaskExecutor ---
class TaskExecutor:
    """A context manager to handle the boilerplate of running a pipeline task."""
    def __init__(self, task_name, block_name, dag_handler, monitor):
        self.task_name: str = task_name
        self.block_name: str = block_name
        self.dag_handler = dag_handler
        self.monitor = monitor
        self.can_run: bool = False
        self.error_msg: str = None

    def __enter__(self):
        if self.dag_handler.can_run(self.task_name):
            self.can_run = True
            print(f"[{self.block_name}] ==> Running task: {self.task_name}")
            if self.monitor is not None:
                self.monitor.update(self.block_name, self.task_name, "RUNNING")
        return self
    
    def __exit__(self, exc_type, exc_value, tb):
        if not self.can_run:
            return

        if exc_type:
            self.error_msg = f"Task '{self.task_name}' failed: {exc_value}"
            print(f"❌ {self.error_msg}\n{traceback.format_exc()}")
            if self.monitor is not None:
                self.monitor.update(self.block_name, self.task_name, "FAILURE", self.error_msg)
            return True
        else:
            self.dag_handler.mark_completed(self.task_name)
            if self.monitor is not None:
                self.monitor.update(self.block_name, self.task_name, "SUCCESS")
        return False


# --- Post-Processing Sub-Flows ---

@flow(name="analyze_pca_components")
def set_xyz_reference_from_gestures_flow(input_files: List[Path], output_dir: Path) -> Tuple[List[Path], Path]:
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
        force_processing=True
    )

    return output_files

@flow(name="segment_single_touches")
def segment_single_touches(input_files: List[Path], output_dir: Path) -> List[Path]:
    """
    Segment the data into single touches for each file in the list.
    """
    print(f"[{output_dir.name}] Segmenting single touches for {len(input_files)} files...")
    return False

    output_files = []
    
    for input_path in input_files:
        if not input_path.exists():
            continue

        df = pd.read_csv(input_path)
        
        # Implementation required: Define logic for segmentation
        # Mock segmentation logic
        if 'touch_id' not in df.columns:
            df['touch_id'] = (df.index // 100) 
        
        output_filename = f"{input_path.stem}_segmented.csv"
        segmented_path = output_dir / output_filename
        df.to_csv(segmented_path, index=False)
        output_files.append(segmented_path)
    
    return output_files

@flow(name="determine_receptive_field")
def determine_receptive_field(input_files: List[Path], output_dir: Path) -> List[Path]:
    """
    Determine the receptive field based on touch locations for each file.
    Returns a list of metadata CSV paths corresponding to the input data files.
    """
    print(f"[{output_dir.name}] Calculating receptive field for {len(input_files)} files...")
    return False
    rf_files = []
    
    for input_path in input_files:
        if not input_path.exists():
            continue

        df = pd.read_csv(input_path)
        
        # Logic: Calculate Convex Hull or Bounding Box of touch points per file
        receptive_field_stats = {
            "source_file": input_path.name,
            "min_x": df['x'].min(),
            "max_x": df['x'].max(),
            "min_y": df['y'].min(),
            "max_y": df['y'].max(),
            "center_x": df['x'].mean(),
            "center_y": df['y'].mean()
        }
        
        output_filename = f"{input_path.stem}_rf_metadata.csv"
        rf_path = output_dir / output_filename
        pd.DataFrame([receptive_field_stats]).to_csv(rf_path, index=False)
        rf_files.append(rf_path)
    
    return rf_files

@flow(name="filter_by_receptive_field")
def filter_by_receptive_field(data_files: List[Path], rf_files: List[Path], output_dir: Path) -> List[Path]:
    """
    Filter the data based on the receptive field.
    Matches data files to RF files by index (assumes strictly ordered 1-to-1 flow).
    """
    print(f"[{output_dir.name}] Filtering data for {len(data_files)} files...")
    return False
    output_files = []
    
    # Ensure we have matching lists
    if len(data_files) != len(rf_files):
        print("❌ Error: Mismatch in count of data files vs RF metadata files.")
        return []

    for data_path, rf_path in zip(data_files, rf_files):
        if not data_path.exists() or not rf_path.exists():
            continue
            
        df = pd.read_csv(data_path)
        rf = pd.read_csv(rf_path).iloc[0]
        
        # Logic: remove points outside the defined receptive field
        mask = (
            (df['x'] >= rf['min_x']) & (df['x'] <= rf['max_x']) &
            (df['y'] >= rf['min_y']) & (df['y'] <= rf['max_y'])
        )
        
        filtered_df = df[mask]
        
        output_filename = f"{data_path.stem}_filtered.csv"
        final_output_path = output_dir / output_filename
        filtered_df.to_csv(final_output_path, index=False)
        output_files.append(final_output_path)
    
    return output_files

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
    session_output_dir = session_configs[0].session_merged_output_dir

    # Resolve input files from Configs
    # We look for the specific file expected from the video processing stage
    session_input_files = []
    for config in session_configs:
        input_dir = config.session_merged_output_dir / "sessions"
        input_path = input_dir / f"{config.session_id}_semicontrolled_{config.block_id}_merged_data.csv"
        session_input_files.append(input_path)

    if monitor_queue is not None:
        monitor = PipelineMonitor(
            report_path=report_file_path, stages=list(dag_handler.tasks.keys()), data_queue=monitor_queue
        )
    else:
        monitor = None

    # Initialize context with the raw list of files
    context = {
        "source_files": session_input_files
    }

    # UPDATED: Pipeline stages now handle List[Path]
    pipeline_stages = [
        # 1. PCA Analysis
        {
            "name": "set_xyz_reference_from_gestures",
            "func": set_xyz_reference_from_gestures_flow,
            "params": lambda: {
                "input_files": context.get("source_files"),
                "output_dir": session_output_dir
            },
            "outputs": ["pca_data_files", "pca_report"]
        },
        # 2. Segmentation
        {
            "name": "segment_single_touches",
            "func": segment_single_touches,
            "params": lambda: {
                "input_files": context.get("pca_data_files"),
                "output_dir": session_output_dir
            },
            "outputs": ["segmented_data_files"]
        },
        # 3. Receptive Field
        {
            "name": "determine_receptive_field",
            "func": determine_receptive_field,
            "params": lambda: {
                "input_files": context.get("segmented_data_files"),
                "output_dir": session_output_dir
            },
            "outputs": ["rf_metadata_files"]
        },
        # 4. Filtering
        {
            "name": "filter_by_receptive_field",
            "func": filter_by_receptive_field,
            "params": lambda: {
                "data_files": context.get("segmented_data_files"),
                "rf_files": context.get("rf_metadata_files"),
                "output_dir": session_output_dir
            },
            "outputs": ["final_data_files"]
        }
    ]

    for stage_idx, stage in enumerate(pipeline_stages):
        task_name = stage["name"]
        executor = TaskExecutor(task_name, session_id, dag_handler, monitor)

        with executor:
            if not executor.can_run:
                continue

            try:
                params = stage["params"]()
            except KeyError as e:
                executor.error_msg = f"Missing dependency in context: {e}"
                print(f"❌ {executor.error_msg}")
                return {"status": "failed", "error": executor.error_msg}
            
            # Validation: Check if list inputs are empty
            input_lists = [v for v in params.values() if isinstance(v, list)]
            if any(len(l) == 0 for l in input_lists):
                 print(f"⚠️ Warning: Empty input list for task {task_name}. Skipping execution.")
                 continue

            result = stage["func"](**params)

            if "outputs" in stage:
                outputs = stage["outputs"]
                if not isinstance(result, tuple):
                    result = (result,)
                for i, key in enumerate(outputs):
                    if key and i < len(result):
                        context[key] = result[i]

        if executor.error_msg:
            return {"status": "failed", "stage": stage_idx, "error": executor.error_msg}

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