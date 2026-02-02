import os
import logging
from pathlib import Path
from multiprocessing import Queue, freeze_support
from typing import List, Set, Dict, Tuple
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
from primary_processing import (
    KinectConfigFileHandler, 
    KinectConfig, 
    get_block_files
)
# Imported the new function here
from analyse.touch_analytics import (
    analyse_number_single_touches, 
    generate_touch_summary_matrix
)

# --- Analysis Flows ---

@flow(name="analyse_number_single_touches")
def analyse_number_single_touches_flow(
    input_items: List[Tuple[Path, Path]], 
    force_processing: bool = False
) -> List[Path]:
    """
    Wrapper flow for the single touches analysis.
    
    Args:
        input_items: A list of tuples, where each tuple contains:
                     (input_file_path, database_path)
    """
    print(f"[Batch Analysis] Processing {len(input_items)} items...")
    
    results = []
    
    for input_file, database_path in input_items:
        try:
            # 1. Construct Output Directory
            output_dir = database_path / "4_analysed"
            
            # 2. Modify Filename
            filename = input_file.name
            if "_semicontrolled_" in filename:
                prefix = filename.split("_semicontrolled_")[0]
                new_filename = f"{prefix}_semicontrolled_single_touches_summary.csv"
            else:
                new_filename = f"{input_file.stem}_single_touches_summary.csv"

            output_file_path = output_dir / new_filename

            # 3. Ensure Output Directory Exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # 4. Execute Analysis
            # Note: We rely on default show=False for individual files in batch
            result_path = analyse_number_single_touches(input_file, output_file_path, show=False)
            results.append(result_path)

        except Exception as e:
            logging.error(f"Failed to process {input_file.name}: {e}")
            import traceback
            traceback.print_exc()
            
    # 5. Generate Aggregate Matrix
    if results:
        try:
            # Anchor output to the first database path
            anchor_db_path = input_items[0][1]
            matrix_output_path = anchor_db_path / "batch_condition_matrix.csv"
            
            logging.info("Generating batch condition matrix...")
            # show=True is the default now, but it's good practice to be explicit in workflows
            generate_touch_summary_matrix(results, matrix_output_path, show=True)
            
        except Exception as e:
            logging.error(f"Failed to generate aggregate matrix: {e}")

    return results


# --- Batch Processing Logic ---

def collect_unique_session_dirs(
    config_dir_names: List[str], 
    root_configs_path: Path, 
    project_data_root: Path
) -> Dict[Path, Path]:
    """
    Iterates through a list of config directory names, loads all valid KinectConfigs.
    """
    session_dir_map = {}
    total_files_scanned = 0

    for dir_name in config_dir_names:
        full_config_dir = root_configs_path / dir_name
        
        if not full_config_dir.exists():
            logging.warning(f"Config directory not found: {full_config_dir}")
            continue
            
        block_files = get_block_files(full_config_dir)
        logging.info(f"Scanning {len(block_files)} files in {dir_name}...")
        
        for block_file in block_files:
            try:
                # Load config to resolve the session output path
                config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
                config = KinectConfig(config_data=config_data, database_path=project_data_root)
                
                if config.session_merged_output_dir and config.database_path:
                    session_dir_map[config.session_merged_output_dir] = config.database_path
                    
                total_files_scanned += 1
            except Exception as e:
                logging.debug(f"Skipping {block_file.name}: {e}")

    logging.info(f"Scanned {total_files_scanned} config files.")
    logging.info(f"Identified {len(session_dir_map)} unique session contexts.")
    
    return session_dir_map

def run_batch_analysis(
    kinect_config_dirs: List[str],
    project_data_root: Path,
    configs_root_path: Path,
    dag_handler: DagConfigHandler,
    report_file_path: Path
):
    """
    Orchestrates the analysis workflow.
    """
    
    # 1. Collect Unique Session Maps
    session_map = collect_unique_session_dirs(
        kinect_config_dirs, 
        configs_root_path, 
        project_data_root
    )

    if not session_map:
        logging.warning("No valid session directories found. Exiting.")
        return

    # 2. Check DAG Configuration
    task_name = "analyse_number_single_touches"
    batch_id = "batch_run_all_sessions"
    
    if task_name not in dag_handler.tasks or not dag_handler.tasks[task_name].get("enabled", True):
        logging.info(f"Task '{task_name}' is disabled in DAG. Exiting.")
        return

    options = dag_handler.get_task_options(task_name)
    monitor = PipelineMonitor(report_path=report_file_path, stages=[task_name], data_queue=Queue())

    # 3. Aggregate target files with their source context
    items_to_process: List[Tuple[Path, Path]] = []
    
    logging.info(f"Scanning {len(session_map)} sessions for data files...")

    for search_dir in sorted(session_map.keys()):
        database_path_context = session_map[search_dir]
        candidates = list(search_dir.glob("*_semicontrolled_aggregated_session.csv"))
        
        if candidates:
            target_file = candidates[0]
            items_to_process.append((target_file, database_path_context))
        else:
            logging.warning(f"No aggregated session file found in {search_dir.name}")

    if not items_to_process:
        logging.warning("No input files found across all sessions. Exiting.")
        return

    # 4. Execute Flow
    logging.info(f"🚀 Starting analysis for {len(items_to_process)} collected items.")
    
    executor = TaskExecutor(task_name, batch_id, dag_handler, monitor)
    
    with executor:
        if executor.can_run:
            try:
                analyse_number_single_touches_flow(
                    input_items=items_to_process,
                    force_processing=options.get("force_processing", False)
                )
                
            except Exception as e:
                executor.error_msg = f"Batch analysis failed: {str(e)}"
                logging.error(f"Error during batch execution: {e}")

    logging.info("✅ Batch analysis finished.")


def main():
    freeze_support()
    
    # Configuration
    project_data_root = path_tools.get_project_data_root() 
    configs_dir = Path("configs")
    dag_config_path = configs_dir / "analyse_workflow_dag.yaml"
    
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_file_path = reports_dir / "analysis_status.xlsx"

    # Load DAG
    if not dag_config_path.exists():
        logging.error(f"DAG config not found at {dag_config_path}")
        exit(1)
        
    dag_handler = DagConfigHandler(dag_config_path)
    
    # Extract list of config directories from DAG parameters
    config_dirs_param = dag_handler.get_parameter('kinect_configs_directories')
    
    if isinstance(config_dirs_param, str):
        kinect_config_dirs = [config_dirs_param]
    elif isinstance(config_dirs_param, list):
        kinect_config_dirs = config_dirs_param
    else:
        logging.error("Parameter 'kinect_configs_directories' must be a string or a list of strings.")
        exit(1)

    run_batch_analysis(
        kinect_config_dirs=kinect_config_dirs,
        project_data_root=project_data_root,
        configs_root_path=configs_dir,
        dag_handler=dag_handler,
        report_file_path=report_file_path
    )

if __name__ == "__main__":
    main()