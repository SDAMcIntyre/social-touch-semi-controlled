# analyse_workflow.py
import argparse
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
from utils.pipeline.session_config_resolver import resolve_session_configs
from primary_processing import (
    KinectConfigFileHandler,
    KinectConfig,
)

# Imported from the updated touch_analysis module (assuming path matches 'analysis/touch_analytics')
from analysis.touch_analytics import (
    generate_unified_summary,
    generate_touch_summary_matrix,
    generate_ap_efficacy_matrix,
    generate_session_summary
)

# --- Analysis Flows ---

@flow(name="generate_session_block_summary")
def generate_session_summary_flow(
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

    output_dir = anchor_db_path / "4_analysed"
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


@flow(name="process_unified_touches")
def process_unified_touches_flow(
    input_items: List[Tuple[Path, Path]], 
    force_processing: bool = False
) -> List[Path]:
    """
    STEP 1: Primary Processing.
    Analyses raw session data and produces a unified summary CSV.
    """
    print(f"[Batch Analysis] Generating unified summaries for {len(input_items)} items...")
    
    results = []
    
    for input_file, database_path in input_items:
        try:
            output_dir = database_path / "4_analysed"
            filename = input_file.name
            
            # Standardized Filename
            if "_semicontrolled_" in filename:
                prefix = filename.split("_semicontrolled_")[0]
                new_filename = f"{prefix}_semicontrolled_touch_summary.csv"
            else:
                new_filename = f"{input_file.stem}_touch_summary.csv"

            output_file_path = output_dir / new_filename
            output_dir.mkdir(parents=True, exist_ok=True)

            # Delegate processing (and checks) to the function
            result_path = generate_unified_summary(
                input_file, 
                output_file_path, 
                show=False, 
                force=force_processing
            )
            results.append(result_path)

        except Exception as e:
            logging.error(f"Failed to process {input_file.name}: {e}")
            import traceback
            traceback.print_exc()
            
    return results


@flow(name="analyse_number_single_touches")
def analyse_number_single_touches_flow(
    input_items: List[Tuple[Path, Path]], 
    force_processing: bool = False
) -> List[Path]:
    """
    STEP 2: Matrix Generation (Counts).
    Generates a matrix of single touch counts into '4_analysed/touch_count'.
    """
    print(f"[Batch Analysis] Generating Count Matrix...")
    
    unified_files = _collect_unified_files(input_items)
    
    if unified_files:
        try:
            # Use the first database path as the anchor for the matrix output
            anchor_db_path = input_items[0][1]
            
            # Define specific subfolder for this analysis flow
            output_dir = anchor_db_path / "4_analysed" / "touch_count"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            matrix_output_path = output_dir / "batch_condition_matrix.csv"
            
            logging.info(f"Calling batch condition matrix generation. Output: {matrix_output_path}")
            # Delegate processing (and checks) to the function
            generate_touch_summary_matrix(
                unified_files, 
                matrix_output_path, 
                show=True, 
                force=force_processing
            )
            return [matrix_output_path]
        except Exception as e:
            logging.error(f"Failed to generate aggregate matrix: {e}")
            return []
    else:
        logging.warning("No unified summary files found. Run 'process_unified_touches' first.")
        return []

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
        logging.warning("No unified summary files found. Run 'process_unified_touches' first.")
        return []

def _collect_unified_files(input_items: List[Tuple[Path, Path]]) -> List[Path]:
    """
    Helper to reconstruct the expected paths of the unified summary files.
    These files are expected to be in the root of '4_analysed' based on Step 1.
    """
    unified_files = []
    for input_file, database_path in input_items:
        output_dir = database_path / "4_analysed"
        filename = input_file.name
        
        if "_semicontrolled_" in filename:
            prefix = filename.split("_semicontrolled_")[0]
            new_filename = f"{prefix}_semicontrolled_touch_summary.csv"
        else:
            new_filename = f"{input_file.stem}_touch_summary.csv"
            
        expected_path = output_dir / new_filename
        if expected_path.exists():
            unified_files.append(expected_path)
        else:
            logging.debug(f"Expected unified file missing: {expected_path}")
            
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
        ("generate_session_summary", generate_session_summary_flow),
        ("process_unified_touches", process_unified_touches_flow),
        ("analyse_number_single_touches", analyse_number_single_touches_flow),
        ("analyse_ap_efficacy", analyse_ap_efficacy_flow)
    ]
    
    task_names = [t[0] for t in available_tasks]
    monitor = PipelineMonitor(report_path=report_file_path, stages=task_names, data_queue=Queue())

    items_to_process: List[Tuple[Path, Path]] = []
    
    logging.info(f"Scanning {len(session_map)} sessions for data files...")
    for search_dir in sorted(session_map.keys()):
        database_path_context = session_map[search_dir]
        candidates = list(search_dir.glob("*_semicontrolled_aggregated_session_filtered.csv"))
        
        if candidates:
            target_file = candidates[0]
            items_to_process.append((target_file, database_path_context))

    if not items_to_process:
        logging.warning("No input files found. Exiting.")
        return

    logging.info(f"🚀 Starting analysis for {len(items_to_process)} collected items.")
    
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
                    flow_func(
                        input_items=items_to_process,
                        force_processing=options.get("force_processing", False)
                    )
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