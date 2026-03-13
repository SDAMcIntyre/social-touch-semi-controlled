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
    generate_unified_summary,
    generate_touch_summary_matrix,
    generate_ap_efficacy_matrix,
    generate_session_summary
)
from analysis.receptive_field_mapping import RFMappingConfig
from analysis.receptive_field_mapping.rf_mapping_engine import RFMappingEngine
from analysis.receptive_field_mapping.rf_data_loader import load_grouped_spatial_data
from analysis.receptive_field_mapping.rf_visualizer import RFVisualizer

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


@flow(name="process_unified_touches")
def process_unified_touches_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
) -> List[Path]:
    """
    STEP 1: Primary Processing.
    Analyses raw session data and produces a unified summary CSV.
    """
    print(f"[Batch Analysis] Generating unified summaries for {len(input_items)} items...")
    
    results = []
    
    for input_file, database_path in input_items:
        try:
            output_dir = database_path / "4_analysed" / "unified_touches"
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
                force=force_processing,
                use_transformed=False,
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

@flow(name="map_receptive_fields")
def map_receptive_fields_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    grouping_columns: List[str] = None,
    monitor: bool = False,
) -> List[Path]:
    """
    RF Mapping: Compute per-group receptive field maps via selectivity + DBSCAN.
    Consumes unified touch summaries and raw merged CSVs.
    """
    from utils.should_process_task import should_process_task

    if grouping_columns is None:
        grouping_columns = ["type_metadata", "direction"]

    print(f"[Batch Analysis] Mapping receptive fields for {len(input_items)} item(s)...")

    config = RFMappingConfig()
    results = []

    for input_file, database_path in input_items:
        try:
            # Locate unified summary
            unified_dir = database_path / "4_analysed" / "unified_touches"
            filename = input_file.name
            if "_semicontrolled_" in filename:
                prefix = filename.split("_semicontrolled_")[0]
                summary_name = f"{prefix}_semicontrolled_touch_summary.csv"
            else:
                summary_name = f"{input_file.stem}_touch_summary.csv"

            summary_path = unified_dir / summary_name
            if not summary_path.exists():
                logging.warning(
                    f"Unified summary not found: {summary_path}. "
                    "Run 'process_unified_touches' first. Skipping."
                )
                continue

            # Output directory for RF maps
            rf_output_dir = database_path / "4_analysed" / "receptive_field_maps"
            rf_output_dir.mkdir(parents=True, exist_ok=True)

            # Collect raw merged CSVs (same session dir as input_file)
            raw_csv_dir = input_file.parent
            raw_csv_paths = sorted(raw_csv_dir.glob("*_semicontrolled_aggregated_session.csv"))
            if not raw_csv_paths:
                raw_csv_paths = [input_file]

            # Define expected outputs for idempotency
            summary_json = rf_output_dir / "rf_mapping_summary.json"
            if not should_process_task(
                input_paths=[summary_path] + raw_csv_paths,
                output_paths=[summary_json],
                force=force_processing,
            ):
                logging.info(f"[{rf_output_dir.name}] RF mapping up-to-date. Skipping.")
                results.append(summary_json)
                continue

            # Load and group spatial data
            grouped_data = load_grouped_spatial_data(
                raw_csv_paths=raw_csv_paths,
                summary_csv_path=summary_path,
                grouping_columns=grouping_columns,
                config=config,
                use_transformed=False,
            )

            if not grouped_data:
                logging.warning(f"No grouped data produced for {input_file.name}. Skipping.")
                continue

            # Process each group
            import json
            all_results = {}
            for group_label, spatial_data in grouped_data.items():
                selectivity = RFMappingEngine.compute_selectivity(
                    spatial_data.spike_counts,
                    spatial_data.total_counts,
                )
                rf_result = RFMappingEngine.cluster_receptive_field(
                    selectivity,
                    config.algorithm,
                    group_label,
                    touch_count=spatial_data.touch_count,
                )

                # Save per-group CSV
                _save_group_csv(rf_result, rf_output_dir)

                # Visualize if requested
                if monitor and rf_result.clusters:
                    session_id = filename.split("_semicontrolled_")[0]
                    forearm_pcd = _load_forearm_pcd(input_file.parent, session_id)
                    RFVisualizer.visualize_rf_map(rf_result, forearm_pcd)

                all_results[group_label] = {
                    "touch_count": rf_result.touch_count,
                    "total_points_evaluated": rf_result.total_points_evaluated,
                    "points_above_threshold": rf_result.points_above_threshold,
                    "num_clusters": len(rf_result.clusters),
                    "cluster_sizes": [c.point_count for c in rf_result.clusters],
                    "cluster_mean_selectivity": [
                        round(c.mean_selectivity, 4) for c in rf_result.clusters
                    ],
                }

            # Save summary JSON
            with open(summary_json, "w") as f:
                json.dump(
                    {
                        "grouping_columns": grouping_columns,
                        "algorithm_config": {
                            "selectivity_threshold": config.algorithm.selectivity_threshold,
                            "dbscan_eps": config.algorithm.dbscan_eps,
                            "dbscan_min_samples": config.algorithm.dbscan_min_samples,
                            "min_cluster_points": config.algorithm.min_cluster_points,
                        },
                        "groups": all_results,
                    },
                    f,
                    indent=2,
                )

            results.append(summary_json)
            logging.info(f"RF mapping complete for {input_file.name}: {len(all_results)} groups.")

        except Exception as e:
            logging.error(f"Failed RF mapping for {input_file.name}: {e}")
            import traceback
            traceback.print_exc()

    return results


def _save_group_csv(rf_result, output_dir: Path) -> Optional[Path]:
    """Save per-group RF points with selectivity and cluster assignment."""
    import pandas as pd

    if not rf_result.clusters:
        return None

    rows = []
    for cluster in rf_result.clusters:
        for i in range(cluster.point_count):
            rows.append({
                "x": cluster.points[i, 0],
                "y": cluster.points[i, 1],
                "z": cluster.points[i, 2],
                "selectivity": cluster.selectivity_scores[i],
                "cluster_id": cluster.cluster_id,
            })

    df = pd.DataFrame(rows)
    safe_label = rf_result.group_label.replace("/", "_").replace(" ", "_")
    csv_path = output_dir / f"rf_map_{safe_label}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _load_forearm_pcd(session_merged_output_dir: Path, session_id: str):
    """Load PCA-calibrated forearm PLY from postprocessing output for visualization."""
    import open3d as o3d

    pcd_path = session_merged_output_dir / "forearm_pca_calibrated" / f"{session_id}_forearm_pca_calibrated.ply"
    if pcd_path.exists():
        return o3d.io.read_point_cloud(str(pcd_path))
    logging.warning(f"Forearm PLY not found: {pcd_path}")
    return None


def _collect_unified_files(input_items: List[Tuple[Path, Path]]) -> List[Path]:
    """
    Helper to reconstruct the expected paths of the unified summary files.
    These files are expected to be in the root of '4_analysed' based on Step 1.
    """
    unified_files = []
    for input_file, database_path in input_items:
        output_dir = database_path / "4_analysed" / "unified_touches"
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
        ("analyse_ap_efficacy", analyse_ap_efficacy_flow),
        ("map_receptive_fields", map_receptive_fields_flow),
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