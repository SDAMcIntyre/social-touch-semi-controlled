import argparse
import csv
import logging
from pathlib import Path
from multiprocessing import freeze_support

# Setup a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import utils.path_tools as path_tools
from utils import DagConfigHandler
from utils.pipeline.session_config_resolver import resolve_session_configs
from primary_processing import KinectConfigFileHandler, KinectConfig

from _3_preprocessing._8_nerve_velocity_adjustment import adjust_nerve_conduction_velocity
from _3_preprocessing._9_nerve_data_extraction import (
    convert_nerve_mat_to_csv,
    rename_nerve_to_block_order,
)

# --- Session-level Flows ---

def convert_mat_to_csv_flow(
    config: KinectConfig,
    dag_handler: DagConfigHandler,
    nerve_mat_dir: Path,
    nerve_csv_output_dir: Path,
) -> list[dict]:
    task_name = "convert_mat_to_csv"
    options = dag_handler.get_task_options(task_name)
    force = options.get("force_processing", False)

    mat_files = sorted(nerve_mat_dir.glob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in: {nerve_mat_dir}")

    session_short = config.session_id.split("_")[-1]  # e.g. "ST14-01" from "2022-06-15_ST14-01"
    session_mat_files = [f for f in mat_files if session_short in f.stem]
    if not session_mat_files:
        logger.warning(f"[{config.session_id}] No .mat files matching '{session_short}' in {nerve_mat_dir}")
        return []

    all_results = []
    for mat_path in session_mat_files:
        logger.info(f"[{config.session_id}] Converting {mat_path.name}")
        results = convert_nerve_mat_to_csv(mat_path, nerve_csv_output_dir, force_processing=force)
        if results:
            all_results.extend(results)
    return all_results


def rename_to_block_order_flow(
    config: KinectConfig,
    dag_handler: DagConfigHandler,
    nerve_csv_output_dir: Path,
    nerve_block_order_output_dir: Path,
    quality_check_xlsx: Path,
) -> list[dict]:
    task_name = "rename_to_block_order"
    options = dag_handler.get_task_options(task_name)
    force = options.get("force_processing", False)

    logger.info(f"[{config.session_id}] Renaming CSVs to block-order convention")
    results = rename_nerve_to_block_order(
        csv_dir=nerve_csv_output_dir,
        output_dir=nerve_block_order_output_dir,
        quality_check_xlsx=quality_check_xlsx,
        session_id=config.session_id,
        force_processing=force,
    )
    return results or []


def adjust_conduction_velocity_flow(
    config: KinectConfig,
    dag_handler: DagConfigHandler,
    metadata_csv_path: Path,
) -> list[dict]:

    if config.nerve_processed_dir is None:
        raise ValueError(f"nerve_processed_dir not set for session {config.session_id}")

    block_order_dir = (
        config.nerve_processed_dir.parent.parent / "2_block-order" / config.session_id
    )

    if not block_order_dir.exists():
        raise FileNotFoundError(f"Block-order input directory not found: {block_order_dir}")

    nerve_files = sorted(block_order_dir.glob("*_nerve.csv"))

    if not nerve_files:
        raise FileNotFoundError(f"No *_nerve.csv files found in {block_order_dir}")

    task_name = "adjust_conduction_velocity"
    options = dag_handler.get_task_options(task_name)
    force = options.get("force_processing", False)

    report_rows = []
    for nerve_file in nerve_files:
        output_csv_path = config.nerve_processed_dir / nerve_file.name
        logger.info(f"[{config.session_id}] Processing {nerve_file.name}")
        result = adjust_nerve_conduction_velocity(
            nerve_file,
            metadata_csv_path,
            output_csv_path,
            force_processing=force,
        )
        if result is not None:
            report_rows.append(result)

    if report_rows:
        report_path = config.nerve_processed_dir / "conduction_velocity_lag_report.csv"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["filename", "lag_sec", "lag_nsample"])
            writer.writeheader()
            writer.writerows(report_rows)
        logger.info(f"[{config.session_id}] Wrote lag report: {report_path.name}")

    return report_rows


# --- Single Session Pipeline ---

def run_single_session_pipeline(
    config: KinectConfig,
    dag_handler: DagConfigHandler,
    metadata_csv_path: Path,
    nerve_mat_dir: Path,
    nerve_csv_output_dir: Path,
    nerve_block_order_output_dir: Path,
    quality_check_xlsx: Path,
) -> dict:
    logger.info(f"Starting pipeline for session: {config.session_id}")

    try:
        task_name = "convert_mat_to_csv"
        if dag_handler.can_run(task_name):
            logger.info(f"[{config.session_id}] ==> Running task: {task_name}")
            convert_mat_to_csv_flow(
                config=config,
                dag_handler=dag_handler,
                nerve_mat_dir=nerve_mat_dir,
                nerve_csv_output_dir=nerve_csv_output_dir,
            )
            dag_handler.mark_completed(task_name)

        task_name = "rename_to_block_order"
        if dag_handler.can_run(task_name):
            logger.info(f"[{config.session_id}] ==> Running task: {task_name}")
            rename_to_block_order_flow(
                config=config,
                dag_handler=dag_handler,
                nerve_csv_output_dir=nerve_csv_output_dir,
                nerve_block_order_output_dir=nerve_block_order_output_dir,
                quality_check_xlsx=quality_check_xlsx,
            )
            dag_handler.mark_completed(task_name)

        task_name = "adjust_conduction_velocity"
        if dag_handler.can_run(task_name):
            logger.info(f"[{config.session_id}] ==> Running task: {task_name}")
            adjust_conduction_velocity_flow(
                config=config,
                dag_handler=dag_handler,
                metadata_csv_path=metadata_csv_path,
            )
            dag_handler.mark_completed(task_name)

    except Exception as e:
        logger.error(f"Pipeline failed for {config.session_id}: {e}")
        return {"status": "failed", "session_id": config.session_id, "error": str(e)}

    logger.info(f"Pipeline finished successfully for session: {config.session_id}")
    return {"status": "success", "session_id": config.session_id, "error": None}


# --- Batch Dispatcher ---

def run_batch_processing(
    block_files: list[Path],
    project_data_root: Path,
    dag_config_path: Path,
):
    logger.info(f"Starting batch processing for {len(block_files)} block files.")

    dag_handler_template = DagConfigHandler(dag_config_path)

    metadata_csv_rel = dag_handler_template.get_parameter("nerve_metadata_csv")
    if not metadata_csv_rel:
        raise ValueError("nerve_metadata_csv parameter is not configured in DAG YAML")
    metadata_csv_path = project_data_root / metadata_csv_rel
    if not metadata_csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv_path}")

    nerve_mat_dir_rel = dag_handler_template.get_parameter("nerve_mat_dir")
    if not nerve_mat_dir_rel:
        raise ValueError("nerve_mat_dir parameter is not configured in DAG YAML")
    nerve_mat_dir = project_data_root / nerve_mat_dir_rel

    nerve_csv_output_dir_rel = dag_handler_template.get_parameter("nerve_csv_output_dir")
    if not nerve_csv_output_dir_rel:
        raise ValueError("nerve_csv_output_dir parameter is not configured in DAG YAML")
    nerve_csv_output_dir = project_data_root / nerve_csv_output_dir_rel

    nerve_block_order_output_dir_rel = dag_handler_template.get_parameter("nerve_block_order_output_dir")
    if not nerve_block_order_output_dir_rel:
        raise ValueError("nerve_block_order_output_dir parameter is not configured in DAG YAML")
    nerve_block_order_output_dir = project_data_root / nerve_block_order_output_dir_rel

    quality_check_xlsx_rel = dag_handler_template.get_parameter("quality_check_xlsx")
    if not quality_check_xlsx_rel:
        raise ValueError("quality_check_xlsx parameter is not configured in DAG YAML")
    quality_check_xlsx = project_data_root / quality_check_xlsx_rel

    processed_sessions: set[str] = set()
    results = []

    for block_file in block_files:
        try:
            config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
            config = KinectConfig(config_data=config_data, database_path=project_data_root)
        except Exception as e:
            logger.error(f"Failed to load config {block_file}: {e}")
            continue

        if config.session_id in processed_sessions:
            logger.info(f"Skipping already-processed session: {config.session_id}")
            continue

        processed_sessions.add(config.session_id)
        dag_handler_instance = dag_handler_template.copy()

        result = run_single_session_pipeline(
            config=config,
            dag_handler=dag_handler_instance,
            metadata_csv_path=metadata_csv_path,
            nerve_mat_dir=nerve_mat_dir,
            nerve_csv_output_dir=nerve_csv_output_dir,
            nerve_block_order_output_dir=nerve_block_order_output_dir,
            quality_check_xlsx=quality_check_xlsx,
        )
        results.append(result)

    failed = [r for r in results if r["status"] == "failed"]
    logger.info(
        f"Batch complete. {len(results)} sessions processed, {len(failed)} failed."
    )
    if failed:
        for r in failed:
            logger.error(f"  FAILED: {r['session_id']} — {r['error']}")


# --- Entry Point ---

def main():
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dag-config", type=Path, required=True)
    args = parser.parse_args()

    project_data_root = path_tools.get_project_data_root()
    configs_dir = Path("configs")

    dag_config_path = args.dag_config
    dag_handler = DagConfigHandler(dag_config_path)
    entries = dag_handler.get_parameter("kinect_configs")
    block_files = resolve_session_configs(entries, configs_dir / "kinect_configs")

    run_batch_processing(block_files, project_data_root, dag_config_path)


if __name__ == "__main__":
    main()
