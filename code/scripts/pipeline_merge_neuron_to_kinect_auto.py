import os
from pathlib import Path
from prefect import flow

import utils.path_tools as path_tools
from utils.pipeline_config_manager import DagConfigHandler

from primary_processing import (
    KinectConfigFileHandler,
    KinectConfig,
    get_block_files
)

from _4_merging import (
    align_and_merge_neural_and_kinect
)


@flow(name="9. Unify Dataset")
def unify_dataset(session_id: str, source_video: Path, output_dir: Path, *, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Generating unified dataset...")
    name_baseline = Path(source_video).stem.replace("_kinect", "")
    contact_filename = output_dir / (name_baseline + "_kinect_contact_and_kinematic_data_withTTL.csv")
    nerve_filename = contact_filename
    output_filename = "output.csv"

    base_database_path = str(nerve_filename).split("2_processed")[0]
    nerve_folder = os.path.join(base_database_path, "2_processed", "nerve\\3_cond-velocity-adj", session_id)
    nerve_filename = Path(os.path.join(nerve_folder, (name_baseline + "_nerve.csv")))

    output_dir = os.path.join(base_database_path, "3_merged", "1_kinect_and_nerve", session_id)
    output_filename = output_dir / Path(name_baseline + "_merged_data.csv")
    
    align_and_merge_neural_and_kinect(
        contact_filename,
        nerve_filename,
        output_filename,
        force_processing=force_processing
    )
    return output_filename


# --- 4. The "Worker" Flow ---
@flow(name="Run Single Session Pipeline")
def run_single_session_pipeline(
    config: KinectConfig,
    dag_handler: DagConfigHandler
):
    """
    Processes a single dataset by explicitly calling processing functions
    based on a DAG configuration.
    """
    block_name = config.source_video.name
    print(f"🚀 Starting pipeline for block: {block_name}")

    # --- Stage 5:  data integration ---
    try:
        if dag_handler.can_run('unify_dataset'):
            print(f"[{block_name}] ==> Running task: unify_dataset")
            force = dag_handler.get_task_options('unify_dataset').get('force_processing', False)
            unify_dataset(
                session_id=config.session_id,
                source_video=config.source_video,
                output_dir=config.video_processed_output_dir,
                force_processing=force
            )
            dag_handler.mark_completed('unify_dataset')
    except Exception as e:
        print(f"❌ Pipeline failed during Stage 5: Data Integration. Error: {e}")
        return {"status": "failed", "stage": 5, "error": str(e)}

    print(f"✅ Pipeline finished successfully for session: {block_name}")
    return {"status": "success", "completed_tasks": list(dag_handler.completed_tasks)}


# --- 5. The "Dispatcher" Flows ---
@flow(name="Batch Process All Sessions", log_prints=True)
def run_batch_in_parallel(kinect_configs_dir: Path, project_data_root: Path, dag_config_path: Path):
    """Finds all session YAML files and triggers a pipeline run for each one in parallel."""
    dag_handler = DagConfigHandler(dag_config_path)

    block_files = get_block_files(kinect_configs_dir)
    for block_file in block_files:
        print(f"Dispatching run for {block_file.name}...")
        config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
        validated_config = KinectConfig(config_data=config_data, database_path=project_data_root)
        dag_handler_copy = dag_handler.copy() # Use copy to ensure isolated state
        run_single_session_pipeline.submit(
            config=validated_config,
            dag_handler=dag_handler_copy,
            flow_run_name=f"session-{validated_config.session_id}"
        )
    print(f"All {len(block_files)} session flows have been submitted.")

@flow(name="Run Batch Sequentially", log_prints=True)
def run_batch_sequentially(kinect_configs_dir: Path, project_data_root: Path, dag_config_path: Path):
    """Runs all session pipelines one by one, waiting for each to complete."""
    dag_handler_template = DagConfigHandler(dag_config_path)

    block_files = get_block_files(kinect_configs_dir)
    for block_file in block_files:
        print(f"--- Running session: {block_file.name} ---")
        config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
        validated_config = KinectConfig(config_data=config_data, database_path=project_data_root)
        dag_handler_instance = dag_handler_template.copy() # Use copy for a fresh run state
        result = run_single_session_pipeline(
            config=validated_config,
            dag_handler=dag_handler_instance
        )
        print(f"--- Completed session: {block_file.name} | Status: {result.get('status', 'unknown')} ---")
    print("✅ All sequential runs have completed.")


# --- 6. Main execution block ---
if __name__ == "__main__":
    print("🛠️ Setting up files for processing...")

    project_data_root = path_tools.get_project_data_root()
    configs_dir = Path("configs")
    dag_config_path = Path(configs_dir / "nerve-kinect-merging_automatic_pipeline_dag.yaml")

    try:
        main_dag_handler = DagConfigHandler(dag_config_path)
        is_parallel = main_dag_handler.get_parameter('parallel_execution', False)
        kinect_dir = main_dag_handler.get_parameter('kinect_configs_directory')
        kinect_configs_dir = configs_dir / kinect_dir
    except FileNotFoundError:
        print(f"❌ Error: '{dag_config_path}' not found.")
        print("Please create it using the example provided and run the script again.")
        exit(1)

    if is_parallel:
        print("🚀 Launching batch processing in PARALLEL.")
        run_batch_in_parallel(
            kinect_configs_dir=kinect_configs_dir,
            project_data_root=project_data_root,
            dag_config_path=dag_config_path
        )
    else:
        print("🚀 Launching batch processing SEQUENTIALLY.")
        run_batch_sequentially(
            kinect_configs_dir=kinect_configs_dir,
            project_data_root=project_data_root,
            dag_config_path=dag_config_path
        )