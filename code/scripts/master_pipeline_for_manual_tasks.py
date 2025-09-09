from pathlib import Path

from prefect import flow

import utils.path_tools as path_tools
from utils.pipeline_config_manager import DagConfigHandler

from primary_processing import (
    KinectConfigFileHandler,
    KinectConfig,
    get_block_files
)

from _3_preprocessing._1_sticker_tracking import (
    review_tracked_objects_in_video,
    view_xyz_stickers_on_depth_data
)

from _3_preprocessing._2_hand_tracking import (
    select_hand_model_characteristics
)

from _3_preprocessing._3_forearm_extraction import (
    is_forearm_valid
)

from _3_preprocessing._4_somatosensory_quantification import (
    view_somatosensory_3d_scene
)


# --- Sub-Flows (Manual Tasks) ---

@flow(name="Manual: Prepare Hand Model")
def prepare_hand_model(
    rgb_video_path: Path,
    hand_models_dir: Path,
    output_dir: Path,
    *,
    force_processing: bool = False
) -> Path:
    """Manually define landmarks for the 3D hand model."""
    print(f"[{output_dir.name}] Preparing hand tracking session...")
    name_baseline = rgb_video_path.stem + "_handmodel"
    metadata_path = output_dir / (name_baseline + "_metadata.json")

    point_labels = ["sticker_yellow", "sticker_blue", "sticker_green"]
    # NOTE: The underlying function `select_hand_model_characteristics` must be updated
    # to accept and use the `force_processing` argument.
    select_hand_model_characteristics(
        rgb_video_path,
        hand_models_dir,
        point_labels,
        metadata_path,
        force_processing=force_processing
    )
    return metadata_path

@flow(name="Manual: Review Stickers")
def review_2d_stickers(
    rgb_video_path: Path,
    root_output_dir: Path,
    *,
    force_processing: bool = False
) -> Path:
    """Manually define ROI and review sticker tracking."""
    output_dir = root_output_dir / "handstickers"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{output_dir.name}] Reviewing sticker tracking...")

    name_baseline = rgb_video_path.stem + "_handstickers"
    metadata_roi_path = output_dir / (name_baseline + "_roi_metadata.json")
    stickers_roi_csv_path = output_dir / (name_baseline + "_roi_tracking.csv")

    # NOTE: The underlying function `review_tracked_objects_in_video` must be updated
    # to accept and use the `force_processing` argument.
    review_tracked_objects_in_video(
        rgb_video_path,
        metadata_roi_path,
        stickers_roi_csv_path,
        force_processing=force_processing
    )
    return stickers_roi_csv_path


def view_xyz_stickers(
    source_video: Path,
    xyz_sticker_csv_dir: Path,
    rgb_video_path: Path,
    session_common_dir: Path,
    session_id: str
) -> Path:
    """Visualize the 3D sticker data on the depth point cloud."""
    print(f"[{rgb_video_path.name}] Validating xyz stickers extraction...")
    forearm_pointcloud_dir = session_common_dir / "forearm_pointclouds"
    metadata_filaname = session_id + "_arm_roi_metadata.json"
    forearm_metadata_path = forearm_pointcloud_dir / metadata_filaname
    if not is_forearm_valid(forearm_pointcloud_dir):
        return False
    
    # Base name derived from the 2D tracking file for consistency
    name = rgb_video_path.stem.replace('_roi_tracking', '_handstickers_xyz_tracked.csv')
    xyz_csv_path = xyz_sticker_csv_dir / name
    
    view_xyz_stickers_on_depth_data(
        xyz_csv_path, 
        source_video, 
        forearm_pointcloud_dir, 
        forearm_metadata_path, 
        rgb_video_path.name
    )
    return True


def view_somatosensory_assessement(
    source_video: Path,
    xyz_sticker_csv_dir: Path,
    rgb_video_path: Path,
    processed_dir: Path,
    session_common_dir: Path,
    session_id: str
) -> Path:
    """Visualize the complete 3D scene for somatosensory assessment."""
    print(f"[{rgb_video_path.name}] Validating somatosensory assessment...")
    forearm_pointcloud_dir = session_common_dir / "forearm_pointclouds"
    metadata_filaname = session_id + "_arm_roi_metadata.json"
    forearm_metadata_path = forearm_pointcloud_dir / metadata_filaname
    if not is_forearm_valid(forearm_pointcloud_dir):
        return False
    
    # Base name derived from the 2D tracking file for consistency
    name_baseline = rgb_video_path.stem.replace('_roi_tracking', '')
    xyz_csv_path = xyz_sticker_csv_dir / (name_baseline + "_handstickers_xyz_tracked.csv")

    name_baseline = rgb_video_path.stem + "_handmodel"
    hand_motion_glb_path = processed_dir / (name_baseline + "_motion.glb")

    view_somatosensory_3d_scene(
        xyz_csv_path, 
        source_video, 
        forearm_pointcloud_dir, 
        forearm_metadata_path, 
        rgb_video_path.name,
        hand_motion_glb_path
    )
    return True


# --- The "Worker" Flow ---
@flow(name="Run Single Session Manual Pipeline")
def run_single_session_pipeline(
    config: KinectConfig,
    dag_handler: DagConfigHandler
):
    """Processes a single dataset by calling manual sub-routines based on DAG config."""
    block_name = config.source_video.name
    print(f"🚀 Starting manual pipeline for block: {block_name}")
    
    # Assume the primary RGB video has been generated by the automatic pipeline.
    # This path is constructed based on convention.
    rgb_video_path = config.video_primary_output_dir / f"{config.source_video.stem}.mp4"
    if not rgb_video_path.exists():
        print(f"❌ Critical Error: RGB video not found at {rgb_video_path}.")
        print("Please run the 'generate_rgb_video' task from the automatic pipeline first.")
        return {"status": "failed", "error": "RGB video not found"}

    try:
        if dag_handler.can_run('prepare_hand_model'):
            print(f"[{block_name}] ==> Running task: prepare_hand_model")
            force = dag_handler.get_task_options('prepare_hand_model').get('force_processing', False)
            prepare_hand_model(
                rgb_video_path=rgb_video_path,
                hand_models_dir=config.hand_models_dir,
                output_dir=config.video_processed_output_dir,
                force_processing=force
            )
            dag_handler.mark_completed('prepare_hand_model')

        if dag_handler.can_run('review_2d_stickers'):
            print(f"[{block_name}] ==> Running task: review_2d_stickers")
            force = dag_handler.get_task_options('review_2d_stickers').get('force_processing', False)
            review_2d_stickers(
                rgb_video_path=rgb_video_path,
                root_output_dir=config.video_processed_output_dir,
                force_processing=force
            )
            dag_handler.mark_completed('review_2d_stickers')
        
        if dag_handler.can_run('view_xyz_stickers'):
            print(f"[{block_name}] ==> Running task: view_xyz_stickers")
            valid_data =  view_xyz_stickers(
                source_video=config.source_video,
                xyz_sticker_csv_dir=config.video_processed_output_dir / "handstickers",
                rgb_video_path=rgb_video_path,
                session_common_dir=config.session_processed_output_dir,
                session_id=config.session_id
            )
            if valid_data:
                dag_handler.mark_completed('view_xyz_stickers')

        if dag_handler.can_run('view_somatosensory_assessement'):
            print(f"[{block_name}] ==> Running task: view_somatosensory_assessement")
            valid_data =  view_somatosensory_assessement(
                source_video=config.source_video,
                xyz_sticker_csv_dir=config.video_processed_output_dir / "handstickers",
                rgb_video_path=rgb_video_path,
                processed_dir=config.video_processed_output_dir,
                session_common_dir=config.session_processed_output_dir,
                session_id=config.session_id
            )
            if valid_data:
                dag_handler.mark_completed('view_somatosensory_assessement')


    except Exception as e:
        print(f"❌ Pipeline failed during manual processing. Error: {e}")
        return {"status": "failed", "error": str(e)}

    print(f"✅ Manual pipeline finished for session: {block_name}")
    return {"status": "success", "completed_tasks": list(dag_handler.completed_tasks)}


# --- The "Dispatcher" Flow ---
@flow(name="Run Manual Batch Sequentially", log_prints=True)
def run_batch_sequentially(kinect_configs_dir: Path, project_data_root: Path, dag_config_path: Path):
    """Runs all session pipelines one by one, waiting for each to complete."""
    dag_handler_template = DagConfigHandler(dag_config_path)

    block_files = get_block_files(kinect_configs_dir)
    for block_file in block_files:
        print(f"--- Running session: {block_file.name} ---")
        try:
            config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
            validated_config = KinectConfig(config_data=config_data, database_path=project_data_root)
            dag_handler_instance = dag_handler_template.copy() # Use copy for a fresh run state
            
            result = run_single_session_pipeline(
                config=validated_config,
                dag_handler=dag_handler_instance
            )
            print(f"--- Completed session: {block_file.name} | Status: {result.get('status', 'unknown')} ---")
        except Exception as e:
            print(f"❌ Failed to initialize session {block_file.name}. Error: {e}")
            continue

    print("✅ All sequential manual runs have completed.")


# --- Main execution block ---
if __name__ == "__main__":
    print("🛠️  Setting up files for manual processing...")

    project_data_root = path_tools.get_project_data_root()

    configs_dir = Path("configs")
    dag_config_path = configs_dir / "kinect_manual_pipeline_dag.yaml"

    try:
        main_dag_handler = DagConfigHandler(dag_config_path)
        kinect_dir = main_dag_handler.get_parameter('kinect_configs_directory')
        kinect_configs_dir = configs_dir / kinect_dir
    except FileNotFoundError:
        print(f"❌ Error: '{dag_config_path}' not found.")
        print("Please create the DAG config file for the manual pipeline and run again.")
        exit(1)

    print("🚀 Launching manual batch processing SEQUENTIALLY.")
    run_batch_sequentially(
        kinect_configs_dir=kinect_configs_dir,
        project_data_root=project_data_root,
        dag_config_path=dag_config_path
    )