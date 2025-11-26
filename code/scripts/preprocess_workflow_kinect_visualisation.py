from pathlib import Path
from prefect import flow
import utils.path_tools as path_tools
from utils.pipeline.pipeline_config_manager import DagConfigHandler

from primary_processing import (
    KinectConfigFileHandler,
    KinectConfig,
    get_block_files
)

from _3_preprocessing._1_sticker_tracking import (
    view_ellipse_tracking,
    view_ellipse_tracking_adjusted,
    view_summary_stickers_on_rgb_data,
    view_xyz_stickers_on_depth_data
)

from _3_preprocessing._3_forearm_extraction import (
    is_forearm_valid
)

from _3_preprocessing._4_somatosensory_quantification import (
    view_somatosensory_3d_scene
)


# --- Sub-Flows (Visualization Tasks) ---

def view_ellipse_tracking_flow(
    rgb_video_path: Path,
    sticker_dir: Path
) -> Path:
    """Visualize the 3D sticker data on the depth point cloud."""
    name_baseline = rgb_video_path.stem + "_handstickers"
    print(f"[{rgb_video_path.name}] Viewing ellipse tracking...")
    
    metadata_colorspace_path = sticker_dir / (name_baseline + "_colorspace_metadata.json")
    binary_video_base_path = sticker_dir / (name_baseline + "_corrmap.mp4")
    fit_ellipses_path = sticker_dir / (name_baseline + "_ellipses.csv")

    view_ellipse_tracking(binary_video_base_path, metadata_colorspace_path, fit_ellipses_path)
    return True
    

def view_ellipse_adjusted_tracking_flow(
    rgb_video_path: Path,
    sticker_dir: Path
) -> Path:
    """Visualize the adjusted ellipse data."""
    name_baseline = rgb_video_path.stem + "_handstickers"
    print(f"[{rgb_video_path.name}] Viewing adjusted ellipse tracking...")
    fit_ellipses_path = sticker_dir / (name_baseline + "_ellipses_center_adjusted.csv")
    view_ellipse_tracking_adjusted(rgb_video_path, fit_ellipses_path)
    return True


def view_consolidated_2d_tracking_data(
    rgb_video_path: Path,
    sticker_dir: Path
) -> Path:
    """Visualize the 2D sticker summaries."""
    print(f"[{rgb_video_path.name}] Viewing summary 2d sticker data...")
    xy_csv_path = sticker_dir / (rgb_video_path.stem + "_handstickers_summary_2d_coordinates.csv")
    
    view_summary_stickers_on_rgb_data(
        xy_csv_path,
        rgb_video_path
    )
    return True


def view_xyz_stickers(
    source_video: Path,
    sticker_dir: Path,
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
        print("Forearm data invalid.")
        return False
    
    name = rgb_video_path.stem.replace('_roi_tracking', '_handstickers_xyz_tracked.csv')
    xyz_csv_path = sticker_dir / name
    
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
    sticker_dir: Path,
    rgb_video_path: Path,
    kinematics_dir: Path,
    session_common_dir: Path,
    session_id: str
) -> Path:
    """Visualize the complete 3D scene for somatosensory assessment."""
    print(f"[{rgb_video_path.name}] Validating somatosensory assessment...")
    forearm_pointcloud_dir = session_common_dir / "forearm_pointclouds"
    metadata_filaname = session_id + "_arm_roi_metadata.json"
    forearm_metadata_path = forearm_pointcloud_dir / metadata_filaname
    
    if not is_forearm_valid(forearm_pointcloud_dir):
        print("Forearm data invalid.")
        return False
    
    name_baseline = rgb_video_path.stem.replace('_roi_tracking', '')
    xyz_csv_path = sticker_dir / (name_baseline + "_handstickers_xyz_tracked.csv")

    name_baseline = rgb_video_path.stem + "_handmodel"
    # UPDATED: Path now resolves to kinematics_analysis directory
    hand_motion_glb_path = kinematics_dir / (name_baseline + "_motion.glb")
    
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
@flow(name="Run Single Session Visualization")
def run_single_session_visualization(
    config: KinectConfig,
    dag_handler: DagConfigHandler
):
    """Processes a single dataset by calling visualization sub-routines."""
    block_name = config.source_video.name
    print(f"🚀 Starting visualization pipeline for block: {block_name}")
    
    rgb_video_path = config.video_primary_output_dir / f"{config.source_video.stem}.mp4"
    if not rgb_video_path.exists():
        print(f"❌ Critical Error: RGB video not found at {rgb_video_path}.")
        return {"status": "failed", "error": "RGB video not found"}

    try:
        # Standardized directories based on manual workflow
        stickers_dir = config.video_processed_output_dir / "handstickers"
        kin_dir = config.video_processed_output_dir / "kinematics_analysis"

        if dag_handler.can_run('view_ellipse_tracking'):
            print(f"[{block_name}] ==> Running task: view_ellipse_tracking")
            view_ellipse_tracking_flow(
                rgb_video_path=rgb_video_path,
                sticker_dir=stickers_dir
            )
            dag_handler.mark_completed('view_ellipse_tracking')

        if dag_handler.can_run('view_ellipse_tracking_adjusted'):
            print(f"[{block_name}] ==> Running task: view_ellipse_tracking_adjusted")
            view_ellipse_adjusted_tracking_flow(
                rgb_video_path=rgb_video_path,
                sticker_dir=stickers_dir
            )
            dag_handler.mark_completed('view_ellipse_tracking_adjusted')

        if dag_handler.can_run('view_consolidated_2d_tracking_data'):
            print(f"[{block_name}] ==> Running task: view_consolidated_2d_tracking_data")
            view_consolidated_2d_tracking_data(
                rgb_video_path=rgb_video_path,
                sticker_dir=stickers_dir
            )
            dag_handler.mark_completed('view_consolidated_2d_tracking_data')

        if dag_handler.can_run('view_xyz_stickers'):
            print(f"[{block_name}] ==> Running task: view_xyz_stickers")
            view_xyz_stickers(
                source_video=config.source_video,
                sticker_dir=stickers_dir,
                rgb_video_path=rgb_video_path,
                session_common_dir=config.session_processed_output_dir,
                session_id=config.session_id
            )
            dag_handler.mark_completed('view_xyz_stickers')

        if dag_handler.can_run('view_somatosensory_assessement'):
            print(f"[{block_name}] ==> Running task: view_somatosensory_assessement")
            view_somatosensory_assessement(
                source_video=config.source_video,
                sticker_dir=stickers_dir,
                rgb_video_path=rgb_video_path,
                kinematics_dir=kin_dir,
                session_common_dir=config.session_processed_output_dir,
                session_id=config.session_id
            )
            dag_handler.mark_completed('view_somatosensory_assessement')

    except Exception as e:
        print(f"❌ Pipeline failed during visualization. Error: {e}")
        return {"status": "failed", "error": str(e)}

    print(f"✅ Visualization finished for session: {block_name}")
    return {"status": "success", "completed_tasks": list(dag_handler.completed_tasks)}


# --- The "Dispatcher" Flow ---
@flow(name="Run Visualization Batch Sequentially", log_prints=True)
def run_batch_sequentially(kinect_configs_dir: Path, project_data_root: Path, dag_config_path: Path):
    """Runs all session pipelines one by one."""
    dag_handler_template = DagConfigHandler(dag_config_path)
    block_files = get_block_files(kinect_configs_dir)
    
    for block_file in block_files:
        print(f"--- Running session: {block_file.name} ---")
        try:
            config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
            validated_config = KinectConfig(config_data=config_data, database_path=project_data_root)
            dag_handler_instance = dag_handler_template.copy()
            
            result = run_single_session_visualization(
                config=validated_config,
                dag_handler=dag_handler_instance
            )
            print(f"--- Completed session: {block_file.name} | Status: {result.get('status', 'unknown')} ---")
        except Exception as e:
            print(f"❌ Failed to initialize session {block_file.name}. Error: {e}")
            continue
    print("✅ All sequential visualization runs have completed.")


if __name__ == "__main__":
    print("🛠️  Setting up files for visualization...")
    project_data_root = path_tools.get_project_data_root()
    configs_dir = Path("configs")
    dag_config_path = configs_dir / "preprocess_workflow_kinect_visualisation_dag.yaml"

    try:
        main_dag_handler = DagConfigHandler(dag_config_path)
        kinect_dir = main_dag_handler.get_parameter('kinect_configs_directory')
        kinect_configs_dir = configs_dir / kinect_dir
    except FileNotFoundError:
        print(f"❌ Error: '{dag_config_path}' not found.")
        exit(1)

    print("🚀 Launching visualization batch processing SEQUENTIALLY.")
    run_batch_sequentially(
        kinect_configs_dir=kinect_configs_dir,
        project_data_root=project_data_root,
        dag_config_path=dag_config_path
    )