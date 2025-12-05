from pathlib import Path
from prefect import flow
import utils.path_tools as path_tools
from utils import DagConfigHandler

from primary_processing import (
    KinectConfigFileHandler,
    KinectConfig,
    get_block_files
)

from _3_preprocessing._1_sticker_tracking import (
    review_tracked_objects_in_video,
    define_handstickers_colorspaces_from_roi,
    define_handstickers_color_threshold,
)

from _3_preprocessing._2_hand_tracking import (
    assign_stickers_location,
    curate_hamer_hand_models
)

from _3_preprocessing._5_led_tracking import (
    define_led_roi
)

from _3_preprocessing._6_metadata_matching import (
    define_trial_chunks,
    review_single_touches
)


# --- Sub-Flows (Manual Tasks) ---
@flow(name="3. Track LED Blinking")
def prepare_led_tracking(
    rgb_video_path: Path, 
    output_dir: Path, 
    *, 
    force_processing: bool = False
) -> Path:
    print(f"[{output_dir.name}] Tracking LED blinking...")
    name_baseline = rgb_video_path.stem + "_LED"
    roi_metadata_path = output_dir / (name_baseline + "_roi_metadata.json")
    define_led_roi(rgb_video_path, roi_metadata_path, force_processing=force_processing)
    return True


@flow(name="Manual: Prepare Hand Model")
def assign_stickers_location_flow(
    rgb_video_path: Path,
    hand_models_dir: Path,
    objects_to_track: list[str],
    output_dir: Path,
    *,
    force_processing: bool = False
) -> Path:
    """Manually define landmarks for the 3D hand model."""
    print(f"[{output_dir.name}] Preparing hand tracking session...")
    name_baseline = rgb_video_path.stem + "_handmodel"
    metadata_path = output_dir / (name_baseline + "_metadata.json")

    assign_stickers_location(
        rgb_video_path,
        hand_models_dir,
        objects_to_track,
        metadata_path,
        force_processing=force_processing
    )
    return metadata_path

@flow(name="Manual: Review Stickers")
def review_2d_stickers(
    rgb_video_path: Path,
    objects_to_track: list[str],
    output_dir: Path,
    *,
    force_processing: bool = False
) -> Path:
    """Manually define ROI and review sticker tracking."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{output_dir.name}] Reviewing sticker tracking...")

    name_baseline = rgb_video_path.stem + "_handstickers"
    metadata_roi_path = output_dir / (name_baseline + "_roi_metadata.json")
    stickers_roi_csv_path = output_dir / (name_baseline + "_roi_tracking.csv")

    review_tracked_objects_in_video(
        rgb_video_path,
        objects_to_track,
        metadata_roi_path,
        stickers_roi_csv_path,
        force_processing=force_processing
    )

    return stickers_roi_csv_path

@flow(name="Manual: Define Colorspace")
def prepare_stickers_colorspace(
    rgb_video_path: Path,
    output_dir: Path,
    *,
    force_processing: bool = False
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{output_dir.name}] Define sticker Colorspace...")

    name_baseline = rgb_video_path.stem + "_handstickers"
    roi_video_base_path = output_dir / (name_baseline + "_roi_unified.mp4")
    metadata_roi_path = output_dir / (name_baseline + "_roi_metadata.json")
    
    metadata_colorspace_path = output_dir / (name_baseline + "_colorspace_metadata.json")
    define_handstickers_colorspaces_from_roi(
        roi_video_base_path,
        metadata_roi_path,
        metadata_colorspace_path,
        force_processing=force_processing
    )
    return

@flow(name="Manual: Define correlation videos thresholding")
def review_handstickers_color_threshold(
    rgb_video_path: Path,
    output_dir: Path,
    *,
    force_processing: bool = False
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{output_dir.name}] Define correlation videos thresholding...")

    name_baseline = rgb_video_path.stem + "_handstickers"
    metadata_colorspace_path = output_dir / (name_baseline + "_colorspace_metadata.json")
    corrmap_video_base_path = output_dir / (name_baseline + "_corrmap.mp4")
    define_handstickers_color_threshold(
        corrmap_video_base_path, 
        md_path=metadata_colorspace_path,
        force_processing=force_processing
    )
    return

@flow(name="Manual: Define trial chunks.")
def define_trial_chunks_flow(
    rgb_video_path: Path,
    sticker_dir: Path,
    output_dir: Path,
    *,
    force_processing: bool = False
) -> Path:
    """Define trial chunks based on sticker data."""
    print(f"[{rgb_video_path.name}] Defining trial chunks...")
    xy_csv_path = sticker_dir / (rgb_video_path.stem + "_handstickers_summary_2d_coordinates.csv")
    output_path = output_dir / (rgb_video_path.stem + '_trial-chunks.csv')
    
    define_trial_chunks(
        xy_csv_path,
        rgb_video_path,
        output_csv_path=output_path,
        force_processing=force_processing
    )
    return True

@flow(name="Manual: Curate Hamer Models")
def run_curate_hamer_hand_models(
    rgb_video_path: Path,
    kinematics_dir: Path,
    temporal_dir: Path,
    *,
    force_processing: bool = False
) -> Path:
    """
    Launch the GUI to curate/validate Hamer hand tracking models.
    """
    print(f"[{rgb_video_path.name}] Curating Hamer hand models...")
    name_baseline = rgb_video_path.stem
    
    # Define paths based on naming conventions
    data_path = kinematics_dir / (name_baseline + "_handmodel_tracked_hands.pkl")
    csv_path = temporal_dir / (name_baseline + "_trial-chunks.csv")

    output_file_path = kinematics_dir / (name_baseline + "_handmodel_tracked_hands_curated.pkl")
    output_success_path = Path(str(output_file_path) + ".SUCCESS")
    
    # Verify Prerequisites
    if not data_path.exists():
        print(f"⚠️ Warning: Input tracking data not found: {data_path}")
        print("   -> Ensure automatic Hamer tracking has run before this step.")
        return False

    if not csv_path.exists():
        print(f"⚠️ Warning: Trial chunks not found: {csv_path}")
        return False

    curate_hamer_hand_models(
        video_path=rgb_video_path,
        data_path=data_path,
        csv_path=csv_path,
        output_file_path=output_file_path,
        output_success_path=output_success_path,
        force_processing=force_processing
    )
    return output_file_path

@flow(name="Manual: Review Single Touches")
def review_single_touches_flow(
    rgb_video_path: Path,
    sticker_dir: Path, 
    output_dir: Path,
    *,
    force_processing: bool = False
) -> Path:
    """Manually review and correct automatically detected single touches."""
    print(f"[{rgb_video_path.name}] Reviewing single touches...")
    
    name_baseline = rgb_video_path.stem
    # Input files generated by the Auto pipeline
    trial_ids_path = output_dir / (name_baseline + "_trial-ids.csv")
    stimuli_metadata_path = output_dir / (name_baseline + "_trial-ids_with-stimuli-data.csv")
    auto_touches_path = output_dir / (name_baseline + "_single-touches-auto.csv")       
    stickers_xyz_path = sticker_dir / (name_baseline + "_handstickers_xyz_tracked.csv")
    # Output file
    output_path = output_dir / (name_baseline + "_single-touches-corrected.csv")
    
    review_single_touches(
        rgb_video_path=rgb_video_path,
        stickers_xyz_path=stickers_xyz_path, 
        stimuli_metadata_path=stimuli_metadata_path,
        trial_data_path=trial_ids_path,
        input_touches_path=auto_touches_path,
        output_path=output_path,
        force_processing=force_processing
    )
    return auto_touches_path


# --- The "Worker" Flow ---
@flow(name="Run Single Session Manual Pipeline")
def run_single_session_pipeline(
    config: KinectConfig,
    dag_handler: DagConfigHandler
):
    """Processes a single dataset by calling manual sub-routines based on DAG config."""
    block_name = config.source_video.name
    print(f"🚀 Starting manual pipeline for block: {block_name}")
    
    rgb_video_path = config.video_primary_output_dir / f"{config.source_video.stem}.mp4"
    if not rgb_video_path.exists():
        print(f"❌ Critical Error: RGB video not found at {rgb_video_path}.")
        return {"status": "failed", "error": "RGB video not found"}

    try:
        led_dir = config.video_processed_output_dir / "temporal_segmentation/LED"
        sticker_dir = config.video_processed_output_dir / "handstickers"
        temp_seg_dir = config.video_processed_output_dir / "temporal_segmentation"
        kin_dir = config.video_processed_output_dir/ "kinematics_analysis"
        
        # 1. LED Tracking
        if dag_handler.can_run('prepare_led_tracking'):
            print(f"[{block_name}] ==> Running task: prepare_led_tracking")
            force = dag_handler.get_task_options('prepare_led_tracking').get('force_processing', False)
            prepare_led_tracking(
                rgb_video_path=rgb_video_path,
                output_dir=led_dir,
                force_processing=force
            )
            dag_handler.mark_completed('prepare_led_tracking')

        # 2. Hand Model
        if dag_handler.can_run('assign_stickers_location'):
            print(f"[{block_name}] ==> Running task: assign_stickers_location")
            force = dag_handler.get_task_options('assign_stickers_location').get('force_processing', False)
            assign_stickers_location_flow(
                rgb_video_path=rgb_video_path,
                hand_models_dir=config.hand_models_dir,
                objects_to_track=config.objects_to_track,
                output_dir= kin_dir,
                force_processing=force
            )
            dag_handler.mark_completed('assign_stickers_location')

        # 3. Review Stickers (ROI)
        if dag_handler.can_run('review_2d_stickers'):
            print(f"[{block_name}] ==> Running task: review_2d_stickers")
            force = dag_handler.get_task_options('review_2d_stickers').get('force_processing', False)
            review_2d_stickers(
                rgb_video_path=rgb_video_path,
                objects_to_track=config.objects_to_track,
                output_dir=sticker_dir,
                force_processing=force
            )
            dag_handler.mark_completed('review_2d_stickers')
        
        # 4. Prepare Colorspace
        if dag_handler.can_run('prepare_stickers_colorspace'):
            print(f"[{block_name}] ==> Running task: prepare_stickers_colorspace")
            force = dag_handler.get_task_options('prepare_stickers_colorspace').get('force_processing', False)
            prepare_stickers_colorspace(
                rgb_video_path=rgb_video_path,
                output_dir=sticker_dir,
                force_processing=force
            )
            dag_handler.mark_completed('prepare_stickers_colorspace')
        
        # 5. Review Thresholds
        if dag_handler.can_run('review_handstickers_color_threshold'):
            print(f"[{block_name}] ==> Running task: review_handstickers_color_threshold")
            force = dag_handler.get_task_options('review_handstickers_color_threshold').get('force_processing', False)
            review_handstickers_color_threshold(
                rgb_video_path=rgb_video_path,
                output_dir=sticker_dir,
                force_processing=force
            )
            dag_handler.mark_completed('review_handstickers_color_threshold')

        # 6. Define Trial Chunks
        if dag_handler.can_run('define_trial_chunks'):
            print(f"[{block_name}] ==> Running task: define_trial_chunks")
            force = dag_handler.get_task_options('define_trial_chunks').get('force_processing', False)
            define_trial_chunks_flow(
                rgb_video_path=rgb_video_path,
                sticker_dir=sticker_dir,
                output_dir=temp_seg_dir,
                force_processing=force
            )
            dag_handler.mark_completed('define_trial_chunks')

        # 7. Curate Hamer Models (NEW)
        if dag_handler.can_run('curate_hamer_hand_models'):
            print(f"[{block_name}] ==> Running task: curate_hamer_hand_models")
            force = dag_handler.get_task_options('curate_hamer_hand_models').get('force_processing', False)
            run_curate_hamer_hand_models(
                rgb_video_path=rgb_video_path,
                kinematics_dir=kin_dir,
                temporal_dir=temp_seg_dir,
                force_processing=force
            )
            dag_handler.mark_completed('curate_hamer_hand_models')

        # 8. Review Single Touches
        if dag_handler.can_run('review_single_touches'):
            print(f"[{block_name}] ==> Running task: review_single_touches")
            force = dag_handler.get_task_options('review_single_touches').get('force_processing', False)
            review_single_touches_flow(
                rgb_video_path=rgb_video_path,
                sticker_dir=sticker_dir,
                output_dir=temp_seg_dir,
                force_processing=force
            )
            dag_handler.mark_completed('review_single_touches')

    except Exception as e:
        print(f"❌ Pipeline failed during manual processing. Error: {e}")
        return {"status": "failed", "error": str(e)}

    print(f"✅ Manual pipeline finished for session: {block_name}")
    return {"status": "success", "completed_tasks": list(dag_handler.completed_tasks)}


# --- The "Dispatcher" Flow ---
@flow(name="Run Manual Batch Sequentially", log_prints=True)
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
            
            result = run_single_session_pipeline(
                config=validated_config,
                dag_handler=dag_handler_instance
            )
            print(f"--- Completed session: {block_file.name} | Status: {result.get('status', 'unknown')} ---")
        except Exception as e:
            print(f"❌ Failed to initialize session {block_file.name}. Error: {e}")
            continue
    print("✅ All sequential manual runs have completed.")


if __name__ == "__main__":
    print("🛠️  Setting up files for manual processing...")
    project_data_root = path_tools.get_project_data_root()
    configs_dir = Path("configs")
    dag_config_path = configs_dir / "preprocess_workflow_kinect_manual_dag.yaml"

    try:
        main_dag_handler = DagConfigHandler(dag_config_path)
        kinect_dir = main_dag_handler.get_parameter('kinect_configs_directory')
        kinect_configs_dir = configs_dir / kinect_dir
    except FileNotFoundError:
        print(f"❌ Error: '{dag_config_path}' not found.")
        exit(1)

    print("🚀 Launching manual batch processing SEQUENTIALLY.")
    run_batch_sequentially(
        kinect_configs_dir=kinect_configs_dir,
        project_data_root=project_data_root,
        dag_config_path=dag_config_path
    )