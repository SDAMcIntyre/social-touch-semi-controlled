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

from _2_primary_processing._2_generate_rgb_depth_video import (
    generate_mkv_stream_analysis,
    extract_depth_to_tiff,
    extract_color_to_mp4
)

from _3_preprocessing._1_sticker_tracking import (
    track_objects_in_video,
    is_2d_stickers_tracking_valid,
    is_correlation_videos_threshold_defined,
    
    generate_standard_roi_size_dataset,
    create_standardized_roi_videos,
    create_color_correlation_videos,
    fit_ellipses_on_correlation_videos,
    adjust_ellipse_centers_to_global_frame,
    consolidate_2d_tracking_data,
    
    extract_stickers_xyz_positions
)

from _3_preprocessing._2_hand_tracking import (
    generate_hand_motion,
    is_hand_model_valid
)

from _3_preprocessing._3_forearm_extraction import (
    is_forearm_valid
)

from _3_preprocessing._4_somatosensory_quantification import (
    compute_somatosensory_characteristics
)

from _3_preprocessing._5_led_tracking import (
    define_led_roi,
    generate_LED_roi_video,
    track_led_states_changes,
    validate_and_correct_led_timing_from_stimuli
)



# --- 3. Sub-Flows (Formerly Tasks) ---
@flow(name="0. Analyse MKV video")
def validate_mkv_video(source_video: Path, output_dir: Path, *, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Analysing MKV video...")
    analysis_csv_path = output_dir / "mkv_analysis_report.csv"
    # Propagate the flag to the underlying implementation function
    return generate_mkv_stream_analysis(source_video, analysis_csv_path, force_processing=force_processing)

@flow(name="1. Generate RGB Video")
def generate_rgb_video(source_video: Path, output_dir: Path, *, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Generating RGB video...")
    base_filename = os.path.splitext(os.path.basename(source_video))[0]
    rgb_path = output_dir / f"{base_filename}.mp4"
    # Propagate the flag to the underlying implementation function
    rgb_video_path = extract_color_to_mp4(source_video, rgb_path, force_processing=force_processing)
    return Path(rgb_video_path) if not isinstance(rgb_video_path, Path) else rgb_video_path

@flow(name="2. Generate Depth Images")
def generate_depth_images(source_video: Path, output_dir: Path, *, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Generating depth images...")
    depth_dir = output_dir / source_video.name.replace(".mkv", "_depth")
    # Propagate the flag to the underlying implementation function
    extract_depth_to_tiff(source_video, depth_dir, force_processing=force_processing)
    return depth_dir

@flow(name="3. Track LED Blinking")
def track_led_blinking(video_path: Path, stimulus_metadata: Path, output_dir: Path, *, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Tracking LED blinking...")
    name_baseline = video_path.stem + "_LED"
    metadata_path = output_dir / (name_baseline + "_roi_metadata.txt")
    # Propagate the flag to all underlying implementation functions in this flow
    define_led_roi(video_path, metadata_path, force_processing=force_processing)
    video_led_path = output_dir / (name_baseline + "_roi.mp4")
    generate_LED_roi_video(video_path, metadata_path, video_led_path, force_processing=force_processing)
    csv_led_path = output_dir / (name_baseline + ".csv")
    metadata_led_state_path = output_dir / (name_baseline + "_metadata.txt")
    track_led_states_changes(video_led_path, csv_led_path, metadata_led_state_path, force_processing=force_processing)
    csv_led_path_corrected = output_dir / (name_baseline + "_corrected.csv")
    validate_and_correct_led_timing_from_stimuli(csv_led_path, stimulus_metadata, csv_led_path_corrected, force_processing=force_processing)
    return csv_led_path_corrected

@flow(name="4. Generate TTL Signal")
def generate_ttl_signal(led_tracking_path: Path, output_dir: Path, *, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Generating TTL signal...")
    # This is a placeholder for your actual logic.
    # If it had an implementation, you would pass `force_processing` to it.
    print("✅ TTL extracted (placeholder).")
    return True

@flow(name="5. Validate Forearm Extraction")
def validate_forearm_extraction(session_output_dir: Path) -> Path:
    print(f"[{session_output_dir.name}] Validating forearm extraction...")
    return is_forearm_valid(session_output_dir / "forearm_pointclouds", verbose=True)

@flow(name="5. Validate Hand Extraction")
def validate_hand_extraction(rgb_video_path: Path, hand_models_dir: Path, output_dir: Path) -> Path:
    print(f"[{output_dir.name}] Validating hand extraction...")
    name_baseline = rgb_video_path.stem + "_handmodel"
    metadata_path = output_dir / (name_baseline + "_metadata.json")
    expected_labels = ["sticker_yellow", "sticker_blue", "sticker_green"]
    return is_hand_model_valid(metadata_path, hand_models_dir, expected_labels, verbose=True)

@flow(name="6. Track Stickers (2D)")
def track_stickers(rgb_video_path: Path, output_dir: Path, *, force_processing: bool = False) -> tuple[Path | None, bool]:
    print(f"[{output_dir.name}] Tracking stickers (2D)...")
    name_baseline = rgb_video_path.stem + "_handstickers"
    metadata_roi_path = output_dir / (name_baseline + "_roi_metadata.json")
    stickers_roi_csv_path = output_dir / (name_baseline + "_roi_tracking.csv")

    track_objects_in_video(rgb_video_path, metadata_roi_path, output_path=stickers_roi_csv_path, force_processing=force_processing)
    
    if not is_2d_stickers_tracking_valid(metadata_roi_path):
        print("❌ --> 2D sticker tracking has not been manually valided. Cannot continue the pipeline.")
        return stickers_roi_csv_path, False
    
    roi_unified_csv_path = output_dir / (name_baseline + "_roi_standard_size.csv")
    generate_standard_roi_size_dataset(stickers_roi_csv_path, roi_unified_csv_path)
    corrmap_video_base_path = output_dir / (name_baseline + "_roi_unified.mp4")
    create_standardized_roi_videos(roi_unified_csv_path, rgb_video_path, corrmap_video_base_path, force_processing=force_processing) # force_processing)
    
    # defined by manual tasks pipeline
    metadata_colorspace_path = output_dir / (name_baseline + "_colorspace_metadata.json")
    binary_video_base_path = output_dir / (name_baseline + "_corrmap.mp4")
    create_color_correlation_videos(corrmap_video_base_path, metadata_colorspace_path, binary_video_base_path, force_processing=force_processing)
    
    if not is_correlation_videos_threshold_defined(metadata_colorspace_path):
        print("❌ --> correlation videos threshold has not been manually valided. Cannot continue the pipeline.Execute the corresponding manual task. ")
        return binary_video_base_path, False
    
    fit_ellipses_path = output_dir / (name_baseline + "_ellipses.csv")
    fit_ellipses_on_correlation_videos(
        video_path=binary_video_base_path,
        md_path=metadata_colorspace_path,
        output_path=fit_ellipses_path,
        force_processing=force_processing)
    
    adj_ellipses_path = output_dir / (name_baseline + "_ellipses_center_adjusted.csv")
    adjust_ellipse_centers_to_global_frame(
        roi_unified_csv_path,
        fit_ellipses_path,
        output_csv_path=adj_ellipses_path,
        force_processing=force_processing
    )
    
    final_csv_path = output_dir / (name_baseline + "_summary_2d_coordinates.csv")
    consolidate_2d_tracking_data(
        roi_unified_csv_path,
        adj_ellipses_path,
        output_csv_path=final_csv_path,
        score_threshold=0.7,
        force_processing=force_processing
    )

    return final_csv_path, True

@flow(name="6. Generate XYZ Sticker Positions (3D)")
def generate_xyz_stickers(stickers_2d_path: Path, source_video: Path, output_dir: Path, *, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Generating XYZ sticker positions (3D)...")
    # Base name derived from the 2D tracking file for consistency
    name_baseline = stickers_2d_path.stem.replace('_summary_2d_coordinates', '')
    result_csv_path = output_dir / (name_baseline + "_xyz_tracked.csv")
    result_md_path = output_dir / (name_baseline + "_xyz_tracked_metadata.json")
    method = "centroid"

    # Propagate the flag to the underlying implementation function
    extract_stickers_xyz_positions(
        source_video, 
        stickers_2d_path,
        method,
        result_csv_path,
        result_md_path,
        force_processing=force_processing
    )
    return result_csv_path

@flow(name="7. Generate 3D Hand Position")
def generate_3d_hand_motion(rgb_video_path: Path, stickers_xyz_path: Path, hand_models_dir: Path, output_dir: Path, *, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Generating 3D hand motion...")
    name_baseline = rgb_video_path.stem + "_handmodel"
    metadata_path = output_dir / (name_baseline + "_metadata.json")
    hand_motion_glb_path = output_dir / (name_baseline + "_motion.glb")
    hand_motion_csv_path = output_dir / (name_baseline + "_motion.csv")
    # Propagate the flag to the underlying implementation function
    generate_hand_motion(stickers_xyz_path, hand_models_dir, metadata_path, hand_motion_glb_path, hand_motion_csv_path, force_processing=force_processing)
    return hand_motion_glb_path, metadata_path

@flow(name="8. Generate Somatosensory Characteristics")
def generate_somatosensory_chars(
    hand_motion_glb_path: Path,
    hand_metadata_path: Path,
    session_processed_dir: Path,
    session_id: str,
    current_video_filename: str,
    output_dir: Path,
    *,
    monitor: bool = False,
    force_processing: bool = False
) -> Path:
    print(f"[{output_dir.name}] Generating somatosensory characteristics...")
    contact_characteristics_path = output_dir / "contact_characteristics.csv"
    forearm_pointcloud_dir = session_processed_dir / "forearm_pointclouds"
    metadata_filaname = session_id + "_arm_roi_metadata.json"
    metadata_path = forearm_pointcloud_dir / metadata_filaname
    # Propagate the flag to the underlying implementation function
    compute_somatosensory_characteristics(
        hand_motion_glb_path, hand_metadata_path,
        metadata_path, forearm_pointcloud_dir,
        current_video_filename, contact_characteristics_path, 
        monitor=monitor, force_processing=force_processing
    )
    return contact_characteristics_path

@flow(name="9. Unify Dataset")
def unify_dataset(contact_chars_path: Path, ttl_path: Path, output_dir: Path, *, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Generating unified dataset...")
    unified_path = output_dir / "unified_dataset.csv"
    # Placeholder for actual implementation. If it existed, it would receive `force_processing`.
    return unified_path


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

    # --- Result placeholders ---
    rgb_video_path = None
    led_tracking_path = None
    ttl_signal_path = None
    sticker_2d_tracking_path = None
    sticker_3d_tracking_path = None
    hand_motion_glb_path = None
    somatosensory_chars_path = None

    # --- Stage 1: Primary Video Processing ---
    try:
        if dag_handler.can_run('validate_mkv_video'):
            print(f"[{block_name}] ==> Running task: validate_mkv_video")
            # Get `force_processing` flag from DAG config for the specific task
            force = dag_handler.get_task_options('validate_mkv_video').get('force_processing', False)
            validate_mkv_video(
                source_video=config.source_video,
                output_dir=config.video_primary_output_dir,
                force_processing=force
            )
            dag_handler.mark_completed('validate_mkv_video')

        if dag_handler.can_run('generate_rgb_video'):
            print(f"[{block_name}] ==> Running task: generate_rgb_video")
            force = dag_handler.get_task_options('generate_rgb_video').get('force_processing', False)
            rgb_video_path = generate_rgb_video(
                source_video=config.source_video,
                output_dir=config.video_primary_output_dir,
                force_processing=force
                    )
            dag_handler.mark_completed('generate_rgb_video')

        if dag_handler.can_run('generate_depth_images'):
            print(f"[{block_name}] ==> Running task: generate_depth_images")
            force = dag_handler.get_task_options('generate_depth_images').get('force_processing', False)
            generate_depth_images(
                source_video=config.source_video,
                output_dir=config.video_primary_output_dir,
                force_processing=force
            )
            dag_handler.mark_completed('generate_depth_images')
    except Exception as e:
        print(f"❌ Pipeline failed during Stage 1: Primary Processing. Error: {e}")
        return {"status": "failed", "stage": 1, "error": str(e)}

    # --- Stage 2: LED Tracking for TTL signal ---
    try:
        if dag_handler.can_run('track_led_blinking'):
            print(f"[{block_name}] ==> Running task: track_led_blinking")
            force = dag_handler.get_task_options('track_led_blinking').get('force_processing', False)
            led_dir = config.video_processed_output_dir / "LED"
            led_tracking_path = track_led_blinking(
                video_path=rgb_video_path,
                stimulus_metadata=config.stimulus_metadata,
                output_dir=led_dir,
                force_processing=force
            )
            dag_handler.mark_completed('track_led_blinking')

        if dag_handler.can_run('generate_ttl_signal'):
            print(f"[{block_name}] ==> Running task: generate_ttl_signal")
            force = dag_handler.get_task_options('generate_ttl_signal').get('force_processing', False)
            ttl_signal_path = generate_ttl_signal(
                led_tracking_path=led_tracking_path,
                output_dir=config.video_processed_output_dir,
                force_processing=force
            )
            dag_handler.mark_completed('generate_ttl_signal')
    except Exception as e:
        print(f"❌ Pipeline failed during Stage 2: LED & TTL. Error: {e}")
        return {"status": "failed", "stage": 2, "error": str(e)}

    # --- Stage 3: Hand and Forearm Validation ---
    try:
        if dag_handler.can_run('validate_forearm_extraction'):
            print(f"[{block_name}] ==> Running task: validate_forearm_extraction")
            valid_data = validate_forearm_extraction(
                config.session_processed_output_dir
            )
            if valid_data:
                dag_handler.mark_completed('validate_forearm_extraction')

        if dag_handler.can_run('validate_hand_extraction'):
            print(f"[{block_name}] ==> Running task: validate_hand_extraction")
            valid_data = validate_hand_extraction(
                rgb_video_path=rgb_video_path,
                hand_models_dir=config.hand_models_dir,
                output_dir=config.video_processed_output_dir
            )
            if valid_data:
                dag_handler.mark_completed('validate_hand_extraction')
    except Exception as e:
        print(f"❌ Pipeline failed during Stage 3: Validation. Error: {e}")
        return {"status": "failed", "stage": 3, "error": str(e)}

    # --- Stage 4: Hand/Sticker Tracking and 3D reconstruction ---
    try:
        handstickers_dir = config.video_processed_output_dir / "handstickers"
        if dag_handler.can_run('track_stickers'):
            print(f"[{block_name}] ==> Running task: track_stickers")
            force = dag_handler.get_task_options('track_stickers').get('force_processing', False)
            sticker_2d_tracking_path, was_valid = track_stickers(
                rgb_video_path=rgb_video_path,
                output_dir=handstickers_dir,
                force_processing=force
            )
            if was_valid and not force:
                dag_handler.mark_completed('track_stickers')

        if dag_handler.can_run('generate_xyz_stickers'):
            print(f"[{block_name}] ==> Running task: generate_xyz_stickers")
            options = dag_handler.get_task_options('generate_xyz_stickers')
            force = options.get('force_processing', False)
            sticker_3d_tracking_path = generate_xyz_stickers(
                stickers_2d_path=sticker_2d_tracking_path,
                source_video=config.source_video,
                output_dir=handstickers_dir,
                force_processing=force
            )
            dag_handler.mark_completed('generate_xyz_stickers')

        if dag_handler.can_run('generate_3d_hand_motion'):
            print(f"[{block_name}] ==> Running task: generate_3d_hand_motion")
            force = dag_handler.get_task_options('generate_3d_hand_motion').get('force_processing', False)
            hand_motion_glb_path, hand_metadata_path = generate_3d_hand_motion(
                rgb_video_path=rgb_video_path,
                stickers_xyz_path=sticker_3d_tracking_path,
                hand_models_dir=config.hand_models_dir,
                output_dir=config.video_processed_output_dir,
                force_processing=force
            )
            dag_handler.mark_completed('generate_3d_hand_motion')

        if dag_handler.can_run('generate_somatosensory_chars'):
            print(f"[{block_name}] ==> Running task: generate_somatosensory_chars")
            options = dag_handler.get_task_options('generate_somatosensory_chars')
            force = options.get('force_processing', False)
            use_monitor = options.get('monitor', False)

            somatosensory_chars_path = generate_somatosensory_chars(
                hand_motion_glb_path=hand_motion_glb_path,
                hand_metadata_path=hand_metadata_path,
                session_processed_dir=config.session_processed_output_dir,
                session_id=config.session_id,
                current_video_filename=rgb_video_path.name,
                output_dir=config.video_processed_output_dir,
                monitor=use_monitor,
                force_processing=force
            )
            dag_handler.mark_completed('generate_somatosensory_chars')
    except Exception as e:
        print(f"❌ Pipeline failed during Stage 4: 3D Reconstruction. Error: {e}")
        return {"status": "failed", "stage": 4, "error": str(e)}

    # --- Stage 5: Final data integration ---
    try:
        if dag_handler.can_run('unify_dataset'):
            print(f"[{block_name}] ==> Running task: unify_dataset")
            force = dag_handler.get_task_options('unify_dataset').get('force_processing', False)
            unify_dataset(
                contact_chars_path=somatosensory_chars_path,
                ttl_path=ttl_signal_path,
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
    print("🛠️  Setting up files for processing...")

    project_data_root = path_tools.get_project_data_root()

    configs_dir = Path("configs")
    dag_config_path = Path(configs_dir / "kinect_automatic_pipeline_dag.yaml")

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