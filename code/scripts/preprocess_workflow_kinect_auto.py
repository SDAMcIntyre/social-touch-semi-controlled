import os
import logging
from pathlib import Path
from datetime import datetime
import shutil
import time
import traceback
from multiprocessing import Queue, freeze_support

from prefect import flow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import utils.path_tools as path_tools
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
    generate_led_roi,
    track_led_states_changes,
    validate_and_correct_led_timing_from_stimuli
)

from _3_preprocessing._6_metadata_matching import (
    calculate_trial_id,
    generate_stimuli_metadata_to_data,
    find_single_touches,
)

from _3_preprocessing._7_unification import (
    unify_datasets          
)

# -----------------------------------------------------------------------------
# PREPROCESSING TASKS (Wrapped Flows)
# -----------------------------------------------------------------------------

@flow(name="3. Track LED Blinking")
def track_led_blinking(video_path: Path, stimulus_metadata: Path, output_dir: Path, *, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Tracking LED blinking...")
    name_baseline = video_path.stem + "_LED"
    roi_metadata_path = output_dir / (name_baseline + "_roi_metadata.json")
    roi_video_path = output_dir / (name_baseline + "_roi.mp4")
    generate_led_roi(video_path, roi_metadata_path, roi_video_path, force_processing=force_processing)
    
    csv_led_path = output_dir / (name_baseline + ".csv")
    metadata_led_state_path = output_dir / (name_baseline + "_metadata.json")
    track_led_states_changes(roi_video_path, csv_led_path, metadata_led_state_path, force_processing=force_processing)

    csv_led_path_corrected = output_dir / (name_baseline + "_corrected.csv")
    validate_and_correct_led_timing_from_stimuli(csv_led_path, stimulus_metadata, csv_led_path_corrected, force_processing=force_processing)
    return csv_led_path_corrected

@flow(name="5. Validate Forearm Extraction")
def validate_forearm_extraction(session_output_dir: Path) -> Path:
    print(f"[{session_output_dir.name}] Validating forearm extraction...")
    is_valid = is_forearm_valid(session_output_dir / "forearm_pointclouds", verbose=True)
    if not is_valid:
        raise ValueError("Forearm needs to be manually extracted first.")
    return is_valid

@flow(name="5. Validate Hand Extraction")
def validate_hand_extraction(rgb_video_path: Path, hand_models_dir: Path, expected_labels: list, output_dir: Path) -> Path:
    print(f"[{output_dir.name}] Validating hand extraction...")
    name_baseline = rgb_video_path.stem + "_handmodel"
    metadata_path = output_dir / (name_baseline + "_metadata.json")
    is_valid, errors = is_hand_model_valid(metadata_path, hand_models_dir, expected_labels, verbose=True)
    if not is_valid:
        raise ValueError("Hand model needs to be manually extracted first.")
    return is_valid

@flow(name="6. Track Stickers (2D)")
def track_stickers(rgb_video_path: Path, output_dir: Path, *, force_processing: bool = False) -> tuple[Path | None, bool]:
    print(f"[{output_dir.name}] Tracking stickers (2D)...")
    name_baseline = rgb_video_path.stem + "_handstickers"
    metadata_roi_path = output_dir / (name_baseline + "_roi_metadata.json")
    stickers_roi_csv_path = output_dir / (name_baseline + "_roi_tracking.csv")

    track_objects_in_video(rgb_video_path, metadata_roi_path, output_path=stickers_roi_csv_path, force_processing=force_processing)
    
    if not is_2d_stickers_tracking_valid(metadata_roi_path):
        raise ValueError("❌ --> 2D sticker tracking has not been manually validated.")
    
    roi_unified_csv_path = output_dir / (name_baseline + "_roi_standard_size.csv")
    generate_standard_roi_size_dataset(stickers_roi_csv_path, roi_unified_csv_path)
    corrmap_video_base_path = output_dir / (name_baseline + "_roi_unified.mp4")
    create_standardized_roi_videos(roi_unified_csv_path, rgb_video_path, corrmap_video_base_path, force_processing=force_processing)
    
    metadata_colorspace_path = output_dir / (name_baseline + "_colorspace_metadata.json")
    binary_video_base_path = output_dir / (name_baseline + "_corrmap.mp4")
    create_color_correlation_videos(corrmap_video_base_path, metadata_colorspace_path, binary_video_base_path, force_processing=force_processing)
    
    if not is_correlation_videos_threshold_defined(metadata_colorspace_path):
        raise ValueError("❌ --> correlation videos threshold has not been manually validated.")
    
    fit_ellipses_path = output_dir / (name_baseline + "_ellipses.csv")
    fit_ellipses_on_correlation_videos(binary_video_base_path, metadata_colorspace_path, fit_ellipses_path, force_processing=force_processing)
    
    adj_ellipses_path = output_dir / (name_baseline + "_ellipses_center_adjusted.csv")
    adjust_ellipse_centers_to_global_frame(roi_unified_csv_path, fit_ellipses_path, adj_ellipses_path, force_processing=force_processing)
    
    final_csv_path = output_dir / (name_baseline + "_summary_2d_coordinates.csv")
    consolidate_2d_tracking_data(roi_unified_csv_path, adj_ellipses_path, final_csv_path, score_threshold=0.7, force_processing=force_processing)
    return final_csv_path, True

@flow(name="6. Generate XYZ Sticker Positions (3D)")
def generate_xyz_stickers(stickers_2d_path: Path, source_video: Path, output_dir: Path, *, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Generating XYZ sticker positions (3D)...")
    name_baseline = stickers_2d_path.stem.replace('_summary_2d_coordinates', '')
    result_csv_path = output_dir / (name_baseline + "_xyz_tracked.csv")
    result_md_path = output_dir / (name_baseline + "_xyz_tracked_metadata.json")
    extract_stickers_xyz_positions(source_video, stickers_2d_path, "centroid", result_csv_path, result_md_path, force_processing=force_processing)
    return result_csv_path

@flow(name="7. Generate 3D Hand Position")
def generate_3d_hand_motion(rgb_video_path: Path, stickers_xyz_path: Path, hand_models_dir: Path, output_dir: Path, *, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Generating 3D hand motion...")
    name_baseline = rgb_video_path.stem + "_handmodel"
    metadata_path = output_dir / (name_baseline + "_metadata.json")
    hand_motion_glb_path = output_dir / (name_baseline + "_motion.glb")
    hand_motion_csv_path = output_dir / (name_baseline + "_motion.csv")
    generate_hand_motion(stickers_xyz_path, hand_models_dir, metadata_path, hand_motion_glb_path, hand_motion_csv_path, force_processing=force_processing)
    return hand_motion_glb_path, metadata_path

@flow(name="8. Generate Somatosensory Characteristics")
def generate_somatosensory_chars(hand_motion_glb_path: Path, hand_metadata_path: Path, session_processed_dir: Path, session_id: str, current_video_filename: str, output_dir: Path, *, monitor: bool = False, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Generating somatosensory characteristics...")
    name_baseline = Path(current_video_filename).stem
    forearm_pointcloud_dir = session_processed_dir / "forearm_pointclouds"
    metadata_filaname = session_id + "_arm_roi_metadata.json"
    metadata_path = forearm_pointcloud_dir / metadata_filaname
    contact_characteristics_path = output_dir / (name_baseline + "_contact_and_kinematic_data.csv")
    compute_somatosensory_characteristics(hand_motion_glb_path, hand_metadata_path, metadata_path, forearm_pointcloud_dir, current_video_filename, contact_characteristics_path, monitor=monitor, force_processing=force_processing)
    return contact_characteristics_path

@flow(name="10. Define Trial IDs")
def define_trial_ids_flow(rgb_video_path: Path, output_dir: Path, *, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Defining Trial IDs (Separate File)...")
    name_baseline = Path(rgb_video_path).stem
    input_chunk_path = output_dir / (name_baseline + "_trial-chunks.csv")
    final_path = output_dir / (name_baseline + "_trial-ids.csv")
    calculate_trial_id(trial_chunk_path=input_chunk_path, output_path=final_path, force_processing=force_processing)
    return final_path

@flow(name="11. Find Single Touches")
def find_single_touches_flow(trial_data_path: Path, stickers_xyz_path: Path, stimuli_metadata_path: Path, output_dir: Path, *, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Finding Single Touches...")
    name_baseline = Path(trial_data_path).stem.replace("_trial_ids_only", "")
    final_path = output_dir / (name_baseline + "_single-touches-auto.csv")
    find_single_touches(stickers_xyz_path=stickers_xyz_path, 
                        trial_data_path=trial_data_path, 
                        stimuli_metadata_path=stimuli_metadata_path, 
                        output_path=final_path, force_processing=force_processing)
    return final_path
    
@flow(name="12. Add Stimuli Metadata")
def generate_stimuli_metadata_flow(trial_data_path: Path, stimulus_metadata: Path, output_dir: Path, *, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Adding Stimuli Metadata...")
    name_baseline = Path(trial_data_path).stem.replace("_trial_ids_only", "")
    final_path = output_dir / (name_baseline + "_with-stimuli-data.csv")
    generate_stimuli_metadata_to_data(trial_id_path=trial_data_path, stimuli_path=stimulus_metadata, output_path=final_path, force_processing=force_processing)
    return final_path

@flow(name="13. Unify Processed Data")
def unify_processed_data_flow(led_path: Path, 
                              contact_path: Path, 
                              trial_path: Path, 
                              single_touch_path: Path, 
                              stimuli_path: Path, 
                              rgb_video_path: Path,
                              output_dir: Path, *, force_processing: bool = False) -> Path:
    print(f"[{output_dir.name}] Unifying all processed data...")
    # Note: trial_path here now refers to the separate CSV file
    final_path = output_dir / (Path(rgb_video_path).stem + "_unified.csv")
    unify_datasets(led_path=led_path, contact_path=contact_path, trial_path=trial_path, single_touch_path=single_touch_path, stimuli_path=stimuli_path, output_path=final_path, force_processing=force_processing)
    return final_path

# -----------------------------------------------------------------------------
# EXECUTION LOGIC
# -----------------------------------------------------------------------------

def run_preprocessing_pipeline_session(config: KinectConfig, dag_handler: DagConfigHandler, monitor_queue: Queue = None, report_file_path: Path = None):
    block_name = config.source_video.stem
    print(f"🚀 Starting PREPROCESSING for: {block_name}")

    if monitor_queue is not None:
        monitor = PipelineMonitor(report_path=report_file_path, stages=list(dag_handler.tasks.keys()), data_queue=monitor_queue)
    else:
        monitor = None
    
    # Context Reconstruction
    # Since we are in a separate script, we must re-derive the RGB video path that 
    # should have been created by the Primary pipeline.
    rgb_video_path = config.video_primary_output_dir / f"{block_name}.mp4"
    
    if not rgb_video_path.exists():
        msg = f"Missing RGB Video at {rgb_video_path}. Did the Primary Pipeline run?"
        print(f"❌ {msg}")
        if monitor: monitor.update(block_name, "Startup", "FAILURE", msg)
        return {"status": "failed", "error": msg}

    context = {"rgb_video_path": rgb_video_path}

    pipeline_stages = [
        {"name": "track_led_blinking", 
         "func": track_led_blinking, 
         "params": lambda: {"video_path": context.get("rgb_video_path"), 
                            "stimulus_metadata": config.stimulus_metadata, 
                            "output_dir": config.video_processed_output_dir / "temporal_segmentation/LED"}, 
         "outputs": ["led_tracking_path"]},

        {"name": "validate_forearm_extraction", 
         "func": validate_forearm_extraction, 
         "params": lambda: {"session_output_dir": config.session_processed_output_dir}},

        {"name": "validate_hand_extraction", 
         "func": validate_hand_extraction, 
         "params": lambda: {"rgb_video_path": context.get("rgb_video_path"), 
                            "hand_models_dir": config.hand_models_dir,
                            "expected_labels": config.objects_to_track, 
                            "output_dir": config.video_processed_output_dir/ "kinematics_analysis"}},

        {"name": "track_stickers", 
         "func": track_stickers, 
         "params": lambda: {"rgb_video_path": context.get("rgb_video_path"), 
                            "output_dir": config.video_processed_output_dir / "handstickers"}, 
         "outputs": ["sticker_2d_tracking_path", None]}, 

        {"name": "generate_xyz_stickers", 
         "func": generate_xyz_stickers, 
         "params": lambda: {"stickers_2d_path": context.get("sticker_2d_tracking_path"), 
                            "source_video": config.source_video, 
                            "output_dir": config.video_processed_output_dir / "handstickers"}, 
         "outputs": ["sticker_3d_tracking_path"]},

        {"name": "generate_3d_hand_motion", 
         "func": generate_3d_hand_motion, 
         "params": lambda: {"rgb_video_path": context.get("rgb_video_path"), 
                            "stickers_xyz_path": context.get("sticker_3d_tracking_path"), 
                            "hand_models_dir": config.hand_models_dir, 
                            "output_dir": config.video_processed_output_dir / "kinematics_analysis"}, 
         "outputs": ["hand_motion_glb_path", "hand_metadata_path"]},

        {"name": "generate_somatosensory_chars", 
         "func": generate_somatosensory_chars, 
         "params": lambda: {"hand_motion_glb_path": context.get("hand_motion_glb_path"), 
                            "hand_metadata_path": context.get("hand_metadata_path"), 
                            "session_processed_dir": config.session_processed_output_dir, 
                            "session_id": config.session_id, 
                            "current_video_filename": context.get("rgb_video_path").name, 
                            "output_dir": config.video_processed_output_dir / "kinematics_analysis"},
         "outputs": ["somatosensory_chars_path"]},

        {"name": "define_trial_id", 
         "func": define_trial_ids_flow,
         "params": lambda: {"rgb_video_path": context.get("rgb_video_path"), 
                            "output_dir": config.video_processed_output_dir / "temporal_segmentation"},
         "outputs": ["trial_data_path"]},

        {"name": "generate_stimuli_metadata", 
         "func": generate_stimuli_metadata_flow, 
         "params": lambda: {"trial_data_path": context.get("trial_data_path"), 
                            "stimulus_metadata": config.stimulus_metadata, 
                            "output_dir": config.video_processed_output_dir / "temporal_segmentation"},
         "outputs": ["stimuli_metadata_path"]},

        {"name": "find_single_touches", 
         "func": find_single_touches_flow,
         "params": lambda: {"stickers_xyz_path": context.get("sticker_3d_tracking_path"), 
                            "trial_data_path": context.get("trial_data_path"),
                            "stimuli_metadata_path": context.get("stimuli_metadata_path"),
                            "output_dir": config.video_processed_output_dir / "temporal_segmentation"},
         "outputs": ["single_touches_path"]},

        {"name": "unify_processed_data", 
         "func": unify_processed_data_flow,
         "params": lambda: {"led_path": context.get("led_tracking_path"),
                            "contact_path": context.get("somatosensory_chars_path"),
                            "trial_path": context.get("trial_data_path"),
                            "single_touch_path": context.get("single_touches_path"),
                            "stimuli_path": context.get("stimuli_metadata_path"),
                            "rgb_video_path": context.get("rgb_video_path"), 
                            "output_dir": config.video_processed_output_dir},
         "outputs": ["unified_dataset_path"]}, 
    ]

    for stage in pipeline_stages:
        task_name = stage["name"]
        executor = TaskExecutor(task_name, block_name, dag_handler, monitor)

        with executor:
            if not executor.can_run: continue
            
            options = dag_handler.get_task_options(task_name)
            params = stage["params"]()
            if 'force_processing' in options:
                params['force_processing'] = options['force_processing']

            result = stage["func"](**params)

            if "outputs" in stage:
                outputs = stage["outputs"]
                if not isinstance(result, tuple): result = (result,)
                for i, key in enumerate(outputs):
                    if key: context[key] = result[i]

        if executor.error_msg:
            print(f"🛑 Failure in {task_name}. Aborting session.")
            return {"status": "failed", "error": executor.error_msg}

    print(f"✅ Preprocessing pipeline finished: {block_name}")
    return {"status": "success"}

def run_batch_preprocessing(kinect_configs_dir, project_data_root, dag_config_path, monitor_queue, report_file_path, parallel):
    dag_template = DagConfigHandler(dag_config_path)
    block_files = get_block_files(kinect_configs_dir)
    submitted_runs = []
    
    for block_file in block_files:
        config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
        validated_config = KinectConfig(config_data=config_data, database_path=project_data_root)
        dag_instance = dag_template.copy()

        if parallel:
            run = run_preprocessing_pipeline_session.submit(validated_config, dag_instance, monitor_queue, report_file_path)
            submitted_runs.append(run)
        else:
            run_preprocessing_pipeline_session(validated_config, dag_instance, monitor_queue, report_file_path)

    if parallel:
        for run in submitted_runs: run.wait()

def setup_environment():
    project_data_root = path_tools.get_project_data_root()
    configs_dir = Path("configs")
    dag_config_path = Path(configs_dir / "preprocess_workflow_kinect_auto_dag.yaml") # Pointing to preprocessing yaml
    
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_file_path = reports_dir / f"{timestamp}_PREPROCESSING_status.xlsx"
    return project_data_root, configs_dir, dag_config_path, report_file_path

def main():
    freeze_support()
    project_data_root, configs_dir, dag_config_path, report_file_path = setup_environment()

    if not dag_config_path.exists():
         print(f"❌ Config {dag_config_path} missing.")
         exit(1)

    main_dag_handler = DagConfigHandler(dag_config_path)
    is_parallel = main_dag_handler.get_parameter('parallel_execution', False)
    kinect_configs_dir = configs_dir / main_dag_handler.get_parameter('kinect_configs_directory')

    main_monitor = PipelineMonitor(str(report_file_path), list(main_dag_handler.tasks.keys()), live_plotting=True)
    main_monitor.show_dashboard()

    run_batch_preprocessing(kinect_configs_dir, project_data_root, dag_config_path, main_monitor.queue, report_file_path, is_parallel)
    
    time.sleep(5)
    main_monitor.close_dashboard(block=True)

if __name__ == "__main__":
    main()