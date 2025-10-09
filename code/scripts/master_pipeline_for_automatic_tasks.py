import os
import logging
from pathlib import Path
from datetime import datetime
import shutil
import time
import traceback
from multiprocessing import Queue, freeze_support

from prefect import flow, get_run_logger

# Setup a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import utils.path_tools as path_tools
from utils.pipeline_config_manager import DagConfigHandler
from utils.pipeline_monitoring.pipeline_monitor import PipelineMonitor

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
    generate_led_roi,
    track_led_states_changes,
    validate_and_correct_led_timing_from_stimuli
)

from _3_preprocessing._6_unification import (
    unify_contact_caracteristics_and_ttl
)


class TaskExecutor:
    """A context manager to handle the boilerplate of running a pipeline task."""
    def __init__(self, task_name, block_name, dag_handler, monitor):
        self.task_name: str = task_name
        self.block_name: str = block_name
        self.dag_handler: DagConfigHandler = dag_handler
        self.monitor: PipelineMonitor = monitor
        self.can_run: bool = False
        self.error_msg: str = None

    def __enter__(self):
        """Called when entering the 'with' block. Prepares and starts the task."""
        if self.dag_handler.can_run(self.task_name):
            self.can_run = True
            print(f"[{self.block_name}] ==> Running task: {self.task_name}")
            if self.monitor is not None:
                self.monitor.update(self.block_name, self.task_name, "RUNNING")
        return self # Returns the executor instance itself

    def __exit__(self, exc_type, exc_value, tb):
        """Called when exiting the 'with' block. Handles success or failure."""
        if not self.can_run:
            return # Task was skipped by the DAG handler

        if exc_type: # An exception occurred
            self.error_msg = f"Task '{self.task_name}' failed: {exc_value}"
            print(f"❌ {self.error_msg}\n{traceback.format_exc()}")
            if self.monitor is not None:
                self.monitor.update(self.block_name, self.task_name, "FAILURE", self.error_msg)
            # Suppress the exception to allow the main loop to handle failure
            return True
        else: # Success
            self.dag_handler.mark_completed(self.task_name)
            if self.monitor is not None:
                self.monitor.update(self.block_name, self.task_name, "SUCCESS")
        return False

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
    
    roi_metadata_path = output_dir / (name_baseline + "_roi_metadata.json")
    roi_video_path = output_dir / (name_baseline + "_roi.mp4")
    success = generate_led_roi(video_path, roi_metadata_path, roi_video_path, force_processing=force_processing)
    
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
    return is_valid

@flow(name="5. Validate Hand Extraction")
def validate_hand_extraction(rgb_video_path: Path, hand_models_dir: Path, expected_labels: list, output_dir: Path) -> Path:
    print(f"[{output_dir.name}] Validating hand extraction...")
    name_baseline = rgb_video_path.stem + "_handmodel"
    metadata_path = output_dir / (name_baseline + "_metadata.json")
    is_valid, errors = is_hand_model_valid(metadata_path, hand_models_dir, expected_labels, verbose=True)
    return is_valid

@flow(name="6. Track Stickers (2D)")
def track_stickers(rgb_video_path: Path, output_dir: Path, *, force_processing: bool = False) -> tuple[Path | None, bool]:
    print(f"[{output_dir.name}] Tracking stickers (2D)...")
    name_baseline = rgb_video_path.stem + "_handstickers"
    metadata_roi_path = output_dir / (name_baseline + "_roi_metadata.json")
    stickers_roi_csv_path = output_dir / (name_baseline + "_roi_tracking.csv")

    track_objects_in_video(rgb_video_path, metadata_roi_path, output_path=stickers_roi_csv_path, force_processing=force_processing)
    
    if not is_2d_stickers_tracking_valid(metadata_roi_path):
        print("❌ --> 2D sticker tracking has not been manually validated. Cannot continue the pipeline.")
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
        print("❌ --> correlation videos threshold has not been manually validated. Cannot continue the pipeline.Execute the corresponding manual task. ")
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
    name_baseline = Path(current_video_filename).stem
    # inputs
    forearm_pointcloud_dir = session_processed_dir / "forearm_pointclouds"
    metadata_filaname = session_id + "_arm_roi_metadata.json"
    metadata_path = forearm_pointcloud_dir / metadata_filaname

    # output
    contact_characteristics_path = output_dir / (name_baseline + "_contact_and_kinematic_data.csv")
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
    name_baseline = Path(contact_chars_path).stem
    unified_path = output_dir / (name_baseline + "_somatosensory_data_withTTL.csv")
    
    unify_contact_caracteristics_and_ttl(contact_chars_path, ttl_path, unified_path)

    return unified_path


# --- 4. The "Worker" Flow ---
# @flow(name="Run Single Session Pipeline")
def run_single_session_pipeline(
    config: KinectConfig,
    dag_handler: DagConfigHandler,
    monitor_queue: Queue = None,
    report_file_path: Path = None
):
    """
    Processes a single dataset ("block") using a data-driven approach
    to eliminate code repetition and improve maintainability.
    """
    block_name = config.source_video.stem
    print(f"🚀 Starting pipeline for block: {block_name}")

    if monitor_queue is not None:
        monitor = PipelineMonitor(
            report_path=report_file_path, stages=list(dag_handler.tasks.keys()), data_queue=monitor_queue
        )
    else:
        monitor = None

    # A dictionary to hold the results from tasks (e.g., file paths)
    context = {}

    # --- Define the entire pipeline as a data structure ---
    pipeline_stages = [
        # --- Stage 1: Primary Video Processing ---
        {"name": "validate_mkv_video", 
         "func": validate_mkv_video, 
         "params": lambda: {"source_video": config.source_video, 
                            "output_dir": config.video_primary_output_dir}},
        {"name": "generate_rgb_video", 
         "func": generate_rgb_video, 
         "params": lambda: {"source_video": config.source_video, 
                            "output_dir": config.video_primary_output_dir}, 
         "outputs": ["rgb_video_path"]},
        {"name": "generate_depth_images", 
         "func": generate_depth_images, 
         "params": lambda: {"source_video": config.source_video, 
                            "output_dir": config.video_primary_output_dir}},

        # --- Stage 2: LED Tracking ---
        {"name": "track_led_blinking", 
         "func": track_led_blinking, 
         "params": lambda: {"video_path": context.get("rgb_video_path"), 
                            "stimulus_metadata": config.stimulus_metadata, 
                            "output_dir": config.video_processed_output_dir / "LED"}, 
         "outputs": ["led_tracking_path"]},

        # --- Stage 3: Validation ---
        {"name": "validate_forearm_extraction", 
         "func": validate_forearm_extraction, 
         "params": lambda: {"session_output_dir": config.session_processed_output_dir}},
        {"name": "validate_hand_extraction", 
         "func": validate_hand_extraction, 
         "params": lambda: {"rgb_video_path": context.get("rgb_video_path"), 
                            "hand_models_dir": config.hand_models_dir,
                            "expected_labels": config.objects_to_track, 
                            "output_dir": config.video_processed_output_dir}},

        # --- Stage 4: 3D Tracking & Reconstruction ---
        {"name": "track_stickers", 
         "func": track_stickers, 
         "params": lambda: {"rgb_video_path": context.get("rgb_video_path"), 
                            "output_dir": config.video_processed_output_dir / "handstickers"}, 
         "outputs": ["sticker_2d_tracking_path", None]}, # Use None for unused return values
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
                            "output_dir": config.video_processed_output_dir}, 
         "outputs": ["hand_motion_glb_path", "hand_metadata_path"]},
        {"name": "generate_somatosensory_chars", 
         "func": generate_somatosensory_chars, 
         "params": lambda: {"hand_motion_glb_path": context.get("hand_motion_glb_path"), 
                            "hand_metadata_path": context.get("hand_metadata_path"), 
                            "session_processed_dir": config.session_processed_output_dir, 
                            "session_id": config.session_id, 
                            "current_video_filename": context.get("rgb_video_path").name, 
                            "output_dir": config.video_processed_output_dir},
         "outputs": ["somatosensory_chars_path"]},

        # --- Stage 5: Final Data Integration ---
        {"name": "unify_dataset", 
         "func": unify_dataset, 
         "params": lambda: {"contact_chars_path": context.get("somatosensory_chars_path"), 
                            "ttl_path": context.get("led_tracking_path"), 
                            "output_dir": config.video_processed_output_dir}},
    ]

    # --- Pipeline Execution Engine ---
    for stage_idx, stage in enumerate(pipeline_stages):
        task_name = stage["name"]
        executor = TaskExecutor(task_name, block_name, dag_handler, monitor)

        with executor:
            if not executor.can_run:
                continue

            # Prepare task arguments
            options = dag_handler.get_task_options(task_name)
            params = stage["params"]()
            force = options.get('force_processing')
            if force is not None:
                params['force_processing'] = force

            # Execute the task
            result = stage["func"](**params)

            # Store outputs in the context dictionary for subsequent tasks
            if "outputs" in stage:
                outputs = stage["outputs"]
                if not isinstance(result, tuple):
                    result = (result,) # Ensure result is always a tuple
                for i, key in enumerate(outputs):
                    if key: # Skip if the key is None
                        context[key] = result[i]

        # If the task failed, stop the pipeline
        if executor.error_msg:
            # The 'mark_remaining_tasks_as_skipped' logic can now be simplified
            all_tasks = list(dag_handler.tasks.keys())
            current_task_index = all_tasks.index(task_name)
            for skipped_task in all_tasks[current_task_index + 1:]:
                monitor.update(block_name, skipped_task, "SKIPPED", "Skipped due to prior failure.")
            return {"status": "failed", "stage": stage_idx, "error": executor.error_msg}

    print(f"✅ Pipeline finished successfully for session: {block_name}")
    return {"status": "success", "completed_tasks": list(dag_handler.completed_tasks)}


# @flow(name="Batch Process All Sessions", log_prints=True)
def run_batch_processing(
    kinect_configs_dir: Path,
    project_data_root: Path,
    dag_config_path: Path,
    monitor_queue: Queue,
    report_file_path: Path,
    parallel: bool,
):
    """
    Dispatches pipeline runs for all session configs found in a directory.

    This single flow handles both parallel and sequential execution based on the
    `parallel` flag, removing code duplication.
    """
    # logger = get_run_logger()
    dag_handler_template = DagConfigHandler(dag_config_path)
    block_files = get_block_files(kinect_configs_dir)
    
    mode = "PARALLEL" if parallel else "SEQUENTIAL"
    logging.info(f"🚀 Starting batch processing for {len(block_files)} sessions in {mode} mode.")

    submitted_runs = []
    for block_file in block_files:
        logging.info(f"Preparing session: {block_file.stem}")
        config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
        validated_config = KinectConfig(config_data=config_data, database_path=project_data_root)
        dag_handler_instance = dag_handler_template.copy()

        if parallel:
            # .submit() creates a new flow run that executes asynchronously
            run = run_single_session_pipeline.submit(
                config=validated_config,
                dag_handler=dag_handler_instance,
                monitor_queue=monitor_queue,
                report_file_path=report_file_path,
                flow_run_name=f"session-{validated_config.session_id}",
            )
            submitted_runs.append(run)
        else:
            # A direct call executes the flow and blocks until completion
            run_single_session_pipeline(
                config=validated_config,
                dag_handler=dag_handler_instance,
                monitor_queue=monitor_queue,
                report_file_path=report_file_path
            )
            logging.info(f"--- Completed session: {block_file.stem} ---")

    # If in parallel mode, robustly wait for all submitted runs to finish
    if parallel:
        logging.info("All flows submitted. Waiting for parallel runs to complete...")
        for i, run in enumerate(submitted_runs):
            run.wait()
            logging.info(f"({i+1}/{len(submitted_runs)}) Completed flow run: {run.name}")
    
    logging.info("✅ All batch processing tasks have finished.")

###
### 2. MODULAR MAIN EXECUTION BLOCK
###
def setup_environment():
    """Handles filesystem setup, cleaning up old reports."""
    project_data_root = path_tools.get_project_data_root()

    configs_dir = Path("configs")
    dag_config_path = Path(configs_dir / "kinect_automatic_pipeline_dag.yaml")

    print("🛠️  Setting up environment...")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_file_path = reports_dir / f"{timestamp}_automatic_pipeline_status.xlsx"
    if report_file_path.exists():
        shutil.rmtree(report_file_path)
        print("🧹 File with the same name found, removing it.")
    return project_data_root, configs_dir, dag_config_path, report_file_path

def main():
    """Main execution function to orchestrate the pipeline."""
    freeze_support() # Essential for multiprocessing on Windows
    project_data_root, configs_dir, dag_config_path, report_file_path = setup_environment()

    # --- Configuration Loading ---
    try:
        main_dag_handler = DagConfigHandler(dag_config_path)
        is_parallel = main_dag_handler.get_parameter('parallel_execution', False)
        kinect_dir_name = main_dag_handler.get_parameter('kinect_configs_directory')
        kinect_configs_dir = configs_dir / kinect_dir_name
    except FileNotFoundError:
        print(f"❌ Error: Configuration file '{dag_config_path}' not found.")
        exit(1)

    # --- Monitoring Setup ---
    print("📊 Initializing pipeline monitor...")
    pipeline_stages = list(main_dag_handler.tasks.keys())
    main_monitor = PipelineMonitor(report_path=str(report_file_path), stages=pipeline_stages, live_plotting=True)
    main_monitor.show_dashboard()

    # --- Flow Execution ---
    # The main logic is now encapsulated in the single dispatcher flow
    run_batch_processing(
        kinect_configs_dir=kinect_configs_dir,
        project_data_root=project_data_root,
        dag_config_path=dag_config_path,
        monitor_queue=main_monitor.queue,
        report_file_path=report_file_path,
        parallel=is_parallel,
    )

    # --- Graceful Shutdown ---
    print("\n🏁 All pipeline tasks have completed.")
    print("✨ Dashboard will close automatically in 10 seconds...")
    time.sleep(10)
    
    main_monitor.close_dashboard(block=True)
    print(f"👋 Processing finished. Final report saved to {report_file_path}")

if __name__ == "__main__":
    main()