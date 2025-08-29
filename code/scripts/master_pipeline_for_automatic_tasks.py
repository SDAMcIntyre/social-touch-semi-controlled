import os
import yaml
from pathlib import Path

from prefect import flow
from pydantic import BaseModel, DirectoryPath, FilePath, Field

from _2_primary_processing._2_generate_rgb_depth_video import (
    generate_mkv_stream_analysis,
    extract_depth_to_tiff,
    extract_color_to_mp4
)

from _3_preprocessing._1_sticker_tracking import (
    track_objects_in_video,
    ensure_tracking_is_valid,
    extract_stickers_xyz_positions
)

from _3_preprocessing._2_hand_tracking import (
    generate_hand_motion
)

from _3_preprocessing._3_forearm_extraction import (
    extract_forearm
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

import utils.path_tools as path_tools




# --- CHOOSE YOUR EXECUTION MODE HERE ---
# ------------------------------------------------------
# ---------------------------------------
# ---------------------
# -----------
PARALLEL_EXECUTION = False
# -----------
# ---------------------
# ---------------------------------------
# ------------------------------------------------------



# --- 1. Configuration Models (Unchanged) ---
# These models define the validated structure for our YAML configuration files.

class ProcessingParams(BaseModel):
    """A model for algorithm-specific parameters."""
    sticker_confidence_threshold: float = Field(
        default=0.95, description="Confidence threshold for sticker tracking."
    )
    led_brightness_min: int = Field(
        default=200, description="Minimum brightness to register an LED blink."
    )
    num_trials_to_process: int = Field(
        default=3, description="Number of trial files to generate."
    )

class SessionInputs(BaseModel):
    """The complete, validated configuration for a single session."""
    source_video: FilePath
    stimulus_metadata: FilePath
    hand_models_dir: DirectoryPath
    video_primary_output_dir: Path
    video_processed_output_dir: Path

def get_session_files(configs_kinect_dir: Path):
    """Helper to find and validate session configuration files."""
    if not configs_kinect_dir.is_dir():
        raise ValueError(f"Sessions folder not found: {configs_kinect_dir}")
    files = list(configs_kinect_dir.glob("*.yaml"))
    print(f"Found {len(files)} session(s) to process.")
    return files


# --- 3. Sub-Flows (Formerly Tasks) ---
# Each major processing step is now a self-contained, runnable flow.


@flow(name="0. Analyse MKV video")
def validate_mkv_video(source_video: Path, output_dir: Path) -> Path:
    print(f"[{output_dir.name}] Analysing MKV video...")
    analysis_csv_path = output_dir / "mkv_analysis_report.csv"
    return generate_mkv_stream_analysis(source_video, analysis_csv_path)

@flow(name="1. Generate RGB Video")
def generate_rgb_video(source_video: Path, output_dir: Path) -> Path:
    """Requires .mo4 video, generates an .mkv video."""
    print(f"[{output_dir.name}] Generating RGB video...")
    base_filename = os.path.splitext(os.path.basename(source_video))[0]

    rgb_path = output_dir / f"{base_filename}.mp4"
    rgb_video_path = extract_color_to_mp4(source_video, rgb_path)
    if not rgb_video_path is Path:
        rgb_video_path = Path(rgb_video_path)

    return rgb_video_path

@flow(name="2. Generate Depth Images")
def generate_depth_images(source_video: Path, output_dir: Path) -> Path:
    """Requires .mkv video, generates a folder of .tiff images."""
    print(f"[{output_dir.name}] Generating depth images...")
    depth_dir = output_dir / source_video.name.replace(".mkv", "_depth")
    extract_depth_to_tiff(source_video, depth_dir)
    return depth_dir

@flow(name="3. Track LED Blinking")
def track_led_blinking(
    video_path: Path, 
    stimulus_metadata: Path, 
    output_dir: Path
) -> Path:
    """Requires .mp4 video, generates a csv file of LED blinking."""
    
    print(f"[{output_dir.name}] Tracking LED blinking...")
    name_baseline = video_path.stem + "_LED"

    metadata_path = output_dir / (name_baseline + "_roi_metadata.txt")
    define_led_roi(video_path, metadata_path)
    
    video_led_path = output_dir / (name_baseline + "_roi.mp4")
    generate_LED_roi_video(video_path, metadata_path, video_led_path)

    csv_led_path = output_dir / (name_baseline + ".csv")
    metadata_led_state_path = output_dir / (name_baseline + "_metadata.txt")
    track_led_states_changes(video_led_path, csv_led_path, metadata_led_state_path)
    
    csv_led_path_corrected = output_dir / (name_baseline + "_corrected.csv")
    validate_and_correct_led_timing_from_stimuli(csv_led_path, stimulus_metadata, csv_led_path_corrected)
    
    return csv_led_path_corrected

@flow(name="4. Track Stickers")
def track_stickers(rgb_video_path: Path,
                   source_video: Path,
                   output_dir: Path,
                   *,
                   monitor_ui: bool = False
) -> Path:
    """Requires .mp4 video, generates a csv file of stickers position."""
    success = True

    print(f"[{output_dir.name}] Tracking stickers...")
    name_baseline =  rgb_video_path.stem + "_handstickers"

    # defined by manual tasks pipeline
    metadata_roi_path = output_dir / (name_baseline + "_roi_metadata.json")
    # --------------------------------

    stickers_roi_csv_path = output_dir / (name_baseline + "_roi_tracking.csv")
    track_objects_in_video(rgb_video_path, metadata_roi_path, output_path=stickers_roi_csv_path)
    
    # defined by manual tasks pipeline
    try:
        ensure_tracking_is_valid(metadata_roi_path)
    except Exception as e:
        success = False
        return stickers_roi_csv_path, success
    
    result_csv_path = output_dir / (name_baseline + "_xyz_tracked.csv")
    result_video_path = output_dir / (name_baseline + "_xyz_tracked.mp4")
    result_md_path = output_dir / (name_baseline + "_xyz_tracked_metadata.json")
    extract_stickers_xyz_positions(
        source_video, 
        stickers_roi_csv_path,
        
        result_csv_path,
        metadata_path=result_md_path,
        video_path=result_video_path,
        monitor=True)
    
    return result_csv_path
    

@flow(name="5. Generate TTL Signal")
def generate_ttl_signal(led_tracking_path: Path, output_dir: Path) -> Path:
    """Requires the csv of LED blinking, generates a TTL csv file."""
    print(f"[{output_dir.name}] Generating TTL signal...")
    ttl_path = output_dir / "ttl_signal.csv"
    
    return ttl_path


@flow(name="6. Generate Forearm Point Cloud")
def generate_forearm_pointcloud(source_video: Path, output_dir: Path) -> Path:
    """Generates a 3D point cloud .ply file."""
    print(f"[{output_dir.name}] Generating forearm 3D point cloud...")
    metadata_path = source_video.parent.parent / (source_video.name.split("_semicontrolled")[0] + "_kinect_arm_roi_metadata.txt")
    forearm_ply_path = output_dir / "forearm_pointcloud.ply"
    extract_forearm(source_video, metadata_path, forearm_ply_path)

    forearm_ply_with_normals_path = output_dir / "forearm_pointcloud_with_normals.ply"
    if not os.path.exists(forearm_ply_with_normals_path):
        raise ValueError(f"Assessing the normals of forearm is necessary before continuing, please run manual_pipeline.py ({forearm_ply_path})")

    return forearm_ply_with_normals_path


@flow(name="7. Generate 3D Hand Position")
def generate_3d_hand_motion(
    rgb_video_path: Path,
    stickers_path: Path, 
    hand_models_dir: Path, 
    output_dir: Path
) -> Path:
    """Generates 3D position over time of the 3D hand model."""
    print(f"[{output_dir.name}] Generating 3D hand position...")
    name_baseline = rgb_video_path.stem + "_handmodel"
    
    metadata_path = output_dir / (name_baseline + "_metadata.json")
    hand_motion_glb_path = output_dir / (name_baseline + "_motion.glb")
    hand_motion_csv_path = output_dir / (name_baseline + "_motion.csv")

    generate_hand_motion(stickers_path, hand_models_dir, metadata_path, 
                         hand_motion_glb_path, 
                         hand_motion_csv_path)
    return hand_motion_glb_path

@flow(name="8. Generate Somatosensory Characteristics")
def generate_somatosensory_chars(hand_motion_glb_path: Path, forearm_ply_path: Path, output_dir: Path) -> Path:
    """Generates a csv file of contact characteristics."""
    print(f"[{output_dir.name}] Generating somatosensory characteristics...")
    contact_characteristics_path = output_dir / "contact_characteristics.csv"
    
    compute_somatosensory_characteristics(
        hand_motion_glb_path, 
        forearm_ply_path, 
        contact_characteristics_path)

    return contact_characteristics_path

@flow(name="9. Unify Dataset")
def unify_dataset(
    contact_chars_path: Path, ttl_path: Path, output_dir: Path
) -> Path:
    """Generates a csv file of unified data."""
    print(f"[{output_dir.name}] Generating unified dataset...")
    unified_path = output_dir / "unified_dataset.csv"
    unified_path.touch()
    return unified_path

# --- 4. The "Worker" Flow ---
# This flow orchestrates the 11 sub-flows for a single session.
@flow(name="Run Single Session Pipeline")
def run_single_session_pipeline(inputs: SessionInputs, *, monitor_ui: bool = False):
    """This flow processes a SINGLE dataset by calling the appropriate sub-flows."""
    print(f"🚀 Starting pipeline for session: {inputs.video_processed_output_dir.name}")

    # Stage 1: Primary processing
    validate_mkv_video(inputs.source_video, inputs.video_primary_output_dir)
    rgb_video_path = generate_rgb_video(inputs.source_video, inputs.video_primary_output_dir)
    
    # ignore for now as it generates 30GB of data per 5GB mkv file.
    # depth_images_path = generate_depth_images(inputs.source_video, inputs.video_primary_output_dir)
        
    # Stage 2&3: Stickers Tracking and 3D Reconstruction
    handstickers_dir = inputs.video_processed_output_dir / "handstickers"
    sticker_tracking, success = track_stickers(rgb_video_path, inputs.source_video, handstickers_dir, monitor_ui=monitor_ui)
    
    if not success:
        return  {"status": "stickers", "result": None}   

    hand_motion_glb_path = generate_3d_hand_motion(rgb_video_path, sticker_tracking, inputs.hand_models_dir, inputs.video_processed_output_dir)
    forearm_pointcloud = generate_forearm_pointcloud(inputs.source_video, inputs.video_processed_output_dir)
    somatosensory_chars = generate_somatosensory_chars(hand_motion_glb_path, forearm_pointcloud, inputs.video_processed_output_dir)
    
    return  {"status": "somatosensory", "result": None}

    # Stage 4: Generate TTL from LED
    led_dir = inputs.video_processed_output_dir / "LED"
    led_TTL = track_led_blinking(rgb_video_path, inputs.stimulus_metadata, led_dir)
    ttl_signal = generate_ttl_signal(led_TTL, inputs.video_processed_output_dir)

    # Stage 5: Data Integration
    unified_data = unify_dataset(somatosensory_chars, ttl_signal, inputs.video_processed_output_dir)
    
    print(f"✅ Pipeline finished for session: {inputs.video_processed_output_dir.name}")
    
    return {"status": "success", "result": None}

# --- 5. The "Dispatcher" Flow ---
# This top-level flow discovers and launches runs of the worker flow.

@flow(name="Batch Process All Sessions", log_prints=True)
def run_batch_in_parallel(configs_kinect_dir: Path, project_data_root: Path):
    """Finds all session YAML files and triggers a pipeline run for each one."""

    if not configs_kinect_dir.is_dir():
        raise ValueError(f"Sessions folder not found: {configs_kinect_dir}")

    session_files = list(configs_kinect_dir.glob("*.yaml"))
    print(f"Found {len(session_files)} session(s) to process in '{configs_kinect_dir}'.")

    # A list to hold our submitted flow runs (futures)
    submitted_runs = []

    for session_file in session_files:
        print(f"Dispatching run for {session_file.name}...")
        with open(session_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config_data_abs = {key: project_data_root / value for key, value in config_data.items()}
        validated_inputs = SessionInputs(**config_data_abs)

        run = run_single_session_pipeline.with_options(
            flow_run_name=f"session-{validated_inputs.video_processed_output_dir.parent.name}/{validated_inputs.video_processed_output_dir.name}"
        )(inputs=validated_inputs)
        
        # Optionally, collect the run objects if you need to wait for them later
        submitted_runs.append(run)

    # If you want the main dispatcher flow to wait until all sub-flows are
    # finished, you can iterate through the results. Accessing the .result()
    # of a future will block until it's available.
    print(f"All {len(submitted_runs)} session flows have been submitted. Waiting for completion...")
    
    # for run in submitted_runs:
    #     run.result() # This line is optional. Add it if the parent flow should only finish after all children are done.

    print("Dispatcher has finished submitting all jobs.")
    return {"status": "success", "result": None}

# --- 5. The "Dispatcher" Flow ---
@flow(name="Run Batch Sequentially", log_prints=True)
def run_batch_sequentially(configs_kinect_dir: Path,
                           project_data_root: Path,
                           *,
                           monitor_ui: bool = False):
    """
    Runs all session pipelines one by one, waiting for each to complete.
    """
    session_files = get_session_files(configs_kinect_dir)
    results = []
    for session_file in session_files:
        print(f"Running session: {session_file.name}")
        with open(session_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config_data_abs = {key: project_data_root / value for key, value in config_data.items()}
        validated_inputs = SessionInputs(**config_data_abs)

        # In Prefect 3, this call is BLOCKING. It runs the subflow to completion.
        result = run_single_session_pipeline(validated_inputs, monitor_ui=monitor_ui)
        results.append(result)
        print(f"Completed session: {session_file.name} | Status: {result['status']}")
    
    print("✅ All sequential runs have completed.")


# --- 6. Main execution block ---
if __name__ == "__main__":
    print("🛠️  Setting up files for processing...")
    
    configs_dir = Path("configs")
    kinect_dir = Path("kinect_configs")

    configs_kinect_dir = configs_dir / kinect_dir
    project_data_root = path_tools.get_project_data_root()

    if PARALLEL_EXECUTION:
        print("🚀 Launching batch processing in PARALLEL.")
        # We await the async parallel flow
        run_batch_in_parallel(
            configs_kinect_dir=configs_kinect_dir, 
            project_data_root=project_data_root
        )
    else:
        print("🚀 Launching batch processing SEQUENTIALLY.")
        # We can also await the sync flow when calling from an async function
        run_batch_sequentially(
            configs_kinect_dir=configs_kinect_dir, 
            project_data_root=project_data_root
        )

