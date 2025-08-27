import os
import yaml
from pathlib import Path

from prefect import  flow
from pydantic import BaseModel, DirectoryPath, FilePath


from _3_preprocessing._1_sticker_tracking import (
    review_tracked_objects_in_video,
    ensure_tracking_is_valid,

    define_object_colorspaces_for_video,
    generate_color_correlation_videos,
    define_handstickers_color_threshold,

    review_tracked_handstickers_position,
    save_tracked_handstickers_position_as_video
)

from _3_preprocessing._2_hand_tracking import select_hand_model_characteristics

from _3_preprocessing._3_forearm_extraction import define_normals

import utils.path_tools as path_tools

# --- 1. Configuration Models (Unchanged) ---
# This model define the validated structure for our YAML configuration files.
class SessionInputs(BaseModel):
    """The complete, validated configuration for a single session."""
    source_video: FilePath
    stimulus_metadata: FilePath
    hand_models_dir: DirectoryPath
    video_primary_output_dir: Path
    video_processed_output_dir: Path


@flow(name="3. Track LED Blinking")
def track_led_blinking(video_path: Path, stimulus_metadata: Path, output_dir: Path) -> Path:
    """Requires .mp4 video, generates a csv file of LED blinking."""
    print(f"[{output_dir.name}] Tracking LED blinking...")
    
    return ""

@flow(name="4. Track Stickers")
def track_stickers(
    rgb_video_path: Path,
    source_video: Path, 
    root_output_dir: Path
) -> Path:
    """Requires .mp4 video, generates a csv file of stickers position."""
    output_dir = root_output_dir / "handstickers"

    """Requires .mp4 video, generates a csv file of stickers position."""
    print(f"[{output_dir.name}] Tracking stickers...")
    name_baseline = rgb_video_path.stem + "_handstickers"
    
    metadata_roi_path = output_dir / (name_baseline + "_roi_metadata.json")
    stickers_roi_csv_path = output_dir / (name_baseline + "_roi_tracking.csv")
    review_tracked_objects_in_video(rgb_video_path, metadata_roi_path, stickers_roi_csv_path)
    
    return stickers_roi_csv_path

    ensure_tracking_is_valid(metadata_roi_path)
    
    bypass_ellipse_tracking = True
    if not bypass_ellipse_tracking:
        roi_video_base_path = output_dir / (name_baseline + "_roi_unified.mp4")
        metadata_colorspace_path = output_dir / (name_baseline + "_colorspace_metadata.json")
        define_object_colorspaces_for_video(roi_video_base_path,
                                            src_metadata_path=metadata_roi_path,
                                            dest_metadata_path=metadata_colorspace_path)
        
        
        corrmap_video_base_path = output_dir / (name_baseline + "_corrmap.mp4")
        generate_color_correlation_videos(
            roi_video_base_path, 
            metadata_colorspace_path,
            corrmap_video_base_path,
            force_processing=True
            )

        define_handstickers_color_threshold(corrmap_video_base_path, md_path=metadata_colorspace_path)
        
        roi_unified_csv_path = output_dir / (name_baseline + "_roi_unified.csv")
        ellipses_csv_path = output_dir / (name_baseline + "_ellipses.csv")
        ellipses_video_path = output_dir / (name_baseline + "_ellipses.mp4")
        if not os.path.exists(ellipses_csv_path):
            return ""
        
        review_tracked_handstickers_position(
            rgb_video_path, 
            roi_unified_csv_path, 
            ellipses_csv_path, 
            metadata_colorspace_path,
            output_video_path=ellipses_video_path)
        return ""

        stickers_pos_video_path = output_dir / (name_baseline + "_pos_tracking.mp4")
        save_tracked_handstickers_position_as_video(rgb_video_path, stickers_pos_csv_path, metadata_pos_path, stickers_pos_video_path)
            
        try:
            ensure_tracking_is_valid(metadata_pos_path)
        except ValueError as e:
            return ""
        
    result_csv_path = output_dir / (name_baseline + "_xyz_tracked.csv")
    return result_csv_path


@flow(name="7. Generate 3D Hand Position")
def prepare_hand_tracking_session(
    rgb_video_path: Path,
    hand_models_dir: Path, 
    output_dir: Path
) -> Path:
    """Generates 3D position over time of the 3D hand model."""
    print(f"[{output_dir.name}] Generating 3D hand position...")
    name_baseline = rgb_video_path.stem + "_handmodel"
    metadata_path = output_dir / (name_baseline + "_metadata.json")

    point_labels = ["sticker_yellow", "sticker_blue", "sticker_green"]
    select_hand_model_characteristics(rgb_video_path, hand_models_dir, point_labels, metadata_path)

    return metadata_path
    

@flow(name="8. Prepare forearm data (normals)")
def prepare_forearm(
    point_cloud_path: Path,
    output_dir: Path
) -> Path:
    """Generates 3D position over time of the 3D hand model."""
    print(f"[{output_dir.name}] Generating forearm data (normals)...")
    output_ply_path = output_dir / "forearm_pointcloud_with_normals.ply"
    output_metadata_path = output_dir / "forearm_pointcloud_with_normals_metadata.json"
    define_normals(point_cloud_path, output_ply_path, output_metadata_path)

    return output_ply_path, output_metadata_path
    

# --- MODIFIED: Added @flow decorator for consistency ---
@flow(name="Single Session Pipeline")
def run_single_session_pipeline(inputs: SessionInputs):
    """This flow processes a SINGLE dataset by calling the appropriate sub-flows."""
    print(f"🚀 Starting pipeline for session: {inputs.video_processed_output_dir.name}")

    # Stage 1: Primary processing
    rgb_video_path = Path(str(inputs.source_video).replace(".mkv", ".mp4"))
    
    # Stage 2: Tracking
    #led_tracking = track_led_blinking(rgb_video_path, inputs.stimulus_metadata, inputs.video_processed_output_dir)
    track_stickers(rgb_video_path, inputs.source_video, inputs.video_processed_output_dir)
    return 

    # Stage 4: 3D Reconstruction
    prepare_hand_tracking_session(rgb_video_path, inputs.hand_models_dir, inputs.video_processed_output_dir)
    prepare_forearm(inputs.video_processed_output_dir / "forearm_pointcloud.ply", inputs.video_processed_output_dir)

    # Stage 5: Data Integration
    #unified_data = unify_dataset(somatosensory_chars, ttl_signal, inputs.video_processed_output_dir)
    
    print(f"✅ Pipeline finished for session: {inputs.video_processed_output_dir.name}")

# --- 5. The "Dispatcher" Flow ---
# --- MODIFIED: This function now runs sessions sequentially ---
def batch_process_all_sessions(configs_kinect_dir: Path, project_data_root: Path):
    """Finds all session YAML files and triggers a pipeline run for each one sequentially."""

    if not configs_kinect_dir.is_dir():
        raise ValueError(f"Sessions folder not found: {configs_kinect_dir}")

    session_files = list(configs_kinect_dir.glob("*.yaml"))
    print(f"Found {len(session_files)} session(s) to process in '{configs_kinect_dir}'.")
    print("Processing sessions one by one...")

    # Iterate through each session file and process it directly.
    # The loop will wait for each call to `run_single_session_pipeline` to finish
    # before starting the next one.
    for i, session_file in enumerate(session_files):
        print(f"\n--- Processing session {i+1}/{len(session_files)}: {session_file.name} ---")
        with open(session_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config_data_abs = {key: project_data_root / value for key, value in config_data.items()}
        validated_inputs = SessionInputs(**config_data_abs)

        # This is now a direct, blocking call.
        # The script will not proceed to the next iteration of the loop
        # until this function completes.
        run_single_session_pipeline(inputs=validated_inputs)

    print("\n🎉 All sessions processed successfully.")


# --- 6. Main execution block (Unchanged) ---
if __name__ == "__main__":
    print("🛠️  Setting up files for processing...")
    
    configs_dir = Path("configs")
    kinect_dir = Path("kinect_configs")

    configs_kinect_dir = configs_dir / kinect_dir
    
    project_data_root = path_tools.get_project_data_root()

    print("🚀 Launching batch processing...")
    batch_process_all_sessions(configs_kinect_dir=configs_kinect_dir, project_data_root=project_data_root)