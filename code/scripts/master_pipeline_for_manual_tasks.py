import os
import yaml
from pathlib import Path

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
# This model defines the validated structure for our YAML configuration files.
class SessionInputs(BaseModel):
    """The complete, validated configuration for a single session."""
    source_video: FilePath
    stimulus_metadata: FilePath
    hand_models_dir: DirectoryPath
    video_primary_output_dir: Path
    video_processed_output_dir: Path


# --- 2. Processing Functions (Prefect decorators removed) ---

def track_led_blinking(video_path: Path, stimulus_metadata: Path, output_dir: Path) -> Path:
    """Requires .mp4 video, generates a csv file of LED blinking."""
    print(f"[{output_dir.name}] Tracking LED blinking...")
    # NOTE: The implementation for this function seems to be a placeholder.
    # It should return the path to the generated CSV file.
    return Path("") # Placeholder return

def track_stickers(
    rgb_video_path: Path,
    source_video: Path,
    root_output_dir: Path
) -> Path:
    """Requires .mp4 video, generates a csv file of stickers position."""
    output_dir = root_output_dir / "handstickers"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{output_dir.name}] Tracking stickers...")
    name_baseline = rgb_video_path.stem + "_handstickers"

    metadata_roi_path = output_dir / (name_baseline + "_roi_metadata.json")
    stickers_roi_csv_path = output_dir / (name_baseline + "_roi_tracking.csv")
    review_tracked_objects_in_video(rgb_video_path, metadata_roi_path, stickers_roi_csv_path)
    
    return stickers_roi_csv_path

    ensure_tracking_is_valid(metadata_roi_path)

    bypass_ellipse_tracking = True
    if not bypass_ellipse_tracking:
        # This block appears to contain unreachable or incomplete logic
        # from the original script. Review if this code is necessary.
        roi_video_base_path = output_dir / (name_baseline + "_roi_unified.mp4")
        metadata_colorspace_path = output_dir / (name_baseline + "_colorspace_metadata.json")
        define_object_colorspaces_for_video(
            roi_video_base_path,
            src_metadata_path=metadata_roi_path,
            dest_metadata_path=metadata_colorspace_path
        )

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
            # This early exit might not be desirable.
            return Path("")

        review_tracked_handstickers_position(
            rgb_video_path,
            roi_unified_csv_path,
            ellipses_csv_path,
            metadata_colorspace_path,
            output_video_path=ellipses_video_path
        )

        # The following section also appears unreachable/incomplete
        # stickers_pos_video_path = output_dir / (name_baseline + "_pos_tracking.mp4")
        # save_tracked_handstickers_position_as_video(rgb_video_path, stickers_pos_csv_path, metadata_pos_path, stickers_pos_video_path)
        #
        # try:
        #     ensure_tracking_is_valid(metadata_pos_path)
        # except ValueError as e:
        #     return Path("")

    # Assuming the final result is the ROI tracking file.
    # Adjust if another file is the intended output.
    result_csv_path = stickers_roi_csv_path
    return result_csv_path


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


def prepare_forearm(
    point_cloud_path: Path,
    output_dir: Path
) -> tuple[Path, Path]:
    """Generates 3D position over time of the 3D hand model."""
    print(f"[{output_dir.name}] Generating forearm data (normals)...")
    output_ply_path = output_dir / "forearm_pointcloud_with_normals.ply"
    output_metadata_path = output_dir / "forearm_pointcloud_with_normals_metadata.json"
    define_normals(point_cloud_path, output_ply_path, output_metadata_path)

    return output_ply_path, output_metadata_path


# --- 3. Main Pipeline Logic ---

def run_single_session_pipeline(inputs: SessionInputs):
    """This function processes a SINGLE dataset by calling the appropriate sub-routines."""
    print(f"🚀 Starting pipeline for session: {inputs.video_processed_output_dir.name}")

    # Stage 1: Primary processing
    rgb_video_path = Path(str(inputs.source_video).replace(".mkv", ".mp4"))

    # Stage 2: Tracking
    # led_tracking_path = track_led_blinking(rgb_video_path, inputs.stimulus_metadata, inputs.video_processed_output_dir)
    track_stickers(rgb_video_path, inputs.source_video, inputs.video_processed_output_dir)
    return

    # Stage 4: 3D Reconstruction
    prepare_hand_tracking_session(rgb_video_path, inputs.hand_models_dir, inputs.video_processed_output_dir)
    
    forearm_pc_path = inputs.video_processed_output_dir / "forearm_pointcloud.ply"
    if forearm_pc_path.exists():
        prepare_forearm(forearm_pc_path, inputs.video_processed_output_dir)
    else:
        print(f"⚠️  Warning: Forearm point cloud not found at '{forearm_pc_path}', skipping normal generation.")


    # Stage 5: Data Integration (Placeholder)
    # unified_data = unify_dataset(somatosensory_chars, ttl_signal, inputs.video_processed_output_dir)

    print(f"✅ Pipeline finished for session: {inputs.video_processed_output_dir.name}")


def batch_process_all_sessions(configs_kinect_dir: Path, project_data_root: Path):
    """Finds all session YAML files and triggers a pipeline run for each one sequentially."""
    if not configs_kinect_dir.is_dir():
        raise ValueError(f"Sessions folder not found: {configs_kinect_dir}")

    session_files = sorted(list(configs_kinect_dir.glob("*.yaml")))
    if not session_files:
        print(f"⚠️  Warning: No session *.yaml files found in '{configs_kinect_dir}'.")
        return
        
    print(f"Found {len(session_files)} session(s) to process in '{configs_kinect_dir}'.")
    print("Processing sessions one by one...")

    for i, session_file in enumerate(session_files):
        print(f"\n--- Processing session {i+1}/{len(session_files)}: {session_file.name} ---")
        try:
            with open(session_file, 'r') as f:
                config_data = yaml.safe_load(f)

            # Resolve paths relative to the project root
            config_data_abs = {key: project_data_root / value for key, value in config_data.items()}
            validated_inputs = SessionInputs(**config_data_abs)

            # Direct, blocking call to the pipeline for a single session.
            run_single_session_pipeline(inputs=validated_inputs)

        except Exception as e:
            print(f"❌ ERROR processing session {session_file.name}: {e}")
            print("Skipping to next session.")
            continue # Move to the next session if an error occurs

    print("\n🎉 All sessions processed.")


# --- 4. Main execution block ---
if __name__ == "__main__":
    print("🛠️  Setting up files for processing...")
    
    # It's good practice to define paths relative to the script location
    # or a well-defined root to avoid ambiguity.
    project_root = Path(__file__).resolve().parent.parent.parent
    configs_dir = project_root / "configs"
    kinect_dir = "kinect_configs" # Subdirectory name
    configs_kinect_dir = configs_dir / kinect_dir
    
    # Assuming path_tools is aware of your project structure
    project_data_root = path_tools.get_project_data_root()

    print("🚀 Launching batch processing...")
    batch_process_all_sessions(configs_kinect_dir=configs_kinect_dir, project_data_root=project_data_root)