import os
from pathlib import Path
from typing import Union, List, Tuple, Optional

import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# --- 1. Project-Specific Imports ---
# Assuming these imports are correct for your project structure
import utils.path_tools as path_tools
from primary_processing import (
    ForearmConfigFileHandler,
    ForearmConfig,
    KinectConfigFileHandler,
    KinectConfig,
)
from preprocessing.forearm_extraction import (
    ForearmFrameParametersFileHandler,
    ForearmParameters,
)
from _3_preprocessing._3_forearm_extraction import (
    define_forearm_extraction_parameters,
    extract_forearm,
    clean_forearm_pointcloud,
    define_normals,
)


# --- CHOOSE YOUR EXECUTION MODE HERE ---
# ------------------------------------------------------
# ---------------------------------------
# ---------------------
# -----------
FORCE_PROCESSING = True
# -----------
# ---------------------
# ---------------------------------------
# ------------------------------------------------------


def create_confirmation_flag():
    """
    Displays a confirmation dialog. If 'Yes' is clicked,
    it creates a flag file.
    """
    # Create the main window but hide it, as we only need the dialog box
    root = tk.Tk()
    root.withdraw()

    # Show the confirmation dialog
    response = messagebox.askyesno(
        title="Processing Confirmation",
        message="Was the processing correct?"
    )

    # Act based on the user's response
    return response


# --- 2. Helper Functions for the Main Pipeline ---

def _setup_session_directories(session_output_dir: Path) -> Path:
    """Creates the necessary output directories for a processing session."""
    print(f"📂 Setting up output directory: {session_output_dir}")
    session_output_dir.mkdir(parents=True, exist_ok=True)
    
    pointclouds_output_dir = session_output_dir / "forearm_pointclouds"
    pointclouds_output_dir.mkdir(exist_ok=True)
    print(f"📦 Point clouds will be saved in: {pointclouds_output_dir}")
    
    return pointclouds_output_dir


def _gather_rgb_video_paths(
    config_links: List[str], project_data_root: Path
) -> List[Path]:
    """Loads linked Kinect configs to find all associated RGB video files."""
    rgb_video_paths: List[Path] = []
    print("🔍 Loading linked Kinect configurations to find RGB videos...")
    
    for config_link in config_links:
        try:
            config_data = KinectConfigFileHandler.load_and_resolve_config(config_link)
            kinect_inputs = KinectConfig(config_data=config_data, database_path=project_data_root)
            
            # The corresponding RGB video is expected to be an .mp4 file
            rgb_video_path = kinect_inputs.source_video.with_suffix('.mp4')
            
            if not rgb_video_path.exists():
                print(f"⚠️ Warning: Corresponding RGB video not found, skipping: {rgb_video_path}")
                continue
                
            rgb_video_paths.append(rgb_video_path)
        except Exception as e:
            print(f"❌ Error loading Kinect config from '{config_link}': {e}")

    if not rgb_video_paths:
        raise FileNotFoundError("Could not find any valid RGB video paths for ROI definition.")
        
    print(f"📹 Found {len(rgb_video_paths)} RGB videos for ROI definition.")
    return rgb_video_paths


def _process_single_forearm_frame(
    params: ForearmParameters,
    rgb_video_paths: List[Path],
    pointclouds_output_dir: Path
) -> Optional[Tuple[Path, Path]]:
    """
    Extracts, processes, and saves the point cloud for a single frame.
    
    Returns a tuple of (normals_path, normals_metadata_path) on success, else None.
    """
    # Find the source video corresponding to the parameters
    matching_rgb_path = next((p for p in rgb_video_paths if p.name == params.video_filename), None)

    if not matching_rgb_path:
        print(f"⚠️ Warning: Could not find source for '{params.video_filename}', skipping frame {params.frame_id}.")
        return None

    # The depth video (.mkv) is assumed to share the same stem as the RGB video (.mp4)
    primary_source_video = matching_rgb_path.with_suffix('.mkv')
    if not primary_source_video.exists():
        print(f"⚠️ Warning: Depth video not found, skipping: {primary_source_video}")
        return None

    # --- Generate unique filenames for this frame's outputs ---
    video_stem = primary_source_video.stem
    base_filename = f"{video_stem}_frame_{params.frame_id:04d}" # Zero-padded for sorting

    forearm_ply_path = pointclouds_output_dir / f"{base_filename}.ply"
    output_params_path = pointclouds_output_dir / f"{base_filename}_extraction_params.json"

    # --- Step 1: Extract the raw forearm point cloud ---
    print(f"🔎 Extracting forearm from: {primary_source_video.name} [Frame: {params.frame_id}]")
    extract_forearm(
        video_path=primary_source_video,
        video_config=params,
        output_ply_path=forearm_ply_path,
        output_params_path=output_params_path,
        interactive=True
    )
    print(f"💾 Raw forearm point cloud saved to: {forearm_ply_path.name}")

    # --- Step 2: Clean the forearm point cloud (NEW STEP) ---
    cleaned_forearm_ply_path = pointclouds_output_dir / f"{base_filename}_cleaned.ply"
    
    # Optional: Define metadata path for cleaning stats if the function supports it
    cleaned_metadata_path = pointclouds_output_dir / f"{base_filename}_cleaning_stats.json"

    print("🧹 Cleaning the extracted forearm point cloud...")
    clean_forearm_pointcloud(
        input_ply_path=forearm_ply_path,
        output_ply_path=cleaned_forearm_ply_path,
        output_metadata_path=cleaned_metadata_path
    )
    print(f"💾 Cleaned point cloud saved to: {cleaned_forearm_ply_path.name}")

    # --- Step 3: Calculate and save normals for the extracted point cloud ---
    # NOTE: Now using the CLEANED point cloud as input
    forearm_ply_normals_path = pointclouds_output_dir / f"{base_filename}_with_normals.ply"
    forearm_metadata_normals_path = pointclouds_output_dir / f"{base_filename}_with_normals_metadata.json"

    print("🧠 Calculating normals for the point cloud...")
    define_normals(cleaned_forearm_ply_path, forearm_ply_normals_path, forearm_metadata_normals_path)
    print(f"💾 Point cloud with normals saved to: {forearm_ply_normals_path.name}")

    return forearm_ply_normals_path, forearm_metadata_normals_path


# --- 3. Main Processing Pipeline ---

def generate_forearm_pointcloud(
    inputs: 'ForearmConfig',
    project_data_root: Path
) -> Tuple[Path, Path]:
    """
    Processes a forearm session to generate a point cloud with normals.
    
    This function orchestrates the pipeline:
    1. Sets up output directories.
    2. Gathers source video files.
    3. Initiates interactive ROI (Region of Interest) definition.
    4. Loops through each defined ROI to extract and process the forearm point cloud.
    """
    print(f"🚀 Starting pipeline for session: {inputs.session_id}")

    # STEP 1: Setup session-specific output directories
    pointclouds_output_dir = _setup_session_directories(inputs.session_processed_path)

    flag_file = pointclouds_output_dir / ".SUCCESS"
    if not FORCE_PROCESSING and os.path.exists(flag_file):
        print(f"⚠️ Flag file found in '{pointclouds_output_dir}'. Skip this session.")
        return True
    
    # STEP 2: Gather all RGB video paths needed for ROI definition
    rgb_video_paths = _gather_rgb_video_paths(inputs.config_file_links, project_data_root)

    # STEP 3: Interactively define the forearm extraction parameters (ROI)
    metadata_path = pointclouds_output_dir / f"{inputs.session_id}_arm_roi_metadata.json"
    print("✍️ Please define the forearm ROI in the upcoming interactive session...")
    define_forearm_extraction_parameters(rgb_video_paths, metadata_path)
    
    forearm_parameters_list: List[ForearmParameters] = ForearmFrameParametersFileHandler.load(metadata_path)

    # STEP 4: Process each defined frame parameter
    for params in forearm_parameters_list:
        try:
            print(f"🚀 Starting forearm extraction for: {params.video_filename}, frame {params.frame_id}")
            _process_single_forearm_frame(
                params, rgb_video_paths, pointclouds_output_dir
            )
        except Exception as e:
            print(f"❌ An unexpected error occurred while processing frame {params.frame_id}: {e}")
            continue
    
    # Act based on the user's response
    if create_confirmation_flag():
        Path(flag_file).touch()
        print(f"✅ Confirmation received. Flag file created at: {flag_file.resolve()}")
        print("✅ Pipeline finished successfully.")
    else:
        Path(flag_file).unlink()
        print(f"🧹 Removed old flag file from previous run.")


    return True


def batch_process_all_sessions(configs_forearm_dir: Path, project_data_root: Path):
    """Finds all session YAML files and runs the processing pipeline for each one."""
    if not configs_forearm_dir.is_dir():
        raise FileNotFoundError(f"Configuration directory not found: {configs_forearm_dir}")

    session_files = sorted(list(configs_forearm_dir.glob("*.yaml")))
    if not session_files:
        print(f"⚠️ No session *.yaml files found in '{configs_forearm_dir}'. Nothing to do.")
        return
    
    total_sessions = len(session_files)
    print(f"Found {total_sessions} session(s) to process in '{configs_forearm_dir}'.")

    for i, session_file in enumerate(session_files):
        print(f"\n{'='*20} Processing Session {i + 1}/{total_sessions}: {session_file.name} {'='*20}")
        try:
            forearm_config: ForearmConfig = ForearmConfigFileHandler.load(session_file)
            generate_forearm_pointcloud(inputs=forearm_config, project_data_root=project_data_root)
        except Exception as e:
            print(f"❌ FATAL ERROR processing session {session_file.name}: {e}")
            print("🛑 Skipping to the next session.")
            continue

    print(f"\n🎉 All {total_sessions} sessions processed.")


# --- 4. Script Execution ---
if __name__ == "__main__":
    print("🛠️ Initializing batch processing script...")
    
    # Define project paths relative to this script's location
    try:
        project_root = Path(__file__).resolve().parents[2]
        configs_dir = project_root / "configs"
        configs_forearm_dir = configs_dir / "forearm_configs"
        
        # Use the utility function to find the data root
        project_data_root = path_tools.get_project_data_root()

        print(f"Project Root: {project_root}")
        print(f"Data Root: {project_data_root}")
        print(f"Forearm Configs: {configs_forearm_dir}")
        
        # Launch the main batch processing function
        batch_process_all_sessions(
            configs_forearm_dir=configs_forearm_dir,
            project_data_root=project_data_root
        )
    except Exception as e:
        print(f"An error occurred during script setup or execution: {e}")