# Standard library imports
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Third-party imports
import pandas as pd
import os
import json
from pathlib import Path
from typing import List


# Local application/library specific imports
# (Assuming these are located in a 'utils' subdirectory relative to this script)
from .utils.roi_to_position_funcs import (
    generate_color_config,
    load_video_frames_bgr,
    prepare_json_update_payload,
    review_frames,
    run_colorspace_definition_tool,
    update_json_object,
)

from .utils.colorspace.ColorspaceFileHandler import ColorspaceFileHandler

def define_object_colorspaces_for_video_wip(
    video_path: str,
    src_metadata_path: str,
    dest_metadata_path: str,
) -> None:
    """
    Orchestrates the colorspace definition for tracked objects in a video
    using the Model-View-Controller (MVC) architecture.
    """
    print(f"🎬 Starting colorspace definition for: '{Path(video_path).name}'")
    video_path = Path(video_path)

    try:
        with open(src_metadata_path, 'r') as f:
            metadata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error loading source files: {e}")
        return

    objects_to_track = metadata.get('objects_to_track', {})
    if not objects_to_track:
        print("❌ Halting: 'obj_to_track' key missing or empty in source metadata.")
        return

    # --- 1. Use ColorspaceFileHandler for robust I/O ---
    try:
        # Initialize the destination handler. It will create the file on save.
        dest_handler = ColorspaceFileHandler(dest_metadata_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error initializing file handlers: {e}")
        return

    # --- 2. Process Each Object with a Dedicated Controller ---
    for object_name in objects_to_track.keys():
        print(f"\n➡️ Processing object: '{object_name}'")
        
        # Load the specific video frames for this object
        object_video_filename = f"{video_path.stem}_{object_name}{video_path.suffix}"
        object_video_path = video_path.parent / object_video_filename
        video_frames = load_video_frames_bgr(object_video_path)
        
        if not video_frames:
            print(f"🟡 Warning: Could not load frames for '{object_name}'. Skipping.")
            continue

        # Step 3b: Manually select the best frames for colorspace definition
        print("   - Waiting for user to select representative frames...")
        _, selected_frame_indices = review_frames(video_frames, title=object_video_filename)
        
        if not selected_frame_indices:
            print(f"   - No frames selected for '{object_name}'. Skipping.")
            continue
            
        selected_frames = [video_frames[i] for i in selected_frame_indices]

        # --- 3. Run the Interactive Session ---
        # The Controller now handles the entire workflow: frame navigation,
        # ROI drawing, analysis, and saving. This single call replaces the
        # old `review_frames` and `run_colorspace_definition_tool` functions.
        print("🚀 Launching interactive annotation session...")
        app_controller = ColorspaceController(
            frames=selected_frames,
            object_name=object_name,
            dest_file_handler=dest_handler
        )
        app_controller.run() # This is a blocking call that runs the UI loop

        print(f"✅ Session for '{object_name}' complete.")

    print(f"\n🎉 Finished processing for: '{Path(video_path).name}'")


def define_object_colorspaces_for_video(
    video_path: str,
    src_metadata_path: str,
    dest_metadata_path: str,
) -> None:
    """Orchestrates the colorspace definition for tracked objects in a video.

    This function performs the following steps:
    1.  Validates that source metadata exists and destination metadata does not.
    2.  Loads video frames and tracking data.
    3.  For each object specified in the metadata:
        a. Extracts the object's tracked regions of interest (ROIs).
        b. Prompts the user to select representative frames.
        c. Defines a colorspace based on the user's selection within the ROI.
        d. Translates the ROI-local coordinates to full-frame coordinates.
        e. Saves the new colorspace data to a destination JSON file.

    Args:
        video_path (str): The absolute path to the source video file.
        tracking_csv_path (str): The absolute path to the CSV with tracking data.
        src_metadata_path (str): The path to the source JSON metadata file.
        dest_metadata_path (str): The path for the output JSON file.
    """
    print(f"🎬 Starting colorspace definition for: '{Path(video_path).name}'")

    # --- 1. Pre-computation Checks & Validation ---
    if not os.path.exists(src_metadata_path):
        print("🟡 Skipping: Source metadata not found. Tracking must be validated first.")
        return

    if os.path.exists(dest_metadata_path):
        print("🟡 Skipping: Destination metadata file already exists.")
        # TODO: Implement colorspace assessment/update for existing files.
        return

    # --- 2. Load Source Data ---
    try:
        with open(src_metadata_path, 'r') as f:
            metadata = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error loading source files: {e}")
        return

    objects_to_track = metadata.get('objects_to_track', {})
    if not objects_to_track:
        print("❌ Halting: 'obj_to_track' key missing or empty in source metadata.")
        return

    # --- 3. Process Each Object ---
    object_names = list(objects_to_track.keys())
    color_config = generate_color_config(object_names)

    for object_name in object_names:
        print(f"\n➡️ Processing object: '{object_name}'")
        # Construct the new filename by inserting your object_name.
        object_video_filename = f"{video_path.stem}_{object_name}{video_path.suffix}"
        object_video_path = video_path.parent / object_video_filename

        video_frames = load_video_frames_bgr(object_video_path)

        # Step 3b: Manually select the best frames for colorspace definition
        print("   - Waiting for user to select representative frames...")
        _, selected_frame_indices = review_frames(video_frames, title=object_video_filename)
        
        if not selected_frame_indices:
            print(f"   - No frames selected for '{object_name}'. Skipping.")
            continue
            
        selected_frames = [video_frames[i] for i in selected_frame_indices]

        # Step 3c: Define colorspaces based on the selected cropped frames (ROIs)
        # The coordinates will be relative to the top-left of the cropped image.
        colorspaces = run_colorspace_definition_tool(
            selected_frames, 
            colors_rgb=color_config.get(object_name),
            title=object_video_filename
        )

        # Step 3e: Prepare payload and update the destination JSON file
        json_payload = prepare_json_update_payload(selected_frame_indices, colorspaces)
        update_json_object(dest_metadata_path, object_name, json_payload)
        print(f"   ✅ Successfully defined and saved colorspace for '{object_name}'.")

    print(f"\n🎉 Finished processing for: '{Path(video_path).name}'")

