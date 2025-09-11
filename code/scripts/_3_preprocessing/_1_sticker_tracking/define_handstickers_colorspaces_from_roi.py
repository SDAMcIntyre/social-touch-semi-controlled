# Standard library imports
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Third-party imports
import pandas as pd
import numpy as np

from preprocessing.common import (
    VideoMP4Manager
)

from preprocessing.stickers_analysis import (
    ROIAnnotationFileHandler,
    ROIAnnotationManager,

    TrackerReviewGUI,
    TrackerReviewOrchestrator,

    FrameROIColor,

    ColorSpaceFileHandler
)


def generate_color_config(objs_to_track: List[str]) -> Dict[str, Any]:
    """Generates a configuration dictionary for object drawing colors.

    This function maps object names (e.g., 'sticker_yellow') to their BGR color
    values. It creates both a bright "live" color for active drawing and a
    darker "final" color for committed shapes.

    Args:
        objs_to_track: A list of strings, where each string is an object name
                       expected to end with a color (e.g., 'sticker_blue').

    Returns:
        A config dictionary where each key is an object name and its value is
        a dictionary containing 'live' and 'final' BGR color tuples.
    """
    # OpenCV uses BGR (Blue, Green, Red) format, not RGB.
    color_map = {
        'yellow': (0, 255, 255),
        'blue': (255, 0, 0),
        'green': (0, 255, 0),
        'red': (0, 0, 255),
        'white': (255, 255, 255)
    }
    color_config = {}
    for obj_name in objs_to_track:
        try:
            color_name = obj_name.split('_')[-1]
            if color_name in color_map:
                live_color = color_map[color_name]
                final_color = tuple(c // 2 for c in live_color)
                color_config[obj_name] = {'live': live_color, 'final': final_color}
            else:
                print(f"⚠️ Warning: Color '{color_name}' not in map. Skipping '{obj_name}'.")
        except IndexError:
            print(f"⚠️ Warning: Could not extract color from '{obj_name}'. Skipping.")
            continue
    return color_config


def prepare_json_update_payload(
    frame_ids: List[int], adjusted_colorspaces: List[Dict[str, Any]], status: str = "pending"
) -> Dict[str, Any]:
    """Formats the final colorspace data into a dictionary for JSON output.

    Args:
        frame_ids: List of frame IDs corresponding to the colorspaces.
        adjusted_colorspaces: List of colorspace dicts with adjusted coordinates.
        status: The review status to assign in the payload.

    Returns:
        A dictionary formatted to be saved as JSON content.
    """
    payload = {'status': status, 'colorspaces': []}
    for frame_id, colorspace in zip(frame_ids, adjusted_colorspaces):
        frame_content = {"frame_id": frame_id, "colorspace": colorspace}
        payload['colorspaces'].append(frame_content)
    return payload


def update_json_object(
    file_path: str, object_name: str, new_content: dict, overwrite: bool = False
) -> bool:
    """Updates or overwrites a specific top-level object within a JSON file.

    This function safely reads a JSON file, modifies a specified object, and
    writes the changes back.

    Args:
        file_path: The path to the JSON file.
        object_name: The key of the top-level object to update.
        new_content: A dictionary containing the new content for the object.
        overwrite: If True, the existing object is completely replaced.
                   If False (default), new content is merged into the existing object.

    Returns:
        bool: False if the update was successful, True if an error occurred.
              (Note: Returning True on error is unconventional.)
    """
    all_data = {}
    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                all_data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"❌ Error reading or parsing {file_path}: {e}")
        return True

    # Update the specific object
    if overwrite or object_name not in all_data:
        print(f"✅ Overwriting or creating object '{object_name}'.")
        all_data[object_name] = new_content
    else:
        # Merge if both existing and new content are dictionaries
        existing_object = all_data.get(object_name, {})
        if isinstance(existing_object, dict) and isinstance(new_content, dict):
            print(f"✅ Merging new content into object '{object_name}'.")
            existing_object.update(new_content)
            all_data[object_name] = existing_object
        else:
            print(f"⚠️ Cannot merge due to incompatible types. Overwriting '{object_name}'.")
            all_data[object_name] = new_content

    # Write the updated data back to the file
    try:
        with open(file_path, "w") as f:
            json.dump(all_data, f, indent=4)
        print(f"💾 Successfully saved data to '{os.path.basename(file_path)}'")
        return False
    except IOError as e:
        print(f"❌ An error occurred while writing to file: {e}")
        return True

def run_colorspace_definition_tool(
    video_frames: List[np.ndarray], 
    colors_rgb: Dict[str, Tuple[int, int, int]],
    title: str = None
) -> List[Dict[str, Any]]:
    """Runs the ROI drawing tool for each frame to define a colorspace.

    This function iterates through a list of frames, launching the FrameROIColor tool
    for each one. The user can then manually draw an ellipse and freehand shape to
    define the target colorspace.

    Args:
        video_frames: A list of frames (as NumPy arrays) to be processed.
        colors_rgb: A dictionary containing 'live' and 'final' RGB color tuples
                    to use for the drawing interface.

    Returns:
        A list of dictionaries, where each dictionary contains the extracted
        colorspace data for the corresponding frame.
    """
    defined_colorspaces = []
    for i, frame_bgr in enumerate(video_frames):
        print(f"\n🎨 Defining colorspace for frame {i+1}/{len(video_frames)}...")
        tracker = FrameROIColor(
            frame_bgr,
            resize_to=(1024, 768),
            is_bgr=True,
            color_live=colors_rgb['live'],
            color_final=colors_rgb['final'],
            window_title=title
        )
        tracker.run()
        tracking_data = tracker.get_tracking_data()

        if tracking_data:
            print("--- ✅ Tracking Data Extracted ---")
            defined_colorspaces.append(tracking_data)

    return defined_colorspaces

def define_handstickers_colorspaces_from_roi(
    video_path: Path,
    roi_metadata_path: Path,
    dest_metadata_path: Path,
    *,
    force_processing: bool = False
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
        roi_metadata_path (str): The path to the source JSON metadata file.
        dest_metadata_path (str): The path for the output JSON file.
    """
    print(f"🎬 Starting colorspace definition for: '{Path(video_path).name}'")

    # --- 1. Pre-computation Checks & Validation ---
    if not os.path.exists(roi_metadata_path):
        print("🟡 Skipping: Source metadata not found. Tracking must be validated first.")
        return

    if not force_processing and os.path.exists(dest_metadata_path):
        print("🟡 Skipping: Destination metadata file already exists.")
        # TODO: Implement colorspace assessment/update for existing files.
        return

    # --- 2. Load Source Data ---
    annotation_data = ROIAnnotationFileHandler.load(roi_metadata_path, create_if_not_exists=False)
    object_names = ROIAnnotationManager(annotation_data).get_object_names()

    if not object_names:
        print("❌ Halting: 'object_names' is empty in source metadata.")
        return

    color_config = generate_color_config(object_names)

    for object_name in object_names:
        print(f"\n➡️ Processing object: '{object_name}'")
        # Construct the new filename by inserting your object_name.
        object_video_filename = f"{video_path.stem}_{object_name}{video_path.suffix}"
        object_video_path = video_path.parent / object_video_filename

        print(f"Loading video '{object_video_path.name}'...")
        video_manager = VideoMP4Manager(object_video_path)
            
           
           
        # Step 3b: Manually select the best frames for colorspace definition
        print("   - Waiting for user to select representative frames...")
        view = TrackerReviewGUI(title=object_video_path.name, windowState='maximized')
        controller = TrackerReviewOrchestrator(model=video_manager, view=view)
        _, selected_frame_indices, _ = controller.run()

        if not selected_frame_indices:
            print(f"   - No frames selected for '{object_name}'. Skipping.")
            continue
            
        selected_frames = [video_manager.get_frame(i) for i in selected_frame_indices]

        # Step 3c: Define colorspaces based on the selected cropped frames (ROIs)
        # The coordinates will be relative to the top-left of the cropped image.
        colorspaces = run_colorspace_definition_tool(
            selected_frames, 
            colors_rgb=color_config.get(object_name),
            title=object_video_filename
        )

        # Step 3e: Prepare payload and update the destination JSON file

        ColorSpaceFileHandler
        json_payload = prepare_json_update_payload(selected_frame_indices, colorspaces)
        update_json_object(dest_metadata_path, object_name, json_payload)
        print(f"   ✅ Successfully defined and saved colorspace for '{object_name}'.")

    print(f"\n🎉 Finished processing for: '{Path(video_path).name}'")

