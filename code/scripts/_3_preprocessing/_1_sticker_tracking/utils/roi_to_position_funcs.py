# ======================================================================================
# Imports
# ======================================================================================
# --- Standard Library Imports ---
import json
import os
import ast  # Used to safely convert string representations of data structures
from typing import List, Dict, Any, Tuple, Optional, Union

# --- Third-party Library Imports ---
import numpy as np
import pandas as pd

# --- Local Application Imports ---
from .FrameROIColor import FrameROIColor
from .ObjectTrackerReviewTool import ObjectTrackerReviewTool


# ======================================================================================
# 1. High-Level UI & Workflow Functions
# (Functions that a user might call directly to orchestrate a major process)
# ======================================================================================

def review_frames(
        video_frames: List[np.ndarray], 
        title="Loaded Frame Review"
) -> Tuple[str, List[int]]:
    """Launches the Tkinter review tool for marking frames that need attention.

    This function initializes and runs the ObjectTrackerReviewTool, allowing a user
    to step through frames and mark specific ones for re-labeling.

    Args:
        video_frames: A list of video frames, where each frame is a NumPy array.

    Returns:
        A tuple containing:
        - The final status string set by the user in the tool (e.g., 'saved', 'cancelled').
        - A list of integer frame indices that the user marked for review.
    """
    print("\n🚀 Launching Tkinter Video Player for review...")
    player = ObjectTrackerReviewTool(video_source=video_frames, title=title)
    final_status, marked_frames = player.play(windowState='maximized')

    print("\n--- Results from player session ---")
    print(f"Final Status: {final_status}")
    print(f"Frames marked for labeling: {marked_frames}")

    return final_status, marked_frames


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
        # Note: The FrameROIColor tool may expect specific parameters like resize_to.
        # These could be passed as arguments for more flexibility.
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


# ======================================================================================
# 2. Data Processing & Transformation Functions
# (Functions that manipulate data structures between major steps)
# ======================================================================================

def crop_frames_by_rois(
    frames: List[np.ndarray], tracked_rois: pd.Series
) -> List[np.ndarray]:
    """Extracts sub-frames (crops) from a list of frames based on ROI data.

    This function pairs each full video frame with its corresponding Region of Interest
    (ROI) and crops the frame to that ROI. It safely handles missing or invalid ROIs.

    Args:
        frames: A list of full video frames (NumPy arrays).
        tracked_rois: A pandas Series containing ROI coordinates [x, y, w, h]
                      or NaN for frames without a valid ROI.

    Returns:
        A list of the extracted sub-frames (cropped images as NumPy arrays).
    """
    extracted_subframes = []
    for frame, roi in zip(frames, tracked_rois):
        # An ROI is considered invalid if it's NaN or contains a NaN value.
 #       if isinstance(roi, list):
 #           is_invalid_roi = any(pd.isna(r) for r in roi)
#        else:
#            is_invalid_roi =  pd.isna(roi)
        is_invalid_roi = np.isnan(np.asarray(roi)).any()


        if not is_invalid_roi:
            # Ensure coordinates are integers before slicing
            x, y, w, h = map(int, roi)
            # Slice the frame using NumPy's [y:y+h, x:x+w] convention
            subframe = frame[y : y + h, x : x + w]
            extracted_subframes.append(subframe)

    return extracted_subframes


def adjust_colorspace_to_frame_coords(
    colorspace: Dict[str, Any], roi_x: int, roi_y: int
) -> Dict[str, Any]:
    """Adjusts colorspace coordinates from a sub-frame (ROI) to the full frame.

    When a colorspace is defined on a cropped sub-frame, its coordinates are
    relative to that crop. This function translates those relative coordinates
    back to the absolute coordinates of the original, full-sized video frame.

    Args:
        colorspace: A dictionary containing colorspace data, including 'ellipse'
                    and 'freehand_pixels' with relative coordinates.
        roi_x: The x-offset of the ROI's top-left corner in the original frame.
        roi_y: The y-offset of the ROI's top-left corner in the original frame.

    Returns:
        The colorspace dictionary with its coordinates updated to be absolute.
    """
    # Update ellipse center coordinates
    ellipse = colorspace['ellipse']
    center_x, center_y = ellipse['center_original_px']
    ellipse['center_original_px'] = (center_x + roi_x, center_y + roi_y)

    # Update freehand pixel coordinates
    freehand = colorspace['freehand_pixels']
    adjusted_coords = [(x + roi_x, y + roi_y) for x, y in freehand['coordinate']]
    freehand['coordinate'] = adjusted_coords

    return colorspace


def parse_rois_from_dataframe(
    tracked_obj_df: pd.DataFrame, obj_name: str, verbose: bool = False
) -> pd.Series:
    """Parses a column of string-formatted ROIs from a DataFrame into a Series of lists.

    This function safely evaluates a DataFrame column (e.g., 'sticker_yellow_roi')
    that contains ROI coordinates stored as strings, like "[10, 20, 50, 50]".
    It uses `ast.literal_eval` to prevent security risks associated with `eval()`.

    Args:
        tracked_obj_df: The DataFrame containing the tracked object data.
        obj_name: The name of the object (e.g., 'sticker_yellow') used to identify
                  the ROI column, which is expected to be f"{obj_name}_roi".
        verbose: If True, prints the parsed series for debugging.

    Returns:
        A pandas Series where each element is a list of ROI coordinates or np.nan
        if parsing failed or the original value was not a string.
    """
    def safe_eval_roi(roi_str: str) -> Optional[list]:
        """Safely evaluates a string to a list/tuple, returning np.nan on failure."""
        if isinstance(roi_str, str):
            try:
                return ast.literal_eval(roi_str)
            except (ValueError, SyntaxError):
                return np.nan  # The string is malformed
        return np.nan  # The input is not a string (e.g., already np.nan)

    roi_column_name = f"{obj_name}_roi"
    if roi_column_name not in tracked_obj_df.columns:
        raise ValueError(f"Column '{roi_column_name}' not found in the DataFrame.")

    parsed_series = tracked_obj_df[roi_column_name].apply(safe_eval_roi)

    if verbose:
        print("--- Series after safe parsing ---")
        print(parsed_series)

    return parsed_series


# --- 1. Centralize dictionary keys for robustness ---
# This makes maintenance easier if the input data format changes.
KEY_OBJECT_NAME = 'object_name'
KEY_FRAME_NUMBER = 'frame_number'
KEY_CENTER_X = 'center_x'
KEY_CENTER_Y = 'center_y'
KEY_AXES_MAJOR = 'axes_major'
KEY_AXES_MINOR = 'axes_minor'
KEY_ANGLE = 'angle'
KEY_SCORE = 'score'
KEY_THRESHOLD = 'optimal_threshold'

def _flatten_and_filter_ellipses(ellipse_results: Dict[Any, List[Dict]], obj_name: str) -> List[Dict]:
    """
    Flattens the nested dictionary of ellipses and filters for a specific object name.

    This is a helper function focused solely on parsing and filtering.

    Args:
        ellipse_results (Dict[Any, List[Dict]]): 
            A dictionary where values are lists of ellipse data dictionaries.
        obj_name (str): 
            The name of the object to filter for.

    Returns:
        List[Dict]: 
            A flat list of dictionaries, where each dictionary represents a filtered ellipse.
    """
    # --- 2. Use a generator expression for efficiency and readability ---
    # This avoids creating a large intermediate list of all ellipses.
    all_ellipses = (
        ellipse_data
        for ellipses_frame in ellipse_results.values()
        for ellipse_data in ellipses_frame
    )

    # --- 3. Use a list comprehension to build the final, restructured list ---
    return [
        {
            'frame_number': ellipse[KEY_FRAME_NUMBER],
            'center': [ellipse[KEY_CENTER_X], ellipse[KEY_CENTER_Y]],
            'axes': [ellipse[KEY_AXES_MAJOR], ellipse[KEY_AXES_MINOR]],
            'angle': ellipse[KEY_ANGLE],
            'score': ellipse[KEY_SCORE],
            'optimal_threshold': ellipse[KEY_THRESHOLD],
        }
        for ellipse in all_ellipses
        if ellipse.get(KEY_OBJECT_NAME) == obj_name # Use .get() for safety
    ]

def parse_ellipses_from_dict(ellipse_results: Dict[Any, List[Dict]], obj_name: str) -> pd.DataFrame:
    """
    Parses ellipse data from a dictionary, filters by object name, and returns a reindexed DataFrame.

    This function orchestrates the parsing and DataFrame creation process, demonstrating
    good separation of concerns.

    Args:
        ellipse_results (Dict[Any, List[Dict]]): 
            A dictionary where values are lists of ellipse data dictionaries.
        obj_name (str): 
            The name of the object to filter for.

    Returns:
        pd.DataFrame: 
            A DataFrame with 'frame_number' as the index, reindexed to be complete.
            Rows for frames without data will be filled with NaN.
    """
    # --- 4. Call the helper function to separate concerns ---
    flat_rows = _flatten_and_filter_ellipses(ellipse_results, obj_name)

    if not flat_rows:
        return pd.DataFrame() # Return an empty DataFrame if no objects were found

    # Create the DataFrame
    df = pd.DataFrame(flat_rows)
    df.set_index('frame_number', inplace=True)

    # Reindex logic remains the same, as it was already correct and clear.
    max_frame = df.index.max()
    complete_index = pd.RangeIndex(start=0, stop=max_frame + 1, name='frame_number')
    df = df.reindex(complete_index)

    return df

    

# ======================================================================================
# 3. Utility, I/O, and Payload Functions
# (Low-level helpers for file operations, data extraction, and formatting)
# ======================================================================================

def load_video_frames_bgr(video_path: str) -> List[np.ndarray]:
    """Loads all frames from a video file into a list of NumPy arrays.

    Args:
        video_path: The absolute or relative path to the video file.

    Returns:
        A list of frames, where each frame is a NumPy array in BGR format.
    """
    # Instantiates the review tool simply to use its frame loading capability.
    # Consider refactoring frame loading into a separate utility if this is common.
    loader = ObjectTrackerReviewTool(str(video_path), as_bgr=True)
    return loader.frames


def extract_specific_frames(
    all_frames: List[np.ndarray], indices_to_extract: List[int]
) -> List[np.ndarray]:
    """Extracts frames from a list of frames by their specific indices.

    Args:
        all_frames: A list containing all video frames as NumPy arrays.
        indices_to_extract: A list of integer indices for the frames to extract.

    Returns:
        A new list containing only the frames at the specified valid indices.
    """
    extracted_frames = []
    total_frames = len(all_frames)
    print(f"\n🔎 Attempting to extract {len(indices_to_extract)} specific frames...")

    for index in indices_to_extract:
        if 0 <= index < total_frames:
            extracted_frames.append(all_frames[index])
        else:
            print(f"⚠️ Warning: Frame index {index} is out of bounds (video has "
                  f"{total_frames} frames). Skipping.")

    print(f"✅ Successfully extracted {len(extracted_frames)} frames.")
    return extracted_frames


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
    

def load_and_parse_colorspace_json(
    file_path: str
) -> Union[
    Tuple[Optional[List[int]], Optional[List[Dict[str, Any]]], Optional[str]],
    Tuple[Optional[Dict[str, Any]], Optional[List[str]]]
]:
    """Loads and parses colorspace data from a JSON file.

    This function operates in two modes:
    1. Single Object Mode (if object_name is provided):
       Parses a specific object and returns a tuple of its contents:
       (frame_ids, adjusted_colorspaces, status).

    2. All Objects Mode (if object_name is None):
       Parses all top-level objects in the file and returns a tuple
       containing a dictionary of the parsed data and a list of the object names:
       ({object_name: (frame_ids, colorspaces, status), ...}, [object_name, ...]).

    Args:
        file_path: The path to the JSON file.
        object_name: The key of the top-level object to parse. If None,
                     all objects are parsed.

    Returns:
        The parsed data in a format determined by the mode.
        Returns (None, None) or (None, None, None) on failure.
    """
    try:
        with open(file_path, "r") as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File not found at '{file_path}'")
        return (None, None, None)
    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode JSON from '{file_path}'")
        return (None, None, None)

    # --- Mode 2: Parse all objects if object_name is None ---
    all_parsed_data = {}
    object_keys = list(all_data.keys())
    for key in object_keys:
        payload = all_data.get(key)
        if not isinstance(payload, dict) or 'status' not in payload or 'colorspaces' not in payload:
            print(f"⚠️ Warning: Skipping object '{key}' due to malformed data or missing keys.")
            continue

        status = payload['status']
        frame_ids = [item.get('frame_id') for item in payload['colorspaces']]
        colorspaces = [item.get('colorspace') for item in payload['colorspaces']]
        
        # Filter out entries where frame_id or colorspace was missing
        valid_entries = [(fid, cs) for fid, cs in zip(frame_ids, colorspaces) if fid is not None and cs is not None]
        
        # Unzip the valid entries back into separate lists
        final_frame_ids = [entry[0] for entry in valid_entries]
        final_colorspaces = [entry[1] for entry in valid_entries]

        all_parsed_data[key] = (final_frame_ids, final_colorspaces, status)
    return all_parsed_data, object_keys


def adjust_colorspace_coordinates_to_frame(
    colorspace: Dict[str, Any], subframe_loc_x: int, subframe_loc_y: int
) -> Dict[str, Any]:
    """
    Adjusts the coordinates within a colorspace dictionary from a sub-frame
    (ROI) to the full video frame.

    Args:
        colorspace: The dictionary containing colorspace data, including 'ellipse'
                    and 'freehand_pixels' with relative coordinates.
        roi_location: A tuple (x, y, w, h) representing the top-left corner
                      of the ROI in the original frame.

    Returns:
        The colorspace dictionary with updated, absolute coordinates.
    """
    # Update ellipse center coordinates
    ellipse = colorspace['ellipse']
    center_x, center_y = ellipse['center_original_px']
    ellipse['center_original_px'] = (center_x + subframe_loc_x, center_y + subframe_loc_y)

    # Update freehand pixel coordinates
    freehand = colorspace['freehand_pixels']
    new_coords = [(x + subframe_loc_x, y + subframe_loc_y) for x, y in freehand['coordinate']]
    freehand['coordinate'] = new_coords
    
    return colorspace
