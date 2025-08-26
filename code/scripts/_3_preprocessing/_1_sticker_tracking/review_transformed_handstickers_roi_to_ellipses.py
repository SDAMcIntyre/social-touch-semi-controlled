# --- 1. Standard Library Imports ---
import ast 
import json
import math
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import copy

# --- 2. Third-party Imports ---
import cv2
import numpy as np
import pandas as pd

# --- 3. Local Application Imports ---
from .utils.roi_to_position_funcs import (
    load_and_parse_colorspace_json,
    run_colorspace_definition_tool,
    generate_color_config,
    load_video_frames_bgr,
    prepare_json_update_payload,
    review_frames,
    update_json_object,
)

from .utils.colorspace.ColorspaceFileHandler import ColorspaceFileHandler

# --- 4. Constants and Enums ---

class Status(Enum):
    """Defines status constants for tracking and annotation review."""
    PENDING = "pending"
    VALID = "valid"
    LABELED = "Labeled"
    INITIAL_POS = "Initial POS"
    TRACKING = "Tracking"


# --- 5. Helper and Utility Functions ---

def get_success_flag_path(target_path: str) -> str:
    """
    Constructs the path for a .SUCCESS flag file based on a target file path.

    Args:
        target_path (str): The path to the file whose success we are flagging.

    Returns:
        str: The full path for the corresponding .SUCCESS file.
    """
    output_dir = os.path.dirname(target_path)
    output_basename = os.path.basename(target_path)
    return os.path.join(output_dir, f"{output_basename}.SUCCESS")

def create_completion_success_flag(target_path: str) -> None:
    """
    Creates an empty .SUCCESS file to signal successful completion of a process.

    Args:
        target_path (str): The path of the file that was successfully processed.
                           The flag will be created alongside it.
    """
    success_filepath = get_success_flag_path(target_path)
    try:
        with open(success_filepath, 'w'):
            pass  # Create an empty file
        print(f"✅ Process marked as complete. Created success flag at '{success_filepath}'")
    except IOError as e:
        print(f"⚠️ Could not create success flag file. Reason: {e}")


# --- 6. Data Loading Functions ---
import pandas as pd
import numpy as np
import ast

def load_sticker_ellipse_dataframes(csv_path: str, sticker_colors: list[str]) -> dict:
    """
    Loads ellipse data from a CSV, handles NaN values gracefully, parses string-formatted 
    tuples (including non-standard '(nan, nan)'), converts numeric data to floats, 
    splits tuple columns into components, and organizes the data into a dictionary 
    of clean DataFrames for each sticker color.

    This revised version ensures that rows with missing tuple data (e.g., 'center') are 
    not dropped. Instead, NaN values are propagated to the resulting split columns 
    (e.g., 'center_x', 'center_y').

    Args:
        csv_path (str): The path to the CSV file with ellipse data.
        sticker_colors (list[str]): A list of the sticker colors to process.

    Returns:
        dict: A dictionary where keys are sticker colors and values are the
              corresponding cleaned DataFrames with split 'center' and 'axes' columns.
    """
    # Helper function 1: Safely parse string literals, now with 'nan' handling.
    def safe_literal_eval(val):
        if isinstance(val, str):
            # Pre-process the string to handle non-standard 'nan' values
            # by replacing them with 'None', which ast.literal_eval can parse.
            processed_val = val.replace('nan', 'None')
            try:
                return ast.literal_eval(processed_val)
            except (ValueError, SyntaxError):
                return val  # Return original string if it's not a valid literal
        return val  # Return original value if not a string

    # Helper function 2: Recursively convert all numeric types to float.
    def to_float_recursive(val):
        # Explicitly handle NaN/None values first to avoid errors
        if pd.isna(val):
            return np.nan
        try:
            if isinstance(val, (list, tuple)):
                return tuple(to_float_recursive(item) for item in val)
            else:
                return float(val)
        except (ValueError, TypeError):
            return val

    # Helper function 3: Split a column of tuples into multiple columns.
    def split_tuple_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Splits a DataFrame column containing tuples into separate columns."""
        if col_name not in df.columns:
            return df
        
        # This approach gracefully handles NaNs.
        split_data = pd.DataFrame(df[col_name].tolist(), index=df.index)
        
        # Create new column names like 'center_x', 'center_y'
        num_cols = len(split_data.columns)
        new_col_names = [f"{col_name}_{axis}" for axis in ('x', 'y', 'z')[:num_cols]]
        split_data.columns = new_col_names
        
        # Drop the original tuple column and join the new split columns
        df = df.drop(columns=[col_name])
        df = df.join(split_data)
        
        return df

    # 1. Read the CSV data.
    df = pd.read_csv(csv_path)

    # 2. Apply safe_literal_eval to convert strings like '(1,2)' and '(nan,nan)' to tuples.
    df = df.applymap(safe_literal_eval)

    # 3. Apply to_float_recursive to ensure all numbers are floats and handle NaNs.
    df = df.applymap(to_float_recursive)

    column_suffixes = ['frame_number', 'center', 'axes', 'angle', 'score', 'optimal_threshold']
    
    sticker_dataframes = {}
    for color in sticker_colors:
        rename_map = {f'{color}_{suffix}': suffix for suffix in column_suffixes}
        cols_for_color = list(rename_map.keys())
        
        if not all(col in df.columns for col in cols_for_color):
            continue
            
        sub_df = df[cols_for_color].rename(columns=rename_map).copy()
        
        # 4. Use the helper to split 'center' and 'axes' columns
        sub_df = split_tuple_column(sub_df, 'center')
        sub_df = split_tuple_column(sub_df, 'axes')

        sticker_dataframes[color] = sub_df
    
    return sticker_dataframes

def _parse_roi_string(value: Any) -> list:
    """
    Helper function to safely parse a string or handle non-string/NaN values.
    """
    # 1. Check if the value is NaN, None, or not a string.
    if pd.isna(value) or not isinstance(value, str):
        # 2. Return a list of NaNs to maintain the DataFrame structure.
        return [np.nan, np.nan, np.nan, np.nan]
    
    try:
        # 3. If it's a string, attempt to parse it.
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # Fallback for strings that are not valid lists (e.g., "invalid_data")
        return [np.nan, np.nan, np.nan, np.nan]

def load_sticker_roi_dataframes(csv_path: str) -> Dict[str, pd.DataFrame]:
    """
    Processes a CSV file where cells can be list-like strings '[x, y, w, h]' or NaN.
    It transforms the data into a dictionary of structured DataFrames.

    Args:
        csv_path (str): The path to the input CSV file.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are original column names
                                 and values are DataFrames with columns
                                 ['x', 'y', 'width', 'height']. Missing or invalid
                                 entries are represented as rows with NaN values.
    """
    try:
        main_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file at '{csv_path}' was not found.")
        return {}
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return {}

    roi_dataframes: Dict[str, pd.DataFrame] = {}
    new_columns = ['x', 'y', 'width', 'height']

    for column_name in main_df.columns:
        # Apply our new, safer parsing function to the entire column.
        parsed_series = main_df[column_name].apply(_parse_roi_string)

        # Create the sub-DataFrame. Rows with NaNs will be handled correctly.
        sub_df = pd.DataFrame(parsed_series.tolist(), columns=new_columns)

        roi_dataframes[column_name] = sub_df
            
    return roi_dataframes


def load_video_to_array(video_path: str) -> Optional[Tuple[List[np.ndarray], float, int, int]]:
    """
    Loads an entire video file into a list of frames in memory.

    ⚠️ This is highly memory-intensive and is not recommended for long or
    high-resolution videos. Consider a frame-by-frame processing approach for scalability.

    Args:
        video_path (str): The path to the source video file.

    Returns:
        A tuple containing (list of frames, fps, width, height), or None on error.
    """
    print(f"Loading video '{video_path}' into memory...")
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at '{video_path}'")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file '{video_path}'")
        return None
    
    # Get video properties for later use
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # The actual frame loading is delegated to the utility function
    frames = load_video_frames_bgr(video_path)

    print(f"✅ Video loaded: {len(frames)} frames, {width}x{height} @ {fps:.2f} FPS.")
    return frames, fps, width, height


# --- Pre-processing functions  ---
def adjust_ellipses_coord_to_full_frame(
    detected_ellipses: Dict[Any, pd.DataFrame],
    roi_dict: Dict[Any, pd.DataFrame],
    objects_to_track: List[Any],
    *,
    # Adjust these column names if needed
    roi_x_col: str = 'x',
    roi_y_col: str = 'y',
    ellipse_x_col: str = 'center_x',
    ellipse_y_col: str = 'center_y'
) -> Dict[Any, pd.DataFrame]:
    """
    Adjusts the coordinates of detected ellipses from ROI-local to full-frame.

    This function takes dictionaries of DataFrames containing ellipse and ROI
    information and translates the ellipse coordinates to be relative to the
    full video frame instead of the local ROI. It correctly handles NaN values,
    propagating them to the result (e.g., a + NaN = NaN).

    Args:
        detected_ellipses: A dictionary where keys are object IDs and values are
                           DataFrames of detected ellipses. These DataFrames must
                           contain 'center_x' and 'center_y' columns.
        roi_dict: A dictionary where keys are object IDs and values are DataFrames
                  of ROI information. These DataFrames must contain 'x' and 'y'
                  columns for the ROI's top-left corner.
        objects_to_track: A list of object IDs to process.
        roi_x_col: Column name for the ROI's top-left x-coordinate.
        roi_y_col: Column name for the ROI's top-left y-coordinate.
        ellipse_x_col: Column name for the ellipse's center x-coordinate.
        ellipse_y_col: Column name for the ellipse's center y-coordinate.

    Returns:
        A new dictionary with the same structure as detected_ellipses, but with
        the 'center_x' and 'center_y' coordinates adjusted to the full frame.
    """
    # Create a deep copy to avoid modifying the original data
    adj_ellipses = copy.deepcopy(detected_ellipses)

    for obj_id in objects_to_track:
        if obj_id not in adj_ellipses or obj_id not in roi_dict:
            print(f"Warning: Object ID {obj_id} not found in both dictionaries. Skipping.")
            continue

        # Get the corresponding DataFrames for the current object
        ellipses_df = adj_ellipses[obj_id]
        rois_df = roi_dict[obj_id]

        # --- Column Validation ---
        required_roi_cols = {roi_x_col, roi_y_col}
        if not required_roi_cols.issubset(rois_df.columns):
            raise KeyError(f"Missing required ROI columns {required_roi_cols - set(rois_df.columns)} for object {obj_id}")
        
        required_ellipse_cols = {ellipse_x_col, ellipse_y_col}
        if not required_ellipse_cols.issubset(ellipses_df.columns):
            raise KeyError(f"Missing required ellipse columns {required_ellipse_cols - set(ellips_df.columns)} for object {obj_id}")

        # --- Coordinate Adjustment ---
        # Add the ROI's top-left coordinates to the ellipse's center coordinates.
        # This works because the indices of both DataFrames are aligned by frame number.
        # Pandas automatically handles NaN propagation.
        ellipses_df[ellipse_x_col] += rois_df[roi_x_col]
        ellipses_df[ellipse_y_col] += rois_df[roi_y_col]

    return adj_ellipses

# --- 7. Core Processing Functions ---
def draw_frame_annotations(
    frame: np.ndarray, 
    annotations: List[Dict[str, Any]], 
    color_map: Dict[str, Tuple[int, int, int]]
) -> np.ndarray:
    """
    Draws all annotations (ellipses and labels) for a single frame, handling cases
    where detection data is missing or invalid (NaN/inf).

    Args:
        frame (np.ndarray): The frame to draw on.
        annotations (List[Dict[str, Any]]): A list of detection dictionaries for this frame.
        color_map (Dict[str, Tuple[int, int, int]]): Maps object names to RGB colors.

    Returns:
        np.ndarray: The frame with annotations drawn on it.
    """
    # Y-coordinate for placing warning messages, starts 30 pixels from the top.
    warning_y_offset = 30

    for detection in annotations:
        object_name = detection['object_name']
        
        # --- Data Validation Step ---
        # Extract all values that will be used for drawing
        center_x = detection.get('center_x')
        center_y = detection.get('center_y')
        axes_major = detection.get('axes_major')
        axes_minor = detection.get('axes_minor')
        angle = detection.get('angle')

        # Check if any required numeric value is None, NaN, or infinity
        # This is a robust way to catch invalid data points
        is_invalid = any(
            val is None or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val)
            for val in [center_x, center_y, axes_major, axes_minor, angle]
        )
        
        if is_invalid:
            # --- Handle Invalid Data ---
            warning_text = f"'{object_name}': Data is missing or invalid"
            position = (15, warning_y_offset)
            
            # Put a red warning text on the frame
            cv2.putText(frame, warning_text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Increment the y-offset to prevent future warnings from overlapping
            warning_y_offset += 25
            
            # Skip to the next detection in the list
            continue

        # --- Draw Valid Detection ---
        # If the code reaches here, the data is valid and can be converted to int
        center = (int(center_x), int(center_y))
        axes = (int(axes_major / 2), int(axes_minor / 2))
        
        try:
            rgb_color = color_map[object_name]["final"]
        except KeyError:
            rgb_color = (255, 255, 255) # Default to white
        
        # Convert RGB to BGR for OpenCV
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0]) 
        
        cv2.ellipse(frame, center, axes, int(angle), 0, 360, bgr_color, thickness=2)
        label_position = (center[0] - 20, center[1] - axes[1] - 10)
        cv2.putText(frame, object_name, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2)
        
    return frame


def annotate_video_frames(
    video_frames: List[np.ndarray],
    detections_by_frame: Dict[int, List[Dict[str, Any]]],
    color_map: Dict[str, Tuple[int, int, int]]
) -> List[np.ndarray]:
    """
    Processes a list of frames to draw all annotations, returning the result in memory.

    This "pure" function takes frame data and annotations and returns a new list of
    annotated frames without causing side effects like writing files.

    Args:
        video_frames: The list of original video frames (as NumPy arrays).
        detections_by_frame: A dictionary mapping frame numbers to annotations.
        color_map: A dictionary mapping object names to RGB colors.

    Returns:
        A list of annotated video frames (as NumPy arrays).
    """
    annotated_frames = []
    total_frames = len(video_frames)
    print("\n🎨 Processing frames to create annotated video in memory...")

    for frame_index, frame in enumerate(video_frames):
        # Create a copy to ensure the original frame data is not modified.
        annotated_frame = frame.copy()
        
        if frame_index in detections_by_frame:
            annotations = detections_by_frame[frame_index]
            draw_frame_annotations(annotated_frame, annotations, color_map)
        
        annotated_frames.append(annotated_frame)
        print(f"\rProcessing frame {frame_index + 1}/{total_frames}...", end="")

    print("\n✅ In-memory annotation complete.")
    return annotated_frames






def define_colorspaces_for_reannotation(
    video_frames: List[np.ndarray],
    frame_ids_to_fix: List[int],
    objects_to_track: List[str],
    color_config: Dict[str, Any],
    dest_metadata_path: str
) -> None:
    """
    Processes frames marked for re-annotation to define new colorspaces for objects.

    For each object, it extracts the relevant ROI from the marked frames,
    defines a new colorspace, adjusts its coordinates to the full frame, and
    updates the destination metadata JSON file.

    Args:
        video_frames: The list of all original video frames.
        frame_ids_to_fix: The integer indices of frames that need correction.
        tracked_obj_df: DataFrame containing all tracking data.
        objects_to_track: A list of object names to process.
        color_config: Configuration for colors used in processing.
        dest_metadata_path: Path to the JSON metadata file to be updated.
    """
    print("\n🔧 Defining new colorspaces based on user corrections...")
    for object_name in objects_to_track:
        print(f"  -> Processing object: {object_name}")
        
        # Use only the frames and ROIs marked by the user for correction
        selected_frames = [video_frames[i] for i in frame_ids_to_fix]
        colors = color_config.get(object_name)
        
        # Define colorspaces with coordinates relative to the cropped ROI
        colorspaces = run_colorspace_definition_tool(selected_frames, colors)
        
        # Prepare final payload and save to the metadata file
        json_payload = prepare_json_update_payload(frame_ids_to_fix, colorspaces)
        update_json_object(dest_metadata_path, object_name, json_payload, overwrite=False)
        print(f"  -> Successfully updated colorspace for {object_name}.")


def save_frames_to_video(
    output_path: str,
    frames: List[np.ndarray],
    fps: float,
    frame_size: Tuple[int, int]
) -> bool:
    """
    Saves a list of frames to an MP4 video file.

    Args:
        output_path (str): The path where the output video will be saved (e.g., 'output.mp4').
        frames (List[np.ndarray]): The list of frames (as NumPy arrays) to be written.
        fps (float): The frames per second for the output video.
        frame_size (Tuple[int, int]): The dimensions of the video (width, height).

    Returns:
        bool: True if the video was saved successfully, False otherwise.
    """
    if not frames:
        print("⚠️ Warning: No frames provided to save. Video creation skipped.")
        return False

    print(f"\n💾 Saving annotated video to '{output_path}'...")
    
    # Define the codec and create VideoWriter object. 'mp4v' is a good codec for .mp4 files.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    if not writer.isOpened():
        print(f"❌ Error: Could not open video writer for path '{output_path}'.")
        return False

    try:
        total_frames = len(frames)
        for i, frame in enumerate(frames):
            writer.write(frame)
            # Print progress without creating a new line each time
            print(f"\rWriting frame {i + 1}/{total_frames}...", end="")
        
        print("\n✅ Video saved successfully.")
        return True
    except Exception as e:
        print(f"\n❌ An error occurred during video writing: {e}")
        return False
    finally:
        # Crucial: Release the writer to finalize the video file.
        writer.release()

# --- 8. Main Orchestration Functions ---
def save_tracked_handstickers_position_as_video(
    video_path: str,
    tracking_pos_csv_path: str,
    metadata_path: str,
    output_video_path: str
):
    """
    Main orchestration function to review, validate, and correct object tracking data.

    It loads video and tracking data, allows a user to review the annotated video,
    and if corrections are made, it processes them and updates the metadata.
    If an output_video_path is provided, it saves the annotated video.

    Args:
        video_path: Absolute path to the source video file.
        tracking_pos_csv_path: Absolute path to the CSV file with tracking data.
        metadata_path: Absolute path to the JSON metadata file to be read and updated.
        output_video_path (Optional[str]): Path to save the annotated MP4 video.
                                           If None, the video is not saved.
    """
    print(f"--- Starting Tracking Review for: '{os.path.basename(video_path)}' ---")

    # 1. Pre-flight check: Skip if already successfully processed
    if os.path.exists(get_success_flag_path(tracking_pos_csv_path)):
        print(f"✅ Success flag found. Skipping processing for '{os.path.basename(tracking_pos_csv_path)}'.")
        return tracking_pos_csv_path

    if not os.path.exists(metadata_path):
        print(f"⚠️ Metadata file not found at '{metadata_path}'. Skipping.")
        return

    # 2. Data Loading
    video_data = load_video_to_array(video_path)
    if not video_data: return
    frames, fps, width, height = video_data

    detections = load_ellipses_from_csv(tracking_pos_csv_path)
    if not detections: return

    # 3. Load Metadata and Configuration
    try:
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"❌ Error loading metadata: {e}. Halting.")
        return

    objects_to_track = list(metadata.keys())
    color_config = generate_color_config(objects_to_track)

    # 4. In-Memory Annotation
    annotated_frames = annotate_video_frames(
        video_frames=frames,
        detections_by_frame=detections,
        color_map=color_config
    )
    
    save_frames_to_video(
        output_path=output_video_path,
        frames=annotated_frames,
        fps=fps,
        frame_size=(width, height)
    )
        
    return output_video_path


def review_tracked_handstickers_position(
    roi_unified_csv_path: str,
    ellipses_csv_path: str,
    metadata_path: str,
    output_csv_base: Optional[str] = None
):
    """
    Main orchestration function to review, validate, and correct object tracking data.

    It loads video and tracking data, allows a user to review the annotated video,
    and if corrections are made, it processes them and updates the metadata.
    If an output_video_path is provided, it saves the annotated video.

    Args:
        video_path: Absolute path to the source video file.
        tracking_pos_csv_path: Absolute path to the CSV file with tracking data.
        metadata_path: Absolute path to the JSON metadata file to be read and updated.
    """
    # 1. Pre-check: Skip if already successfully processed
    if os.path.exists(get_success_flag_path(ellipses_csv_path)):
        print(f"✅ Success flag found. Skipping processing for '{os.path.basename(ellipses_csv_path)}'.")
        return ellipses_csv_path

    if not os.path.exists(metadata_path):
        print(f"⚠️ Metadata file not found at '{metadata_path}'. Skipping.")
        return

    # 3. Load Metadata and Configuration
    try:
        colorspaces = ColorspaceFileHandler(metadata_path)
        objects_to_track = colorspaces.get_object_names()
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"❌ Error loading metadata: {e}. Halting.")
        return


    # 2. Data Loading
    roi_dict = load_sticker_roi_dataframes(roi_unified_csv_path)
    if roi_dict is None:
        return

    detected_ellipses = load_sticker_ellipse_dataframes(ellipses_csv_path, objects_to_track)
    if not detected_ellipses: 
        return

    adj_ellipses = adjust_ellipses_coord_to_full_frame(
        detected_ellipses, 
        roi_dict, 
        objects_to_track)

    
    video_data = load_video_to_array(video_path)
    if not video_data: return
    frames, fps, width, height = video_data


    color_config = generate_color_config(objects_to_track)

    if output_video_path:
        save_frames_to_video(
            output_path=output_video_path,
            frames=annotated_frames,
            fps=fps,
            frame_size=(width, height)
        )
    
    # 4. In-Memory Annotation
    annotated_frames = annotate_video_frames(
        video_frames=frames,
        detections_by_frame=detected_ellipses,
        color_map=color_config
    )

    # 6. Interactive User Review
    final_status, marked_frame_indices = review_frames(annotated_frames)

    # 7. Process Review Results
    if final_status == Status.VALID.value:
        print("✅ Tracking marked as valid by user.")
        create_completion_success_flag(ellipses_csv_path)
    elif marked_frame_indices:
        print(f"⚠️ User marked {len(marked_frame_indices)} frames for re-annotation. Launching correction process.")
        
        define_colorspaces_for_reannotation(
            video_frames=frames,
            frame_ids_to_fix=marked_frame_indices,
            objects_to_track=objects_to_track,
            color_config=color_config,
            dest_metadata_path=metadata_path
        )
        print("\nRe-annotation complete. Please run the process again to verify the new tracking.")
    else:
        print("🟡 Review session ended without validation or marking new frames.")
    
    print(f"--- Finished Tracking Review for: '{os.path.basename(video_path)}' ---")
    return ellipses_csv_path

