# --- 1. Standard Library Imports ---
import ast 
import json
import math
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import copy
import sys

# --- 2. Third-party Imports ---
import cv2
import numpy as np
import pandas as pd

# --- 3. Local Application Imports ---
from .utils.roi_to_position_funcs import (
    generate_color_config,
    load_video_frames_bgr
)

from .utils.colorspace.ColorspaceFileHandler import ColorspaceFileHandler


# --- 4. Constants and Enums ---
def load_and_reconstruct_dataframes(csv_path: str, prefixes: list) -> dict:
    """
    Loads a CSV file created by merge_and_prefix_dataframes and reconstructs
    the original dictionary of DataFrames using a given list of prefixes.

    Args:
        csv_path: Path to the merged CSV file.
        prefixes: A list of strings representing the original dictionary keys.

    Returns:
        A dictionary of pandas DataFrames, reconstructed from the CSV.
    """
    if not os.path.exists(csv_path):
        print(f"❌ File not found at '{csv_path}'.")
        return {}
        
    merged_df = pd.read_csv(csv_path)
    dataframe_dict = {}
    
    # Iterate through the provided list of prefixes
    for prefix in prefixes:
        # Select columns belonging to the current prefix
        prefix_cols = [col for col in merged_df.columns if col.startswith(f"{prefix}_")]
        
        if not prefix_cols:
            print(f"⚠️ No columns found for prefix '{prefix}' in '{csv_path}'.")
            continue
            
        # Create a new dataframe for the current prefix
        sub_df = merged_df[prefix_cols].copy()
        
        # Remove the prefix from the column names
        sub_df.columns = [col.replace(f"{prefix}_", '', 1) for col in sub_df.columns]
        
        dataframe_dict[prefix] = sub_df
        
    return dataframe_dict


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
    frames = _load_video_frames(video_path)

    print(f"✅ Video loaded: {len(frames)} frames, {width}x{height} @ {fps:.2f} FPS.")
    return frames, fps, width, height

def _load_video_frames(video_path, as_bgr=True):
    """
    Loads all frames from the video into a list of images with progress feedback,
    respecting the color format set during initialization.
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise IOError(f"Error: Could not open video file at {video_path}")

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print("Warning: Video contains no frames.")
        return []

    frames = []
    processed_frames = 0
    print("Starting video processing...")

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        
        # Check the instance's color format property
        if as_bgr:
            # Append the frame as is (BGR)
            frames.append(frame)
        else:
            # Convert the frame from BGR to RGB and then append
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        processed_frames += 1

        # Calculate and display the progress
        percentage = (processed_frames / total_frames) * 100
        sys.stdout.write(f"\rLoading frames: {processed_frames}/{total_frames} frames ({percentage:.2f}%)")
        sys.stdout.flush()

    video_capture.release()
    print("\nVideo processing complete.")
    return frames


# --- Pre-processing functions  ---
def is_valid_ellipse(ellipse: pd.Series):
    center_x = ellipse['center_x']
    center_y = ellipse['center_y']
    axes_major = ellipse['axes_major']
    axes_minor = ellipse['axes_minor']
    angle = ellipse['angle']

    is_invalid = any(
        val is None or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val)
        for val in [center_x, center_y, axes_major, axes_minor, angle]
    )
    return not is_invalid

# --- 7. Core Processing Functions ---
def draw_frame_annotations(
    frame: np.ndarray, 
    ellipse: pd.Series, 
    color_map: Tuple[int, int, int], 
    object_name: str
) -> np.ndarray:
    """
    Draws all annotations (ellipses and labels) for a single frame, handling cases
    where detection data is missing or invalid (NaN/inf).

    Args:
        frame (np.ndarray): The frame to draw on.
        annotations pd.Series: A list of detection dictionaries for this frame.
        color_map (Dict[str, Tuple[int, int, int]]): Maps object names to RGB colors.

    Returns:
        np.ndarray: The frame with annotations drawn on it.
    """
    # Y-coordinate for placing warning messages, starts 30 pixels from the top.
    warning_y_offset = 30

    # --- Data Validation Step ---
    if is_valid_ellipse(ellipse):
        # Extract all values that will be used for drawing
        center_x = ellipse['center_x']
        center_y = ellipse['center_y']
        axes_major = ellipse['axes_major']
        axes_minor = ellipse['axes_minor']
        angle = ellipse['angle']

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
            
    else:
        # --- Handle Invalid Data ---
        warning_text = f"'{object_name}': Data is missing or invalid"
        position = (15, warning_y_offset)
        
        # Put a red warning text on the frame
        cv2.putText(frame, warning_text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Increment the y-offset to prevent future warnings from overlapping
        warning_y_offset += 25
        
    return frame

def split_ellipse_data(ellipses_row: pd.Series, objects_to_track: list[str]) -> dict[str, pd.Series]:
    """
    Splits a single series containing data for multiple objects into a dictionary of series.

    Each key in the output dictionary corresponds to an object prefix from objects_to_track,
    and its value is a series containing the data for that object with the prefix stripped
    from the index.

    Args:
        ellipses_row (pd.Series): A series where the index is formatted as
                                  'prefix_attribute' (e.g., 'sticker_yellow_center_x').
        objects_to_track (list[str]): A list of prefixes to identify and split the data by
                                      (e.g., ['sticker_yellow', 'sticker_blue']).

    Returns:
        dict[str, pd.Series]: A dictionary where each key is an object prefix and each value
                              is a pd.Series of its corresponding data.
    """
    ellipses_by_color = {}
    for prefix in objects_to_track:
        # 1. Filter the series to get rows for the current prefix
        # We add a '_' to ensure we match the full prefix exactly
        prefix_with_underscore = f"{prefix}_"
        object_data = ellipses_row[ellipses_row.index.str.startswith(prefix_with_underscore)]

        # 2. If data was found, clean the index and store it
        if not object_data.empty:
            # Create a new index by removing the prefix
            new_index = object_data.index.str.replace(prefix_with_underscore, '', n=1)
            object_data.index = new_index
            ellipses_by_color[prefix] = object_data
            
    return ellipses_by_color

def annotate_video_frames(
    video_frames: List[np.ndarray],
    objects_to_track: List[str],
    ellipses_df: pd.DataFrame,
    color_map: Dict[str, Any]
) -> List[np.ndarray]:
    """
    Processes a list of frames to draw all annotations from a DataFrame.

    This function assumes a 1-to-1 relationship where the row at index `i` in the
    DataFrame corresponds to the frame at index `i` in the video_frames list.

    Args:
        video_frames (List[np.ndarray]): The list of original video frames.
        ellipses_df (pd.DataFrame): DataFrame where each row corresponds to a frame.
        color_map (Dict[str, Any]): Maps object names to colors.

    Returns:
        List[np.ndarray]: A new list of annotated video frames.
    """
    # --- Input Validation ---
    if ellipses_df.empty:
        print("\n⚠️  Warning: Detections DataFrame is empty. Returning original frames.")
        return video_frames

    if len(video_frames) != len(ellipses_df):
        print(f"\n❌ Error: Frame count ({len(video_frames)}) and detection row count "
              f"({len(ellipses_df)}) do not match. Cannot proceed.")
        return video_frames

    annotated_frames = []
    total_frames = len(video_frames)
    print("\n🎨 Processing frames to create annotated video in memory...")

    # Iterate through each frame and its corresponding detection data row
    for frame_index, frame in enumerate(video_frames):
        # Create a copy to ensure the original frame data is not modified.
        annotated_frame = frame.copy()
        
        # Get the corresponding row of detection data using its index.
        ellipses_row = ellipses_df.iloc[frame_index]
        ellipses_by_object = split_ellipse_data(ellipses_row, objects_to_track)

        for (sticker_name, ellipse) in ellipses_by_object.items():
            if is_valid_ellipse(ellipse):
                color = color_map[sticker_name]
                draw_frame_annotations(annotated_frame, ellipse, color, sticker_name)
            
        annotated_frames.append(annotated_frame)
        print(f"\rProcessing frame {frame_index + 1}/{total_frames}...", end="")

    print("\n✅ In-memory annotation complete.")
    return annotated_frames


# --- Main Processing Functions ---
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

def process_video_frame_by_frame(
    video_path: str,
    output_path: str,
    ellipses_df: pd.DataFrame,
    objects_to_track: List[str],
    color_map: Dict[str, Any]
) -> bool:
    """
    Reads a video, annotates it, and saves it by processing one frame at a time.

    This memory-efficient function opens a video stream for reading and another for
    writing. It iterates through each frame, applies annotations based on the
    provided DataFrame, and writes the result directly to the output file,
    avoiding loading the entire video into memory.

    Args:
        video_path (str): The path to the source video file.
        output_path (str): The path where the annotated video will be saved.
        ellipses_df (pd.DataFrame): DataFrame where each row corresponds to a frame's
                                    annotation data.
        objects_to_track (List[str]): List of object names to find in the DataFrame.
        color_map (Dict[str, Any]): A dictionary mapping object names to their colors.

    Returns:
        bool: True if processing was successful, False otherwise.
    """
    # 1. --- Initialization and Validation ---
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at '{video_path}'")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file '{video_path}'")
        return False

    # Get video properties from the source
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Critical validation: ensure data matches video length
    if total_frames != len(ellipses_df):
        print(f"❌ Error: Frame count in video ({total_frames}) does not match "
              f"row count in DataFrame ({len(ellipses_df)}).")
        cap.release()
        return False

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"❌ Error: Could not open video writer for path '{output_path}'.")
        cap.release()
        return False
        
    print(f"🚀 Starting video processing: '{video_path}' -> '{output_path}'")
    print(f"Video details: {total_frames} frames, {width}x{height} @ {fps:.2f} FPS.")

    # 2. --- Frame-by-Frame Processing Loop ---
    frame_index = 0
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break # End of video

            # Get the corresponding row of detection data
            ellipses_row = ellipses_df.iloc[frame_index]
            ellipses_by_object = split_ellipse_data(ellipses_row, objects_to_track)

            # Annotate the current frame
            for sticker_name, ellipse in ellipses_by_object.items():
                if is_valid_ellipse(ellipse):
                    color = color_map[sticker_name]
                    draw_frame_annotations(frame, ellipse, color, sticker_name)
            
            # Write the annotated frame to the output file
            writer.write(frame)

            # Update progress
            percentage = ((frame_index + 1) / total_frames) * 100
            sys.stdout.write(f"\rProcessing frame {frame_index + 1}/{total_frames} ({percentage:.2f}%)")
            sys.stdout.flush()
            
            frame_index += 1
            
    except Exception as e:
        print(f"\n❌ An error occurred during processing: {e}")
        return False
    finally:
        # 3. --- Cleanup ---
        cap.release()
        writer.release()
        print("\n✅ Processing complete. Video saved.")

    return True

# --- 8. Main Orchestration Functions ---
def generate_ellipses_on_frame_video(
    video_path: str,
    ellipses_adj_path: str,
    metadata_path: str,
    output_path: str,
    process_by_frame: bool = True
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
    if os.path.exists(output_path):
        print(f"✅ Video file found. Skipping processing for '{os.path.basename(video_path)}'.")
        return output_path

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
    if not os.path.exists(ellipses_adj_path):
        print(f"❌ File not found at '{ellipses_adj_path}'.")
        return {}
    ellipses = pd.read_csv(ellipses_adj_path)
    if ellipses is None: 
        return
    
    color_config = generate_color_config(objects_to_track)
    
    if process_by_frame:
        success = process_video_frame_by_frame(
            video_path=video_path,
            output_path=output_path,
            ellipses_df=ellipses,
            objects_to_track=objects_to_track,
            color_map=color_config
        )
        
        if success:
            print("Workflow completed successfully.")
        else:
            print("Workflow failed.")
    else:
        video_data = load_video_to_array(video_path)
        if not video_data: return
        frames, fps, width, height = video_data


        # 4. In-Memory Annotation
        annotated_frames = annotate_video_frames(
            video_frames=frames,
            objects_to_track=objects_to_track,
            ellipses_df=ellipses,
            color_map=color_config
        )

        save_frames_to_video(
            output_path=output_path,
            frames=annotated_frames,
            fps=fps,
            frame_size=(width, height)
        )

    return output_path

