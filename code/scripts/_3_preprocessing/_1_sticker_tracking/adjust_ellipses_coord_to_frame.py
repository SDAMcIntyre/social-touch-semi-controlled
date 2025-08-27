# --- 1. Standard Library Imports ---
import ast
import copy
import json
import os
from typing import Any, Dict, List

# --- 2. Third-party Imports ---
import numpy as np
import pandas as pd

from .utils.colorspace.ColorspaceFileHandler import ColorspaceFileHandler

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
            processed_val = val
            processed_val = processed_val.replace('nan', 'None')
            processed_val = processed_val.replace('-inf', 'None')
            processed_val = processed_val.replace('inf', 'None')
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
        try:
            split_data.columns = new_col_names
        except ValueError as e:
            mask = df[col_name].str.len() > 2
            filtered_df = df[mask]
            print(f"--- Rows where length of '{col_name}' > {2} ---")
            print(filtered_df[col_name])
            print("-------------------------------------------------\n")
            raise RuntimeError("Failed to process DataFrame due to column mismatch.")
        
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

    column_suffixes = [
        'frame_number', 
        'center_x', 'center_y', 
        'axes_major', 'axes_minor',
        'angle', 'score', 'optimal_threshold']
    
    sticker_dataframes = {}
    for color in sticker_colors:
        rename_map = {f'{color}_{suffix}': suffix for suffix in column_suffixes}
        cols_for_color = list(rename_map.keys())
        
        if not all(col in df.columns for col in cols_for_color):
            continue
        
        sub_df = df[cols_for_color].rename(columns=rename_map).copy()
        
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


def merge_and_prefix_dataframes(dataframe_dict: dict) -> pd.DataFrame:
    """
    Merges a dictionary of pandas DataFrames into a single DataFrame,
    prefixing each column with its corresponding dictionary key.

    Args:
        dataframe_dict: A dictionary where keys are strings (used as prefixes)
                        and values are pandas DataFrames.

    Returns:
        A single merged pandas DataFrame with prefixed columns.
    """
    prefixed_dfs = []
    for name, df in dataframe_dict.items():
        # Create a copy to avoid modifying the original DataFrame in the dictionary
        df_copy = df.copy()
        
        # Add the prefix to each column name
        df_copy.columns = [f"{name}_{col}" for col in df_copy.columns]
        
        prefixed_dfs.append(df_copy)

    if not prefixed_dfs:
        return pd.DataFrame()

    # Concatenate all the dataframes horizontally
    # This assumes that all DataFrames are aligned by their index (e.g., frame number)
    merged_df = pd.concat(prefixed_dfs, axis=1)
    
    return merged_df

# --- Pre-processing functions  ---
def _adjust_ellipses_coord_to_full_frame(
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
            raise KeyError(f"Missing required ellipse columns {required_ellipse_cols - set(ellipses_df.columns)} for object {obj_id}")

        # --- Coordinate Adjustment ---
        # Add the ROI's top-left coordinates to the ellipse's center coordinates.
        # This works because the indices of both DataFrames are aligned by frame number.
        # Pandas automatically handles NaN propagation.
        ellipses_df[ellipse_x_col] += rois_df[roi_x_col]
        ellipses_df[ellipse_y_col] += rois_df[roi_y_col]

    return adj_ellipses

# --- 8. Main Orchestration Functions ---
def adjust_ellipses_coord_to_frame(
    roi_unified_csv_path: str,
    ellipses_csv_path: str,
    metadata_path: str,
    output_path: str
):
    """
    Main orchestration function to review, validate, and correct object tracking data.

    It loads video and tracking data, allows a user to review the annotated video,
    and if corrections are made, it processes them and updates the metadata, and 
    saves the processed dataframes to a directory.

    Args:
        roi_unified_csv_path: Path to the unified ROI data.
        ellipses_csv_path: Path to the detected ellipses data.
        metadata_path: Path to the JSON metadata file.
        output_csv_base: base path for saving output files.
                         The final output will be a directory derived from this base.
    """
    # 1. Pre-check: Skip if already successfully processed
    if False and os.path.exists(output_path):
        print(f"✅ File found. Skipping processing for '{os.path.basename(output_path)}'.")
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
    roi_dict = load_sticker_roi_dataframes(roi_unified_csv_path)
    if roi_dict is None:
        return

    detected_ellipses = load_sticker_ellipse_dataframes(ellipses_csv_path, objects_to_track)
    if not detected_ellipses: 
        return

    adj_ellipses = _adjust_ellipses_coord_to_full_frame(
        detected_ellipses, 
        roi_dict, 
        objects_to_track)
    
    if adj_ellipses:
        merged_adj_ellipses = merge_and_prefix_dataframes(adj_ellipses)
        # Create the directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)
        print(f"\n💾 Saving adjusted ellipses to directory: '{output_path.parent}'")
        merged_adj_ellipses.to_csv(output_path, index=False)
        
    # This new directory path could be returned as a confirmation of output
    return output_path

