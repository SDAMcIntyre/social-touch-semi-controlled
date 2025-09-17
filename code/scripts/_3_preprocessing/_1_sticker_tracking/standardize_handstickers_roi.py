# Standard library imports
import os
import pandas as pd
from typing import Tuple

# Third-party imports
from preprocessing.stickers_analysis import (
    ROITrackedFileHandler
)
from utils.should_process_task import should_process_task

# --- Helper & Core Logic Functions ---

def get_max_roi_values(tracked_rois: pd.DataFrame) -> Tuple[int, int, int, int]:
    """Calculates the bounding box that encompasses all ROIs in a DataFrame."""
    roi_cols = ['roi_x', 'roi_y', 'roi_width', 'roi_height']
    if tracked_rois.empty or not all(col in tracked_rois.columns for col in roi_cols):
        return (0, 0, 0, 0)

    # Calculate max, fill potential NaN columns with 0, convert to int, and return as tuple
    max_values = tracked_rois[roi_cols].max().fillna(0).astype(int)
    return tuple(max_values)


def standardise_roi_size(tracked_rois: pd.DataFrame) -> pd.DataFrame:
    """
    Standardises all ROIs in a DataFrame to the maximum dimensions found,
    preserving the original center of each ROI.
    """
    _, _, max_w, max_h = get_max_roi_values(tracked_rois)

    # If no valid dimensions exist, there's nothing to standardise.
    if max_w == 0 or max_h == 0:
        return tracked_rois.copy()

    df = tracked_rois.copy()

    # Identify rows with valid ROI data to avoid errors on NaN values.
    valid_rois = df['roi_width'].notna() & df['roi_height'].notna()
    
    # Use vectorized operations on the valid subset for efficiency.
    # Calculate original centers
    center_x = df.loc[valid_rois, 'roi_x'] + df.loc[valid_rois, 'roi_width'] / 2
    center_y = df.loc[valid_rois, 'roi_y'] + df.loc[valid_rois, 'roi_height'] / 2

    # Update ROI position and dimensions based on the new standard size
    df.loc[valid_rois, 'roi_x'] = (center_x - max_w / 2).astype(int)
    df.loc[valid_rois, 'roi_y'] = (center_y - max_h / 2).astype(int)
    df.loc[valid_rois, 'roi_width'] = max_w
    df.loc[valid_rois, 'roi_height'] = max_h

    return df

# --- Main Orchestration Function ---

def generate_standard_roi_size_dataset(
    roi_path: str,
    output_roi_path: str,
    *,
    force_processing: bool = False
) -> None:
    """
    Loads ROI tracking data, standardises ROI sizes for each tracked object,
    and saves the result to a new file.
    """
    if not should_process_task(
        output_paths=output_roi_path,
        input_paths=roi_path,
        force=force_processing
    ):
        print(f"Unified ROI dataset already exists: {output_roi_path}. Skipping.")
        return # Skips if outputs are up-to-date


    print("--- Starting ROI Standardisation Process ---")

    # Load data sources
    roi_df_dict = ROITrackedFileHandler(roi_path).load_all_data()

    # Process all DataFrames using a dictionary comprehension for conciseness
    standardised_results = {
        name: standardise_roi_size(df)
        for name, df in roi_df_dict.items()
    }
    
    print(f"✅ All objects processed.")

    # Save the consolidated results
    ROITrackedFileHandler(output_roi_path).save_all_data(standardised_results)
    print(f"--- Processing Complete. Output saved to: {output_roi_path} ---")