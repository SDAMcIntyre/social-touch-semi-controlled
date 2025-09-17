# Standard library imports
from pathlib import Path
from typing import Optional, Dict

# Third-party imports
import numpy as np
import pandas as pd

# Local application/library specific imports
from utils.should_process_task import should_process_task
from preprocessing.stickers_analysis import (
    ROITrackedFileHandler,
    ROITrackedObjects,

    FittedEllipsesFileHandler,
    FittedEllipsesManager,

    ConsolidatedTracksFileHandler,
    ConsolidatedTracksManager
)

def convert_to_int_where_finite(series: pd.Series) -> pd.Series:
    """
    Converts finite float values in a Series to integers,
    leaving non-finite values (NaN, inf) untouched.
    """
    # Create a boolean mask to identify valid, finite numbers.
    # pd.notna() handles None/NaN, and np.isfinite() handles +/- infinity.
    is_finite_mask = pd.notna(series) & np.isfinite(series)

    # Use the mask with .loc to apply the .astype(int) conversion
    # only to the finite values. This avoids errors on NaNs or infs.
    # We work on a copy to avoid SettingWithCopyWarning.
    result = series.copy()
    result.loc[is_finite_mask] = result.loc[is_finite_mask].astype(int)
    
    return result


def consolidate_2d_tracking_data(
    roi_csv_path: Path,
    ellipse_csv_path: Path,
    output_csv_path: Path,
    score_threshold: float = 0.7,
    *,
    force_processing: bool = False
) -> Optional[ConsolidatedTracksManager]:
    """
    Determines final object center coordinates by selecting between ellipse fits and ROI centers.

    This function returns a structured ConsolidatedTracksManager object
    and uses a dedicated FileHandler for all I/O, improving separation of concerns.

    Args:
        roi_csv_path (Path): Path to the long-format CSV with ROI data.
        ellipse_csv_path (Path): Path to the wide-format CSV with pre-adjusted,
                                 global-frame ellipse data.
        output_csv_path (Path): Path to save the resulting CSV file.
        score_threshold (float): The confidence score to use the ellipse center.
        force_processing (bool): If True, re-processes even if output exists.

    Returns:
        Optional[ConsolidatedTracksManager]: A manager object with final coordinates.
    """
    if not should_process_task(output_paths=output_csv_path, input_paths=[roi_csv_path, ellipse_csv_path], force=force_processing):
        print(f"✅ Output file '{output_csv_path}' already exists. Use --force to overwrite.")
        return

    try:
        # 1. Load Data using robust handlers
        print("📖 Loading input data...")
        roi_handler = ROITrackedFileHandler(roi_csv_path)
        roi_objects: ROITrackedObjects = roi_handler.load_all_data()
        ellipse_manager: FittedEllipsesManager = FittedEllipsesFileHandler.load(ellipse_csv_path)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Please check your file paths.")
        return None

    # 2. Prepare Data for Merging
    if not roi_objects:
        print("⚠️ Warning: ROI data is empty. Cannot proceed.")
        return None
    
    rois_df = pd.concat(
        [df.assign(object_name=name) for name, df in roi_objects.items()],
        ignore_index=True
    ).rename(columns={'frame_id': 'frame_number'})
    
    ellipse_results = ellipse_manager.get_all_results()
    if not ellipse_results:
        print("⚠️ Warning: Ellipse data is empty.")
        ellipses_long_df = pd.DataFrame(columns=['frame_number', 'center_x', 'center_y', 'score', 'object_name'])
    else:
        ellipses_long_df = pd.concat(
            [df.assign(object_name=name) for name, df in ellipse_results.items()],
            ignore_index=True
        )
    
    # --- MODIFICATION START 1 ---
    # Rename original ellipse center columns to avoid name conflicts and preserve the data.
    ellipses_long_df.rename(
        columns={'center_x': 'ellipse_center_x', 'center_y': 'ellipse_center_y'},
        inplace=True
    )
    # --- MODIFICATION END 1 ---
    
    # 3. Merge ROI and Ellipse Data
    print("🤝 Merging ROI and ellipse information...")
    merged_df = pd.merge(rois_df, ellipses_long_df, on=['frame_number', 'object_name'], how='left')

    # 4. Apply Conditional Logic for Center Coordinates
    print("🧠 Calculating center coordinates based on score threshold...")
    merged_df['score'].fillna(0, inplace=True)
    
    # Calculate the ROI center as a fallback
    roi_center_x = merged_df['roi_x'] + (merged_df['roi_width'] / 2)
    roi_center_y = merged_df['roi_y'] + (merged_df['roi_height'] / 2)

    # Use the renamed 'ellipse_center_x/y' columns for the calculation.
    # The np.where function directly uses the ellipse's coordinates when the score is high.
    merged_df['center_x_float'] = np.where(
        merged_df['score'] > score_threshold,
        merged_df['ellipse_center_x'], # Use global ellipse center directly
        roi_center_x                   # Fallback to ROI center
    )
    merged_df['center_y_float'] = np.where(
        merged_df['score'] > score_threshold,
        merged_df['ellipse_center_y'], # Use global ellipse center directly
        roi_center_y                   # Fallback to ROI center
    )

    # 5. Finalize DataFrame for the Manager
    # Keep all original input columns and add the final calculated centers.
    output_df = merged_df.copy()
    
    # Create the final 'center_x' and 'center_y' columns required by the manager, converting to int.# Assume 'merged_df' is your starting DataFrame
    output_df['center_x'] = convert_to_int_where_finite(output_df['center_x_float'])
    output_df['center_y'] = convert_to_int_where_finite(output_df['center_y_float'])

    
    # Remove the intermediate floating-point columns used for calculation.
    output_df.drop(columns=['center_x_float', 'center_y_float'], inplace=True)

    # 6. Instantiate Manager, Save, and Return
    coordinate_manager = ConsolidatedTracksManager(output_df)
    
    ConsolidatedTracksFileHandler.save(coordinate_manager, output_csv_path)
    
    print("\n--- Processing Summary ---")
    print(coordinate_manager)
    print("\n--- Data Preview ---")
    print(coordinate_manager.get_all_data().head(10))
    
    return coordinate_manager