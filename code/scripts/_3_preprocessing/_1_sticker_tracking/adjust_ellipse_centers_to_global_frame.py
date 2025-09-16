# main_adjust_coordinates_refactored.py

# Standard library imports
import argparse
from pathlib import Path

# Third-party imports
import pandas as pd

# Local application/library specific imports
from preprocessing.stickers_analysis import (
    ROITrackedFileHandler,
    ROITrackedObjects,
    FittedEllipsesFileHandler,
    FittedEllipsesManager
)

def adjust_ellipse_centers_to_global_frame(
    roi_csv_path: Path,
    ellipse_csv_path: Path,
    output_csv_path: Path,
    *,
    force_processing: bool = False
) -> None:
    """
    Adjusts ellipse center coordinates from a local ROI to the global frame
    by iterating through objects and processing them individually.

    This refactored function leverages the structure of FittedEllipsesManager
    to process data object-by-object, avoiding unnecessary data flattening
    and reconstruction for a more efficient and readable workflow.

    Args:
        roi_csv_path (Path): Path to the CSV containing ROI data.
        ellipse_csv_path (Path): Path to the CSV containing ellipse data.
        output_csv_path (Path): Path where the output CSV will be saved.
        force_processing (bool): If True, re-processes and overwrites the output file.
    """
    print("🚀 Starting coordinate adjustment process...")

    if not force_processing and output_csv_path.exists():
        print(f"✅ Output file '{output_csv_path}' already exists. Use --force to overwrite.")
        return

    # --- 1. Load Input Data using Custom Handlers ---
    try:
        print(f"📖 Loading ROI data from '{roi_csv_path}'...")
        roi_handler = ROITrackedFileHandler(roi_csv_path)
        roi_objects: ROITrackedObjects = roi_handler.load_all_data()

        print(f"📖 Loading ellipse data from '{ellipse_csv_path}'...")
        ellipse_manager: FittedEllipsesManager = FittedEllipsesFileHandler.load(ellipse_csv_path)
    except FileNotFoundError as e:
        print(f"❌ Error: Input file not found. {e}")
        return
    except Exception as e:
        print(f"❌ An unexpected error occurred while loading data: {e}")
        return

    # --- 2. Process Data Object-by-Object ---
    print("🧠 Processing objects individually...")
    output_manager = FittedEllipsesManager()
    total_records_processed = 0
    
    ellipse_results = ellipse_manager.get_all_results()

    if not ellipse_results or not roi_objects:
        print("⚠️ Warning: Input data is empty. Saving an empty result file.")
        FittedEllipsesFileHandler.save(output_manager, output_csv_path)
        return

    for name, ellipse_df in ellipse_results.items():
        # Check for corresponding ROI data
        if name not in roi_objects:
            print(f"⚠️ Warning: No ROI data found for object '{name}'. Skipping.")
            continue
            
        # Ensure 'frame_number' column consistency for merging
        roi_df = roi_objects[name].rename(columns={'frame_id': 'frame_number'})
        
        # Merge data for this specific object
        merged_df = roi_df.merge(ellipse_df, on='frame_number', how='inner')

        if merged_df.empty:
            print(f"ℹ️ Info: No matching frames for object '{name}'. Skipping.")
            continue
            
        # Perform the coordinate adjustment.
        # Adding a number to NaN results in NaN, which is the desired behavior.
        merged_df['center_x'] = merged_df['roi_x'] + merged_df['center_x']
        merged_df['center_y'] = merged_df['roi_y'] + merged_df['center_y']
        
        # ✨ --- MODIFICATION: Preserve all original ellipse columns --- ✨
        # Dynamically get the column names from the original ellipse dataframe.
        # This ensures that all data (axes_major, axes_minor, angle, etc.) is kept.
        output_columns = list(ellipse_df.columns)
        
        # Create the clean output DataFrame with the original columns and adjusted data.
        # We select these columns from the merged dataframe, which now contains
        # the globally adjusted center_x and center_y values.
        adjusted_df = merged_df[output_columns]

        # Use nullable integer types ('Int64') for coordinates to handle potential NaN values.
        # This prevents the script from crashing when a coordinate is missing.
        dtype_mapping = {
            'center_x': 'Int64',
            'center_y': 'Int64'
        }
        # Apply casting only to columns that actually exist in the DataFrame.
        existing_cols_for_casting = {k: v for k, v in dtype_mapping.items() if k in adjusted_df.columns}
        if existing_cols_for_casting:
            adjusted_df = adjusted_df.astype(existing_cols_for_casting)

        # Add the processed result directly to the output manager
        output_manager.add_result(name, adjusted_df)
        total_records_processed += len(adjusted_df)
        print(f"   -> Processed {len(adjusted_df)} records for '{name}'.")

    # --- 3. Save the Adjusted Data ---
    # The FileHandler will internally call get_combined_dataframe() on the
    # output_manager to create the final wide-format CSV.
    FittedEllipsesFileHandler.save(output_manager, output_csv_path)
    
    print(f"\n--- ✅ Process Complete ---")
    print(f"Total records processed: {total_records_processed}")
    print(f"Data for {len(output_manager.get_all_results())} objects saved to '{output_csv_path}'.")
    
    final_output_df = output_manager.get_combined_dataframe()
    if not final_output_df.empty:
        print("\n--- Data Preview (first 5 rows of combined data) ---")
        print(final_output_df.head())