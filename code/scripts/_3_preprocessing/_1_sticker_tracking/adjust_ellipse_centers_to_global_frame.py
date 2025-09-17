# main_adjust_coordinates_refactored.py

# Standard library imports
import argparse
from pathlib import Path

# Third-party imports
import pandas as pd
import numpy as np
import cv2

# Local application/library specific imports
from utils.should_process_task import should_process_task
from preprocessing.stickers_analysis import (
    ROITrackedFileHandler,
    ROITrackedObjects,
    FittedEllipsesFileHandler,
    FittedEllipsesManager
)

def visualize_data(
    merged_df: pd.DataFrame, 
    window_title: str = 'ROI and Ellipse Visualization', 
    width: int = 1920, 
    height: int = 1080
) -> None:
    """
    Creates an interactive OpenCV window with a slider to visualize ROI and ellipses.

    Args:
        merged_df (pd.DataFrame): The DataFrame containing the shape data per frame.
                                  Must include columns like 'roi_x', 'roi_y', 'center_x', etc.
        window_title (str): The title for the visualization window.
        width (int): The width of the black background frame.
        height (int): The height of the black background frame.
    """
    
    # Define the drawing function inside the main function's scope
    # This allows it to access 'merged_df', 'width', and 'height' directly.
    def draw_shapes(frame_index: int):
        """Callback function executed when the slider changes."""
        # Create a black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Get the data for the current frame
        try:
            row = merged_df.iloc[frame_index]
        except IndexError:
            return

        # --- Draw the ROI rectangle (Green) ---
        if pd.notna(row['roi_x']):
            pt1 = (int(row['roi_x']), int(row['roi_y']))
            pt2 = (int(row['roi_x'] + row['roi_width']), int(row['roi_y'] + row['roi_height']))
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame, 'ROI', (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # --- Draw the ellipse (Blue) ---
        if pd.notna(row['center_x']) and pd.notna(row['axes_major']):
            center = (int(row['center_x']), int(row['center_y']))
            axes = (int(row['axes_major'] / 2), int(row['axes_minor'] / 2))
            angle = int(row['angle'])
            cv2.ellipse(frame, center, axes, angle, 0, 360, (255, 0, 0), 2)
            
            if pd.notna(row['score']):
                score_text = f"Score: {row['score']:.2f}"
                cv2.putText(frame, score_text, (center[0] + 20, center[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Display the final frame
        cv2.imshow(window_title, frame)

    # --- Main function logic ---
    if merged_df.empty:
        print("Error: The provided DataFrame is empty.")
        return

    # Create a resizable window
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 1280, 720) 

    # Create the slider (trackbar)
    max_frame = len(merged_df) - 1
    cv2.createTrackbar('Frame', window_title, 0, max_frame, draw_shapes)

    # Initialize the view with the first frame
    draw_shapes(0)

    print(f"\n[{window_title}] window is active.")
    print("Drag the slider to change frames. Press 'q' or ESC to quit.")

    # Loop to keep the window open and responsive
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
            break
            
    cv2.destroyAllWindows()
    print("Window closed.")


def adjust_ellipse_centers_to_global_frame(
    roi_csv_path: Path,
    ellipse_csv_path: Path,
    output_csv_path: Path,
    *,
    force_processing: bool = False,
    debug_view: bool = False
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
    if not should_process_task(output_paths=output_csv_path, input_paths=[roi_csv_path, ellipse_csv_path], force=force_processing):
        print(f"✅ Output file '{output_csv_path}' already exists. Use --force to overwrite.")
        return

    print("🚀 Starting coordinate adjustment process...")

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
        roi_df: pd.DataFrame = roi_objects[name].rename(columns={'frame_id': 'frame_number'})
        
        # Merge data for this specific object
        merged_df = roi_df.merge(ellipse_df, on='frame_number', how='inner')

        if merged_df.empty:
            print(f"ℹ️ Info: No matching frames for object '{name}'. Skipping.")
            continue
            
        # Perform the coordinate adjustment.
        # Adding a number to NaN results in NaN, which is the desired behavior.
        merged_df['center_x'] = merged_df['roi_x'] + merged_df['center_x']
        merged_df['center_y'] = merged_df['roi_y'] + merged_df['center_y']
        if debug_view: visualize_data(merged_df, f'ROI and Ellipse Visualization: {name}')

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