import pandas as pd
import re
import os

def convert_sticker_roi_to_center(input_csv_path: str, output_csv_path: str) -> None:
    """
    Reads a CSV file containing sticker tracking data with Regions of Interest (ROI),
    calculates the center point of each ROI for every frame, and saves the results
    to a new CSV file.

    Args:
        input_csv_path (str): The file path for the input CSV.
            The CSV should have a 'frame' column and columns for each sticker's
            ROI, formatted as 'sticker_<color>_roi'. The ROI data should be
            a string like '(x, y, width, height)' or similar.
        output_csv_path (str): The file path where the output CSV will be saved.
            The output CSV will contain the 'frame' column and new columns for
            the x and y coordinates of each sticker's center, formatted as
            'sticker_<color>_x' and 'sticker_<color>_y'.
    """
    if os.path.exists(output_csv_path):
        print(f"Center extraction has already been done (result file: {output_csv_path}). Skipping...")
        return output_csv_path
    
    try:
        # Step 1: Read the input CSV file into a pandas DataFrame.
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"Error: The file at {input_csv_path} was not found.")
        return

    # Step 2: Initialize the output DataFrame with the 'frame' column.
    # This ensures the frame numbering is preserved.
    output_df = pd.DataFrame()
    output_df['frame'] = df['frame']

    # Step 3: Identify all columns that represent sticker ROIs.
    # We assume these columns have names ending with '_roi'.
    roi_columns = [col for col in df.columns if col.endswith('_roi')]

    # Step 4: Iterate over each identified ROI column to process its data.
    for col_name in roi_columns:
        # Extract the color name from the column title (e.g., 'blue' from 'sticker_blue_roi').
        # This assumes the format 'sticker_<color>_roi'.
        try:
            color = col_name.split('_')[1]
        except IndexError:
            print(f"Warning: Could not parse color from column name '{col_name}'. Skipping.")
            continue

        # Define the names for the new output columns.
        x_col_name = f'sticker_{color}_x'
        y_col_name = f'sticker_{color}_y'

        # Lists to hold the calculated center coordinates for the current color.
        centers_x = []
        centers_y = []

        # Step 5: Process each cell (row) in the current ROI column.
        for roi_entry in df[col_name]:
            try:
                # Use regex to find number sequences inside brackets or parentheses.
                # This handles formats like '(n,n,n,n)' and '[n,n,n,n]' and ignores text like 'Initial ROI'.
                match = re.search(r'\[(.*?)\]|\((.*?)\)', str(roi_entry))
                if match:
                    # One of the groups will contain the numbers string (e.g., '1057, 343, 41, 32').
                    numbers_str = match.group(1) or match.group(2)
                    x, y, w, h = [int(num.strip()) for num in numbers_str.split(',')]
                    
                    # Calculate the center point of the rectangle.
                    center_x = x + w / 2
                    center_y = y + h / 2
                    
                    centers_x.append(center_x)
                    centers_y.append(center_y)
                else:
                    # If the format doesn't match, append None as a placeholder.
                    centers_x.append(None)
                    centers_y.append(None)
            except (ValueError, TypeError):
                # Handle cases where parsing fails (e.g., malformed string).
                centers_x.append(None)
                centers_y.append(None)

        # Add the new lists as columns to our output DataFrame.
        output_df[x_col_name] = centers_x
        output_df[y_col_name] = centers_y

    # Step 6: Save the resulting DataFrame to the specified output CSV file.
    # index=False prevents pandas from writing the DataFrame index as a column.
    output_df.to_csv(output_csv_path, index=False)
    print(f"Successfully processed data and saved to {output_csv_path}")
