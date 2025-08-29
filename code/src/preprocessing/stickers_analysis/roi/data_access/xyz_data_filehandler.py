import pandas as pd
from typing import List, Dict, Any

class XYZDataFileHandler:
    """Handles saving the processed 3D coordinate data to a CSV file."""

    def save(self, results_data: Dict[str, Dict[str, Any]], sticker_names: List[str], output_path: str):
        """
        Saves the final 3D coordinates to a CSV file.
        
        Args:
            results_data (List[Dict[str, Any]]): A list of dictionaries, where each dict holds a frame's results.
            sticker_names (List[str]): The names of the stickers to create header columns.
            output_path (str): The path to the output CSV file.
        """
        if not results_data:
            print("\nNo data was processed. The output file was not created.")
            return

        # 1. Create a list of DataFrames, one for each sticker
        all_dfs = []
        for name, data_list in results_data.items():
            # Create a DF from the list of dicts for the current sticker
            temp_df = pd.DataFrame(data_list)
            
            # We only care about x, y, z coordinates
            temp_df = temp_df[['x_mm', 'y_mm', 'z_mm']]
            
            # Rename columns to be specific to this sticker (e.g., 'x_mm' -> 'sticker_blue_x_mm')
            temp_df.columns = [f"{name}_{axis}_mm" for axis in ['x', 'y', 'z']]
            all_dfs.append(temp_df)

        # 2. Concatenate all DataFrames horizontally
        result_df = pd.concat(all_dfs, axis=1)

        # 3. Add the 'frame' column at the beginning
        result_df.insert(0, 'frame', range(len(result_df)))
        result_df.to_csv(output_path, index=False, float_format='%.2f')
        print(f"\nSuccessfully saved 3D sticker positions to: {output_path}")
