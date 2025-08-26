import pandas as pd
from typing import List, Dict, Any

class XYZDataFileHandler:
    """Handles saving the processed 3D coordinate data to a CSV file."""

    def save(self, results_data: List[Dict[str, Any]], sticker_names: List[str], output_path: str):
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

        result_df = pd.DataFrame(results_data)
        # Ensure column order is consistent
        cols = ['frame'] + [f"{name}_{axis}_mm" for name in sticker_names for axis in ['x', 'y', 'z']]
        result_df = result_df[cols]
        
        result_df.to_csv(output_path, index=False, float_format='%.2f')
        print(f"\nSuccessfully saved 3D sticker positions to: {output_path}")
