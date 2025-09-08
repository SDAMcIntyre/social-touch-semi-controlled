import pandas as pd
from typing import Dict

class XYZDataFileHandler:
    """
    Handles saving and loading processed 3D coordinate data.
    
    This class converts between a dictionary of DataFrames (one per sticker)
    and a single, flat CSV file for persistent storage.
    """

    @staticmethod
    def save(data_dict: Dict[str, pd.DataFrame], output_path: str):
        """
        Saves a dictionary of DataFrames to a single, flat CSV file.

        Each key in the dictionary is the sticker name, and each value is a
        DataFrame indexed by 'frame' with columns for coordinates ('x_mm', 'y_mm', etc.).

        Args:
            data_dict (Dict[str, pd.DataFrame]): The dictionary of DataFrames to save.
            output_path (str): The path to the output CSV file.
        """
        if not data_dict:
            print("\nInput dictionary is empty. The output file was not created.")
            return

        # 1. Prepare a list of DataFrames, renaming columns to include the sticker name.
        # e.g., 'x_mm' in the 'sticker_blue' df becomes 'sticker_blue_x_mm'.
        all_dfs = []
        for sticker_name, df in data_dict.items():
            renamed_df = df.rename(columns=lambda col: f"{sticker_name}_{col}")
            all_dfs.append(renamed_df)

        # 2. Concatenate all DataFrames horizontally. The 'frame' index ensures
        #    that rows are aligned correctly.
        flat_df = pd.concat(all_dfs, axis=1)
        
        # 3. Sort columns alphabetically for consistent file output.
        flat_df = flat_df.sort_index(axis=1)

        # 4. Save to CSV. The 'frame' index will be written as the first column.
        flat_df.to_csv(output_path, index=True, float_format='%.2f')
        print(f"\nSuccessfully saved 3D sticker positions to: {output_path}")


    @staticmethod
    def load(file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Loads a flat CSV file into a dictionary of DataFrames.

        This method reverses the 'save' operation, creating a structured dictionary
        where each key is a sticker name and each value is its coordinate DataFrame.

        Args:
            file_path (str): The path to the input CSV file.

        Returns:
            A dictionary where keys are sticker names and values are DataFrames
            indexed by 'frame' with coordinate columns ('x_mm', 'y_mm', etc.).

        Raises:
            FileNotFoundError: If the specified file_path does not exist.
        """
        try:
            # 1. Read the CSV, using the first column ('frame') as the index.
            df = pd.read_csv(file_path, index_col='frame')
        except FileNotFoundError:
            print(f"Error: The file at {file_path} was not found.")
            raise

        # 2. Create a MultiIndex for the columns from the flat names.
        #    e.g., 'sticker_blue_x_mm' -> ('sticker_blue', 'x_mm')
        #    The logic is now robust for any number of parts in the coordinate.
        
        # ✅ CORRECTED LOGIC
        parsed_columns = []
        for col in df.columns:
            parts = col.split('_')
            sticker_name = f"{parts[0]}_{parts[1]}"  # Assumes 'sticker_color' format
            coord_name = '_'.join(parts[2:])         # The rest is the coordinate
            parsed_columns.append((sticker_name, coord_name))

        df.columns = pd.MultiIndex.from_tuples(
            parsed_columns,
            names=['sticker', 'coord']
        )

        # 3. Split the DataFrame into a dictionary based on the 'sticker' level
        #    of the column index.
        output_dict = {
            sticker_name: df[sticker_name]
            for sticker_name in df.columns.get_level_values('sticker').unique()
        }
        
        print(f"Successfully loaded and structured data from: {file_path}")
        return output_dict