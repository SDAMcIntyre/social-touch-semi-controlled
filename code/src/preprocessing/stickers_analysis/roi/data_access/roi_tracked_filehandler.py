
import pandas as pd
from pathlib import Path
from typing import Any

# Assuming these are defined elsewhere, as in your original code
from ..models.roi_tracked_data import ROITrackedObjects, ROI_TRACKED_SCHEMA


class ROITrackedFileHandler:
    """
    Handles reading and writing all tracked objects' data to a single "long" 
    or "tidy" format CSV file. In this format, an 'object_name' column is
    used to identify which object each record belongs to.
    
    This class acts as a translation layer between the on-disk format (one CSV)
    and the in-memory format (a dictionary of DataFrames).
    """
    def __init__(self, storage_path: str):
        """
        Initializes the handler.

        Args:
            storage_path (str): The full path to the single CSV file for storage.
        """
        self.file_path = Path(storage_path)
        # Ensure the parent directory for the file exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def is_roi_tracked_objects(variable: Any) -> bool:
        """
        Checks if a variable conforms to the ROITrackedObjects type structure.

        The variable is considered a valid ROITrackedObjects if it is a dictionary
        where all keys are strings and all values are pandas DataFrames. An empty 
        dictionary is also considered valid.

        Args:
            variable (Any): The variable to check.

        Returns:
            bool: True if the variable is of the correct structure, False otherwise.
        """
        # 1. Ensure the variable is a dictionary.
        if not isinstance(variable, dict) or not variable:
            return False

        # 2. Check if all keys are strings and all values are pandas DataFrames.
        # The all() function on an empty dictionary will return True, correctly
        # handling the case of an empty collection.
        return all(
            isinstance(key, str) and isinstance(value, pd.DataFrame)
            for key, value in variable.items()
        )
    
    def load_all_data(self) -> ROITrackedObjects:
        """
        Loads all object data from the single long-format CSV file.

        It reads the CSV and then uses groupby('object_name') to split the data 
        back into a dictionary of individual DataFrames.

        Returns:
            ROITrackedObjects: A dictionary mapping object names to their DataFrames.
                               Example: {'car': DataFrame, 'person': DataFrame}
        """
        if not self.file_path.exists():
            return {}

        try:
            # Read the entire dataset from the single CSV
            long_df = pd.read_csv(self.file_path)
        except pd.errors.EmptyDataError:
            return {}
            
        if 'object_name' not in long_df.columns:
            print(f"Warning: File {self.file_path} is missing 'object_name' column. Cannot load data.")
            return {}

        # Use groupby to split the single DataFrame into a dictionary of DataFrames.
        # The keys of the dictionary will be the unique values from 'object_name' column.
        objects_data = {
            name: group.drop('object_name', axis=1).reset_index(drop=True).astype(ROI_TRACKED_SCHEMA)
            for name, group in long_df.groupby('object_name')
        }
        
        return objects_data

    def save_all_data(self, all_objects_data: ROITrackedObjects) -> None:
        """
        Saves data for all objects into a single long-format CSV file.

        Args:
            all_objects_data (ROITrackedObjects): 
                A dictionary of object names to DataFrames that should be saved.
                Example: {'car': DataFrame, 'person': DataFrame}
        """
        if not all_objects_data:
            # If there's no data, ensure the file is empty.
            return
            open(self.file_path, 'w').close()
            return

        # Create a list to hold the DataFrames before concatenation
        all_dfs_list = []
        for name, data in all_objects_data.items():
            # Add the 'object_name' column to each DataFrame to identify its data
            df_with_id = data.copy()
            df_with_id['object_name'] = name
            all_dfs_list.append(df_with_id)
        
        # Concatenate all DataFrames vertically into one "long" DataFrame
        long_df = pd.concat(all_dfs_list, ignore_index=True)
        
        # Reorder columns to have 'object_name' first for better human readability
        cols = ['object_name'] + [col for col in ROI_TRACKED_SCHEMA.keys()]
        long_df = long_df[cols]
        
        # Save the combined DataFrame to the single CSV file
        long_df.to_csv(self.file_path, index=False)