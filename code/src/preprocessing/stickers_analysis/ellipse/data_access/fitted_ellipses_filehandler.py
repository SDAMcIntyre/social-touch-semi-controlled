# Standard library imports
from __future__ import annotations
from pathlib import Path

# Third-party imports
import pandas as pd

from ..models.fitted_ellipses_manager import FittedEllipsesManager

# ==============================================================================
# No changes are needed for the FileHandler. Its interface remains the same.
# ==============================================================================
class FittedEllipsesFileHandler:
    """
    Handles file I/O by saving an FittedEllipsesManager to a file
    or loading a file to create an FittedEllipsesManager.
    """
    @staticmethod
    def save(manager: FittedEllipsesManager, path: Path) -> None:
        """
        Saves the state of an FittedEllipsesManager to a CSV file.

        Args:
            manager (FittedEllipsesManager): The manager instance to save.
            path (Path): The file path where the CSV will be saved.
        """
        print(f"💾 Aggregating data from manager and writing to '{path}'...")
        dataframe = manager.get_combined_dataframe()
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False)

    @staticmethod
    def load(path: Path) -> FittedEllipsesManager:
        """
        Loads data from a CSV file and reconstructs an FittedEllipsesManager.

        Args:
            path (Path): The path to the CSV file.

        Returns:
            FittedEllipsesManager: A new manager instance populated with the loaded data.
        """
        if not path.is_file():
            raise FileNotFoundError(f"File not found at '{path}'")
        
        print(f"📖 Reading '{path}' to reconstruct a results manager...")
        combined_df = pd.read_csv(path)
        manager = FittedEllipsesManager()

        if combined_df.empty or 'frame_number' not in combined_df.columns:
            return manager

        # Identify unique object prefixes from column names
        prefixed_cols = [col for col in combined_df.columns if col != 'frame_number']
        prefixes = sorted(list(set('_'.join(col.split('_')[:2]) for col in prefixed_cols)))

        # Reconstruct the original DataFrame for each object
        for prefix in prefixes:
            cols_for_object = ['frame_number'] + [
                col for col in prefixed_cols if col.startswith(f"{prefix}_")
            ]
            object_df = combined_df[cols_for_object].copy()
            
            # Remove prefixes from column names
            rename_map = {
                col: col.removeprefix(f"{prefix}_") for col in object_df.columns
            }
            object_df.rename(columns=rename_map, inplace=True)
            
            # Add the reconstructed DataFrame to the manager
            manager.add_result(prefix, object_df)
            print(f"🧠 Reconstructed and stored results for '{prefix}'.")

        return manager