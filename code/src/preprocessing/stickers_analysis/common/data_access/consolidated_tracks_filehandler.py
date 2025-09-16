from pathlib import Path
import pandas as pd

from ..models.consolidated_tracks_manager import ConsolidatedTracksManager

class ConsolidatedTracksFileHandler:
    """
    Handles the reading and writing of ConsolidatedTracksManager data to/from CSV files.
    
    This class uses only static methods as it does not need to maintain any state.
    It acts as a dedicated namespace for I/O operations related to center coordinates.
    """

    @staticmethod
    def save(manager: ConsolidatedTracksManager, output_path: Path) -> None:
        """
        Saves the data from a ConsolidatedTracksManager to a CSV file.

        This method calls manager.get_all_data() to get a unified DataFrame
        that is suitable for CSV serialization.

        Args:
            manager (ConsolidatedTracksManager): The data manager instance to save.
            output_path (Path): The file path where the CSV will be saved.
        """
        try:
            # Reconstruct the single DataFrame before saving
            df_to_save = manager.get_all_data()
            
            # Make sure that the columns 'object_name' appears early the csv document
            cols = df_to_save.columns.tolist()
            cols.remove('object_name')
            cols.insert(1, 'object_name')
            df_to_save = df_to_save[cols]

            df_to_save.to_csv(output_path, index=False)
            print(f"✅ Success! Center coordinates saved to '{output_path}'.")
        except Exception as e:
            print(f"❌ Failed to save file to '{output_path}': {e}")
            raise

    @staticmethod
    def load(input_path: Path) -> ConsolidatedTracksManager:
        """
        Loads center coordinate data from a CSV file into a ConsolidatedTracksManager.
        
        The loaded DataFrame is passed to the manager's constructor, which handles
        the conversion to the internal dictionary format.

        Args:
            input_path (Path): The path to the CSV file to load.

        Returns:
            ConsolidatedTracksManager: An instance containing the loaded data.
            
        Raises:
            FileNotFoundError: If the input_path does not exist.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"The file '{input_path}' was not found.")
        
        try:
            loaded_df = pd.read_csv(input_path)
            print(f"📖 Successfully loaded data from '{input_path}'.")
            # The manager's __init__ will handle validation and transformation.
            return ConsolidatedTracksManager(loaded_df)
        except Exception as e:
            print(f"❌ Failed to load or parse file from '{input_path}': {e}")
            raise