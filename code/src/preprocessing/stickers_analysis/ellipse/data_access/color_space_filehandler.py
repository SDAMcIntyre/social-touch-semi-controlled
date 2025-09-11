import json
from ..models.color_space_manager_model import ColorSpaceManager

class ColorSpaceFileHandler:
    """
    A handler class for reading from and writing to colorspace metadata files.
    """
    @staticmethod
    def load(filepath: str) -> ColorSpaceManager:
        """
        Reads a JSON file and returns a ColorSpace object.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return ColorSpaceManager(data)
        except FileNotFoundError:
            print(f"Error: The file at {filepath} was not found.")
            raise
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode JSON from {filepath}. Details: {e}")
            raise

    # --- NEW METHOD ---
    @staticmethod
    def write(filepath: str, metadata: ColorSpaceManager):
        """
        Writes a ColorSpace object to a JSON file with pretty-printing.

        Args:
            filepath (str): The path to the file where data will be saved.
            metadata (ColorSpace): The metadata object to serialize.

        Raises:
            IOError: If an error occurs during file writing.
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # Use the to_dict() method to get the serializable data
                json.dump(metadata.to_dict(), f, indent=4)
        except IOError as e:
            print(f"Error: Could not write to file at {filepath}. Details: {e}")
            raise