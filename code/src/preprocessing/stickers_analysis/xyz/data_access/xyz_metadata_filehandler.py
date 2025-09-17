from pathlib import Path
import json
from typing import Optional

from ..models.xyz_metadata_model import XYZMetadataModel

class XYZMetadataFileHandler:
    """Orchestrates the creation, saving, and loading of processing metadata."""

    def __init__(self):
        pass

    @staticmethod
    def save(metadata: XYZMetadataModel, output_path: Path): 
        """
        Serializes the metadata to a dictionary and saves it as a JSON file.
        """
        # 🧙‍♂️ Best Practice: Use pathlib for robust path manipulation.
        output_path = Path(output_path)
        
        # Ensure the parent directory exists to prevent FileNotFoundError.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get the dictionary representation of your metadata object.
        data_to_save = metadata.to_dict()
        
        # Use a 'with' statement to safely open the file and handle closing it.
        # 'w' opens the file for writing, creating it if it doesn't exist.
        # 'encoding='utf-8'' is crucial for handling a wide range of characters.
        with open(output_path, 'w', encoding='utf-8') as f:
            # json.dump() writes the dictionary to the file object 'f'.
            # 'indent=4' formats the JSON with 4-space indentation for readability.
            json.dump(data_to_save, f, indent=4)
            
        print(f"✅ Metadata successfully saved to: {output_path}")

    @staticmethod
    def load(file_path: Path) -> Optional[XYZMetadataModel]:
        """
        Loads metadata from a JSON file and reconstructs the XYZMetadataModel object.

        Args:
            file_path: The path to the JSON metadata file.

        Returns:
            An instance of XYZMetadataModel if successful, otherwise None.
        """
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Re-create the model instance using the constructor's required arguments.
            # The 'display_dims' argument is not saved in the JSON, so it will
            # correctly fall back to the default value defined in the constructor.
            metadata_obj = XYZMetadataModel(
                source_video_path=data['inputs']['source_video'],
                input_csv_path=data['inputs']['input_csv_path'],
                output_csv_path=data['outputs']['output_csv_path'],
            )

            # Populate the remaining fields from the loaded data.
            # Using .get() provides default values if a key is missing.
            metadata_obj.start_time_utc = data.get('start_time_utc')
            metadata_obj.end_time_utc = data.get('end_time_utc')
            metadata_obj.status = data.get('status', 'Unknown')
            metadata_obj.error_message = data.get('error_message')
            metadata_obj.mkv_metadata = data.get('mkv_metadata', {})
            metadata_obj.processing_details = data.get('processing_details', {})
            
            print(f"✅ Metadata successfully loaded from: {file_path}")
            return metadata_obj

        except FileNotFoundError:
            print(f"❌ Error: Metadata file not found at {file_path}")
            return None
        except json.JSONDecodeError:
            print(f"❌ Error: Failed to decode JSON from {file_path}. File may be invalid or empty.")
            return None
        except KeyError as e:
            print(f"❌ Error: Missing essential key {e} in metadata file {file_path}.")
            return None