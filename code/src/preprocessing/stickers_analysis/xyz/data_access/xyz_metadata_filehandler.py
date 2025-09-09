from typing import Any, Optional
from pyk4a import PyK4APlayback
from pathlib import Path
import json

from ..models.xyz_metadata_model import XYZMetadataModel, XYZMetadataConfig

class XYZMetadataFileHandler:
    """Orchestrates the creation and saving of processing metadata."""

    def __init__(self, config: XYZMetadataConfig):
        """
        Initializes the manager with a configuration and a repository.

        Args:
            config (XYZMetadataConfig): The configuration object containing paths and parameters.
            repository (MetadataRepository, optional): The repository to use for persistence.
                                                     Defaults to MetadataRepository.
        """
        self.metadata = XYZMetadataModel(
            source_video_path=config.source_video_path,
            input_csv_path=config.input_csv_path,
            output_csv_path=config.output_csv_path,
            video_path=config.video_path,
            metadata_path=config.metadata_path,
            monitor=config.monitor
        )

    def get_metadata(self):
        return self.metadata

    def save(self, metadata: XYZMetadataModel, output_path: str): 
        """
        Serializes the metadata to a dictionary and saves it as a JSON file.
        """
        self.metadata = metadata
        
        # 🧙‍♂️ Best Practice: Use pathlib for robust path manipulation.
        output_path = Path(output_path)
        
        # Ensure the parent directory exists to prevent FileNotFoundError.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get the dictionary representation of your metadata object.
        data_to_save = self.metadata.to_dict()
        
        # Use a 'with' statement to safely open the file and handle closing it.
        # 'w' opens the file for writing, creating it if it doesn't exist.
        # 'encoding='utf-8'' is crucial for handling a wide range of characters.
        with open(output_path, 'w', encoding='utf-8') as f:
            # json.dump() writes the dictionary to the file object 'f'.
            # 'indent=4' formats the JSON with 4-space indentation for readability.
            json.dump(data_to_save, f, indent=4)
            
        print(f"✅ Metadata successfully saved to: {output_path}")
