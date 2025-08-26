import json

from ..models.xyz_metadata_model import XYZMetadataModel

class XYZMetadataFileHandler:
    """Handles the persistence (saving/loading) of metadata."""

    def save_json(self, metadata: XYZMetadataModel):
        """
        Saves the provided MetadataModel to its specified path as a JSON file.

        Args:
            metadata (MetadataModel): The metadata object to save.
        
        Raises:
            ValueError: If the metadata_path is not set in the model.
            IOError: If the file cannot be written.
        """
        if not metadata.metadata_path:
            print("\nWarning: metadata_path is not set. Skipping save.")
            return

        try:
            with open(metadata.metadata_path, 'w') as f:
                # Convert the model to a dict before dumping
                json.dump(metadata.to_dict(), f, indent=4)
            print(f"\nSuccessfully saved metadata to: {metadata.metadata_path}")
        except IOError as e:
            print(f"\nError: Could not write metadata file to {metadata.metadata_path}. Reason: {e}")
            raise # Re-raise the exception so the caller can handle it if needed
    