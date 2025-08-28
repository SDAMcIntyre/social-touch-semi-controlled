from typing import Any, Optional
import importlib
from pathlib import Path
from .xyz_metadata_model import XYZMetadataModel, XYZMetadataConfig

class XYZMetadataManager:
    """Orchestrates the creation and saving of processing metadata."""

    def __init__(self, config: XYZMetadataConfig):
        """
        Initializes the manager with a configuration.

        Args:
            config (XYZMetadataConfig): The configuration object containing paths and parameters.
        """
        self.metadata = XYZMetadataModel(
            source_path=config.source_path,
            input_csv_path=config.input_csv_path,
            output_csv_path=config.output_csv_path,
            video_path=config.video_path,
            metadata_path=config.metadata_path,
            monitor=config.monitor,
            input_type=config.input_type
        )

    def _load_pyk4a(self):
        """Loads pyk4a only if needed for MKV processing."""
        try:
            pyk4a = importlib.import_module('pyk4a')
            return pyk4a.PyK4APlayback
        except ImportError:
            return None

    def add_input_metadata(self, input_data: Any) -> int:
        """
        Adds metadata based on input type (MKV or TIFF).
        
        Args:
            input_data: Either PyK4APlayback object for MKV or Path for TIFF folder
            
        Returns:
            int: Frame rate for MKV or number of frames for TIFF
        """
        if self.metadata.input_type == 'mkv':
            PyK4APlayback = self._load_pyk4a()
            if PyK4APlayback is None:
                raise ImportError("pyk4a is required for MKV processing")
            if not isinstance(input_data, PyK4APlayback):
                raise TypeError("Invalid playback object type")
            return self.metadata.populate_from_mkv(input_data)
        else:  # tiff
            if not isinstance(input_data, (str, Path)):
                raise TypeError("TIFF input must be a path")
            return self.metadata.populate_from_tiff(Path(input_data))

    def update_processing_details(self, key: str, value: Any):
        """Delegates updating processing details to the model."""
        self.metadata.update_processing_detail(key, value)

    def set_status(self, status: str, error_message: Optional[str] = None):
        """Delegates setting the status to the model."""
        self.metadata.set_status(status, error_message)

    def finalize(self, frames_processed: int):
        """Finalizes the metadata and instructs the repository to save it."""
        self.metadata.finalize(frames_processed)
