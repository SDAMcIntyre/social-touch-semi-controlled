from typing import Any, Optional
import importlib
from utils.package_utils import load_pyk4a

from .xyz_metadata_model import XYZMetadataModel, XYZMetadataConfig

class XYZMetadataManager:
    """Orchestrates the creation and saving of processing metadata."""

    def __init__(self, config: XYZMetadataConfig):
        """
        Initializes the manager with a configuration.

        Args:
            config (XYZMetadataConfig): The configuration object containing paths and parameters.
            repository (MetadataRepository, optional): The repository to use for persistence.
                                                     Defaults to MetadataRepository.
        """
        self.metadata = XYZMetadataModel(
            source_path=config.source_path,  # Updated from source_video_path
            input_csv_path=config.input_csv_path,
            output_csv_path=config.output_csv_path,
            video_path=config.video_path,
            metadata_path=config.metadata_path,
            monitor=config.monitor,
            input_type=config.input_type  # Add input_type
        )

    def add_mkv_metadata(self, playback: Any) -> int:
        """Delegates MKV metadata extraction to the model."""
        if self.metadata.input_type == 'mkv':
            PyK4APlayback = load_pyk4a()
            if PyK4APlayback is None:
                raise ImportError("pyk4a is required for MKV processing")
            if not isinstance(playback, PyK4APlayback):
                raise TypeError("Invalid playback object type")
        return self.metadata.populate_from_mkv(playback)

    def update_processing_details(self, key: str, value: Any):
        """Delegates updating processing details to the model."""
        self.metadata.update_processing_detail(key, value)

    def set_status(self, status: str, error_message: Optional[str] = None):
        """Delegates setting the status to the model."""
        self.metadata.set_status(status, error_message)

    def finalize(self, frames_processed: int):
        """
        Finalizes the metadata and instructs the repository to save it.
        """
        # 1. Finalize the data state in the model
        self.metadata.finalize(frames_processed)
