from typing import Any, Optional
from pyk4a import PyK4APlayback

from .xyz_metadata_model import XYZMetadataModel, XYZMetadataConfig

class XYZMetadataManager:
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
            center_csv_path=config.center_csv_path,
            output_csv_path=config.output_csv_path,
            video_path=config.video_path,
            metadata_path=config.metadata_path,
            monitor=config.monitor
        )

    def add_mkv_metadata(self, playback: PyK4APlayback) -> int:
        """Delegates MKV metadata extraction to the model."""
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
