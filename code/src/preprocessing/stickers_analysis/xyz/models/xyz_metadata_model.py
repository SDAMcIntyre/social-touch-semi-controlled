import os
from typing import Any, Optional, Dict, Tuple
from datetime import datetime, timezone

from pyk4a import PyK4APlayback


class XYZMetadataModel:
    """
    A self-contained class representing the configuration, dynamic state, 
    and results of a processing job.
    
    The constructor directly accepts all necessary configuration parameters.
    """
    
    def __init__(
        self, 
        source_video_path: str, 
        input_csv_path: str, 
        output_csv_path: str, 
        display_dims: Tuple[int, int] = (1080, 1920)
    ):
        """
        Initializes the metadata model with all configuration parameters.
        
        Args:
            source_video_path: Path to the source MKV video file.
            input_csv_path: Path to the input CSV file.
            output_csv_path: Path where the output CSV will be saved.
            display_dims (optional): The display dimensions as a (height, width) tuple.
        """
        # --- Input and Configuration ---
        self.source_video_path = source_video_path
        self.input_csv_path = input_csv_path
        self.output_csv_path = output_csv_path
        self.display_dims = display_dims

        # --- Timestamps and Status ---
        self.start_time_utc: str = datetime.now(timezone.utc).isoformat()
        self.end_time_utc: Optional[str] = None
        self.status: str = "In Progress"
        self.error_message: Optional[str] = None

        # --- Dynamic Data (Populated during processing) ---
        self.mkv_metadata: Dict[str, Any] = {}
        self.processing_details: Dict[str, Any] = {}

    def populate_from_mkv(self, playback: PyK4APlayback) -> int:
        """
        Extracts and adds metadata from the k4a playback object.
        Returns the camera FPS as an integer.
        """
        k4a_fps_to_int = {0: 0, 1: 5, 2: 15, 3: 30}
        config = playback.configuration
        fps_int = k4a_fps_to_int.get(config['camera_fps'], 30)

        self.mkv_metadata = {
            "camera_fps": fps_int,
            "color_format": str(config['color_format']),
            "color_resolution": str(config['color_resolution']),
            "depth_mode": str(config['depth_mode']),
            "wired_sync_mode": str(config['wired_sync_mode']),
        }
        return fps_int

    def update_processing_detail(self, key: str, value: Any):
        """Updates a single key-value pair in the processing details."""
        self.processing_details[key] = value

    def set_status(self, status: str, error_message: Optional[str] = None):
        """Sets the final status and an optional error message."""
        self.status = status
        if error_message:
            self.error_message = error_message

    def finalize(self, success: bool = True):
        """
        Finalizes the metadata before saving, setting the end time and status.
        """
        self.end_time_utc = datetime.now(timezone.utc).isoformat()
        if success and self.status == "In Progress":
            self.set_status("Completed")
        elif not success and self.status == "In Progress":
            self.set_status("Failed")

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the metadata model to a dictionary suitable for JSON serialization.
        """
        return {
            "start_time_utc": self.start_time_utc,
            "end_time_utc": self.end_time_utc,
            "status": self.status,
            "error_message": self.error_message,
            "inputs": {
                "source_video": os.path.abspath(self.source_video_path),
                "input_csv_path": os.path.abspath(self.input_csv_path),
            },
            "outputs": {
                "output_csv_path": os.path.abspath(self.output_csv_path),
            },
            "mkv_metadata": self.mkv_metadata,
            "processing_details": self.processing_details,
        }
