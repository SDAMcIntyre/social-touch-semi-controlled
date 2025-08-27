import os
from typing import Any, Optional, Dict, Tuple, Union
import PIL.Image
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Make pyk4a import optional
try:
    from pyk4a import PyK4APlayback
    HAVE_PYK4A = True
except ImportError:
    HAVE_PYK4A = False
    PyK4APlayback = Any  # type placeholder


@dataclass
class XYZMetadataConfig:
    """Holds all configuration for the XYZ extraction process."""
    source_path: str  # Changed from source_video_path to handle both video and TIFF
    input_csv_path: str
    output_csv_path: str
    metadata_path: Optional[str] = None
    monitor: bool = False
    video_path: Optional[str] = None
    display_dims: Tuple[int, int] = (1080, 1920)
    input_type: str = 'mkv'  # Added input_type field


@dataclass
class XYZMetadataModel:
    """A data class representing the structure and state of processing metadata."""
    
    # --- Input and Configuration ---
    source_path: str  # Changed from source_video_path
    input_csv_path: str
    output_csv_path: str
    metadata_path: Optional[str] = None
    monitor: bool = False 
    video_path: Optional[str] = None
    display_dims: Tuple[int, int] = (1080, 1920)
    input_type: str = 'mkv'  # New field for input type

    # --- Timestamps and Status ---
    start_time_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time_utc: Optional[str] = None
    status: str = "In Progress"
    error_message: Optional[str] = None

    # --- Dynamic Data ---
    mkv_metadata: Dict[str, Any] = field(default_factory=dict)
    tiff_metadata: Dict[str, Any] = field(default_factory=dict)  # New field for TIFF metadata
    processing_details: Dict[str, Any] = field(default_factory=dict)

    def populate_from_mkv(self, playback: PyK4APlayback) -> int:
        """
        Extracts and adds metadata from the k4a playback object.
        Returns the camera FPS as an integer.
        """
        if not HAVE_PYK4A:
            raise ImportError("pyk4a is required for MKV processing")
            
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

    def populate_from_tiff(self, tiff_folder: Union[str, Path]) -> None:
        """
        Extracts and adds metadata from TIFF files.
        """
        from PIL import Image  # Import here to avoid top-level import

        tiff_folder = Path(tiff_folder)
        tiff_files = sorted(list(tiff_folder.glob("*.tiff")))
        if not tiff_files:
            tiff_files = sorted(list(tiff_folder.glob("*.tif")))
        
        if not tiff_files:
            raise ValueError(f"No TIFF files found in {tiff_folder}")

        # Get basic TIFF metadata from first frame
        with Image.open(tiff_files[0]) as img:
            self.tiff_metadata = {
                "frame_count": len(tiff_files),
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format,
                "info": dict(img.info)
            }

    def update_processing_detail(self, key: str, value: Any):
        """Updates a single key-value pair in the processing details."""
        self.processing_details[key] = value

    def set_status(self, status: str, error_message: Optional[str] = None):
        """Sets the final status and an optional error message."""
        self.status = status
        if error_message:
            self.error_message = error_message

    def finalize(self, frames_processed: int):
        """Finalizes the metadata before saving (e.g., sets end time)."""
        self.end_time_utc = datetime.now(timezone.utc).isoformat()
        self.update_processing_detail("frames_processed", frames_processed)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the metadata model to a dictionary suitable for JSON serialization."""
        return {
            "start_time_utc": self.start_time_utc,
            "end_time_utc": self.end_time_utc,
            "status": self.status,
            "error_message": self.error_message,
            "inputs": {
                "source": os.path.abspath(self.source_path),
                "input_type": self.input_type,
                "input_csv_path": os.path.abspath(self.input_csv_path),
            },
            "outputs": {
                "output_csv_path": os.path.abspath(self.output_csv_path),
                "video_path": os.path.abspath(self.video_path) if self.video_path else None,
                "metadata_path": os.path.abspath(self.metadata_path) if self.metadata_path else None,
            },
            "parameters": {
                "monitor": self.monitor,
                "input_type": self.input_type
            },
            "metadata": self.mkv_metadata if self.input_type == 'mkv' else self.tiff_metadata,
            "processing_details": self.processing_details,
        }