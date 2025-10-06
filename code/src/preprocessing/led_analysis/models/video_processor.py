import numpy as np
from typing import Dict, List, Any

# Assuming these imports are correct for your project structure
from preprocessing.common.data_access.video_mp4_manager import VideoMP4Manager, ColorFormat

class VideoProcessor:
    """
    Handles reliable video reading and processing by leveraging VideoMP4Manager.
    
    This version is modified to be a context manager, making it compatible with
    the 'with' statement for robust resource management.
    """
    def __init__(self, video_path: str):
        """
        Initializes the VideoProcessor.

        Args:
            video_path (str): The path to the video file.
        """
        # The VideoMP4Manager is the core engine for all video I/O.
        self._manager = VideoMP4Manager(video_path, color_format=ColorFormat.RGB)
        
        # Public properties are derived directly from the manager.
        self.video_path = str(self._manager.video_path)
        self.width = self._manager.width
        self.height = self._manager.height
        self.fps = self._manager.fps
        self.frame_count = self._manager.total_frames

    # --- START: CONTEXT MANAGER IMPLEMENTATION ---
    def __enter__(self):
        """
        Enables the use of the 'with' statement. Returns the instance itself.
        """
        # The resource (_manager) is already initialized in __init__.
        # We just need to return the instance for the 'as video:' part.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Ensures resources are cleaned up when exiting the 'with' block.
        """
        # Here you would add any explicit cleanup logic for VideoMP4Manager,
        # such as calling a .close() method if it exists.
        # For now, we'll signal that the context is being exited.
        # De-referencing the manager can help signal to the garbage collector.
        self._manager = None
    # --- END: CONTEXT MANAGER IMPLEMENTATION ---

    def get_properties(self) -> Dict[str, Any]:
        """
        Returns a dictionary of the video's properties.

        Returns:
            Dict[str, Any]: A dictionary containing frame width, height, fps, 
                            and total number of frames.
        """
        return {
            "frame_width": self.width,
            "frame_height": self.height,
            "fps": self.fps,
            "nframes": self.frame_count,
        }

    def get_frame(self, frame_index: int) -> np.ndarray | None:
        """
        Retrieves a single frame from the video using the manager's indexing.

        Args:
            frame_index (int): The index of the frame to retrieve.

        Returns:
            np.ndarray | None: The frame as a NumPy array, or None if the
                               index is out of bounds.
        """
        if not self._manager:
            raise RuntimeError("VideoProcessor has been closed and cannot be used.")
        try:
            return self._manager[frame_index]
        except IndexError:
            # Return None to maintain compatibility with the original method's signature.
            return None

    def get_montage_frame_indices(self, num_frames: int = 19) -> List[int]:
        """
        Gets a list of frame indices evenly spaced throughout the first half of the video.

        Args:
            num_frames (int): The number of frame indices to generate.

        Returns:
            List[int]: A list of evenly spaced frame indices.
        """
        # We scan only the first half of the video for the montage.
        effective_frame_count = self.frame_count // 2
        if effective_frame_count < 1:
            return [] # Avoids errors with very short videos
            
        return np.linspace(0, effective_frame_count - 1, num=num_frames, dtype=int).tolist()

    def extract_roi_from_video(self, roi_coords: Dict[str, Any]) -> List[np.ndarray]:
        """
        Extracts a Region of Interest (ROI) from every frame of the video.

        Args:
            roi_coords (Dict[str, Any]): A dictionary with 'x', 'y', 'width', and 'height'.

        Returns:
            List[np.ndarray]: A list of the cropped ROI frames.
        """
        if not self._manager:
            raise RuntimeError("VideoProcessor has been closed and cannot be used.")

        # Calculate the slicing coordinates for the ROI.
        x_start = roi_coords['x']
        y_start = roi_coords['y']
        x_end = x_start + roi_coords['width']
        y_end = y_start + roi_coords['height']

        roi_frames = []
        # The VideoMP4Manager is iterable, providing a clean way to process all frames.
        for frame in self._manager:
            # Crop the frame to the calculated ROI coordinates.
            cropped_frame = frame[y_start:y_end, x_start:x_end]
            roi_frames.append(cropped_frame)
            
        return roi_frames