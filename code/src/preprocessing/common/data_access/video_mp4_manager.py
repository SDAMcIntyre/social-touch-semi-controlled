import cv2
import numpy as np
import sys
from typing import List, Optional

# --- MODULE 1: Video Handling ---
class VideoMP4Manager:
    """
    Handles video file operations, providing frames on demand without
    loading the entire video into memory.
    """
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.capture = cv2.VideoCapture(video_path)
        if not self.capture.isOpened():
            raise IOError(f"Cannot open video at {self.video_path}")

        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        fourcc_int = int(self.capture.get(cv2.CAP_PROP_FOURCC))
        self.fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])


    def get_frame(self, frame_num: int) -> Optional[np.ndarray]:
        """
        Retrieves a specific frame from the video file.

        Args:
            frame_num: The 0-indexed frame number to retrieve.

        Returns:
            The frame as a NumPy array in RGB format, or None if reading fails.
        """
        if not (0 <= frame_num < self.total_frames):
            return None
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        success, frame = self.capture.read()
        if success:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def get_frames_range(self, start: int, end: int) -> List[np.ndarray]:
        """Loads a specific range of frames with progress feedback."""
        frames = []
        # Ensure 'end' does not exceed the total number of frames
        end = min(end, self.total_frames)
        total = end - start
        if total <= 0:
            return []
            
        print(f"Loading frames {start} to {end-1}...")
        for i, frame_num in enumerate(range(start, end)):
            frame = self.get_frame(frame_num)
            if frame is not None:
                frames.append(frame)
            
            # Simple progress reporting
            percentage = ((i + 1) / total) * 100
            sys.stdout.write(f"\rProgress: {i+1}/{total} ({percentage:.2f}%)")
            sys.stdout.flush()
        print("\nLoading complete.")
        return frames

    def get_frames(self) -> List[np.ndarray]:
        """
        Retrieves all frames from the video.

        Returns:
            A list containing all frames of the video as NumPy arrays.
        """
        return self.get_frames_range(0, self.total_frames)

    def release(self):
        """Releases the video capture object."""
        self.capture.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()