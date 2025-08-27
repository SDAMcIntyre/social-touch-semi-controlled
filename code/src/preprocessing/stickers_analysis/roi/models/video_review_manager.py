import cv2
import sys
import logging
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Union
import time 

from .roi_tracked_data import ROITrackedObjects


# 2. Define a custom exception for clear error handling
class VideoLoadError(Exception):
    """Custom exception raised for video loading failures."""
    pass

# 3. Create a context manager for safe resource handling
@contextmanager
def video_capture(source: str):
    """
    A context manager for cv2.VideoCapture to ensure resources are always released.
    This is a more Pythonic and robust alternative to a try...finally block.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        # Raise an error that the retry mechanism can catch
        raise IOError(f"Could not open video source: {source}")
    try:
        yield cap
    finally:
        cap.release()
        logging.info(f"Video capture for '{source}' released.")


class VideoReviewManager:
    """
    Handles loading a video, optionally associating it with tracking data for multiple objects,
    and playing it back with overlays for review.
    
    Note: This implementation loads the entire video into memory upon initialization.
    See the architectural review for suggestions on a more memory-efficient approach.
    """
    STATUS_COLORS = {
        "Tracking": (0, 255, 0),      # Green
        "Out of Frame": (0, 0, 255),  # Red
        "Failure": (0, 165, 255),     # Orange
        "Re-initialized": (255, 255, 0),# Cyan/Aqua
        "default": (255, 0, 255)      # Magenta
    }

    def __init__(self,
                 video_source: Union[str, Path, List[np.ndarray]],
                 tracking_history: ROITrackedObjects = None,
                 as_bgr: bool = True):
        """
        Initializes the reviewer by loading video frames and optional tracking data.

        Args:
            video_source (Union[str, Path, List[np.ndarray]]): Path to the video file or a list/array of pre-loaded frames.
            tracking_history (ROITrackedObjects, optional): A dictionary where keys are object names
                                                             and values are pandas DataFrames with tracking
                                                             history. Defaults to None.
            as_bgr (bool): If True, loads frames in BGR format; otherwise, converts to RGB.
        """
        self._is_bgr = as_bgr
        self.frames = self._load_frames(video_source)
        if not self.frames:
            raise ValueError("Video source could not be loaded or resulted in no frames.")
        
        self.tracking_history: ROITrackedObjects = tracking_history if tracking_history is not None else {}
            
        self.total_frames = len(self.frames)
        self.frame_height, self.frame_width = self.frames[0].shape[:2]

    def get_frame(self, frame_index: int, ignore_tracking_history: bool = False) -> np.ndarray:
        """
        Returns a single frame with its tracking annotations drawn on it.
        If no tracking history is available, it returns the original frame.
        
        Args:
            frame_index (int): The index of the frame to retrieve.
            ignore_tracking_history (bool): If True, returns the raw frame without annotations.

        Returns:
            np.ndarray: A copy of the frame with annotations. Returns None if the frame index is invalid.
        """
        if not (0 <= frame_index < self.total_frames):
            return None

        frame = self.frames[frame_index].copy() # Work on a copy
        
        if not self.tracking_history or ignore_tracking_history:
            return frame

        # Iterate over each tracked object and its history.
        for object_name, history_df in self.tracking_history.items():
            # Find the tracking data for the current frame for this specific object.
            track_data = history_df.loc[history_df['frame_id'] == frame_index]

            if not track_data.empty:
                result = track_data.iloc[0]
                status = result.get('status', 'No Status')
                base_status = status.split(':')[0]
                color = self.STATUS_COLORS.get(base_status, self.STATUS_COLORS["default"])

                if all(col in result and pd.notna(result[col]) for col in ['roi_x', 'roi_y', 'roi_width', 'roi_height']):
                    x, y, w, h = map(int, [result['roi_x'], result['roi_y'], result['roi_width'], result['roi_height']])
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    label = f"{object_name}: {status}"
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    label_bg_y2 = y - 10
                    label_bg_y1 = label_bg_y2 - text_height - 5
                    label_bg_y1 = max(label_bg_y1, 0)
                    
                    cv2.rectangle(frame, (x, label_bg_y1), (x + text_width, label_bg_y2), color, cv2.FILLED)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        
        return frame

    def play(self, fps=30):
        """
        Plays the video with tracking results overlaid in a window.
        Press 'q' to quit the playback.
        
        Args:
            fps (int): The playback speed in frames per second.
        """
        print("\nStarting playback... Press 'q' to exit.")
        delay = int(1000 / fps)
        
        for i in range(self.total_frames):
            frame = self.get_frame(i)
            if frame is None:
                continue

            cv2.imshow("Video Review", frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print("Playback finished.")

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Centralized method to handle frame processing, like color conversion.
        This respects the DRY (Don't Repeat Yourself) principle.
        """
        if not self._is_bgr and frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def _load_from_path(self, video_path: str, retry_attempts: int, retry_delay: float) -> List[np.ndarray]:
        """
        Loads all frames from a video file into memory with a retry mechanism.
        """
        frames = []
        for attempt in range(retry_attempts):
            try:
                # Use the context manager for safe handling
                with video_capture(video_path) as cap:
                    logging.info(f"Successfully opened '{video_path}'. Starting frame extraction.")
                    
                    # Use a robust while loop instead of relying on frame count
                    while True:
                        success, frame = cap.read()
                        if not success:
                            # End of video or read error
                            break
                        
                        processed_frame = self._process_frame(frame)
                        frames.append(processed_frame)

                    logging.info(f"Finished loading {len(frames)} frames from '{video_path}'.")
                    return frames # Success!

            except IOError as e:
                logging.warning(f"Attempt {attempt + 1}/{retry_attempts} failed: {e}")
                if attempt + 1 == retry_attempts:
                    # Raise the final, specific error after all retries fail
                    raise VideoLoadError(f"Failed to open video after {retry_attempts} attempts.") from e
                
                # Wait with exponential backoff before the next retry
                time.sleep(retry_delay * (2 ** attempt))
        
        return [] # Should not be reached due to the raise, but here for completeness


    def _load_frames(self, video_source: Union[str, Path, List, np.ndarray],
                            retry_attempts: int = 3, retry_delay: float = 1.0) -> List[np.ndarray]:
        """
        Loads a complete video into a list of frames in memory.

        This method acts as a dispatcher, handling either a path or a pre-loaded
        list of frames. It is designed to be error-resilient for file paths.

        Args:
            video_source: The source of the video (path or list/array of frames).
            retry_attempts: Number of times to retry opening a video file.
            retry_delay: Initial delay in seconds between retries.

        Returns:
            A list of all frames from the video as NumPy arrays.

        Raises:
            VideoLoadError: If a video file cannot be opened after all retries.
            TypeError: If the video_source is of an unsupported type.
        """
        if isinstance(video_source, (str, Path)):
            return self._load_from_path(str(video_source), retry_attempts, retry_delay)
        
        elif isinstance(video_source, (list, np.ndarray)):
            logging.info(f"Processing {len(video_source)} pre-loaded frames.")
            # Use a list comprehension for a concise transformation
            return [self._process_frame(frame) for frame in video_source]
        
        else:
            raise TypeError("video_source must be a path or a list/numpy array of frames.")

