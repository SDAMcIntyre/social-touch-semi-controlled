import cv2
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Union

from .roi_tracked_data import ROITrackedObjects

class VideoReviewManager:
    """
    Handles loading a video, optionally associating it with tracking data for multiple objects,
    and playing it back with overlays for review.
    
    Note: This implementation loads the entire video into memory upon initialization.
    See the architectural review for suggestions on a more memory-efficient approach.
    """
    STATUS_COLORS = {
        "Tracking": (0, 255, 0),       # Green
        "Out of Frame": (0, 0, 255),   # Red
        "Failure": (0, 165, 255),      # Orange
        "Re-initialized": (255, 255, 0),# Cyan/Aqua
        "default": (255, 0, 255)       # Magenta
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
                                                             history. Each DataFrame must conform to the
                                                             ROI_TRACKED_SCHEMA. Defaults to None.
            as_bgr (bool): If True, loads frames in BGR format; otherwise, converts to RGB.
        """
        self._is_bgr = as_bgr
        self.frames = self._load_frames(video_source)
        if not self.frames:
            raise ValueError("Video source resulted in no frames.")
        
        # MODIFICATION: Default tracking_history is now an empty dictionary.
        self.tracking_history: ROITrackedObjects = tracking_history if tracking_history is not None else {}
            
        self.total_frames = len(self.frames)
        self.frame_height, self.frame_width = self.frames[0].shape[:2]

    def get_frame(self, frame_index: int, ignore_tracking_history: bool = False) -> np.ndarray:
        """
        Returns a single frame with its tracking annotations drawn on it.
        If no tracking history is available, it returns the original frame.
        
        Args:
            frame_index (int): The index of the frame to retrieve.

        Returns:
            np.ndarray: A copy of the frame with annotations for all tracked objects present
                        in that frame. Returns None if the frame index is invalid.
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

            # If there's data for this object in this frame, draw it.
            if not track_data.empty:
                # Get the first (and only) row of data for this frame.
                result = track_data.iloc[0]
                
                status = result.get('status', 'No Status')
                
                # Determine color from the base status.
                base_status = status.split(':')[0]
                color = self.STATUS_COLORS.get(base_status, self.STATUS_COLORS["default"])

                # Check if box coordinates are valid numbers before drawing.
                if all(col in result and pd.notna(result[col]) for col in ['roi_x', 'roi_y', 'roi_width', 'roi_height']):
                    x, y, w, h = map(int, [result['roi_x'], result['roi_y'], result['roi_width'], result['roi_height']])
                    
                    # Draw the bounding box.
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Create and draw a label with the object name and status above the box.
                    label = f"{object_name}: {status}"
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    label_bg_y2 = y - 10
                    label_bg_y1 = label_bg_y2 - text_height - 5
                    # Ensure label background does not go off-screen
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

    def _load_frames(self, video_source):
        # This private method remains unchanged.
        if isinstance(video_source, (str, Path)):
            return self._load_from_path(str(video_source))
        elif isinstance(video_source, (list, np.ndarray)):
            return list(video_source)
        else:
            raise TypeError("video_source must be a path or a list/numpy array of frames.")

    def _load_from_path(self, video_path):
        # This private method remains unchanged.
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            raise IOError(f"Error: Could not open video file at {video_path}")

        total = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            return []

        frames = []
        print("Starting video processing...")
        for i in range(total):
            success, frame = video_capture.read()
            if not success:
                break
            
            if not self._is_bgr:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                frames.append(frame)

            percentage = ((i + 1) / total) * 100
            sys.stdout.write(f"\rLoading frames: {i+1}/{total} ({percentage:.2f}%)")
            sys.stdout.flush()

        video_capture.release()
        print("\nVideo processing complete.")
        return frames