import cv2
import logging
import numpy as np
import pandas as pd
from typing import Union, List, Optional
from pathlib import Path

# Assuming the other classes are in these files respectively
from preprocessing.common import VideoMP4Manager, ColorFormat
from .roi_tracked_data import ROITrackedObjects

class VideoReviewManager(VideoMP4Manager):
    """
    Handles playback and review of a video with tracking data overlays.

    This class inherits from VideoMP4Manager and overrides its item access
    methods (`__getitem__`, `__array__`) to return frames with tracking
    history drawn on them by default. This allows for intuitive, array-like
    access to the final annotated video frames.
    """
    STATUS_COLORS = {
        "Tracking": (0, 255, 0),      # Green
        "Out of Frame": (0, 0, 255),  # Red
        "Failure": (0, 165, 255),     # Orange
        "Re-initialized": (255, 255, 0), # Cyan/Aqua
        "default": (255, 0, 255)      # Magenta
    }

    def __init__(self,
                 video_source: Union[str, Path],
                 tracking_history: ROITrackedObjects = None,
                 as_bgr: bool = True):
        """Initializes the reviewer by loading video metadata and tracking data."""
        color_fmt = ColorFormat.BGR if as_bgr else ColorFormat.RGB
        super().__init__(video_path=video_source, color_format=color_fmt)
        self.tracking_history: ROITrackedObjects = tracking_history if tracking_history is not None else {}

    def _draw_overlays(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """
        Private helper method to draw all tracking annotations for a given frame.
        """
        if not self.tracking_history:
            return frame

        # Iterate over tracking data and draw overlays.
        for object_name, history_df in self.tracking_history.items():
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
                    label_bg_y1 = max(0, label_bg_y2 - text_height - 5)
                    cv2.rectangle(frame, (x, label_bg_y1), (x + text_width, label_bg_y2), color, cv2.FILLED)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        return frame

    def __getitem__(self, index: Union[int, slice]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Overrides parent method to return frames with tracking history applied.
        This enables direct, intuitive access like `manager[10]` or `manager[10:20]`.
        """
        if isinstance(index, int):
            # 1. Get the raw frame from the parent class.
            raw_frame = super().__getitem__(index).copy()
            # 2. Apply overlays and return the result.
            return self._draw_overlays(raw_frame, index)

        elif isinstance(index, slice):
            # 1. Get the list of raw frames from the parent.
            raw_frames = super().__getitem__(index)
            # 2. Get the corresponding original frame indices for the slice.
            indices = range(*index.indices(len(self)))
            # 3. Apply overlays to each frame using its original index.
            return [self._draw_overlays(frame.copy(), frame_idx) for frame, frame_idx in zip(raw_frames, indices)]
        
        else:
            raise TypeError("Index must be an integer or slice")
            
    def __array__(self, dtype: Optional[np.dtype] = None) -> np.ndarray:
        """
        Overrides the NumPy array protocol to convert the *entire video with overlays*
        into a single NumPy array. Called via `np.array(manager)`.
        """
        logging.info("Converting VideoReviewManager to a NumPy array with tracking overlays...")
        # Use our overridden slice-based __getitem__ to get all frames with annotations.
        all_frames_with_overlays = self[:] 
        array = np.stack(all_frames_with_overlays, axis=0)
        
        if dtype is not None:
            return array.astype(dtype, copy=False)
        return array

    def get_frame(self, frame_index: int, ignore_tracking_history: bool = False) -> np.ndarray:
        """
        Retrieves a single frame, providing an option to bypass overlay drawing.
        
        Args:
            frame_index (int): The index of the frame to retrieve.
            ignore_tracking_history (bool): If True, returns the raw frame without annotations.
                                            If False (default), returns the frame with annotations.
        Returns:
            np.ndarray: The requested frame.
        """
        if ignore_tracking_history:
            # Get the raw frame directly from the parent.
            return super().__getitem__(frame_index)
        else:
            # Use our default, intuitive behavior to get the frame with overlays.
            return self[frame_index]

    def play(self, fps: int = None):
        """Plays the video with tracking results overlaid in a window."""
        print("\nStarting playback... Press 'q' to exit.")
        playback_fps = fps if fps is not None else self.fps
        delay = int(1000 / playback_fps) if playback_fps > 0 else 1
        
        # This loop now transparently gets frames with overlays due to the
        # overridden __getitem__ method. We can iterate directly over the instance.
        for frame_to_show in self:
            cv2.imshow("Video Review", frame_to_show)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print("Playback finished.")