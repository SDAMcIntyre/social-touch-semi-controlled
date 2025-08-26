import cv2
import numpy as np
from typing import Optional, Tuple

class TrackerWrapper:
    """A wrapper for the OpenCV tracking algorithm."""
    def __init__(self, tracker_name: str = "CSRT"):
        # Using a factory pattern allows for easy extension to other trackers
        if tracker_name == "CSRT":
            self.tracker = cv2.TrackerCSRT_create()
        else:
            # In a real application, you might support KCF, MOSSE, etc.
            raise ValueError(f"Unsupported tracker: {tracker_name}")
            
    def init(self, frame: np.ndarray, roi: Tuple[int, int, int, int]):
        """
        Initializes the tracker with the first frame and a region of interest.

        Args:
            frame: The first frame (must be BGR for OpenCV).
            roi: The initial bounding box tuple (x, y, w, h).
        """
        self.tracker.init(frame, roi)

    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        """
        Updates the tracker with a new frame.

        Args:
            frame: The next frame in the sequence (must be BGR).

        Returns:
            A tuple (success, roi) where success is a boolean and roi is the
            new bounding box tuple (x, y, w, h).
        """
        success, box = self.tracker.update(frame)
        if success:
            return True, tuple(map(int, box))
        return False, None