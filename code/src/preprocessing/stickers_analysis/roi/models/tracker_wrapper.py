import cv2
import sys
import numpy as np
import sys
from typing import Optional, Tuple, Any

class TrackerWrapper:
    """
    A robust wrapper for OpenCV tracking algorithms, handling API fragmentation
    across different OpenCV versions (standard vs. contrib/legacy).
    """

    def __init__(self, tracker_name: str = "CSRT"):
        """
        Initialize the tracker using a factory approach that resolves 
        API location dynamically.

        Args:
            tracker_name (str): The name of the tracker algorithm (e.g., "CSRT", "KCF").
        
        Raises:
            ValueError: If the tracker name is unsupported.
            ImportError: If the required OpenCV modules (contrib) are missing.
        """
        self.tracker_name = tracker_name.upper()
        self.tracker = self._create_tracker(self.tracker_name)

    def _create_tracker(self, name: str) -> Any:
        """
        Internal factory method to instantiate the correct OpenCV tracker 
        regardless of API version.
        """
        # Map string names to specific tracker creators
        # Note: In modern OpenCV (4.5+), these often reside in cv2.legacy
        tracker_types = {
            "CSRT": "TrackerCSRT_create",
            "KCF": "TrackerKCF_create",
            "MOSSE": "TrackerMOSSE_create",
            "MIL": "TrackerMIL_create",
        }

        if name not in tracker_types:
            raise ValueError(f"Unsupported tracker: {name}. Supported: {list(tracker_types.keys())}")

        method_name = tracker_types[name]

        # 1. Try accessing directly from cv2 (Older versions < 4.5)
        if hasattr(cv2, method_name):
            return getattr(cv2, method_name)()

        # 2. Try accessing from cv2.legacy (Newer versions 4.5+)
        # This requires opencv-contrib-python
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, method_name):
            return getattr(cv2.legacy, method_name)()

        # 3. If neither works, the environment is likely missing opencv-contrib-python
        raise ImportError(
            f"Could not instantiate {name} tracker using method '{method_name}'. "
            "This usually indicates that 'opencv-contrib-python' is not installed "
            "or the OpenCV version is incompatible. "
            "Try running: pip install opencv-contrib-python"
        )

    def init(self, frame: np.ndarray, roi: Tuple[int, int, int, int]) -> None:
        """
        Initializes the tracker with the first frame and a region of interest.

        Args:
            frame: The first frame (must be BGR for OpenCV).
            roi: The initial bounding box tuple (x, y, w, h).
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame cannot be empty.")
            
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
        if frame is None or frame.size == 0:
            return False, None

        success, box = self.tracker.update(frame)
        
        if success:
            # OpenCV returns box as float, convert to int tuple
            return True, tuple(map(int, box))
        
        return False, None

def diagnose_opencv():
    print(f"Python Version: {sys.version}")
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"OpenCV File Path: {cv2.__file__}")
    
    # Check for legacy module specifically
    has_legacy = hasattr(cv2, 'legacy')
    print(f"Has 'cv2.legacy' module: {has_legacy}")

    # Check for CSRT in root
    has_csrt_root = hasattr(cv2, 'TrackerCSRT_create')
    print(f"Has 'cv2.TrackerCSRT_create': {has_csrt_root}")

    # Check for CSRT in legacy
    has_csrt_legacy = False
    if has_legacy:
        has_csrt_legacy = hasattr(cv2.legacy, 'TrackerCSRT_create')
    print(f"Has 'cv2.legacy.TrackerCSRT_create': {has_csrt_legacy}")

    if not (has_csrt_root or has_csrt_legacy):
        print("\n[CRITICAL FAILURE] The tracker is missing.")
        print("Reason: The currently loaded 'cv2' does not contain contrib modules.")
        print("Action: Ensure you uninstalled 'opencv-python' and have ONLY 'opencv-contrib-python'.")
    else:
        print("\n[SUCCESS] Tracker detected. The Wrapper code will now work.")

if __name__ == "__main__":
    diagnose_opencv()