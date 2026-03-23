import json
import logging
import tkinter as tk
from pathlib import Path
from typing import Optional

import cv2

from preprocessing.common import FrameROISquare, VideoFrameSelector

logger = logging.getLogger(__name__)


def select_roi_on_video(
    video_path: Path,
    existing_roi: Optional[dict] = None,
) -> Optional[dict]:
    """
    Interactive ROI selection on a navigable video frame.

    Step 1: the user navigates frames using the shared VideoFrameSelector widget.
    Step 2: FrameROISquare opens on that frame for rectangle drawing.

    Args:
        video_path: Path to the input video.
        existing_roi: Optional pre-existing sidecar dict with keys
                      x_min, x_max, y_min, y_max (int or None).
                      Used to pre-populate the drawn rectangle.

    Returns:
        dict with integer keys x_min, x_max, y_min, y_max, or None if cancelled.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None

    frame_num = None
    root = tk.Tk()
    root.geometry("1x1+10000+10000")  # off-screen; withdraw() hides transient children on Windows
    try:
        selector = VideoFrameSelector(root, cap, title=f"Select Frame for Hand Tracking ROI — {video_path.name}")
        frame_num = selector.select_frame()
    finally:
        root.destroy()

    if frame_num is None:
        cap.release()
        logger.info("Frame selection cancelled.")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    success, frame_bgr = cap.read()
    cap.release()

    if not success or frame_bgr is None:
        logger.error("Failed to read selected frame from video.")
        return None

    frame_h, frame_w = frame_bgr.shape[:2]

    # Convert existing sidecar format to FrameROISquare predefined_roi format
    predefined = None
    if existing_roi:
        x_min = existing_roi.get("x_min") or 0
        x_max = existing_roi.get("x_max") or frame_w
        y_min = existing_roi.get("y_min") or 0
        y_max = existing_roi.get("y_max") or frame_h
        predefined = {
            "x": int(x_min),
            "y": int(y_min),
            "width": int(x_max - x_min),
            "height": int(y_max - y_min),
        }

    roi_selector = FrameROISquare(
        frame_bgr,
        is_rgb=False,
        window_title=f"Draw Hand Tracking ROI — {video_path.name}  |  left-click drag, then Proceed / Enter",
        predefined_roi=predefined,
    )
    roi_selector.run()

    rect = roi_selector.get_roi_data()
    if rect is None:
        logger.info("ROI selection cancelled — no rectangle drawn.")
        return None

    x, y, w, h = rect["x"], rect["y"], rect["width"], rect["height"]
    return {
        "x_min": int(x),
        "x_max": int(x + w),
        "y_min": int(y),
        "y_max": int(y + h),
    }
