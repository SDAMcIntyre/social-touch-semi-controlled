import json
import logging
from pathlib import Path
from typing import Optional

import cv2

from preprocessing.common import VideoMP4Manager, FrameROISquare

logger = logging.getLogger(__name__)


def _select_frame(video_manager: VideoMP4Manager) -> Optional[object]:
    """
    Display video frames with keyboard navigation. Returns the selected BGR frame
    or None if the user cancels.

    Controls:
        a / ← : previous frame
        d / → : next frame
        q      : jump back 10% of total frames
        e      : jump forward 10% of total frames
        Enter / Space : confirm selected frame
        ESC    : cancel
    """
    n_frames = len(video_manager)
    if n_frames == 0:
        return None

    idx = 0
    window_name = "Select Frame for ROI  |  a/d=prev/next  q/e=jump  Enter=confirm  ESC=cancel"
    cv2.namedWindow(window_name)

    while True:
        frame_bgr = video_manager[idx]  # defaults to BGR

        overlay = frame_bgr.copy()
        label = f"Frame {idx + 1} / {n_frames}"
        cv2.putText(overlay, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(window_name, overlay)

        key = cv2.waitKey(50) & 0xFF

        if key == 27:  # ESC — cancel
            cv2.destroyWindow(window_name)
            return None
        elif key in (13, 32):  # Enter or Space — confirm
            cv2.destroyWindow(window_name)
            return frame_bgr
        elif key == ord('a'):  # prev frame
            idx = max(0, idx - 1)
        elif key == ord('d'):  # next frame
            idx = min(n_frames - 1, idx + 1)
        elif key == ord('q'):  # jump back 10%
            idx = max(0, idx - max(1, n_frames // 10))
        elif key == ord('e'):  # jump forward 10%
            idx = min(n_frames - 1, idx + max(1, n_frames // 10))

    cv2.destroyWindow(window_name)
    return None


def select_roi_on_video(
    video_path: Path,
    existing_roi: Optional[dict] = None,
) -> Optional[dict]:
    """
    Interactive ROI selection on a navigable video frame.

    Step 1: the user navigates frames and confirms one with Enter.
    Step 2: FrameROISquare opens on that frame for rectangle drawing.

    Args:
        video_path: Path to the input video.
        existing_roi: Optional pre-existing sidecar dict with keys
                      x_min, x_max, y_min, y_max (int or None).
                      Used to pre-populate the drawn rectangle.

    Returns:
        dict with integer keys x_min, x_max, y_min, y_max, or None if cancelled.
    """
    video_manager = VideoMP4Manager(video_path)

    frame_bgr = _select_frame(video_manager)
    if frame_bgr is None:
        logger.info("Frame selection cancelled.")
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
        window_title="Draw Hand Tracking ROI  |  left-click drag, then Proceed / Enter",
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
