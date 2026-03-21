import json
import logging
from pathlib import Path
from typing import Optional

from preprocessing.motion_analysis.hand_tracking.gui.roi_definition_gui import select_roi_on_video
from utils.should_process_task import should_process_task

logger = logging.getLogger(__name__)


def define_hand_tracking_roi(
    rgb_video_path: Path,
    output_dir: Path,
    *,
    force_processing: bool = False,
) -> Optional[Path]:
    """
    Launch an interactive GUI to define a static ROI crop region for hand tracking.

    Saves a sidecar JSON file ``{video_stem}_handmodel_roi.json`` to output_dir.
    The auto pipeline reads this file (if present) and crops the video before
    uploading it to the HaMeR API, narrowing the search space for hand detection.

    Sidecar JSON format:
        {
          "x_min": int,   -- left crop boundary  (pixel column, inclusive)
          "x_max": int,   -- right crop boundary (pixel column, exclusive)
          "y_min": int,   -- top crop boundary   (pixel row, inclusive)
          "y_max": int    -- bottom crop boundary (pixel row, exclusive)
        }

    null values are supported when the file is edited by hand:
    a null value means "use the full extent on that edge" (no cropping on that side).
    The GUI always writes integer values.

    Args:
        rgb_video_path:  Path to the source video.
        output_dir:      Directory where the sidecar JSON will be saved
                         (typically kinematics_analysis/).
        force_processing: If True, re-run even when output already exists and
                          is newer than the input.

    Returns:
        Path to the saved JSON, or None if the user cancelled.
    """
    output_path = output_dir / (rgb_video_path.stem + "_handmodel_roi.json")

    if not should_process_task(
        input_paths=[rgb_video_path],
        output_paths=[output_path],
        force=force_processing,
    ):
        logger.info(f"Skipping: {output_path} is up to date.")
        return output_path

    # Pre-populate the GUI with any existing ROI so the user can refine it.
    existing_roi = None
    if output_path.exists():
        try:
            with open(output_path) as f:
                existing_roi = json.load(f)
        except Exception:
            pass

    roi = select_roi_on_video(rgb_video_path, existing_roi=existing_roi)

    if roi is None:
        logger.warning("ROI definition cancelled — no file saved.")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(roi, f, indent=2)

    logger.info(f"Saved hand tracking ROI to: {output_path}")
    return output_path
