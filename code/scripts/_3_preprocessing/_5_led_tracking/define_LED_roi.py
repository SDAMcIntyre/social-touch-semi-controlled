import logging
import sys
from pathlib import Path

# Setup a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from utils.should_process_task import should_process_task  # noqa: E402

from preprocessing.led_analysis import (
    ROIManager,
    LEDROIFilesHandler
)


def define_led_roi(
        video_path: Path, 
        output_metadata_path: Path,
        *,
        force_processing: bool = False
) -> bool:
    """
    Identifies and saves the LED Region of Interest (ROI) from a video file.

    This function is idempotent: it checks if the output metadata file
    already exists. If it does, the function logs a message and skips
    processing.

    Args:
        video_path (Path): The path to the input MP4 video file.
        output_path (Path): The path where the output metadata (e.g., JSON) will be saved.

    Returns:
        bool: True if the process completed successfully (or was skipped), False on failure.
    """
    # 1. Idempotency Check (using pathlib and with logging)
    if not should_process_task(
         input_paths=video_path, 
         output_paths=[output_metadata_path], 
         force=force_processing):
        logging.info(f"Output file already exists. Skipping ROI definition for '{video_path.name}'.")
        return True

    logging.info(f"Starting LED ROI definition for '{video_path.name}'.")
    
    try:
        # 3. Core Logic with improved control flow
        led_roi = ROIManager(video_path)
        
        # ✅ Pass the specific title for this task
        led_roi.choose_parameters(window_title=f"Define LED ROI: {video_path.name}")
        
        fileHandler = LEDROIFilesHandler()
        fileHandler.save_metadata(output_metadata_path, led_roi.metadata)
        return True

    # 4. Robust Error Handling
    except FileNotFoundError:
        logging.error(f"Video file not found at '{video_path}'.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during ROI definition for '{video_path.name}': {e}")
        return False