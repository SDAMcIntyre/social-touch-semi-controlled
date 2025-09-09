import logging
import sys
from pathlib import Path

# Setup a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from .utils.KinectLEDRegionOfInterestMP4 import KinectLEDRegionOfInterestMP4  # noqa: E402


def define_led_roi(
        video_path: Path, 
        output_path: Path,
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
    if output_path.exists():
        logging.info(f"Output file already exists. Skipping ROI definition for '{video_path.name}'.")
        return True

    logging.info(f"Starting LED ROI definition for '{video_path.name}'.")
    
    try:
        # 2. Ensure output directory exists (using pathlib)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 3. Core Logic with improved control flow
        led_roi = KinectLEDRegionOfInterestMP4(video_path)
        led_roi.initialise_video()

        while True:
            frame_number = led_roi.get_frame_id_with_led()
            
            # Exit condition: No more frames with a detectable LED
            if frame_number is None:
                logging.warning(f"Could not find a suitable frame with an LED in '{video_path.name}'.")
                break  # Exit the loop

            led_roi.set_reference_frame(frame_number)
            
            # Attempt to finalize the ROI
            if led_roi.draw_led_location() is not None:
                logging.info(f"Successfully defined LED ROI using frame {frame_number}.")
                led_roi.save_result_metadata(output_path)
                logging.info(f"Result metadata saved to '{output_path}'.")
                return True # Success

        # This part is reached if the loop finishes without saving (e.g., no LED found)
        logging.error(f"Failed to define and save LED ROI for '{video_path.name}'.")
        return False

    # 4. Robust Error Handling
    except FileNotFoundError:
        logging.error(f"Video file not found at '{video_path}'.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during ROI definition for '{video_path.name}': {e}")
        return False