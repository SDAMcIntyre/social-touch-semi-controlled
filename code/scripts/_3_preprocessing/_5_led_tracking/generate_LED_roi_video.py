import os
from typing import List, Optional
from pathlib import Path

from .utils.KinectLEDRegionOfInterestMP4 import KinectLEDRegionOfInterestMP4

def generate_LED_roi_video(
    video_path: Path,
    roi_path: Path,
    output_path: Path,
    *,
    force_processing: bool = False,
    use_specific_blocks: bool = False,
    specific_blocks: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Processes a single metadata file to find and extract an LED Region of
    Interest (ROI) from the corresponding video.

    This function performs the following steps:
    1.  Optionally filters files based on specific block names.
    2.  Skips processing if the output already exists and `force_processing` is False.
    3.  Uses the `KinectLEDRegionOfInterestMP4` class to load data,
        extract the ROI, and save it as a new .mp4 video.
    4.  Handles path corrections for cross-platform compatibility (e.g., Linux paths on Windows).

    Args:
        video_path (Path): The absolute path to the corresponding video file.
        roi_path (Path): The absolute path to the metadata file to process.
        output_path (Path): The absolute path for the output ROI video.
        force_processing (bool): If True, re-processes the file even if the
                             output already exists. Defaults to False.
        use_specific_blocks (bool): If True, only processes files whose names
                                contain a string from `specific_blocks`.
        specific_blocks (Optional[List[str]]): A list of substrings to filter
                                           filenames by.

    Returns:
        Optional[str]: The absolute path to the created ROI video file if
                       processing was successful, otherwise None.
    """
    file = roi_path.name

    # 1. Filter based on specific blocks
    if use_specific_blocks and specific_blocks:
        if not any(block in file for block in specific_blocks):
            print(f"INFO: Skipping '{file}' as it's not in a specific block.")
            return None

    # 2. Check if file already exists
    if not force_processing and output_path.exists():
        print(f"INFO: Output for '{file}' already exists. Skipping.")
        return str(output_path) # Return existing path

    # 3. Process the video
    try:
        led_roi = KinectLEDRegionOfInterestMP4()
        led_roi.load_metadata(roi_path)

        if not led_roi.led_in_frame:
            print(f"INFO: Skipping '{file}': LED not in frame.")
            return None
        
        # 4. Handle cross-platform path issues
        # This check corrects paths that may be stored in an absolute Linux format
        # when running the script on Windows.
        if os.path.sep == '\\' and led_roi.video_path.startswith('/'):
            print("DEBUG: Correcting Linux-style path for Windows OS.")
            led_roi.video_path = video_path

        # 5. Extract and save ROI
        led_roi.initialise_video()
        led_roi.set_reference_frame(led_roi.reference_frame_idx)
        led_roi.extract_roi()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        led_roi.save_roi_as_video(output_path)
        
        print(f"INFO: Successfully processed '{file}' -> '{output_path}'")
        return str(output_path)

    except Exception as e:
        # Using traceback to get more detailed error information
        import traceback
        print(f"ERROR: An error occurred while processing '{file}': {e}")
        traceback.print_exc()
        return None