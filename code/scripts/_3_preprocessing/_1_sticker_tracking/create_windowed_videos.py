# Standard library imports
import ast
import os
from pathlib import Path
from typing import Dict, Any, List

# Third-party imports
import cv2
import numpy as np
import pandas as pd

from .utils.roi_to_position_funcs import load_video_frames_bgr


def load_unified_rois(roi_file_path: str) -> Dict[str, pd.Series]:
    """
    Loads and parses the unified ROI data from a CSV file.

    Args:
        roi_file_path: The path to the CSV file containing unified ROI data.
                       The file is expected to have an index column (frame number)
                       and one column per tracked object.

    Returns:
        A dictionary mapping object names (str) to a pandas Series of their ROIs.
        Each ROI is a list [x, y, w, h].
    """
    print(f"🔄 Loading unified ROI data from '{roi_file_path}'...")
    if not os.path.exists(roi_file_path):
        raise FileNotFoundError(f"ROI file not found at '{roi_file_path}'.")

    # Read the CSV. The first column is automatically treated as the index.
    df = pd.read_csv(roi_file_path)

    unified_roi_results = {}
    for obj_name in df.columns:
        # The ROIs are stored as strings, e.g., "[10, 20, 30, 40]".
        # We need to convert them back to lists of numbers.
        # ast.literal_eval is a safe way to evaluate a string containing a Python literal.
        def safe_eval_roi(roi_str: Any) -> Any:
            if isinstance(roi_str, str):
                try:
                    return ast.literal_eval(roi_str)
                except (ValueError, SyntaxError):
                    return np.nan # Return NaN if string is not a valid list
            return roi_str # Keep NaNs as they are

        unified_roi_results[obj_name] = df[obj_name].apply(safe_eval_roi)

    print(f"✅ Successfully loaded and parsed ROI data for objects: {list(unified_roi_results.keys())}")
    return unified_roi_results





import cv2
import numpy as np
import pandas as pd
from typing import List

def create_windowed_video(
    video_frames: List[np.ndarray],
    output_video_path: str,
    roi_series: pd.Series,
    fps: float = 30
):
    """
    Creates a cropped video from preloaded frames based on a series of ROIs.

    For each frame, it uses the corresponding ROI from the series to crop
    the frame. If an ROI is missing for a frame, a black frame is written
    instead to ensure the output frame count matches the input.

    Args:
        video_frames: A list of video frames, where each frame is a NumPy array.
        output_video_path: Path where the new cropped video will be saved.
        roi_series: A pandas Series where the index is the frame number and
                    the value is the ROI list [x, y, w, h].
        fps: The frames per second for the output video.
    """
    # 1. Validate input
    if not video_frames:
        print("⚠️ Warning: Empty list of video frames provided. No video will be created.")
        return

    # 2. Get video properties for the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files

    # Determine output video dimensions from the first valid ROI.
    # Since all ROIs for an object are unified, we can take any valid one.
    valid_rois = roi_series.dropna()
    if valid_rois.empty:
        print("⚠️ Error: No valid ROIs found in the series. Cannot determine output dimensions.")
        return
        
    first_valid_roi = valid_rois.iloc[0]
    out_w, out_h = int(first_valid_roi[2]), int(first_valid_roi[3])
    # If the crop results in an empty frame, write a black frame
    black_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    # 3. Create the VideoWriter object
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise IOError(f"Error opening output video writer for: '{output_video_path}'")

    print(f"🎬 Processing video for '{output_video_path}' with frame size ({out_w}, {out_h})...")
    
    # 4. Iterate through preloaded frames
    for frame_idx, frame in enumerate(video_frames):
        # Get the ROI for the current frame
        roi = roi_series.get(frame_idx)

        # 5. Crop the frame if the ROI is valid, otherwise write a black frame
        if isinstance(roi, list) and len(roi) == 4:
            x, y, w, h = map(int, roi) # Ensure coordinates are integers
            
            # Clamp coordinates to be within the frame boundaries to prevent errors
            frame_h, frame_w, _ = frame.shape
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame_w, x + w)
            y2 = min(frame_h, y + h)

            # Crop the frame
            cropped_frame = frame[y1:y2, x1:x2]

            # The cropped frame must be resized to the target dimensions (out_w, out_h)
            # in case clamping changed the size.
            if cropped_frame.size > 0:
                resized_cropped_frame = cv2.resize(cropped_frame, (out_w, out_h))
                writer.write(resized_cropped_frame)
            else:
                writer.write(black_frame)
        else:
            # This ensures the output video has the same number of frames as the input.
            writer.write(black_frame)

    # 6. Release resources
    writer.release()
    print(f"✅ Finished. Video saved to '{output_video_path}'")



def create_windowed_videos(
    roi_file_path: str,
    video_path: str,
    output_video_base_path: str
):
    """
    Main orchestrator to generate a windowed video for each object from preloaded frames.

    Args:
        roi_file_path: Path to the CSV file with unified ROI data.
        video_frames: A list of video frames, where each frame is a NumPy array.
        fps: The frames per second of the source video.
        output_video_base_path: The base path for output files. The object name
                                and extension will be appended. E.g., 'results/run1'
                                becomes 'results/run1_objectA.mp4'.
    """
    # Load all ROI data from the file
    unified_rois = load_unified_rois(roi_file_path)

    # Use pathlib for robust path manipulation
    output_path = Path(output_video_base_path)
    output_dir = output_path.parent
    base_name = output_path.stem
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    video_frames = None
    # Process each object
    for obj_name, roi_series in unified_rois.items():
        # Create a unique output path for this object's video
        output_video_path = output_dir / f"{base_name}_{obj_name}.mp4"
        
        if os.path.exists(output_video_path):
            print(f"Video {output_video_path} has been already processed. Skipping...")
            continue

        if not video_frames:
            video_frames = load_video_frames_bgr(video_path)

        create_windowed_video(
            video_frames=video_frames,
            output_video_path=str(output_video_path),
            roi_series=roi_series
        )