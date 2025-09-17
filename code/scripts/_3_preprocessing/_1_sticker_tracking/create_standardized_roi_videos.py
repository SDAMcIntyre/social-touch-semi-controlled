# Standard library imports
import ast
import os
from pathlib import Path
from typing import Dict, List

# Third-party imports
import cv2
import numpy as np
import pandas as pd

# Third-party imports
from utils.should_process_task import should_process_task
from preprocessing.stickers_analysis import ROITrackedFileHandler
from preprocessing.common import VideoMP4Manager, ColorFormat


def create_windowed_video(
    video_frames: List[np.ndarray],
    output_video_path: str,
    roi_df: pd.DataFrame,
    fps: float = 30
):
    """
    Creates a cropped video from preloaded frames based on a DataFrame of ROIs.

    For each frame, it uses the corresponding ROI from the DataFrame to crop
    the frame. If an ROI is missing for a frame or the status is 'Black Frame',
    a black frame is written instead to ensure the output frame count matches the input.

    Args:
        video_frames: A list of video frames, where each frame is a NumPy array.
        output_video_path: Path where the new cropped video will be saved.
        roi_df: A pandas DataFrame with columns including 'frame_id', 'roi_x',
                'roi_y', 'roi_width', 'roi_height', and 'status'.
        fps: The frames per second for the output video.
    """
    # 1. Validate input
    if not video_frames:
        print("⚠️ Warning: Empty list of video frames provided. No video will be created.")
        return

    # 2. Get video properties for the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files

    # Determine output video dimensions from the first valid ROI.
    # We filter out 'Black Frame' statuses to find a tracked ROI for dimensions.
    valid_roi_df = roi_df[roi_df['status'] != 'Black Frame'].dropna(
        subset=['roi_width', 'roi_height']
    )
    if valid_roi_df.empty:
        print("⚠️ Error: No valid ROIs found in the DataFrame. Cannot determine output dimensions.")
        return

    first_valid_roi = valid_roi_df.iloc[0]
    out_w = int(first_valid_roi['roi_width'])
    out_h = int(first_valid_roi['roi_height'])
    
    # Create a black frame template based on the determined dimensions
    black_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    # 3. Create the VideoWriter object
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise IOError(f"Error opening output video writer for: '{output_video_path}'")

    print(f"🎬 Processing video for '{output_video_path}' with frame size ({out_w}, {out_h})...")

    # For efficient lookup, set 'frame_id' as the index of the DataFrame
    roi_df_indexed = roi_df.set_index('frame_id')

    # 4. Iterate through preloaded frames
    for frame_idx, frame in enumerate(video_frames):
        try:
            # Get the ROI row for the current frame
            roi_row = roi_df_indexed.loc[frame_idx]

            # 5. Crop the frame if the ROI is valid, otherwise write a black frame
            if roi_row['status'] != 'Black Frame':
                x = int(roi_row['roi_x'])
                y = int(roi_row['roi_y'])
                w = int(roi_row['roi_width'])
                h = int(roi_row['roi_height'])

                # Clamp coordinates to be within the frame boundaries
                frame_h, frame_w, _ = frame.shape
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(frame_w, x + w)
                y2 = min(frame_h, y + h)

                # Crop the frame
                cropped_frame = frame[y1:y2, x1:x2]

                # Resize cropped frame to the target dimensions in case clamping changed the size.
                if cropped_frame.size > 0:
                    resized_cropped_frame = cv2.resize(cropped_frame, (out_w, out_h))
                    writer.write(resized_cropped_frame)
                else:
                    writer.write(black_frame)
            else:
                # The status is 'Black Frame', so write a black frame
                writer.write(black_frame)

        except KeyError:
            # If frame_idx is not in the ROI data, write a black frame.
            # This ensures the output video has the same number of frames as the input.
            writer.write(black_frame)

    # 6. Release resources
    writer.release()
    print(f"✅ Finished. Video saved to '{output_video_path}'")


def create_standardized_roi_videos(
    roi_file_path: str,
    video_path: str,
    output_video_base_path: str,
    *,
    force_processing: bool = False
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
    rois_df_dict: Dict = ROITrackedFileHandler(roi_file_path).load_all_data()
    
    # Use pathlib for robust path manipulation
    output_path = Path(output_video_base_path)
    output_dir = output_path.parent
    base_name = output_path.stem
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    video_manager = VideoMP4Manager(video_path)
    video_manager.color_format = ColorFormat.BGR
    frames = None
    
    # Process each object
    for obj_name, roi_df in rois_df_dict.items():
        # Create a unique output path for this object's video
        output_video_path = output_dir / f"{base_name}_{obj_name}.mp4"

        if not should_process_task(
            output_paths=output_video_path,
            input_paths=[roi_file_path, video_path],
            force=force_processing
        ):
            print(f"Video {output_video_path} has been already processed. Skipping...")
            continue

        if not frames:
            video_manager.preload()
            frames = video_manager.get_frames()

        create_windowed_video(
            video_frames=frames,
            roi_df=roi_df,
            output_video_path=str(output_video_path)
        )