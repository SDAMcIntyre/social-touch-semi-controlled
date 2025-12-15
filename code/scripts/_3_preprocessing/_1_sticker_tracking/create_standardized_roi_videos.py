# Standard library imports
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# Third-party imports
import cv2
import numpy as np
import pandas as pd

# Local application imports
from preprocessing.common import ColorFormat, VideoMP4Manager
from preprocessing.stickers_analysis import ROITrackedFileHandler
from utils.should_process_task import should_process_task


@contextmanager
def video_capture_manager(path: str) -> Iterator[cv2.VideoCapture]:
    """A context manager for cv2.VideoCapture to ensure it's always released."""
    cap = cv2.VideoCapture(path)
    try:
        yield cap
    finally:
        cap.release()


def success_video_file_quality_check(video_path: str) -> bool:
    """
    Verifies a video file by trying to open it and read the first frame.

    Args:
        video_path: The path to the video file to verify.

    Returns:
        True if the video is valid and readable, False otherwise.
    """
    try:
        print(f"🔎 Verifying video file: '{video_path}'...")
        with video_capture_manager(video_path) as cap:
            # Check 1: Was the file opened successfully?
            if not cap.isOpened():
                print("Verification failed: Could not open the video file.")
                return False

            # Check 2: Can we read at least one frame?
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Verification failed: Could not read a frame from the video.")
                return False

        print("✅ Verification successful.")
        return True
    except Exception as e:
        print(f"Verification failed with an exception: {e}")
        return False


@contextmanager
def video_writer_manager(
    path: str, fourcc: int, fps: float, frame_size: tuple[int, int]
) -> Iterator[cv2.VideoWriter]:
    """A context manager for cv2.VideoWriter to ensure it's always released."""
    writer = cv2.VideoWriter(path, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise IOError(f"Error opening video writer for: '{path}'")

    try:
        # Yield the writer object to the 'with' block
        yield writer
    finally:
        # This code is guaranteed to run, ensuring the file is finalized
        print("Finalizing video stream...")
        writer.release()


def create_windowed_video(
    video_manager: VideoMP4Manager, output_video_path: str, roi_df: pd.DataFrame
):
    """
    Creates a cropped video from a source, ensuring the output is never corrupted.

    This version uses a context manager for the VideoWriter and an atomic move
    operation to guarantee file integrity. Includes progress feedback.
    """
    # 1. Input Validation
    required_cols = ['frame_id', 'roi_x', 'roi_y', 'roi_width', 'roi_height', 'status']
    if not all(col in roi_df.columns for col in required_cols):
        raise ValueError(f"roi_df is missing required columns: {required_cols}")

    output_dir = Path(output_video_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Determine output dimensions from the first valid ROI
    valid_roi_df = roi_df[roi_df['status'] != 'Black Frame'].dropna(
        subset=['roi_width', 'roi_height']
    )
    if valid_roi_df.empty:
        raise ValueError("No valid ROIs found. Cannot determine output dimensions.")

    first_valid_roi = valid_roi_df.iloc[0]
    out_w = int(first_valid_roi['roi_width'])
    out_h = int(first_valid_roi['roi_height'])

    # 3. Setup for Atomic Write Operation
    p = Path(output_video_path)
    temp_output_path = p.with_suffix(f".tmp{p.suffix}")

    try:
        # Use a modern and widely compatible codec (H.264).
        # 'avc1' is for H.264, often better than the older 'mp4v' (MPEG-4 Part 2).
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # The 'with' statement handles opening and (crucially) releasing the writer
        with video_writer_manager(
            str(temp_output_path), fourcc, video_manager.fps, (out_w, out_h)
        ) as writer:
            print(f"🎬 Processing video for '{output_video_path}' with frame size ({out_w}, {out_h})...")
            black_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            roi_df_indexed = roi_df.set_index('frame_id')
            roi_coord_cols = ['roi_x', 'roi_y', 'roi_width', 'roi_height']

            total_frames = len(video_manager)
            # Log progress every 5% of frames, or at least every 10 frames
            log_interval = max(10, int(total_frames * 0.05))

            # 4. Main writing loop
            for frame_idx in range(total_frames):
                # --- Progress Feedback Logic ---
                if frame_idx % log_interval == 0:
                    percent_complete = (frame_idx / total_frames) * 100
                    print(f"   ⏳ Progress: Frame {frame_idx}/{total_frames} ({percent_complete:.1f}%)")
                # -------------------------------

                try:
                    roi_row = roi_df_indexed.loc[frame_idx]
                    if roi_row['status'] == 'Black Frame' or roi_row[roi_coord_cols].isnull().any():
                        writer.write(black_frame)
                        continue

                    frame = video_manager[frame_idx]
                    x, y, _, _ = roi_row[roi_coord_cols].astype(int)
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(frame.shape[1], x + out_w), min(frame.shape[0], y + out_h)

                    cropped_frame = frame[y1:y2, x1:x2]

                    if cropped_frame.size > 0:
                        writer.write(cv2.resize(cropped_frame, (out_w, out_h)))
                    else:
                        writer.write(black_frame)

                except KeyError:
                    # Frame index not found in ROI data, write a black frame
                    writer.write(black_frame)

        # 5. Atomic Move: This line is only reached if the 'with' block completes.
        # The temp file is guaranteed to be closed and finalized at this point.
        shutil.move(str(temp_output_path), output_video_path)
        print(f"✅ File successfully created at '{output_video_path}'")

        # Step 5: ✨ NEW - VERIFICATION STEP ✨
        if not success_video_file_quality_check(output_video_path):
            # If verification fails, clean up and raise an error.
            print(f"🗑️ Deleting corrupted file: '{output_video_path}'")
            Path(output_video_path).unlink()
            raise IOError(f"💥 Failed to create a valid video file at '{output_video_path}'. The output was corrupted.")

        print(f"🎉 Successfully created and verified video: '{output_video_path}'")

    except Exception as e:
        print(f"❌ An error occurred during video creation: {e}. Cleaning up...")
        # If anything fails, delete the temporary file if it exists
        if temp_output_path.exists():
            temp_output_path.unlink()
        # Re-raise the exception so the calling code knows something went wrong
        raise


def create_standardized_roi_videos(
    roi_file_path: str,
    video_path: str,
    output_video_base_path: str,
    *,
    force_processing: bool = False,
):
    """
    Main orchestrator to generate a windowed video for each object.

    Processes frames on-the-fly to minimize memory usage.
    """
    rois_df_dict: dict = ROITrackedFileHandler(roi_file_path).load_all_data()

    output_path = Path(output_video_base_path)
    output_dir = output_path.parent
    base_name = output_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reuse manager for each object without reloading the video data.
    video_manager = VideoMP4Manager(video_path)
    video_manager.color_format = ColorFormat.BGR

    for obj_name, roi_df in rois_df_dict.items():
        output_video_path = output_dir / f"{base_name}_{obj_name}.mp4"

        if not should_process_task(
            output_paths=output_video_path,
            input_paths=[roi_file_path, video_path],
            force=force_processing,
        ):
            print(f"Video {output_video_path} has already been processed. Skipping...")
            continue

        # Corrected function call to match the robust version defined above
        create_windowed_video(
            video_manager=video_manager,
            roi_df=roi_df,
            output_video_path=str(output_video_path),
        )