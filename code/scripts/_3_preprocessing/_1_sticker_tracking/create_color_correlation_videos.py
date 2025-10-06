# --------------------------------------------------------------------------- #
# Standard Library Imports
# --------------------------------------------------------------------------- #
from pathlib import Path
from typing import List, Optional, Union

# --------------------------------------------------------------------------- #
# Third-Party Imports
# --------------------------------------------------------------------------- #
import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Local Application Imports
# --------------------------------------------------------------------------- #
from preprocessing.common import ColorFormat, VideoMP4Manager
from preprocessing.stickers_analysis import (
    ColorCorrelationVisualizer,
    ColorFamilyModel,
    ColorSpace,
    ColorSpaceFileHandler,
    ColorSpaceManager,
    ColorSpaceStatus,
)
from utils.should_process_task import should_process_task


# --------------------------------------------------------------------------- #
# Core Processing Functions
# --------------------------------------------------------------------------- #

def generate_correlation_maps(
    frames_bgr: List[np.ndarray],
    metadata: ColorSpace,
    *,
    show: bool = False,
    conversion_mode: str = 'xyz'
) -> List[np.ndarray]:
    """
    Generates color correlation maps for a sequence of BGR frames.

    This function contains the core computational logic for calculating the
    Mahalanobis distance of each pixel from a defined color family.

    Args:
        frames_bgr (List[np.ndarray]): A list of video frames as BGR NumPy arrays.
        metadata (ColorSpace): An object containing the color family definitions.
        show (bool, optional): If True, displays the frames and their
                               correlation maps using matplotlib. Defaults to False.
        conversion_mode (str, optional): The color space conversion mode to
                                         use in the model. Defaults to 'xyz'.

    Returns:
        List[np.ndarray]: A list of correlation maps (float arrays), one for
                          each input frame.
    """
    family_colors = metadata.extract_rgb_triplets(output_format='array')
    model = ColorFamilyModel(
        family_colors, color_space='rgb', conversion_mode=conversion_mode
    )

    correlation_maps = [
        model.calculate_mahalanobis_map(frame_bgr, color_space='bgr')
        for frame_bgr in frames_bgr
    ]

    if show:
        try:
            import matplotlib.pyplot as plt
            visualizer = ColorCorrelationVisualizer(model)
            for frame_bgr in frames_bgr:
                visualizer.update(frame_bgr)
        finally:
            print("Processing finished. Close the plot window to exit.")
            plt.ioff()
            plt.show()

    return correlation_maps


def invert_correlation_maps(corr_maps: List[np.ndarray]) -> List[np.ndarray]:
    """
    Inverts correlation map values for intuitive grayscale visualization.

    This transformation ensures that low distance values (strong correlation)
    become high intensity values (bright pixels), using a consistent scale
    across all frames. The formula used is:
    `inverted_value = global_max - original_value`

    Args:
        corr_maps (List[np.ndarray]): A list of 2D float arrays, where each
                                      array is a correlation map.

    Returns:
        List[np.ndarray]: A new list containing the inverted correlation maps.

    Raises:
        ValueError: If the input list `corr_maps` is empty.
    """
    if not corr_maps:
        raise ValueError("Input list 'corr_maps' cannot be empty.")

    # Find the global maximum value across all frames for a consistent scale
    # A generator expression is used for memory efficiency
    global_max = max(np.max(frame) for frame in corr_maps)

    # Invert each frame using the global max. A list comprehension is used
    # for a concise creation of the new list.
    inverted_maps = [global_max - frame for frame in corr_maps]

    return inverted_maps


# --------------------------------------------------------------------------- #
# I/O and Utility Functions
# --------------------------------------------------------------------------- #

def save_correlation_results_to_mp4(
    corr_maps: List[np.ndarray],
    output_path: Union[str, Path],
    fps: int = 30
) -> None:
    """
    Saves a list of correlation maps as a grayscale MP4 video.

    The function normalizes the maps globally to an 8-bit range [0, 255]
    to ensure consistent brightness across the entire video.

    Args:
        corr_maps (List[np.ndarray]): A list of 2D NumPy arrays (float values).
        output_path (Union[str, Path]): The full path for the output MP4 file.
        fps (int, optional): Frames per second for the output video. Defaults to 30.
    """
    if not corr_maps:
        print("⚠️ Warning: The list of correlation maps is empty. No video created.")
        return

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get frame dimensions and setup video writer
    height, width = corr_maps[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (width, height), isColor=False
    )

    if not video_writer.isOpened():
        print(f"❌ Error: Could not open video writer for path: {output_path}")
        return

    # Find global min and max for consistent normalization across all frames
    global_min = np.min([np.min(m) for m in corr_maps])
    global_max = np.max([np.max(m) for m in corr_maps])
    scaled_range = global_max - global_min

    print(f"📹 Saving grayscale video ({len(corr_maps)} frames) to '{output_path}'...")
    print(f"   - Normalizing values from range [{global_min:.2f}, {global_max:.2f}] to [0, 255].")

    try:
        for corr_map in corr_maps:
            # Handle the edge case where all values are identical (prevents division by zero)
            if scaled_range == 0.0:
                normalized_float = np.zeros_like(corr_map, dtype=float)
            else:
                # Apply min-max normalization to scale values to [0.0, 1.0]
                normalized_float = (corr_map - global_min) / scaled_range

            # Scale to 0-255 and convert to 8-bit integer for video encoding
            frame_uint8 = (normalized_float * 255).astype(np.uint8)
            video_writer.write(frame_uint8)
    finally:
        video_writer.release()
        print(f"✅ Video successfully saved to '{output_path}'.")


# --------------------------------------------------------------------------- #
# Main Workflow Orchestrator
# --------------------------------------------------------------------------- #

def create_color_correlation_videos(
    video_standard_format_path: Path,
    md_path: Path,
    output_standard_format_path: Path,
    *,
    force_processing: bool = False
) -> Optional[Path]:
    """
    Orchestrates the workflow to generate color correlation videos.

    This function iterates through objects defined in a metadata file, processes
    videos for objects marked as 'TO_BE_PROCESSED', generates inverted
    correlation maps, saves them as MP4 files, and updates the metadata status.

    Args:
        video_standard_format_path (Path): Base path for input videos.
        md_path (Path): Path to the metadata JSON file.
        output_standard_format_path (Path): Base path for output videos.
        force_processing (bool, optional): If True, processes all objects
                                           regardless of status. Defaults to False.
    Returns:
        Optional[Path]: The path of the last generated video, or None if no
                        videos were processed.
    """
    colorspace_manager: ColorSpaceManager = ColorSpaceFileHandler.load(md_path)
    metadata_file_modified = False
    last_output_path = None

    print("--- Starting Color Correlation Video Processing ---")

    for name in colorspace_manager.colorspace_names:
        current_colorspace = colorspace_manager.get_colorspace(name)
        
        # Define I/O paths for the current object
        input_video_path = video_standard_format_path.with_name(
            f"{video_standard_format_path.stem}_{name}.mp4"
        )
        output_video_path = output_standard_format_path.with_name(
            f"{output_standard_format_path.stem}_{name}.mp4"
        )

        # Decide whether to process this object
        is_to_be_processed = (
            current_colorspace.status == ColorSpaceStatus.TO_BE_PROCESSED.value
        )
        needs_processing = should_process_task(
            output_paths=output_video_path,
            input_paths=input_video_path,
            force=force_processing
        )

        if not (is_to_be_processed or needs_processing):
            print(f"Skipping '{name}' (Status: '{current_colorspace.status}', files up-to-date).")
            continue

        print(f"\nProcessing '{name}'...")
        print(f"-> Loading video: '{input_video_path}'")
        video_manager = VideoMP4Manager(input_video_path)
        video_manager.color_format = ColorFormat.BGR

        # Generate, invert, and save correlation maps
        corr_maps = generate_correlation_maps(
            frames_bgr=video_manager,
            metadata=current_colorspace,
            show=False,
            conversion_mode='circular'
        )

        inverted_maps = invert_correlation_maps(corr_maps)
        save_correlation_results_to_mp4(inverted_maps, output_video_path)

        # Update metadata status
        colorspace_manager.update_status(name, ColorSpaceStatus.TO_BE_REVIEWED.value)
        metadata_file_modified = True
        last_output_path = output_video_path
        print(f"✅ Finished processing for '{name}'.")

    if metadata_file_modified:
        ColorSpaceFileHandler.write(md_path, colorspace_manager)
        print("\n--- Metadata file updated. Processing complete. ---")
    else:
        print("\n✅ No objects required processing. Nothing to do.")

    return last_output_path