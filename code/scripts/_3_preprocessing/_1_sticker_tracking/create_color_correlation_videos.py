# Standard library imports
import ast
import itertools
import json
import math
import os
import sys
import time
import tkinter as tk  # Used to get screen dimensions
from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import h5py


# Third-party imports
import cv2
import numpy as np
import pandas as pd

# Local application imports
from preprocessing.common import VideoMP4Manager, ColorFormat
from preprocessing.stickers_analysis import (
    ColorSpaceFileHandler, 
    ColorSpaceManager,
    ColorSpace,
    ColorSpaceStatus,

    ColorFamilyModel,
    ColorCorrelationVisualizer
)


def generate_correlation_maps(
        frames_bgr: List[Any], 
        metadata: ColorSpace, 
        *,
        show: bool = False,
        conversion_mode: str = 'xyz'
) -> List[Any]:
    """
    Generates color correlation maps for a sequence of frames.

    This function contains the core computational logic without any GUI.

    Args:
        frames (List[Any]): A list of video frames (e.g., numpy arrays).
        metadata (Dict[str, Any]): A dictionary containing metadata, including
                                   'colorspaces' for model initialization.

    Returns:
        List[Any]: A list of correlation maps, one for each frame.
    """
    family_colors = metadata.extract_rgb_triplets(output_format='array')
    model = ColorFamilyModel(
        family_colors, color_space='rgb',
        conversion_mode=conversion_mode)

    correlation_maps = [
        model.calculate_mahalanobis_map(frame_bgr, color_space='bgr')
        for frame_bgr in frames_bgr
    ]

    if show:
        try:
            import matplotlib.pyplot as plt
            # Use the new, more explicit class name
            visualizer = ColorCorrelationVisualizer(model)

            for frame_bgr in frames_bgr:
                visualizer.update(frame_bgr)

        finally:
            print("Processing finished. Close the plot window to exit.")
            plt.ioff()
            plt.show()

    return correlation_maps


def save_correlation_results_to_mp4(
    corr_maps: List[np.ndarray],
    output_path: str,
    fps: int = 30
) -> None:
    """
    Saves a list of 2D correlation maps as a grayscale MP4 video.

    This function takes a list of floating-point correlation maps, normalizes them
    globally to an 8-bit grayscale format, and writes them to a video file.

    Args:
        corr_maps (List[np.ndarray]): A list of 2D NumPy arrays, where each array
                                      is a correlation map with float values.
        output_path (str): The full path for the output MP4 video file.
        fps (int, optional): The frames per second for the output video.
                             Defaults to 30.
    """
    # 1. Validate inputs
    if not corr_maps:
        print("⚠️ Warning: The list of correlation maps is empty. No video will be created.")
        return

    # Create the directory if it doesn't exist to prevent errors
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 2. Get frame dimensions and setup video writer
    height, width = corr_maps[0].shape
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size, isColor=False)

    if not video_writer.isOpened():
        print(f"❌ Error: Could not open video writer for path: {output_path}")
        return

    # 3. Find global min and max for consistent normalization across all frames
    global_min = np.min([np.min(m) for m in corr_maps])
    global_max = np.max([np.max(m) for m in corr_maps])
    
    # Handle the edge case where all values are the same (prevents division by zero)
    if global_max == global_min:
        print("⚠️ Warning: All correlation values are identical. The video will be uniformly black.")
        scaled_range = 0.0  # This will result in black frames
    else:
        scaled_range = global_max - global_min

    print(f"📹 Saving grayscale video ({len(corr_maps)} frames) to '{output_path}'...")
    print(f"   - Normalizing correlation values from range [{global_min:.2f}, {global_max:.2f}] to [0, 255].")

    # 4. Process frames and write to video
    try:
        for corr_map in corr_maps:
            # a. Handle the normalization based on the calculated range
            if scaled_range == 0.0:
                # If all values were the same, create a black frame
                normalized_float = np.zeros_like(corr_map, dtype=float)
            else:
                # Apply min-max normalization to scale values to [0.0, 1.0]
                normalized_float = (corr_map - global_min) / scaled_range

            # b. Scale to the 0-255 range and convert to an 8-bit integer.
            frame_uint8 = (normalized_float * 255).astype(np.uint8)

            # c. Write the single-channel (grayscale) frame to the video.
            video_writer.write(frame_uint8)
    finally:
        # 5. Finalize and release the video writer
        video_writer.release()
        print(f"✅ Video successfully saved to '{output_path}'.")

def invert_correlation_maps(corr_maps: List[np.ndarray]) -> List[np.ndarray]:
    """
    Inverts the values of correlation maps for grayscale video visualization.

    This function processes a list of 2D numpy arrays (correlation maps)
    and inverts their values based on the global maximum value across all maps.
    The transformation ensures that low values (e.g., low distance) in the
    original maps will correspond to high values (bright pixels) in the
    output maps, maintaining a consistent brightness scale across all frames.

    The inversion formula used for each pixel in each map is:
    inverted_value = global_max - original_value

    Args:
        corr_maps (List[np.ndarray]): A list of 2D NumPy arrays, where each
                                      array represents a correlation map frame.

    Returns:
        List[np.ndarray]: A new list containing the inverted correlation maps.

    Raises:
        ValueError: If the input list is empty.
    """
    if not corr_maps:
        raise ValueError("Input list 'corr_maps' cannot be empty.")

    # Find the global maximum value across all frames for a consistent scale
    # A generator expression is used for memory efficiency
    global_max = max(np.max(frame) for frame in corr_maps)

    # Invert each frame using the global max value.
    # A list comprehension creates a new list with the inverted frames.
    inverted_maps = [global_max - frame for frame in corr_maps]

    return inverted_maps



def create_color_correlation_videos(
    video_path: Path,
    md_path: Path,
    output_path: Path,
    *,
    force_processing: bool = False
):
    """
    Orchestrates the video processing workflow, processing only objects
    marked as 'TO_BE_PROCESSED' unless 'force_processing' is enabled.
    """
    print("--- Starting Video Processing ---")

    # 1. Load the metadata manager
    colorspace_manager: ColorSpaceManager = ColorSpaceFileHandler.load(md_path)

    # 2. Early exit: If not forcing and no objects need processing, stop.
    if not force_processing and colorspace_manager.are_no_objects_with_status(ColorSpaceStatus.TO_BE_PROCESSED):
        print("✅ No objects are marked 'to_be_processed'. Nothing to do.")
        return output_path

    # 3. Process each object sequentially based on its status
    metadata_file_modified = False
    object_names = colorspace_manager.colorspace_names
    for name in object_names:
        current_colorspace = colorspace_manager.get_colorspace(name)

        # 4. Decide whether to process this specific object
        should_process = force_processing or (current_colorspace.status == ColorSpaceStatus.TO_BE_PROCESSED.value)

        if not should_process:
            print(f"Skipping '{name}' (status: '{current_colorspace.status}').")
            continue
            
        print(f"Processing '{name}'...")
        output_object_path = output_path.parent / (output_path.stem + f"_{name}.mp4")

        if not force_processing and os.path.exists(output_object_path):
            print(f"Output file already exists, skipping: {output_object_path}")
            continue

        input_video_path = video_path.parent / (video_path.stem + f"_{name}.mp4")
        print(f"Loading video '{input_video_path}'...")
        
        video_manager = VideoMP4Manager(input_video_path)
        video_manager.color_format = ColorFormat.BGR
        
        conversion_mode = 'circular'
        corr_maps = generate_correlation_maps(
            frames_bgr=video_manager,
            metadata=current_colorspace,
            show=False,
            conversion_mode=conversion_mode
        )
        
        corr_maps = invert_correlation_maps(corr_maps)
        
        save_correlation_results_to_mp4(corr_maps, output_object_path)
        print(f"✅ Finished processing for '{name}'.")
        colorspace_manager.update_status(name, ColorSpaceStatus.TO_BE_REVIEWED.value)
        metadata_file_modified = True

    if metadata_file_modified:
        ColorSpaceFileHandler.write(md_path, colorspace_manager)
    
    print("--- Standalone Processing Complete ---")
    return output_path