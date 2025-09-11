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

    ColorFamilyModel,
    ColorCorrelationVisualizer
)



def extract_pixel_rgb_values(
    colorspaces_data: list,
    output_format: str = 'array'
) -> Union[np.ndarray, list]:
    """
    Extracts all RGB color triplets from the nested 'freehand_pixels'
    dictionary within a list of colorspace data.

    This function iterates through a list of frame data dictionaries. For each
    frame, it safely navigates to ['colorspace']['freehand_pixels']['rgb']
    to find the list of RGB triplets. All found triplets are aggregated and
    returned in the specified format.

    Args:
        colorspaces_data (list): The list of dictionaries, corresponding to
                                 metadata.colorspaces.
        output_format (str, optional): The desired output format. Can be
                                'array' for a NumPy array (default) or
                                'list' for a standard Python list.

    Returns:
        Union[np.ndarray, list]: A collection of all extracted RGB triplets.
                                 The type is a 2D NumPy array by default,
                                 or a list of lists if specified.

    Raises:
        ValueError: If an unsupported output_format is provided.
    """
    all_rgb_triplets = []

    # Core extraction logic (remains the same)
    for frame_data in colorspaces_data:
        freehand_pixels_data = frame_data.get('colorspace', {}).get('freehand_pixels', {})

        if freehand_pixels_data:
            rgb_list = freehand_pixels_data.get('rgb')
            if rgb_list:
                all_rgb_triplets.extend(rgb_list)

    # --- New: Format the output based on the parameter ---

    # Handle the case where no colors were found
    if not all_rgb_triplets:
        if output_format == 'array':
            # Return an empty NumPy array with the correct shape (0 rows, 3 columns)
            return np.empty((0, 3), dtype=np.uint8)
        else:
            return []

    # Convert to the specified format
    if output_format == 'array':
        # Convert the list of lists to a 2D NumPy array.
        # np.uint8 is the standard data type for 8-bit color values (0-255).
        return np.array(all_rgb_triplets, dtype=np.uint8)
    elif output_format == 'list':
        return all_rgb_triplets
    else:
        raise ValueError(f"Invalid output_format: '{output_format}'. Please choose 'array' or 'list'.")


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
    family_colors = extract_pixel_rgb_values(metadata.colorspaces, output_format='array')
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


# --- Core Component 1: Loading ---
def _load_metadata(md_path: str) -> Dict[str, Any]:
    """
    Responsibility: Loads and validates metadata and tracking data from files.
    """
    if not os.path.exists(md_path):
        raise FileNotFoundError(f"Metadata file not found at '{md_path}'.")

    try:
        with open(md_path, "r") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        raise ValueError(f"Could not read or parse metadata file '{md_path}'.") from e

    print(f"✅ Successfully loaded metadata.")
    return metadata


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
    Orchestrates the video processing workflow.
    """
    print(f"--- Starting Video Processing ---")

    # 2. Identify objects to process
    colorspace_manager = ColorSpaceFileHandler.load(md_path)
    object_names = colorspace_manager.colorspace_names

    # 3. Process each object sequentially
    for name in object_names:
        print(f"Processing '{name}'...")

        input_video_path = video_path.parent / (video_path.stem + f"_{name}.mp4")
        output_object_path = output_path.parent / (output_path.stem + f"_{name}.mp4")

        if not force_processing and os.path.exists(output_object_path):
            print(f"output file already exists (file = {output_object_path}): Skipping...")
            continue

        print(f"Loading video '{input_video_path}'...")
        video_manager = VideoMP4Manager(input_video_path)
        video_manager.color_format = ColorFormat.BGR

        current_metadata = colorspace_manager.get_colorspace(name)
        
        conversion_mode = 'circular' # 'xyz', 'circular'
        corr_maps = generate_correlation_maps(
            frames_bgr=video_manager,
            metadata=current_metadata,
            show=False,
            conversion_mode=conversion_mode)
        
        corr_maps = invert_correlation_maps(corr_maps)
        
        save_correlation_results_to_mp4(corr_maps, output_object_path)
        print(f"✅ Finished processing for '{name}'.")

    print(f"--- Standalone Processing Complete ---")