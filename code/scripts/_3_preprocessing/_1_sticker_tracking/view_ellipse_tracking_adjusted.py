# Standard library imports
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

# Third-party imports
import cv2
import numpy as np
import pandas as pd

# Local application/library specific imports
from preprocessing.common import VideoMP4Manager
from preprocessing.stickers_analysis import (
    FittedEllipsesFileHandler,
    FittedEllipsesManager,
    
    EllipseFitViewGUI
)


def fit_ellipse_on_frame(binary_frame: np.ndarray) -> Optional[Dict]:
    """
    Fits an ellipse to the largest contour in a binary frame.

    Args:
        binary_frame (np.ndarray): The input binary frame (single channel, 0 or 255).

    Returns:
        Optional[Dict]: A dictionary with the ellipse data or None if no suitable contour is found.
    """
    # --- Contour Detection and Validation ---
    if binary_frame.ndim != 2 or binary_frame.dtype != np.uint8:
        print("Error: Input frame must be a single-channel 8-bit image (CV_8UC1).")
        return None

    # --- Contour Detection and Validation ---
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    # An ellipse needs at least 5 points to be fitted
    if len(largest_contour) < 5:
        return None

    # --- Ellipse Fitting ---
    ellipse = cv2.fitEllipse(largest_contour)
    
    # --- Scoring and Return Structure ---
    contour_area = cv2.contourArea(largest_contour)
    axes = ellipse[1]

    # A valid ellipse must have non-zero axes
    if axes[0] <= 0 or axes[1] <= 0:
        return None
        
    ellipse_area = math.pi * (axes[0] / 2.0) * (axes[1] / 2.0)
    
    # Score is the ratio of the smaller area to the larger area
    score = min(contour_area, ellipse_area) / max(contour_area, ellipse_area)

    return {
        'center_x': ellipse[0][0],
        'center_y': ellipse[0][1],
        'axes_major': axes[1],
        'axes_minor': axes[0],
        'angle': ellipse[2],
        'score': score
    }


def fit_ellipses_on_grayscale_frames(frames: List[np.ndarray], threshold: int) -> pd.DataFrame:
    """
    Processes a list of grayscale frames by first applying a binary threshold
    and then finding the best-fit ellipse on the result.

    Args:
        frames (List[np.ndarray]): A list of grayscale video frames.
        threshold (int): The pixel intensity value (0-255) to use for binary thresholding.

    Returns:
        pd.DataFrame: A DataFrame with the best-fit ellipse data for each frame.
    """
    all_frame_results = []

    for frame_number, frame in enumerate(frames):
        if frame is None or frame.size == 0:
            print(f"Frame {frame_number}: Frame is empty, adding NaN row.")
            best_result_for_frame = None
        else:
            # Convert the grayscale frame to a binary image using the provided threshold.
            _, binary_frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
            
            # Fit ellipse directly on the binary frame
            best_result_for_frame = fit_ellipse_on_frame(binary_frame)

        if best_result_for_frame:
            best_result_for_frame['frame_number'] = frame_number
            all_frame_results.append(best_result_for_frame)
        else:
            # If no ellipse was found, append a row with NaN values
            nan_result = {
                'frame_number': frame_number, 'center_x': np.nan, 'center_y': np.nan,
                'axes_major': np.nan, 'axes_minor': np.nan, 'angle': np.nan, 'score': np.nan,
            }
            all_frame_results.append(nan_result)
            
    return pd.DataFrame(all_frame_results)


def view_ellipse_tracking_adjusted(
    video_path: Path,
    tracking_ellipses_path: Path
):
    """
    Orchestrates the video processing workflow for grayscale videos.
    It loads grayscale frames, applies a binary threshold from metadata,
    fits ellipses, and saves the results using the dedicated manager and file handler.
    """
    print(f"--- Starting Video Processing ---")
    
    # --- MODIFICATION START ---
    # 1. Load metadata and perform pre-flight checks
    # ---
    ellipse_manager: FittedEllipsesManager = FittedEllipsesFileHandler.load(tracking_ellipses_path)
    ellipse_data_dict = ellipse_manager.get_all_results()
    frames_bgr = VideoMP4Manager(video_path)

    gui = EllipseFitViewGUI(
        ellipse_df=ellipse_data_dict,
        frames=frames_bgr,
        title="Multi-Object Ellipse Visualization (Color Video)",
        windowState='maximized'
    )
    gui.start()
