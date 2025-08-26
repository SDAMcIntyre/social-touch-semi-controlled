# Standard library imports
import json
import math
import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


# Third-party imports
import cv2
import numpy as np
import pandas as pd

# Local application imports
from .utils.roi_to_position_funcs import (
    load_video_frames_bgr,
    review_frames
)


def get_obj_name(metadata: dict) -> list[str]:
    keys_list = list(metadata.keys())
    return keys_list

def fit_ellipse_on_frame(gray_frame: np.ndarray, threshold_value: int, spot_type: str = 'bright') -> Optional[Dict]:
    """
    Applies a binary threshold to a frame and fits an ellipse to the largest contour,
    using an area-based similarity score.

    This function combines the flexibility of detecting bright or dark spots with the
    scoring method and return structure of the original implementation.

    Args:
        gray_frame (np.ndarray): The input grayscale frame.
        threshold_value (int): The threshold value (0-255).
        spot_type (str): Type of spot to detect. 'bright' for light spots on a
                         dark background, or 'dark' for dark spots on a light background.

    Returns:
        Optional[Dict]: A dictionary with the ellipse data or None if no suitable contour is found.
                        The dictionary format is:
                        {'center': (x, y), 'axes': (height, width), 'angle': angle, 'score': score}
    """
    # --- Flexible Thresholding (from frame2) ---
    if spot_type == 'bright':
        thresh_type = cv2.THRESH_BINARY
    elif spot_type == 'dark':
        thresh_type = cv2.THRESH_BINARY_INV
    else:
        raise ValueError("spot_type must be 'bright' or 'dark'")

    _, binary_image = cv2.threshold(gray_frame, threshold_value, 255, thresh_type)

    # --- Contour Detection and Validation (from frame2) ---
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)

    # An ellipse needs at least 5 points to be fitted
    if len(largest_contour) < 5:
        return None

    # --- Ellipse Fitting ---
    ellipse = cv2.fitEllipse(largest_contour)
    
    # --- Scoring and Return Structure (from frame1) ---
    contour_area = cv2.contourArea(largest_contour)
    axes = ellipse[1]  # The (width, height) of the bounding box of the ellipse

    # A valid ellipse must have non-zero axes
    if axes[0] <= 0 or axes[1] <= 0:
        return None
        
    # Area of ellipse = pi * (width/2) * (height/2)
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


def fit_ellipses_with_adaptive_search(
    frames: List[np.ndarray],
    initial_threshold_range: Tuple[int, int] = (100, 250),
    step: int = 5,
    satisfaction_score: float = 0.95,
    search_radius: int = 25,
    n_avg_frames: int = 5,
    spot_type: str = 'bright'
) -> pd.DataFrame:
    """
    Processes a list of frames to find the best-fit ellipse on the
    brightest or darkest spot, using an adaptive threshold search.

    This function returns a DataFrame where each row corresponds to a frame.
    If no ellipse could be fitted for a frame, the row will contain NaN values.

    Args:
        frames (List[np.ndarray]): A list of video frames (as NumPy arrays).
        initial_threshold_range (Tuple[int, int]): The (start, end) for the first frame's search.
        step (int): The increment for the threshold search.
        satisfaction_score (float): If a score above this is found, the search for
                                    the frame stops immediately.
        search_radius (int): How far to search around the last optimal threshold.
        n_avg_frames (int): The number of recent frames to average for the next search center.
        spot_type (str): The type of spot to search for, either 'bright' or 'dark'. Defaults to 'bright'.

    Returns:
        pd.DataFrame: A DataFrame with the best-fit ellipse data for each frame.
                      Columns include 'frame_number', 'center', 'axes', 'angle',
                      'score', and 'optimal_threshold'.
    """
    # --- Input Validation ---
    if spot_type not in ['bright', 'dark']:
        raise ValueError(f"spot_type must be 'bright' or 'dark', but got '{spot_type}'")

    all_frame_results = []
    recent_optimal_thresholds = deque(maxlen=n_avg_frames)

    for frame_number, frame in enumerate(frames):
        if frame is None or frame.size == 0:
            print(f"Frame {frame_number}: Frame is empty, adding NaN row.")
            best_result_for_frame = None
        else:
            # --- Pre-processing ---
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame

            # Normalize frame to full 0-255 range to make thresholds consistent
            min_val, max_val = np.min(gray_frame), np.max(gray_frame)
            if max_val == min_val:
                gray_frame = np.zeros(gray_frame.shape, dtype=np.uint8)
            else:
                gray_frame = gray_frame.astype(np.float32)
                gray_frame = 255 * (gray_frame - min_val) / (max_val - min_val)
                gray_frame = gray_frame.astype(np.uint8)

            best_result_for_frame = None
            highest_score = -1.0

            # --- Define the search space for the current frame ---
            if not recent_optimal_thresholds:
                search_center = (initial_threshold_range[0] + initial_threshold_range[1]) // 2
                current_radius = (initial_threshold_range[1] - initial_threshold_range[0]) // 2
            else:
                search_center = int(np.mean(list(recent_optimal_thresholds)))
                current_radius = search_radius
            
            # --- Build the prioritized list of thresholds to test (center-out) ---
            thresholds_to_test = [search_center]
            for i in range(1, (current_radius // step) + 1):
                thresholds_to_test.append(search_center - i * step)
                thresholds_to_test.append(search_center + i * step)
            
            thresholds_to_test = [t for t in thresholds_to_test if 0 < t < 255]

            # --- Adaptive Search with Early Exit ---
            for th_val in thresholds_to_test:
                current_result = fit_ellipse_on_frame(gray_frame, th_val, spot_type=spot_type)

                if current_result:
                    if current_result['score'] > highest_score:
                        highest_score = current_result['score']
                        best_result_for_frame = current_result
                        best_result_for_frame['optimal_threshold'] = th_val
                    
                    if current_result['score'] >= satisfaction_score:
                        break 
        
        if best_result_for_frame:
            optimal_th = best_result_for_frame['optimal_threshold']
            best_result_for_frame['frame_number'] = frame_number
            all_frame_results.append(best_result_for_frame)
            recent_optimal_thresholds.append(optimal_th)
            print(f"Frame {frame_number}: Found fit with score {highest_score:.4f} at threshold {optimal_th}.")
        else:
            # If no ellipse was found, append a row with NaN values
            print(f"Frame {frame_number}: No suitable ellipse found. Adding NaN row.")
            nan_result = {
                'frame_number': frame_number,
                'center_x': np.nan,
                'center_y': np.nan,
                'axes_major': np.nan,
                'axes_minor': np.nan,
                'angle': np.nan,
                'score': np.nan,
                'optimal_threshold': np.nan
            }
            all_frame_results.append(nan_result)
            recent_optimal_thresholds.clear()
            
    # --- Convert the list of dictionaries to a pandas DataFrame ---
    return pd.DataFrame(all_frame_results)


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


# --- Image Preparation Functions ---
def prepare_full_frame_display(
    full_frame: np.ndarray,
    frame_id: int,
    roi_location: Tuple[int, int, int, int],
    ellipse_data: Dict[str, Any]
) -> np.ndarray:
    """
    Prepares the full frame visualization by drawing overlays.

    Args:
        full_frame (np.ndarray): The full-size original video frame.
        frame_id (int): The current frame index.
        roi_location (Tuple[int, int, int, int]): The (x, y, w, h) of the ROI.
        ellipse_data (Dict[str, Any]): Dictionary with fitted ellipse parameters.

    Returns:
        np.ndarray: The annotated full frame image in BGR format.
    """
    # Create a BGR copy to draw on
    if len(full_frame.shape) == 2:
        display_frame = cv2.cvtColor(full_frame, cv2.COLOR_GRAY2BGR)
    else:
        display_frame = full_frame.copy()

    # Draw the ROI bounding box and frame ID
    roi_x, roi_y, roi_w, roi_h = roi_location
    cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
    cv2.putText(display_frame, f"Frame {frame_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Safely check and draw the ellipse
    center_val = ellipse_data.get('center', (math.nan, math.nan))
    axes_val = ellipse_data.get('axes', (math.nan, math.nan))
    angle_val = ellipse_data.get('angle', math.nan)

    if not any(math.isnan(c) for c in center_val) and \
       not any(math.isnan(a) for a in axes_val) and \
       not math.isnan(angle_val):
        
        axes = (int(axes_val[0] / 2), int(axes_val[1] / 2))
        angle = int(angle_val)
        center = (int(center_val[0]), int(center_val[1]))
        cv2.ellipse(display_frame, center, axes, angle, 0, 360, (0, 0, 255), 2)

    return display_frame


def _monitor_single_object(
    object_name: str,
    frames_bgr: List,
    ellipses: Dict[str, Any]
):
    """
    Pre-processes all frames and then monitors a single object with smooth playback.

    This function operates in two phases:
    1. Preparation: It iterates through all data to generate display frames,
       which are stored in memory. This introduces an initial delay.
    2. Display: It plays back the pre-generated frames at a consistent FPS,
       decoupled from the processing time.

    Args:
        object_name (str): Name of the object being tracked.
        frames_bgr (List): List of video frames (BGR format).
        tracked_rois (pd.Series): Series containing ROI coordinates [x, y, w, h].
        ellipses (Dict[str, Any]): Dictionary of ellipse data.
        corr_maps [List[Any]]: List of correlation maps.
    """
    # --- 1. Preparation Phase ---
    print("Preparing all frames for display. This may take a moment...")
    full_prepared_frames = [] 
    
    zipped_data = zip(frames_bgr, ellipses)

    for frame_id, (frame_bgr, ellipse) in enumerate(zipped_data):
        try:
            # Create a copy to avoid drawing on the original frame data
            frame_copy = frame_bgr.copy()
            
            # Prepare the main display frame
            full_frame_display = prepare_full_frame_display(frame_copy, frame_id, ellipse)
            full_prepared_frames.append(full_frame_display)

        except (TypeError, ValueError) as e:
            print(f"  -> Invalid data for frame {frame_id}. Skipping preparation. Reason: {e}")
    
    print(f"Preparation complete. {len(full_prepared_frames)} frames are ready for display.")

    if not full_prepared_frames:
        print("No valid frames were prepared. Exiting monitor.")
        return

    # --- 2. Display Phase ---
    win1_name = f"Full Frame Monitor: {object_name}"
    review_frames(full_prepared_frames, title=win1_name)


def fit_ellipses_on_correlation_videos(
    video_path: str,
    md_path: str,
    output_path: str,
    *,
    monitor_ui: bool = False
):
    """
    Orchestrates the video processing workflow.
    """
    print(f"--- Starting Video Processing ---")
    
    # 1. Load all data
    metadata = _load_metadata(md_path)
    
    # 2. Identify objects to process
    object_names = get_obj_name(metadata)

    ellipses_results = {}
    # 3. Process each object sequentially
    for name in object_names:            
        print(f"Processing '{name}'...")

        input_video_path = video_path.parent / (video_path.stem + f"_{name}.mp4")
        
        print(f"Loading video '{input_video_path}'...")
        frames_bgr = load_video_frames_bgr(input_video_path)
        
        ellipse_results = fit_ellipses_with_adaptive_search(
            frames=frames_bgr, 
            satisfaction_score=0.97,
            spot_type='dark')
        print(f"✅ Finished processing for '{name}'.")

        if monitor_ui:
            _monitor_single_object(name, frames_bgr, ellipse_results)

        ellipses_results[name] = ellipse_results
        
    # 4. Save all results
    # Create a list of DataFrames with prefixed columns
    prefixed_dfs = [df.add_prefix(f'{key}_') for key, df in ellipses_results.items()]
    # Concatenate the list of DataFrames horizontally (axis=1)
    results_df = pd.concat(prefixed_dfs, axis=1)
    results_df.to_csv(output_path, index=False)
    print(f"--- Processing Complete ---")
