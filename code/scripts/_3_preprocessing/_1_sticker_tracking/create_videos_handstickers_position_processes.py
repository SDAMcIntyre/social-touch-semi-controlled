# --- 1. Standard Library Imports ---
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

# Note: tkinter is often used for simple GUI tasks like getting screen dimensions,
# but can be a heavy dependency if that's its only use.
import tkinter as tk

# --- 2. Third-party Imports ---
import cv2
import h5py
import numpy as np
import pandas as pd

# --- 3. Local Application Imports ---
from .utils.ColorFamilyModel import ColorFamilyModel
from .utils.roi_to_position_funcs import (
    crop_frames_by_rois,
    generate_color_config,
    load_video_frames_bgr,
    parse_rois_from_dataframe,
    parse_ellipses_from_dict
)



# --- Presentation Components ---
# --- Helper Function ---
def _get_screen_resolution() -> Tuple[int, int]:
    """
    Gets the primary screen resolution using tkinter.
    Falls back to a default size if tkinter is unavailable.
    """
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main tkinter window
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except tk.TclError:
        print("Warning: tkinter not available. Falling back to 1920x1080 screen size.")
        return 1920, 1080

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


def prepare_roi_correlation_display(
    frame_bgr: np.ndarray,
    roi_location: Tuple[int, int, int, int],
    correlation_map: np.ndarray,
    ellipse_data: Dict[str, Any]
) -> Optional[np.ndarray]:
    """
    Prepares a combined view of the zoomed ROI and the correlation map.

    Args:
        full_frame (np.ndarray): The full-size original video frame (for extracting the ROI).
        roi_location (Tuple[int, int, int, int]): The (x, y, w, h) of the ROI.
        correlation_map (np.ndarray): The generated correlation map.
        ellipse_data (Dict[str, Any]): Dictionary with fitted ellipse parameters.

    Returns:
        Optional[np.ndarray]: The combined BGR image, or None if dimensions are invalid.

    """
    
    # Normalize and colorize the correlation map
    vis_map = cv2.normalize(correlation_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    vis_map_bgr = cv2.cvtColor(vis_map, cv2.COLOR_GRAY2BGR)
    # import matplotlib.pyplot as plt;; plt.imshow(vis_map_bgr); plt.show(block=True)

    # Extract the zoomed-in ROI
    roi_x, roi_y, roi_w, roi_h = roi_location
    end_x = min(roi_x + roi_w, frame_bgr.shape[1])
    end_y = min(roi_y + roi_h, frame_bgr.shape[0])
    
    # Need to handle the color conversion for the ROI source frame
    if len(frame_bgr.shape) == 2:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
    else:
        frame_bgr = frame_bgr
        
    zoomed_roi = frame_bgr[roi_y:end_y, roi_x:end_x]

    # Safely check and draw the ellipses
    center_val = ellipse_data.get('center', (math.nan, math.nan))
    axes_val = ellipse_data.get('axes', (math.nan, math.nan))
    angle_val = ellipse_data.get('angle', math.nan)

    if not any(math.isnan(c) for c in center_val) and \
       not any(math.isnan(a) for a in axes_val) and \
       not math.isnan(angle_val):
        
        # Draw ellipse on zoomed ROI (local ROI coordinates)
        axes = (int(axes_val[0] / 2), int(axes_val[1] / 2))
        angle = int(angle_val)
        local_center = (int(center_val[0])-roi_x, int(center_val[1])-roi_y)
        cv2.ellipse(zoomed_roi, local_center, axes, angle, 0, 360, (0, 0, 255), 2)

    # Combine the two images side-by-side
    if zoomed_roi.shape[0] == 0 or vis_map_bgr.shape[0] == 0:
        return None # Return None if either image is empty

    # Resize correlation map to match ROI height for clean stacking
    h_roi, _ = zoomed_roi.shape[:2]
    h_map, w_map = vis_map_bgr.shape[:2]
    scale_ratio = h_roi / h_map
    new_w_map = int(w_map * scale_ratio)
    resized_vis_map_bgr = cv2.resize(vis_map_bgr, (new_w_map, h_roi), interpolation=cv2.INTER_AREA)

    combined_view = np.hstack([zoomed_roi, resized_vis_map_bgr])
    return combined_view

def normalize_frame_size(
    frame: Optional[np.ndarray], 
    target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Ensures a frame fits within target dimensions, handling both color and grayscale.

    Args:
        frame: The input image frame (can be None, grayscale, or color).
        target_size: A tuple (width, height) for the output frame dimensions.

    Returns:
        A new frame that has the exact target dimensions.
    """
    target_width, target_height = target_size

    # --- 1. Handle Empty Frame ---
    if frame is None or frame.size == 0:
        # Default to a 3-channel black frame if input is empty
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # --- 2. Dynamically handle Grayscale vs. Color ---
    if frame.ndim == 2:
        # Grayscale image
        fh, fw = frame.shape
        channels = 1
        canvas = np.zeros((target_height, target_width), dtype=np.uint8)
    elif frame.ndim == 3:
        # Color image
        fh, fw, channels = frame.shape
        canvas = np.zeros((target_height, target_width, channels), dtype=frame.dtype)
    else:
        # Unsupported shape, raise an error for clarity
        raise ValueError(f"Unsupported frame shape: {frame.shape}")

    # --- 3. Main Logic (Resizing and Padding) ---
    if fh == target_height and fw == target_width:
        return frame

    scale = min(target_width / fw, target_height / fh)
    
    # Only resize if the frame is larger than the target canvas
    if scale < 1.0:
        new_size = (int(fw * scale), int(fh * scale))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
        if frame.ndim == 2: # cv2.resize can sometimes alter dimensions
            fh, fw = frame.shape
        else:
            fh, fw, _ = frame.shape


    # Calculate offsets for centering
    x_offset = (target_width - fw) // 2
    y_offset = (target_height - fh) // 2

    # Paste the frame onto the canvas
    canvas[y_offset:y_offset+fh, x_offset:x_offset+fw] = frame
    
    return canvas

def get_max_roi_values(tracked_rois: pd.Series) -> Tuple[int, int, int, int]:
    """
    Calculates the maximum value for each column across all ROIs in a Series.

    This function is useful for finding the bounding box that encompasses all
    individual ROIs, for instance, to determine the required size of a canvas
    or a crop area.

    Args:
        tracked_rois: A pandas Series where each element is a list or tuple
                      representing an ROI, typically as (x, y, width, height).

    Returns:
        A tuple containing the maximum x, y, width, and height found across
        all ROIs. Returns (0, 0, 0, 0) if the input Series is empty.
    """
    # 0. Filter out any NaN values from the Series to prevent processing errors.
    valid_rois = tracked_rois.dropna()

    # 1. Handle edge case of an empty input to prevent errors
    if valid_rois.empty:
        return (0, 0, 0, 0)

    # 2. Convert the Series of lists/tuples into a DataFrame
    # This is a highly efficient, vectorized operation.
    # Column names are added for clarity.
    roi_df = pd.DataFrame(
        valid_rois.tolist(), 
        columns=['x', 'y', 'width', 'height']
    )

    # 3. Calculate the maximum value for each column at once
    max_values = roi_df.max()

    # 4. Return the results as a clean tuple of integers
    return (
        int(max_values['x']),
        int(max_values['y']),
        int(max_values['width']),
        int(max_values['height'])
    )


# --- Sentinel Object Definition ---
# https://python-patterns.guide/python/sentinel-object/
# Create a unique object to act as a sentinel. Its only purpose is to be
# checked by identity. This is the standard way to solve the problem of 
# keeping an unspecified/ungiven default value if None is a useful value.
_SENTINEL = object()

# --- Encapsulate validation logic (Refactored v3) ---
def are_data_valid(
    frame: Optional[np.ndarray], 
    tracked_roi: Optional[Tuple[int, int, int, int]],
    ellipse: Optional[Dict[str, Any]],
    corr_map: Any = _SENTINEL  # Use the sentinel as the default
) -> bool:
    """
    Checks if the frame and all associated tracking data are valid.

    This function validates that required data is present and well-formed.
    - The `corr_map` argument is optional.
    - However, if `corr_map` IS provided, its value cannot be `None`.

    Args:
        frame (Optional[np.ndarray]): The video frame. Must not be None.
        tracked_roi (Optional[Sequence[float]]): The tracked ROI.
            Must not be None, empty, or contain NaNs.
        ellipse (Optional[Dict[str, Any]]): A dictionary for the fitted 
            ellipse. Must not be None.
        corr_map (Any, optional): The correlation map. If provided, it cannot
            be None. Defaults to an internal sentinel value.

    Returns:
        bool: True if all provided data is valid, False otherwise.
    """
    # 1. Validate the always-required arguments
    if frame is None or ellipse is None:
        return False

    # 2. Validate the ROI separately for clarity
    if not isinstance(tracked_roi, tuple) or len(tracked_roi) == 0:
        return False
    if any(math.isnan(v) for v in tracked_roi):
        return False

    # 3. Validate corr_map using the sentinel logic
    # This check is only failed if the user *explicitly* passed `corr_map=None`.
    # If the argument was omitted, `corr_map` is `_SENTINEL`, so this is skipped.
    if corr_map is None:
        return False
    
    # If all checks pass, the data is valid.
    return True


def save_monitoring_video_streamed(
    frames_bgr: List,
    tracked_rois: pd.Series,
    ellipses: pd.DataFrame,
    corr_maps: List[Any],
    full_frame_output_path: str,
    roi_corr_output_path: str,
    fps: int = 30
):
    """
    Generates and saves monitoring videos using a memory-efficient streaming approach.

    If an output video file already exists, its processing will be skipped.
    """
    print("Initializing video generation...")
    
    # Determine which videos need to be processed
    process_full_video = not os.path.exists(full_frame_output_path)
    process_roi_video = not os.path.exists(roi_corr_output_path)

    # Early exit if both videos already exist
    if not process_full_video and not process_roi_video:
        print("✅ Both output videos already exist. Skipping all processing.")
        print(f"  - Found: {full_frame_output_path}")
        print(f"  - Found: {roi_corr_output_path}")
        return

    if not frames_bgr:
        print("Error: No input frames provided.")
        return

    # --- 2. Initialize Video Writers Conditionally ---
    height, width, _ = frames_bgr[0].shape
    _, _, width_roi, height_roi = get_max_roi_values(tracked_rois)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    full_writer = None
    roi_writer = None

    if process_full_video:
        print(f"➡️ Will generate full frame video: {full_frame_output_path}")
        full_writer = cv2.VideoWriter(full_frame_output_path, fourcc, fps, (width, height))
        if not full_writer.isOpened():
            print(f"Error: Could not open video writer for {full_frame_output_path}.")
            # Disable processing for this writer if it fails to open
            process_full_video = False
    else:
        print(f"☑️ Skipping full frame video, file already exists.")

    if process_roi_video:
        print(f"➡️ Will generate ROI/Correlation video: {roi_corr_output_path}")
        roi_writer = cv2.VideoWriter(roi_corr_output_path, fourcc, fps, (height_roi, width_roi))
        if not roi_writer.isOpened():
            print(f"Error: Could not open video writer for {roi_corr_output_path}.")
            process_roi_video = False
    else:
        print(f"☑️ Skipping ROI/Correlation video, file already exists.")

    # Final check in case writers failed to open
    if not process_full_video and not process_roi_video:
        print("No valid video writers to use. Aborting.")
        return

    # --- 3. Processing and Writing Loop ---
    print("\nProcessing frames...")
    processing_df = pd.DataFrame({
        'frame_bgr': frames_bgr,
        'tracked_roi': tracked_rois,
        'corr_map': corr_maps
    })
    master_df = processing_df.join(ellipses)
    
    zipped_data = zip(frames_bgr, tracked_rois, ellipses.iterrows(), corr_maps)
    frames_processed = 0
    try:
        for frame_id, (frame_bgr, tracked_roi, ellipse, corr_map) in enumerate(zipped_data):
            try:
                frame_copy = frame_bgr.copy()

                # Prepare and write the first video's frame if needed
                if process_full_video:
                    if not are_data_valid(frame_copy, tracked_roi, ellipse):
                        full_writer.write(normalize_frame_size(None, (width, height)))
                    else:
                        full_frame_display = prepare_full_frame_display(frame_copy, frame_id, tracked_roi, ellipse)
                        full_writer.write(full_frame_display)
                
                # Prepare and write the second video's frame if needed
                if process_roi_video:
                    if not are_data_valid(frame_copy, tracked_roi, ellipse, corr_map):
                        roi_writer.write(normalize_frame_size(None, (width_roi, height_roi)))
                    else:
                        roi_corr_display = prepare_roi_correlation_display(frame_copy, tracked_roi, corr_map, ellipse)
                        normalized_roi_frame = normalize_frame_size(roi_corr_display, (width_roi, height_roi))
                        roi_writer.write(normalized_roi_frame)

                frames_processed += 1

            except (TypeError, ValueError) as e:
                print(f"  -> Skipping frame {frame_id} due to invalid data: {e}")

        print(f"\nProcessing complete. {frames_processed} frames analyzed.")

    finally:
        # --- 4. Finalization Phase ---
        print("Releasing video writers...")
        if full_writer:
            full_writer.release()
        if roi_writer:
            roi_writer.release()
        print("Done.")

def get_video_metadata(video_path: str) -> Optional[Tuple[float, int, int]]:
    print(f"Loading video '{video_path}' into memory...")
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at '{video_path}'")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file '{video_path}'")
        return None
    
    # Get video properties for later use
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return fps, width, height



def load_ellipses_from_csv(csv_path: str) -> Optional[Dict[int, List[Dict[str, Any]]]]:
    """
    Loads object detection data from a CSV file.

    It organizes detections by frame number into a dictionary for efficient O(1) lookups.

    Args:
        csv_path (str): The path to the input CSV file.

    Returns:
        A dictionary mapping frame numbers to a list of detection records,
        or None if an error occurs.
    """
    print(f"Loading detection data from '{csv_path}'...")
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found at '{csv_path}'")
        return None
        
    try:
        detections_df = pd.read_csv(csv_path)
        detections_by_frame = {
            frame_num: group.to_dict('records') 
            for frame_num, group in detections_df.groupby('frame_number')
        }
        print(f"✅ Data loaded for {len(detections_by_frame)} frames.")
        return detections_by_frame
    except Exception as e:
        print(f"❌ Error reading or processing CSV file: {e}")
        return None
    

def load_correlation_results_from_hdf5(input_path: str) -> Optional[Dict[str, List[np.ndarray]]]:
    """
    Loads correlation data from an HDF5 file into a dictionary of frame lists.

    This function is the inverse of _save_correlation_results_to_hdf5. It assumes the
    HDF5 file contains groups, and each group contains datasets named 'frame_xxxxx'
    that should be loaded in alphanumeric order.

    Args:
        input_path (str): The path to the input HDF5 file.

    Returns:
        Optional[Dict[str, List[np.ndarray]]]: A dictionary where keys are group names
                                               and values are lists of NumPy arrays (frames).
                                               Returns None if the file cannot be read.
    """
    print(f"Loading data from '{input_path}'...")
    loaded_data: Dict[str, List[np.ndarray]] = {}

    try:
        with h5py.File(input_path, 'r') as hf:
            # Iterate over the groups in the file (which correspond to our dictionary keys)
            for key in hf.keys():
                group = hf[key]
                loaded_data[key] = []
                
                # It's crucial to sort the dataset names to ensure correct frame order
                dataset_names = sorted(group.keys())

                # Read each dataset (frame) in the group and append it to the list
                for dataset_name in dataset_names:
                    frame = group[dataset_name][:]  # [:] reads the full array into memory
                    loaded_data[key].append(frame)
        
        print("Load complete.")
        return loaded_data

    except FileNotFoundError:
        print(f"❌ Error: The file '{input_path}' was not found.")
        return None
    except OSError as e:
        print(f"❌ Error reading HDF5 file '{input_path}'. It may be corrupted or not a valid HDF5 file. Details: {e}")
        return None


def save_video_handstickers_position_processes(
    video_path: str,
    stickers_roi_csv_path,
    tracking_pos_csv_path: str,
    tracking_corr_hdf5_path: str,
    track_video_path: Optional[str] = None,
    roi_corr_video_path: Optional[str] = None
):
    """
    Main orchestration function to review, validate, and correct object tracking data.

    It loads video and tracking data, allows a user to review the annotated video,
    and if corrections are made, it processes them and updates the metadata.
    If an output_video_path is provided, it saves the annotated video.

    Args:
        video_path: Absolute path to the source video file.
        tracking_pos_csv_path: Absolute path to the CSV file with tracking data.
        metadata_path: Absolute path to the JSON metadata file to be read and updated.
        output_video_path (Optional[str]): Path to save the annotated MP4 video.
                                           If None, the video is not saved.
    """
    print(f"--- Starting video creation for: '{os.path.basename(video_path)}' ---")

    # 2. Data Loading
    frames = load_video_frames_bgr(video_path)
    fps, width, height = get_video_metadata(video_path)

    ellipse_results = load_ellipses_from_csv(tracking_pos_csv_path)
    if not ellipse_results: 
        return
    corr_map_results = load_correlation_results_from_hdf5(tracking_corr_hdf5_path)
    
    tracking_df = pd.read_csv(stickers_roi_csv_path)

    print(f"✅ Successfully loaded data.")


    for object_name in list(corr_map_results.keys()):
        obj_track_video_path = str(track_video_path).replace('.mp4', f'_{object_name}.mp4')
        obj_roi_corr_video_path = str(roi_corr_video_path).replace('.mp4', f'_{object_name}.mp4')
    
        current_tracked_roi = parse_rois_from_dataframe(tracking_df, object_name) 
        ellipse_result = parse_ellipses_from_dict(ellipse_results, object_name)
        corr_map = corr_map_results[object_name]

        save_monitoring_video_streamed(
            frames, 
            current_tracked_roi,
            ellipse_result, 
            corr_map,
            full_frame_output_path=obj_track_video_path,
            roi_corr_output_path=obj_roi_corr_video_path,
            fps=fps
        )
    
