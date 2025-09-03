import os
import sys
import cv2
import numpy as np
import yaml

from utils.kinect_mkv_manager import (
    KinectMKV,
    KinectFrame
)

from preprocessing.common import PointCloudDataHandler

from preprocessing.forearm_extraction import (
    ForearmFrameParametersFileHandler,
    ForearmParameters,
    RegionOfInterest,

    ArmSegmentation,

    ForearmSegmentationParamsFileHandler
)


CONFIG_PATH = 'config.yaml'

# -----------------------------------------------------------------
# 1. Custom Exceptions for Better Error Handling
# -----------------------------------------------------------------
class ConfigError(Exception):
    """Exception raised for errors in the config file."""
    pass

# -----------------------------------------------------------------
# 3. Helper Functions (Separation of Concerns)
# -----------------------------------------------------------------
def load_config(config_path: str) -> dict:
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigError(f"Configuration file not found at: {config_path}")
    except yaml.YAMLError as e:
        raise ConfigError(f"Error parsing YAML configuration: {e}")

def get_corners_from_roi(roi: RegionOfInterest) -> np.ndarray:
    p1 = (roi.top_left_corner.x, roi.top_left_corner.y)
    p2 = (roi.bottom_right_corner.x, roi.bottom_right_corner.y)
    return np.array([p1, p2])

def get_3d_cuboid_from_roi(frame: KinectFrame, roi: RegionOfInterest) -> np.ndarray:
    """Converts the 2D ROI into 3D corner points for the box filter."""
    top_left_corner, bottom_right_corner = get_corners_from_roi(roi)
    p1 = frame.convert_xy_to_xyz(top_left_corner)
    p2 = frame.convert_xy_to_xyz(bottom_right_corner)
    return np.array([p1, p2])

def show_annotated_frames(
        roi: RegionOfInterest, 
        frame: KinectFrame):
    """Displays the depth and color frames with the ROI rectangle."""
    pt1, pt2 = get_corners_from_roi(roi)
    
    depth_mat = frame.get_depth_for_viewing()
    cv2.rectangle(depth_mat, pt1, pt2, (0, 0, 255), 2)
    cv2.imshow("Kinect Depth Frame with ROI", depth_mat)

    color_mat = frame.color
    cv2.rectangle(color_mat, pt1, pt2, (0, 0, 255), 2)
    cv2.imshow("Kinect Color Frame with ROI", color_mat)
    
    print("Press any key in an image window to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _resize_with_padding(image: np.ndarray, target_dims: tuple) -> np.ndarray:
    """
    Resizes an image to a target dimension while preserving the aspect ratio
    by padding the background with black.
    """
    target_w, target_h = target_dims
    if image.shape[0] == 0 or image.shape[1] == 0:
        if len(image.shape) == 3:
            return np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        else:
            return np.zeros((target_h, target_w), dtype=image.dtype)

    src_h, src_w = image.shape[:2]
    src_ratio = src_w / src_h
    target_ratio = target_w / target_h

    if src_ratio > target_ratio:
        new_w = target_w
        new_h = int(new_w / src_ratio)
    else:
        new_h = target_h
        new_w = int(new_h * src_ratio)

    resized_image = cv2.resize(image, (new_w, new_h))

    if len(image.shape) == 3:
        padded_image = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
    else:
        padded_image = np.zeros((target_h, target_w), dtype=image.dtype)
        
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    return padded_image

def _create_monitoring_frame(frame_index: int,
                             depth_image: np.ndarray,
                             target_dims: tuple = (1080, 1920),
                             display_dims: tuple = (1080, 1920)) -> np.ndarray:
    """
    Creates a monitoring visualization frame but does not display it.
    This function generates the canvas, draws images, overlays, and text, then returns the final image.
    """
    # --- 0. Setup Dimensions ---
    display_h, display_w = display_dims
    panel_width = 450
    images_total_width = display_w - panel_width
    if images_total_width <= 0:
        raise ValueError("Display width is too small for the text panel.")

    # --- 1. Prepare Visualizations and Draw Overlays ---
    target_h, target_w = target_dims
    depth_present = depth_image is not None

    # Use a black image as a placeholder if the source is missing
    
    if depth_present and depth_image.shape[:2] == target_dims:
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_vis = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    else:
        depth_vis = np.zeros((target_h, target_w, 3), dtype=np.uint8)


    # --- 2. Create Final Canvas and Assemble Components ---
    final_canvas = np.zeros((display_h, display_w, 3), dtype=np.uint8)

    image_target_box = (images_total_width, display_h)
    resized_depth = _resize_with_padding(depth_vis, image_target_box)
    final_canvas[0:display_h, 0:images_total_width] = resized_depth

    # --- 3. Draw Text on the Right Panel ---
    text_x_start = images_total_width + 15
    text_y = 30
    cv2.putText(final_canvas, f"Frame: {frame_index}", (text_x_start, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    text_y += 40
    return final_canvas



def _display_frame(frame: np.ndarray) -> bool:
    """
    Displays a given frame in a window and handles user input.
    Returns True if the user presses 'q', False otherwise.
    """
    cv2.imshow("XYZ Monitoring", frame)
    key = cv2.waitKey(1) & 0xFF
    return key == ord('q')

def show_video(source_video, frame_index = 542):
    from pyk4a import PyK4APlayback
    from datetime import timedelta

    display_dims = (1080, 1920)

    playback = PyK4APlayback(source_video)
    playback.open()
    print(f"Successfully opened MKV: {source_video}")

    timestamp_td = timedelta(seconds=frame_index / 30)
    timestamp_usec = int(timestamp_td.total_seconds() * 1_000_000)
    playback.seek(timestamp_usec)

    while True:
        capture = playback.get_next_capture()
        if capture is None: break

        visual_frame = _create_monitoring_frame(
            frame_index=frame_index,
            depth_image=capture.transformed_depth, 
            display_dims=display_dims
        )
        
        _display_frame(visual_frame)

        frame_index += 1
    

# -----------------------------------------------------------------
# 4. Main Orchestrator (uses Dependency Injection)
# -----------------------------------------------------------------
def extract_forearm(
        video_path: str,
        video_config: ForearmParameters,
        output_ply_path: str,
        output_params_path: str,
        *,
        monitor: str = False,
        interactive: str = False,
):
    """
    Orchestrates the entire processing pipeline for a single file.
    
    Args:
        config (dict): The loaded configuration dictionary.
    """
    
    # Load configuration
    config = load_config(os.path.join(os.path.dirname(__file__), CONFIG_PATH))

    if os.path.exists(output_params_path):
        segmentation_params = ForearmSegmentationParamsFileHandler.load(output_params_path)
    else:
        segmentation_params = config['segmentation_params']

    try:
        # 1. Load Data
        print(f"--- Processing {os.path.basename(video_path)} ---")
        
        # 2. Setup Dependencies
        # Dependencies are created here and "injected" into the functions that need them.
        with KinectMKV(video_path) as mkv:
            frame: KinectFrame = mkv[video_config.frame_id]
            point_cloud = frame.generate_o3d_point_cloud()
            
            segmenter = ArmSegmentation(segmentation_params, interactive=interactive)

            cuboid_oppposed_corners = get_3d_cuboid_from_roi(frame, video_config.region_of_interest)
            pcd = segmenter.preprocess(
                point_cloud,
                cuboid_oppposed_corners,
                monitor #  config['visualization']['show_intermediate_steps']
            )
            
            pcd = segmenter.extract_arm(
                pcd,
                monitor #  config['visualization']['show_intermediate_steps']
            )
            
            # 4. Finalize
            if monitor: #  config['visualization']['show_intermediate_steps']
                show_annotated_frames(video_config.region_of_interest, frame)
            PointCloudDataHandler.save(pcd, output_path=output_ply_path)
            # save the parameters
            ForearmSegmentationParamsFileHandler.save(segmenter.params, output_params_path)
            
        return output_ply_path

    except (ConfigError) as e:
        print(f"❌ ERROR: A pipeline failure occurred: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

