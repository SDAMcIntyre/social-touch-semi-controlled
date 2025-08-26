import yaml
import json
import pathlib
import open3d as o3d
import numpy as np
import cv2
from dataclasses import dataclass

from ..forearm_analysis.capture_manager import CaptureManager


# -----------------------------------------------------------------
# 1. Custom Exceptions for Better Error Handling
# -----------------------------------------------------------------
class ConfigError(Exception):
    """Exception raised for errors in the config file."""
    pass

class DataIOError(Exception):
    """Exception raised for errors related to data input/output."""
    pass

class PipelineError(Exception):
    """Exception raised for errors during the processing pipeline."""
    pass


@dataclass
class RegionOfInterest:
    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int

@dataclass
class VideoData:
    video_path: str
    reference_frame_idx: int
    roi: RegionOfInterest
    frame_width: int
    frame_height: int
    fps: float


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

def extract_video_config(file_path: str) -> VideoData:
    """Reads metadata from the specified JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            j = json.load(f)
        roi_data = j["region_of_interest"]
        return VideoData(
            video_path=j["video_path"].replace('', 'ö'),
            reference_frame_idx=j["reference_frame_idx"],
            roi=RegionOfInterest(
                top_left_x=roi_data["top_left_corner"]["x"], top_left_y=roi_data["top_left_corner"]["y"],
                bottom_right_x=roi_data["bottom_right_corner"]["x"], bottom_right_y=roi_data["bottom_right_corner"]["y"]
            ),
            frame_width=j["frame_width"], frame_height=j["frame_height"], fps=j["fps"]
        )
    except FileNotFoundError:
        raise DataIOError(f"Metadata file not found: {file_path}")
    except (KeyError, json.JSONDecodeError) as e:
        raise DataIOError(f"Error reading or parsing metadata file {file_path}: {e}")


def get_3d_cuboid_from_roi(capture_manager: CaptureManager, roi: RegionOfInterest) -> np.ndarray:
    """Converts the 2D ROI into 3D corner points for the box filter."""
    p1 = capture_manager.convert_xy_coordinate_to_xyz_mm([roi.top_left_x, roi.top_left_y])
    p2 = capture_manager.convert_xy_coordinate_to_xyz_mm([roi.bottom_right_x, roi.bottom_right_y])
    return np.array([p1, p2])

def show_annotated_frames(
        data: 'VideoData', 
        cm: 'CaptureManager'):
    """Displays the depth and color frames with the ROI rectangle."""
    pt1 = (data.roi.top_left_x, data.roi.top_left_y)
    pt2 = (data.roi.bottom_right_x, data.roi.bottom_right_y)
    
    depth_mat = cm.get_depth_as_rgb()
    cv2.rectangle(depth_mat, pt1, pt2, (0, 0, 255), 2)
    cv2.imshow("Kinect Depth Frame with ROI", depth_mat)

    color_mat = cm.color
    cv2.rectangle(color_mat, pt1, pt2, (0, 0, 255), 2)
    cv2.imshow("Kinect Color Frame with ROI", color_mat)
    
    print("Press any key in an image window to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


