"""
This package provides modules for data handling and graphical user interfaces
related to 3D scenes, point clouds, and video data.
"""

# --- Data Access Layer ---
# For handling and managing data sources like point clouds, videos, and Kinect captures.
from .data_access.glb_data_handler import GLBDataHandler
from .data_access.kinect_mkv_manager import KinectFrame, KinectMKV
from .data_access.kinect_pointcloud_wrapper import KinectPointCloudView
from .data_access.pc_data_handler import PointCloudDataHandler
from .data_access.video_mp4_manager import VideoMP4Manager

# --- GUI Layer ---
# For visualizing data and interacting with it through graphical components.
from .gui.frame_roi_square import FrameROISquare
from .gui.scene_viewer import (
    LazyPointCloudSequence,
    PersistentOpen3DPointCloudSequence,
    PersistentPointCloudSequence,
    PointCloudData,
    PointCloudSequence,
    Open3DTriangleMeshSequence,
    Trajectory,
    SceneViewer
)
from .gui.video_frame_selector import VideoFrameSelector
from .gui.video_frames_selector import VideoFramesSelector

# --- Public API Definition ---
# Explicitly lists all names that are part of the public API.
__all__ = [
    # Data Access Classes
    "GLBDataHandler",
    "KinectFrame",
    "KinectMKV",
    "KinectPointCloudView",
    "PointCloudDataHandler",
    "VideoMP4Manager",
    
    # GUI Classes
    "FrameROISquare",
    "SceneViewer",
    "VideoFrameSelector",
    "VideoFramesSelector",

    # Scene Viewer Data Structures
    "LazyPointCloudSequence",
    "PersistentOpen3DPointCloudSequence",
    "PersistentPointCloudSequence",
    "PointCloudData",
    "PointCloudSequence",
    "Open3DTriangleMeshSequence",
    "Trajectory",
]