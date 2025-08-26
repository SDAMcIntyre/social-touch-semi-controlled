import numpy as np
import cv2
import open3d as o3d
from pyk4a import PyK4APlayback, ImageFormat
from pyk4a.config import FPS
from datetime import timedelta
import logging
import struct

def pack_rgb(r, g, b):
    """
    Packs 8-bit R, G, B channels into a single 32-bit integer and
    reinterprets it as a float.

    Args:
        r (int): Red channel value (0-255).
        g (int): Green channel value (0-255).
        b (int): Blue channel value (0-255).

    Returns:
        float: The packed RGB color as a 32-bit float.
    """
    # Ensure values are within the valid 8-bit range
    r = int(r) & 0xFF
    g = int(g) & 0xFF
    b = int(b) & 0xFF

    # Bit-shift to pack into a single 32-bit integer
    # (R << 16) | (G << 8) | B
    rgb_int = (r << 16) | (g << 8) | b

    # Reinterpret the 32-bit integer as a 32-bit float
    # 'I' = unsigned int (4 bytes), 'f' = float (4 bytes)
    packed_float = struct.unpack('f', struct.pack('I', rgb_int))[0]
    
    return packed_float


def unpack_rgb(rgb_float):
    """
    Unpacks a 32-bit float used by PCL into 8-bit R, G, B channels.

    Args:
        rgb_float (float): The packed RGB color as a 32-bit float.

    Returns:
        tuple: A tuple containing the (R, G, B) integer values (0-255).
    """
    # Reinterpret the 32-bit float as a 32-bit integer
    rgb_int = struct.unpack('I', struct.pack('f', rgb_float))[0]
    
    # Use bitwise operations to extract the channels
    r = (rgb_int >> 16) & 0xFF
    g = (rgb_int >> 8) & 0xFF
    b = rgb_int & 0xFF
    
    return (r, g, b)

class CaptureManager:
    """
    Manages reading and processing of Azure Kinect data from an MKV recording.
    This version is compatible with pyk4a >= 1.5.0.
    """

    def __init__(self, video_path: str, ref_frame_idx: int):
        """
        Initializes the CaptureManager and loads the reference frame.

        Args:
            video_path (str): Path to the .mkv Kinect recording.
            ref_frame_idx (int): The index of the frame to load initially.
        """
        self._video_path = video_path
        self._ref_frame_idx = ref_frame_idx

        # Initialize pyk4a playback object
        self._playback = PyK4APlayback(self._video_path)
        self._playback.open()
        self.color = None
        
        fps_map = {
            FPS.FPS_30: 30
        }
        self._fps = fps_map.get(self._playback.configuration['camera_fps'], 30) # Default to 30 if not found
        
        # Internal state for the current loaded capture
        self._current_capture = None
        
        # Load the initial reference frame
        print(f"INFO: CaptureManager initializing with video: '{self._video_path}'")
        self.load_frame(self._ref_frame_idx)
    
    def load_frame(self, frame_idx: int):
        """
        Seeks to and loads a specific frame from the video file.
        The data from this frame can then be accessed via properties like
        .color, .depth, .xyz_map, etc.

        Args:
            frame_idx (int): The index of the frame to load.
        """
        try:
            # Calculate the timestamp for the desired frame
            timestamp_td = timedelta(microseconds=frame_idx * (1_000_000 / self._fps))
            timestamp_usec = int(timestamp_td.total_seconds() * 1_000_000)
            self._playback.seek(timestamp_usec)
            
            # Read the capture from the new position, until a frame with both color and depth is found
            read = True
            while read:
                self.get_next_capture()
                if not(self.color is None or self._current_capture.depth is None):
                    read = False
                else:
                    self._ref_frame_idx += 1
            
            print(f"INFO: Successfully loaded frame {frame_idx}.")
        except (StopIteration, EOFError): # StopIteration is the new EOF for `next()`
            print(f"ERROR: Frame index {frame_idx} is out of bounds.")
            self._current_capture = None

    def get_next_capture(self):
        self.color = None
        self._current_capture = self._playback.get_next_capture()
        # Check if the color format is MJPG and decompress it if so.
        # If the format is uncompressed (like BGRA), do nothing
        if not(self._current_capture.color is None) and self._playback.configuration['color_format'] == ImageFormat.COLOR_MJPG:
            # cv2.imdecode takes the 1D byte buffer and returns a 3-channel BGR image.
            self.color = cv2.imdecode(self._current_capture.color, cv2.IMREAD_COLOR)
        else:
            self.color = self._current_capture.color

    @property
    def depth_colorpov(self) -> np.ndarray | None:
        """
        Returns the depth map transformed to the color camera's perspective.
        Values are in millimeters.
        """
        if self._current_capture is None or self._current_capture.depth is None:
            return None
        return self._current_capture.transformed_depth

    @property
    def xyz_map(self) -> np.ndarray | None:
        """
        Returns the point cloud as a 3D map (image format) in the color camera's perspective.
        Each pixel (x, y) contains the (X, Y, Z) coordinates in millimeters.
        """
        if self._current_capture is None:
            return None
        return self._current_capture.transformed_depth_point_cloud

    def generate_point_cloud(self):
        """
        Generates a PCL PointCloud_PointXYZRGB object from the current frame.
        
        Returns:
            pcl.PointCloud_PointXYZRGB or None if data is unavailable.
        """
        xyz = self.xyz_map
        color_img = self.color
        
        if xyz is None or color_img is None:
            # Use logging instead of print for better application management
            logging.warning("Cannot generate point cloud, xyz_map or color image is missing.")
            return None

        # Validate shapes to ensure data integrity
        expected_dims = 3 # Expecting (height, width, channels)
        if xyz.ndim != expected_dims or color_img.ndim != expected_dims:
            logging.error(f"Shape mismatch: xyz_map has {xyz.ndim} dims (shape:{xyz.shape}), color_img has {color_img.ndim} dims (shape:{color_img.shape}). Both must be {expected_dims}.")
            return None
        
        if xyz.shape[:2] != color_img.shape[:2]:
            logging.error(f"Dimension mismatch: xyz_map shape {xyz.shape[:2]} != color_img shape {color_img.shape[:2]}.")
            return None

        valid_mask = (xyz[:, :, 2] > 0) & ~np.isnan(xyz).any(axis=2)
        points_xyz = xyz[valid_mask]
        
        points_bgra = color_img[valid_mask]
        points_rgb = points_bgra[:, [2, 1, 0]] / 255

        if len(points_xyz) == 0:
            print("WARN: No valid points found to create a point cloud.")
            return None, None

        # Ensure the points array is float32
        points_xyz = points_xyz.astype(np.float32)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz)
        pcd.colors = o3d.utility.Vector3dVector(points_rgb)

        return pcd

    def convert_xy_coordinate_to_xyz_mm(self, xy: tuple[int, int]) -> np.ndarray:
        """
        Looks up the 3D coordinate (in mm) for a given 2D pixel coordinate.

        Args:
            xy (tuple): The (x, y) pixel coordinate.

        Returns:
            np.ndarray: The [X, Y, Z] coordinate in millimeters, or [0,0,0] if invalid.
        """
        x, y = xy
        point_cloud_map = self.xyz_map
        if point_cloud_map is not None and 0 <= y < point_cloud_map.shape[0] and 0 <= x < point_cloud_map.shape[1]:
            return point_cloud_map[y, x]
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def get_depth_as_rgb(self) -> np.ndarray | None:
        """
        Converts the raw depth map (in mm) to a visualizable 8-bit RGB image.
        """
        depth = self.depth_colorpov
        if depth is None:
            return None
        depth_clipped = np.clip(depth, 0, 5000)
        norm = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
