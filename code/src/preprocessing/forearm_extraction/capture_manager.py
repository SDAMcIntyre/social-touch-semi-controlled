# File: kinect_frame_processor.py

import logging
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

# Import the high-level facade and its data structure
from preprocessing.common import (
    KinectMKV, 
    KinectFrame
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class KinectFrameProcessor:
    """
    Processes a single KinectFrame to provide specialized data representations.

    This class acts as a dedicated tool for operating on a frame's data,
    such as converting depth maps for visualization or looking up specific
    3D coordinates. It uses a KinectFrame object as its data source,
    promoting a clean separation of concerns.

    Attributes:
        frame (KinectFrame): The source frame data object.
    """
    def __init__(self, frame: KinectFrame):
        """
        Initializes the processor with a specific KinectFrame.

        Args:
            frame: An instance of KinectFrame, typically obtained from a
                   KinectMKV object (e.g., `mkv[frame_index]`).
        """
        if not isinstance(frame, KinectFrame):
            raise TypeError("KinectFrameProcessor must be initialized with a KinectFrame object.")
        self._frame = frame
        logging.info("KinectFrameProcessor initialized for a new frame.")

    # --- Properties delegating directly to the KinectFrame ---

    @property
    def color(self) -> np.ndarray | None:
        """The color image in BGR format."""
        return self._frame.color

    @property
    def depth(self) -> np.ndarray | None:
        """The depth map transformed to the color camera's perspective (in mm)."""
        return self._frame.transformed_depth

    @property
    def xyz_map(self) -> np.ndarray | None:
        """The point cloud as a 3D map (image format) from the color camera's perspective."""
        return self._frame.transformed_depth_point_cloud
    
    @property
    def point_cloud(self) -> o3d.geometry.PointCloud | None:
        """The generated Open3D PointCloud object."""
        # This logic is no longer duplicated; we delegate it to the frame object.
        return self._frame.generate_o3d_point_cloud()

    # --- Specialized Processing Methods (Retained from original CaptureManager) ---

    def convert_xy_to_xyz(self, xy: tuple[int, int]) -> np.ndarray:
        """
        Looks up the 3D coordinate (in mm) for a given 2D pixel coordinate.

        Args:
            xy (tuple): The (x, y) pixel coordinate.

        Returns:
            np.ndarray: The [X, Y, Z] coordinate in millimeters, or [0,0,0] if invalid.
        """
        x, y = xy
        point_cloud_map = self._frame.transformed_depth_point_cloud
        if point_cloud_map is not None and 0 <= y < point_cloud_map.shape[0] and 0 <= x < point_cloud_map.shape[1]:
            xyz_point = point_cloud_map[y, x]
            # Check for invalid points (often represented as NaNs or zeros)
            if np.any(np.isnan(xyz_point)) or np.all(xyz_point == 0):
                 return np.array([0.0, 0.0, 0.0], dtype=np.float32)
            return xyz_point
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def get_visual_depth_map(self) -> np.ndarray | None:
        """
        Converts the raw depth map (in mm) to a visualizable 8-bit RGB image.
        Clips depth values at 5 meters for better contrast.
        """
        depth_map = self.depth
        if depth_map is None:
            return None
        
        # Clip depth to a practical range (e.g., 0 to 5000mm) for visualization
        depth_clipped = np.clip(depth_map, 0, 5000)
        
        # Normalize the clipped depth map to the 0-255 range
        norm_depth = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply a colormap for better visual distinction
        return cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)


# --- Example Usage ---
if __name__ == "__main__":
    # Ensure you have a sample MKV file at this path
    video_file = Path("path/to/your/video.mkv")
    
    if not video_file.exists():
        print(f"ERROR: Video file not found at '{video_file}'. Please update the path.")
    else:
        FRAME_TO_PROCESS = 150  # Example frame index

        # 1. Use KinectMKV to open the file and access a frame
        print(f"Opening '{video_file}'...")
        with KinectMKV(video_file) as mkv:
            if FRAME_TO_PROCESS < len(mkv):
                # 2. Get the desired frame object
                target_frame = mkv[FRAME_TO_PROCESS]
                
                # 3. Pass the frame to our new processor
                processor = KinectFrameProcessor(target_frame)
                
                # 4. Use the processor's methods and properties
                
                # Get the color image
                color_image = processor.color
                if color_image is not None:
                    cv2.imshow("Color Image (Frame 150)", color_image)
                    print("Displayed color image.")
                
                # Get the visualized depth map
                visual_depth = processor.get_visual_depth_map()
                if visual_depth is not None:
                    cv2.imshow("Visual Depth Map (Frame 150)", visual_depth)
                    print("Displayed visual depth map.")

                # Get the 3D coordinate of a pixel (e.g., center of a 720p image)
                pixel_coord = (640, 360)
                xyz_coord = processor.convert_xy_to_xyz(pixel_coord)
                print(f"The 3D coordinate at pixel {pixel_coord} is {xyz_coord.tolist()} mm.")
                
                # Generate and visualize the point cloud
                pcd = processor.point_cloud
                if pcd:
                    print("Generated point cloud. Visualizing with Open3D...")
                    o3d.visualization.draw_geometries([pcd], window_name=f"Point Cloud - Frame {FRAME_TO_PROCESS}")
                
                print("\nPress any key in an image window to exit.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Error: Frame index {FRAME_TO_PROCESS} is out of bounds for video with {len(mkv)} frames.")
