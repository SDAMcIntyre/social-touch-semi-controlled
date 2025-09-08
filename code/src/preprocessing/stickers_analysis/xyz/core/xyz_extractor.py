import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Callable, Optional

# Assuming these are the data structures passed in.
from ...roi.models.roi_tracked_data import ROITrackedObject
from pyk4a import PyK4APlayback

class Sticker3DPositionExtractor:
    """
    Processes video frames and 2D coordinates to extract 3D positions.
    This version supports a callback function for frame-by-frame monitoring.
    """

    def __init__(self, playback: PyK4APlayback, tracked_data: ROITrackedObject):
        self.playback = playback
        self.tracked_data = tracked_data
        self.sticker_names = list(tracked_data.keys())
    
    @staticmethod
    def get_xyz(roi: pd.Series, point_cloud: np.ndarray) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Extracts 3D coordinates and prepares monitoring data for a single frame.
        """
        # Calculate the center of the ROI in pixel coordinates
        origin = np.array([roi["roi_x"], roi["roi_y"]])
        size = np.array([roi["roi_width"], roi["roi_height"]])
        center = origin + size / 2.0
        
        px, py = int(center[0]), int(center[1])

        # Get the 3D coordinates from the point cloud at the calculated pixel
        x_mm, y_mm, z_mm = Sticker3DPositionExtractor._get_xyz_from_point_cloud(point_cloud, px, py)

        # Structure the output to match the type hint
        coords_3d = {"x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm}
        monitor_data = {"px": px, "py": py}

        return coords_3d, monitor_data
    
    @staticmethod
    def get_empty_xyz():
        x_mm = np.nan
        y_mm = np.nan
        z_mm = np.nan
        
        px = np.nan
        py = np.nan
        # Structure the output to match the type hint
        coords_3d = {"x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm}
        monitor_data = {"px": px, "py": py}

        return coords_3d, monitor_data

    def extract_positions(
        self,
        on_frame_processed: Optional[Callable[[int, 'Capture', Dict], bool]] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Iterates through video playback, extracts 3D positions, and calls a
        callback on each frame if provided.

        Args:
            on_frame_processed (Optional[Callable]): A function to call after
                processing each frame. It should accept (frame_index, capture,
                monitoring_data) and return True to stop processing.

        Returns:
            A tuple containing the results list and the number of processed frames.
        """
        results_data = []
        frame_index = 0
        
        for frame_index, capture in enumerate(self.playback):
            print(f"Processing frame: {frame_index}", end='\r')

            frame_centers = self.tracked_data.get_coordinates_for_frame(frame_index)
            if frame_centers.empty:
                continue

            frame_results, monitoring_data = self._process_frame(
                capture.transformed_depth_point_cloud,
                frame_centers
            )
            
            frame_results['frame'] = frame_index
            results_data.append(frame_results)
            
            if on_frame_processed:
                should_stop = on_frame_processed(frame_index, capture, monitoring_data)
                if should_stop:
                    print("\nProcessing stopped by callback.")
                    break
        
        processed_count = frame_index + 1
        print(f"\nCompleted processing {processed_count} frames.")
        return results_data, processed_count
    
    def _process_frame(self, point_cloud: np.ndarray, frame_centers: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Extracts 3D coordinates and prepares monitoring data for a single frame.
        """
        frame_results = {}
        monitoring_data = {}
        for name in self.sticker_names:
            px = frame_centers[f"{name}_x"].iloc[0]
            py = frame_centers[f"{name}_y"].iloc[0]
            
            x_mm, y_mm, z_mm = self._get_xyz_from_point_cloud(point_cloud, px, py)
            
            frame_results.update({f"{name}_x_mm": x_mm, f"{name}_y_mm": y_mm, f"{name}_z_mm": z_mm})
            monitoring_data[name] = {'px': px, 'py': py, 'x_mm': x_mm, 'y_mm': y_mm, 'z_mm': z_mm}

        return frame_results, monitoring_data   
     
    @staticmethod
    def _get_xyz_from_point_cloud(point_cloud: np.ndarray, px: float, py: float) -> Tuple[float, float, float]:
        """Retrieves the (x, y, z) coordinates from a point cloud at a given pixel location."""
        # This method remains unchanged from the previous version.
        if point_cloud is None or np.isnan(px) or np.isnan(py):
            return (np.nan, np.nan, np.nan)
        height, width, _ = point_cloud.shape
        ix, iy = int(round(px)), int(round(py))
        if 0 <= iy < height and 0 <= ix < width:
            coords = point_cloud[iy, ix]
            if np.all(coords == 0):
                return (np.nan, np.nan, np.nan)
            return float(coords[0]), float(coords[1]), float(coords[2])
        return (np.nan, np.nan, np.nan)
