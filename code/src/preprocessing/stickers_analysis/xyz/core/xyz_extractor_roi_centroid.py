# modified file: xyz_extractors.py
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

# Assuming these are the data structures passed in.
from ...roi.models.roi_tracked_data import ROITrackedObjects
from .xyz_extractor_interface import XYZExtractorInterface

class ROICentroidPointCloudExtractor(XYZExtractorInterface):
    """
    Extracts a 3D coordinate from a point cloud using the 2D centroid
    of a tracked ROI's bounding box.
    """

    def __init__(self):
        """Initializes the extractor."""
        pass

    def extract(self, tracked_obj_row: pd.Series, point_cloud: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extracts 3D coordinates from the center of an ROI's bounding box for a single frame.
        """
        # Calculate the center of the ROI in pixel coordinates
        origin = np.array([tracked_obj_row["roi_x"], tracked_obj_row["roi_y"]])
        size = np.array([tracked_obj_row["roi_width"], tracked_obj_row["roi_height"]])
        center = origin + size / 2.0
        
        px, py = int(center[0]), int(center[1])

        # Get the 3D coordinates from the point cloud at the calculated pixel
        x_mm, y_mm, z_mm = self.get_xyz_from_point_cloud(point_cloud, px, py)
        
        coords_3d = {"x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm}
        monitor_data = {"px": px, "py": py}

        return coords_3d, monitor_data
    
    def get_empty_result(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Returns a structured dictionary with NaN values for a failed extraction.
        """
        coords_3d = {"x_mm": np.nan, "y_mm": np.nan, "z_mm": np.nan}
        monitor_data = {"px": np.nan, "py": np.nan}
        return coords_3d, monitor_data
    
    @staticmethod
    def get_xyz_from_point_cloud(point_cloud: np.ndarray, px: float, py: float) -> Tuple[float, float, float]:
        """Retrieves the (x, y, z) coordinates from a point cloud at a given pixel location."""
        if point_cloud is None or np.isnan(px) or np.isnan(py):
            return (np.nan, np.nan, np.nan)
        height, width, _ = point_cloud.shape
        ix, iy = int(round(px)), int(round(py))
        if 0 <= iy < height and 0 <= ix < width:
            coords = point_cloud[iy, ix]
            # A value of all zeros often indicates no data from the depth sensor
            if np.all(coords == 0):
                return (np.nan, np.nan, np.nan)
            return float(coords[0]), float(coords[1]), float(coords[2])
        return (np.nan, np.nan, np.nan)

    @classmethod
    def can_process(cls, tracked_data: Any) -> bool:
        """
        Checks if the provided data is of the correct type (ROITrackedObjects)
        and has the necessary DataFrame columns for processing.
        """
        if not isinstance(tracked_data, ROITrackedObjects):
            return False
        
        first_object_name = next(iter(tracked_data), None)
        if first_object_name:
            # Check for columns required by the 'extract' method
            required_cols = {"roi_x", "roi_y", "roi_width", "roi_height", "status"}
            df_cols = set(tracked_data[first_object_name].columns)
            if not required_cols.issubset(df_cols):
                return False
        
        return True
    
    @classmethod
    def should_process_row(cls, tracked_obj_row: pd.Series) -> bool:
        """Determines if a row should be processed based on its 'status' field."""
        return tracked_obj_row["status"] != "Failed" and tracked_obj_row["status"] != "Black Frame"