import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Callable, Optional, Union
from pathlib import Path
from PIL import Image

from ..models.roi_tracked_data import ROITrackedObjects
from utils.package_utils import load_pyk4a

class BasePositionExtractor(ABC):
    """Abstract base class for position extraction."""
    
    def __init__(self, tracked_data: ROITrackedObjects):
        self.tracked_data = tracked_data
        self.sticker_names = self.tracked_data.sticker_names

    @abstractmethod
    def extract_positions(
        self,
        on_frame_processed: Optional[Callable[[int, Any, Dict], bool]] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        pass

    def _process_frame(self, point_cloud: np.ndarray, frame_centers: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Shared frame processing logic."""
        frame_results = {}
        monitoring_data = {}
        for name in self.sticker_names:
            px = frame_centers[f"{name}_x"].iloc[0]
            py = frame_centers[f"{name}_y"].iloc[0]
            
            x_mm, y_mm, z_mm = self._get_xyz_from_point_cloud(point_cloud, px, py)
            
            frame_results.update({f"{name}_x_mm": x_mm, f"{name}_y_mm": y_mm, f"{name}_z_mm": z_mm})
            monitoring_data[name] = {'px': px, 'py': py, 'x_mm': x_mm, 'y_mm': y_mm, 'z_mm': z_mm}

        return frame_results, monitoring_data

    def _get_xyz_from_point_cloud(self, point_cloud: np.ndarray, px: float, py: float) -> Tuple[float, float, float]:
        """Retrieves the (x, y, z) coordinates from a point cloud at a given pixel location."""
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

class MKVPositionExtractor(BasePositionExtractor):
    """Handles position extraction from MKV files."""
    
    def __init__(self, playback: 'PyK4APlayback', tracked_data: ROITrackedObjects):
        super().__init__(tracked_data)
        self.playback = playback

    def extract_positions(
        self,
        on_frame_processed: Optional[Callable[[int, 'Capture', Dict], bool]] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
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

class TIFFPositionExtractor(BasePositionExtractor):
    """Handles position extraction from TIFF files."""
    
    def __init__(self, tiff_folder: Union[str, Path], tracked_data: ROITrackedObjects):
        super().__init__(tracked_data)
        self.tiff_folder = Path(tiff_folder)
        self.tiff_files = sorted(list(self.tiff_folder.glob("*.tiff")))
        if not self.tiff_files:
            self.tiff_files = sorted(list(self.tiff_folder.glob("*.tif")))
        if not self.tiff_files:
            raise ValueError(f"No TIFF files found in {self.tiff_folder}")

    def extract_positions(
        self,
        on_frame_processed: Optional[Callable[[int, np.ndarray, Dict], bool]] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        results_data = []
        
        for frame_index, tiff_path in enumerate(self.tiff_files):
            print(f"Processing frame: {frame_index}", end='\r')
            
            frame_centers = self.tracked_data.get_coordinates_for_frame(frame_index)
            if frame_centers.empty:
                continue

            with Image.open(tiff_path) as img:
                depth_frame = np.array(img, dtype=np.float32)
                
                frame_results, monitoring_data = self._process_frame(
                    depth_frame,
                    frame_centers
                )
                
                frame_results['frame'] = frame_index
                results_data.append(frame_results)
                
                if on_frame_processed:
                    should_stop = on_frame_processed(frame_index, depth_frame, monitoring_data)
                    if should_stop:
                        print("\nProcessing stopped by callback.")
                        break

        processed_count = frame_index + 1
        print(f"\nCompleted processing {processed_count} frames.")
        return results_data, processed_count
