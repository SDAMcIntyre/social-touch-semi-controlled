# xyz_extractors.py
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional

# Explicitly register the 3D projection for Matplotlib
from mpl_toolkits.mplot3d import Axes3D

from ...roi.models.roi_tracked_data import ROITrackedObjects
from .xyz_extractor_interface import XYZExtractorInterface


class XYZExtractionVisualizer:
    """
    Encapsulates all visualization logic for XYZ extraction debugging.
    Separates presentation logic (Matplotlib/OpenCV) from business logic.
    """

    @staticmethod
    def plot_tracked_object_info(tracked_obj_row: pd.Series) -> None:
        """
        Generates a debug image showing ROI, ellipse, and text data for a tracked object row.
        Does not trigger display; creates a Matplotlib figure in the current state.
        """
        width, height = 1920, 1080
        image = np.full((height, width, 3), 255, dtype=np.uint8)

        # Geometric Primitive Configuration (BGR)
        color_text = (0, 0, 0)        # Black
        color_roi = (0, 0, 255)       # Red
        color_ellipse = (255, 0, 0)   # Blue

        # Text Configuration
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        line_spacing = 25
        x_text_start = 50
        y_text_start = 50

        cv2.putText(image, "Tracked Object Data:", (x_text_start, y_text_start), 
                    font, font_scale * 1.2, color_text, 2)

        current_y = y_text_start + 30
        for key, value in tracked_obj_row.items():
            if isinstance(value, float):
                text_str = f"{key}: {value:.2f}"
            else:
                text_str = f"{key}: {value}"
                
            cv2.putText(image, text_str, (x_text_start, current_y), 
                        font, font_scale, color_text, font_thickness)
            current_y += line_spacing

        # Draw ROI (Rectangle)
        rx = int(tracked_obj_row.get('roi_x', 0))
        ry = int(tracked_obj_row.get('roi_y', 0))
        rw = int(tracked_obj_row.get('roi_width', 0))
        rh = int(tracked_obj_row.get('roi_height', 0))

        pt1 = (rx, ry)
        pt2 = (rx + rw, ry + rh)

        cv2.rectangle(image, pt1, pt2, color_roi, 2)
        cv2.putText(image, "ROI", (rx, ry - 10), font, 0.5, color_roi, 1)

        # Draw Ellipse if parameters exist
        if 'ellipse_center_x' in tracked_obj_row:
            ex = int(tracked_obj_row['ellipse_center_x'])
            ey = int(tracked_obj_row['ellipse_center_y'])
            center = (ex, ey)

            ax_maj = int(tracked_obj_row.get('axes_major', 0))
            ax_min = int(tracked_obj_row.get('axes_minor', 0))
            axes = (ax_maj, ax_min)

            angle = tracked_obj_row.get('angle', 0.0)

            cv2.ellipse(image, center, axes, angle, 0, 360, color_ellipse, 2)
            cv2.circle(image, center, 2, (0, 255, 0), -1)

        # Plotting Result (Matplotlib)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 7))
        plt.imshow(image_rgb)
        plt.axis('off')
        plt.title("Tracking Visualization")
        plt.tight_layout()
        # Removed plt.show() to allow batch rendering

    @staticmethod
    def _get_3d_trace_from_2d_pixels(point_cloud: np.ndarray, pixels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Helper method to map a list of 2D pixel coordinates to 3D space via point cloud lookup.
        Inserts NaNs for invalid points to create discontinuous lines in Matplotlib.
        
        Args:
            point_cloud: (H, W, 3) XYZ array.
            pixels: (N, 2) array of [x, y] coordinates.
            
        Returns:
            Tuple of (X, Y, Z) arrays.
        """
        h, w, _ = point_cloud.shape
        X, Y, Z = [], [], []
        
        for px, py in pixels:
            ix, iy = int(round(px)), int(round(py))
            
            # Bounds check
            if 0 <= ix < w and 0 <= iy < h:
                val = point_cloud[iy, ix]
                # Check for zero (missing data) or NaN
                if np.all(val == 0) or np.any(np.isnan(val)):
                    X.append(np.nan); Y.append(np.nan); Z.append(np.nan)
                else:
                    X.append(val[0]); Y.append(val[1]); Z.append(val[2])
            else:
                X.append(np.nan); Y.append(np.nan); Z.append(np.nan)
                
        return np.array(X), np.array(Y), np.array(Z)

    @staticmethod
    def plot_point_cloud_extraction(point_cloud: np.ndarray, tracked_obj_row: pd.Series, xyz: Tuple[float, float, float], *, z_range: float = 50.0) -> None:
        """
        Visualizes the point cloud in interactive 3D space, highlighting the target centroid,
        ROI boundary, and Ellipse fit.

        Modification:
        - Camera View: Facing Upward (+Z direction) -> elev=-90.
        - Plots the ROI rectangle (Red) and Ellipse (Blue) projected onto the 3D surface.
        - Filters points based on Depth ROI (+/- z_range).
        - Decimation Strategy: Selects the closest 5000 points to the target.

        Args:
            point_cloud: (H, W, 3) numpy array of XYZ coordinates.
            tracked_obj_row: Pandas Series containing ROI/Ellipse parameters and pixel centroid.
            xyz: Tuple (x_mm, y_mm, z_mm) of the target point in 3D space.
            z_range: The +/- depth range in mm to visualize around the target Z.
        """
        if point_cloud is None:
            print("Debug Warning: Point Cloud is None, cannot plot.")
            return

        x_mm, y_mm, z_mm = xyz
        px = tracked_obj_row.get("center_x", np.nan)
        py = tracked_obj_row.get("center_y", np.nan)
        
        # 1. Vectorization
        # Reshape (H, W, 3) -> (N, 3) to treat as a purely geometric point set
        points_flat = point_cloud.reshape(-1, 3)
        
        # Extract columns for vectorized operations
        Z_flat = points_flat[:, 2]
        
        # 2. Filter Data (ROI & Validity)
        # Base validity: Filter out Z=0 (missing data) and NaNs
        valid_mask = (Z_flat != 0) & (~np.isnan(Z_flat))
        
        if not np.isnan(z_mm):
            # Depth ROI: Points within +/- z_range of the target
            depth_mask = (Z_flat >= (z_mm - z_range)) & (Z_flat <= (z_mm + z_range))
            final_mask = valid_mask & depth_mask
        else:
            final_mask = valid_mask

        # Apply mask to get candidate points
        filtered_points = points_flat[final_mask]

        # 3. Proximity Decimation (Top-K Closest in XY)
        target_count = 5000
        num_points = filtered_points.shape[0]
        
        if num_points > target_count and not np.isnan(x_mm) and not np.isnan(y_mm):
            # Calculate squared Euclidean distance in XY plane (faster than sqrt)
            diff_x = filtered_points[:, 0] - x_mm
            diff_y = filtered_points[:, 1] - y_mm
            dist_sq = (diff_x ** 2) + (diff_y ** 2)
            
            # O(n) selection of top K closest points
            partition_indices = np.argpartition(dist_sq, target_count)[:target_count]
            points_to_plot = filtered_points[partition_indices]
        
        elif num_points > target_count:
            # Random sampling if no target centroid to focus on
            indices = np.random.choice(num_points, size=target_count, replace=False)
            points_to_plot = filtered_points[indices]
            
        else:
            points_to_plot = filtered_points

        # Handle empty case
        if points_to_plot.shape[0] == 0:
            print(f"Debug Warning: No points found within Z range +/- {z_range}mm of target.")
            X_plot, Y_plot, Z_plot = np.array([]), np.array([]), np.array([])
        else:
            X_plot = points_to_plot[:, 0]
            Y_plot = points_to_plot[:, 1]
            Z_plot = points_to_plot[:, 2]

        # 4. 3D Plotting
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the scene context (Cloud)
        if len(X_plot) > 0:
            scatter = ax.scatter(X_plot, Y_plot, Z_plot, 
                                 c=Z_plot, cmap='viridis', s=3, alpha=0.5, label='Scene Cloud')
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1)
            cbar.set_label('Depth Z (mm)')

        # Plot the Target Point (Centroid)
        if not any(np.isnan([x_mm, y_mm, z_mm])):
            ax.scatter([x_mm], [y_mm], [z_mm], 
                       c='red', s=50, marker='X', linewidths=2, label='Target Centroid', depthshade=False)

        # 5. Plot ROI Rectangle (Projected on Surface)
        if all(k in tracked_obj_row for k in ['roi_x', 'roi_y', 'roi_width', 'roi_height']):
            rx, ry = int(tracked_obj_row['roi_x']), int(tracked_obj_row['roi_y'])
            rw, rh = int(tracked_obj_row['roi_width']), int(tracked_obj_row['roi_height'])
            
            # Define rectangle corners in pixel space
            roi_pixels = []
            # Top edge
            for x in range(rx, rx + rw): roi_pixels.append([x, ry])
            # Right edge
            for y in range(ry, ry + rh): roi_pixels.append([rx + rw, y])
            # Bottom edge
            for x in range(rx + rw, rx, -1): roi_pixels.append([x, ry + rh])
            # Left edge
            for y in range(ry + rh, ry, -1): roi_pixels.append([rx, y])
            
            roi_pixels = np.array(roi_pixels)
            # Assuming _get_3d_trace_from_2d_pixels is available in class context
            roi_X, roi_Y, roi_Z = XYZExtractionVisualizer._get_3d_trace_from_2d_pixels(point_cloud, roi_pixels)
            
            if len(roi_X) > 0:
                ax.plot(roi_X, roi_Y, roi_Z, c='red', linewidth=2, label='ROI Boundary')

        # 6. Plot Ellipse (Projected on Surface)
        if all(k in tracked_obj_row for k in ['ellipse_center_x', 'ellipse_center_y', 'axes_major', 'axes_minor', 'angle']):
            ex, ey = int(tracked_obj_row['ellipse_center_x']), int(tracked_obj_row['ellipse_center_y'])
            maj, min_ax = int(tracked_obj_row['axes_major'] / 2), int(tracked_obj_row['axes_minor'] / 2)
            angle = tracked_obj_row['angle']
            
            # Generate Ellipse Polygon (List of points)
            ellipse_pts = cv2.ellipse2Poly((ex, ey), (maj, min_ax), int(angle), 0, 360, 5)
            
            if ellipse_pts is not None and len(ellipse_pts) > 0:
                el_X, el_Y, el_Z = XYZExtractionVisualizer._get_3d_trace_from_2d_pixels(point_cloud, ellipse_pts)
                ax.plot(el_X, el_Y, el_Z, c='blue', linewidth=2, linestyle='--', label='Ellipse Fit')

        # Labels and Style
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        
        # Force Camera Centering
        if not np.isnan(x_mm) and not np.isnan(y_mm) and not np.isnan(z_mm):
            ax.set_xlim(x_mm - z_range, x_mm + z_range)
            ax.set_ylim(y_mm - z_range, y_mm + z_range)
            ax.set_zlim(z_mm - z_range, z_mm + z_range)
        
        ax.set_box_aspect([1, 1, 1]) 
        
        # --- MODIFICATION: Camera Facing Upward (+Z Direction) ---
        # elev=-90: Camera is at -Z looking towards +Z (Bottom-Up)
        # Change to elev=90 if you intended Top-Down view (Camera at +Z looking down)
        ax.view_init(elev=-90, azim=-90)

        title_text = (f"3D Point Cloud Extraction\n"
                      f"Pixel Target: ({px:.1f}, {py:.1f})\n"
                      f"Extracted World Coords: X={x_mm:.1f}, Y={y_mm:.1f}, Z={z_mm:.1f} mm\n"
                      f"View Volume: Target +/- {z_range}mm | Points Shown: {len(X_plot)}")
        ax.set_title(title_text)
        
        ax.legend()
        plt.tight_layout()
        plt.show() # Ensure plot is displayed

    @staticmethod
    def show_plots(block: bool = True) -> None:
        """
        Triggers the display of all active Matplotlib figures.
        Blocks execution until all windows are closed if block=True.
        """
        if plt.get_fignums():
            plt.show(block=block)

class CentroidPointCloudExtractor(XYZExtractorInterface):
    """
    Extracts a 3D coordinate from a point cloud using the 2D centroid
    of a tracked ROI's bounding box.
    """

    def __init__(self, debug: bool = False):
        """
        Initializes the extractor.
        
        Args:
            debug (bool): If True, enables debug mode (delegates to XYZExtractionVisualizer).
        """
        self.debug = debug

    def extract(self, tracked_obj_row: pd.Series, point_cloud: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extracts 3D coordinates from the center of an ROI's bounding box for a single frame.
        """
        # Data Extraction (Pure Logic)
        px = tracked_obj_row["center_x"]
        py = tracked_obj_row["center_y"]
        
        x_mm, y_mm, z_mm = self.get_xyz_from_point_cloud(point_cloud, px, py)
        
        coords_3d = {"x_mm": x_mm, "y_mm": y_mm, "z_mm": z_mm}
        monitor_data = {"px": px, "py": py}

        # Visualization Delegation
        if self.debug:
            XYZExtractionVisualizer.plot_tracked_object_info(tracked_obj_row)
            # Pass full row to allow ROI/Ellipse plotting
            XYZExtractionVisualizer.plot_point_cloud_extraction(
                point_cloud, 
                tracked_obj_row, 
                (x_mm, y_mm, z_mm)
            )
            # Blocking call triggered here, after both figures have been generated
            XYZExtractionVisualizer.show_plots(block=True)

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
        """
        Retrieves the (x, y, z) coordinates from a point cloud at a given pixel location.
        """
        if point_cloud is None or pd.isna(px) or pd.isna(py):
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
        """
        Determines if a row should be processed based on its 'status' field.
        """
        invalid_statuses = {"Failed", "Black Frame", "Ignored"}
        return tracked_obj_row["status"] not in invalid_statuses