import numpy as np
import open3d as o3d
import colorsys
from typing import Tuple, Dict

class ArmSegmentation:
    """
    A class providing stateless methods for arm segmentation using Open3D and NumPy.
    It processes 3D points and their corresponding colors to isolate the largest cluster,
    assumed to be the user's arm.
    """

    def preprocess(self,
                   pcd: o3d.geometry.PointCloud,
                   params: Dict,
                   box_corners: np.ndarray,
                   show: bool = False) -> o3d.geometry.PointCloud:
        """
        Applies a series of filters to the input point cloud data.

        Args:
            points_xyz (np.ndarray): An (N, 3) NumPy array of 3D point coordinates.
            points_rgb (np.ndarray): An (N, 3) NumPy array of RGB colors, normalized to [0, 1].
            params (Dict): A dictionary containing processing parameters.
            box_corners (np.ndarray): A (2, 2) NumPy array [[min_x, min_y], [max_x, max_y]]
                                      defining the cropping box.
            show (bool): If True, visualizes the point cloud at each step.

        Returns:
            A tuple (np.ndarray, np.ndarray) containing the processed points and colors.
        """
        if show: self._visualize_cloud(pcd, "1. Input Point Cloud")

        if params['down_sampling']['enabled']:
            leaf_size = params['down_sampling']['leaf_size']
            pcd = pcd.voxel_down_sample(voxel_size=leaf_size)
            print(f"Downsampled cloud size: {len(pcd.points)}")
            if show: self._visualize_cloud(pcd, "2. After Downsampling")
        
        if params['box_filter']['enabled']:
            min_bound = np.array([box_corners[0, 0], box_corners[0, 1], -np.inf])
            max_bound = np.array([box_corners[1, 0], box_corners[1, 1], np.inf])
            bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
            pcd = pcd.crop(bounding_box)
            print(f"Box-filtered cloud size: {len(pcd.points)}")
            if show: self._visualize_cloud(pcd, "3. After Box Filter")

        if params['color_skin_filter']['enabled'] and len(pcd.points) > 0:
            pcd = self._apply_skin_color_filter(pcd, params['color_skin_filter'])
            print(f"Color-filtered cloud size: {len(pcd.points)}")
            if show: self._visualize_cloud(pcd, "4. After Skin Color Filter")
            
        return pcd

    def extract_arm(self,
                    pcd: o3d.geometry.PointCloud,
                    params: Dict,
                    show: bool = False) -> o3d.geometry.PointCloud:
        """
        Extracts the largest cluster (assumed to be the arm) via DBSCAN clustering.

        Args:
            processed_xyz (np.ndarray): The preprocessed (N, 3) point coordinates.
            processed_rgb (np.ndarray): The preprocessed (N, 3) point colors.
            params (Dict): A dictionary containing clustering parameters.
            show (bool): If True, visualizes the clustering results.

        Returns:
            A tuple (np.ndarray, np.ndarray) containing the points and colors of the
            largest cluster, or empty arrays if no cluster is found.
        """
        points_color = np.asarray(pcd.colors)
        
        if np.asarray(pcd.points).shape[0] < params['min_cluster_size']:
            print("WARNING: Not enough points to process for arm extraction.")
            return np.array([]), np.array([])
        
        # Use DBSCAN for clustering, which does not require normals.
        # 'eps' is the distance to neighbors in a cluster.
        # 'min_points' is the minimum number of points to form a cluster.
        labels = np.array(pcd.cluster_dbscan(
            eps=params.get('dbscan_eps', 10),
            min_points=params['min_cluster_size'],
            print_progress=False
        ))

        # Find the largest cluster (ignoring noise points with label -1)
        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(counts) == 0:
            print("No clusters found.")
            return np.array([]), np.array([])
            
        print(f"Found {len(unique_labels)} clusters.")
        if show:
            # --- Start of modified section ---

            # 1. Define a specific color palette for clusters (normalized RGB).
            # You can easily add more colors or change the existing ones.
            palette = np.array([
                [255, 0, 0],   # Red
                [0, 255, 0],   # Green
                [0, 0, 255],   # Blue
                [255, 255, 0], # Yellow
                [0, 255, 255], # Cyan
                [255, 0, 255], # Magenta
                [255, 128, 0], # Orange
                [128, 0, 128], # Purple
            ]) / 255.0  # Normalize to [0, 1] range

            # 2. Define the color for noise points (label -1).
            noise_color = np.array([0.0, 0.0, 0.0]) # Black

            # 3. Create the final color array for each point using a fast, vectorized method.
            max_label = labels.max()
            
            # Handle the case where only noise points are found
            if max_label < 0:
                colors = np.tile(noise_color, (len(labels), 1))
            else:
                # Create a color map for actual cluster labels (0, 1, 2, ...).
                # We use the modulo operator to cycle through the palette if there are
                # more clusters than predefined colors.
                color_map = palette[np.arange(max_label + 1) % len(palette)]
            
                # Build a lookup table: first row for noise, subsequent rows for clusters.
                # We add 1 to the labels array to use it as indices for this table.
                # label -1 becomes index 0 (noise_color)
                # label 0 becomes index 1 (color_map[0])
                # label 1 becomes index 2 (color_map[1]), and so on.
                lookup_table = np.vstack((noise_color, color_map))
                colors = lookup_table[labels + 1]

            # 4. Assign the calculated colors to the point cloud and visualize.
            pcd.colors = o3d.utility.Vector3dVector(colors)
            self._visualize_cloud(pcd, "5. All Clusters with Specific Colors")
            
        largest_cluster_label = unique_labels[counts.argmax()]
        mask = (labels == largest_cluster_label)
        
        arm_pcd = o3d.geometry.PointCloud()
        arm_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[mask])
        arm_pcd.colors = o3d.utility.Vector3dVector(points_color[mask])
        
        print(f"Extracted arm with {np.asarray(arm_pcd.points).shape[0]} points.")
        if show:
            self._visualize_cloud(arm_pcd, "6. Segmented Arm")

        return arm_pcd
    
    def _apply_skin_color_filter(self,
                                 pcd: o3d.geometry.PointCloud,
                                 params: dict) -> o3d.geometry.PointCloud:
        """Applies an HSV-based color filter to isolate skin tones."""
        # The input points_rgb is already a normalized NumPy array

        points_color_rgb = np.asarray(pcd.colors)

        hsv = np.array([colorsys.rgb_to_hsv(c[0], c[1], c[2]) for c in points_color_rgb])
        
        # HSV values in colorsys are [0, 1]. Scale Hue to [0, 360] for parameter consistency.
        hsv[:, 0] *= 360
        
        lb, ub = params['hsv_lower_bound'], params['hsv_upper_bound']
        # Create a mask based on HSV thresholds
        mask = (hsv[:, 0] >= lb[0]) & (hsv[:, 0] <= ub[0]) & \
               (hsv[:, 1] >= lb[1]) & (hsv[:, 1] <= ub[1]) & \
               (hsv[:, 2] >= lb[2]) & (hsv[:, 2] <= ub[2])

        arm_pcd = o3d.geometry.PointCloud()
        arm_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[mask])
        arm_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[mask])

        return arm_pcd

    def _visualize_cloud(self, 
                                pcd: o3d.geometry.PointCloud, 
                                window_name: str,
                                point_size: float = 3.0,
                                background_color: tuple = (0.0, 0.0, 0.0)):
        """
        Visualizes an Open3D PointCloud object with custom render options.

        Args:
            pcd (o3d.geometry.PointCloud): The Open3D point cloud to visualize.
            window_name (str): The name of the visualization window.
            point_size (float): The size of the points in the renderer. Defaults to 3.0.
            background_color (tuple): A tuple of (R, G, B) values for the background, 
                                      where each value is between 0 and 1. Defaults to black.
        """
        # 1. Create a Visualizer object
        vis = o3d.visualization.Visualizer()
        
        # 2. Create a window with the specified name
        vis.create_window(window_name=window_name)
        
        # 3. Add the point cloud to the visualizer
        vis.add_geometry(pcd)
        
        # 4. Get the render option and modify it
        render_option = vis.get_render_option()
        render_option.point_size = point_size
        render_option.background_color = np.asarray(background_color)
        
        # 5. Run the visualizer
        vis.run()
        
        # 6. Destroy the window when it's closed
        vis.destroy_window()