import numpy as np
import open3d as o3d
import colorsys
from typing import Dict, Callable, Tuple, Any

# Matplotlib is used for generating distinct colors for clusters
import matplotlib.pyplot as plt
import copy
import collections.abc


def deep_update(source: Dict, overrides: Dict) -> Dict:
    """
    Recursively updates a dictionary.
    Sub-dictionaries are updated instead of being replaced.
    """
    for key, value in overrides.items():
        if isinstance(value, collections.abc.Mapping) and key in source:
            source[key] = deep_update(source[key], value)
        else:
            source[key] = value
    return source


class ArmSegmentation:
    """
    A class providing stateful methods for arm segmentation using Open3D and NumPy.
    It processes 3D points and their corresponding colors to isolate the largest cluster,
    assumed to be the user's arm.

    The processing parameters are provided during initialization. An optional interactive
    mode allows for real-time parameter tuning and visualization at each step.

    Skin-colour filtering uses a cyclic hue range (``hsv_h_range``) so that
    wrap-around ranges spanning the 0°/360° boundary — e.g. ``[330, 30]`` for
    pink/red hues — are fully supported without any special-casing by the caller.
    """
    # --- Default parameters for all processing steps ---
    _DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
        'down_sampling': {
            'enabled': True,
            'leaf_size': 5.0
        },
        'box_filter': {
            'enabled': True,
            'min_z': -np.inf,
            'max_z': +np.inf,
        },
        'color_skin_filter': {
            'enabled': True,
            'hsv_h_range': [335, 25],         # [H_start, H_end] degrees, cyclic (0-360)
            'hsv_s_range': [0.1, 1.0],      # [S_low, S_high]
            'hsv_v_range': [0.0, 1.0],      # [V_low, V_high]
        },
        'region_growing': {
            'dbscan_eps': 18.0,
            'min_cluster_size': 50
        }
    }

    # --- Slider configuration: min, max, type ('int' or 'float'), and optional
    #     per-component labels for list parameters.
    #
    #     Ranges are derived from the physical scale of the data:
    #       H  : 0–360°  (full hue circle; skin ≈ 0–50°)
    #       S,V: 0–1.0   (normalised saturation / brightness)
    #       dbscan_eps      : 1–100 mm  (point-cloud units; arm spans ~100 mm)
    #       min_cluster_size: 10–2000   (points after voxel downsampling)
    #       leaf_size       : 0.5–20 mm (voxel size)
    # ---
    _SLIDER_CONFIGS: Dict[str, Dict] = {
        'down_sampling': {
            'leaf_size': {'min': 0.5, 'max': 20.0, 'type': 'float'},
        },
        'color_skin_filter': {
            'hsv_h_range': {'is_hue_circle': True},
            'hsv_s_range': {
                'is_range': True,
                'cfgs': [
                    {'min': 0.0, 'max': 1.0, 'type': 'float', 'label': 'S low'},
                    {'min': 0.0, 'max': 1.0, 'type': 'float', 'label': 'S high'},
                ],
            },
            'hsv_v_range': {
                'is_range': True,
                'cfgs': [
                    {'min': 0.0, 'max': 1.0, 'type': 'float', 'label': 'V low'},
                    {'min': 0.0, 'max': 1.0, 'type': 'float', 'label': 'V high'},
                ],
            },
        },
        'region_growing': {
            'dbscan_eps':       {'min': 1.0,  'max': 100.0, 'type': 'float'},
            'min_cluster_size': {'min': 10,   'max': 2000,  'type': 'int'},
        },
    }

    def __init__(self, params: Dict = None, interactive: bool = True):
        """
        Initializes the ArmSegmentation instance.

        Args:
            params (Dict): A dictionary containing parameters to override the defaults.
            interactive (bool): If True, enables a GUI for real-time parameter adjustment.
        """
        # Start with a deep copy of the default parameters
        self.params = copy.deepcopy(self._DEFAULT_PARAMS)
        # Warn callers that pass the old flat-bound keys
        if params:
            csf = params.get('color_skin_filter', {})
            if 'hsv_lower_bound' in csf or 'hsv_upper_bound' in csf:
                import warnings
                warnings.warn(
                    "ArmSegmentation: 'hsv_lower_bound' and 'hsv_upper_bound' were removed. "
                    "Use 'hsv_h_range', 'hsv_s_range', and 'hsv_v_range' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            deep_update(self.params, params)

        self.interactive = interactive
        self._view = None
        # Store original colors for clustering visualization
        self._original_colors = None
        # Set to True if the operator clicked "Process" at least once during an
        # interactive session; remains False if the window was closed without
        # applying any changes (so the caller can skip re-saving unchanged outputs).
        self.was_modified: bool = False

    def preprocess(self,
                   pcd: o3d.geometry.PointCloud,
                   box_corners: np.ndarray,
                   show: bool = False) -> o3d.geometry.PointCloud:
        """
        Applies a series of filters to the input point cloud data.

        Args:
            pcd (o3d.geometry.PointCloud): The input point cloud with points and colors.
            box_corners (np.ndarray): A (2, 2) array [[min_x, min_y], [max_x, max_y]] for cropping.
            show (bool): If True (and not in interactive mode), visualizes the point cloud at each step.

        Returns:
            o3d.geometry.PointCloud: The processed point cloud.
        """
        if show and not self.interactive: self._display_pointcloud(pcd, "1. Input Point Cloud")

        # --- 1. Downsampling ---
        if self.params['down_sampling']['enabled']:
            pcd = self._display_pointcloud(
                pcd, "Step 1: Voxel Downsampling",
                params_key='down_sampling',
                processing_func=lambda p, pa: p.voxel_down_sample(voxel_size=pa['leaf_size'])
            )
            print(f"Downsampled cloud size: {len(pcd.points)}")
            if show and not self.interactive: self._display_pointcloud(pcd, "2. After Downsampling")

        # --- 2. Box Filter ---
        if self.params['box_filter']['enabled']:
            min_b = np.array([box_corners[0, 0], box_corners[0, 1], self.params['box_filter']['min_z']])
            max_b = np.array([box_corners[1, 0], box_corners[1, 1], self.params['box_filter']['max_z']])
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_b, max_b)
            pcd = pcd.crop(bbox)
            print(f"Box-filtered cloud size: {len(pcd.points)}")
            if show and not self.interactive: self._display_pointcloud(pcd, "3. After Box Filter")

        # --- 3. Skin Color Filter ---
        if self.params['color_skin_filter']['enabled'] and len(pcd.points) > 0:
            pcd = self._display_pointcloud(
                pcd, "Step 2: Skin Color Filter (HSV)",
                params_key='color_skin_filter',
                processing_func=self._apply_skin_color_filter
            )
            print(f"Color-filtered cloud size: {len(pcd.points)}")
            if show and not self.interactive: self._display_pointcloud(pcd, "4. After Skin Color Filter")

        return pcd

    def extract_arm(self,
                    pcd: o3d.geometry.PointCloud,
                    show: bool = False) -> o3d.geometry.PointCloud:
        """
        Extracts the largest cluster (assumed to be the arm) via DBSCAN clustering.

        Args:
            pcd (o3d.geometry.PointCloud): The preprocessed point cloud.
            show (bool): If True (and not in interactive mode), visualizes the clustering results.

        Returns:
            o3d.geometry.PointCloud: A point cloud containing only the largest cluster.
        """
        if len(pcd.points) < self.params['region_growing']['min_cluster_size']:
            print("WARNING: Not enough points to process for arm extraction.")
            return o3d.geometry.PointCloud()

        self._original_colors = np.asarray(pcd.colors)

        arm_pcd = self._display_pointcloud(
            pcd, "Step 3: DBSCAN Clustering",
            params_key='region_growing',
            processing_func=self._apply_clustering,
            is_cluster_step=True
        )

        if arm_pcd and len(arm_pcd.points) > 0:
            print(f"Extracted arm with {len(arm_pcd.points)} points.")
            if show and not self.interactive:
                self._display_pointcloud(arm_pcd, "6. Segmented Arm")

        return arm_pcd

    def _apply_clustering(self, pcd: o3d.geometry.PointCloud, params: Dict) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """Helper function to perform DBSCAN and extract the largest cluster."""
        labels = np.array(pcd.cluster_dbscan(
            eps=params['dbscan_eps'],
            min_points=int(params['min_cluster_size']),
            print_progress=False
        ))

        pcd_all_clusters = o3d.geometry.PointCloud(pcd)
        # Visualization for all clusters
        max_label = labels.max()
        if max_label >= 0:
            # Use a perceptually uniform colormap
            colors = plt.get_cmap("viridis")(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0  # noise points are black
            pcd_all_clusters.colors = o3d.utility.Vector3dVector(colors[:, :3])

        unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(counts) == 0:
            print("No clusters found.")
            return pcd_all_clusters, o3d.geometry.PointCloud()

        largest_cluster_label = unique_labels[counts.argmax()]
        mask = (labels == largest_cluster_label)

        arm_pcd = o3d.geometry.PointCloud()
        arm_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[mask])
        # Restore original colors for the final output
        if self._original_colors is not None and self._original_colors.shape[0] == np.asarray(pcd.points).shape[0]:
            arm_pcd.colors = o3d.utility.Vector3dVector(self._original_colors[mask])

        return pcd_all_clusters, arm_pcd

    def _apply_skin_color_filter(self, pcd: o3d.geometry.PointCloud, params: Dict) -> o3d.geometry.PointCloud:
        """Applies an HSV-based color filter to isolate skin tones.

        Reads hsv_h_range, hsv_s_range, and hsv_v_range from params.
        Hue is handled cyclically so wrap-around ranges (e.g. [330, 30]) work correctly.
        """
        points_color_rgb = np.asarray(pcd.colors)
        if points_color_rgb.shape[0] == 0:
            return o3d.geometry.PointCloud()

        hsv = np.array([colorsys.rgb_to_hsv(c[0], c[1], c[2]) for c in points_color_rgb])
        hsv[:, 0] *= 360  # scale hue to 0-360 degrees

        h_start, h_end = params['hsv_h_range']
        s_lo,   s_hi   = params['hsv_s_range']
        v_lo,   v_hi   = params['hsv_v_range']

        mask = self._hue_in_range(hsv[:, 0], h_start, h_end) & \
               (hsv[:, 1] >= s_lo) & (hsv[:, 1] <= s_hi) & \
               (hsv[:, 2] >= v_lo) & (hsv[:, 2] <= v_hi)

        return pcd.select_by_index(np.where(mask)[0])

    # ------------------------------------------------------------------
    # Processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hue_in_range(H: np.ndarray, h_start: float, h_end: float) -> np.ndarray:
        """Returns a boolean mask: True where H (degrees, 0–360) falls within the
        clockwise arc from h_start to h_end.  Handles wrap-around correctly, so
        _hue_in_range(H, 330, 30) selects red hues that span the 0°/360° boundary."""
        if h_start <= h_end:
            return (H >= h_start) & (H <= h_end)
        else:  # wrap-around (e.g. 330°–30° spans the red/pink end)
            return (H >= h_start) | (H <= h_end)

    # ------------------------------------------------------------------
    # Display / interactive loop
    # ------------------------------------------------------------------

    def _display_pointcloud(self,
                            pcd_input: o3d.geometry.PointCloud,
                            window_name: str,
                            params_key: str = None,
                            processing_func: Callable = None,
                            is_cluster_step: bool = False) -> o3d.geometry.PointCloud:
        """
        Displays or processes a point cloud.

        - Non-interactive path: runs processing_func silently (no window).
        - Interactive path: delegates to arm_segmentation_view via lazy import
          so that GUI/OpenGL dependencies are never loaded in batch mode.
        """
        # --- Non-Interactive (or simple view) Path ---
        if not self.interactive or params_key is None or processing_func is None:
            # Batch path: if a processing function was provided, run it silently
            # using the saved params. No window, no blocking. Mirror the
            # interactive path's cluster-step unpacking so callers get the
            # selected cluster (not the all-clusters preview).
            if processing_func is not None and params_key is not None:
                result = processing_func(pcd_input, self.params[params_key])
                if is_cluster_step:
                    _all_clusters, arm_pcd = result
                    return arm_pcd
                return result
            # Display-only call — caller has already gated on `show`.
            if len(pcd_input.points) > 0:
                o3d.visualization.draw_geometries([pcd_input], window_name=window_name)
            else:
                print(f"Skipping visualization for '{window_name}': No points to show.")
            return pcd_input

        # --- Interactive Path — lazy import avoids loading GUI deps in batch mode ---
        from preprocessing.forearm_extraction import arm_segmentation_view as _view
        return _view.display_pointcloud_interactive(
            segmenter=self,
            pcd_input=pcd_input,
            window_name=window_name,
            params_key=params_key,
            processing_func=processing_func,
            is_cluster_step=is_cluster_step,
        )


# Example usage:
if __name__ == '__main__':
    # Create a dummy point cloud for demonstration
    dummy_points = np.random.rand(50000, 3) * 100
    dummy_points[:, 2] *= 0.2 # Make it flatter
    # Add a denser "arm" cluster with skin-like color
    arm_points = np.random.rand(5000, 3) * 20 + np.array([40, 40, 5])
    skin_color_rgb = np.array([234, 192, 183]) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([dummy_points, arm_points]))

    # Assign random colors and skin color to the arm
    colors = np.random.rand(50000, 3)
    arm_colors = np.tile(skin_color_rgb, (5000, 1)) + np.random.randn(5000, 3) * 0.05
    pcd.colors = o3d.utility.Vector3dVector(np.vstack([colors, arm_colors]))

    # --- Define parameters to OVERRIDE the defaults ---
    # You no longer need to define every single parameter.
    # For example, we can omit the 'box_filter' if the default is acceptable.
    user_defined_params = {
        'down_sampling': {
            'leaf_size': 2.5  # Override default leaf size
        },
        'color_skin_filter': {
            # Widen the hue range slightly (H 0°–35°, full S/V range)
            'hsv_h_range': [0, 35],
        },
        'region_growing': { # Changed from 'clustering' to match internal key
            'dbscan_eps': 6.0,
            'min_cluster_size': 50
        }
    }

    # Bounding box for filtering
    box_corners = np.array([[0, 0], [100, 100]])

    # --- Run in INTERACTIVE mode ---
    print("--- Starting INTERACTIVE segmentation ---")

    # The class now handles merging user params with defaults internally
    segmenter_interactive = ArmSegmentation(params=user_defined_params, interactive=True)

    # 1. Preprocessing
    preprocessed_pcd = segmenter_interactive.preprocess(pcd, box_corners)

    # 2. Arm Extraction
    if len(preprocessed_pcd.points) > 0:
        arm_pcd = segmenter_interactive.extract_arm(preprocessed_pcd)
        if arm_pcd and len(arm_pcd.points) > 0:
            print("\n✅ Interactive segmentation complete. Final arm point cloud:")
            segmenter_interactive._display_pointcloud(arm_pcd, "Final Result from Interactive Mode")
        else:
            print("\n❌ Interactive segmentation did not yield a result.")
    else:
        print("\n❌ Preprocessing removed all points. Cannot extract arm.")