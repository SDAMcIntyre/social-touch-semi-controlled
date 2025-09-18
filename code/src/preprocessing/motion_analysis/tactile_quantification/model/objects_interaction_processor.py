import numpy as np
import pandas as pd
import open3d as o3d
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation # <-- Essential for quaternion handling
from typing import List, Optional, Dict, Union

from ..gui.debug_visualiser import DebugVisualizer

class ObjectsInteractionProcessor:
    """
    Processes the interaction between a moving set of vertices and a static,
    selectable reference point cloud from a dictionary. This class contains
    only the core numerical logic.
    """
    def __init__(self,
                 references_pcd: Dict[int, o3d.geometry.PointCloud],
                 base_vertices: np.ndarray,
                 tracked_points_groups_indices: Optional[Union[List[int], List[List[int]]]] = None,
                 tracked_points_groups_labels: Optional[List[str]] = None,
                 *,
                 fps: int = 30,
                 use_debug: bool = True
        ):
        """
        Initializes the processor with a dictionary of reference point clouds.
        
        Args:
            references_pcd (Dict[int, o3d.geometry.PointCloud]): A dictionary
                mapping integer keys to reference point clouds.
            base_vertices (np.ndarray): The base vertices of the moving object.
            tracked_points_groups_indices (Optional[Union[List[int], List[List[int]]]]):
                Indices of vertices to track from `base_vertices`. This can be a
                **flat list** of integers for individual vertices, or a **nested list**
                where each sublist groups vertex indices to be **averaged** into a
                single, more stable tracked point.
            tracked_points_groups_labels (Optional[List[str]]): Labels for
                the tracked point groups.
            fps (int): Frames per second for velocity calculations.
        """
        if not references_pcd:
            raise ValueError("references_pcd dictionary cannot be empty.")
        
        for key, pcd in references_pcd.items():
            if not pcd.has_points() or not pcd.has_normals():
                raise ValueError(f"Point cloud with key {key} must contain both points and normals.")
        
        self.references_pcd = references_pcd
        
        # Initialize attributes that will be set by `set_current_pcd`
        self.current_pcd_key: Optional[int] = None
        self.ref_pcd: Optional[o3d.geometry.PointCloud] = None
        self.ref_points: Optional[np.ndarray] = None
        self.ref_normals: Optional[np.ndarray] = None
        self.ref_colors: Optional[np.ndarray] = None
        self.kdtree: Optional[KDTree] = None
        self.point_area: Optional[float] = None

        # Set the initial active point cloud (defaults to the first one)
        initial_key = next(iter(self.references_pcd))
        self.set_current_pcd(initial_key)
        
        if tracked_points_groups_indices and tracked_points_groups_labels:
            if len(tracked_points_groups_indices) != len(tracked_points_groups_labels):
                raise ValueError("tracked_points_groups_indices and tracked_points_groups_labels must have the same length.")
            self.tracked_points_groups_indices = tracked_points_groups_indices
            self.tracked_points_groups_labels = tracked_points_groups_labels
        else:
            self.tracked_points_groups_indices = []
            self.tracked_points_groups_labels = []

        self.base_vertices = base_vertices
        self._previous_tracked_points_pos: Dict[str, np.ndarray] = {}
        self.fps = fps
        self.dt = 1.0 / fps
        
        self._debug = use_debug

          # Or set based on some config
        if self._debug:
            self.visualizer = DebugVisualizer(self.ref_pcd)
        else:
            self.visualizer = None

    def set_current_pcd(self, key: int) -> None:
        """
        Sets the active reference point cloud for all subsequent processing.

        This method updates all internal references (points, normals, KDTree, etc.)
        to correspond to the point cloud associated with the given key.

        Args:
            key (int): The key of the point cloud to set as active.
        
        Raises:
            KeyError: If the provided key does not exist in the dictionary.
        """
        if not self.is_valid_pcd_key(key):
            raise KeyError(f"Key {key} not found in the provided point cloud dictionary.")
        
        self.current_pcd_key = key
        pcd = self.references_pcd[key]
        
        self.ref_pcd = pcd
        self.ref_points = np.asarray(pcd.points)
        self.ref_normals = np.asarray(pcd.normals)
        self.ref_colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        self.kdtree = KDTree(self.ref_points)
        
        ref_pcd_neighbor_dist = pcd.compute_nearest_neighbor_distance()
        point_cloud_resolution = np.mean(ref_pcd_neighbor_dist)
        self.point_area = point_cloud_resolution**2

    def is_valid_pcd_key(self, key: int) -> bool:
        """
        Checks if the given integer is a valid key in the point cloud dictionary.
        
        Args:
            key (int): The key to check.
            
        Returns:
            bool: True if the key exists, False otherwise.
        """
        return key in self.references_pcd

    @staticmethod
    def _calculate_rotation_from_planes(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        """Calculates the 3x3 rotation matrix to align the plane of source_points to target_points."""
        def create_basis(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
            v1 = p1 - p0
            v2 = p2 - p0
            x_axis = v1 / np.linalg.norm(v1)
            z_axis_unnorm = np.cross(x_axis, v2)
            if np.linalg.norm(z_axis_unnorm) < 1e-6:
                temp_vec = np.array([0, 1, 0]) if np.allclose(np.abs(x_axis), [1, 0, 0]) else np.array([1, 0, 0])
                z_axis_unnorm = np.cross(x_axis, temp_vec)
            z_axis = z_axis_unnorm / np.linalg.norm(z_axis_unnorm)
            y_axis = np.cross(z_axis, x_axis)
            return np.column_stack([x_axis, y_axis, z_axis])

        source_basis = create_basis(source_points[0], source_points[1], source_points[2])
        target_basis = create_basis(target_points[0], target_points[1], target_points[2])
        rotation_matrix = target_basis @ source_basis.T
        return rotation_matrix

    @staticmethod
    def _calculate_translation_for_origin(source_origin: np.ndarray, target_origin: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        """Calculates the translation vector to align a rotated source origin to a target origin."""
        rotated_source_origin = rotation_matrix @ source_origin
        translation_vector = target_origin - rotated_source_origin
        return translation_vector

    @staticmethod
    def calculate_prioritized_rigid_transformation(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
        """Calculates a 4x4 rigid transformation matrix with an explicit, prioritized alignment."""
        if source_points.shape != (3, 3) or target_points.shape != (3, 3):
            raise ValueError("Input point clouds must be of shape (3, 3)")
        rotation = ObjectsInteractionProcessor._calculate_rotation_from_planes(source_points, target_points)
        source_origin = source_points[0]
        target_origin = target_points[0]
        translation = ObjectsInteractionProcessor._calculate_translation_for_origin(source_origin, target_origin, rotation)
        transform_matrix = np.eye(4)
        transform_matrix[0:3, 0:3] = rotation
        transform_matrix[0:3, 3] = translation
        return transform_matrix
    
    def _transform_vertices(self, translation: np.ndarray, rotation_quat: np.ndarray) -> np.ndarray:
        """
        Applies rotation (from quaternion) and translation to the base vertices.
        
        Args:
            translation (np.ndarray): The (3,) translation vector.
            rotation_quat (np.ndarray): The (4,) quaternion [x, y, z, w].
        """
        rot = Rotation.from_quat(rotation_quat)
        transformed_vertices = rot.apply(self.base_vertices) + translation
        return transformed_vertices

    def _calculate_tactile_data(self, v_transformed: np.ndarray) -> tuple[dict, dict]:
        """Calculates contact-related metrics for a single frame."""
        # --- All the calculation logic remains exactly the same ---
        distances, min_indices = self.kdtree.query(v_transformed)
        hand_arm_vectors = v_transformed - self.ref_points[min_indices]
        normals_at_closest_points = self.ref_normals[min_indices]
        dot_products = np.einsum('ij,ij->i', hand_arm_vectors, normals_at_closest_points)
        inside_mask = dot_products <= 0

        if self.ref_colors is not None and len(self.ref_colors) > 0:
            colors_at_closest_points = self.ref_colors[min_indices]
            color_mask = np.all(colors_at_closest_points < 0.9, axis=1)
            inside_mask &= color_mask
            
        if not np.any(inside_mask):
            contact_quantities = {"contact_detected": 0, "contact_depth": 0.0, "contact_area": 0.0, "contact_points": [], "contact_location_x": np.nan, "contact_location_y": np.nan, "contact_location_z": np.nan}
            contact_info = {"contact_points": np.array([]), "contact_normals": np.array([])}
            return contact_quantities, contact_info

        unique_contact_indices = np.unique(min_indices[inside_mask])
        arm_intersect_unique = self.ref_points[unique_contact_indices]
        
        contact_quantities = {
            "contact_detected": 1,
            "contact_depth": np.mean(distances[inside_mask]),
            "contact_area": len(unique_contact_indices) * self.point_area,
            "contact_points": arm_intersect_unique,
            "contact_location_x": np.mean(arm_intersect_unique[:, 0]),
            "contact_location_y": np.mean(arm_intersect_unique[:, 1]),
            "contact_location_z": np.mean(arm_intersect_unique[:, 2]),
        }
        
        contact_info = {"contact_points": arm_intersect_unique, "contact_normals": self.ref_normals[unique_contact_indices]}
        
        if self._debug and self.visualizer and (not np.any(inside_mask)):
            self.visualizer.show(
                transformed_vertices=v_transformed,
                contact_points=contact_info["contact_points"],
                min_indices=min_indices,
                dot_products=dot_products,
                inside_mask=inside_mask,
                normals_at_closest_points=normals_at_closest_points,
                unique_contact_indices=unique_contact_indices
            )
        
        return contact_quantities, contact_info

    def _calculate_proprioceptive_data(
            self,
            v_transformed: np.ndarray,
            contact_info: dict) -> tuple[dict, dict]:
        """Calculates motion-related metrics for a single frame."""
        if not self.tracked_points_groups_labels:
            return {}, {}
        proprio_data, current_positions = {}, {}
        for label, indices in zip(self.tracked_points_groups_labels, self.tracked_points_groups_indices):
            position_xyz = np.mean(v_transformed[np.atleast_1d(indices), :], axis=0)
            current_positions[label] = position_xyz
            proprio_data[f"{label}_position_x"], proprio_data[f"{label}_position_y"], proprio_data[f"{label}_position_z"] = position_xyz
        primary_label = self.tracked_points_groups_labels[0]
        primary_position = current_positions[primary_label]
        primary_pos_prev = self._previous_tracked_points_pos.get(primary_label)
        velocity = np.zeros(3) if primary_pos_prev is None else (primary_position - primary_pos_prev) / self.dt
        vel_abs, vel_normal, vel_lateral, vel_longitudinal = np.linalg.norm(velocity), np.nan, np.nan, np.nan
        if contact_info["contact_points"].any():
            arm_normal = np.mean(contact_info["contact_normals"], axis=0)
            arm_normal /= np.linalg.norm(arm_normal)
            cam_y_axis = np.array([0, 1, 0])
            arm_longitudinal = cam_y_axis - np.dot(cam_y_axis, arm_normal) * arm_normal
            arm_longitudinal /= np.linalg.norm(arm_longitudinal)
            arm_lateral = np.cross(arm_longitudinal, arm_normal)
            vel_normal = np.dot(velocity, arm_normal)
            vel_lateral = np.dot(velocity, arm_lateral)
            vel_longitudinal = np.dot(velocity, arm_longitudinal)
        proprio_data.update({"velocity_absolute": vel_abs, "velocity_normal": vel_normal, "velocity_lateral": vel_lateral, "velocity_longitudinal": vel_longitudinal})
        return proprio_data, current_positions
    
    def process_single_frame(
            self, 
            translation: np.ndarray, 
            rotation_quat: np.ndarray,
            _debug: bool = False
        ) -> dict:
        """
        Processes a single frame of interaction against the active point cloud.

        Args:
            translation (np.ndarray): The (3,) translation vector for the current frame.
            rotation_quat (np.ndarray): The (4,) rotation quaternion [x, y, z, w]
                                        for the current frame.

        Returns:
            A dictionary containing calculated metrics and data for visualization.
        """
        self._debug = _debug

        # Check for zero transformation input. If so, return a dictionary of NaNs.
        if np.all(rotation_quat == 0) or np.all(translation == 0):
            contact_keys = [
                "contact_detected", "contact_depth", "contact_area", "contact_location_x",
                "contact_location_y", "contact_location_z"
            ]
            proprio_keys = []
            if self.tracked_points_groups_labels:
                for label in self.tracked_points_groups_labels:
                    proprio_keys.extend([
                        f"{label}_position_x", f"{label}_position_y", f"{label}_position_z"
                    ])
                proprio_keys.extend([
                    "velocity_absolute", "velocity_normal", "velocity_lateral", "velocity_longitudinal"
                ])
            
            all_metric_keys = contact_keys + proprio_keys
            nan_metrics = {key: np.nan for key in all_metric_keys}
            self._previous_tracked_points_pos = {}
            return {
                "metrics": nan_metrics,
                "visualization": {
                    "transformed_vertices": np.copy(self.base_vertices), 
                    "contact_points": np.array([])
                }
            }
            
        v_transformed = self._transform_vertices(translation, rotation_quat)
        contact_quantities, contact_info = self._calculate_tactile_data(v_transformed)
        proprio_quantities, current_tracked_points_pos = self._calculate_proprioceptive_data(
            v_transformed, 
            contact_info
        )
        if proprio_quantities:
            self._previous_tracked_points_pos = current_tracked_points_pos

        return {
            "metrics": {**contact_quantities, **proprio_quantities},
            "visualization": {
                "transformed_vertices": v_transformed,
                "contact_points": contact_info["contact_points"]
            }
        }