import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
from typing import Optional, Dict, Union, Tuple

from ..gui.debug_visualiser import DebugVisualizer

class ObjectsInteractionProcessor:
    """
    Processes the interaction between a provided set of vertices (current state) 
    and a specific reference geometry (PointCloud or Mesh).
    
    Refactored:
    - No internal dictionary handling.
    - No velocity/proprioceptive tracking.
    - Operates on pre-transformed vertices.
    """
    def __init__(self,
                 reference_geometry: Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh],
                 *,
                 use_debug: bool = False
        ):
        """
        Initializes the processor with a single reference geometry.
        
        Args:
            reference_geometry: The static environment object to check collisions against.
            use_debug (bool): Enable debug visualizer.
        """
        self.ref_pcd: Optional[o3d.geometry.PointCloud] = None
        self.ref_points: Optional[np.ndarray] = None
        self.ref_normals: Optional[np.ndarray] = None
        self.ref_colors: Optional[np.ndarray] = None
        self.kdtree: Optional[KDTree] = None
        self.point_area: Optional[float] = None
        
        self._debug = use_debug
        
        # Initialize with the provided reference
        self.update_reference(reference_geometry)
        self.visualizer = DebugVisualizer(self.ref_pcd) if self._debug else None

    def update_reference(self, geometry: Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]) -> None:
        """
        Updates the reference geometry and rebuilds the KDTree.
        
        Args:
            geometry: The new reference geometry.
        """
        if geometry is None:
            raise ValueError("Reference geometry cannot be None.")
            
        # Ensure we have a point cloud
        if isinstance(geometry, o3d.geometry.TriangleMesh):
            pcd = geometry.sample_points_poisson_disk(number_of_points=len(geometry.vertices))
        else:
            pcd = geometry

        if not pcd.has_points() or not pcd.has_normals():
            # Attempt to compute normals if missing
            if pcd.has_points():
                 pcd.estimate_normals()
            else:
                raise ValueError("Reference point cloud must contain points.")
        
        self.ref_pcd = pcd
        self.ref_points = np.asarray(pcd.points)
        self.ref_normals = np.asarray(pcd.normals)
        self.ref_colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        # Rebuild Spatial Index
        self.kdtree = KDTree(self.ref_points)
        
        # Estimate resolution for area calculation
        # We catch potential errors if point cloud is too sparse or single point
        if len(self.ref_points) > 1:
            ref_pcd_neighbor_dist = pcd.compute_nearest_neighbor_distance()
            point_cloud_resolution = np.mean(ref_pcd_neighbor_dist)
            self.point_area = point_cloud_resolution**2
        else:
            self.point_area = 0.0

    def _calculate_tactile_data(self, v_transformed: np.ndarray) -> tuple[dict, dict]:
        """Calculates contact-related metrics for a single frame using provided vertices."""
        # Query KDTree for nearest neighbors
        distances, min_indices = self.kdtree.query(v_transformed)
        
        # Calculate vectors from reference points to transformed vertices
        hand_arm_vectors = v_transformed - self.ref_points[min_indices]
        normals_at_closest_points = self.ref_normals[min_indices]
        
        # Dot product to check if inside/contacting (vector opposes normal)
        dot_products = np.einsum('ij,ij->i', hand_arm_vectors, normals_at_closest_points)
        inside_mask = dot_products <= 0

        # Filter by color if available (assuming specific color logic from original requirement)
        if self.ref_colors is not None and len(self.ref_colors) > 0:
            colors_at_closest_points = self.ref_colors[min_indices]
            # Keeping original threshold logic
            color_mask = np.all(colors_at_closest_points < 0.9, axis=1)
            inside_mask &= color_mask
            
        if not np.any(inside_mask):
            contact_quantities = {
                "contact_detected": 0, 
                "contact_depth": 0.0, 
                "contact_area": 0.0, 
                "contact_location_x": np.nan, 
                "contact_location_y": np.nan, 
                "contact_location_z": np.nan
            }
            contact_info = {"contact_points": np.array([]), "contact_normals": np.array([])}
            return contact_quantities, contact_info

        # Extract contact data
        unique_contact_indices = np.unique(min_indices[inside_mask])
        arm_intersect_unique = self.ref_points[unique_contact_indices]
        
        contact_quantities = {
            "contact_detected": 1,
            "contact_depth": np.mean(distances[inside_mask]),
            "contact_area": len(unique_contact_indices) * self.point_area,
            "contact_location_x": np.mean(arm_intersect_unique[:, 0]),
            "contact_location_y": np.mean(arm_intersect_unique[:, 1]),
            "contact_location_z": np.mean(arm_intersect_unique[:, 2]),
        }
        
        contact_info = {
            "contact_points": arm_intersect_unique, 
            "contact_normals": self.ref_normals[unique_contact_indices]
        }
        
        if self._debug and self.visualizer:
             # Logic implies showing visualizer when NOT in mask in original code? 
             # Or generally updating. Adjusted to be safe.
            if not np.any(inside_mask): 
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

    def process_single_frame(
            self, 
            current_mesh: o3d.geometry.TriangleMesh,
            _debug: bool = False
        ) -> dict:
        """
        Processes a single frame of interaction against the active point cloud using
        the provided mesh.

        Args:
            current_mesh (o3d.geometry.TriangleMesh): The mesh object for the current frame
                                                      already transformed to world space.
            _debug (bool): Enable debug visualization for this frame.

        Returns:
            A dictionary containing calculated metrics and data for visualization.
        """
        self._debug = _debug
        
        # Extract vertices from the mesh for calculation
        current_vertices = np.asarray(current_mesh.vertices)
        
        contact_data, contact_info = self._calculate_tactile_data(current_vertices)

        viz = {"contact_points": contact_info["contact_points"]}
        return contact_data, viz
    
    
    def empty_structure(self) -> dict:
        # If data is invalid (e.g., zero quaternion in source), return NaNs.
        contact_keys = [
            "contact_detected", "contact_depth", "contact_area", 
            "contact_location_x", "contact_location_y", "contact_location_z"
        ]
        contact_data = {key: np.nan for key in contact_keys}
        visualization = {"contact_points": np.array([])}
        return contact_data, visualization
    
