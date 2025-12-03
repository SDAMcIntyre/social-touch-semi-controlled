import numpy as np
import open3d as o3d
from typing import Optional, Dict, Union, Tuple, Any

class ObjectsInteractionProcessor:
    """
    Processes the interaction between a static 2.5D terrain mesh and a dynamic 3D object.
    
    Refactored Architecture for Non-Watertight Support:
    - Broad Phase: Axis-Aligned Bounding Box (AABB) cropping.
    - Narrow Phase: Signed Distance check using Open3D Tensor Raycasting.
      (allows interaction detection on open shells/non-manifold geometry).
    - Metric: Surface area accumulation of penetrating terrain triangles.
    """
    def __init__(self,
                 reference_geometry: Union[o3d.geometry.TriangleMesh, o3d.geometry.PointCloud],
                 *,
                 use_debug: bool = False
        ):
        """
        Initializes the processor.
        
        Args:
            reference_geometry: The static 2.5D terrain environment. 
                                MUST be a TriangleMesh for area calculations.
            use_debug (bool): Enable debug visualizer.
        """
        self.ref_mesh: Optional[o3d.geometry.TriangleMesh] = None
        self.ref_triangle_areas: Optional[np.ndarray] = None
        self.ref_vertices: Optional[np.ndarray] = None
        
        self._debug = use_debug
        
        # Initialize with the provided reference
        self.update_reference(reference_geometry)
        
        self.visualizer = None 

    def update_reference(self, geometry: Union[o3d.geometry.PointCloud, o3d.geometry.TriangleMesh]) -> None:
        """
        Updates the reference geometry (Terrain). Pre-calculates triangle areas for O(1) lookup.
        
        Args:
            geometry: The new reference geometry.
        """
        if geometry is None:
            raise ValueError("Reference geometry cannot be None.")
            
        # Architecture Requirement: Reference must be a Mesh to have "Surface Area"
        if isinstance(geometry, o3d.geometry.PointCloud):
            raise TypeError("Reference geometry must be o3d.geometry.TriangleMesh for 2.5D Area Estimation.")
        
        self.ref_mesh = geometry
        
        # Ensure normals for visualization and correct orientation checks
        if not self.ref_mesh.has_vertex_normals():
            self.ref_mesh.compute_vertex_normals()

        # Pre-compute Surface Areas
        # We calculate area of all triangles using cross product of edges.
        vertices = np.asarray(self.ref_mesh.vertices)
        triangles = np.asarray(self.ref_mesh.triangles)
        
        # Vectors for edges
        v0 = vertices[triangles[:, 0]]
        v1 = vertices[triangles[:, 1]]
        v2 = vertices[triangles[:, 2]]
        
        # Area = 0.5 * |(v1 - v0) x (v2 - v0)|
        cross_product = np.cross(v1 - v0, v2 - v0)
        self.ref_triangle_areas = 0.5 * np.linalg.norm(cross_product, axis=1)
        self.ref_vertices = vertices

    def _calculate_intersection_volume(self, input_mesh: o3d.geometry.TriangleMesh) -> tuple[dict, dict]:
        """
        Executes Broad Phase (Cropping) and Narrow Phase (Signed Distance Check).
        
        Args:
            input_mesh: The dynamic object mesh.
            
        Returns:
            Tuple containing contact metrics and visualization data.
        """
        # 1. Broad Phase: 2.5D Projection / AABB Crop
        # We perform an AABB crop on the Reference Terrain using the Object's bounding box.
        # This acts as a spatial hash to limit the number of vertices we check.
        aabb = input_mesh.get_axis_aligned_bounding_box()
        
        # Crop the terrain. Returns a NEW mesh containing only potentially intersecting triangles.
        cropped_terrain = self.ref_mesh.crop(aabb)
        
        if len(cropped_terrain.triangles) == 0:
            return self.empty_structure()

        # 2. Narrow Phase: Signed Distance Check (Watertight Independent)
        # Instead of Occupancy (Inside/Outside volume), we use Signed Distance.
        # Negative distance implies the point is behind the surface normal (penetration).
        
        # Convert dynamic object to Tensor Geometry for RaycastingScene
        # Using legacy-to-tensor conversion. 
        # Note: We do NOT compute convex hull here, preserving open-mesh geometry.
        try:
            t_object = o3d.t.geometry.TriangleMesh.from_legacy(input_mesh)
        except Exception as e:
            # Fallback for empty or malformed meshes that passed the None check
            return self.empty_structure()

        # Initialize Scene
        scene = o3d.t.geometry.RaycastingScene()
        # Add the dynamic object to the scene
        scene.add_triangles(t_object)
        
        # Prepare Query Points: Vertices of the cropped terrain
        # Float32 is standard for Open3D Tensor ops
        cropped_vertices_np = np.asarray(cropped_terrain.vertices)
        t_terrain_verts = o3d.core.Tensor.from_numpy(cropped_vertices_np.astype(np.float32))
        
        # Compute Signed Distance
        # Returns a tensor of distances. Negative = Penetration (assuming consistent normals).
        signed_distances = scene.compute_signed_distance(t_terrain_verts)
        distances_np = signed_distances.numpy()
        
        # 3. Determine Interaction
        # We identify triangles where vertices are penetrating (Distance < 0).
        # Tolerance epsilon can be adjusted if grazing contact is needed.
        EPSILON = 1e-5
        
        # Map vertex state to triangles
        # cropped_terrain has re-indexed triangles specific to this crop
        cropped_triangles = np.asarray(cropped_terrain.triangles)
        
        # Get distance for each vertex of the triangle
        # shape: (N_triangles, 3)
        tri_distances = distances_np[cropped_triangles]
        
        # Heuristic: A triangle is interacting if ALL vertices are penetrating (Strict)
        # or ANY vertex is penetrating (Lenient).
        # For non-watertight meshes (e.g. sheets), normals define the "underside".
        # We use a strict check here for robust area calculation, or a centroid check.
        # Let's use: If the Centroid is penetrating (approximated by mean of vertex distances < 0).
        
        # Check: Are all vertices of the triangle below the surface?
        inside_mask = np.all(tri_distances < EPSILON, axis=1)
        
        if not np.any(inside_mask):
            return self.empty_structure()

        # 4. Area Accumulation & Metrics
        active_tris = cropped_triangles[inside_mask]
        
        # Recalculate areas for the active subset
        # We use the cropped_vertices_np array we extracted earlier
        v0 = cropped_vertices_np[active_tris[:, 0]]
        v1 = cropped_vertices_np[active_tris[:, 1]]
        v2 = cropped_vertices_np[active_tris[:, 2]]
        
        cross_prod = np.cross(v1 - v0, v2 - v0)
        active_areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
        
        total_contact_area = np.sum(active_areas)
        
        # Calculate centroids of contacting triangles for location data
        centroids = (v0 + v1 + v2) / 3.0
        mean_location = np.mean(centroids, axis=0)
        
        # Depth Estimation:
        # For open meshes, "Depth" is the magnitude of the negative signed distance.
        # We take the max of the absolute distances of the active vertices.
        active_vertex_distances = tri_distances[inside_mask]
        contact_depth = np.max(np.abs(active_vertex_distances))

        # Visualization Data
        unique_contact_indices = np.unique(active_tris)
        contact_points = cropped_vertices_np[unique_contact_indices]

        contact_quantities = {
            "contact_detected": 1,
            "contact_points": contact_points,
            "contact_depth": float(contact_depth),
            "contact_area": float(total_contact_area),
            "contact_location_x": float(mean_location[0]),
            "contact_location_y": float(mean_location[1]),
            "contact_location_z": float(mean_location[2]),
        }
        
        contact_info = {
            "contact_points": contact_points,
            "contact_normals": np.asarray(cropped_terrain.vertex_normals)[unique_contact_indices] if cropped_terrain.has_vertex_normals() else np.array([])
        }
        
        return contact_quantities, contact_info

    def process_single_frame(
            self, 
            current_mesh: o3d.geometry.TriangleMesh,
            _debug: bool = False
        ) -> tuple[dict, dict]:
        """
        Processes a single frame of interaction.

        Args:
            current_mesh (o3d.geometry.TriangleMesh): The dynamic object mesh in world space.
            _debug (bool): Enable debug visualization for this frame.

        Returns:
            Tuple[dict, dict]: contact_data, visualization_data
        """
        self._debug = _debug
        
        if current_mesh is None:
            return self.empty_structure()
            
        return self._calculate_intersection_volume(current_mesh)
    
    def empty_structure(self) -> tuple[dict, dict]:
        """Returns standard empty data structure."""
        contact_keys = [
            "contact_detected", "contact_depth", "contact_area", 
            "contact_location_x", "contact_location_y", "contact_location_z"
        ]
        contact_data = {key: 0.0 if key == "contact_detected" else np.nan for key in contact_keys}
        contact_data["contact_detected"] = 0
        contact_data["contact_area"] = 0.0
        contact_data["contact_depth"] = 0.0
        
        visualization = {
            "contact_points": np.array([]),
            "contact_normals": np.array([])
        }
        return contact_data, visualization