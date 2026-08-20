import numpy as np
import open3d as o3d
from typing import Optional, Dict, Union, Tuple, Any
from preprocessing.forearm_extraction import serialize_contact_points

from .contact_depth_field import ContactDepthFrame, signed_contact_depth_mm

class ObjectsInteractionProcessor:
    """
    Processes the interaction between a static 2.5D terrain mesh and a dynamic 3D object.

    Refactored Architecture for Non-Watertight Support:
    - Broad Phase: Axis-Aligned Bounding Box (AABB) cropping.
    - Narrow Phase: Signed Distance check using Open3D Tensor Raycasting.
      (allows interaction detection on open shells/non-manifold geometry).
    - Metric: Surface area accumulation of penetrating terrain triangles.

    The geometry itself lives in ``contact_depth_field.signed_contact_depth_mm``;
    this class orchestrates per-frame calls and owns the CSV row schema only.
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

    def _calculate_intersection_volume(
            self,
            input_mesh: o3d.geometry.TriangleMesh,
            frame_index: int,
            time_s: float
        ) -> tuple[dict, dict, Optional[ContactDepthFrame]]:
        """
        Summarises one frame of contact into the CSV row schema.

        The geometry — AABB broad phase, signed-distance narrow phase, and the
        contact-patch triangle mask — lives in ``contact_depth_field``.  This
        method delegates to it and derives every scalar it reports from the
        returned per-vertex field, so ``contact_depth`` and the field can never
        disagree.

        Args:
            input_mesh: The dynamic object mesh.
            frame_index: Kinect frame index, stamped onto the returned field.
            time_s: Frame timestamp in seconds, stamped onto the returned field.

        Returns:
            Tuple containing contact metrics, visualization data, and the
            per-vertex depth field (``None`` when there is no contact).
        """
        frame = signed_contact_depth_mm(
            input_mesh,
            self.ref_mesh,
            frame_index=frame_index,
            time_s=time_s,
        )

        if frame is None:
            # No contact this frame — a legitimate empty result, not a failure.
            contact_quantities, contact_info = self.empty_structure()
            return contact_quantities, contact_info, None

        # Depth Estimation:
        # For open meshes, "Depth" is the magnitude of the negative signed distance.
        # Derived from the field: max(|.|) is invariant to the duplicate vertex
        # entries the legacy per-triangle formulation carried, so this is
        # bit-identical to the historical value.
        contact_depth = frame.max_penetration_depth_mm

        contact_points = frame.points

        contact_quantities = {
            "contact_detected": 1,
            "contact_points": serialize_contact_points([tuple(pt) for pt in contact_points]),
            "contact_depth": contact_depth,
            "contact_area": float(frame.total_area_mm2),
            "contact_location_x": float(frame.mean_location[0]),
            "contact_location_y": float(frame.mean_location[1]),
            "contact_location_z": float(frame.mean_location[2]),
        }

        contact_info = {
            "contact_points": contact_points,
            "contact_normals": frame.normals,
        }

        return contact_quantities, contact_info, frame

    def process_single_frame(
            self,
            current_mesh: o3d.geometry.TriangleMesh,
            frame_index: int,
            time_s: float,
            _debug: bool = False
        ) -> tuple[dict, dict, Optional[ContactDepthFrame]]:
        """
        Processes a single frame of interaction.

        Args:
            current_mesh (o3d.geometry.TriangleMesh): The dynamic object mesh in world space.
            frame_index (int): Kinect frame index; stamped onto the depth field so
                the field can be joined back to the CSV row for this frame.
            time_s (float): Frame timestamp in seconds; stamped onto the depth field.
            _debug (bool): Enable debug visualization for this frame.

        Returns:
            Tuple[dict, dict, Optional[ContactDepthFrame]]: contact_data,
            visualization_data, and the per-vertex depth field.  The field is
            ``None`` when the hand does not touch the forearm this frame; that
            is *zero contact*, never *absent measurement* — an absent hand pose
            raises instead.
        """
        self._debug = _debug

        if current_mesh is None:
            # An absent hand pose is NOT a zero-depth frame.  Collapsing the two
            # would silently report "no contact" for a frame we simply could not
            # measure, which corrupts any downstream weighting of neural firing
            # rate.  (Unreachable from ObjectsInteractionController, which
            # dereferences ``current_mesh.vertices`` before calling this.)
            raise ValueError(
                "current_mesh is None: the hand pose for this frame is absent. "
                "Record it as absent — do not process it as a zero-depth frame."
            )

        return self._calculate_intersection_volume(current_mesh, frame_index, time_s)
    
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
        contact_data["contact_points"] = "[]"
        
        visualization = {
            "contact_points": np.array([]),
            "contact_normals": np.array([])
        }
        return contact_data, visualization