# hand_motion.py

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from typing import Dict, Optional

class HandMotion:
    """
    Encapsulates hand geometry and motion data.

    This class manages the base mesh of the hand and its trajectory,
    which is defined by a series of translations and rotations. It provides
    a method to compute the hand's transformed state (vertices and mesh)
    for any given frame in the trajectory.
    """
    def __init__(self, hand_motion_data: Dict):
        """
        Initializes the HandMotion object with geometry and trajectory data.

        Args:
            hand_motion_data (Dict): A dictionary containing:
                - 'vertices': Base vertices of the hand mesh (np.ndarray).
                - 'faces': Triangle faces of the hand mesh (np.ndarray).
                - 'time_points': Array of timestamps (np.ndarray).
                - 'translations': Array of translation vectors (np.ndarray).
                - 'rotations': Array of rotation quaternions (np.ndarray).
        """
        self._base_vertices: np.ndarray = hand_motion_data['vertices']
        self._base_triangles: np.ndarray = hand_motion_data['faces']
        
        self.time_points: np.ndarray = hand_motion_data['time_points']
        self.translations: np.ndarray = hand_motion_data['translations']
        self.rotations: np.ndarray = hand_motion_data['rotations']
        
        # Create and store the base Open3D TriangleMesh object
        self._base_mesh = o3d.geometry.TriangleMesh()
        self._base_mesh.vertices = o3d.utility.Vector3dVector(self._base_vertices)
        self._base_mesh.triangles = o3d.utility.Vector3iVector(self._base_triangles)
        
        # Define and store the color as an instance attribute
        hand_color = [228/255, 178/255, 148/255]
        self._base_mesh.paint_uniform_color(hand_color)
        self._base_mesh.compute_vertex_normals()

    @property
    def base_vertices(self) -> np.ndarray:
        """Returns the base vertices of the hand mesh."""
        return self._base_vertices

    @property
    def base_triangles_vector(self) -> o3d.utility.Vector3iVector:
        """Returns the base triangles as an Open3D Vector3iVector."""
        return self._base_mesh.triangles

    @property
    def base_mesh(self) -> o3d.geometry.TriangleMesh:
        """Returns the base hand mesh in its default pose."""
        return self._base_mesh

    @property
    def num_frames(self) -> int:
        """Returns the total number of frames in the motion sequence."""
        return len(self.time_points)

    def get_transformed_vertices_for_frame(self, frame_id: int) -> Optional[np.ndarray]:
        """
        Calculates the transformed vertices for a specific frame.

        Args:
            frame_id (int): The index of the frame.

        Returns:
            Optional[np.ndarray]: The vertices of the hand mesh after applying the
                                  frame-specific rotation and translation, or None
                                  if the motion data for the frame is invalid
                                  (e.g., NaN values or zero-norm quaternion).
        """
        if not 0 <= frame_id < self.num_frames:
            raise IndexError("frame_id is out of bounds.")
            
        rotation_quat = self.rotations[frame_id]
        translation = self.translations[frame_id]
        
        # Check for invalid motion data (NaNs or zero-norm quaternions).
        # A zero-norm quaternion cannot represent a rotation and will cause an error.
        is_quat_invalid = np.isnan(rotation_quat).any() or np.isclose(np.linalg.norm(rotation_quat), 0)
        is_trans_invalid = np.isnan(translation).any()
        
        if is_quat_invalid or is_trans_invalid:
            return None
        
        # Apply rotation (from quaternion) and then translation.
        # A try-except block is included as a fail-safe against other potential
        # scipy errors, though the checks above should be sufficient.
        try:
            rot = R.from_quat(rotation_quat)
            transformed_vertices = rot.apply(self._base_vertices) + translation
            return transformed_vertices
        except ValueError:
            return None

    def get_hand_mesh_for_frame(self, frame_id: int) -> Optional[o3d.geometry.TriangleMesh]:
        """
        Generates a new hand mesh for a specific frame's pose.

        Args:
            frame_id (int): The index of the frame.

        Returns:
            Optional[o3d.geometry.TriangleMesh]: A new mesh object representing the hand
                                                 at the specified frame, or None if the
                                                 motion data is invalid.
        """
        transformed_vertices = self.get_transformed_vertices_for_frame(frame_id)
        
        if transformed_vertices is None:
            return None
        
        # Create a new mesh for the transformed state
        transformed_mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(transformed_vertices),
            self.base_triangles_vector
        )
        # Copy visual and geometric properties from the base mesh
        # This is more robust than re-painting, as it preserves per-vertex colors if they exist.
        transformed_mesh.vertex_colors = self._base_mesh.vertex_colors
        transformed_mesh.compute_vertex_normals()
        
        return transformed_mesh
    
    def get_all_transformed_meshes(self) -> Dict[int, o3d.geometry.TriangleMesh]:
        """
        Generates a dictionary of transformed hand meshes for all valid frames.

        This method iterates through the entire motion sequence, computes the
        transformed mesh for each frame, and returns them in a dictionary.
        Frames with invalid motion data (e.g., NaNs or zero-norm quaternions)
        are skipped and not included in the output.

        Returns:
            Dict[int, o3d.geometry.TriangleMesh]: A dictionary where keys are
                frame IDs (int) and values are the corresponding transformed
                Open3D TriangleMesh objects for valid frames.
        """
        valid_meshes = {}
        for frame_id in range(self.num_frames):
            mesh = self.get_hand_mesh_for_frame(frame_id)
            if mesh is not None:
                valid_meshes[frame_id] = mesh
        return valid_meshes