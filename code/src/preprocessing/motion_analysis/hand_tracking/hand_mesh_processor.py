from pathlib import Path
import numpy as np
import open3d as o3d

from .imported.hand_mesh import HandMesh
from .imported import config


class HandMeshProcessor:
    """
    Handles loading, processing, and transforming a 3D hand mesh.

    This class now acts as a stateless utility. The create_mesh method
    is a static function that takes inputs and produces an Open3D mesh object.
    """
    @staticmethod
    def create_mesh(handmesh_vertices_path: Path, parameters: dict, SCALE_M_TO_MM: bool = True) -> o3d.geometry.TriangleMesh:
        """
        Loads vertices and faces to create an Open3D TriangleMesh.

        Args:
            handmesh_vertices_path (Path): Path to the .txt file containing mesh vertices.
            parameters (dict): A dictionary containing configuration, e.g., {'left': False}.

        Returns:
            o3d.geometry.TriangleMesh: The constructed and correctly oriented hand mesh.
        """
        v_handMesh = np.loadtxt(handmesh_vertices_path)
        # It's good practice to define "magic numbers" as named constants
        if SCALE_M_TO_MM:
            SCALE_FACTOR_M_TO_MM = 1000
            v_handMesh *= SCALE_FACTOR_M_TO_MM

        hand_mesh_model = HandMesh(config.HAND_MESH_MODEL_PATH)
        t_handMesh = np.asarray(hand_mesh_model.faces)

        # Using .get() with a default value is a safe way to access dict keys
        if not parameters.get('left', False):
            v_handMesh[:, 0] *= -1
            t_handMesh = t_handMesh[:, [0, 2, 1]]

        handmesh = o3d.geometry.TriangleMesh()
        handmesh.vertices = o3d.utility.Vector3dVector(v_handMesh)
        handmesh.triangles = o3d.utility.Vector3iVector(t_handMesh)
        handmesh.paint_uniform_color(config.HAND_COLOR)
        handmesh.compute_vertex_normals()

        return handmesh