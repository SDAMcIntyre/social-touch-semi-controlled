"""Mesh loading, scalar mapping, and format conversion for RF surface rendering."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from numpy import ndarray
import open3d as o3d
import pyvista as pv
from scipy.spatial import Delaunay, KDTree
import trimesh

logger = logging.getLogger(__name__)


def load_or_build_forearm_mesh(
    forearm_ply_path: Path,
    mesh_cache_path: Path = None,
) -> Optional[trimesh.Trimesh]:
    if mesh_cache_path is None:
        mesh_cache_path = forearm_ply_path.with_name(
            forearm_ply_path.stem + "_mesh_filtered.obj"
        )

    try:
        ply_mtime = forearm_ply_path.stat().st_mtime

        if mesh_cache_path.exists() and mesh_cache_path.stat().st_mtime >= ply_mtime:
            return trimesh.load(str(mesh_cache_path), force="mesh")

        pcd = o3d.io.read_point_cloud(str(forearm_ply_path))
        if pcd.is_empty():
            return None

        points = np.asarray(pcd.points)
        if points.shape[0] < 4:
            return None

        input_normals = np.asarray(pcd.normals) if pcd.has_normals() else None
        input_colors = np.asarray(pcd.colors) if pcd.has_colors() else None

        tri = Delaunay(points[:, :2])
        simplices = tri.simplices

        # Remove triangles whose longest 3D edge exceeds the local point spacing.
        # This filters spurious cross-surface connections (e.g. between fingertips)
        # that Delaunay creates from the 2D XY projection.
        v0 = points[simplices[:, 0]]
        v1 = points[simplices[:, 1]]
        v2 = points[simplices[:, 2]]
        edge_lens = np.stack([
            np.linalg.norm(v1 - v0, axis=1),
            np.linalg.norm(v2 - v1, axis=1),
            np.linalg.norm(v0 - v2, axis=1),
        ], axis=1)
        max_edge = edge_lens.max(axis=1)
        nn_dists, _ = KDTree(points).query(points, k=2)
        threshold = max(float(np.median(nn_dists[:, 1])) * 8, 10.0)
        simplices = simplices[max_edge <= threshold]

        n_removed = len(tri.simplices) - len(simplices)
        if n_removed > 0:
            logger.info(
                "Edge filtering: removed %d/%d triangles (threshold=%.1f mm) from %s",
                n_removed, len(tri.simplices), threshold, forearm_ply_path.name,
            )
        if len(simplices) == 0:
            logger.warning("All triangles filtered from %s", forearm_ply_path)
            return None

        mesh = trimesh.Trimesh(
            vertices=points,
            faces=simplices,
            vertex_colors=input_colors,
        )

        if input_normals is not None and len(input_normals) == len(points):
            face_vertex_normals = input_normals[mesh.faces]
            avg_input_normals = face_vertex_normals.mean(axis=1)
            dots = np.einsum("ij,ij->i", mesh.face_normals, avg_input_normals)
            if np.sum(dots < 0) > len(dots) / 2:
                mesh.invert()
        else:
            # Z-up assumption: forearm after tangent-plane rotation faces +Z
            if np.mean(mesh.face_normals[:, 2]) < 0:
                mesh.invert()

        mesh.fix_normals()

        mesh_cache_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(mesh_cache_path))

        return mesh

    except Exception:
        logger.exception("Failed to load or build forearm mesh from %s", forearm_ply_path)
        return None


def map_scalars_to_mesh(
    mesh: trimesh.Trimesh,
    contact_points: ndarray,
    scalar_values: ndarray,
    radius: float = 5.0,
) -> ndarray:
    per_vertex = np.full(len(mesh.vertices), np.nan, dtype=np.float64)

    if contact_points is None or len(contact_points) == 0:
        return per_vertex

    tree = KDTree(mesh.vertices)
    # Group by vertex so max scalar wins when multiple contacts map to the same vertex
    vertex_max: dict[int, float] = {}
    for pt, val in zip(contact_points, scalar_values):
        indices = tree.query_ball_point(pt, radius)
        for idx in indices:
            if idx not in vertex_max or val > vertex_max[idx]:
                vertex_max[idx] = val

    for idx, val in vertex_max.items():
        per_vertex[idx] = val

    return per_vertex


def mesh_to_pyvista(mesh: trimesh.Trimesh) -> pv.PolyData:
    vertices = mesh.vertices
    # VTK face format: [n_verts, v0, v1, ..., n_verts, v0, v1, ...]
    faces = mesh.faces
    faces_vtk = np.hstack(
        [np.full((len(faces), 1), 3, dtype=np.int64), faces]
    ).ravel()
    return pv.PolyData(vertices, faces_vtk)


def apply_rotation_to_mesh(mesh: trimesh.Trimesh, R: ndarray) -> trimesh.Trimesh:
    rotated_verts = (R @ mesh.vertices.T).T
    new_mesh = trimesh.Trimesh(vertices=rotated_verts, faces=mesh.faces.copy())
    new_mesh.fix_normals()
    return new_mesh
