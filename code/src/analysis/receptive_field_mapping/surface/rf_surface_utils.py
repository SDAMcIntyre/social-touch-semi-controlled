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

# Multi-scale ball radii expressed as multiples of the cloud's average
# nearest-neighbour distance. Smaller radii fill dense regions; larger ones
# bridge minor gaps. Stay below ~5x avg-NN to avoid stitching across the
# forearm cylinder (front-to-back surface tunneling).
_BPA_RADIUS_MULTIPLIERS = (1.5, 2.0, 2.5, 3.0)


_VALID_MESH_METHODS = ("bpa", "delaunay")


def _build_and_cache_delaunay(
    ply_path: Path,
    mesh_cache_path: Path,
    max_edge_mm: Optional[float],
) -> Optional[trimesh.Trimesh]:
    """Load a PLY point cloud, build a 2.5D Delaunay mesh, cache it, and return it.

    If *max_edge_mm* is None it is defaulted to ``3.0 * avg_nn`` where
    ``avg_nn`` is the average nearest-neighbour distance of the cloud.
    """
    pcd = o3d.io.read_point_cloud(str(ply_path))
    if pcd.is_empty():
        return None

    points = np.asarray(pcd.points)
    if points.shape[0] < 4:
        return None

    avg_nn = float(np.mean(np.asarray(pcd.compute_nearest_neighbor_distance())))

    if max_edge_mm is None:
        max_edge_mm = 3.0 * avg_nn

    mesh = build_delaunay_mesh(points, max_edge_mm=max_edge_mm)
    if mesh is None:
        logger.warning(
            "Delaunay produced no triangles for %s "
            "(avg_NN=%.3f mm, max_edge_mm=%.3f mm)",
            ply_path, avg_nn, max_edge_mm,
        )
        return None

    logger.info(
        "Delaunay reconstruction: %d triangles, %d vertices "
        "(avg_NN=%.2f mm, max_edge_mm=%.2f mm) for %s",
        len(mesh.faces), len(mesh.vertices), avg_nn, max_edge_mm, ply_path.name,
    )

    mesh_cache_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(mesh_cache_path))

    return mesh


def load_or_build_forearm_mesh(
    forearm_ply_path: Path,
    mesh_cache_path: Path = None,
    mesh_method: str = "bpa",
    max_edge_mm: Optional[float] = None,
) -> Optional[trimesh.Trimesh]:
    if mesh_method not in _VALID_MESH_METHODS:
        raise ValueError(
            f"Unknown mesh_method {mesh_method!r}. "
            f"Valid options are: {_VALID_MESH_METHODS}"
        )

    if mesh_cache_path is None:
        mesh_cache_path = forearm_ply_path.with_name(
            forearm_ply_path.stem + f"_mesh_{mesh_method}.obj"
        )

    try:
        ply_mtime = forearm_ply_path.stat().st_mtime

        if mesh_cache_path.exists() and mesh_cache_path.stat().st_mtime >= ply_mtime:
            return trimesh.load(str(mesh_cache_path), force="mesh")

        if mesh_method == "delaunay":
            return _build_and_cache_delaunay(
                ply_path=forearm_ply_path,
                mesh_cache_path=mesh_cache_path,
                max_edge_mm=max_edge_mm,
            )

        # --- BPA branch (mesh_method == "bpa") ---
        pcd = o3d.io.read_point_cloud(str(forearm_ply_path))
        if pcd.is_empty():
            return None

        points = np.asarray(pcd.points)
        if points.shape[0] < 4:
            return None

        input_colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        had_input_normals = pcd.has_normals()

        # BPA requires normals. Estimate + orient when missing.
        if not had_input_normals:
            nn_dists_pre, _ = KDTree(points).query(points, k=2)
            search_radius = max(float(np.median(nn_dists_pre[:, 1])) * 4.0, 5.0)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=search_radius, max_nn=30
                )
            )
            pcd.orient_normals_consistent_tangent_plane(k=15)

        input_normals = np.asarray(pcd.normals)

        # Multi-scale ball pivoting using avg-NN derived radii.
        avg_dist = float(np.mean(np.asarray(pcd.compute_nearest_neighbor_distance())))
        radii = [avg_dist * m for m in _BPA_RADIUS_MULTIPLIERS]

        o3d_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )

        if len(o3d_mesh.triangles) == 0:
            logger.warning(
                "BPA produced 0 triangles for %s (avg_NN=%.3f mm, radii=%s)",
                forearm_ply_path, avg_dist, radii,
            )
            return None

        o3d_mesh.remove_duplicated_vertices()
        o3d_mesh.remove_duplicated_triangles()
        o3d_mesh.remove_degenerate_triangles()
        o3d_mesh.remove_unreferenced_vertices()

        vertices = np.asarray(o3d_mesh.vertices)
        triangles = np.asarray(o3d_mesh.triangles)

        # Vertex order is preserved by BPA only when no merging/cleanup dropped
        # points; carry colors over only when the lengths still line up.
        vertex_colors = (
            input_colors
            if input_colors is not None and len(input_colors) == len(vertices)
            else None
        )

        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=triangles,
            vertex_colors=vertex_colors,
            process=False,
        )

        # Re-orient face winding so normals predominantly agree with input
        # normals (when available) or face +Z (post tangent-plane rotation).
        if had_input_normals and len(input_normals) == len(vertices):
            face_vertex_normals = input_normals[mesh.faces]
            avg_face_input_normals = face_vertex_normals.mean(axis=1)
            dots = np.einsum("ij,ij->i", mesh.face_normals, avg_face_input_normals)
            if np.sum(dots < 0) > len(dots) / 2:
                mesh.invert()
        else:
            if np.mean(mesh.face_normals[:, 2]) < 0:
                mesh.invert()

        mesh.fix_normals()

        logger.info(
            "BPA reconstruction: %d triangles, %d vertices "
            "(avg_NN=%.2f mm, radii=%s) for %s",
            len(mesh.faces), len(mesh.vertices), avg_dist,
            [f"{r:.2f}" for r in radii], forearm_ply_path.name,
        )

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


def build_delaunay_mesh(
    vertices: ndarray,
    max_edge_mm: Optional[float] = None,
) -> Optional[trimesh.Trimesh]:
    """Build a 2.5D Delaunay mesh from a forearm point cloud.

    Projects vertices onto XY for triangulation, keeping original 3D
    coordinates.  Optionally removes triangles whose longest edge
    exceeds *max_edge_mm*.
    """
    if vertices is None or len(vertices) < 3:
        return None

    tri = Delaunay(vertices[:, :2])
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=tri.simplices, process=False
    )

    if np.mean(mesh.face_normals[:, 2]) < 0:
        mesh.invert()
    mesh.fix_normals()

    if max_edge_mm is not None:
        faces = mesh.faces
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        longest = np.maximum(
            np.linalg.norm(v1 - v0, axis=1),
            np.maximum(
                np.linalg.norm(v2 - v1, axis=1),
                np.linalg.norm(v0 - v2, axis=1),
            ),
        )
        keep = longest <= max_edge_mm
        if not np.any(keep):
            return None
        mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces[keep], process=False
        )
        mesh.fix_normals()

    return mesh


def apply_rotation_to_mesh(mesh: trimesh.Trimesh, R: ndarray) -> trimesh.Trimesh:
    rotated_verts = (R @ mesh.vertices.T).T
    new_mesh = trimesh.Trimesh(vertices=rotated_verts, faces=mesh.faces.copy(), process=False)
    new_mesh.fix_normals()
    return new_mesh
