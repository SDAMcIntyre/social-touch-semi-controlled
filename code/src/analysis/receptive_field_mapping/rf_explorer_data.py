"""Per-frame data model and loader for the RF Feature-Space Explorer GUI."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import cKDTree

from .rf_data_loader import load_forearm_vertices
from .rf_surface_utils import build_delaunay_mesh, mesh_to_pyvista
from .tangent_plane_alignment import compute_tangent_plane_rotation


@dataclass
class ExplorerSessionData:
    forearm_mesh: pv.PolyData
    vertices: np.ndarray
    tangent_rotation: np.ndarray


@dataclass
class ExplorerData:
    pressure: np.ndarray
    velocity_signed: np.ndarray
    gesture_types: np.ndarray
    spikes: np.ndarray
    frame_vertex_idx: np.ndarray
    session_data: ExplorerSessionData

    @property
    def n_frames(self) -> int:
        return len(self.pressure)


def load_explorer_data(
    series_csv_path: Path,
    forearm_ply_path: Path,
    max_edge_mm: float = 20.0,
) -> ExplorerData:
    df = pd.read_csv(series_csv_path)

    mask = df["contact_location_x"].notna()
    df = df[mask].reset_index(drop=True)

    pressure = df["pressure"].to_numpy(dtype=np.float64)
    velocity_signed = df["hand_velocity_signed"].to_numpy(dtype=np.float64)
    gesture_types = df["gesture_type"].to_numpy(dtype=object)
    spikes = df["Nerve_spike"].to_numpy(dtype=bool)
    contact_pts = df[
        ["contact_location_x", "contact_location_y", "contact_location_z"]
    ].to_numpy(dtype=np.float64)

    vertices = load_forearm_vertices(forearm_ply_path)
    if vertices is None:
        raise ValueError(
            f"load_explorer_data: could not load forearm vertices from {forearm_ply_path}"
        )

    trimesh_mesh = build_delaunay_mesh(vertices, max_edge_mm=max_edge_mm)
    if trimesh_mesh is None:
        raise ValueError(
            f"load_explorer_data: build_delaunay_mesh produced no mesh for {forearm_ply_path}"
        )

    contact_centroid = contact_pts.mean(axis=0)
    rotation = compute_tangent_plane_rotation(vertices, contact_centroid)
    if rotation is None:
        raise ValueError(
            f"load_explorer_data: compute_tangent_plane_rotation returned None "
            f"for {forearm_ply_path}"
        )

    rotated_vertices = (rotation @ trimesh_mesh.vertices.T).T
    rotated_contacts = (rotation @ contact_pts.T).T

    import trimesh as _trimesh
    rotated_mesh = _trimesh.Trimesh(
        vertices=rotated_vertices,
        faces=trimesh_mesh.faces.copy(),
        process=False,
    )
    rotated_mesh.fix_normals()
    forearm_mesh = mesh_to_pyvista(rotated_mesh)

    tree = cKDTree(rotated_vertices)
    distances, frame_vertex_idx = tree.query(rotated_contacts)

    bad = distances > 15.0
    if np.any(bad):
        raise ValueError(
            f"load_explorer_data: {bad.sum()} frame(s) have nearest-vertex distance "
            f"exceeding 15mm (max={distances.max():.2f}mm)"
        )

    session_data = ExplorerSessionData(
        forearm_mesh=forearm_mesh,
        vertices=rotated_vertices,
        tangent_rotation=rotation,
    )

    return ExplorerData(
        pressure=pressure,
        velocity_signed=velocity_signed,
        gesture_types=gesture_types,
        spikes=spikes,
        frame_vertex_idx=frame_vertex_idx,
        session_data=session_data,
    )
