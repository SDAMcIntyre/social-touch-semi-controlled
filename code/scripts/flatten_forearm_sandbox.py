#!/usr/bin/env python
"""
Flatten-Forearm Sandbox
=======================

Purpose
-------
Standalone development script for iterating on 3D-to-2D forearm-surface
flattening algorithms outside the Prefect DAG / receptive-field-mapping
pipeline.  Loads a single forearm PLY (path hard-coded below), builds a
triangle mesh, cleans it, and runs three flattening baselines side-by-side.

How to use
----------
1. Edit ``PLY_PATH`` below to point at a forearm segmentation PLY (e.g.
   ``{session_processed_path}/forearm_pointclouds/forearm_0001.ply``).
2. Optionally change ``MESH_METHOD`` (``"bpa"`` or ``"delaunay"``).
3. Run:  ``python code/scripts/flatten_forearm_sandbox.py``
   A single matplotlib window with four panels appears.

Mesh methods
------------
- ``"bpa"``      — Ball-Pivoting Algorithm via Open3D; uses
                   ``load_or_build_forearm_mesh()`` from
                   ``analysis/receptive_field_mapping/rf_surface_utils.py``.
                   Produces a high-quality manifold mesh and caches the
                   result as an OBJ next to the PLY.
- ``"delaunay"`` — Fast 2.5-D Delaunay triangulation via
                   ``scipy.spatial.Delaunay`` in the PCA-projected plane.
                   No caching; useful for quick geometry checks.

Flattening methods (implemented in Phase 2)
-------------------------------------------
- LSCM     — Least-Squares Conformal Maps; minimises angle distortion
             via a sparse linear system.  Pin two boundary vertices to
             fix the gauge freedom.
- Harmonic — Solve the Laplace equation with boundary vertices mapped to
             a unit circle.  Smooth but may produce more area distortion
             than LSCM on elongated surfaces.
- ARAP     — As-Rigid-As-Possible; non-linear energy that locally
             preserves rigidity.  Initialised with the harmonic map and
             refined by alternating local-global iterations (~20 steps).

Algorithm reference
-------------------
``docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md``
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
import trimesh.repair

# ---------------------------------------------------------------------------
# Extend sys.path so pipeline helpers can be imported without installing
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent   # …/social-touch-semi-controlled
_SRC = _REPO_ROOT / "code" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# User-editable constants — edit these before running
# ---------------------------------------------------------------------------

#: Absolute or relative path to the forearm segmentation PLY.
#: Must be set by the user before running this script.
PLY_PATH: str = ""

#: Mesh construction method: ``"bpa"`` (Ball-Pivoting Algorithm, recommended)
#: or ``"delaunay"`` (fast 2.5-D triangulation, no caching).
MESH_METHOD: str = "bpa"


# ---------------------------------------------------------------------------
# Phase 1 — load, mesh, clean
# ---------------------------------------------------------------------------

def load_pcd(path: str) -> np.ndarray:
    """Load a PLY point cloud and return the (N, 3) point array.

    Parameters
    ----------
    path:
        Path to the PLY file.

    Returns
    -------
    np.ndarray
        Shape (N, 3), dtype float64.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the point cloud is empty after loading.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"PLY file not found: {p}\n"
            "Edit PLY_PATH at the top of this script to point at a valid forearm PLY."
        )

    pcd = o3d.io.read_point_cloud(str(p))
    if pcd.is_empty():
        raise ValueError(
            f"Point cloud loaded from '{p}' is empty — "
            "check that the file is a valid PLY with XYZ data."
        )

    points = np.asarray(pcd.points, dtype=np.float64)
    return points


def build_mesh(ply_path: Path, method: str) -> trimesh.Trimesh:
    """Build a triangle mesh from a forearm PLY using the chosen method.

    Parameters
    ----------
    ply_path:
        Path to the forearm segmentation PLY.
    method:
        ``"bpa"``  — Ball-Pivoting Algorithm (via
                     ``rf_surface_utils.load_or_build_forearm_mesh``).
        ``"delaunay"`` — 2.5-D Delaunay in the PCA-projected plane.

    Returns
    -------
    trimesh.Trimesh

    Raises
    ------
    FileNotFoundError
        If the PLY does not exist (propagated from ``load_pcd``).
    ValueError
        If an unsupported method is requested, or if the BPA mesh builder
        returns ``None`` (empty cloud or degenerate input).
    """
    if method == "bpa":
        # Import lazily so the delaunay path has no dep on the analysis package.
        from analysis.receptive_field_mapping.rf_surface_utils import (
            load_or_build_forearm_mesh,
        )
        mesh = load_or_build_forearm_mesh(ply_path)
        if mesh is None:
            raise ValueError(
                f"BPA mesh builder returned None for '{ply_path}'. "
                "The point cloud may be too sparse or degenerate."
            )
        return mesh

    elif method == "delaunay":
        points = load_pcd(str(ply_path))

        # Project into the dominant plane via PCA (forearm is roughly planar
        # when viewed along its principal axis).
        mean = points.mean(axis=0)
        centered = points - mean
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        # The two principal axes span the forearm plane.
        projected_2d = centered @ Vt[:2].T  # (N, 2)

        from scipy.spatial import Delaunay as _Delaunay

        tri = _Delaunay(projected_2d)
        faces = tri.simplices.astype(np.int64)

        mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
        return mesh

    else:
        raise ValueError(
            f"Unknown MESH_METHOD '{method}'. "
            "Must be one of: 'bpa', 'delaunay'."
        )


def clean_mesh(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Keep the largest connected component, fix winding, drop orphan vertices.

    Parameters
    ----------
    mesh:
        Input trimesh (may have multiple components, flipped faces, or
        unreferenced vertices).

    Returns
    -------
    V : np.ndarray
        Vertex positions, shape (N, 3), float64.
    F : np.ndarray
        Face indices (into V), shape (M, 3), int64.

    Raises
    ------
    ValueError
        If the mesh has zero faces after cleaning (completely degenerate
        input or wrong PLY file).
    """
    # Split into connected components; keep the one with the most vertices.
    components = mesh.split(only_watertight=False)
    if len(components) == 0:
        raise ValueError(
            "mesh.split() returned no components — the mesh is empty or "
            "entirely degenerate."
        )

    largest = max(components, key=lambda m: len(m.vertices))

    # Fix face winding for consistent outward normals.
    trimesh.repair.fix_winding(largest)

    # Remove vertices that are not referenced by any face.
    largest.remove_unreferenced_vertices()

    V = np.asarray(largest.vertices, dtype=np.float64)
    F = np.asarray(largest.faces, dtype=np.int64)

    if F.shape[0] == 0:
        raise ValueError(
            "Zero faces remain after cleaning the mesh. "
            "Check that PLY_PATH points to a segmented forearm cloud, "
            "not a raw scan or an empty file."
        )

    return V, F


# ---------------------------------------------------------------------------
# Phase 2 — flattening baselines
# ---------------------------------------------------------------------------

import igl


def boundary_loop(F: np.ndarray) -> np.ndarray:
    """Return the ordered boundary vertex indices for the mesh."""
    b = igl.boundary_loop(F)
    if b is None or len(b) == 0:
        raise RuntimeError(
            "No boundary loop found — the mesh appears to be a closed surface. "
            "Forearm meshes must have an open boundary for flattening."
        )
    return b


def flatten_lscm(V: np.ndarray, F: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Least-Squares Conformal Map with two pinned boundary vertices."""
    # Pin boundary[0] → (0, 0) and the opposite vertex → (1, 0) to fix gauge freedom.
    b = np.array([boundary[0], boundary[len(boundary) // 2]], dtype=np.int32)
    bc = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    _, uv = igl.lscm(V, F, b, bc)
    if np.any(np.isnan(uv)):
        raise RuntimeError("LSCM produced NaN values — mesh may be degenerate.")
    return uv.astype(np.float64)


def flatten_harmonic(V: np.ndarray, F: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """Harmonic map with boundary vertices uniformly distributed on the unit circle."""
    n_b = len(boundary)
    angles = np.linspace(0.0, 2.0 * np.pi, n_b, endpoint=False)
    boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)])
    uv = igl.harmonic(V, F, boundary.astype(np.int32), boundary_uv, 1)
    if np.any(np.isnan(uv)):
        raise RuntimeError("Harmonic map produced NaN values — mesh may be degenerate.")
    return uv.astype(np.float64)


def flatten_arap(V: np.ndarray, F: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """As-Rigid-As-Possible flattening, initialised from the harmonic map."""
    n_b = len(boundary)
    angles = np.linspace(0.0, 2.0 * np.pi, n_b, endpoint=False)
    bc = np.column_stack([np.cos(angles), np.sin(angles)])

    uv = flatten_harmonic(V, F, boundary)

    arap = igl.ARAP(V, F, 2, boundary.astype(np.int32))
    for _ in range(20):
        uv_new = arap.solve(bc, uv)
        rel_change = np.linalg.norm(uv_new - uv) / (np.linalg.norm(uv) + 1e-12)
        uv = uv_new
        if rel_change < 1e-6:
            break

    if np.any(np.isnan(uv)):
        raise RuntimeError("ARAP produced NaN values — mesh may be degenerate.")
    return uv.astype(np.float64)


# ---------------------------------------------------------------------------
# Phase 3 — visualisation + entry point
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.tri


def plot_panels(
    V: np.ndarray,
    F: np.ndarray,
    uv_lscm: np.ndarray,
    uv_arap: np.ndarray,
    uv_harmonic: np.ndarray,
) -> None:
    """Render the 3D mesh and three 2D flattenings in a single figure."""
    fig = plt.figure(figsize=(16, 4))

    # -- 3D input mesh --------------------------------------------------------
    ax3d = fig.add_subplot(141, projection="3d")
    ax3d.plot_trisurf(
        V[:, 0], V[:, 1], V[:, 2],
        triangles=F,
        alpha=0.7,
        color="steelblue",
        edgecolor="none",
    )
    ax3d.set_title("Input mesh (3D)")

    # -- 2D flattening panels -------------------------------------------------
    panels = [
        (142, uv_lscm, "LSCM"),
        (143, uv_arap, "ARAP"),
        (144, uv_harmonic, "Harmonic"),
    ]
    for subplot_id, uv, title in panels:
        ax = fig.add_subplot(subplot_id)
        tri = matplotlib.tri.Triangulation(uv[:, 0], uv[:, 1], F)
        ax.triplot(tri, color="steelblue", linewidth=0.4)
        ax.set_title(title)
        ax.set_aspect("equal")


if __name__ == "__main__":
    if not PLY_PATH:
        raise ValueError(
            "PLY_PATH is not set. "
            "Edit the PLY_PATH constant at the top of this script."
        )

    ply_path = Path(PLY_PATH)
    raw_mesh = build_mesh(ply_path, MESH_METHOD)
    V_raw, F_raw = clean_mesh(raw_mesh)

    boundary = boundary_loop(F_raw)

    uv_lscm = flatten_lscm(V_raw, F_raw, boundary)
    uv_harmonic = flatten_harmonic(V_raw, F_raw, boundary)
    uv_arap = flatten_arap(V_raw, F_raw, boundary)

    plot_panels(V_raw, F_raw, uv_lscm, uv_arap, uv_harmonic)
    plt.tight_layout()
    plt.show()
