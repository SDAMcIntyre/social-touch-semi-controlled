#!/usr/bin/env python
"""
Compare Six Forearm Flattening Methods (SLIM Stage-1 Study)
===========================================================

Purpose
-------
Standalone comparison harness that flattens the same cleaned forearm mesh
with six methods side-by-side and emits comparison figures plus a per-method
distortion summary CSV.

    docs/development/plans/active/compare-flattening-methods.md

The three baseline mesh-parametric methods (LSCM, Harmonic, ARAP) and all
mesh/cleaning helpers are imported directly from ``flatten_forearm_sandbox``.
SLIM is implemented here.  The two production projections (tangent-plane and
cylindrical-unwrap) are imported from ``analysis.receptive_field_mapping``.

How to use
----------
1. Edit ``PLY_PATH`` below.
2. Optionally set ``CENTER_POINT`` to a (x, y, z) tuple from a previous run.
3. Run:  ``python code/scripts/compare_flattening_methods.py``

Output (saved next to the PLY)
-------------------------------
- ``*_compare_{timestamp}.png``            — 7-panel figure (3D + 6 UVs)
- ``*_compare_{timestamp}_distortion.png`` — 6-column distortion maps
- ``*_compare_{timestamp}_summary.csv``    — per-method distortion statistics
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.tri
from matplotlib.collections import PolyCollection

import numpy as np

# ---------------------------------------------------------------------------
# Extend sys.path so pipeline helpers and the sandbox script can be imported
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC = _REPO_ROOT / "code" / "src"
_SCRIPTS = _REPO_ROOT / "code" / "scripts"

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# Import reused functions from the existing sandbox (no re-implementation).
from flatten_forearm_sandbox import (  # noqa: E402
    load_pcd,
    build_mesh,
    clean_mesh,
    pick_center_point,
    boundary_loop,
    canonicalise_uv,
    flatten_lscm,
    flatten_harmonic,
    flatten_arap,
    compute_face_distortion,
    plot_panels,
    plot_distortion_panels,
    show_mesh_inspector,
)

import igl  # noqa: E402  (must come after flatten_forearm_sandbox which also imports it)

from analysis.receptive_field_mapping._slim_helpers import (  # noqa: E402
    _has_flipped_triangles,
    flatten_slim as _slim_base,
)

from analysis.receptive_field_mapping.rf_projection import (  # noqa: E402
    project_tangent_plane,
    project_cylindrical_unwrap,
)

# ---------------------------------------------------------------------------
# User-editable constants — edit these before running
# ---------------------------------------------------------------------------

PLY_PATH: str = (
    r"F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/02_data"
    r"/semi-controlled/3_merged/2022-06-17_ST16-05/2022-06-17_ST16-05_forearm.ply"
)

# Second session — uncomment to run on 2022-06-16_ST15-01:
# PLY_PATH = (
#     r"F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/02_data"
#     r"/semi-controlled/3_merged/2022-06-16_ST15-01/2022-06-16_ST15-01_forearm.ply"
# )

MESH_METHOD: str = "bpa"

#: Center point for centre-weighted flattening.  Three modes:
#:
#: * ``None``            — skip picker; no centre weighting.
#: * ``"interactive"``   — open the PyVista picker window.
#: * ``(x, y, z)`` tuple — use this hard-coded 3D coordinate directly.
CENTER_POINT: tuple[float, float, float] | str | None = "interactive"


# ---------------------------------------------------------------------------
# PCA rotation helper
# ---------------------------------------------------------------------------

def _pca_rotation(V: np.ndarray) -> np.ndarray:
    """Return PCA principal axes as a (3, 3) rotation matrix.

    Row 0 = first principal axis (longest extent), Row 1 = second, Row 2 = third.
    Used as a stand-in for the saved RF-camera rotation when applying the
    production projections (tangent-plane, cylindrical-unwrap) to the whole
    mesh in this geometry-level comparison.  Do **not** use this matrix in
    production; load the per-session saved camera rotation instead.

    Parameters
    ----------
    V:
        Vertex positions, shape (N, 3), float64.

    Returns
    -------
    Vt : np.ndarray
        (3, 3) orthonormal matrix whose rows are the PCA principal axes.
    """
    centered = V - V.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt  # (3, 3) orthonormal


# ---------------------------------------------------------------------------
# SLIM flattening
# ---------------------------------------------------------------------------

# _has_flipped_triangles is imported from _slim_helpers above.


def _tutte_uniform_map(
    F: np.ndarray,
    n_vertices: int,
    boundary: np.ndarray,
    boundary_uv: np.ndarray,
) -> np.ndarray:
    # Tutte (1963): uniform-weight Laplacian + convex boundary => bijective.
    # Used as a foldover-free fallback when the cotangent-harmonic init flips.
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    e_all = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    edges = np.unique(np.sort(e_all, axis=1), axis=0)
    a = edges[:, 0]
    b = edges[:, 1]
    n_e = len(edges)
    rows = np.concatenate([a, b, a, b])
    cols = np.concatenate([b, a, a, b])
    vals = np.concatenate([-np.ones(n_e), -np.ones(n_e), np.ones(n_e), np.ones(n_e)])
    L = coo_matrix((vals, (rows, cols)), shape=(n_vertices, n_vertices)).tolil()

    boundary = np.asarray(boundary, dtype=np.int64)
    for bi in boundary:
        L.rows[bi] = [int(bi)]
        L.data[bi] = [1.0]
    rhs = np.zeros((n_vertices, 2), dtype=np.float64)
    rhs[boundary] = boundary_uv

    return np.asarray(spsolve(L.tocsr(), rhs))


def flatten_slim(
    V: np.ndarray,
    F: np.ndarray,
    boundary: np.ndarray,
    center_vid: int | None = None,
    n_iter: int = 40,
) -> np.ndarray:
    """Sandbox wrapper around the production ``flatten_slim``.

    Delegates to ``_slim_base`` (the production implementation in
    ``_slim_helpers``).  When the cotangent-harmonic initialisation has
    flipped triangles, falls back to the Tutte (uniform-weight Laplacian)
    map for robustness during sandbox exploration.  The production module
    does **not** include this fallback — it raises immediately.
    """
    try:
        return _slim_base(V, F, boundary, center_vid=center_vid, n_iter=n_iter)
    except RuntimeError as exc:
        if "flipped triangles" not in str(exc):
            raise
        # Tutte fallback for sandbox exploration only.
        n_b = len(boundary)
        angles = np.linspace(0.0, 2.0 * np.pi, n_b, endpoint=False)
        boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)])
        uv_init = _tutte_uniform_map(F, V.shape[0], boundary, boundary_uv)
        if _has_flipped_triangles(uv_init, F):
            raise RuntimeError(
                "Tutte fallback still produced flipped triangles — the mesh "
                "topology is likely invalid (non-manifold edges, holes, or "
                "boundary not mapping homeomorphically)."
            )
        # Run SLIM from the Tutte init.
        empty_b = np.empty((0,), dtype=np.int32)
        empty_bc = np.empty((0, 2), dtype=np.float64)
        data = igl.slim_precompute(
            V, F, uv_init,
            igl.MappingEnergyType.SYMMETRIC_DIRICHLET,
            empty_b, empty_bc, 0.0,
        )
        uv = uv_init
        for _ in range(n_iter):
            uv = igl.slim_solve(data, 1)
        if np.any(np.isnan(uv)):
            raise RuntimeError("SLIM produced NaN values after Tutte fallback.")
        uv = uv.astype(np.float64)
        from analysis.receptive_field_mapping._slim_helpers import canonicalise_uv
        if center_vid is not None:
            b0 = int(boundary[0])
            uv = canonicalise_uv(uv, int(center_vid), b0)
        return uv


# ---------------------------------------------------------------------------
# 7-panel comparison figure
# ---------------------------------------------------------------------------

def plot_comparison_panels(
    V: np.ndarray,
    F: np.ndarray,
    uvs: dict[str, np.ndarray],
    colors: np.ndarray | None = None,
    center_vid: int | None = None,
) -> plt.Figure:
    """7-panel figure: 3D input + one UV panel per method.

    Parameters
    ----------
    V:
        Vertex positions, shape (N, 3), float64.
    F:
        Face indices, shape (M, 3), int32.
    uvs:
        Dict mapping method name → UV array of shape (N, 2).  Dict insertion
        order determines panel order (left to right).
    colors:
        Per-vertex RGB colours, shape (N, 3), float in [0, 1].  When provided,
        faces are coloured using the mean vertex colour.  When None, ``triplot``
        wireframe is drawn instead.
    center_vid:
        Vertex index to mark as a red dot in every UV panel.  When None, no
        marker is drawn.

    Returns
    -------
    fig : plt.Figure
    """
    n_methods = len(uvs)
    fig = plt.figure(figsize=(4 * (n_methods + 1), 4))

    ax3d = fig.add_subplot(1, n_methods + 1, 1, projection="3d")
    ax3d.plot_trisurf(
        V[:, 0], V[:, 1], V[:, 2],
        triangles=F,
        color="lightgray",
        edgecolor="none",
        alpha=0.7,
    )
    ax3d.set_title("3D input")
    ax3d.set_axis_off()

    for col_idx, (method_name, uv) in enumerate(uvs.items(), start=2):
        ax = fig.add_subplot(1, n_methods + 1, col_idx)

        if colors is not None:
            face_colors = colors[F].mean(axis=1)
            pc = PolyCollection(
                uv[F],
                facecolors=face_colors,
                edgecolors="none",
            )
            ax.add_collection(pc)
            ax.set_xlim(uv[:, 0].min(), uv[:, 0].max())
            ax.set_ylim(uv[:, 1].min(), uv[:, 1].max())
        else:
            tri = matplotlib.tri.Triangulation(uv[:, 0], uv[:, 1], F)
            ax.triplot(tri, color="steelblue", linewidth=0.3)

        if center_vid is not None:
            ax.plot(
                uv[center_vid, 0],
                uv[center_vid, 1],
                "r.",
                markersize=6,
            )

        ax.set_aspect("equal")
        ax.set_title(method_name)

    return fig


# ---------------------------------------------------------------------------
# Entry point — Phase 2 six-method orchestration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import csv
    from datetime import datetime

    if not PLY_PATH:
        raise ValueError(
            "PLY_PATH is not set. "
            "Edit the PLY_PATH constant at the top of this script."
        )

    ply_path = Path(PLY_PATH)

    # ------------------------------------------------------------------
    # 1. Load point cloud and pick centre point
    # ------------------------------------------------------------------
    orig_points, orig_colors = load_pcd(str(ply_path))

    if CENTER_POINT == "interactive":
        center_3d = pick_center_point(orig_points, orig_colors)
    elif isinstance(CENTER_POINT, tuple):
        center_3d = np.array(CENTER_POINT, dtype=np.float64)
    elif CENTER_POINT is None:
        center_3d = None
    else:
        raise ValueError(
            f"Invalid CENTER_POINT value: {CENTER_POINT!r}. "
            "Must be None, 'interactive', or a (x, y, z) tuple."
        )

    # ------------------------------------------------------------------
    # 2. Build and clean mesh
    # ------------------------------------------------------------------
    raw_mesh = build_mesh(ply_path, MESH_METHOD)
    V, F = clean_mesh(raw_mesh)

    # Map original PLY colours onto cleaned mesh vertices via KDTree
    if orig_colors is not None:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(orig_points).query(V)
        vertex_colors = orig_colors[idx]
    else:
        vertex_colors = None

    # ------------------------------------------------------------------
    # 3. Resolve centre point → vertex index
    # ------------------------------------------------------------------
    if center_3d is not None:
        from scipy.spatial import cKDTree as _cKDTree
        _, center_vid = _cKDTree(V).query(center_3d)
        center_vid = int(center_vid)
    else:
        center_vid = None

    # ------------------------------------------------------------------
    # 4. Boundary loop (warn if suspiciously short)
    # ------------------------------------------------------------------
    boundary = boundary_loop(F)
    if len(boundary) < V.shape[0] * 0.01:
        print(
            f"WARNING: boundary loop is very short ({len(boundary)} vertices "
            f"vs {V.shape[0]} total). The mesh may have poor boundary topology."
        )

    # ------------------------------------------------------------------
    # 5. Run four mesh-parametric flattening methods
    # ------------------------------------------------------------------
    print("Running LSCM ...")
    uv_lscm = flatten_lscm(V, F, boundary, center_vid=center_vid)
    print("  LSCM converged.")

    print("Running ARAP ...")
    uv_arap = flatten_arap(V, F, boundary, center_vid=center_vid)
    print("  ARAP converged.")

    print("Running Harmonic ...")
    uv_harmonic = flatten_harmonic(V, F, boundary, center_vid=center_vid)
    print("  Harmonic converged.")

    print("Running SLIM ...")
    uv_slim = flatten_slim(V, F, boundary, center_vid=center_vid)
    print("  SLIM converged.")

    # ------------------------------------------------------------------
    # 6. Synthesise PCA rotation matrix and run production projections
    # ------------------------------------------------------------------
    # The PCA rotation is used only as a stand-in for the saved per-session
    # RF-camera rotation.  It gives the projections a geometrically meaningful
    # coordinate frame (longest forearm axis = row 0) without requiring a
    # previously run set_rf_camera_settings session.
    rotation_matrix = _pca_rotation(V)

    print("Running TangentPlane projection ...")
    uv_tangent = project_tangent_plane(
        points_3d=V,
        forearm_vertices=V,
        contact_centroid=V.mean(axis=0),
        rotation_matrix=rotation_matrix,
    )
    print("  TangentPlane done.")

    print("Running Cylindrical projection ...")
    uv_cyl = project_cylindrical_unwrap(
        points_3d=V,
        forearm_vertices=V,
        contact_centroid=V.mean(axis=0),
        per_point_radius=False,
        rotation_matrix=rotation_matrix,
    )
    print("  Cylindrical done.")

    # ------------------------------------------------------------------
    # 7. Collect UV arrays
    # ------------------------------------------------------------------
    uvs: dict[str, np.ndarray] = {
        "LSCM": uv_lscm,
        "ARAP": uv_arap,
        "Harmonic": uv_harmonic,
        "SLIM": uv_slim,
        "TangentPlane": uv_tangent,
        "Cylindrical": uv_cyl,
    }

    # ------------------------------------------------------------------
    # 8. Compute per-face distortion metrics for every method
    # ------------------------------------------------------------------
    distortions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method_name, uv in uvs.items():
        conf, area = compute_face_distortion(V, F, uv)
        distortions[method_name] = (conf, area)

    # ------------------------------------------------------------------
    # 9. Build and persist summary CSV
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = ply_path.with_name(
        f"{ply_path.stem}_compare_{timestamp}_summary.csv"
    )
    fieldnames = [
        "method",
        "conformal_median",
        "conformal_p95",
        "area_median",
        "area_p95",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method_name, (conf, area) in distortions.items():
            writer.writerow(
                {
                    "method": method_name,
                    "conformal_median": float(np.median(conf)),
                    "conformal_p95": float(np.percentile(conf, 95)),
                    "area_median": float(np.median(area)),
                    "area_p95": float(np.percentile(area, 95)),
                }
            )
    print(f"\nSummary CSV saved: {csv_path}")

    # ------------------------------------------------------------------
    # 10. Print summary table to stdout
    # ------------------------------------------------------------------
    col_w = 14
    header = (
        f"{'Method':<{col_w}}"
        f"{'conf_median':>{col_w}}"
        f"{'conf_p95':>{col_w}}"
        f"{'area_median':>{col_w}}"
        f"{'area_p95':>{col_w}}"
    )
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    for method_name, (conf, area) in distortions.items():
        print(
            f"{method_name:<{col_w}}"
            f"{np.median(conf):>{col_w}.4f}"
            f"{np.percentile(conf, 95):>{col_w}.4f}"
            f"{np.median(area):>{col_w}.4f}"
            f"{np.percentile(area, 95):>{col_w}.4f}"
        )
    print(separator)

    # ------------------------------------------------------------------
    # 11. Emit comparison PNG
    # ------------------------------------------------------------------
    fig_compare = plot_comparison_panels(V, F, uvs, vertex_colors, center_vid)
    fig_compare.tight_layout()
    compare_png = ply_path.with_name(f"{ply_path.stem}_compare_{timestamp}.png")
    fig_compare.savefig(compare_png, dpi=150)
    os.startfile(compare_png)
    print(f"Comparison PNG saved: {compare_png}")

    # ------------------------------------------------------------------
    # 12. Emit distortion PNG (plot_distortion_panels handles N columns)
    # ------------------------------------------------------------------
    fig_distortion = plot_distortion_panels(V, F, uvs, center_vertex_idx=center_vid)
    fig_distortion.tight_layout()
    distortion_png = ply_path.with_name(
        f"{ply_path.stem}_compare_{timestamp}_distortion.png"
    )
    fig_distortion.savefig(distortion_png, dpi=150)
    os.startfile(distortion_png)
    print(f"Distortion PNG saved: {distortion_png}")
