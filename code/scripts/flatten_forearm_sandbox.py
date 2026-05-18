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

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Qt5Agg")

import numpy as np
import open3d as o3d
import pyvista as pv
import trimesh
import trimesh.repair
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget,
)
from pyvistaqt import QtInteractor

# ---------------------------------------------------------------------------
# Extend sys.path so pipeline helpers can be imported without installing
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent   # …/social-touch-semi-controlled
_SRC = _REPO_ROOT / "code" / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analysis.receptive_field_mapping._slim_helpers import (  # noqa: E402
    clean_mesh,
    boundary_loop,
    canonicalise_uv,
    compute_face_distortion,
)

# ---------------------------------------------------------------------------
# User-editable constants — edit these before running
# ---------------------------------------------------------------------------

#: Absolute or relative path to the forearm segmentation PLY.
#: Must be set by the user before running this script.
PLY_PATH: str = r"F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/02_data/semi-controlled/3_merged/2022-06-17_ST16-05/2022-06-17_ST16-05_forearm.ply"

#: Mesh construction method: ``"bpa"`` (Ball-Pivoting Algorithm, recommended)
#: or ``"delaunay"`` (fast 2.5-D triangulation, no caching).
MESH_METHOD: str = "bpa"

#: Center point for center-weighted flattening.  Three modes:
#:
#: * ``None``            — skip the picker; no center weighting applied.
#: * ``"interactive"``   — open the interactive PyVista picker window so the
#:                         user can Ctrl+click the desired center vertex.
#: * ``(x, y, z)`` tuple — use this hard-coded 3D coordinate directly,
#:                         bypassing the picker (useful for reproducible runs:
#:                         copy-paste the coordinate printed by the interactive
#:                         mode).
CENTER_POINT: tuple[float, float, float] | str | None = "interactive"


# ---------------------------------------------------------------------------
# Phase 1 — load, mesh, clean
# ---------------------------------------------------------------------------

def load_pcd(path: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a PLY point cloud and return points and optional vertex colors.

    Parameters
    ----------
    path:
        Path to the PLY file.

    Returns
    -------
    points : np.ndarray
        Shape (N, 3), dtype float64.
    colors : np.ndarray or None
        Shape (N, 3), dtype float64, values in [0, 1].  ``None`` if the
        PLY has no vertex colors.

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
    colors = np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors() else None
    return points, colors


class _CenterPickerWindow(QWidget):
    """Qt window with a 3D point-cloud view, Ctrl+click picking, and
    vertex-ID text entry.

    Two ways to select a vertex:
    - **Ctrl+click** on the point cloud in the 3D view.
    - Type a vertex index in the text field and click **Show**.

    Both methods place a red sphere on the selected vertex.  Click
    **Confirm** to accept.  Closing the window without confirming leaves
    ``result`` as ``None``.
    """

    def __init__(self, points: np.ndarray, colors: np.ndarray | None):
        import vtk as _vtk
        from scipy.spatial import cKDTree

        super().__init__()
        self.points = points
        self.result: np.ndarray | None = None
        self._current_idx: int | None = None
        self._tree = cKDTree(points)

        self.setWindowTitle("Select Center Point")
        self.resize(900, 650)

        root = QVBoxLayout(self)

        # -- 3D view --
        self.plotter = QtInteractor(self)
        cloud = pv.PolyData(points)
        if colors is not None:
            cloud["RGB"] = (colors * 255).astype(np.uint8)
            self.plotter.add_mesh(
                cloud, name="forearm_cloud", scalars="RGB", rgb=True,
                point_size=4, render_points_as_spheres=True,
            )
        else:
            self.plotter.add_mesh(
                cloud, name="forearm_cloud", color="lightblue",
                point_size=4, render_points_as_spheres=True,
            )
        root.addWidget(self.plotter.interactor, stretch=1)

        # -- Ctrl+click picking via vtkPointPicker --
        self._vtk_picker = _vtk.vtkPointPicker()
        self._vtk_picker.SetTolerance(0.025)
        self.plotter.iren.interactor.AddObserver(
            "LeftButtonPressEvent", self._on_ctrl_click, 1.0,
        )

        # -- Controls row --
        row = QHBoxLayout()
        row.addWidget(QLabel(f"Vertex ID (0–{len(points) - 1}):"))
        self._id_edit = QLineEdit()
        self._id_edit.setText("8000")
        row.addWidget(self._id_edit)

        self._show_btn = QPushButton("Show")
        self._show_btn.clicked.connect(self._on_show)
        row.addWidget(self._show_btn)

        self._confirm_btn = QPushButton("Confirm")
        self._confirm_btn.setEnabled(False)
        self._confirm_btn.clicked.connect(self._on_confirm)
        row.addWidget(self._confirm_btn)

        self._status = QLabel("Ctrl+Click on the cloud or type a vertex ID.")
        row.addWidget(self._status, stretch=1)
        root.addLayout(row)

    # -- Ctrl+click handler ---------------------------------------------------

    def _on_ctrl_click(self, interactor, _event):
        if not interactor.GetControlKey():
            return

        x, y = interactor.GetEventPosition()
        self._vtk_picker.Pick(x, y, 0, self.plotter.renderer)

        if self._vtk_picker.GetPointId() < 0:
            return

        pick_pos = np.array(self._vtk_picker.GetPickPosition())
        _, closest_idx = self._tree.query(pick_pos)
        self._select_vertex(int(closest_idx))

    # -- Text-entry handler ---------------------------------------------------

    def _on_show(self):
        text = self._id_edit.text().strip()
        try:
            idx = int(text)
        except ValueError:
            self._status.setText(f"Invalid input: '{text}' — enter an integer.")
            return
        if idx < 0 or idx >= len(self.points):
            self._status.setText(f"Out of range: must be 0–{len(self.points) - 1}.")
            return
        self._select_vertex(idx)

    # -- Shared selection logic -----------------------------------------------

    def _select_vertex(self, idx: int):
        coord = self.points[idx]
        self._current_idx = idx
        self._id_edit.setText(str(idx))
        self.plotter.add_mesh(
            pv.Sphere(radius=4.0, center=coord.tolist()),
            color="red", name="center_pick_sphere",
        )
        self.plotter.render()
        self._confirm_btn.setEnabled(True)
        self._status.setText(
            f"Vertex {idx}: ({coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.6f})"
        )

    def _on_confirm(self):
        if self._current_idx is not None:
            self.result = self.points[self._current_idx].copy()
        self.close()

    def closeEvent(self, event):
        self.plotter.close()
        super().closeEvent(event)


def pick_center_point(
    points: np.ndarray,
    colors: np.ndarray | None,
) -> np.ndarray:
    """Open an interactive PyVista+Qt window and return the selected vertex.

    The user enters a vertex index in a text field, clicks **Show** to
    preview it as a red sphere on the point cloud, then clicks **Confirm**
    to accept.

    Parameters
    ----------
    points:
        Shape (N, 3), dtype float64.  The forearm point cloud vertices.
    colors:
        Shape (N, 3), dtype float64, values in [0, 1], or ``None`` if the
        PLY has no vertex colors.

    Returns
    -------
    np.ndarray
        Shape (3,), dtype float64 — the 3D coordinate of the selected vertex.

    Raises
    ------
    RuntimeError
        If the user closes the window without confirming a point.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    win = _CenterPickerWindow(points, colors)
    win.show()
    app.exec_()

    if win.result is None:
        raise RuntimeError(
            "No point was confirmed. Run the script again and enter a vertex "
            "ID, click Show, then click Confirm."
        )

    coord = win.result
    print(f"Picked center point: ({coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.6f})")
    return np.array(coord, dtype=np.float64)


def show_mesh_inspector(
    V: np.ndarray,
    F: np.ndarray,
    vertex_colors: np.ndarray | None,
    center_vid: int | None,
) -> None:
    """Open a standalone PyVista window for interactive 3D mesh inspection.

    Blocks until the user closes the window.
    """
    # Ensure a QApplication exists (the picker may have already created one).
    _ = QApplication.instance() or QApplication(sys.argv)

    # PyVista expects face arrays in [n, v0, v1, v2, n, v0, v1, v2, ...] format.
    faces_pv = np.column_stack(
        [np.full(len(F), 3, dtype=np.int64), F.astype(np.int64)]
    ).ravel()
    mesh = pv.PolyData(V, faces_pv)

    plotter = pv.Plotter(title="Forearm mesh — 3D inspection")
    if vertex_colors is not None:
        mesh["RGB"] = (vertex_colors * 255).astype(np.uint8)
        plotter.add_mesh(mesh, scalars="RGB", rgb=True, show_edges=False)
    else:
        plotter.add_mesh(mesh, color="lightblue", show_edges=False)

    if center_vid is not None:
        c = V[center_vid]
        # Sphere radius scaled to ~1% of mesh diagonal for visibility.
        diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
        plotter.add_mesh(
            pv.Sphere(radius=diag * 0.01, center=c.tolist()), color="red"
        )

    plotter.show()


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
        points, _ = load_pcd(str(ply_path))

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


# ---------------------------------------------------------------------------
# Phase 2 — flattening baselines
# ---------------------------------------------------------------------------

import igl


def flatten_lscm(
    V: np.ndarray,
    F: np.ndarray,
    boundary: np.ndarray,
    center_vid: int | None = None,
) -> np.ndarray:
    """Least-Squares Conformal Map with two pinned vertices.

    When ``center_vid`` is given, pin it to UV origin ``(0, 0)`` and
    ``boundary[0]`` to ``(1, 0)``. Otherwise pin two opposite boundary
    vertices (legacy behaviour).
    """
    if center_vid is not None:
        # Pin centre→(0,0) for translation gauge, boundary[0]→(1,0) for scale/rotation gauge.
        b = np.array([int(center_vid), int(boundary[0])], dtype=np.int64)
    else:
        b = np.array(
            [int(boundary[0]), int(boundary[len(boundary) // 2])], dtype=np.int64
        )
    bc = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    uv, _ = igl.lscm(V, F, b, bc)
    if np.any(np.isnan(uv)):
        raise RuntimeError("LSCM produced NaN values — mesh may be degenerate.")
    uv = uv.astype(np.float64)

    if center_vid is not None:
        # LSCM already pins these by construction; reapply for layout symmetry
        # with the other methods (centre at origin, boundary[0] on +x).
        uv = canonicalise_uv(uv, int(center_vid), int(boundary[0]))

    return uv


def flatten_harmonic(
    V: np.ndarray,
    F: np.ndarray,
    boundary: np.ndarray,
    center_vid: int | None = None,
) -> np.ndarray:
    """Harmonic map with boundary vertices uniformly distributed on the unit circle.

    When ``center_vid`` is given, the centre is **not** added as a Dirichlet
    constraint — the Tutte / Rado-Kneser-Choquet bijectivity guarantee only
    holds when interior vertices are free.  The centre is placed at the UV
    origin via a rigid post-processing similarity transform instead.
    """
    n_b = len(boundary)
    angles = np.linspace(0.0, 2.0 * np.pi, n_b, endpoint=False)
    boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)])

    b = boundary.astype(np.int32)
    bc = boundary_uv

    uv = igl.harmonic(V, F, b, bc, 1)
    if np.any(np.isnan(uv)):
        raise RuntimeError("Harmonic map produced NaN values — mesh may be degenerate.")
    uv = uv.astype(np.float64)

    if center_vid is not None:
        if int(center_vid) in set(int(v) for v in boundary):
            raise ValueError(
                f"center_vid {int(center_vid)} is on the mesh boundary — "
                "it must be an interior vertex for centre-aligned flattening."
            )
        uv = canonicalise_uv(uv, int(center_vid), int(boundary[0]))

    return uv


def flatten_arap(
    V: np.ndarray,
    F: np.ndarray,
    boundary: np.ndarray,
    center_vid: int | None = None,
) -> np.ndarray:
    """As-Rigid-As-Possible flattening, initialised from the harmonic map.

    Only boundary vertices are pinned (to the unit circle).  When
    ``center_vid`` is given, the centre is moved to the UV origin via a
    rigid similarity transform after the solve — the same reasoning as
    ``flatten_harmonic``: pinning an interior vertex inside the solve
    breaks the bijectivity guarantee and induces fold-overs.
    """
    n_b = len(boundary)
    angles = np.linspace(0.0, 2.0 * np.pi, n_b, endpoint=False)
    boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)])

    if center_vid is not None and int(center_vid) in set(int(v) for v in boundary):
        raise ValueError(
            f"center_vid {int(center_vid)} is on the mesh boundary — "
            "it must be an interior vertex for centre-aligned flattening."
        )

    pin_idx = boundary.astype(np.int32)
    bc = boundary_uv

    # Use boundary-only harmonic init (no centre pin) so the initial guess
    # is itself flip-free per Rado-Kneser-Choquet.
    uv = flatten_harmonic(V, F, boundary, center_vid=None)

    data = igl.ARAPData()
    igl.arap_precomputation(V, F.astype(np.int64), 2, pin_idx, data)
    for _ in range(20):
        uv_new = igl.arap_solve(bc, data, uv)
        rel_change = np.linalg.norm(uv_new - uv) / (np.linalg.norm(uv) + 1e-12)
        uv = uv_new
        if rel_change < 1e-6:
            break

    if np.any(np.isnan(uv)):
        raise RuntimeError("ARAP produced NaN values — mesh may be degenerate.")
    uv = uv.astype(np.float64)

    if center_vid is not None:
        uv = canonicalise_uv(uv, int(center_vid), int(boundary[0]))

    return uv


# ---------------------------------------------------------------------------
# Phase 3 — visualisation + entry point
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.tri
from matplotlib.collections import PolyCollection


def plot_panels(
    V: np.ndarray,
    F: np.ndarray,
    uv_lscm: np.ndarray,
    uv_arap: np.ndarray,
    uv_harmonic: np.ndarray,
    colors: np.ndarray | None = None,
    center_vertex_idx: int | None = None,
) -> plt.Figure:
    """Render the 3D mesh and three 2D flattenings in a single figure."""
    from datetime import datetime

    fig = plt.figure(figsize=(16, 4))

    # Per-face colors: average the three vertex colors for each triangle.
    if colors is not None:
        face_colors = colors[F].mean(axis=1)
    else:
        face_colors = None

    # -- 3D input mesh --------------------------------------------------------
    ax3d = fig.add_subplot(141, projection="3d")
    surf = ax3d.plot_trisurf(
        V[:, 0], V[:, 1], V[:, 2],
        triangles=F,
        edgecolor="none",
        shade=False,
    )
    if face_colors is not None:
        surf.set_facecolors(face_colors)
    else:
        surf.set_facecolors("steelblue")
    if center_vertex_idx is not None:
        c = V[center_vertex_idx]
        ax3d.scatter(c[0], c[1], c[2], color="red", s=60, zorder=10)
    ax3d.set_title("Input mesh (3D)")

    # -- 2D flattening panels -------------------------------------------------
    panels = [
        (142, uv_lscm, "LSCM"),
        (143, uv_arap, "ARAP"),
        (144, uv_harmonic, "Harmonic"),
    ]
    for subplot_id, uv, title in panels:
        ax = fig.add_subplot(subplot_id)
        if face_colors is not None:
            polys = uv[F]  # (M, 3, 2)
            pc = PolyCollection(polys, facecolors=face_colors, edgecolors="none")
            ax.add_collection(pc)
            ax.autoscale_view()
        else:
            tri = matplotlib.tri.Triangulation(uv[:, 0], uv[:, 1], F)
            ax.triplot(tri, color="steelblue", linewidth=0.4)
        if center_vertex_idx is not None:
            cx, cy = uv[center_vertex_idx]
            ax.plot(cx, cy, "o", color="red", markersize=6, zorder=10)
        ax.set_title(title)
        ax.set_aspect("equal")

    fig.text(
        0.99, 0.01, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ha="right", va="bottom", fontsize=7, color="gray",
    )

    return fig


def plot_distortion_panels(
    V: np.ndarray,
    F: np.ndarray,
    uvs: dict[str, np.ndarray],
    center_vertex_idx: int | None = None,
) -> plt.Figure:
    """Render per-face conformal and area distortion for each flattening method.

    Layout: 2 rows (conformal, area) x N columns (one per method).
    """
    from datetime import datetime

    methods = list(uvs.keys())
    n = len(methods)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))
    if n == 1:
        axes = axes[:, np.newaxis]

    for col, name in enumerate(methods):
        uv = uvs[name]
        conformal, area_dist = compute_face_distortion(V, F, uv)
        polys = uv[F]

        ax = axes[0, col]
        pc = PolyCollection(polys, array=conformal, cmap="YlOrRd", edgecolors="none")
        pc.set_clim(1.0, max(np.percentile(conformal, 95), 1.01))
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_title(f"{name} — conformal (σ₁/σ₂)")
        fig.colorbar(pc, ax=ax)
        if center_vertex_idx is not None:
            ax.plot(*uv[center_vertex_idx], "o", color="blue", markersize=4, zorder=10)

        ax = axes[1, col]
        vmax = max(abs(np.percentile(area_dist, 5)), abs(np.percentile(area_dist, 95)), 0.01)
        pc = PolyCollection(polys, array=area_dist, cmap="RdBu_r", edgecolors="none")
        pc.set_clim(-vmax, vmax)
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_title(f"{name} — area (log₂ scale)")
        fig.colorbar(pc, ax=ax)
        if center_vertex_idx is not None:
            ax.plot(*uv[center_vertex_idx], "o", color="blue", markersize=4, zorder=10)

    fig.text(
        0.99, 0.01, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ha="right", va="bottom", fontsize=7, color="gray",
    )

    return fig


if __name__ == "__main__":
    if not PLY_PATH:
        raise ValueError(
            "PLY_PATH is not set. "
            "Edit the PLY_PATH constant at the top of this script."
        )

    ply_path = Path(PLY_PATH)
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
    # center_3d is resolved to a cleaned-mesh vertex index below and passed to flatten_*().

    raw_mesh = build_mesh(ply_path, MESH_METHOD)
    V_raw, F_raw = clean_mesh(raw_mesh)

    # Map original PLY vertex colors onto the cleaned (reindexed) vertices.
    if orig_colors is not None:
        from scipy.spatial import cKDTree
        _, idx = cKDTree(orig_points).query(V_raw)
        vertex_colors = orig_colors[idx]
    else:
        vertex_colors = None

    # Resolve picked 3D point to nearest cleaned-mesh vertex index.
    if center_3d is not None:
        from scipy.spatial import cKDTree as _cKDTree
        _, center_vid = _cKDTree(V_raw).query(center_3d)
        center_vid = int(center_vid)
    else:
        center_vid = None

    boundary = boundary_loop(F_raw)
    uv_lscm = flatten_lscm(V_raw, F_raw, boundary, center_vid=center_vid)
    uv_harmonic = flatten_harmonic(V_raw, F_raw, boundary, center_vid=center_vid)
    uv_arap = flatten_arap(V_raw, F_raw, boundary, center_vid=center_vid)

    fig = plot_panels(V_raw, F_raw, uv_lscm, uv_arap, uv_harmonic, vertex_colors, center_vid)
    fig.tight_layout()
    from datetime import datetime as _dt
    timestamp = _dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = ply_path.with_name(f"{ply_path.stem}_flattening_{timestamp}.png")
    fig.savefig(out, dpi=150)
    os.startfile(out)

    distortion_fig = plot_distortion_panels(
        V_raw, F_raw,
        {"LSCM": uv_lscm, "ARAP": uv_arap, "Harmonic": uv_harmonic},
        center_vertex_idx=center_vid,
    )
    distortion_fig.tight_layout()
    distortion_out = ply_path.with_name(
        f"{ply_path.stem}_flattening_distortion_{timestamp}.png"
    )
    distortion_fig.savefig(distortion_out, dpi=150)
    os.startfile(distortion_out)

    show_mesh_inspector(V_raw, F_raw, vertex_colors, center_vid)
