"""
_slim_qc_figures.py
===================

Diagnostic PNG figures for the SLIM UV precompute step.

Saved next to the ``.npz`` cache at 300 DPI so the researcher can visually
verify the UV map and distortion before running downstream RF heatmap tasks.

Uses the matplotlib OOP interface (``Figure`` + ``FigureCanvasAgg``) so that
no interactive backend is initialised — safe inside Prefect, tests, or any
context that does not have a Qt/Tk display available.

Public API
----------
save_slim_qc_figures           — Write both QC figures and return their paths.
plot_slim_uv_panel             — 3D mesh + SLIM UV layout (1 × 2).
plot_slim_distortion_panel     — Conformal + area distortion (2 × 1).
save_slim_diagnostic_figures   — Write 6 step-numbered diagnostic PNGs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib.tri
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers "3d" projection

from .slim_helpers import compute_face_distortion, _find_boundary_loops


# ---------------------------------------------------------------------------
# Individual figure builders
# ---------------------------------------------------------------------------

def plot_slim_uv_panel(
    V: np.ndarray,
    F: np.ndarray,
    uv: np.ndarray,
    center_vid: int,
) -> Figure:
    """1 × 2 figure: 3D mesh (left) and SLIM UV flattening (right)."""
    fig = Figure(figsize=(14, 7))
    FigureCanvasAgg(fig)

    # Left — 3D mesh
    ax3d = fig.add_subplot(121, projection="3d")
    surf = ax3d.plot_trisurf(
        V[:, 0], V[:, 1], V[:, 2],
        triangles=F,
        edgecolor="none",
        shade=True,
    )
    surf.set_facecolors("steelblue")
    c3d = V[center_vid]
    ax3d.scatter(c3d[0], c3d[1], c3d[2], color="red", s=60, zorder=10)
    ax3d.set_title("Forearm mesh (3D)")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    # Right — SLIM UV
    ax2d = fig.add_subplot(122)
    polys = uv[F]
    pc = PolyCollection(polys, facecolors="steelblue", edgecolors="none", alpha=0.6)
    ax2d.add_collection(pc)
    tri = matplotlib.tri.Triangulation(uv[:, 0], uv[:, 1], F)
    ax2d.triplot(tri, color="white", linewidth=0.25, alpha=0.5)
    ax2d.plot(*uv[center_vid], "o", color="red", markersize=6, zorder=10,
              label=f"centre (vid={center_vid})")
    ax2d.autoscale_view()
    ax2d.set_aspect("equal")
    ax2d.legend(fontsize=8, loc="upper right")
    ax2d.set_title("SLIM UV flattening")
    ax2d.set_xlabel("u")
    ax2d.set_ylabel("v")

    fig.text(
        0.99, 0.01, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ha="right", va="bottom", fontsize=7, color="gray",
    )

    return fig


def plot_slim_distortion_panel(
    V: np.ndarray,
    F: np.ndarray,
    uv: np.ndarray,
    center_vid: int,
) -> Figure:
    """2 × 1 figure: conformal distortion (top) and area distortion (bottom)."""
    conformal, area_dist = compute_face_distortion(V, F, uv)
    polys = uv[F]

    fig = Figure(figsize=(8, 12))
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 1)

    # Top — conformal (σ₁/σ₂)
    ax = axes[0]
    clim_high = max(np.percentile(conformal, 95), 1.01)
    pc = PolyCollection(polys, array=conformal, cmap="YlOrRd", edgecolors="none")
    pc.set_clim(1.0, clim_high)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title("Conformal distortion (σ₁/σ₂) — 1 = perfect")
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    fig.colorbar(pc, ax=ax)
    ax.plot(*uv[center_vid], "o", color="blue", markersize=5, zorder=10,
            label=f"centre (vid={center_vid})")
    ax.legend(fontsize=8, loc="upper right")

    # Bottom — area (log₂ scale)
    ax = axes[1]
    vmax = max(abs(np.percentile(area_dist, 5)), abs(np.percentile(area_dist, 95)), 0.01)
    pc = PolyCollection(polys, array=area_dist, cmap="RdBu_r", edgecolors="none")
    pc.set_clim(-vmax, vmax)
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_title("Area distortion (log₂ scale) — 0 = median area")
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    fig.colorbar(pc, ax=ax)
    ax.plot(*uv[center_vid], "o", color="blue", markersize=5, zorder=10,
            label=f"centre (vid={center_vid})")
    ax.legend(fontsize=8, loc="upper right")

    fig.text(
        0.99, 0.01, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ha="right", va="bottom", fontsize=7, color="gray",
    )

    return fig


# ---------------------------------------------------------------------------
# Combined save entry point
# ---------------------------------------------------------------------------

def save_slim_qc_figures(
    V: np.ndarray,
    F: np.ndarray,
    uv: np.ndarray,
    center_vid: int,
    cache_path: Path,
) -> tuple[Path, Path]:
    """Save the UV and distortion QC figures at 300 DPI next to *cache_path*.

    Parameters
    ----------
    V, F:
        Cleaned mesh vertices and face indices (from the SLIM precompute).
    uv:
        SLIM UV coordinates, shape (N_v, 2).
    center_vid:
        Interior vertex placed at the UV origin.
    cache_path:
        Path to the ``.npz`` cache file; figures are saved alongside it.

    Returns
    -------
    (qc_path, distortion_path)
        Paths to the two written PNG files.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    qc_path = cache_path.with_name(f"{cache_path.stem}_qc_{timestamp}.png")
    dist_path = cache_path.with_name(f"{cache_path.stem}_distortion_{timestamp}.png")

    fig_uv = plot_slim_uv_panel(V, F, uv, center_vid)
    fig_uv.tight_layout()
    fig_uv.savefig(qc_path, dpi=300)

    fig_dist = plot_slim_distortion_panel(V, F, uv, center_vid)
    fig_dist.tight_layout()
    fig_dist.savefig(dist_path, dpi=300)

    return qc_path, dist_path


# ---------------------------------------------------------------------------
# Step-by-step diagnostic figure helpers
# ---------------------------------------------------------------------------

def _compute_skin_normal_view(V: np.ndarray, F: np.ndarray) -> tuple[float, float]:
    """Compute area-weighted mean face normal and return matplotlib (elev, azim)."""
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]
    crosses = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(crosses, axis=1, keepdims=True)
    weighted = crosses * areas
    mean_normal = weighted.sum(axis=0)
    norm = np.linalg.norm(mean_normal)
    if norm < 1e-15:
        raise ValueError(
            "_compute_skin_normal_view: area-weighted mean normal is zero — "
            "the mesh may be degenerate or have cancelling faces."
        )
    mean_normal = mean_normal / norm
    elev = float(np.arcsin(np.clip(mean_normal[2], -1.0, 1.0)) * 180.0 / np.pi)
    azim = float(np.arctan2(mean_normal[1], mean_normal[0]) * 180.0 / np.pi)
    return elev, azim


def _rotation_matrix_to_elev_azim(R: np.ndarray) -> tuple[float, float]:
    """Convert a 3×3 camera rotation matrix to matplotlib (elev, azim).

    R is from camera_settings_to_rotation(): R[2] is the view direction
    (camera → scene).  Matplotlib elev/azim describe the viewer position,
    so we negate R[2] to point from scene toward viewer.
    """
    view_dir = -R[2]
    elev = float(np.arcsin(np.clip(view_dir[2], -1.0, 1.0)) * 180.0 / np.pi)
    azim = float(np.arctan2(view_dir[1], view_dir[0]) * 180.0 / np.pi)
    return elev, azim


def _plot_mesh_3d(
    ax: Axes3D,
    V: np.ndarray,
    F: np.ndarray,
    elev: float,
    azim: float,
    title: str,
    overlay_fn: Callable[[Axes3D], None] | None = None,
) -> None:
    """Render a triangular mesh into *ax* from the given view angle."""
    ax.plot_trisurf(
        V[:, 0], V[:, 1], V[:, 2],
        triangles=F,
        color="lightsteelblue",
        edgecolor="none",
        alpha=0.85,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    if overlay_fn is not None:
        overlay_fn(ax)


def _plot_step1_raw_mesh(
    V_raw: np.ndarray,
    F_raw: np.ndarray,
    elev: float,
    azim: float,
    cam_elev: float,
    cam_azim: float,
    cam_label: str,
) -> Figure:
    """Step 1 — raw BPA mesh: skin-normal view (left) + camera view (right)."""
    fig = Figure(figsize=(10, 4.5))
    FigureCanvasAgg(fig)
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")
    _plot_mesh_3d(ax_left, V_raw, F_raw, elev, azim, "Step 1 — raw mesh (skin-normal view)")
    _plot_mesh_3d(ax_right, V_raw, F_raw, cam_elev, cam_azim, f"Step 1 — raw mesh ({cam_label})")
    fig.tight_layout()
    return fig


def _plot_step2_cleaned_mesh(
    V: np.ndarray,
    F: np.ndarray,
    elev: float,
    azim: float,
    cam_elev: float,
    cam_azim: float,
    cam_label: str,
) -> Figure:
    """Step 2 — cleaned mesh: dual view with vertex/face/boundary-loop annotations."""
    n_loops = len(_find_boundary_loops(F))
    ann = f"V={V.shape[0]:,}  F={F.shape[0]:,}  boundary loops={n_loops}"

    fig = Figure(figsize=(10, 4.5))
    FigureCanvasAgg(fig)
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")
    _plot_mesh_3d(ax_left, V, F, elev, azim, "Step 2 — cleaned mesh (skin-normal view)")
    _plot_mesh_3d(ax_right, V, F, cam_elev, cam_azim, f"Step 2 — cleaned mesh ({cam_label})")
    fig.text(0.5, 0.02, ann, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def _plot_step3_centroid_boundary(
    V: np.ndarray,
    F: np.ndarray,
    center_vid: int,
    boundary: np.ndarray,
    elev: float,
    azim: float,
    cam_elev: float,
    cam_azim: float,
    cam_label: str,
) -> Figure:
    """Step 3 — centroid (red) and boundary loop (yellow) overlays."""
    def _overlay(ax: Axes3D) -> None:
        c = V[center_vid]
        ax.scatter(c[0], c[1], c[2], color="red", s=40, zorder=10, depthshade=False)
        bv = V[boundary]
        ax.scatter(bv[:, 0], bv[:, 1], bv[:, 2], color="gold", s=4, zorder=9, depthshade=False)

    fig = Figure(figsize=(10, 4.5))
    FigureCanvasAgg(fig)
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")
    _plot_mesh_3d(
        ax_left, V, F, elev, azim,
        "Step 3 — centroid + boundary (skin-normal view)",
        overlay_fn=_overlay,
    )
    _plot_mesh_3d(
        ax_right, V, F, cam_elev, cam_azim,
        f"Step 3 — centroid + boundary ({cam_label})",
        overlay_fn=_overlay,
    )
    fig.tight_layout()
    return fig


def _plot_step4_uv_init(
    V: np.ndarray,
    F: np.ndarray,
    uv_init: np.ndarray,
    init_method: str,
    elev: float,
    azim: float,
) -> Figure:
    """Step 4 — UV initialisation: 3D mesh coloured by init-u (left) + 2D UV scatter (right)."""
    p0, p1, p2 = uv_init[F[:, 0]], uv_init[F[:, 1]], uv_init[F[:, 2]]
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (
        p1[:, 1] - p0[:, 1]
    ) * (p2[:, 0] - p0[:, 0])
    n_flipped = int(np.sum(cross < 0) if np.any(cross > 0) else np.sum(cross > 0))

    fig = Figure(figsize=(10, 4.5))
    FigureCanvasAgg(fig)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    u_vals = uv_init[:, 0]
    u_norm = (u_vals - u_vals.min()) / max(float(u_vals.max() - u_vals.min()), 1e-15)
    face_u = u_norm[F].mean(axis=1)
    surf = ax3d.plot_trisurf(
        V[:, 0], V[:, 1], V[:, 2],
        triangles=F,
        edgecolor="none",
        alpha=0.9,
    )
    surf.set_array(face_u)
    surf.set_cmap("viridis")
    ax3d.view_init(elev=elev, azim=azim)
    ax3d.set_title("Step 4 — mesh coloured by init-u", fontsize=9)
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])

    polys_2d = uv_init[F]
    pc2d = PolyCollection(polys_2d, array=face_u, cmap="viridis", edgecolors="white", linewidths=0.15)
    ax2d.add_collection(pc2d)
    tri_2d = matplotlib.tri.Triangulation(uv_init[:, 0], uv_init[:, 1], F)
    ax2d.triplot(tri_2d, color="white", linewidth=0.15, alpha=0.4)
    ax2d.autoscale_view()
    ax2d.set_aspect("equal")
    ax2d.set_title("Step 4 — UV init (2D)", fontsize=9)
    ax2d.set_xlabel("u")
    ax2d.set_ylabel("v")

    ann = f"init method: {init_method}   flipped faces: {n_flipped}"
    fig.text(0.5, 0.02, ann, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def _plot_step5_slim_final(
    uv_init: np.ndarray,
    F: np.ndarray,
    uv_final: np.ndarray,
    trimmed: bool,
) -> Figure:
    """Step 5 — init UV (left) vs final SLIM UV (right), both colormapped meshes."""
    fig = Figure(figsize=(10, 4.5))
    FigureCanvasAgg(fig)
    ax_left = fig.add_subplot(1, 2, 1)
    ax_right = fig.add_subplot(1, 2, 2)

    u_vals = uv_init[:, 0]
    u_norm = (u_vals - u_vals.min()) / max(float(u_vals.max() - u_vals.min()), 1e-15)
    face_u = u_norm[F].mean(axis=1)

    polys_init = uv_init[F]
    pc_left = PolyCollection(polys_init, array=face_u, cmap="viridis", edgecolors="white", linewidths=0.15)
    ax_left.add_collection(pc_left)
    tri_init = matplotlib.tri.Triangulation(uv_init[:, 0], uv_init[:, 1], F)
    ax_left.triplot(tri_init, color="white", linewidth=0.15, alpha=0.4)
    ax_left.autoscale_view()
    ax_left.set_aspect("equal")
    ax_left.set_title("Step 5 — UV init", fontsize=9)
    ax_left.set_xlabel("u")
    ax_left.set_ylabel("v")

    polys_final = uv_final[F]
    pc_right = PolyCollection(polys_final, array=face_u, cmap="viridis", edgecolors="white", linewidths=0.15)
    ax_right.add_collection(pc_right)
    tri_final = matplotlib.tri.Triangulation(uv_final[:, 0], uv_final[:, 1], F)
    ax_right.triplot(tri_final, color="white", linewidth=0.15, alpha=0.4)
    ax_right.autoscale_view()
    ax_right.set_aspect("equal")
    ax_right.set_title("Step 5 — SLIM UV final", fontsize=9)
    ax_right.set_xlabel("u")
    ax_right.set_ylabel("v")

    trimmed_label = "mesh trimmed: yes" if trimmed else "mesh trimmed: no"
    fig.text(0.5, 0.02, trimmed_label, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


# ---------------------------------------------------------------------------
# Diagnostic orchestrator
# ---------------------------------------------------------------------------

def save_slim_diagnostic_figures(
    out_dir: Path,
    session_id: str,
    V_raw: np.ndarray,
    F_raw: np.ndarray,
    V: np.ndarray,
    F: np.ndarray,
    center_vid: int,
    boundary: np.ndarray,
    slim_diag: dict,
    uv_final: np.ndarray,
    cam_settings: dict | None,
) -> list[Path]:
    """Build and save 6 step-numbered diagnostic PNGs.

    Parameters
    ----------
    out_dir:
        Parent directory; a ``diagnostics/`` subdirectory is created inside.
    session_id:
        Session identifier used only for logging context.
    V_raw, F_raw:
        Raw BPA mesh vertices and faces before cleaning.
    V, F:
        Cleaned mesh vertices and faces (from clean_mesh).
    center_vid:
        Centroid vertex index into V (the cleaned mesh).
    boundary:
        Boundary vertex indices into V (the cleaned mesh).
    slim_diag:
        Dict with keys ``"init_uv"``, ``"init_method"``, ``"trimmed"``,
        ``"V_trimmed"``, ``"F_trimmed"`` populated by flatten_slim() with
        diagnostics enabled.
    uv_final:
        Final SLIM UV coordinates, shape (N_v, 2).
    cam_settings:
        RF camera settings dict (keys: camera_position, focal_point,
        up_vector) or None when unavailable.

    Returns
    -------
    List of Paths to the 6 written PNG files, in step order.
    """
    diag_dir = out_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    elev, azim = _compute_skin_normal_view(V, F)

    if cam_settings is not None:
        from .tangent_plane_alignment import camera_settings_to_rotation
        R = camera_settings_to_rotation(cam_settings)
        cam_elev, cam_azim = _rotation_matrix_to_elev_azim(R)
        cam_label = "camera view"
    else:
        cam_elev, cam_azim = 30.0, -60.0
        cam_label = "no camera settings"

    raw_elev, raw_azim = _compute_skin_normal_view(V_raw, F_raw)

    init_uv: np.ndarray = slim_diag["init_uv"]
    init_method: str = slim_diag["init_method"]
    trimmed: bool = slim_diag["trimmed"]
    V_trim: np.ndarray = slim_diag["V_trimmed"]
    F_trim: np.ndarray = slim_diag["F_trimmed"]

    paths: list[Path] = []

    fig1 = _plot_step1_raw_mesh(V_raw, F_raw, raw_elev, raw_azim, cam_elev, cam_azim, cam_label)
    p1 = diag_dir / "step1_raw_mesh.png"
    fig1.savefig(p1, dpi=200)
    paths.append(p1)

    fig2 = _plot_step2_cleaned_mesh(V, F, elev, azim, cam_elev, cam_azim, cam_label)
    p2 = diag_dir / "step2_cleaned_mesh.png"
    fig2.savefig(p2, dpi=200)
    paths.append(p2)

    fig3 = _plot_step3_centroid_boundary(V, F, center_vid, boundary, elev, azim, cam_elev, cam_azim, cam_label)
    p3 = diag_dir / "step3_centroid_boundary.png"
    fig3.savefig(p3, dpi=200)
    paths.append(p3)

    trim_elev, trim_azim = _compute_skin_normal_view(V_trim, F_trim)
    fig4 = _plot_step4_uv_init(V_trim, F_trim, init_uv, init_method, trim_elev, trim_azim)
    p4 = diag_dir / "step4_uv_initialization.png"
    fig4.savefig(p4, dpi=200)
    paths.append(p4)

    fig5 = _plot_step5_slim_final(init_uv, F_trim, uv_final, trimmed)
    p5 = diag_dir / "step5_slim_uv_final.png"
    fig5.savefig(p5, dpi=200)
    paths.append(p5)

    fig6 = plot_slim_distortion_panel(V_trim, F_trim, uv_final, center_vid)
    fig6.tight_layout()
    p6 = diag_dir / "step6_distortion.png"
    fig6.savefig(p6, dpi=200)
    paths.append(p6)

    return paths
