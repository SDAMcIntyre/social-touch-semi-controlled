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

from .slim_helpers import (
    compute_face_distortion,
    _compute_face_aspect_ratios,
    _find_boundary_loops,
)


# ---------------------------------------------------------------------------
# Individual figure builders
# ---------------------------------------------------------------------------

def plot_slim_uv_panel(
    V: np.ndarray,
    F: np.ndarray,
    uv: np.ndarray,
    center_vid: int,
    vertex_colors: np.ndarray | None = None,
    cam_settings: dict | None = None,
) -> Figure:
    """1 × 2 figure: 3D mesh (left) and SLIM UV flattening (right)."""
    fig = Figure(figsize=(14, 7))
    FigureCanvasAgg(fig)

    face_rgba = vertex_colors[F].mean(axis=1) if vertex_colors is not None else None

    if cam_settings is not None:
        from .tangent_plane_alignment import camera_settings_to_rotation
        R = camera_settings_to_rotation(cam_settings)
        cam_elev, cam_azim = _rotation_matrix_to_elev_azim(R)
    else:
        cam_elev, cam_azim = _compute_skin_normal_view(V, F)

    # Left — 3D mesh
    ax3d = fig.add_subplot(121, projection="3d")
    surf = ax3d.plot_trisurf(
        V[:, 0], V[:, 1], V[:, 2],
        triangles=F,
        edgecolor="none",
        shade=True,
    )
    if face_rgba is not None:
        surf.set_facecolors(face_rgba)
    else:
        surf.set_facecolors("steelblue")
    ax3d.view_init(elev=cam_elev, azim=cam_azim)
    c3d = V[center_vid]
    ax3d.scatter(c3d[0], c3d[1], c3d[2], color="red", s=60, zorder=10)
    ax3d.set_title("Forearm mesh (3D)")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")

    # Right — SLIM UV
    ax2d = fig.add_subplot(122)
    polys = uv[F]
    if face_rgba is not None:
        pc = PolyCollection(polys, facecolors=face_rgba, edgecolors="none", alpha=0.6)
    else:
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
    vertex_colors: np.ndarray | None = None,
    cam_settings: dict | None = None,
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
    vertex_colors:
        Per-vertex RGBA colours, shape (N_v, 4) float [0, 1].  When
        ``None``, figures fall back to flat ``steelblue`` colouring.
    cam_settings:
        RF camera settings dict (keys: camera_position, focal_point,
        up_vector) or ``None``.  When provided the 3D mesh panel uses this
        viewpoint; otherwise a skin-normal view is computed automatically.

    Returns
    -------
    (qc_path, distortion_path)
        Paths to the two written PNG files.
    """
    qc_path = cache_path.with_name(f"{cache_path.stem}_qc.png")
    dist_path = cache_path.with_name(f"{cache_path.stem}_distortion.png")

    fig_uv = plot_slim_uv_panel(V, F, uv, center_vid, vertex_colors=vertex_colors,
                                cam_settings=cam_settings)
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
    edgecolor: str = "none",
    linewidth: float = 0.0,
    vertex_colors: np.ndarray | None = None,
) -> None:
    """Render a triangular mesh into *ax* from the given view angle."""
    surf = ax.plot_trisurf(
        V[:, 0], V[:, 1], V[:, 2],
        triangles=F,
        color="lightsteelblue",
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=0.85,
    )
    if vertex_colors is not None:
        face_colors = vertex_colors[F].mean(axis=1)
        surf.set_facecolors(face_colors)
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
    vertex_colors: np.ndarray | None = None,
) -> Figure:
    """Step 1 — raw BPA mesh: skin-normal view (left) + camera view (right)."""
    fig = Figure(figsize=(10, 4.5))
    FigureCanvasAgg(fig)
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")
    _plot_mesh_3d(ax_left, V_raw, F_raw, elev, azim, "Step 1 — raw mesh (skin-normal view)",
                  vertex_colors=vertex_colors)
    _plot_mesh_3d(ax_right, V_raw, F_raw, cam_elev, cam_azim, f"Step 1 — raw mesh ({cam_label})",
                  vertex_colors=vertex_colors)
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
    vertex_colors: np.ndarray | None = None,
) -> Figure:
    """Step 2 — cleaned mesh: 2x2 layout with wireframe and aspect-ratio heatmap."""
    n_loops = len(_find_boundary_loops(F))
    ar = _compute_face_aspect_ratios(V, F)
    ar_max = float(ar.max())
    ar_p95 = float(np.percentile(ar, 95))
    ar_median = float(np.median(ar))
    n_bad = int(np.sum(ar > 10.0))

    fig = Figure(figsize=(10, 9))
    FigureCanvasAgg(fig)

    ax_tl = fig.add_subplot(2, 2, 1, projection="3d")
    ax_tr = fig.add_subplot(2, 2, 2, projection="3d")
    ax_bl = fig.add_subplot(2, 2, 3)
    ax_br = fig.add_subplot(2, 2, 4)

    _plot_mesh_3d(
        ax_tl, V, F, elev, azim,
        "Step 2 — cleaned mesh (skin-normal view)",
        edgecolor="gray", linewidth=0.15,
        vertex_colors=vertex_colors,
    )
    _plot_mesh_3d(
        ax_tr, V, F, cam_elev, cam_azim,
        f"Step 2 — cleaned mesh ({cam_label})",
        edgecolor="gray", linewidth=0.15,
        vertex_colors=vertex_colors,
    )

    # Bottom-left: face aspect-ratio heatmap (2D UV-space proxy: use XY projection)
    # Project faces to 2D using first two PCA components of V for a flat view.
    V_c = V - V.mean(axis=0)
    _, _, Vt = np.linalg.svd(V_c, full_matrices=False)
    V2d = V_c @ Vt[:2].T          # (N, 2)
    polys2d = V2d[F]               # (M, 3, 2)
    clim_high = max(ar_p95, 1.1)
    pc = PolyCollection(polys2d, array=ar, cmap="YlOrRd", edgecolors="none")
    pc.set_clim(1.0, clim_high)
    ax_bl.add_collection(pc)
    ax_bl.autoscale_view()
    ax_bl.set_aspect("equal")
    ax_bl.set_title("Step 2 — face aspect ratio (longest/shortest edge)", fontsize=8)
    ax_bl.set_xlabel("PCA-1")
    ax_bl.set_ylabel("PCA-2")
    fig.colorbar(pc, ax=ax_bl, label="AR (clipped at 95th pct)")

    # Bottom-right: quality statistics text
    ax_br.axis("off")
    stats_text = (
        f"Mesh statistics\n"
        f"─────────────────────\n"
        f"Vertices      : {V.shape[0]:,}\n"
        f"Faces         : {F.shape[0]:,}\n"
        f"Boundary loops: {n_loops}\n"
        f"\nFace aspect ratio (longest/shortest edge)\n"
        f"─────────────────────\n"
        f"Median AR     : {ar_median:.2f}\n"
        f"95th pct AR   : {ar_p95:.2f}\n"
        f"Max AR        : {ar_max:.2f}\n"
        f"Faces AR > 10 : {n_bad} ({100.0 * n_bad / max(len(ar), 1):.1f}%)\n"
    )
    ax_br.text(
        0.05, 0.95, stats_text,
        transform=ax_br.transAxes,
        fontsize=8, verticalalignment="top", fontfamily="monospace",
    )
    ax_br.set_title("Step 2 — quality statistics", fontsize=8)

    fig.tight_layout()
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
    vertex_colors: np.ndarray | None = None,
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
        vertex_colors=vertex_colors,
    )
    _plot_mesh_3d(
        ax_right, V, F, cam_elev, cam_azim,
        f"Step 3 — centroid + boundary ({cam_label})",
        overlay_fn=_overlay,
        vertex_colors=vertex_colors,
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
    vertex_colors: np.ndarray | None = None,
) -> Figure:
    """Step 4 — UV initialisation: skin colors (top) + viridis init-u (bottom).

    When *vertex_colors* is provided the figure is 2×2: the top row shows the
    mesh with real skin colours (3D + 2D UV) and the bottom row shows the
    viridis init-u analytical view.  Without colours, falls back to the
    original 1×2 viridis-only layout.
    """
    p0, p1, p2 = uv_init[F[:, 0]], uv_init[F[:, 1]], uv_init[F[:, 2]]
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (
        p1[:, 1] - p0[:, 1]
    ) * (p2[:, 0] - p0[:, 0])
    n_flipped = int(np.sum(cross < 0) if np.any(cross > 0) else np.sum(cross > 0))

    u_vals = uv_init[:, 0]
    u_norm = (u_vals - u_vals.min()) / max(float(u_vals.max() - u_vals.min()), 1e-15)
    face_u = u_norm[F].mean(axis=1)
    polys_2d = uv_init[F]

    has_colors = vertex_colors is not None
    fig = Figure(figsize=(10, 9 if has_colors else 4.5))
    FigureCanvasAgg(fig)

    if has_colors:
        face_rgba = vertex_colors[F].mean(axis=1)

        ax_tl = fig.add_subplot(2, 2, 1, projection="3d")
        surf_tl = ax_tl.plot_trisurf(
            V[:, 0], V[:, 1], V[:, 2], triangles=F, edgecolor="none", alpha=0.9,
        )
        surf_tl.set_facecolors(face_rgba)
        ax_tl.view_init(elev=elev, azim=azim)
        ax_tl.set_title("Step 4 — mesh (skin colours)", fontsize=9)
        ax_tl.set_xticks([]); ax_tl.set_yticks([]); ax_tl.set_zticks([])

        ax_tr = fig.add_subplot(2, 2, 2)
        pc_tr = PolyCollection(polys_2d, facecolors=face_rgba, edgecolors="white", linewidths=0.15)
        ax_tr.add_collection(pc_tr)
        ax_tr.autoscale_view()
        ax_tr.set_aspect("equal")
        ax_tr.set_title("Step 4 — UV init (skin colours)", fontsize=9)
        ax_tr.set_xlabel("u"); ax_tr.set_ylabel("v")

        ax_bl = fig.add_subplot(2, 2, 3, projection="3d")
        ax_br = fig.add_subplot(2, 2, 4)
    else:
        ax_bl = fig.add_subplot(1, 2, 1, projection="3d")
        ax_br = fig.add_subplot(1, 2, 2)

    surf_bl = ax_bl.plot_trisurf(
        V[:, 0], V[:, 1], V[:, 2], triangles=F, edgecolor="none", alpha=0.9,
    )
    surf_bl.set_array(face_u)
    surf_bl.set_cmap("viridis")
    ax_bl.view_init(elev=elev, azim=azim)
    ax_bl.set_title("Step 4 — mesh coloured by init-u", fontsize=9)
    ax_bl.set_xticks([]); ax_bl.set_yticks([]); ax_bl.set_zticks([])

    pc_br = PolyCollection(polys_2d, array=face_u, cmap="viridis", edgecolors="white", linewidths=0.15)
    ax_br.add_collection(pc_br)
    tri_2d = matplotlib.tri.Triangulation(uv_init[:, 0], uv_init[:, 1], F)
    ax_br.triplot(tri_2d, color="white", linewidth=0.15, alpha=0.4)
    ax_br.autoscale_view()
    ax_br.set_aspect("equal")
    ax_br.set_title("Step 4 — UV init (2D)", fontsize=9)
    ax_br.set_xlabel("u"); ax_br.set_ylabel("v")

    ann = f"init method: {init_method}   flipped faces: {n_flipped}"
    fig.text(0.5, 0.02, ann, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig


def _plot_step5_slim_final(
    uv_init: np.ndarray,
    F: np.ndarray,
    uv_final: np.ndarray,
    trimmed: bool,
    vertex_colors: np.ndarray | None = None,
) -> Figure:
    """Step 5 — init UV vs final SLIM UV: skin colours (top) + viridis (bottom).

    When *vertex_colors* is provided the figure is 2×2: the top row shows the
    UV layouts with real skin colours and the bottom row shows the viridis
    face-u analytical view.  Without colours, falls back to the original
    1×2 viridis-only layout.
    """
    u_vals = uv_init[:, 0]
    u_norm = (u_vals - u_vals.min()) / max(float(u_vals.max() - u_vals.min()), 1e-15)
    face_u = u_norm[F].mean(axis=1)

    polys_init = uv_init[F]
    polys_final = uv_final[F]

    has_colors = vertex_colors is not None
    fig = Figure(figsize=(10, 9 if has_colors else 4.5))
    FigureCanvasAgg(fig)

    if has_colors:
        face_rgba = vertex_colors[F].mean(axis=1)

        ax_tl = fig.add_subplot(2, 2, 1)
        pc_tl = PolyCollection(polys_init, facecolors=face_rgba, edgecolors="white", linewidths=0.15)
        ax_tl.add_collection(pc_tl)
        ax_tl.autoscale_view()
        ax_tl.set_aspect("equal")
        ax_tl.set_title("Step 5 — UV init (skin colours)", fontsize=9)
        ax_tl.set_xlabel("u"); ax_tl.set_ylabel("v")

        ax_tr = fig.add_subplot(2, 2, 2)
        pc_tr = PolyCollection(polys_final, facecolors=face_rgba, edgecolors="white", linewidths=0.15)
        ax_tr.add_collection(pc_tr)
        ax_tr.autoscale_view()
        ax_tr.set_aspect("equal")
        ax_tr.set_title("Step 5 — SLIM UV final (skin colours)", fontsize=9)
        ax_tr.set_xlabel("u"); ax_tr.set_ylabel("v")

        ax_bl = fig.add_subplot(2, 2, 3)
        ax_br = fig.add_subplot(2, 2, 4)
    else:
        ax_bl = fig.add_subplot(1, 2, 1)
        ax_br = fig.add_subplot(1, 2, 2)

    pc_bl = PolyCollection(polys_init, array=face_u, cmap="viridis", edgecolors="white", linewidths=0.15)
    ax_bl.add_collection(pc_bl)
    tri_init = matplotlib.tri.Triangulation(uv_init[:, 0], uv_init[:, 1], F)
    ax_bl.triplot(tri_init, color="white", linewidth=0.15, alpha=0.4)
    ax_bl.autoscale_view()
    ax_bl.set_aspect("equal")
    ax_bl.set_title("Step 5 — UV init", fontsize=9)
    ax_bl.set_xlabel("u"); ax_bl.set_ylabel("v")

    pc_br = PolyCollection(polys_final, array=face_u, cmap="viridis", edgecolors="white", linewidths=0.15)
    ax_br.add_collection(pc_br)
    tri_final = matplotlib.tri.Triangulation(uv_final[:, 0], uv_final[:, 1], F)
    ax_br.triplot(tri_final, color="white", linewidth=0.15, alpha=0.4)
    ax_br.autoscale_view()
    ax_br.set_aspect("equal")
    ax_br.set_title("Step 5 — SLIM UV final", fontsize=9)
    ax_br.set_xlabel("u"); ax_br.set_ylabel("v")

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
    raw_mesh_colors: np.ndarray | None = None,
    clean_mesh_colors: np.ndarray | None = None,
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
    raw_mesh_colors:
        Per-vertex RGBA colours for V_raw, shape (N_raw, 4) float [0, 1].
    clean_mesh_colors:
        Per-vertex RGBA colours for V, shape (N_clean, 4) float [0, 1].

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

    # Compute colours for the potentially trimmed mesh (steps 4-5).
    trim_colors = clean_mesh_colors
    if trimmed and clean_mesh_colors is not None:
        from scipy.spatial import KDTree
        _, idx = KDTree(V).query(V_trim)
        trim_colors = clean_mesh_colors[idx]

    paths: list[Path] = []

    fig1 = _plot_step1_raw_mesh(V_raw, F_raw, raw_elev, raw_azim, cam_elev, cam_azim, cam_label,
                                vertex_colors=raw_mesh_colors)
    p1 = diag_dir / "step1_raw_mesh.png"
    fig1.savefig(p1, dpi=200)
    paths.append(p1)

    fig2 = _plot_step2_cleaned_mesh(V, F, elev, azim, cam_elev, cam_azim, cam_label,
                                    vertex_colors=clean_mesh_colors)
    p2 = diag_dir / "step2_cleaned_mesh.png"
    fig2.savefig(p2, dpi=200)
    paths.append(p2)

    fig3 = _plot_step3_centroid_boundary(V, F, center_vid, boundary, elev, azim, cam_elev, cam_azim, cam_label,
                                         vertex_colors=clean_mesh_colors)
    p3 = diag_dir / "step3_centroid_boundary.png"
    fig3.savefig(p3, dpi=200)
    paths.append(p3)

    trim_elev, trim_azim = _compute_skin_normal_view(V_trim, F_trim)
    fig4 = _plot_step4_uv_init(V_trim, F_trim, init_uv, init_method, trim_elev, trim_azim,
                               vertex_colors=trim_colors)
    p4 = diag_dir / "step4_uv_initialization.png"
    fig4.savefig(p4, dpi=200)
    paths.append(p4)

    fig5 = _plot_step5_slim_final(init_uv, F_trim, uv_final, trimmed,
                                  vertex_colors=trim_colors)
    p5 = diag_dir / "step5_slim_uv_final.png"
    fig5.savefig(p5, dpi=200)
    paths.append(p5)

    fig6 = plot_slim_distortion_panel(V_trim, F_trim, uv_final, center_vid)
    fig6.tight_layout()
    p6 = diag_dir / "step6_distortion.png"
    fig6.savefig(p6, dpi=200)
    paths.append(p6)

    return paths
