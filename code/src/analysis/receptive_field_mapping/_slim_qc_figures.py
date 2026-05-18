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
save_slim_qc_figures       — Write both QC figures and return their paths.
plot_slim_uv_panel         — 3D mesh + SLIM UV layout (1 × 2).
plot_slim_distortion_panel — Conformal + area distortion (2 × 1).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.tri
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers "3d" projection

from ._slim_helpers import compute_face_distortion


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
