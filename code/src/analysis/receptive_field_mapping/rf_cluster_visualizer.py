"""3D forearm heatmap rendering for cluster-based RF mapping.

Renders the forearm point cloud (PLY) as a subtle grey background with
spike-count contact points overlaid as a coloured heatmap. Camera is oriented
normal to the contact surface when possible, with a sensible default fallback.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _compute_surface_normal(
    forearm_vertices: np.ndarray,
    contact_centroid: np.ndarray,
    k: int = 50,
) -> np.ndarray:
    """Estimate the forearm surface normal at the contact centroid.

    Uses a KD-tree to find k nearest forearm vertices to the centroid, then
    PCA on those neighbours to extract the smallest eigenvector (surface normal).

    Parameters
    ----------
    forearm_vertices:
        (N, 3) array of forearm point cloud vertices.
    contact_centroid:
        (3,) array — centroid of contact points.
    k:
        Number of neighbours to use for local PCA.

    Returns
    -------
    Unit normal vector (3,), or None if computation fails or is degenerate.
    """
    try:
        from scipy.spatial import KDTree

        tree = KDTree(forearm_vertices)
        _, idx = tree.query(contact_centroid, k=min(k, len(forearm_vertices)))
        neighbors = forearm_vertices[idx]
        centered = neighbors - neighbors.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        # Smallest singular vector = surface normal direction
        normal = Vt[-1]
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            return None
        return normal / norm
    except Exception:
        logger.debug("Surface normal computation failed", exc_info=True)
        return None


def _normal_to_view_angles(normal: np.ndarray) -> tuple:
    """Convert a surface normal to matplotlib 3D view_init angles.

    Parameters
    ----------
    normal:
        Unit 3-vector pointing away from the surface.

    Returns
    -------
    (elev, azim) in degrees, suitable for ``ax.view_init()``.
    """
    # Elevation = arcsin of the y-component (assumed up-axis in camera coords)
    elev = float(np.degrees(np.arcsin(np.clip(normal[1], -1.0, 1.0))))
    # Azimuth = atan2 of x and z
    azim = float(np.degrees(np.arctan2(normal[0], normal[2])))
    return elev, azim


def render_forearm_heatmap(
    forearm_ply_path: Path,
    spike_counts_df: pd.DataFrame,
    output_path: Path,
    session_id: str,
    cluster_label: str,
) -> None:
    """Render a 3D forearm heatmap of spike-count contact points and save as PNG.

    Plots the forearm point cloud as a subtle grey scatter, then overlays
    contact points coloured by spike_count using the YlOrRd colormap. Camera
    is oriented normal to the contact surface when possible; falls back to
    (30°, 45°) if PLY is unavailable or normal computation fails.

    All coordinate axes are labelled in mm (Kinect SDK native units).

    Parameters
    ----------
    forearm_ply_path:
        Path to the PCA-calibrated forearm PLY file.
    spike_counts_df:
        DataFrame with columns (x, y, z, spike_count).
    output_path:
        Destination PNG file path (parent directories are created if needed).
    session_id:
        Used in the figure title.
    cluster_label:
        Used in the figure title.
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for offscreen rendering
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

    if spike_counts_df.empty:
        logger.warning(
            "Empty spike_counts_df for session %s, cluster %s — skipping render.",
            session_id, cluster_label,
        )
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- Plot forearm point cloud ---
    forearm_vertices = None
    if forearm_ply_path.exists():
        try:
            import open3d as o3d

            pcd = o3d.io.read_point_cloud(str(forearm_ply_path))
            pts = np.asarray(pcd.points)
            if pts.size > 0:
                forearm_vertices = pts
                # Subsample to ~10k points for rendering speed
                stride = max(1, len(pts) // 10000)
                ax.scatter(
                    pts[::stride, 0], pts[::stride, 1], pts[::stride, 2],
                    c='lightgrey', s=0.5, alpha=0.3, rasterized=True,
                    linewidths=0,
                )
        except Exception:
            logger.warning("Could not load forearm PLY: %s", forearm_ply_path, exc_info=True)
    else:
        logger.warning("Forearm PLY not found: %s", forearm_ply_path)

    # --- Overlay contact points coloured by spike_count ---
    xs = spike_counts_df['x'].to_numpy()
    ys = spike_counts_df['y'].to_numpy()
    zs = spike_counts_df['z'].to_numpy()
    counts = spike_counts_df['spike_count'].to_numpy()

    sc = ax.scatter(
        xs, ys, zs,
        c=counts, cmap='YlOrRd', s=20, alpha=0.9,
        vmin=counts.min(), vmax=counts.max(),
    )
    plt.colorbar(sc, ax=ax, label='Spike count', shrink=0.6, pad=0.1)

    # --- Camera orientation: normal to contact surface ---
    elev, azim = 30.0, 45.0  # sensible default
    if forearm_vertices is not None and len(xs) > 0:
        contact_centroid = np.array([xs.mean(), ys.mean(), zs.mean()])
        normal = _compute_surface_normal(forearm_vertices, contact_centroid)
        if normal is not None:
            elev, azim = _normal_to_view_angles(normal)

    ax.view_init(elev=elev, azim=azim)

    # --- Labels and title ---
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(
        f'RF Heatmap — {session_id} | cluster {cluster_label}',
        fontsize=11,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches='tight')
    plt.close(fig)

    logger.info("Saved heatmap: %s", output_path)
