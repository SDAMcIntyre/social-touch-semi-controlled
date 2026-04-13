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
    interactive: bool = False,
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
    interactive:
        If True, display an interactive 3D window (blocking) in addition to
        saving the PNG. Skips the Agg backend so the window is navigable.
        If False (default), uses the Agg backend for offscreen PNG-only rendering.
    """
    import matplotlib
    if not interactive:
        matplotlib.use('Agg')  # Non-interactive backend for offscreen rendering
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

    if spike_counts_df.empty:
        logger.warning(
            "Empty spike_counts_df for session %s, cluster %s — skipping render.",
            session_id, cluster_label,
        )
        return

    fig = plt.figure(figsize=(10, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')

    # --- Plot forearm point cloud ---
    forearm_vertices = None
    sc_forearm = None
    if forearm_ply_path is not None and forearm_ply_path.exists():
        try:
            import open3d as o3d
            from scipy.spatial import KDTree

            pcd = o3d.io.read_point_cloud(str(forearm_ply_path))
            pts = np.asarray(pcd.points)
            if pts.size > 0:
                forearm_vertices = pts
                # Subsample to ~10k points for rendering speed
                stride = max(1, len(pts) // 10000)
                forearm_sub = pts[::stride]
                colors = np.asarray(pcd.colors)
                sub_colors = colors[::stride] if colors.size > 0 else None
                # Remove forearm points near spike contact locations
                spike_xyz = spike_counts_df[['x', 'y', 'z']].to_numpy()
                if len(spike_xyz) > 0:
                    tree = KDTree(forearm_sub[:, :3])
                    nearby = tree.query_ball_point(spike_xyz, r=2.0)
                    exclude = set().union(*nearby)
                    mask = np.ones(len(forearm_sub), dtype=bool)
                    mask[list(exclude)] = False
                    forearm_sub = forearm_sub[mask]
                    if sub_colors is not None:
                        sub_colors = sub_colors[mask]
                if sub_colors is not None:
                    point_colors = sub_colors
                else:
                    point_colors = 'lightgrey'
                sc_forearm = ax.scatter(
                    forearm_sub[:, 0], forearm_sub[:, 1], forearm_sub[:, 2],
                    c=point_colors, s=20, alpha=1.0, rasterized=True,
                    linewidths=0, depthshade=False,
                )
        except Exception:
            logger.warning("Could not load forearm PLY: %s", forearm_ply_path, exc_info=True)
    elif forearm_ply_path is None:
        logger.warning("No forearm PLY resolved for session %s", session_id)
    else:
        logger.warning("Forearm PLY not found: %s", forearm_ply_path)

    # --- Overlay contact points coloured by spike_count ---
    xs = spike_counts_df['x'].to_numpy()
    ys = spike_counts_df['y'].to_numpy()
    zs = spike_counts_df['z'].to_numpy()
    counts = spike_counts_df['spike_count'].to_numpy()

    vmin = max(1, counts.min())
    vmax = counts.max()
    # Guard against vmin == vmax (all counts identical) — fall back to linear
    if vmin >= vmax:
        norm = None  # matplotlib default linear
    else:
        norm = LogNorm(vmin=vmin, vmax=vmax)

    sc = ax.scatter(
        xs, ys, zs,
        c=counts, cmap='RdYlBu_r', s=20, alpha=0.9,
        norm=norm, depthshade=False,
    )
    cbar = plt.colorbar(sc, ax=ax, label='Spike count', shrink=0.6, pad=0.1)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.yaxis.label.set_color('white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # --- Camera orientation: normal to contact surface ---
    elev, azim = 30.0, 45.0  # sensible default
    if forearm_vertices is not None and len(xs) > 0:
        contact_centroid = np.array([xs.mean(), ys.mean(), zs.mean()])
        normal = _compute_surface_normal(forearm_vertices, contact_centroid)
        if normal is not None:
            elev, azim = _normal_to_view_angles(normal)

    ax.view_init(elev=elev, azim=azim)

    # --- Labels and title ---
    ax.set_xlabel('X (mm)', color='white')
    ax.set_ylabel('Y (mm)', color='white')
    ax.set_zlabel('Z (mm)', color='white')
    ax.set_title(
        f'RF Heatmap — {session_id} | cluster {cluster_label}',
        fontsize=11, color='white',
    )
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    # --- Point-size slider (interactive mode only) ---
    if interactive:
        from matplotlib.widgets import Slider

        fig.subplots_adjust(bottom=0.15)
        ax_slider = fig.add_axes([0.2, 0.04, 0.6, 0.03], facecolor='#222222')
        slider = Slider(
            ax=ax_slider,
            label='Point size',
            valmin=1,
            valmax=100,
            valinit=20,
            color='#555555',
        )
        slider.label.set_color('white')
        slider.valtext.set_color('white')

        def _on_size_change(val: float) -> None:
            s = [val]
            if sc_forearm is not None:
                sc_forearm.set_sizes(s)
            sc.set_sizes(s)
            fig.canvas.draw_idle()

        slider.on_changed(_on_size_change)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    logger.info("Saved heatmap: %s", output_path)

    if interactive:
        plt.show()  # Blocking — waits for user to close the window

    plt.close(fig)
