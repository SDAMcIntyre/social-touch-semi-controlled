"""3D forearm heatmap rendering for cluster-based RF mapping.

Renders the forearm point cloud (PLY) as a subtle grey background with
spike-count contact points overlaid as a coloured heatmap. Camera is oriented
normal to the contact surface when possible, with a sensible default fallback.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .rf_surface_utils import load_or_build_forearm_mesh, map_scalars_to_mesh
from .tangent_plane_alignment import (  # noqa: F401
    _compute_surface_normal,
    align_points,
    compute_tangent_plane_rotation,
)

logger = logging.getLogger(__name__)


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

    # --- Compute contact centroid and tangent-plane rotation early ---
    xs = spike_counts_df['x'].to_numpy()
    ys = spike_counts_df['y'].to_numpy()
    zs = spike_counts_df['z'].to_numpy()
    contact_centroid = np.array([xs.mean(), ys.mean(), zs.mean()]) if len(xs) > 0 else None

    # --- Plot forearm point cloud ---
    forearm_vertices = None
    sc_forearm = None
    R = None
    sc = None
    cbar_created = False

    if forearm_ply_path is not None and forearm_ply_path.exists():
        try:
            import open3d as o3d
            from scipy.spatial import KDTree

            pcd = o3d.io.read_point_cloud(str(forearm_ply_path))
            pts = np.asarray(pcd.points)
            if pts.size > 0:
                forearm_vertices = pts

                R = (
                    compute_tangent_plane_rotation(forearm_vertices, contact_centroid)
                    if contact_centroid is not None
                    else None
                )

                forearm_mesh = load_or_build_forearm_mesh(forearm_ply_path)

                if forearm_mesh is not None:
                    from .rf_surface_utils import apply_rotation_to_mesh
                    if R is not None:
                        forearm_mesh = apply_rotation_to_mesh(forearm_mesh, R)

                    verts = forearm_mesh.vertices
                    faces = forearm_mesh.faces

                    ax.plot_trisurf(
                        verts[:, 0], verts[:, 1], verts[:, 2],
                        triangles=faces,
                        color='lightgrey',
                        shade=True,
                        edgecolor='none',
                        alpha=1.0,
                    )

                    spike_xyz = spike_counts_df[['x', 'y', 'z']].to_numpy()
                    counts = spike_counts_df['spike_count'].to_numpy()
                    if R is not None:
                        spike_xyz = align_points(spike_xyz, R)

                    per_vertex = map_scalars_to_mesh(forearm_mesh, spike_xyz, counts.astype(float))

                    has_any = np.any(~np.isnan(per_vertex))
                    if has_any:
                        vmin = max(1, float(np.nanmin(per_vertex[~np.isnan(per_vertex)])))
                        vmax = float(np.nanmax(per_vertex[~np.isnan(per_vertex)]))
                        if vmin < vmax:
                            norm = LogNorm(vmin=vmin, vmax=vmax)
                        else:
                            norm = None

                        import matplotlib.cm as cm
                        cmap_obj = cm.get_cmap('RdYlBu_r')
                        if norm is not None:
                            normed = norm(np.nan_to_num(per_vertex, nan=vmin))
                        else:
                            if vmax > vmin:
                                normed = (np.nan_to_num(per_vertex, nan=vmin) - vmin) / (vmax - vmin)
                            else:
                                normed = np.zeros_like(per_vertex)
                        vertex_rgba = cmap_obj(normed)
                        vertex_rgba[np.isnan(per_vertex), 3] = 0.0

                        # plot_trisurf cannot accept facecolors kwarg directly —
                        # it passes it internally and would get duplicates. Set
                        # per-face colors on the returned Poly3DCollection instead.
                        face_rgba = vertex_rgba[faces].mean(axis=1)
                        nan_mask = np.isnan(per_vertex)
                        face_rgba[nan_mask[faces].all(axis=1), 3] = 0.0

                        surf = ax.plot_trisurf(
                            verts[:, 0], verts[:, 1], verts[:, 2],
                            triangles=faces,
                            shade=False,
                            edgecolor='none',
                        )
                        surf.set_facecolor(face_rgba)

                        from matplotlib.cm import ScalarMappable
                        sm = ScalarMappable(cmap='RdYlBu_r', norm=norm)
                        sm.set_array([])
                        cbar = plt.colorbar(sm, ax=ax, label='Spike count', shrink=0.6, pad=0.1)
                        cbar.ax.yaxis.set_tick_params(color='white')
                        cbar.ax.yaxis.label.set_color('white')
                        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')
                        cbar_created = True

                else:
                    # Scatter fallback
                    stride = max(1, len(pts) // 10000)
                    forearm_sub = pts[::stride]
                    colors = np.asarray(pcd.colors)
                    sub_colors = colors[::stride] if colors.size > 0 else None
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

                    if R is not None:
                        forearm_sub = align_points(forearm_sub, R)

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

    # --- Overlay contact points coloured by spike_count (scatter fallback path only) ---
    counts = spike_counts_df['spike_count'].to_numpy()

    if not cbar_created:
        if R is not None:
            spike_pts_rot = align_points(np.stack([xs, ys, zs], axis=1), R)
            plot_xs, plot_ys, plot_zs = spike_pts_rot[:, 0], spike_pts_rot[:, 1], spike_pts_rot[:, 2]
        else:
            plot_xs, plot_ys, plot_zs = xs, ys, zs

        vmin = max(1, counts.min())
        vmax = counts.max()
        if vmin >= vmax:
            norm = None
        else:
            norm = LogNorm(vmin=vmin, vmax=vmax)

        sc = ax.scatter(
            plot_xs, plot_ys, plot_zs,
            c=counts, cmap='RdYlBu_r', s=20, alpha=0.9,
            norm=norm, depthshade=False,
        )
        cbar = plt.colorbar(sc, ax=ax, label='Spike count', shrink=0.6, pad=0.1)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.label.set_color('white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # --- Camera orientation ---
    if R is not None:
        ax.view_init(elev=90, azim=-90)
    else:
        elev, azim = 30.0, 45.0  # sensible default
        if forearm_vertices is not None and contact_centroid is not None:
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
            if sc is not None:
                sc.set_sizes(s)
            fig.canvas.draw_idle()

        slider.on_changed(_on_size_change)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    logger.info("Saved heatmap: %s", output_path)

    if interactive:
        plt.show()  # Blocking — waits for user to close the window

    plt.close(fig)
