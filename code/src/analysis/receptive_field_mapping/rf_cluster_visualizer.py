"""3D forearm heatmap rendering for cluster-based RF mapping.

Renders the forearm point cloud (PLY) as a subtle grey background with
spike-count contact points overlaid as a coloured heatmap. Camera is oriented
normal to the contact surface when possible, with a sensible default fallback.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .rf_2d_renderer import render_2d_heatmap
from .rf_projection import project_to_2d
from .rf_surface_utils import load_or_build_forearm_mesh, map_scalars_to_mesh
from .tangent_plane_alignment import (  # noqa: F401
    _compute_surface_normal,
    align_points,
    compute_tangent_plane_rotation,
)

logger = logging.getLogger(__name__)


@dataclass
class RFRenderContext:
    """Context object carrying neuron-scoped counts and contact arrays for rendering.

    Parameters
    ----------
    neuron_touches:
        Total touch count for this neuron (session) across all clusters.
    neuron_cluster_touches:
        Touch count for this neuron in the current cluster only.
    neuron_contacts_xyz:
        (N, 3) array of mean_contact_x/y/z for this neuron across all clusters.
    neuron_cluster_contacts_xyz:
        (M, 3) array of mean_contact_x/y/z for this neuron in the current cluster.
    feature_ranges:
        Dict from cluster_description['feature_ranges']: {name: {min, max, mean}}.
    """

    neuron_touches: int
    neuron_cluster_touches: int
    neuron_contacts_xyz: np.ndarray   # (N, 3)
    neuron_cluster_contacts_xyz: np.ndarray   # (M, 3)
    feature_ranges: dict


def _format_metadata_overlay(context: RFRenderContext) -> str:
    """Build the metadata text string for figure overlays.

    Returns a multi-line string:
        Touches: {cluster} / {neuron} ({pct:.1f}%)
        {feature}: [{min}, {max}]
        ...
    """
    nt = context.neuron_touches
    nct = context.neuron_cluster_touches

    if nt > 0:
        pct = 100.0 * nct / nt
        touch_line = f"Touches: {nct} / {nt} ({pct:.1f}%)"
    else:
        touch_line = f"Touches: {nct} / {nt} (—%)"

    lines = [touch_line]
    for feature_name, ranges in context.feature_ranges.items():
        fmin = ranges.get("min", "?")
        fmax = ranges.get("max", "?")
        # Round floats for readability
        if isinstance(fmin, float):
            fmin = round(fmin, 3)
        if isinstance(fmax, float):
            fmax = round(fmax, 3)
        lines.append(f"{feature_name}: [{fmin}, {fmax}]")

    return "\n".join(lines)


def _draw_hull_3d(ax, points_3d: np.ndarray, color: str, label: str) -> None:
    """Draw convex hull edges as 3D polylines on a 3D axes.

    Skips gracefully (with a log info) if fewer than 3 points are provided or
    if scipy raises QhullError.

    Parameters
    ----------
    ax:
        Matplotlib 3D axes object.
    points_3d:
        (N, 3) array of 3D points.
    color:
        Line colour string.
    label:
        Legend label — applied to the first plotted simplex only to avoid
        duplicate legend entries.
    """
    from scipy.spatial import ConvexHull, QhullError

    if points_3d is None or len(points_3d) < 3:
        logger.info(
            "_draw_hull_3d: too few points (%d) for hull '%s' — skipping.",
            len(points_3d) if points_3d is not None else 0,
            label,
        )
        return

    try:
        hull = ConvexHull(points_3d)
    except QhullError:
        logger.info("_draw_hull_3d: QhullError for hull '%s' — skipping.", label)
        return

    first = True
    for simplex in hull.simplices:
        # Each simplex is a triangle: draw all three edges
        for i in range(len(simplex)):
            p1 = points_3d[simplex[i]]
            p2 = points_3d[simplex[(i + 1) % len(simplex)]]
            kwargs = dict(color=color, linewidth=0.8, linestyle='--', alpha=0.7)
            if first:
                kwargs['label'] = label
                first = False
            ax.plot3D(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                **kwargs,
            )


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
    projection_method: str = None,
    cluster_description: str = '',
    display_metric: str = "spike_count",
    render_context: 'RFRenderContext' = None,
    disjoint_mask_distance_mm: float = 8.0,
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
        DataFrame with columns (x, y, z, spike_count, unique_touch_spike_count).
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
    projection_method:
        If set, projects spike points to 2D using the named method and renders
        a 2D scatter + heatmap figure instead of the 3D plot. The output filename
        should include the method name (caller's responsibility via output_path).
        Pass None (default) for the original 3D rendering path.
    display_metric:
        ``"spike_count"`` (default) uses raw spike counts as the colour driver.
        ``"spike_ratio"`` uses unique_touch_spike_count / neuron_cluster_touches
        (range [0, 1]).  Any other value raises ``ValueError``.
    render_context:
        Optional :class:`RFRenderContext` carrying neuron-scoped touch counts,
        contact arrays, and feature ranges. Required when
        ``display_metric="spike_ratio"``.
    disjoint_mask_distance_mm:
        In the 2D projection path, grid cells further than this distance (mm)
        from the nearest sample point are masked to NaN. Default 8.0 mm.
    """
    _VALID_METRICS = {"spike_count", "spike_ratio"}
    if display_metric not in _VALID_METRICS:
        raise ValueError(
            f"display_metric must be one of {_VALID_METRICS!r}, got {display_metric!r}."
        )

    if display_metric == "spike_ratio":
        if render_context is None or render_context.neuron_cluster_touches == 0:
            raise ValueError(
                "display_metric='spike_ratio' requires render_context with "
                "neuron_cluster_touches > 0, but got "
                f"render_context={render_context!r}."
            )

    if spike_counts_df.empty:
        logger.warning(
            "Empty spike_counts_df for session %s, cluster %s — skipping render.",
            session_id, cluster_label,
        )
        return

    # --- 2D projection path (early return) ---
    if projection_method is not None:
        spike_xyz = spike_counts_df[['x', 'y', 'z']].to_numpy()
        counts = spike_counts_df['spike_count'].to_numpy()
        contact_centroid = spike_xyz.mean(axis=0) if len(spike_xyz) > 0 else None

        forearm_vertices = None
        if forearm_ply_path is not None and forearm_ply_path.exists():
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(str(forearm_ply_path))
                pts = np.asarray(pcd.points)
                if pts.size > 0:
                    forearm_vertices = pts
            except Exception:
                logger.warning("Could not load forearm PLY for 2D projection: %s", forearm_ply_path, exc_info=True)

        uv_points = project_to_2d(spike_xyz, forearm_vertices, contact_centroid, method=projection_method)

        forearm_uv = None
        if forearm_vertices is not None:
            try:
                forearm_uv = project_to_2d(forearm_vertices, forearm_vertices, contact_centroid, method=projection_method)
            except Exception:
                logger.debug("Could not project forearm vertices to 2D for background.", exc_info=True)

        # --- Compute ratio counts if requested ---
        ratio_counts = None
        if display_metric == "spike_ratio":
            ratio_counts = (
                spike_counts_df['unique_touch_spike_count'].to_numpy()
                / render_context.neuron_cluster_touches
            )

        # --- Project neuron hull contact points if context provided ---
        neuron_contacts_uv = None
        neuron_cluster_contacts_uv = None
        if render_context is not None:
            hull_centroid = (
                render_context.neuron_contacts_xyz.mean(axis=0)
                if len(render_context.neuron_contacts_xyz) > 0
                else contact_centroid
            )
            if len(render_context.neuron_contacts_xyz) > 0:
                try:
                    neuron_contacts_uv = project_to_2d(
                        render_context.neuron_contacts_xyz,
                        forearm_vertices,
                        hull_centroid,
                        method=projection_method,
                    )
                except Exception:
                    logger.debug(
                        "Could not project neuron_contacts_xyz to 2D for hull.", exc_info=True
                    )
            if len(render_context.neuron_cluster_contacts_xyz) > 0:
                try:
                    neuron_cluster_contacts_uv = project_to_2d(
                        render_context.neuron_cluster_contacts_xyz,
                        forearm_vertices,
                        hull_centroid,
                        method=projection_method,
                    )
                except Exception:
                    logger.debug(
                        "Could not project neuron_cluster_contacts_xyz to 2D for hull.", exc_info=True
                    )

        render_2d_heatmap(
            uv_points=uv_points,
            counts=counts,
            output_path=output_path,
            session_id=session_id,
            cluster_label=cluster_label,
            projection_method=projection_method,
            forearm_uv=forearm_uv,
            interactive=interactive,
            cluster_description=cluster_description,
            display_metric=display_metric,
            render_context=render_context,
            ratio_counts=ratio_counts,
            neuron_contacts_uv=neuron_contacts_uv,
            neuron_cluster_contacts_uv=neuron_cluster_contacts_uv,
            disjoint_mask_distance_mm=disjoint_mask_distance_mm,
        )
        return

    import matplotlib
    if not interactive:
        matplotlib.use('Agg')  # Non-interactive backend for offscreen rendering
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

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
                        # Determine colour driver
                        if display_metric == "spike_ratio":
                            ratio_vals = (
                                spike_counts_df['unique_touch_spike_count'].to_numpy()
                                / render_context.neuron_cluster_touches
                            )
                            per_vertex_color = map_scalars_to_mesh(
                                forearm_mesh, spike_xyz, ratio_vals.astype(float)
                            )
                            vmin_c = 0.0
                            vmax_c = 1.0
                            norm = None
                            cbar_label = "Spike ratio"
                        else:
                            per_vertex_color = per_vertex
                            vmin_c = max(1, float(np.nanmin(per_vertex[~np.isnan(per_vertex)])))
                            vmax_c = float(np.nanmax(per_vertex[~np.isnan(per_vertex)]))
                            norm = LogNorm(vmin=vmin_c, vmax=vmax_c) if vmin_c < vmax_c else None
                            cbar_label = "Spike count"

                        import matplotlib.cm as cm
                        cmap_obj = cm.get_cmap('RdYlBu_r')
                        if norm is not None:
                            normed = norm(np.nan_to_num(per_vertex_color, nan=vmin_c))
                        else:
                            if vmax_c > vmin_c:
                                normed = (np.nan_to_num(per_vertex_color, nan=vmin_c) - vmin_c) / (vmax_c - vmin_c)
                            else:
                                normed = np.zeros_like(per_vertex_color)
                        vertex_rgba = cmap_obj(normed)
                        vertex_rgba[np.isnan(per_vertex_color), 3] = 0.0

                        # plot_trisurf cannot accept facecolors kwarg directly —
                        # it passes it internally and would get duplicates. Set
                        # per-face colors on the returned Poly3DCollection instead.
                        face_rgba = vertex_rgba[faces].mean(axis=1)
                        nan_mask = np.isnan(per_vertex_color)
                        face_rgba[nan_mask[faces].all(axis=1), 3] = 0.0

                        surf = ax.plot_trisurf(
                            verts[:, 0], verts[:, 1], verts[:, 2],
                            triangles=faces,
                            shade=False,
                            edgecolor='none',
                        )
                        surf.set_facecolor(face_rgba)

                        from matplotlib.cm import ScalarMappable
                        from matplotlib.colors import Normalize
                        sm_norm = norm if norm is not None else Normalize(vmin=vmin_c, vmax=vmax_c)
                        sm = ScalarMappable(cmap='RdYlBu_r', norm=sm_norm)
                        sm.set_array([])
                        cbar = plt.colorbar(sm, ax=ax, label=cbar_label, shrink=0.6, pad=0.1)
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

        if display_metric == "spike_ratio":
            color_values = (
                spike_counts_df['unique_touch_spike_count'].to_numpy()
                / render_context.neuron_cluster_touches
            )
            vmin_s = 0.0
            vmax_s = 1.0
            norm = None
            cbar_label = "Spike ratio"
        else:
            color_values = counts
            vmin_s = max(1, counts.min())
            vmax_s = counts.max()
            norm = LogNorm(vmin=vmin_s, vmax=vmax_s) if vmin_s < vmax_s else None
            cbar_label = "Spike count"

        sc = ax.scatter(
            plot_xs, plot_ys, plot_zs,
            c=color_values, cmap='RdYlBu_r', s=20, alpha=0.9,
            norm=norm, depthshade=False,
            vmin=vmin_s if norm is None else None,
            vmax=vmax_s if norm is None else None,
        )
        cbar = plt.colorbar(sc, ax=ax, label=cbar_label, shrink=0.6, pad=0.1)
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.label.set_color('white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # --- Draw convex hull perimeters (3D path) ---
    if render_context is not None:
        if R is not None:
            all_xyz_rot = (
                align_points(render_context.neuron_contacts_xyz, R)
                if len(render_context.neuron_contacts_xyz) > 0
                else render_context.neuron_contacts_xyz
            )
            cluster_xyz_rot = (
                align_points(render_context.neuron_cluster_contacts_xyz, R)
                if len(render_context.neuron_cluster_contacts_xyz) > 0
                else render_context.neuron_cluster_contacts_xyz
            )
        else:
            all_xyz_rot = render_context.neuron_contacts_xyz
            cluster_xyz_rot = render_context.neuron_cluster_contacts_xyz

        _draw_hull_3d(ax, all_xyz_rot, '#00aaff', 'neuron (all clusters)')
        _draw_hull_3d(ax, cluster_xyz_rot, '#ffaa00', 'neuron ∩ cluster')

        metadata_text = _format_metadata_overlay(render_context)
        ax.text2D(
            0.02, 0.02,
            metadata_text,
            transform=ax.transAxes,
            va='bottom',
            ha='left',
            color='#aaaaaa',
            fontsize=7,
            fontfamily='monospace',
        )

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
    title_3d = f'RF Heatmap — {session_id} | cluster {cluster_label}'
    if cluster_description:
        title_3d += f'\n{cluster_description}'
    ax.set_title(title_3d, fontsize=11, color='white')
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
