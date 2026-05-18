"""3D forearm heatmap rendering for cluster-based RF mapping.

Renders the forearm surface mesh as a coloured heatmap using PyVista offscreen
rendering. Camera is set directly from saved camera settings, guaranteeing
orientation fidelity with the RF Camera Settings Viewer.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.spatial import ConvexHull, QhullError

from .rf_2d_renderer import render_2d_heatmap
from .rf_data_loader import load_forearm_vertices
from .rf_projection import project_to_2d
from .rf_surface_utils import load_or_build_forearm_mesh, map_scalars_to_mesh, mesh_to_pyvista
from .tangent_plane_alignment import camera_settings_to_rotation

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
        (N, 3) unique mm-rounded contact points parsed from the session's aggregated CSV
        ``contact_points`` column, across all clusters.
        Invariant: heatmap spike contacts ⊆ neuron_cluster_contacts_xyz ⊆ neuron_contacts_xyz.
    neuron_cluster_contacts_xyz:
        (M, 3) unique mm-rounded contact points parsed from the session's aggregated CSV
        ``contact_points`` column, for this cluster only.
        Invariant: heatmap spike contacts ⊆ neuron_cluster_contacts_xyz ⊆ neuron_contacts_xyz.
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

    Returns a single-line string:
        Touches: {cluster} / {neuron} ({pct:.1f}%)
    """
    nt = context.neuron_touches
    nct = context.neuron_cluster_touches

    if nt > 0:
        pct = 100.0 * nct / nt
        return f"Touches: {nct} / {nt} ({pct:.1f}%)"
    return f"Touches: {nct} / {nt} (—%)"


def _pyvista_hull_mesh(points: np.ndarray) -> pv.PolyData | None:
    """Build a PyVista PolyData wireframe from the convex hull of the given points.

    Returns None if points are too few or the hull computation fails.
    """
    if points is None or len(points) < 4:
        return None
    try:
        hull = ConvexHull(points)
    except QhullError:
        return None
    lines = []
    for simplex in hull.simplices:
        for i in range(len(simplex)):
            lines.extend([2, simplex[i], simplex[(i + 1) % len(simplex)]])
    mesh = pv.PolyData(points)
    mesh.lines = np.array(lines, dtype=np.int64)
    return mesh


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
    camera_settings: dict = None,
) -> None:
    """Render a 3D forearm heatmap of spike-count contact points and save as PNG.

    Uses PyVista offscreen rendering for the 3D path. Camera is oriented directly
    from saved camera_settings, guaranteeing fidelity with the RF Camera Settings Viewer.

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
        If True, display an interactive 3D window (blocking). If False (default),
        renders offscreen and saves PNG.
    projection_method:
        If set, projects spike points to 2D using the named method and renders
        a 2D scatter + heatmap figure instead of the 3D plot. Pass None (default)
        for the 3D rendering path.
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
    camera_settings:
        Dict with keys ``camera_position``, ``focal_point``, ``up_vector``,
        ``view_angle`` as saved by the RF Camera Settings Viewer. Applied directly
        to the PyVista camera. When None, falls back to isometric view.
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

    if render_context is None or len(render_context.neuron_contacts_xyz) == 0:
        raise ValueError(
            "render_forearm_heatmap: render_context with non-empty neuron_contacts_xyz "
            f"is required (session={session_id}, cluster={cluster_label}). "
            "Pipeline contract violation — see rf_cluster_pipeline.py."
        )
    projection_centroid = render_context.neuron_contacts_xyz.mean(axis=0)

    if spike_counts_df.empty:
        logger.warning(
            "Empty spike_counts_df for session %s, cluster %s — skipping render.",
            session_id, cluster_label,
        )
        return

    # --- 2D projection path (early return) ---
    if projection_method is not None:
        rotation_matrix = None
        if camera_settings is not None:
            rotation_matrix = camera_settings_to_rotation(camera_settings)

        slim_cache_path = None
        if projection_method == "slim" and forearm_ply_path is not None:
            slim_cache_path = forearm_ply_path.with_name(
                forearm_ply_path.stem + "_slim_uv.npz"
            )

        spike_xyz = spike_counts_df[['x', 'y', 'z']].to_numpy()
        counts = spike_counts_df['spike_count'].to_numpy()

        forearm_vertices = None
        if forearm_ply_path is not None:
            try:
                forearm_vertices = load_forearm_vertices(forearm_ply_path)
            except Exception:
                logger.warning("Could not load forearm PLY for 2D projection: %s", forearm_ply_path, exc_info=True)

        uv_points = project_to_2d(spike_xyz, forearm_vertices, projection_centroid, method=projection_method, rotation_matrix=rotation_matrix, slim_cache_path=slim_cache_path)

        forearm_uv = None
        if forearm_vertices is not None:
            try:
                forearm_uv = project_to_2d(forearm_vertices, forearm_vertices, projection_centroid, method=projection_method, rotation_matrix=rotation_matrix, slim_cache_path=slim_cache_path)
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
        if len(render_context.neuron_contacts_xyz) > 0:
            try:
                neuron_contacts_uv = project_to_2d(
                    render_context.neuron_contacts_xyz,
                    forearm_vertices,
                    projection_centroid,
                    method=projection_method,
                    rotation_matrix=rotation_matrix,
                    slim_cache_path=slim_cache_path,
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
                    projection_centroid,
                    method=projection_method,
                    rotation_matrix=rotation_matrix,
                    slim_cache_path=slim_cache_path,
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

    # --- 3D PyVista rendering path ---
    plotter = pv.Plotter(off_screen=not interactive, window_size=[2000, 1600])
    plotter.set_background("black")

    spike_xyz = spike_counts_df[['x', 'y', 'z']].to_numpy()

    if display_metric == "spike_ratio":
        scalar_values = (
            spike_counts_df['unique_touch_spike_count'].to_numpy()
            / render_context.neuron_cluster_touches
        ).astype(float)
        clim = [0.0, 1.0]
        cbar_label = "Spike ratio"
        scalar_bar_args = {"title": cbar_label, "color": "white"}
        use_log = False
    else:
        scalar_values = spike_counts_df['spike_count'].to_numpy().astype(float)
        cbar_label = "Spike count"
        scalar_bar_args = {"title": cbar_label, "color": "white"}
        use_log = True

    forearm_mesh = None
    if forearm_ply_path is None:
        logger.warning("No forearm PLY resolved for session %s", session_id)
    elif not forearm_ply_path.exists():
        logger.warning("Forearm PLY not found: %s", forearm_ply_path)
    else:
        try:
            forearm_mesh = load_or_build_forearm_mesh(forearm_ply_path)
        except Exception:
            logger.warning("Could not load forearm mesh: %s", forearm_ply_path, exc_info=True)

    if forearm_mesh is not None:
        mesh_pv = mesh_to_pyvista(forearm_mesh)
        per_vertex = map_scalars_to_mesh(forearm_mesh, spike_xyz, scalar_values)

        valid = per_vertex[~np.isnan(per_vertex)]
        if use_log and len(valid) > 0:
            vmin = max(1.0, float(np.nanmin(valid)))
            vmax = float(np.nanmax(valid))
            if vmin < vmax:
                per_vertex_plot = np.where(np.isnan(per_vertex), np.nan, np.log1p(per_vertex))
                clim = [np.log1p(vmin), np.log1p(vmax)]
            else:
                per_vertex_plot = per_vertex
                clim = [vmin, vmax]
        else:
            per_vertex_plot = per_vertex

        plotter.add_mesh(
            mesh_pv,
            scalars=per_vertex_plot,
            cmap="RdYlBu_r",
            nan_color="lightgrey",
            smooth_shading=True,
            clim=clim,
            scalar_bar_args=scalar_bar_args,
        )
    else:
        # Point cloud fallback when mesh build fails
        forearm_vertices = None
        if forearm_ply_path is not None and forearm_ply_path.exists():
            try:
                forearm_vertices = load_forearm_vertices(forearm_ply_path)
            except Exception:
                logger.warning("Could not load forearm PLY vertices: %s", forearm_ply_path, exc_info=True)

        if forearm_vertices is not None:
            cloud = pv.PolyData(forearm_vertices)
            plotter.add_mesh(cloud, color="lightgrey", point_size=3, render_points_as_spheres=False, name="forearm")

        if use_log and len(scalar_values) > 0:
            vmin = max(1.0, float(np.nanmin(scalar_values)))
            vmax = float(np.nanmax(scalar_values))
            if vmin < vmax:
                scalar_vals_plot = np.log1p(scalar_values)
                clim = [np.log1p(vmin), np.log1p(vmax)]
            else:
                scalar_vals_plot = scalar_values
                clim = [vmin, vmax]
        else:
            scalar_vals_plot = scalar_values

        contact_cloud = pv.PolyData(spike_xyz)
        contact_cloud["scalars"] = scalar_vals_plot
        plotter.add_mesh(
            contact_cloud,
            scalars="scalars",
            cmap="RdYlBu_r",
            point_size=8,
            clim=clim,
            scalar_bar_args=scalar_bar_args,
        )

    # --- Convex hull wireframes ---
    if render_context is not None:
        for pts, color in [
            (render_context.neuron_contacts_xyz, '#00aaff'),
            (render_context.neuron_cluster_contacts_xyz, '#ffaa00'),
        ]:
            hull_mesh = _pyvista_hull_mesh(pts)
            if hull_mesh is not None:
                plotter.add_mesh(hull_mesh, color=color, style="wireframe", line_width=1.5)

    # --- Text overlays ---
    if render_context is not None:
        metadata_text = _format_metadata_overlay(render_context)
        plotter.add_text(metadata_text, position="lower_left", font_size=10, color="#aaaaaa")

    title_text = f"RF Heatmap — {session_id} | cluster {cluster_label}"
    if cluster_description:
        title_text = f"{title_text}\n{cluster_description}"
    plotter.add_text(title_text, position="upper_left", font_size=11, color="white")

    # --- Camera ---
    if camera_settings is not None:
        plotter.camera.position = camera_settings["camera_position"]
        plotter.camera.focal_point = camera_settings["focal_point"]
        plotter.camera.up = camera_settings["up_vector"]
        plotter.camera.view_angle = camera_settings["view_angle"]
        plotter.renderer.ResetCameraClippingRange()
    else:
        plotter.view_isometric()
        plotter.reset_camera()

    # --- Output ---
    if interactive:
        plotter.show()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(str(output_path))
        logger.info("Saved heatmap: %s", output_path)
    plotter.close()
