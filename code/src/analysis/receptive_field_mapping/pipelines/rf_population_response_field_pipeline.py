"""Per-session pipeline that renders population RF heatmap PNGs via SLIM UV."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from scipy.spatial import KDTree

from analysis.receptive_field_mapping.data.rf_data_loader import (
    load_forearm_vertex_colors,
    resolve_forearm_ply,
)
from analysis.receptive_field_mapping.surface.forearm_slim_uv import (
    load_slim_uv_cache,
    uv_points_to_xyz,
)
from analysis.receptive_field_mapping.data.touch_population_data import (
    load_population_data,
    load_population_rf_data,
)
from analysis.pipeline.shared_constants import GESTURE_TYPES, IFF_METRICS, single_touch_npz_filename
from analysis.receptive_field_mapping.data.rf_population_heatmap import (
    apply_vertex_threshold,
    build_gesture_touch_indices,
    compute_rf_heatmap,
    compute_threshold_from_ratio,
    compute_unique_touch_count,
)
from analysis.receptive_field_mapping.metrics.rf_inflection_boundary import (
    compute_inflection_boundary,
    compute_laplacian_arrays,
    inflection_boundary_to_dict,
)
from analysis.receptive_field_mapping.metrics.rf_gradient_boundary import (
    compute_gradient_magnitude,
    compute_gradient_ridge,
    gradient_boundary_to_dict,
)
from analysis.receptive_field_mapping.metrics.rf_pca_alignment import (
    apply_uv_alignment,
    compute_rf_pca_alignment,
)
from analysis.receptive_field_mapping.rendering.rf_population_map_renderer import (
    compute_highest_contour_center,
    compute_interpolated_grid,
    compute_standalone_figwidth,
    compute_uv_to_mm_scale,
    render_population_rf_circular_crop,
    render_population_rf_map,
    render_population_rf_composite,
    render_population_rf_standalone_interpolated,
    render_population_rf_colorbar,
)
from analysis.pipeline.output_dirs import (
    SPATIAL_EXTRACT_BOUNDARIES,
    SPATIAL_MAP_SINGLE_TOUCH,
    SPATIAL_SLIM_UV,
    TOUCH_COMPUTE_SERIES,
)
from analysis.pipeline.shared_constants import session_id_from_path

logger = logging.getLogger(__name__)


@dataclass
class _SessionCompositeData:
    session_id: str
    output_dir: Path
    forearm_uv: np.ndarray
    forearm_faces: np.ndarray
    forearm_V: np.ndarray
    results: dict
    session_vmax: float
    session_vmin: float
    min_overlap_pct: float
    sentinel: Path
    produced: List[Path] = field(default_factory=list)
    gesture_boundaries: dict = field(default_factory=dict)
    gesture_inflection_boundaries: dict = field(default_factory=dict)
    gesture_gradient_boundaries: dict = field(default_factory=dict)
    per_gesture_grids: dict = field(default_factory=dict)
    vertex_data_npz: Path | None = None
    alignment_center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    alignment_rotation_matrix: np.ndarray = field(default_factory=lambda: np.eye(2))
    alignment_angle_deg: float = 0.0
    forearm_ply_path: Path | None = field(default=None)
    slim_vertex_colors: np.ndarray | None = field(default=None)


def run_population_response_field_extraction(
    session_configs: list,
    neuron_mode: str,
    output_dir: Path,
    min_overlap_pct: float = 25.0,
    force_processing: bool = False,
    median_filter_size: int | None = None,
    inflection_sigma: float | None = None,
    heatmap_space: str = "linear",
    cmap: str = "inferno",
    iff_metric: str = "mean",
    flip_u: bool = False,
    contour_color: str = "red",
    circular_crop_margin: float = 0.0,
    boundary_method: str = "gradient",
) -> None:
    """Render per-session 2D population RF heatmap PNGs projected via SLIM UV.

    For each session config, produces one PNG per gesture subset (all, stroke,
    tap, stroke_proximal, stroke_distal) plus two composite PNGs (scatter and
    interpolated) under ``4_analysed/spatial_extract_boundaries/{session_id}/``.
    ``stroke`` is a virtual subset combining stroke_proximal + stroke_distal.

    Composites use a global colour scale and UV axis range across all sessions
    so they are directly comparable.

    Parameters
    ----------
    session_configs:
        List of ``(aggregated_csv_path, database_path)`` tuples.
    neuron_mode:
        ``"iff"`` or ``"spike"`` — must match the mode used by
        ``run_single_touch_rf_mapping``.
    output_dir:
        Root output directory for this task
        (e.g. ``database_path / '4_analysed' / SPATIAL_EXTRACT_BOUNDARIES``).
    min_overlap_pct:
        Minimum percentage of touches that must contact a vertex for it to be
        included in the heatmap (default 25 %).
    force_processing:
        If True, reprocess sessions even when the sentinel file exists.
    inflection_sigma:
        Gaussian smoothing sigma for Laplacian inflection boundary detection.
        Pass ``None`` to disable boundary computation entirely.
    iff_metric:
        Which IFF aggregation NPZ to consume — ``"mean"`` (default) or
        ``"max"``.  Must be one of ``IFF_METRICS``.
    flip_u:
        If True, negate the U-axis (column 0) of the aligned forearm UV
        coordinates after PCA alignment. Mirrors the heatmap and all
        boundary metrics along the vertical axis of the output plots.
    """
    if iff_metric not in IFF_METRICS:
        raise ValueError(
            f"run_population_response_field_extraction: invalid iff_metric "
            f"{iff_metric!r}. Expected one of {IFF_METRICS}."
        )
    if boundary_method not in ("gradient", "inflection"):
        raise ValueError(
            f"run_population_response_field_extraction: invalid boundary_method "
            f"{boundary_method!r}. Expected 'gradient' or 'inflection'."
        )
    npz_filename = single_touch_npz_filename(iff_metric)
    # ---- Pass 1: compute heatmaps + render per-gesture PNGs ----
    composite_queue: List[_SessionCompositeData] = []

    for csv_path, database_path in session_configs:
        csv_path = Path(csv_path)
        database_path = Path(database_path)

        session_id = session_id_from_path(csv_path)
        session_output_dir = output_dir / session_id
        sentinel = session_output_dir / f'{session_id}_population_response_fields_done.json'

        if sentinel.exists() and not force_processing:
            print(f"[Population Response Fields] {session_id}: up-to-date, skipping.")
            continue

        print(f"[Population Response Fields] {session_id}: processing...")

        # --- Resolve paths ---
        series_csv_path = (
            database_path / '4_analysed' / TOUCH_COMPUTE_SERIES
            / f'{session_id}_series_augmented.csv'
        )
        if not series_csv_path.exists():
            raise FileNotFoundError(
                f"[Population Response Fields] {session_id}: series-augmented CSV not found — "
                f"run 'touch_compute_series' first: {series_csv_path}"
            )

        forearm_ply_path = resolve_forearm_ply(csv_path.parent, session_id)
        if forearm_ply_path is None:
            raise FileNotFoundError(
                f"[Population Response Fields] {session_id}: forearm PLY not found in "
                f"{csv_path.parent} — RF-centred PLY must exist."
            )

        npz_path = (
            database_path / '4_analysed' / SPATIAL_MAP_SINGLE_TOUCH
            / session_id / npz_filename
        )
        if not npz_path.exists():
            raise FileNotFoundError(
                f"[Population Response Fields] {session_id}: {npz_filename} not found: "
                f"{npz_path}. Enable 'spatial_map_single_touch' in the DAG config and re-run."
            )

        slim_cache_path = (
            database_path / '4_analysed' / SPATIAL_SLIM_UV
            / session_id / f'{session_id}_slim_uv.npz'
        )
        if not slim_cache_path.exists():
            raise FileNotFoundError(
                f"[Population Response Fields] {session_id}: SLIM UV cache not found: "
                f"{slim_cache_path}. Enable 'spatial_precompute_slim_uv' in the DAG "
                f"config and re-run to generate the cache."
            )

        # --- Load data ---
        pop_data = load_population_data(series_csv_path, forearm_ply_path)
        n_verts = len(pop_data.forearm_vertices)
        rf_data = load_population_rf_data(npz_path, pop_data.touch_triple_keys, n_verts)

        # --- Load SLIM UV cache ---
        cache = load_slim_uv_cache(slim_cache_path)
        forearm_uv = cache.uv
        slim_V = cache.V
        slim_faces = cache.F

        # --- Build KDTree mapping: original forearm vertices → SLIM vertices ---
        # The SLIM cache stores cleaned-mesh vertices (after BPA + clean_mesh +
        # flatten_slim), which can include synthetic centroid vertices from
        # interior hole filling.  These vertices are not in the raw PLY, so a
        # handful will be >1 mm from their nearest raw neighbour.  Genuine
        # misalignment (wrong PLY) would show distances of 10+ mm.
        orig_tree = KDTree(pop_data.forearm_vertices)
        distances, nearest_orig_for_slim = orig_tree.query(slim_V)
        max_dist_mm = float(distances.max())
        _WARN_THRESHOLD_MM = 1.0
        _ERROR_THRESHOLD_MM = 5.0
        if max_dist_mm > _ERROR_THRESHOLD_MM:
            raise ValueError(
                f"[Population Response Fields] {session_id}: KDTree nearest-neighbour "
                f"mapping from SLIM vertices to original forearm vertices has a "
                f"maximum distance of {max_dist_mm:.3f} mm (threshold: "
                f"{_ERROR_THRESHOLD_MM} mm). The SLIM mesh and forearm PLY are "
                "misaligned — re-run 'spatial_precompute_slim_uv' after "
                "verifying the forearm PLY."
            )
        if max_dist_mm > _WARN_THRESHOLD_MM:
            n_over = int((distances > _WARN_THRESHOLD_MM).sum())
            logger.info(
                "[Population Response Fields] %s: %d / %d SLIM vertices are >%.0f mm "
                "from the nearest raw PLY vertex (max=%.3f mm) — expected for "
                "hole-fill centroid vertices.",
                session_id, n_over, len(slim_V),
                _WARN_THRESHOLD_MM, max_dist_mm,
            )

        # --- Load PLY vertex colors and map to SLIM vertices via KDTree ---
        raw_colors = load_forearm_vertex_colors(forearm_ply_path)
        if raw_colors is not None:
            slim_colors_rgb = raw_colors[nearest_orig_for_slim].astype(np.float64) / 255.0
            alpha_col = np.ones((len(slim_colors_rgb), 1), dtype=np.float64)
            slim_vertex_colors = np.hstack([slim_colors_rgb, alpha_col])
        else:
            slim_vertex_colors = None

        # --- Compute heatmaps for all gesture subsets ---
        subsets = ['all'] + list(GESTURE_TYPES)
        results: dict = {}

        for gtype in subsets:
            if gtype == 'all':
                gesture_touch_indices = np.arange(len(pop_data.touch_triple_keys))
            else:
                gesture_touch_indices = build_gesture_touch_indices(
                    pop_data.gesture_types, gtype
                )

            n_gesture_touches = len(gesture_touch_indices)
            if n_gesture_touches == 0:
                logger.warning(
                    "[Population Response Fields] %s: no touches for gesture type '%s' — skipping.",
                    session_id, gtype,
                )
                continue

            cp_mask = np.isin(pop_data.cp_touch_idx, gesture_touch_indices)
            heatmap = compute_rf_heatmap(
                gesture_touch_indices,
                rf_data.rf_vertex_indices,
                rf_data.rf_values,
                n_verts,
            )
            unique_count = compute_unique_touch_count(
                pop_data.cp_vertex_idx,
                pop_data.cp_touch_idx,
                cp_mask,
                n_verts,
            )
            threshold = compute_threshold_from_ratio(min_overlap_pct, n_gesture_touches)
            thresholded = apply_vertex_threshold(heatmap, unique_count, threshold)
            slim_heatmap = thresholded[nearest_orig_for_slim]

            results[gtype] = (slim_heatmap, n_gesture_touches, threshold)

        # --- Synthesize 'stroke' = stroke_proximal + stroke_distal ---
        _sp_in = 'stroke_proximal' in results
        _sd_in = 'stroke_distal' in results
        if _sp_in or _sd_in:
            parts = []
            if _sp_in:
                parts.append(build_gesture_touch_indices(pop_data.gesture_types, 'stroke_proximal'))
            if _sd_in:
                parts.append(build_gesture_touch_indices(pop_data.gesture_types, 'stroke_distal'))
            stroke_touch_indices = np.concatenate(parts)

            n_stroke_touches = len(stroke_touch_indices)
            if n_stroke_touches > 0:
                cp_mask = np.isin(pop_data.cp_touch_idx, stroke_touch_indices)
                heatmap = compute_rf_heatmap(
                    stroke_touch_indices,
                    rf_data.rf_vertex_indices,
                    rf_data.rf_values,
                    n_verts,
                )
                unique_count = compute_unique_touch_count(
                    pop_data.cp_vertex_idx,
                    pop_data.cp_touch_idx,
                    cp_mask,
                    n_verts,
                )
                threshold = compute_threshold_from_ratio(min_overlap_pct, n_stroke_touches)
                thresholded = apply_vertex_threshold(heatmap, unique_count, threshold)
                slim_heatmap = thresholded[nearest_orig_for_slim]
                results['stroke'] = (slim_heatmap, n_stroke_touches, threshold)

        _canonical_order = ['all', 'stroke', 'tap', 'stroke_proximal', 'stroke_distal']
        results = {k: results[k] for k in _canonical_order if k in results}

        if not results:
            logger.warning(
                "[Population Response Fields] %s: no gesture subsets had touches — no PNGs produced.",
                session_id,
            )
            session_output_dir.mkdir(parents=True, exist_ok=True)
            _write_sentinel(sentinel, session_id, produced=[])
            continue

        if 'all' not in results:
            raise ValueError(
                f"[Population Response Fields] {session_id}: 'all' gesture type missing from "
                f"results — cannot compute PCA alignment. This should not happen."
            )

        all_heatmap, _, _ = results['all']
        alignment_center, alignment_rotation_matrix, alignment_angle_deg = compute_rf_pca_alignment(
            forearm_uv, all_heatmap
        )
        logger.info(
            "[Population Response Fields] %s: PCA alignment — center=(%.3f, %.3f) angle=%.1f°",
            session_id, alignment_center[0], alignment_center[1], alignment_angle_deg,
        )
        forearm_uv = apply_uv_alignment(forearm_uv, alignment_center, alignment_rotation_matrix)

        if flip_u:
            forearm_uv[:, 0] *= -1
            logger.info(
                "[Population Response Fields] %s: U-axis flipped (flip_u=True)",
                session_id,
            )

        # --- Render per-gesture PNGs with session-wide colour scale ---
        finite_maxima = [
            float(np.nanmax(h))
            for (h, _, _) in results.values()
            if np.any(np.isfinite(h) & (h >= 0))
        ]
        if not finite_maxima:
            raise ValueError(
                f"[Population Response Fields] {session_id}: all heatmaps are empty or "
                f"below-threshold — no valid colour scale can be determined. "
                f"Check upstream RF data."
            )
        session_vmax = max(finite_maxima)
        finite_minima = [
            float(np.nanmin(h[np.isfinite(h) & (h > 0)]))
            for (h, _, _) in results.values()
            if np.any(np.isfinite(h) & (h > 0))
        ]
        session_vmin = min(finite_minima) if finite_minima else session_vmax * 1e-3

        session_output_dir.mkdir(parents=True, exist_ok=True)
        aggregated_dir = session_output_dir / "aggregated"
        aggregated_dir.mkdir(parents=True, exist_ok=True)
        inspection_dir = session_output_dir / "inspection"
        inspection_dir.mkdir(parents=True, exist_ok=True)
        produced: List[Path] = []
        gesture_boundaries: dict = {}
        gesture_inflection_boundaries: dict = {}
        gesture_gradient_boundaries: dict = {}
        per_gesture_grids: dict = {}

        for gtype, (slim_heatmap, n_touches, threshold) in results.items():
            title = (
                f"{session_id} | {gtype} | {n_touches} touches | "
                f"threshold={threshold} ({min_overlap_pct:.0f}%)"
            )
            png_path = aggregated_dir / f'{session_id}_rf_population_{gtype}.png'

            print(f"[Population Response Fields] {session_id}: rendering '{gtype}'...")
            grid_u, grid_v, grid_z = compute_interpolated_grid(
                forearm_uv, slim_faces, slim_V, slim_heatmap,
                median_filter_size=median_filter_size,
            )
            per_gesture_grids[gtype] = (grid_u, grid_v, grid_z)
            boundary = (
                compute_inflection_boundary(
                    grid_u, grid_v, grid_z, inflection_sigma,
                    snapshot_dir=inspection_dir, snapshot_label=gtype,
                    contour_color=contour_color,
                )
                if inflection_sigma is not None
                else None
            )
            gesture_inflection_boundaries[gtype] = boundary
            if inflection_sigma is not None:
                smoothed_for_grad, _ = compute_laplacian_arrays(grid_z, inflection_sigma)
                grad_boundary = compute_gradient_ridge(
                    grid_u, grid_v, grid_z, smoothed_for_grad,
                    snapshot_dir=inspection_dir, snapshot_label=gtype,
                )
            else:
                grad_boundary = None
            gesture_gradient_boundaries[gtype] = grad_boundary
            if boundary_method == "gradient":
                gesture_boundaries[gtype] = grad_boundary
            else:
                gesture_boundaries[gtype] = boundary
            render_population_rf_map(
                forearm_uv=forearm_uv,
                heatmap_val=slim_heatmap,
                vmax=session_vmax,
                vmin=session_vmin,
                title=title,
                output_path=png_path,
                forearm_faces=slim_faces,
                forearm_V=slim_V,
                median_filter_size=median_filter_size,
                precomputed_grid=(grid_u, grid_v, grid_z),
                boundary=gesture_boundaries[gtype],
                heatmap_space=heatmap_space,
                cmap=cmap,
                vertex_colors=slim_vertex_colors,
                contour_color=contour_color,
            )
            produced.append(png_path)
            print(f"[Population Response Fields] {session_id}: saved {png_path.name}")

        vertex_data_npz = _save_response_fields_npz(
            output_dir=session_output_dir,
            session_id=session_id,
            forearm_uv=forearm_uv,
            forearm_faces=slim_faces,
            forearm_V=slim_V,
            results=results,
            per_gesture_grids=per_gesture_grids,
            neuron_mode=neuron_mode,
            min_overlap_pct=min_overlap_pct,
            gesture_boundaries=gesture_boundaries,
            gesture_gradient_boundaries=gesture_gradient_boundaries,
            gesture_inflection_boundaries=gesture_inflection_boundaries,
            boundary_method=boundary_method,
            inflection_sigma=inflection_sigma,
            alignment_center=alignment_center,
            alignment_rotation_matrix=alignment_rotation_matrix,
            alignment_angle_deg=alignment_angle_deg,
            flip_u=flip_u,
            slim_vertex_colors=slim_vertex_colors,
        )
        _write_sentinel(
            sentinel, session_id, produced=produced,
            inflection_boundaries=gesture_boundaries,
            vertex_data_npz=vertex_data_npz,
        )
        print(
            f"[Population Response Fields] {session_id}: done — {len(produced)} PNG(s) written."
        )

        composite_queue.append(_SessionCompositeData(
            session_id=session_id,
            output_dir=session_output_dir,
            forearm_uv=forearm_uv,
            forearm_faces=slim_faces,
            forearm_V=slim_V,
            results=results,
            session_vmax=session_vmax,
            session_vmin=session_vmin,
            min_overlap_pct=min_overlap_pct,
            sentinel=sentinel,
            produced=produced,
            gesture_boundaries=gesture_boundaries,
            gesture_inflection_boundaries=gesture_inflection_boundaries,
            gesture_gradient_boundaries=gesture_gradient_boundaries,
            per_gesture_grids=per_gesture_grids,
            vertex_data_npz=vertex_data_npz,
            alignment_center=alignment_center,
            alignment_rotation_matrix=alignment_rotation_matrix,
            alignment_angle_deg=alignment_angle_deg,
            forearm_ply_path=forearm_ply_path,
            slim_vertex_colors=slim_vertex_colors,
        ))

    # ---- Pass 2: render composite PNGs with global colour scale + UV limits ----
    if not composite_queue:
        return

    # Pre-compute all standalone titles and measure the longest to get a shared
    # figure width, so every *_interpolated.png has identical pixel dimensions.
    standalone_titles: dict[tuple[str, str], str] = {}
    for sd in composite_queue:
        for gtype, (_, n_touches, threshold) in sd.results.items():
            standalone_titles[(sd.session_id, gtype)] = (
                f"{sd.session_id} | {gtype} | {n_touches} touches | "
                f"threshold={threshold} ({sd.min_overlap_pct:.0f}%)"
            )
    longest_title = max(standalone_titles.values(), key=len) if standalone_titles else ""
    standalone_figwidth = compute_standalone_figwidth(longest_title) if longest_title else 6.0

    global_vmax = max(sd.session_vmax for sd in composite_queue)
    _all_grid_positive = [
        float(grid_z[np.isfinite(grid_z) & (grid_z > 0)].min())
        for sd in composite_queue
        for _, (_, _, grid_z) in sd.per_gesture_grids.items()
        if np.any(np.isfinite(grid_z) & (grid_z > 0))
    ]
    global_vmin = min(_all_grid_positive) if _all_grid_positive else global_vmax * 1e-3
    all_u = np.concatenate([sd.forearm_uv[:, 0] for sd in composite_queue])
    all_v = np.concatenate([sd.forearm_uv[:, 1] for sd in composite_queue])
    uv_margin = 0.02
    u_range = all_u.max() - all_u.min()
    v_range = all_v.max() - all_v.min()
    global_uv_xlim = (float(all_u.min() - uv_margin * u_range),
                      float(all_u.max() + uv_margin * u_range))
    global_uv_ylim = (float(all_v.min() - uv_margin * v_range),
                      float(all_v.max() + uv_margin * v_range))
    del all_u, all_v

    print(
        f"[Population Response Fields] Rendering composites: global_vmax={global_vmax:.2f}, "
        f"U=[{global_uv_xlim[0]:.1f}, {global_uv_xlim[1]:.1f}], "
        f"V=[{global_uv_ylim[0]:.1f}, {global_uv_ylim[1]:.1f}]"
    )

    for sd in composite_queue:
        aggregated_dir = sd.output_dir / "aggregated"
        aggregated_dir.mkdir(parents=True, exist_ok=True)
        inspection_dir = sd.output_dir / "inspection"
        inspection_dir.mkdir(parents=True, exist_ok=True)
        for panel_type in ('scatter', 'interpolated'):
            composite_path = (
                aggregated_dir / f'{sd.session_id}_rf_population_{panel_type}_composite.png'
            )
            print(
                f"[Population Response Fields] {sd.session_id}: "
                f"rendering '{panel_type}' composite..."
            )
            if panel_type == 'interpolated':
                precomputed_grids = {}
                inflection_boundaries = {}
                for gtype in sd.results:
                    slim_heatmap_g, _, _ = sd.results[gtype]
                    grid_u_g, grid_v_g, grid_z_g = compute_interpolated_grid(
                        sd.forearm_uv, sd.forearm_faces, sd.forearm_V, slim_heatmap_g,
                        median_filter_size=median_filter_size,
                    )
                    precomputed_grids[gtype] = (grid_u_g, grid_v_g, grid_z_g)
                    if inflection_sigma is not None:
                        if boundary_method == "gradient":
                            smoothed_comp, _ = compute_laplacian_arrays(grid_z_g, inflection_sigma)
                            inflection_boundaries[gtype] = compute_gradient_ridge(
                                grid_u_g, grid_v_g, grid_z_g, smoothed_comp,
                                snapshot_dir=inspection_dir,
                                snapshot_label=f"{sd.session_id}_{gtype}_composite",
                            )
                        else:
                            inflection_boundaries[gtype] = compute_inflection_boundary(
                                grid_u_g, grid_v_g, grid_z_g, inflection_sigma,
                                snapshot_dir=inspection_dir,
                                snapshot_label=f"{sd.session_id}_{gtype}_composite",
                                contour_color=contour_color,
                            )
                    else:
                        inflection_boundaries[gtype] = None
            else:
                precomputed_grids = None
                inflection_boundaries = None
            render_population_rf_composite(
                forearm_uv=sd.forearm_uv,
                results=sd.results,
                vmax=global_vmax,
                vmin=global_vmin,
                session_id=sd.session_id,
                panel_type=panel_type,
                output_path=composite_path,
                min_overlap_pct=sd.min_overlap_pct,
                uv_xlim=global_uv_xlim,
                uv_ylim=global_uv_ylim,
                forearm_faces=sd.forearm_faces,
                forearm_V=sd.forearm_V,
                median_filter_size=median_filter_size,
                precomputed_grids=precomputed_grids,
                boundaries=inflection_boundaries,
                heatmap_space=heatmap_space,
                cmap=cmap,
                vertex_colors=sd.slim_vertex_colors,
                contour_color=contour_color,
            )
            sd.produced.append(composite_path)
            print(
                f"[Population Response Fields] {sd.session_id}: saved {composite_path.name}"
            )

        for gtype, (grid_u, grid_v, grid_z) in sd.per_gesture_grids.items():
            boundary = sd.gesture_boundaries.get(gtype)
            standalone_path = sd.output_dir / f'{sd.session_id}_rf_population_{gtype}_interpolated.png'
            render_population_rf_standalone_interpolated(
                u_grid=grid_u,
                v_grid=grid_v,
                interp_grid=grid_z,
                forearm_uv=sd.forearm_uv,
                boundary_u=boundary.contour_uv[:, 0] if boundary is not None else None,
                boundary_v=boundary.contour_uv[:, 1] if boundary is not None else None,
                output_path=standalone_path,
                vmax=global_vmax,
                vmin=global_vmin,
                title=standalone_titles[(sd.session_id, gtype)],
                figwidth=standalone_figwidth,
                xlim=global_uv_xlim,
                ylim=global_uv_ylim,
                heatmap_space=heatmap_space,
                cmap=cmap,
                vertex_colors=sd.slim_vertex_colors,
                forearm_faces=sd.forearm_faces,
                contour_color=contour_color,
            )
            sd.produced.append(standalone_path)
            print(f"[Population Response Fields] {sd.session_id}: saved {standalone_path.name}")

        colorbar_path = sd.output_dir / f'{sd.session_id}_rf_population_colorbar.png'
        render_population_rf_colorbar(output_path=colorbar_path, vmax=global_vmax, vmin=global_vmin, heatmap_space=heatmap_space, cmap=cmap)
        sd.produced.append(colorbar_path)
        print(f"[Population Response Fields] {sd.session_id}: saved {colorbar_path.name}")

        colorbar_local_path = sd.output_dir / f'{sd.session_id}_rf_population_colorbar_local.png'
        render_population_rf_colorbar(
            output_path=colorbar_local_path,
            vmax=sd.session_vmax,
            vmin=sd.session_vmin,
            heatmap_space=heatmap_space,
            cmap=cmap,
        )
        sd.produced.append(colorbar_local_path)
        print(f"[Population Response Fields] {sd.session_id}: saved {colorbar_local_path.name}")

        all_boundary = sd.gesture_boundaries.get('all')
        if all_boundary is None:
            raise ValueError(
                "render_population_rf_circular_crop requires an inflection boundary for gesture 'all' "
                "but none is available — ensure inflection_sigma is configured"
            )
        all_grid_u, all_grid_v, all_grid_z = sd.per_gesture_grids['all']

        centroid_uv = np.array(all_boundary.centroid_uv)
        peak_uv = np.array(all_boundary.peak_uv)
        contour_center_uv = compute_highest_contour_center(
            all_grid_u, all_grid_v, all_grid_z, n_levels=6,
        )

        crop_jobs: list[tuple[str, np.ndarray, np.ndarray | None]] = [
            ('centroid', centroid_uv, centroid_uv),
            ('peak', peak_uv, peak_uv),
        ]
        if contour_center_uv is not None:
            crop_jobs.append(('contour_center', contour_center_uv, contour_center_uv))

        radius_mm = 50.0
        scale = compute_uv_to_mm_scale(sd.forearm_uv, sd.forearm_V, sd.forearm_faces)
        radius_uv = radius_mm / scale
        margin = radius_uv * circular_crop_margin
        all_centers = np.array([c for _, c, _ in crop_jobs])
        shared_xlim = (
            float(all_centers[:, 0].min()) - radius_uv - margin,
            float(all_centers[:, 0].max()) + radius_uv + margin,
        )
        shared_ylim = (
            float(all_centers[:, 1].min()) - radius_uv - margin,
            float(all_centers[:, 1].max()) + radius_uv + margin,
        )

        for center_label, center_uv, marker_uv in crop_jobs:
            for vmax_val, vmin_val, suffix in (
                (global_vmax, global_vmin, ''),
                (sd.session_vmax, sd.session_vmin, '_local'),
            ):
                for use_marker, marker_suffix in ((True, ''), (False, '_clean')):
                    out_path = (
                        sd.output_dir
                        / f'{sd.session_id}_rf_population_all_circular_{center_label}{marker_suffix}{suffix}.png'
                    )
                    print(
                        f"[Population Response Fields] {sd.session_id}: "
                        f"rendering 'all' circular {center_label}{marker_suffix}{suffix}..."
                    )
                    crop_kwargs = dict(
                        u_grid=all_grid_u,
                        v_grid=all_grid_v,
                        interp_grid=all_grid_z,
                        forearm_uv=sd.forearm_uv,
                        forearm_V=sd.forearm_V,
                        forearm_faces=sd.forearm_faces,
                        center_uv=center_uv,
                        radius_mm=radius_mm,
                        vmax=vmax_val,
                        vmin=vmin_val,
                        vertex_colors=sd.slim_vertex_colors,
                        heatmap_space=heatmap_space,
                        cmap=cmap,
                        contour_levels=6,
                        centroid_uv=marker_uv if use_marker else None,
                        xlim=shared_xlim,
                        ylim=shared_ylim,
                    )
                    # Original crop — preserved unchanged (no RF boundary overlay).
                    render_population_rf_circular_crop(output_path=out_path, **crop_kwargs)
                    sd.produced.append(out_path)
                    print(f"[Population Response Fields] {sd.session_id}: saved {out_path.name}")
                    # Duplicate crop with the red closed RF boundary overlaid.
                    boundary_path = out_path.with_name(f'{out_path.stem}_rfboundary{out_path.suffix}')
                    render_population_rf_circular_crop(
                        output_path=boundary_path,
                        boundary_contour_uv=all_boundary.contour_uv,
                        boundary_color=contour_color,
                        **crop_kwargs,
                    )
                    sd.produced.append(boundary_path)
                    print(f"[Population Response Fields] {sd.session_id}: saved {boundary_path.name}")

        _write_sentinel(sd.sentinel, sd.session_id, produced=sd.produced,
                        inflection_boundaries=sd.gesture_boundaries,
                        vertex_data_npz=sd.vertex_data_npz)


def _save_boundary_fields(
    data_dict: dict,
    prefix: str,
    gtype: str,
    boundary,
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    forearm_V: np.ndarray,
) -> None:
    data_dict[f'{prefix}_contour_uv_{gtype}'] = boundary.contour_uv.astype(np.float64)
    data_dict[f'{prefix}_centroid_uv_{gtype}'] = np.array(boundary.centroid_uv, dtype=np.float64)
    data_dict[f'{prefix}_peak_uv_{gtype}'] = np.array(boundary.peak_uv, dtype=np.float64)
    data_dict[f'{prefix}_perimeter_uv_{gtype}'] = np.float64(boundary.perimeter_uv)
    data_dict[f'{prefix}_area_uv_{gtype}'] = np.float64(boundary.area_uv)
    data_dict[f'{prefix}_circularity_{gtype}'] = np.float64(boundary.circularity)
    data_dict[f'{prefix}_pca_major_uv_{gtype}'] = np.float64(boundary.pca_major_uv)
    data_dict[f'{prefix}_pca_minor_uv_{gtype}'] = np.float64(boundary.pca_minor_uv)
    data_dict[f'{prefix}_pca_orientation_deg_{gtype}'] = np.float64(boundary.pca_orientation_deg)
    data_dict[f'{prefix}_mean_iff_on_contour_{gtype}'] = np.float64(boundary.mean_iff_on_contour)
    data_dict[f'{prefix}_iff_at_centroid_{gtype}'] = np.float64(boundary.iff_at_centroid)

    contour_xyz = uv_points_to_xyz(
        boundary.contour_uv, forearm_uv, forearm_faces, forearm_V,
    )
    centroid_uv_arr = np.array(boundary.centroid_uv, dtype=np.float64).reshape(1, 2)
    centroid_xyz = uv_points_to_xyz(
        centroid_uv_arr, forearm_uv, forearm_faces, forearm_V,
    )[0]

    contour_xyz_closed = np.vstack([contour_xyz, contour_xyz[:1]])
    perimeter_xyz_mm = float(
        np.sum(np.linalg.norm(np.diff(contour_xyz_closed, axis=0), axis=1))
    )

    c = centroid_xyz
    edges_i = contour_xyz[:-1] - c
    edges_j = contour_xyz[1:] - c
    closing_i = contour_xyz[-1] - c
    closing_j = contour_xyz[0] - c
    cross_vecs = np.vstack([
        np.cross(edges_i, edges_j),
        np.cross(closing_i, closing_j).reshape(1, 3),
    ])
    area_xyz_mm2 = 0.5 * float(np.sum(np.linalg.norm(cross_vecs, axis=1)))

    peak_uv_arr = np.array(boundary.peak_uv, dtype=np.float64).reshape(1, 2)
    peak_xyz = uv_points_to_xyz(
        peak_uv_arr, forearm_uv, forearm_faces, forearm_V,
    )[0]

    data_dict[f'{prefix}_contour_xyz_{gtype}'] = contour_xyz.astype(np.float64)
    data_dict[f'{prefix}_centroid_xyz_{gtype}'] = centroid_xyz.astype(np.float64)
    data_dict[f'{prefix}_peak_xyz_{gtype}'] = peak_xyz.astype(np.float64)
    data_dict[f'{prefix}_perimeter_xyz_mm_{gtype}'] = np.float64(perimeter_xyz_mm)
    data_dict[f'{prefix}_area_xyz_mm2_{gtype}'] = np.float64(area_xyz_mm2)


def _save_response_fields_npz(
    output_dir: Path,
    session_id: str,
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    forearm_V: np.ndarray,
    results: dict,
    per_gesture_grids: dict,
    neuron_mode: str,
    min_overlap_pct: float,
    gesture_boundaries: dict,
    gesture_gradient_boundaries: dict | None = None,
    gesture_inflection_boundaries: dict | None = None,
    boundary_method: str = "gradient",
    inflection_sigma: float | None = None,
    alignment_center: np.ndarray = None,
    alignment_rotation_matrix: np.ndarray = None,
    alignment_angle_deg: float = 0.0,
    flip_u: bool = False,
    slim_vertex_colors: np.ndarray | None = None,
) -> Path:
    npz_path = output_dir / f'{session_id}_population_response_fields.npz'

    data_dict: dict = {
        'forearm_uv': forearm_uv.astype(np.float64),
        'forearm_faces': forearm_faces.astype(np.int32),
        'forearm_V': forearm_V.astype(np.float64),
        'session_id': np.array(session_id, dtype=object),
        'neuron_mode': np.array(neuron_mode, dtype=object),
        'min_overlap_pct': np.float64(min_overlap_pct),
        'gesture_types': np.array(list(results.keys()), dtype=object),
        'inflection_sigma': np.float64(inflection_sigma if inflection_sigma is not None else float('nan')),
        'flip_u': np.bool_(flip_u),
        'boundary_method': np.array(boundary_method, dtype=object),
    }

    for gtype in results.keys():
        slim_heatmap, n_touches, threshold = results[gtype]
        grid_u, grid_v, grid_z = per_gesture_grids[gtype]
        data_dict[f'heatmap_{gtype}'] = slim_heatmap.astype(np.float64)
        data_dict[f'n_touches_{gtype}'] = np.int64(n_touches)
        data_dict[f'threshold_{gtype}'] = np.int64(threshold)
        data_dict[f'grid_u_{gtype}'] = grid_u.astype(np.float64)
        data_dict[f'grid_v_{gtype}'] = grid_v.astype(np.float64)
        data_dict[f'grid_z_{gtype}'] = grid_z.astype(np.float64)

        if inflection_sigma is not None:
            smoothed, lap = compute_laplacian_arrays(grid_z, inflection_sigma)
            data_dict[f'smoothed_{gtype}'] = smoothed.astype(np.float64)
            data_dict[f'laplacian_{gtype}'] = lap.astype(np.float64)

        boundary = gesture_boundaries.get(gtype)
        if boundary is not None:
            _save_boundary_fields(data_dict, 'boundary', gtype, boundary,
                                  forearm_uv, forearm_faces, forearm_V)

    # ---- Inflection boundary (explicit prefix) ----
    if gesture_inflection_boundaries:
        for gtype in results.keys():
            infl_boundary = gesture_inflection_boundaries.get(gtype)
            if infl_boundary is not None:
                _save_boundary_fields(data_dict, 'inflection', gtype, infl_boundary,
                                      forearm_uv, forearm_faces, forearm_V)

    # ---- Gradient ridge boundary ----
    if gesture_gradient_boundaries:
        for gtype in results.keys():
            grad_boundary = gesture_gradient_boundaries.get(gtype)
            if inflection_sigma is not None and f'smoothed_{gtype}' in data_dict:
                nan_mask = np.isnan(per_gesture_grids[gtype][2])
                grad_mag = compute_gradient_magnitude(data_dict[f'smoothed_{gtype}'], nan_mask)
                data_dict[f'gradient_mag_{gtype}'] = grad_mag.astype(np.float64)
            if grad_boundary is not None:
                _save_boundary_fields(data_dict, 'gradient', gtype, grad_boundary,
                                      forearm_uv, forearm_faces, forearm_V)

    data_dict['alignment_center_uv'] = alignment_center.astype(np.float64)
    data_dict['alignment_rotation_matrix'] = alignment_rotation_matrix.astype(np.float64)
    data_dict['alignment_rotation_deg'] = np.float64(alignment_angle_deg)

    if slim_vertex_colors is not None:
        data_dict['slim_vertex_colors'] = slim_vertex_colors.astype(np.float64)

    np.savez(npz_path, **data_dict)
    logger.info("[Population Response Fields] %s: saved response fields NPZ → %s", session_id, npz_path.name)
    return npz_path


def _write_sentinel(
    sentinel: Path,
    session_id: str,
    produced: List[Path],
    inflection_boundaries: dict | None = None,
    vertex_data_npz: Path | None = None,
) -> None:
    serialized_boundaries = {}
    if inflection_boundaries:
        for gtype, b in inflection_boundaries.items():
            serialized_boundaries[gtype] = inflection_boundary_to_dict(b) if b is not None else None

    data = {
        'session_id': session_id,
        'n_pngs': len(produced),
        'pngs': [str(p) for p in produced],
    }
    if serialized_boundaries:
        data['inflection_boundaries'] = serialized_boundaries
    if vertex_data_npz is not None:
        data['vertex_data_npz'] = str(vertex_data_npz)

    with open(sentinel, 'w') as f:
        json.dump(data, f, indent=2)
