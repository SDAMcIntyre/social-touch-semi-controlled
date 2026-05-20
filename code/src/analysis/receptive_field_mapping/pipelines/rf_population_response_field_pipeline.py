"""Per-session pipeline that renders population RF heatmap PNGs via SLIM UV."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from scipy.spatial import KDTree

from analysis.receptive_field_mapping.data.rf_data_loader import resolve_forearm_ply
from analysis.receptive_field_mapping.surface.forearm_slim_uv import (
    load_slim_uv_cache,
    uv_points_to_xyz,
)
from analysis.receptive_field_mapping.data.touch_population_data import (
    load_population_data,
    load_population_rf_data,
)
from analysis.pipeline.shared_constants import GESTURE_TYPES
from analysis.receptive_field_mapping.data.rf_population_heatmap import (
    apply_vertex_threshold,
    build_gesture_touch_indices,
    compute_rf_heatmap,
    compute_threshold_from_ratio,
    compute_unique_touch_count,
)
from analysis.receptive_field_mapping.metrics.rf_inflection_boundary import (
    compute_inflection_boundary,
    inflection_boundary_to_dict,
)
from analysis.receptive_field_mapping.rendering.rf_population_map_renderer import (
    compute_interpolated_grid,
    render_population_rf_map,
    render_population_rf_composite,
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
    min_overlap_pct: float
    sentinel: Path
    produced: List[Path] = field(default_factory=list)
    gesture_boundaries: dict = field(default_factory=dict)
    vertex_data_npz: Path | None = None


def run_population_response_field_extraction(
    session_configs: list,
    neuron_mode: str,
    min_overlap_pct: float = 25.0,
    force_processing: bool = False,
    median_filter_size: int | None = None,
    inflection_sigma: float | None = None,
) -> None:
    """Render per-session 2D population RF heatmap PNGs projected via SLIM UV.

    For each session config, produces one PNG per gesture subset (all, tap,
    stroke_proximal, stroke_distal) plus two composite PNGs (scatter and
    interpolated) under ``4_analysed/population_response_fields/{session_id}/``.

    Composites use a global colour scale and UV axis range across all sessions
    so they are directly comparable.

    Parameters
    ----------
    session_configs:
        List of ``(aggregated_csv_path, database_path)`` tuples.
    neuron_mode:
        ``"iff"`` or ``"spike"`` — must match the mode used by
        ``run_single_touch_rf_mapping``.
    min_overlap_pct:
        Minimum percentage of touches that must contact a vertex for it to be
        included in the heatmap (default 25 %).
    force_processing:
        If True, reprocess sessions even when the sentinel file exists.
    inflection_sigma:
        Gaussian smoothing sigma for Laplacian inflection boundary detection.
        Pass ``None`` to disable boundary computation entirely.
    """
    # ---- Pass 1: compute heatmaps + render per-gesture PNGs ----
    composite_queue: List[_SessionCompositeData] = []

    for csv_path, database_path in session_configs:
        csv_path = Path(csv_path)
        database_path = Path(database_path)

        session_id = session_id_from_path(csv_path)
        output_dir = database_path / '4_analysed' / 'population_response_fields' / session_id
        sentinel = output_dir / f'{session_id}_population_response_fields_done.json'

        if sentinel.exists() and not force_processing:
            print(f"[Population Response Fields] {session_id}: up-to-date, skipping.")
            continue

        print(f"[Population Response Fields] {session_id}: processing...")

        # --- Resolve paths ---
        series_csv_path = (
            database_path / '4_analysed' / 'series_transforms'
            / f'{session_id}_series_augmented.csv'
        )
        if not series_csv_path.exists():
            raise FileNotFoundError(
                f"[Population Response Fields] {session_id}: series-augmented CSV not found — "
                f"run 'touch_series_transforms' first: {series_csv_path}"
            )

        forearm_ply_path = resolve_forearm_ply(csv_path.parent, session_id)
        if forearm_ply_path is None:
            raise FileNotFoundError(
                f"[Population Response Fields] {session_id}: forearm PLY not found in "
                f"{csv_path.parent} — RF-centred PLY must exist."
            )

        npz_path = (
            database_path / '4_analysed' / 'single_touch_rf_maps'
            / session_id / 'single_touch_rf_maps.npz'
        )
        if not npz_path.exists():
            raise FileNotFoundError(
                f"[Population Response Fields] {session_id}: single_touch_rf_maps.npz not found: "
                f"{npz_path}. Enable 'map_single_touch_rf' in the DAG config and re-run."
            )

        slim_cache_path = (
            database_path / '4_analysed' / 'forearm_slim_uv'
            / session_id / f'{session_id}_slim_uv.npz'
        )
        if not slim_cache_path.exists():
            raise FileNotFoundError(
                f"[Population Response Fields] {session_id}: SLIM UV cache not found: "
                f"{slim_cache_path}. Enable 'precompute_forearm_slim_uv' in the DAG "
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
                "misaligned — re-run 'precompute_forearm_slim_uv' after "
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

        if not results:
            logger.warning(
                "[Population Response Fields] %s: no gesture subsets had touches — no PNGs produced.",
                session_id,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_sentinel(sentinel, session_id, produced=[])
            continue

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

        output_dir.mkdir(parents=True, exist_ok=True)
        produced: List[Path] = []
        gesture_boundaries: dict = {}
        per_gesture_grids: dict = {}

        for gtype, (slim_heatmap, n_touches, threshold) in results.items():
            title = (
                f"{session_id} | {gtype} | {n_touches} touches | "
                f"threshold={threshold} ({min_overlap_pct:.0f}%)"
            )
            png_path = output_dir / f'{session_id}_rf_population_{gtype}.png'

            print(f"[Population Response Fields] {session_id}: rendering '{gtype}'...")
            grid_u, grid_v, grid_z = compute_interpolated_grid(
                forearm_uv, slim_faces, slim_V, slim_heatmap,
                median_filter_size=median_filter_size,
            )
            per_gesture_grids[gtype] = (grid_u, grid_v, grid_z)
            boundary = (
                compute_inflection_boundary(
                    grid_u, grid_v, grid_z, inflection_sigma,
                    snapshot_dir=output_dir, snapshot_label=gtype,
                )
                if inflection_sigma is not None
                else None
            )
            gesture_boundaries[gtype] = boundary
            render_population_rf_map(
                forearm_uv=forearm_uv,
                heatmap_val=slim_heatmap,
                vmax=session_vmax,
                title=title,
                output_path=png_path,
                forearm_faces=slim_faces,
                forearm_V=slim_V,
                median_filter_size=median_filter_size,
                precomputed_grid=(grid_u, grid_v, grid_z),
                inflection_boundary=boundary,
            )
            produced.append(png_path)
            print(f"[Population Response Fields] {session_id}: saved {png_path.name}")

        vertex_data_npz = _save_response_fields_npz(
            output_dir=output_dir,
            session_id=session_id,
            forearm_uv=forearm_uv,
            forearm_faces=slim_faces,
            forearm_V=slim_V,
            results=results,
            per_gesture_grids=per_gesture_grids,
            neuron_mode=neuron_mode,
            min_overlap_pct=min_overlap_pct,
            gesture_boundaries=gesture_boundaries,
            inflection_sigma=inflection_sigma,
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
            output_dir=output_dir,
            forearm_uv=forearm_uv,
            forearm_faces=slim_faces,
            forearm_V=slim_V,
            results=results,
            session_vmax=session_vmax,
            min_overlap_pct=min_overlap_pct,
            sentinel=sentinel,
            produced=produced,
            gesture_boundaries=gesture_boundaries,
            vertex_data_npz=vertex_data_npz,
        ))

    # ---- Pass 2: render composite PNGs with global colour scale + UV limits ----
    if not composite_queue:
        return

    global_vmax = max(sd.session_vmax for sd in composite_queue)
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
        for panel_type in ('scatter', 'interpolated'):
            composite_path = (
                sd.output_dir / f'{sd.session_id}_rf_population_{panel_type}_composite.png'
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
                    inflection_boundaries[gtype] = (
                        compute_inflection_boundary(
                            grid_u_g, grid_v_g, grid_z_g, inflection_sigma,
                            snapshot_dir=sd.output_dir, snapshot_label=f"{sd.session_id}_{gtype}_composite",
                        )
                        if inflection_sigma is not None
                        else None
                    )
            else:
                precomputed_grids = None
                inflection_boundaries = None
            render_population_rf_composite(
                forearm_uv=sd.forearm_uv,
                results=sd.results,
                vmax=global_vmax,
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
                inflection_boundaries=inflection_boundaries,
            )
            sd.produced.append(composite_path)
            print(
                f"[Population Response Fields] {sd.session_id}: saved {composite_path.name}"
            )

        _write_sentinel(sd.sentinel, sd.session_id, produced=sd.produced,
                        inflection_boundaries=sd.gesture_boundaries,
                        vertex_data_npz=sd.vertex_data_npz)


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
    inflection_sigma: float | None,
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

        boundary = gesture_boundaries.get(gtype)
        if boundary is not None:
            data_dict[f'boundary_contour_uv_{gtype}'] = boundary.contour_uv.astype(np.float64)
            data_dict[f'boundary_centroid_uv_{gtype}'] = np.array(boundary.centroid_uv, dtype=np.float64)
            data_dict[f'boundary_perimeter_uv_{gtype}'] = np.float64(boundary.perimeter_uv)
            data_dict[f'boundary_area_uv_{gtype}'] = np.float64(boundary.area_uv)
            data_dict[f'boundary_circularity_{gtype}'] = np.float64(boundary.circularity)
            data_dict[f'boundary_pca_major_uv_{gtype}'] = np.float64(boundary.pca_major_uv)
            data_dict[f'boundary_pca_minor_uv_{gtype}'] = np.float64(boundary.pca_minor_uv)
            data_dict[f'boundary_pca_orientation_deg_{gtype}'] = np.float64(boundary.pca_orientation_deg)
            data_dict[f'boundary_mean_iff_on_contour_{gtype}'] = np.float64(boundary.mean_iff_on_contour)

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

            data_dict[f'boundary_contour_xyz_{gtype}'] = contour_xyz.astype(np.float64)
            data_dict[f'boundary_centroid_xyz_{gtype}'] = centroid_xyz.astype(np.float64)
            data_dict[f'boundary_perimeter_xyz_mm_{gtype}'] = np.float64(perimeter_xyz_mm)
            data_dict[f'boundary_area_xyz_mm2_{gtype}'] = np.float64(area_xyz_mm2)

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
