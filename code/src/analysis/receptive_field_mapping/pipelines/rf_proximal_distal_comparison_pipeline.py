"""Per-session + cross-session pipeline for proximal-distal RF comparison."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pipeline.output_dirs import SPATIAL_EXTRACT_BOUNDARIES
from analysis.pipeline.shared_constants import IFF_METRICS, session_id_from_path
from analysis.receptive_field_mapping.rendering.neuron_type_colors import (
    SessionColorScheme,
    build_session_color_scheme,
)
from analysis.receptive_field_mapping.rendering.rf_proximal_distal_comparison_renderer import (
    render_center_marked_heatmap,
    render_proximal_distal_aggregate,
    render_proximal_distal_contour_center_aggregate,
    render_proximal_distal_contour_overlay,
    render_proximal_distal_heatmap_triptych,
    render_proximal_distal_hotspot_aggregate,
    render_proximal_distal_metric_deltas,
    render_proximal_distal_population_strips,
    render_shift_decomposition,
)
from analysis.receptive_field_mapping.rendering.rf_population_map_renderer import (
    compute_highest_contour_peak,
    compute_standalone_figwidth,
    compute_uv_to_mm_scale,
    render_population_rf_circular_crop,
)

logger = logging.getLogger(__name__)

_REQUIRED_GTYPES = ('all', 'stroke_proximal', 'stroke_distal')
_HOTSPOT_GTYPES = ('stroke', 'stroke_proximal', 'stroke_distal')

_BOUNDARY_SCALAR_KEYS = (
    'boundary_area_xyz_mm2',
    'boundary_perimeter_xyz_mm',
    'boundary_circularity',
    'boundary_pca_major_uv',
    'boundary_pca_minor_uv',
    'boundary_pca_orientation_deg',
    'boundary_mean_iff_on_contour',
)


def run_proximal_distal_comparison(
    session_configs: list[tuple[Path, Path]],
    output_dir: Path,
    force_processing: bool = False,
    heatmap_space: str = "linear",
    cmap: str = "inferno",
    iff_metric: str = "mean",
    contour_color: str = "red",
    neuron_summary_xlsx: Path | None = None,
) -> None:
    if iff_metric not in IFF_METRICS:
        raise ValueError(
            f"[RF Proximal-Distal Comparison] Invalid iff_metric {iff_metric!r}. "
            f"Expected one of {IFF_METRICS}."
        )
    if not session_configs:
        raise ValueError("[RF Proximal-Distal Comparison] session_configs is empty.")
    sentinel_path = output_dir / 'rf_proximal_distal_comparison_done.json'
    # Backward-compat: also recognise the old sentinel name
    _old_sentinel_path = output_dir / 'rf_center_proximal_distal_done.json'

    if (sentinel_path.exists() or _old_sentinel_path.exists()) and not force_processing:
        logger.info("[RF Proximal-Distal Comparison] up-to-date, skipping.")
        return

    logger.info("[RF Proximal-Distal Comparison] building comparison from %d sessions...", len(session_configs))

    # Pass 1 — load all session data, compute global limits
    valid_data: list[dict] = []
    all_grid_z_arrays: list[np.ndarray] = []
    all_forearm_uv: list[np.ndarray] = []
    all_titles: list[str] = []
    summary_rows: list[dict] = []

    for csv_path, db_path_item in session_configs:
        csv_path = Path(csv_path)
        db_path_item = Path(db_path_item)
        session_id = session_id_from_path(csv_path)

        npz_path = (
            db_path_item / '4_analysed' / SPATIAL_EXTRACT_BOUNDARIES
            / f"iff_{iff_metric}" / session_id / f'{session_id}_population_response_fields.npz'
        )

        if not npz_path.exists():
            raise FileNotFoundError(
                f"[RF Proximal-Distal Comparison] {session_id}: NPZ not found at "
                f"{npz_path} — run spatial_extract_boundaries first."
            )

        npz = np.load(npz_path, allow_pickle=True)

        forearm_uv = npz['forearm_uv'].astype(np.float64)
        forearm_V = npz['forearm_V'].astype(np.float64)
        forearm_faces = npz['forearm_faces'].astype(np.int32)
        slim_vertex_colors = (
            npz['slim_vertex_colors'].astype(np.float64)
            if 'slim_vertex_colors' in npz
            else None
        )
        uv_to_mm = compute_uv_to_mm_scale(forearm_uv, forearm_V, forearm_faces)

        missing = [g for g in _REQUIRED_GTYPES if f'boundary_centroid_uv_{g}' not in npz]
        if missing:
            for mg in missing:
                logger.warning(
                    "[RF Proximal-Distal Comparison] %s: missing centroid for '%s' — skipping session.",
                    session_id, mg,
                )
            continue

        centroid_all = npz['boundary_centroid_uv_all'].astype(np.float64)
        centroid_proximal = npz['boundary_centroid_uv_stroke_proximal'].astype(np.float64)
        centroid_distal = npz['boundary_centroid_uv_stroke_distal'].astype(np.float64)

        offset_proximal = (centroid_proximal - centroid_all) * uv_to_mm
        offset_distal = (centroid_distal - centroid_all) * uv_to_mm
        dist_mm = float(np.linalg.norm(centroid_proximal - centroid_distal)) * uv_to_mm

        # Task 4.1 — attempt hotspot loading; degrade gracefully if keys missing
        hotspot_available = True
        missing_hotspot = [g for g in _HOTSPOT_GTYPES if f'boundary_peak_uv_{g}' not in npz]
        if missing_hotspot:
            for mg in missing_hotspot:
                logger.warning(
                    "[RF Proximal-Distal Comparison] %s: missing hotspot key 'boundary_peak_uv_%s' — "
                    "hotspot data unavailable for this session.",
                    session_id, mg,
                )
            hotspot_available = False

        if hotspot_available:
            hotspot_stroke = npz['boundary_peak_uv_stroke'].astype(np.float64)
            hotspot_proximal = npz['boundary_peak_uv_stroke_proximal'].astype(np.float64)
            hotspot_distal = npz['boundary_peak_uv_stroke_distal'].astype(np.float64)

            hotspot_offset_proximal = (hotspot_proximal - hotspot_stroke) * uv_to_mm
            hotspot_offset_distal = (hotspot_distal - hotspot_stroke) * uv_to_mm
            hotspot_dist_mm = float(np.linalg.norm(hotspot_proximal - hotspot_distal)) * uv_to_mm
            centroid_hotspot_distance_mm_stroke = float(np.linalg.norm(centroid_all - hotspot_stroke)) * uv_to_mm
        else:
            hotspot_stroke = None
            hotspot_proximal = None
            hotspot_distal = None
            hotspot_offset_proximal = None
            hotspot_offset_distal = None
            hotspot_dist_mm = None
            centroid_hotspot_distance_mm_stroke = None

        # Task 1.2–1.6 — extract boundary scalar metrics for both directions
        nan = float('nan')
        _dir_metrics: dict[str, dict] = {}
        for _direction in ('stroke_proximal', 'stroke_distal'):
            _contour_key = f'boundary_contour_uv_{_direction}'
            if _contour_key in npz:
                _area_mm2 = float(npz[f'boundary_area_xyz_mm2_{_direction}'])
                _perimeter_mm = float(npz[f'boundary_perimeter_xyz_mm_{_direction}'])
                _circularity = float(npz[f'boundary_circularity_{_direction}'])
                _pca_major_mm = float(npz[f'boundary_pca_major_uv_{_direction}']) * uv_to_mm
                _pca_minor_mm = float(npz[f'boundary_pca_minor_uv_{_direction}']) * uv_to_mm
                _pca_orientation_deg = float(npz[f'boundary_pca_orientation_deg_{_direction}'])
                _mean_iff_on_contour = float(npz[f'boundary_mean_iff_on_contour_{_direction}'])
                _n_touches = int(npz[f'n_touches_{_direction}'])
                # Task 1.3 — pca_aspect_ratio
                _pca_aspect_ratio = _pca_major_mm / _pca_minor_mm if _pca_minor_mm != 0.0 else nan
                # Task 1.6 — peak_to_centroid_mm
                _peak_key = f'boundary_peak_uv_{_direction}'
                _centroid_key = f'boundary_centroid_uv_{_direction}'
                if _peak_key in npz and _centroid_key in npz:
                    _peak_uv_dir = npz[_peak_key].astype(np.float64)
                    _centroid_uv_dir = npz[_centroid_key].astype(np.float64)
                    _peak_to_centroid_mm = float(np.linalg.norm(_peak_uv_dir - _centroid_uv_dir)) * uv_to_mm
                else:
                    _peak_to_centroid_mm = nan
                # peak_iff — maximum IFF intensity on the heatmap grid
                _grid_z_key = f'grid_z_{_direction}'
                if _grid_z_key in npz:
                    _gz = npz[_grid_z_key]
                    _finite_positive = _gz[np.isfinite(_gz) & (_gz > 0)]
                    _peak_iff = float(np.nanmax(_finite_positive)) if _finite_positive.size > 0 else nan
                else:
                    _peak_iff = nan
                # equivalent_diameter_mm — diameter of a circle with the same area
                _equivalent_diameter_mm = float(np.sqrt(4 * _area_mm2 / np.pi)) if not np.isnan(_area_mm2) and _area_mm2 > 0 else nan
                # rf_sharpness — peak IFF / mean IFF on contour
                _rf_sharpness = _peak_iff / _mean_iff_on_contour if (_mean_iff_on_contour > 0 and not np.isnan(_mean_iff_on_contour) and not np.isnan(_peak_iff)) else nan
                # iff_at_centroid — IFF value at the boundary centroid
                _iff_at_centroid_key = f'boundary_iff_at_centroid_{_direction}'
                _iff_at_centroid = float(npz[_iff_at_centroid_key]) if _iff_at_centroid_key in npz else nan
                _dir_metrics[_direction] = {
                    'area_mm2': _area_mm2,
                    'perimeter_mm': _perimeter_mm,
                    'circularity': _circularity,
                    'pca_major_mm': _pca_major_mm,
                    'pca_minor_mm': _pca_minor_mm,
                    'pca_aspect_ratio': _pca_aspect_ratio,
                    'pca_orientation_deg': _pca_orientation_deg,
                    'mean_iff_on_contour': _mean_iff_on_contour,
                    'n_touches': _n_touches,
                    'peak_to_centroid_mm': _peak_to_centroid_mm,
                    'peak_iff': _peak_iff,
                    'equivalent_diameter_mm': _equivalent_diameter_mm,
                    'rf_sharpness': _rf_sharpness,
                    'iff_at_centroid': _iff_at_centroid,
                }
            else:
                logger.warning(
                    "[RF Proximal-Distal Comparison] %s: no boundary for '%s' — metrics will be NaN.",
                    session_id, _direction,
                )
                _dir_metrics[_direction] = {
                    'area_mm2': nan,
                    'perimeter_mm': nan,
                    'circularity': nan,
                    'pca_major_mm': nan,
                    'pca_minor_mm': nan,
                    'pca_aspect_ratio': nan,
                    'pca_orientation_deg': nan,
                    'mean_iff_on_contour': nan,
                    'n_touches': None,
                    'peak_to_centroid_mm': nan,
                    'peak_iff': nan,
                    'equivalent_diameter_mm': nan,
                    'rf_sharpness': nan,
                    'iff_at_centroid': nan,
                }

        _pm = _dir_metrics['stroke_proximal']
        _dm = _dir_metrics['stroke_distal']

        # Task 1.4 — compute delta metrics (proximal − distal)
        _delta_area_mm2 = _pm['area_mm2'] - _dm['area_mm2']
        _area_ratio = (
            _pm['area_mm2'] / _dm['area_mm2']
            if not (np.isnan(_dm['area_mm2']) or _dm['area_mm2'] == 0.0)
            else nan
        )
        _delta_perimeter_mm = _pm['perimeter_mm'] - _dm['perimeter_mm']
        _delta_circularity = _pm['circularity'] - _dm['circularity']
        _delta_pca_aspect_ratio = _pm['pca_aspect_ratio'] - _dm['pca_aspect_ratio']
        _orient_p = _pm['pca_orientation_deg']
        _orient_d = _dm['pca_orientation_deg']
        _delta_pca_orientation_deg = (
            ((_orient_p - _orient_d + 90) % 180) - 90
            if not (np.isnan(_orient_p) or np.isnan(_orient_d))
            else nan
        )
        _delta_mean_iff_on_contour = _pm['mean_iff_on_contour'] - _dm['mean_iff_on_contour']

        # Task 1.5 — delta_area_pct normalized by combined stroke boundary area
        _stroke_area_key = 'boundary_area_xyz_mm2_stroke'
        if _stroke_area_key in npz:
            _stroke_area = float(npz[_stroke_area_key])
            _delta_area_pct = (
                _delta_area_mm2 / _stroke_area * 100.0
                if _stroke_area > 0 and not np.isnan(_delta_area_mm2)
                else nan
            )
        else:
            _delta_area_pct = nan

        # Task 1.6 — delta_peak_to_centroid_mm
        _delta_peak_to_centroid_mm = _pm['peak_to_centroid_mm'] - _dm['peak_to_centroid_mm']

        # New delta metrics (proximal − distal)
        _delta_peak_iff = _pm['peak_iff'] - _dm['peak_iff']
        _delta_equivalent_diameter_mm = _pm['equivalent_diameter_mm'] - _dm['equivalent_diameter_mm']
        _delta_rf_sharpness = _pm['rf_sharpness'] - _dm['rf_sharpness']
        _delta_iff_at_centroid = _pm['iff_at_centroid'] - _dm['iff_at_centroid']

        # Task 2.3 — centroid shift decomposition (proximal − distal in UV → mm)
        _centroid_shift_uv = centroid_proximal - centroid_distal
        _centroid_shift_along_arm_mm = float(_centroid_shift_uv[0]) * uv_to_mm  # U component
        _centroid_shift_across_arm_mm = float(_centroid_shift_uv[1]) * uv_to_mm  # V component

        # Task 2.4 — contour overlap (IoU, Dice)
        _prox_contour_key = 'boundary_contour_uv_stroke_proximal'
        _dist_contour_key = 'boundary_contour_uv_stroke_distal'
        if _prox_contour_key in npz and _dist_contour_key in npz:
            _contour_prox = npz[_prox_contour_key].astype(np.float64)
            _contour_dist = npz[_dist_contour_key].astype(np.float64)
            # Use grid from stroke_proximal (same meshgrid dimensions for both)
            _grid_u_prox = npz['grid_u_stroke_proximal']
            _grid_v_prox = npz['grid_v_stroke_proximal']
            _contour_overlap_iou, _contour_overlap_dice = _compute_contour_overlap(
                _contour_prox, _contour_dist, _grid_u_prox, _grid_v_prox,
            )
        else:
            _contour_overlap_iou = nan
            _contour_overlap_dice = nan

        # Task 2.4 — heatmap correlation
        if 'grid_z_stroke_proximal' in npz and 'grid_z_stroke_distal' in npz:
            _heatmap_pearson_r = _compute_heatmap_similarity(
                npz['grid_z_stroke_proximal'],
                npz['grid_z_stroke_distal'],
            )
        else:
            _heatmap_pearson_r = nan

        # Asymmetric containment — fraction of one RF contained within the other
        _prox_contour_key_c = 'boundary_contour_uv_stroke_proximal'
        _dist_contour_key_c = 'boundary_contour_uv_stroke_distal'
        if _prox_contour_key_c in npz and _dist_contour_key_c in npz and 'grid_u_stroke_proximal' in npz:
            from matplotlib.path import Path as MplPath
            _contour_prox_c = npz[_prox_contour_key_c].astype(np.float64)
            _contour_dist_c = npz[_dist_contour_key_c].astype(np.float64)
            _grid_u_c = npz['grid_u_stroke_proximal']
            _grid_v_c = npz['grid_v_stroke_proximal']
            _points_c = np.column_stack([_grid_u_c.ravel(), _grid_v_c.ravel()])
            _mask_prox = MplPath(_contour_prox_c).contains_points(_points_c)
            _mask_dist = MplPath(_contour_dist_c).contains_points(_points_c)
            _n_prox = np.sum(_mask_prox)
            _n_dist = np.sum(_mask_dist)
            _containment_proximal_in_distal = float(np.sum(_mask_prox & _mask_dist) / _n_prox) if _n_prox > 0 else nan
            _containment_distal_in_proximal = float(np.sum(_mask_prox & _mask_dist) / _n_dist) if _n_dist > 0 else nan
        else:
            _containment_proximal_in_distal = nan
            _containment_distal_in_proximal = nan

        gesture_types = list(npz['gesture_types'])
        gestures_with_centroid: dict[str, tuple] = {}
        for gtype in gesture_types:
            if f'boundary_centroid_uv_{gtype}' not in npz:
                continue
            grid_u = npz[f'grid_u_{gtype}']
            grid_v = npz[f'grid_v_{gtype}']
            grid_z = npz[f'grid_z_{gtype}']
            centroid_uv_gtype = npz[f'boundary_centroid_uv_{gtype}'].astype(np.float64)
            n_touches = int(npz[f'n_touches_{gtype}'])
            threshold = float(npz[f'threshold_{gtype}'])
            peak_uv_gtype = (
                npz[f'boundary_peak_uv_{gtype}'].astype(np.float64)
                if f'boundary_peak_uv_{gtype}' in npz
                else None
            )
            gestures_with_centroid[gtype] = (grid_u, grid_v, grid_z, centroid_uv_gtype, n_touches, threshold, peak_uv_gtype)

            finite_z = grid_z[np.isfinite(grid_z) & (grid_z > 0)]
            if finite_z.size > 0:
                all_grid_z_arrays.append(finite_z)

            title = (
                f"{session_id} | {gtype} | {n_touches} touches | "
                f"{centroid_uv_gtype[0]:.3f}, {centroid_uv_gtype[1]:.3f}"
            )
            all_titles.append(title)

        all_forearm_uv.append(forearm_uv)

        # Contour-center shift decomposition (proximal − distal)
        _cc_proximal = None
        _cc_distal = None
        if 'stroke_proximal' in gestures_with_centroid:
            _gu, _gv, _gz = gestures_with_centroid['stroke_proximal'][:3]
            _cc_proximal = compute_highest_contour_peak(_gu, _gv, _gz, n_levels=6)
        if 'stroke_distal' in gestures_with_centroid:
            _gu, _gv, _gz = gestures_with_centroid['stroke_distal'][:3]
            _cc_distal = compute_highest_contour_peak(_gu, _gv, _gz, n_levels=6)

        if _cc_proximal is not None and _cc_distal is not None:
            _cc_shift_uv = _cc_proximal - _cc_distal
            _cc_shift_along_arm_mm = float(_cc_shift_uv[0]) * uv_to_mm
            _cc_shift_across_arm_mm = float(_cc_shift_uv[1]) * uv_to_mm
        else:
            _cc_shift_along_arm_mm = nan
            _cc_shift_across_arm_mm = nan

        # Peak (hotspot) shift decomposition (proximal − distal in UV → mm)
        if hotspot_available:
            _peak_shift_uv = hotspot_proximal - hotspot_distal
            _peak_shift_along_arm_mm = float(_peak_shift_uv[0]) * uv_to_mm
            _peak_shift_across_arm_mm = float(_peak_shift_uv[1]) * uv_to_mm
        else:
            _peak_shift_along_arm_mm = nan
            _peak_shift_across_arm_mm = nan

        # Contour-center for 'all' gesture type (reference for aggregate scatter)
        _cc_all = None
        if 'all' in gestures_with_centroid:
            _gu_all, _gv_all, _gz_all = gestures_with_centroid['all'][:3]
            _cc_all = compute_highest_contour_peak(_gu_all, _gv_all, _gz_all, n_levels=6)

        if _cc_all is not None and _cc_proximal is not None and _cc_distal is not None:
            _cc_offset_proximal = (_cc_proximal - _cc_all) * uv_to_mm
            _cc_offset_distal = (_cc_distal - _cc_all) * uv_to_mm
        else:
            _cc_offset_proximal = None
            _cc_offset_distal = None

        valid_data.append({
            'session_id': session_id,
            'db_path': db_path_item,
            'gestures_with_centroid': gestures_with_centroid,
            'forearm_uv': forearm_uv,
            'forearm_V': forearm_V,
            'forearm_faces': forearm_faces,
            'slim_vertex_colors': slim_vertex_colors,
            'centroid_all': centroid_all,
            'centroid_proximal': centroid_proximal,
            'centroid_distal': centroid_distal,
            'offset_proximal': offset_proximal,
            'offset_distal': offset_distal,
            'dist_mm': dist_mm,
            'hotspot_available': hotspot_available,
            'hotspot_stroke': hotspot_stroke,
            'hotspot_proximal': hotspot_proximal,
            'hotspot_distal': hotspot_distal,
            'hotspot_offset_proximal': hotspot_offset_proximal,
            'hotspot_offset_distal': hotspot_offset_distal,
            'hotspot_dist_mm': hotspot_dist_mm,
            'centroid_hotspot_distance_mm_stroke': centroid_hotspot_distance_mm_stroke,
            # Phase 1 — per-direction boundary scalar metrics
            'boundary_metrics_proximal': _pm,
            'boundary_metrics_distal': _dm,
            # Phase 1 — delta metrics
            'delta_area_mm2': _delta_area_mm2,
            'area_ratio': _area_ratio,
            'delta_area_pct': _delta_area_pct,
            'delta_perimeter_mm': _delta_perimeter_mm,
            'delta_circularity': _delta_circularity,
            'delta_pca_aspect_ratio': _delta_pca_aspect_ratio,
            'delta_pca_orientation_deg': _delta_pca_orientation_deg,
            'delta_mean_iff_on_contour': _delta_mean_iff_on_contour,
            'delta_peak_to_centroid_mm': _delta_peak_to_centroid_mm,
            'delta_peak_iff': _delta_peak_iff,
            'delta_equivalent_diameter_mm': _delta_equivalent_diameter_mm,
            'delta_rf_sharpness': _delta_rf_sharpness,
            'delta_iff_at_centroid': _delta_iff_at_centroid,
            'containment_proximal_in_distal': _containment_proximal_in_distal,
            'containment_distal_in_proximal': _containment_distal_in_proximal,
            # Phase 2 — overlap, correlation, centroid shift decomposition
            'contour_overlap_iou': _contour_overlap_iou,
            'contour_overlap_dice': _contour_overlap_dice,
            'heatmap_pearson_r': _heatmap_pearson_r,
            'centroid_shift_along_arm_mm': _centroid_shift_along_arm_mm,
            'centroid_shift_across_arm_mm': _centroid_shift_across_arm_mm,
            'contour_center_shift_along_arm_mm': _cc_shift_along_arm_mm,
            'contour_center_shift_across_arm_mm': _cc_shift_across_arm_mm,
            'peak_shift_along_arm_mm': _peak_shift_along_arm_mm,
            'peak_shift_across_arm_mm': _peak_shift_across_arm_mm,
            'cc_all': _cc_all,
            'cc_proximal': _cc_proximal,
            'cc_distal': _cc_distal,
            'cc_offset_proximal': _cc_offset_proximal,
            'cc_offset_distal': _cc_offset_distal,
            # Phase 3 — contour and grid data for per-session overlay/triptych figures
            'contour_proximal_uv': (
                npz['boundary_contour_uv_stroke_proximal'].astype(np.float64)
                if 'boundary_contour_uv_stroke_proximal' in npz else None
            ),
            'contour_distal_uv': (
                npz['boundary_contour_uv_stroke_distal'].astype(np.float64)
                if 'boundary_contour_uv_stroke_distal' in npz else None
            ),
            'contour_stroke_uv': (
                npz['boundary_contour_uv_stroke'].astype(np.float64)
                if 'boundary_contour_uv_stroke' in npz else None
            ),
            'grid_u_proximal': (
                npz['grid_u_stroke_proximal']
                if 'grid_u_stroke_proximal' in npz else None
            ),
            'grid_v_proximal': (
                npz['grid_v_stroke_proximal']
                if 'grid_v_stroke_proximal' in npz else None
            ),
            'grid_z_proximal': (
                npz['grid_z_stroke_proximal']
                if 'grid_z_stroke_proximal' in npz else None
            ),
            'grid_z_distal': (
                npz['grid_z_stroke_distal']
                if 'grid_z_stroke_distal' in npz else None
            ),
        })

        summary_rows.append({
            'session_id': session_id,
            'uv_to_mm_scale': uv_to_mm,
            'centroid_u_all': float(centroid_all[0]),
            'centroid_v_all': float(centroid_all[1]),
            'centroid_u_proximal': float(centroid_proximal[0]),
            'centroid_v_proximal': float(centroid_proximal[1]),
            'centroid_u_distal': float(centroid_distal[0]),
            'centroid_v_distal': float(centroid_distal[1]),
            'offset_u_proximal_mm': float(offset_proximal[0]),
            'offset_v_proximal_mm': float(offset_proximal[1]),
            'offset_u_distal_mm': float(offset_distal[0]),
            'offset_v_distal_mm': float(offset_distal[1]),
            'proximal_distal_distance_mm': dist_mm,
            'hotspot_u_stroke': float(hotspot_stroke[0]) if hotspot_available else nan,
            'hotspot_v_stroke': float(hotspot_stroke[1]) if hotspot_available else nan,
            'hotspot_u_proximal': float(hotspot_proximal[0]) if hotspot_available else nan,
            'hotspot_v_proximal': float(hotspot_proximal[1]) if hotspot_available else nan,
            'hotspot_u_distal': float(hotspot_distal[0]) if hotspot_available else nan,
            'hotspot_v_distal': float(hotspot_distal[1]) if hotspot_available else nan,
            'hotspot_offset_u_proximal_mm': float(hotspot_offset_proximal[0]) if hotspot_available else nan,
            'hotspot_offset_v_proximal_mm': float(hotspot_offset_proximal[1]) if hotspot_available else nan,
            'hotspot_offset_u_distal_mm': float(hotspot_offset_distal[0]) if hotspot_available else nan,
            'hotspot_offset_v_distal_mm': float(hotspot_offset_distal[1]) if hotspot_available else nan,
            'hotspot_proximal_distal_distance_mm': hotspot_dist_mm if hotspot_available else nan,
            'centroid_hotspot_distance_mm_stroke': centroid_hotspot_distance_mm_stroke if hotspot_available else nan,
            # Phase 1 — per-direction boundary scalar metrics
            'area_mm2_proximal': _pm['area_mm2'],
            'area_mm2_distal': _dm['area_mm2'],
            'perimeter_mm_proximal': _pm['perimeter_mm'],
            'perimeter_mm_distal': _dm['perimeter_mm'],
            'circularity_proximal': _pm['circularity'],
            'circularity_distal': _dm['circularity'],
            'pca_major_mm_proximal': _pm['pca_major_mm'],
            'pca_major_mm_distal': _dm['pca_major_mm'],
            'pca_minor_mm_proximal': _pm['pca_minor_mm'],
            'pca_minor_mm_distal': _dm['pca_minor_mm'],
            'pca_aspect_ratio_proximal': _pm['pca_aspect_ratio'],
            'pca_aspect_ratio_distal': _dm['pca_aspect_ratio'],
            'pca_orientation_deg_proximal': _pm['pca_orientation_deg'],
            'pca_orientation_deg_distal': _dm['pca_orientation_deg'],
            'mean_iff_on_contour_proximal': _pm['mean_iff_on_contour'],
            'mean_iff_on_contour_distal': _dm['mean_iff_on_contour'],
            'n_touches_proximal': _pm['n_touches'],
            'n_touches_distal': _dm['n_touches'],
            'peak_to_centroid_mm_proximal': _pm['peak_to_centroid_mm'],
            'peak_to_centroid_mm_distal': _dm['peak_to_centroid_mm'],
            'peak_iff_proximal': _pm['peak_iff'],
            'peak_iff_distal': _dm['peak_iff'],
            'equivalent_diameter_mm_proximal': _pm['equivalent_diameter_mm'],
            'equivalent_diameter_mm_distal': _dm['equivalent_diameter_mm'],
            'rf_sharpness_proximal': _pm['rf_sharpness'],
            'rf_sharpness_distal': _dm['rf_sharpness'],
            'iff_at_centroid_proximal': _pm['iff_at_centroid'],
            'iff_at_centroid_distal': _dm['iff_at_centroid'],
            # Phase 1 — delta metrics
            'delta_area_mm2': _delta_area_mm2,
            'area_ratio': _area_ratio,
            'delta_area_pct': _delta_area_pct,
            'delta_perimeter_mm': _delta_perimeter_mm,
            'delta_circularity': _delta_circularity,
            'delta_pca_aspect_ratio': _delta_pca_aspect_ratio,
            'delta_pca_orientation_deg': _delta_pca_orientation_deg,
            'delta_mean_iff_on_contour': _delta_mean_iff_on_contour,
            'delta_peak_to_centroid_mm': _delta_peak_to_centroid_mm,
            'delta_peak_iff': _delta_peak_iff,
            'delta_equivalent_diameter_mm': _delta_equivalent_diameter_mm,
            'delta_rf_sharpness': _delta_rf_sharpness,
            'delta_iff_at_centroid': _delta_iff_at_centroid,
            'containment_proximal_in_distal': _containment_proximal_in_distal,
            'containment_distal_in_proximal': _containment_distal_in_proximal,
            # Phase 2 — overlap, correlation, centroid shift decomposition
            'contour_overlap_iou': _contour_overlap_iou,
            'contour_overlap_dice': _contour_overlap_dice,
            'heatmap_pearson_r': _heatmap_pearson_r,
            'centroid_shift_along_arm_mm': _centroid_shift_along_arm_mm,
            'centroid_shift_across_arm_mm': _centroid_shift_across_arm_mm,
            'contour_center_shift_along_arm_mm': _cc_shift_along_arm_mm,
            'contour_center_shift_across_arm_mm': _cc_shift_across_arm_mm,
            'peak_shift_along_arm_mm': _peak_shift_along_arm_mm,
            'peak_shift_across_arm_mm': _peak_shift_across_arm_mm,
            'cc_u_all': float(_cc_all[0]) if _cc_all is not None else nan,
            'cc_v_all': float(_cc_all[1]) if _cc_all is not None else nan,
            'cc_offset_u_proximal_mm': float(_cc_offset_proximal[0]) if _cc_offset_proximal is not None else nan,
            'cc_offset_v_proximal_mm': float(_cc_offset_proximal[1]) if _cc_offset_proximal is not None else nan,
            'cc_offset_u_distal_mm': float(_cc_offset_distal[0]) if _cc_offset_distal is not None else nan,
            'cc_offset_v_distal_mm': float(_cc_offset_distal[1]) if _cc_offset_distal is not None else nan,
        })

    if not valid_data:
        logger.warning("[RF Proximal-Distal Comparison] no sessions had valid centroid data — nothing to do.")
        return

    if not all_grid_z_arrays:
        raise ValueError(
            "rf_proximal_distal_comparison: no valid heatmap data found across all sessions."
        )

    all_z_concat = np.concatenate(all_grid_z_arrays)
    global_vmax = float(all_z_concat.max())
    global_vmin_candidates = all_z_concat[all_z_concat > 0]
    global_vmin = float(global_vmin_candidates.min()) if global_vmin_candidates.size > 0 else global_vmax * 1e-3

    all_uv = np.vstack(all_forearm_uv)
    u_min_uv, u_max_uv = float(all_uv[:, 0].min()), float(all_uv[:, 0].max())
    v_min_uv, v_max_uv = float(all_uv[:, 1].min()), float(all_uv[:, 1].max())
    u_margin = 0.02 * (u_max_uv - u_min_uv) if u_max_uv > u_min_uv else 1.0
    v_margin = 0.02 * (v_max_uv - v_min_uv) if v_max_uv > v_min_uv else 1.0
    global_uv_xlim = (u_min_uv - u_margin, u_max_uv + u_margin)
    global_uv_ylim = (v_min_uv - v_margin, v_max_uv + v_margin)

    longest_title = max(all_titles, key=len) if all_titles else ""
    standalone_figwidth = compute_standalone_figwidth(longest_title)

    aggregate_mm_limits = _compute_aggregate_mm_limits(valid_data)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pass 2 — render per-session PNGs
    for d in valid_data:
        session_id = d['session_id']
        session_output_dir = output_dir / session_id
        session_output_dir.mkdir(parents=True, exist_ok=True)

        # Pre-pass: collect all crop centers across gesture types for shared limits
        radius_mm = 50.0
        scale = compute_uv_to_mm_scale(
            d['forearm_uv'], d['forearm_V'], d['forearm_faces'],
        )
        radius_uv = radius_mm / scale
        margin = radius_uv * 0.05

        all_crop_centers: list[np.ndarray] = []
        contour_centers: dict[str, np.ndarray | None] = {}
        for _gt, (_gu, _gv, _gz, _cent, _nt, _thr, _pk) in d['gestures_with_centroid'].items():
            all_crop_centers.append(_cent)
            if _pk is not None:
                all_crop_centers.append(_pk)
            _cc = compute_highest_contour_peak(_gu, _gv, _gz, n_levels=6)
            contour_centers[_gt] = _cc
            if _cc is not None:
                all_crop_centers.append(_cc)

        all_centers_arr = np.array(all_crop_centers)
        shared_crop_xlim = (
            float(all_centers_arr[:, 0].min()) - radius_uv - margin,
            float(all_centers_arr[:, 0].max()) + radius_uv + margin,
        )
        shared_crop_ylim = (
            float(all_centers_arr[:, 1].min()) - radius_uv - margin,
            float(all_centers_arr[:, 1].max()) + radius_uv + margin,
        )

        for gtype, (grid_u, grid_v, grid_z, centroid_uv_gtype, n_touches, _threshold, peak_uv_gtype) in d['gestures_with_centroid'].items():
            render_center_marked_heatmap(
                u_grid=grid_u,
                v_grid=grid_v,
                interp_grid=grid_z,
                forearm_uv=d['forearm_uv'],
                centroid_uv=centroid_uv_gtype,
                output_path=session_output_dir / f'{session_id}_rf_center_{gtype}_{cmap}.png',
                vmax=global_vmax,
                vmin=global_vmin,
                title=f"{session_id} | {gtype} | centroid ({centroid_uv_gtype[0]:.3f}, {centroid_uv_gtype[1]:.3f})",
                figwidth=standalone_figwidth,
                xlim=global_uv_xlim,
                ylim=global_uv_ylim,
                heatmap_space=heatmap_space,
                cmap=cmap,
                vertex_colors=d['slim_vertex_colors'],
                forearm_faces=d['forearm_faces'],
                contour_color=contour_color,
            )

            # Task 4.3 — render hotspot PNG when peak_uv is available for this gtype
            if peak_uv_gtype is not None:
                render_center_marked_heatmap(
                    u_grid=grid_u,
                    v_grid=grid_v,
                    interp_grid=grid_z,
                    forearm_uv=d['forearm_uv'],
                    centroid_uv=centroid_uv_gtype,
                    output_path=session_output_dir / f'{session_id}_rf_hotspot_{gtype}_{cmap}.png',
                    vmax=global_vmax,
                    vmin=global_vmin,
                    title=f"{session_id} | {gtype} | hotspot ({peak_uv_gtype[0]:.3f}, {peak_uv_gtype[1]:.3f})",
                    figwidth=standalone_figwidth,
                    xlim=global_uv_xlim,
                    ylim=global_uv_ylim,
                    heatmap_space=heatmap_space,
                    cmap=cmap,
                    peak_uv=peak_uv_gtype,
                    vertex_colors=d['slim_vertex_colors'],
                    forearm_faces=d['forearm_faces'],
                    contour_color=contour_color,
                )

            # Circular crops per center type, each in its own subdirectory
            centroid_crop_dir = session_output_dir / 'centroid'
            peak_crop_dir = session_output_dir / 'peak'
            contour_center_crop_dir = session_output_dir / 'contour_center'
            for d_path in (centroid_crop_dir, peak_crop_dir, contour_center_crop_dir):
                d_path.mkdir(parents=True, exist_ok=True)

            contour_center = contour_centers[gtype]

            crop_jobs: list[tuple[np.ndarray, np.ndarray, Path]] = [
                (centroid_uv_gtype, centroid_uv_gtype,
                 centroid_crop_dir / f'{session_id}_rf_circular_centroid_{gtype}_{cmap}.png'),
            ]
            if peak_uv_gtype is not None:
                crop_jobs.append((
                    peak_uv_gtype, centroid_uv_gtype,
                    peak_crop_dir / f'{session_id}_rf_circular_peak_{gtype}_{cmap}.png',
                ))
            if contour_center is not None:
                crop_jobs.append((
                    contour_center, contour_center,
                    contour_center_crop_dir / f'{session_id}_rf_circular_contour_center_{gtype}_{cmap}.png',
                ))

            for center_uv, marker_uv, out_path in crop_jobs:
                render_population_rf_circular_crop(
                    u_grid=grid_u,
                    v_grid=grid_v,
                    interp_grid=grid_z,
                    forearm_uv=d['forearm_uv'],
                    forearm_V=d['forearm_V'],
                    forearm_faces=d['forearm_faces'],
                    center_uv=center_uv,
                    radius_mm=radius_mm,
                    vmax=global_vmax,
                    vmin=global_vmin,
                    output_path=out_path,
                    vertex_colors=d['slim_vertex_colors'],
                    heatmap_space=heatmap_space,
                    dpi=300,
                    cmap=cmap,
                    contour_levels=6,
                    centroid_uv=marker_uv,
                    xlim=shared_crop_xlim,
                    ylim=shared_crop_ylim,
                )

        # Phase 3 — contour overlay and heatmap triptych (per-session, outside gesture loop)
        _contour_prox = d.get('contour_proximal_uv')
        _contour_dist = d.get('contour_distal_uv')
        if _contour_prox is not None and _contour_dist is not None:
            render_proximal_distal_contour_overlay(
                forearm_uv=d['forearm_uv'],
                forearm_faces=d['forearm_faces'],
                contour_proximal_uv=_contour_prox,
                contour_distal_uv=_contour_dist,
                centroid_proximal_uv=d['centroid_proximal'],
                centroid_distal_uv=d['centroid_distal'],
                output_path=session_output_dir / f'{session_id}_contour_overlay_{cmap}.png',
                contour_stroke_uv=d.get('contour_stroke_uv'),
                hotspot_proximal_uv=d.get('hotspot_proximal'),
                hotspot_distal_uv=d.get('hotspot_distal'),
                vertex_colors=d['slim_vertex_colors'],
                iou=d.get('contour_overlap_iou'),
                area_ratio=d.get('area_ratio'),
                title=f"{session_id} | Contour Overlay",
                xlim=global_uv_xlim,
                ylim=global_uv_ylim,
            )
        else:
            logger.warning(
                "[RF Proximal-Distal Comparison] %s: missing contour — skipping overlay.",
                session_id,
            )

        _gz_prox = d.get('grid_z_proximal')
        _gz_dist = d.get('grid_z_distal')
        _gu_prox = d.get('grid_u_proximal')
        _gv_prox = d.get('grid_v_proximal')
        if _gz_prox is not None and _gz_dist is not None and _gu_prox is not None and _gv_prox is not None:
            render_proximal_distal_heatmap_triptych(
                grid_u=_gu_prox,
                grid_v=_gv_prox,
                grid_z_proximal=_gz_prox,
                grid_z_distal=_gz_dist,
                forearm_uv=d['forearm_uv'],
                forearm_faces=d['forearm_faces'],
                output_path=session_output_dir / f'{session_id}_heatmap_triptych_{cmap}.png',
                vmax=global_vmax,
                vmin=global_vmin,
                vertex_colors=d['slim_vertex_colors'],
                heatmap_space=heatmap_space,
                cmap=cmap,
                title=f"{session_id} | Heatmap Triptych",
                xlim=global_uv_xlim,
                ylim=global_uv_ylim,
            )
        else:
            logger.warning(
                "[RF Proximal-Distal Comparison] %s: missing heatmap grid — skipping triptych.",
                session_id,
            )

    # Pass 3 — cross-session outputs
    df = pd.DataFrame(summary_rows).astype({
        'session_id': str,
        'uv_to_mm_scale': np.float64,
        'centroid_u_all': np.float64,
        'centroid_v_all': np.float64,
        'centroid_u_proximal': np.float64,
        'centroid_v_proximal': np.float64,
        'centroid_u_distal': np.float64,
        'centroid_v_distal': np.float64,
        'offset_u_proximal_mm': np.float64,
        'offset_v_proximal_mm': np.float64,
        'offset_u_distal_mm': np.float64,
        'offset_v_distal_mm': np.float64,
        'proximal_distal_distance_mm': np.float64,
        'hotspot_u_stroke': np.float64,
        'hotspot_v_stroke': np.float64,
        'hotspot_u_proximal': np.float64,
        'hotspot_v_proximal': np.float64,
        'hotspot_u_distal': np.float64,
        'hotspot_v_distal': np.float64,
        'hotspot_offset_u_proximal_mm': np.float64,
        'hotspot_offset_v_proximal_mm': np.float64,
        'hotspot_offset_u_distal_mm': np.float64,
        'hotspot_offset_v_distal_mm': np.float64,
        'hotspot_proximal_distal_distance_mm': np.float64,
        'centroid_hotspot_distance_mm_stroke': np.float64,
        # Phase 1 — per-direction boundary scalar metrics
        'area_mm2_proximal': np.float64,
        'area_mm2_distal': np.float64,
        'perimeter_mm_proximal': np.float64,
        'perimeter_mm_distal': np.float64,
        'circularity_proximal': np.float64,
        'circularity_distal': np.float64,
        'pca_major_mm_proximal': np.float64,
        'pca_major_mm_distal': np.float64,
        'pca_minor_mm_proximal': np.float64,
        'pca_minor_mm_distal': np.float64,
        'pca_aspect_ratio_proximal': np.float64,
        'pca_aspect_ratio_distal': np.float64,
        'pca_orientation_deg_proximal': np.float64,
        'pca_orientation_deg_distal': np.float64,
        'mean_iff_on_contour_proximal': np.float64,
        'mean_iff_on_contour_distal': np.float64,
        'n_touches_proximal': 'Int64',
        'n_touches_distal': 'Int64',
        'peak_to_centroid_mm_proximal': np.float64,
        'peak_to_centroid_mm_distal': np.float64,
        'peak_iff_proximal': np.float64,
        'peak_iff_distal': np.float64,
        'equivalent_diameter_mm_proximal': np.float64,
        'equivalent_diameter_mm_distal': np.float64,
        'rf_sharpness_proximal': np.float64,
        'rf_sharpness_distal': np.float64,
        'iff_at_centroid_proximal': np.float64,
        'iff_at_centroid_distal': np.float64,
        # Phase 1 — delta metrics
        'delta_area_mm2': np.float64,
        'area_ratio': np.float64,
        'delta_area_pct': np.float64,
        'delta_perimeter_mm': np.float64,
        'delta_circularity': np.float64,
        'delta_pca_aspect_ratio': np.float64,
        'delta_pca_orientation_deg': np.float64,
        'delta_mean_iff_on_contour': np.float64,
        'delta_peak_to_centroid_mm': np.float64,
        'delta_peak_iff': np.float64,
        'delta_equivalent_diameter_mm': np.float64,
        'delta_rf_sharpness': np.float64,
        'delta_iff_at_centroid': np.float64,
        'containment_proximal_in_distal': np.float64,
        'containment_distal_in_proximal': np.float64,
        # Phase 2 — overlap, correlation, centroid shift decomposition
        'contour_overlap_iou': np.float64,
        'contour_overlap_dice': np.float64,
        'heatmap_pearson_r': np.float64,
        'centroid_shift_along_arm_mm': np.float64,
        'centroid_shift_across_arm_mm': np.float64,
        'contour_center_shift_along_arm_mm': np.float64,
        'contour_center_shift_across_arm_mm': np.float64,
        'peak_shift_along_arm_mm': np.float64,
        'peak_shift_across_arm_mm': np.float64,
        'cc_u_all': np.float64,
        'cc_v_all': np.float64,
        'cc_offset_u_proximal_mm': np.float64,
        'cc_offset_v_proximal_mm': np.float64,
        'cc_offset_u_distal_mm': np.float64,
        'cc_offset_v_distal_mm': np.float64,
    })
    csv_path = output_dir / 'rf_proximal_distal_comparison_summary.csv'
    df.to_csv(csv_path, index=False)
    logger.info("[RF Proximal-Distal Comparison] wrote %s", csv_path.name)

    # Task 4.1 — build per-session color scheme when neuron_summary_xlsx is provided
    session_colors: dict[str, str] | None = None
    neuron_type_legend: dict[str, str] | None = None
    if neuron_summary_xlsx is not None:
        session_ids = df['session_id'].unique().tolist()
        scheme: SessionColorScheme = build_session_color_scheme(session_ids, neuron_summary_xlsx)
        session_colors = scheme.session_color
        neuron_type_legend = scheme.type_color

    session_centroids = {
        d['session_id']: {
            'stroke_proximal': d['offset_proximal'],
            'stroke_distal': d['offset_distal'],
        }
        for d in valid_data
    }

    render_proximal_distal_aggregate(
        session_centroids=session_centroids,
        output_path=output_dir / 'rf_center_proximal_distal_aggregate.png',
        mm_limits=aggregate_mm_limits,
    )

    hotspot_valid_data = [d for d in valid_data if d['hotspot_available']]
    if hotspot_valid_data:
        session_hotspots = {
            d['session_id']: {
                'stroke_proximal': d['hotspot_offset_proximal'],
                'stroke_distal': d['hotspot_offset_distal'],
            }
            for d in hotspot_valid_data
        }
        hotspot_aggregate_mm_limits = _compute_aggregate_mm_limits_from_offsets(
            [(d['hotspot_offset_proximal'], d['hotspot_offset_distal']) for d in hotspot_valid_data]
        )
        render_proximal_distal_hotspot_aggregate(
            session_hotspots=session_hotspots,
            output_path=output_dir / 'rf_hotspot_proximal_distal_aggregate.png',
            mm_limits=hotspot_aggregate_mm_limits,
        )
    else:
        logger.warning("[RF Proximal-Distal Comparison] no sessions with hotspot data — skipping hotspot aggregate plot.")

    cc_valid_data = [d for d in valid_data if d['cc_offset_proximal'] is not None]
    if cc_valid_data:
        session_cc_offsets = {
            d['session_id']: {
                'stroke_proximal': d['cc_offset_proximal'],
                'stroke_distal': d['cc_offset_distal'],
            }
            for d in cc_valid_data
        }
        cc_aggregate_mm_limits = _compute_aggregate_mm_limits_from_offsets(
            [(d['cc_offset_proximal'], d['cc_offset_distal']) for d in cc_valid_data]
        )
        render_proximal_distal_contour_center_aggregate(
            session_contour_centers=session_cc_offsets,
            output_path=output_dir / 'rf_contour_center_proximal_distal_aggregate.png',
            mm_limits=cc_aggregate_mm_limits,
        )
    else:
        logger.warning(
            "[RF Proximal-Distal Comparison] no sessions with contour-center data — "
            "skipping contour-center aggregate plot."
        )

    # Phase 4 — cross-session aggregate figures
    render_proximal_distal_metric_deltas(
        df=df,
        output_path=output_dir / 'rf_proximal_distal_metric_deltas.png',
        session_colors=session_colors,
        neuron_type_legend=neuron_type_legend,
    )

    render_proximal_distal_population_strips(
        df=df,
        output_path=output_dir / 'rf_proximal_distal_population_strips.png',
        session_colors=session_colors,
        neuron_type_legend=neuron_type_legend,
    )

    render_shift_decomposition(
        df=df,
        output_path=output_dir / 'rf_centroid_shift_decomposition.png',
        along_col='centroid_shift_along_arm_mm',
        across_col='centroid_shift_across_arm_mm',
        title='Centroid Shift Decomposition (Proximal − Distal)',
        session_colors=session_colors,
        neuron_type_legend=neuron_type_legend,
    )

    render_shift_decomposition(
        df=df,
        output_path=output_dir / 'rf_contour_center_shift_decomposition.png',
        along_col='contour_center_shift_along_arm_mm',
        across_col='contour_center_shift_across_arm_mm',
        title='Contour Center Shift Decomposition (Proximal − Distal)',
        session_colors=session_colors,
        neuron_type_legend=neuron_type_legend,
    )

    render_shift_decomposition(
        df=df,
        output_path=output_dir / 'rf_peak_shift_decomposition.png',
        along_col='peak_shift_along_arm_mm',
        across_col='peak_shift_across_arm_mm',
        title='Peak Shift Decomposition (Proximal − Distal)',
        session_colors=session_colors,
        neuron_type_legend=neuron_type_legend,
    )

    with open(sentinel_path, 'w') as f:
        json.dump({'done': True, 'n_sessions': len(valid_data)}, f)
    logger.info("[RF Proximal-Distal Comparison] wrote sentinel → %s", sentinel_path.name)


def _compute_aggregate_mm_limits(
    valid_data: list[dict],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    pairs = [(d['offset_proximal'], d['offset_distal']) for d in valid_data]
    return _compute_aggregate_mm_limits_from_offsets(pairs)


def _compute_aggregate_mm_limits_from_offsets(
    offset_pairs: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    all_offsets = []
    for prox, dist in offset_pairs:
        all_offsets.append(prox)
        all_offsets.append(dist)
    if not all_offsets:
        return None
    pts = np.array(all_offsets)
    u_min, u_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    v_min, v_max = float(pts[:, 1].min()), float(pts[:, 1].max())
    u_margin = 0.1 * (u_max - u_min) if u_max > u_min else 1.0
    v_margin = 0.1 * (v_max - v_min) if v_max > v_min else 1.0
    return (u_min - u_margin, u_max + u_margin), (v_min - v_margin, v_max + v_margin)


def _compute_contour_overlap(
    contour_a_uv: np.ndarray,
    contour_b_uv: np.ndarray,
    grid_u: np.ndarray,
    grid_v: np.ndarray,
) -> tuple[float, float]:
    """Compute IoU and Dice overlap between two UV contour polygons.

    Rasterizes both contours onto the given grid using matplotlib.path.Path.
    Returns (iou, dice). Both are 0.0 when neither contour covers any grid cell.
    """
    from matplotlib.path import Path as MplPath

    # Build flat grid of (u, v) points from the meshgrid
    points = np.column_stack([grid_u.ravel(), grid_v.ravel()])

    mask_a = MplPath(contour_a_uv).contains_points(points)
    mask_b = MplPath(contour_b_uv).contains_points(points)

    intersection = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)
    sum_ab = np.sum(mask_a) + np.sum(mask_b)

    iou = float(intersection / union) if union > 0 else 0.0
    dice = float(2 * intersection / sum_ab) if sum_ab > 0 else 0.0

    return iou, dice


def _compute_heatmap_similarity(
    grid_z_a: np.ndarray,
    grid_z_b: np.ndarray,
) -> float:
    """Pearson correlation between two heatmap grids (finite cells only)."""
    mask = np.isfinite(grid_z_a) & np.isfinite(grid_z_b)
    if mask.sum() < 3:
        return float('nan')
    a = grid_z_a[mask]
    b = grid_z_b[mask]
    if np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])
