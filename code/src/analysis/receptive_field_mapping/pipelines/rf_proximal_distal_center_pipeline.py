"""Per-session + cross-session pipeline for RF center proximal-distal comparison."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.pipeline.shared_constants import session_id_from_path
from analysis.receptive_field_mapping.rendering.rf_center_comparison_renderer import (
    render_center_marked_heatmap,
    render_proximal_distal_aggregate,
)
from analysis.receptive_field_mapping.rendering.rf_population_map_renderer import (
    compute_standalone_figwidth,
)

logger = logging.getLogger(__name__)

_REQUIRED_GTYPES = ('all', 'stroke_proximal', 'stroke_distal')


def run_proximal_distal_center_comparison(
    session_configs: list[tuple[Path, Path]],
    force_processing: bool = False,
    heatmap_space: str = "linear",
    cmap: str = "jet",
) -> None:
    if not session_configs:
        raise ValueError("[RF Center Comparison] session_configs is empty.")

    _, db_path = session_configs[0]
    db_path = Path(db_path)
    output_dir = db_path / '4_analysed' / 'rf_center_proximal_distal'
    sentinel_path = output_dir / 'rf_center_proximal_distal_done.json'

    if sentinel_path.exists() and not force_processing:
        logger.info("[RF Center Comparison] up-to-date, skipping.")
        return

    logger.info("[RF Center Comparison] building comparison from %d sessions...", len(session_configs))

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
            db_path_item / '4_analysed' / 'population_response_fields'
            / session_id / f'{session_id}_population_response_fields.npz'
        )

        if not npz_path.exists():
            raise FileNotFoundError(
                f"[RF Center Comparison] {session_id}: NPZ not found at "
                f"{npz_path} — run extract_population_rf_response_field_boundaries first."
            )

        npz = np.load(npz_path, allow_pickle=True)

        missing = [g for g in _REQUIRED_GTYPES if f'boundary_centroid_uv_{g}' not in npz]
        if missing:
            for mg in missing:
                logger.warning(
                    "[RF Center Comparison] %s: missing centroid for '%s' — skipping session.",
                    session_id, mg,
                )
            continue

        centroid_all = npz['boundary_centroid_uv_all'].astype(np.float64)
        centroid_proximal = npz['boundary_centroid_uv_stroke_proximal'].astype(np.float64)
        centroid_distal = npz['boundary_centroid_uv_stroke_distal'].astype(np.float64)

        offset_proximal = centroid_proximal - centroid_all
        offset_distal = centroid_distal - centroid_all
        dist_uv = float(np.linalg.norm(centroid_proximal - centroid_distal))

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
            gestures_with_centroid[gtype] = (grid_u, grid_v, grid_z, centroid_uv_gtype, n_touches, threshold)

            finite_z = grid_z[np.isfinite(grid_z) & (grid_z > 0)]
            if finite_z.size > 0:
                all_grid_z_arrays.append(finite_z)

            title = (
                f"{session_id} | {gtype} | {n_touches} touches | "
                f"{centroid_uv_gtype[0]:.3f}, {centroid_uv_gtype[1]:.3f}"
            )
            all_titles.append(title)

        forearm_uv = npz['forearm_uv'].astype(np.float64)
        all_forearm_uv.append(forearm_uv)

        valid_data.append({
            'session_id': session_id,
            'db_path': db_path_item,
            'gestures_with_centroid': gestures_with_centroid,
            'forearm_uv': forearm_uv,
            'centroid_all': centroid_all,
            'centroid_proximal': centroid_proximal,
            'centroid_distal': centroid_distal,
            'offset_proximal': offset_proximal,
            'offset_distal': offset_distal,
            'dist_uv': dist_uv,
        })

        summary_rows.append({
            'session_id': session_id,
            'centroid_u_all': float(centroid_all[0]),
            'centroid_v_all': float(centroid_all[1]),
            'centroid_u_proximal': float(centroid_proximal[0]),
            'centroid_v_proximal': float(centroid_proximal[1]),
            'centroid_u_distal': float(centroid_distal[0]),
            'centroid_v_distal': float(centroid_distal[1]),
            'offset_u_proximal': float(offset_proximal[0]),
            'offset_v_proximal': float(offset_proximal[1]),
            'offset_u_distal': float(offset_distal[0]),
            'offset_v_distal': float(offset_distal[1]),
            'proximal_distal_distance_uv': dist_uv,
        })

    if not valid_data:
        logger.warning("[RF Center Comparison] no sessions had valid centroid data — nothing to do.")
        return

    if not all_grid_z_arrays:
        raise ValueError(
            "rf_center_proximal_distal: no valid heatmap data found across all sessions."
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

    aggregate_uv_limits = _compute_aggregate_uv_limits(valid_data)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Pass 2 — render per-session PNGs
    for d in valid_data:
        session_id = d['session_id']
        session_output_dir = output_dir / session_id
        session_output_dir.mkdir(parents=True, exist_ok=True)

        for gtype, (grid_u, grid_v, grid_z, centroid_uv_gtype, n_touches, _threshold) in d['gestures_with_centroid'].items():
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
            )

    # Pass 3 — cross-session outputs
    df = pd.DataFrame(summary_rows).astype({
        'session_id': str,
        'centroid_u_all': np.float64,
        'centroid_v_all': np.float64,
        'centroid_u_proximal': np.float64,
        'centroid_v_proximal': np.float64,
        'centroid_u_distal': np.float64,
        'centroid_v_distal': np.float64,
        'offset_u_proximal': np.float64,
        'offset_v_proximal': np.float64,
        'offset_u_distal': np.float64,
        'offset_v_distal': np.float64,
        'proximal_distal_distance_uv': np.float64,
    })
    csv_path = output_dir / 'rf_center_proximal_distal_summary.csv'
    df.to_csv(csv_path, index=False)
    logger.info("[RF Center Comparison] wrote %s", csv_path.name)

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
        uv_limits=aggregate_uv_limits,
    )

    with open(sentinel_path, 'w') as f:
        json.dump({'done': True, 'n_sessions': len(valid_data)}, f)
    logger.info("[RF Center Comparison] wrote sentinel → %s", sentinel_path.name)


def _compute_aggregate_uv_limits(
    valid_data: list[dict],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    all_offsets = []
    for d in valid_data:
        all_offsets.append(d['offset_proximal'])
        all_offsets.append(d['offset_distal'])
    if not all_offsets:
        return None
    pts = np.array(all_offsets)
    u_min, u_max = float(pts[:, 0].min()), float(pts[:, 0].max())
    v_min, v_max = float(pts[:, 1].min()), float(pts[:, 1].max())
    u_margin = 0.1 * (u_max - u_min) if u_max > u_min else 1.0
    v_margin = 0.1 * (v_max - v_min) if v_max > v_min else 1.0
    return (u_min - u_margin, u_max + u_margin), (v_min - v_margin, v_max + v_margin)
