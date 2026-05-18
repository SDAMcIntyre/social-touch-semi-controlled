"""Per-session pipeline that renders population RF heatmap PNGs via SLIM UV."""

import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from analysis.receptive_field_mapping.rf_data_loader import resolve_forearm_ply
from analysis.receptive_field_mapping.rf_projection import project_to_2d
from analysis.receptive_field_mapping.touch_population_data import (
    load_population_data,
    load_population_rf_data,
)
from analysis.receptive_field_mapping.rf_population_heatmap import (
    GESTURE_TYPES,
    apply_vertex_threshold,
    build_gesture_touch_indices,
    compute_rf_heatmap,
    compute_threshold_from_ratio,
    compute_unique_touch_count,
)
from analysis.receptive_field_mapping.rf_population_map_renderer import render_population_rf_map
from analysis.touch_analytics.pipeline_shared import session_id_from_path

logger = logging.getLogger(__name__)


def run_population_rf_maps(
    session_configs: list,
    neuron_mode: str,
    min_overlap_pct: float = 25.0,
    disjoint_mask_distance_mm: float = 10.0,
    force_processing: bool = False,
) -> None:
    """Render per-session 2D population RF heatmap PNGs projected via SLIM UV.

    For each session config, produces one PNG per gesture subset (all, tap,
    stroke_proximal, stroke_distal) under
    ``4_analysed/population_rf_maps/{session_id}/``.

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
    disjoint_mask_distance_mm:
        NaN-mask radius for the interpolated panel (passed to renderer).
    force_processing:
        If True, reprocess sessions even when the sentinel file exists.
    """
    for csv_path, database_path in session_configs:
        csv_path = Path(csv_path)
        database_path = Path(database_path)

        session_id = session_id_from_path(csv_path)
        output_dir = database_path / '4_analysed' / 'population_rf_maps' / session_id
        sentinel = output_dir / f'{session_id}_rf_population_maps_done.json'

        if sentinel.exists() and not force_processing:
            print(f"[Population RF Maps] {session_id}: up-to-date, skipping.")
            continue

        print(f"[Population RF Maps] {session_id}: processing...")

        # --- Resolve paths ---
        series_csv_path = (
            database_path / '4_analysed' / 'series_transforms'
            / f'{session_id}_series_augmented.csv'
        )
        if not series_csv_path.exists():
            raise FileNotFoundError(
                f"[Population RF Maps] {session_id}: series-augmented CSV not found — "
                f"run 'touch_series_transforms' first: {series_csv_path}"
            )

        forearm_ply_path = resolve_forearm_ply(csv_path.parent, session_id)
        if forearm_ply_path is None:
            raise FileNotFoundError(
                f"[Population RF Maps] {session_id}: forearm PLY not found in "
                f"{csv_path.parent} — RF-centred PLY must exist."
            )

        npz_path = (
            database_path / '4_analysed' / 'single_touch_rf_maps'
            / session_id / 'single_touch_rf_maps.npz'
        )
        if not npz_path.exists():
            raise FileNotFoundError(
                f"[Population RF Maps] {session_id}: single_touch_rf_maps.npz not found: "
                f"{npz_path}. Enable 'map_single_touch_rf' in the DAG config and re-run."
            )

        slim_cache_path = (
            database_path / '4_analysed' / 'forearm_slim_uv'
            / session_id / f'{session_id}_slim_uv.npz'
        )
        if not slim_cache_path.exists():
            raise FileNotFoundError(
                f"[Population RF Maps] {session_id}: SLIM UV cache not found: "
                f"{slim_cache_path}. Enable 'precompute_forearm_slim_uv' in the DAG "
                f"config and re-run to generate the cache."
            )

        # --- Load data ---
        pop_data = load_population_data(series_csv_path, forearm_ply_path)
        n_verts = len(pop_data.forearm_vertices)
        rf_data = load_population_rf_data(npz_path, pop_data.touch_triple_keys, n_verts)

        # --- Project all forearm vertices to SLIM UV ---
        forearm_uv = project_to_2d(
            pop_data.forearm_vertices,
            pop_data.forearm_vertices,
            pop_data.forearm_vertices.mean(axis=0),
            method='slim',
            slim_cache_path=slim_cache_path,
        )

        # --- Pass 1: compute heatmaps for all gesture subsets ---
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
                    "[Population RF Maps] %s: no touches for gesture type '%s' — skipping.",
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
            threshold = compute_threshold_from_ratio(min_overlap_pct, int(cp_mask.sum()))
            thresholded = apply_vertex_threshold(heatmap, unique_count, threshold)

            results[gtype] = (thresholded, n_gesture_touches, threshold)

        if not results:
            logger.warning(
                "[Population RF Maps] %s: no gesture subsets had touches — no PNGs produced.",
                session_id,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            _write_sentinel(sentinel, session_id, produced=[])
            continue

        # --- Pass 2: render with consistent session-wide colour scale ---
        finite_maxima = [
            float(np.nanmax(h))
            for (h, _, _) in results.values()
            if np.any(np.isfinite(h) & (h >= 0))
        ]
        if not finite_maxima:
            raise ValueError(
                f"[Population RF Maps] {session_id}: all heatmaps are empty or "
                f"below-threshold — no valid colour scale can be determined. "
                f"Check upstream RF data."
            )
        session_vmax = max(finite_maxima)

        output_dir.mkdir(parents=True, exist_ok=True)
        produced: List[Path] = []

        for gtype, (thresholded, n_touches, threshold) in results.items():
            title = (
                f"{session_id} | {gtype} | {n_touches} touches | "
                f"threshold={threshold} ({min_overlap_pct:.0f}%)"
            )
            png_path = output_dir / f'{session_id}_rf_population_{gtype}.png'

            print(f"[Population RF Maps] {session_id}: rendering '{gtype}'...")
            render_population_rf_map(
                forearm_uv=forearm_uv,
                heatmap_val=thresholded,
                vmax=session_vmax,
                title=title,
                output_path=png_path,
                disjoint_mask_distance_mm=disjoint_mask_distance_mm,
            )
            produced.append(png_path)
            print(f"[Population RF Maps] {session_id}: saved {png_path.name}")

        _write_sentinel(sentinel, session_id, produced=produced)
        print(
            f"[Population RF Maps] {session_id}: done — {len(produced)} PNG(s) written."
        )


def _write_sentinel(sentinel: Path, session_id: str, produced: List[Path]) -> None:
    with open(sentinel, 'w') as f:
        json.dump(
            {
                'session_id': session_id,
                'n_pngs': len(produced),
                'pngs': [str(p) for p in produced],
            },
            f,
            indent=2,
        )
