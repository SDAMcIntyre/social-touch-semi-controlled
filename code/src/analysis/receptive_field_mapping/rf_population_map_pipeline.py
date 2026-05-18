"""Per-session pipeline that renders population RF heatmap PNGs via SLIM UV."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
from scipy.spatial import KDTree

from analysis.receptive_field_mapping.rf_data_loader import resolve_forearm_ply
from analysis.receptive_field_mapping.forearm_slim_uv import load_slim_uv_cache
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
from analysis.receptive_field_mapping.rf_population_map_renderer import (
    render_population_rf_map,
    render_population_rf_composite,
)
from analysis.touch_analytics.pipeline_shared import session_id_from_path

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


def run_population_rf_maps(
    session_configs: list,
    neuron_mode: str,
    min_overlap_pct: float = 25.0,
    force_processing: bool = False,
) -> None:
    """Render per-session 2D population RF heatmap PNGs projected via SLIM UV.

    For each session config, produces one PNG per gesture subset (all, tap,
    stroke_proximal, stroke_distal) plus two composite PNGs (scatter and
    interpolated) under ``4_analysed/population_rf_maps/{session_id}/``.

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
    """
    # ---- Pass 1: compute heatmaps + render per-gesture PNGs ----
    composite_queue: List[_SessionCompositeData] = []

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

        # --- Load SLIM UV cache ---
        cache = load_slim_uv_cache(slim_cache_path)
        forearm_uv = cache.uv
        slim_V = cache.V
        slim_faces = cache.F

        # --- Build KDTree mapping: original forearm vertices → SLIM vertices ---
        orig_tree = KDTree(pop_data.forearm_vertices)
        distances, nearest_orig_for_slim = orig_tree.query(slim_V)
        max_dist_mm = float(distances.max())
        if max_dist_mm > 1.0:
            raise ValueError(
                f"[Population RF Maps] {session_id}: KDTree nearest-neighbour "
                f"mapping from SLIM vertices to original forearm vertices has a "
                f"maximum distance of {max_dist_mm:.3f} mm (threshold: 1 mm). "
                "The SLIM mesh and forearm PLY are misaligned — re-run "
                "'precompute_forearm_slim_uv' after verifying the forearm PLY."
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
            threshold = compute_threshold_from_ratio(min_overlap_pct, n_gesture_touches)
            thresholded = apply_vertex_threshold(heatmap, unique_count, threshold)
            slim_heatmap = thresholded[nearest_orig_for_slim]

            results[gtype] = (slim_heatmap, n_gesture_touches, threshold)

        if not results:
            logger.warning(
                "[Population RF Maps] %s: no gesture subsets had touches — no PNGs produced.",
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
                f"[Population RF Maps] {session_id}: all heatmaps are empty or "
                f"below-threshold — no valid colour scale can be determined. "
                f"Check upstream RF data."
            )
        session_vmax = max(finite_maxima)

        output_dir.mkdir(parents=True, exist_ok=True)
        produced: List[Path] = []

        for gtype, (slim_heatmap, n_touches, threshold) in results.items():
            title = (
                f"{session_id} | {gtype} | {n_touches} touches | "
                f"threshold={threshold} ({min_overlap_pct:.0f}%)"
            )
            png_path = output_dir / f'{session_id}_rf_population_{gtype}.png'

            print(f"[Population RF Maps] {session_id}: rendering '{gtype}'...")
            render_population_rf_map(
                forearm_uv=forearm_uv,
                heatmap_val=slim_heatmap,
                vmax=session_vmax,
                title=title,
                output_path=png_path,
                forearm_faces=slim_faces,
                forearm_V=slim_V,
            )
            produced.append(png_path)
            print(f"[Population RF Maps] {session_id}: saved {png_path.name}")

        _write_sentinel(sentinel, session_id, produced=produced)
        print(
            f"[Population RF Maps] {session_id}: done — {len(produced)} PNG(s) written."
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
        f"[Population RF Maps] Rendering composites: global_vmax={global_vmax:.2f}, "
        f"U=[{global_uv_xlim[0]:.1f}, {global_uv_xlim[1]:.1f}], "
        f"V=[{global_uv_ylim[0]:.1f}, {global_uv_ylim[1]:.1f}]"
    )

    for sd in composite_queue:
        for panel_type in ('scatter', 'interpolated'):
            composite_path = (
                sd.output_dir / f'{sd.session_id}_rf_population_{panel_type}_composite.png'
            )
            print(
                f"[Population RF Maps] {sd.session_id}: "
                f"rendering '{panel_type}' composite..."
            )
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
            )
            sd.produced.append(composite_path)
            print(
                f"[Population RF Maps] {sd.session_id}: saved {composite_path.name}"
            )

        _write_sentinel(sd.sentinel, sd.session_id, produced=sd.produced)


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
