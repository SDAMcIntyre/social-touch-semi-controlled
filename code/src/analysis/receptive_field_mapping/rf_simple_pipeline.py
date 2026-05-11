"""Simple per-neuron receptive field mapping pipeline.

For each session, reads all aggregated spike-associated contact positions
(forward-filled contact_points at Nerve_spike==1 rows), saves them as
spike_positions.csv (x, y, z), and renders a forearm heatmap PNG.

Runs early in the analysis workflow — no dependency on feature extraction
or clustering.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import open3d as o3d  # type: ignore
import pandas as pd
from scipy.spatial import cKDTree  # type: ignore

from analysis.receptive_field_mapping.rf_cluster_visualizer import (
    RFRenderContext,
    render_forearm_heatmap,
)
from analysis.receptive_field_mapping.rf_data_loader import (
    parse_contact_points,
    resolve_forearm_ply,
)
from analysis.receptive_field_mapping.rf_extraction_io import (
    RF_CAMERA_SETTINGS_FILENAME,
    load_rf_camera_rotation,
)
from analysis.touch_analytics.pipeline_shared import session_id_from_path
from utils.should_process_task import should_process_task

logger = logging.getLogger(__name__)

_KEY_COLS = ['block_order_id', 'trial_id', 'single_touch_id']
_DATA_COLS = ['contact_points', 'Nerve_spike']
_NEEDED_COLS = _KEY_COLS + _DATA_COLS


def _aggregate_spike_counts(
    positions_df: pd.DataFrame,
    forearm_ply_path: Optional[Path],
    session_id: str,
) -> pd.DataFrame:
    """Aggregate spike positions by nearest PLY vertex index.

    Snaps each (x, y, z) spike position to the nearest forearm PLY vertex
    using a KD-tree query, then groups by integer vertex index.  This merges
    near-duplicate positions that arise from serialization rounding and
    cross-block coordinate instability.

    Falls back to exact-float (x, y, z) groupby if the PLY is unavailable,
    empty, or in an unexpected coordinate space (mean snap distance > 2 mm).
    """
    if forearm_ply_path is not None and forearm_ply_path.exists():
        try:
            pcd = o3d.io.read_point_cloud(str(forearm_ply_path))
            ply_verts = np.asarray(pcd.points)

            if ply_verts.size > 0:
                query_pts = positions_df[['x', 'y', 'z']].to_numpy()
                tree = cKDTree(ply_verts)
                dists, vertex_indices = tree.query(query_pts, k=1)

                mean_dist = float(dists.mean())
                if mean_dist > 2.0:
                    # PLY appears to be in a different coordinate space — do not
                    # merge positions whose snap targets may be wrong.
                    logger.warning(
                        "[RF Simple] %s: mean snap distance %.2f mm > 2 mm — "
                        "PLY may be in a different coordinate space. "
                        "Falling back to exact-float groupby.",
                        session_id,
                        mean_dist,
                    )
                else:
                    # Group by vertex index, then map back to full-precision PLY
                    # coordinates (avoids .1f rounding artifacts in the output).
                    vertex_counts = (
                        pd.Series(vertex_indices)
                        .value_counts()
                        .rename_axis('vertex_idx')
                        .reset_index(name='spike_count')
                    )
                    coords = ply_verts[vertex_counts['vertex_idx'].to_numpy()]
                    spike_counts_df = pd.DataFrame({
                        'x': coords[:, 0],
                        'y': coords[:, 1],
                        'z': coords[:, 2],
                        'spike_count': vertex_counts['spike_count'].to_numpy(),
                    })
                    print(
                        f"[RF Simple] {session_id}: vertex-index aggregation → "
                        f"{len(spike_counts_df)} unique vertices "
                        f"(mean snap dist {mean_dist:.3f} mm)."
                    )
                    return spike_counts_df

        except Exception:
            logger.warning(
                "[RF Simple] %s: vertex-index aggregation failed, falling back to exact-float groupby.",
                session_id,
                exc_info=True,
            )

    # Fallback: exact-float (x, y, z) groupby — used when PLY is unavailable
    # or the snap distance check fails.
    return (
        positions_df.groupby(['x', 'y', 'z'])
        .size()
        .reset_index(name='spike_count')
    )


def run_simple_rf_mapping(
    input_items: List[Tuple[Path, Path]],
    output_dir: Path,
    force: bool = False,
    show_interactive: bool = False,
    projection_method: Optional[str] = None,
) -> List[Path]:
    """Extract raw spike-associated contact positions and render forearm heatmaps.

    For each session in ``input_items``:
    1. Reads the aggregated CSV, forward-fills ``contact_points`` within each
       touch group (30 Hz -> 1 kHz alignment), and filters rows where
       ``Nerve_spike == 1``.
    2. Saves ``spike_positions.csv`` (columns: x, y, z) — one row per parsed
       contact vertex per spike frame.
    3. Aggregates positions by PLY vertex index (KD-tree snap) and renders
       ``<session_id>_rf_simple.png`` overlaid on the forearm PLY.

    Parameters
    ----------
    input_items:
        List of (aggregated_csv_path, database_path) tuples from the DAG runner.
    output_dir:
        Root output directory. Session results go under
        ``output_dir / <session_id> /``.
    force:
        If True, reprocess even if outputs are up-to-date.
    show_interactive:
        If True, display an interactive 3D matplotlib window per session
        (blocking — waits for user to close before processing the next session)
        in addition to saving the PNG.

    Returns
    -------
    List of paths to produced spike_positions.csv files.
    """
    produced: List[Path] = []
    camera_settings_dir = output_dir.parent / 'rf_camera_settings'

    for csv_path, _ in input_items:
        session_id = session_id_from_path(csv_path)
        session_out = output_dir / session_id
        sentinel = session_out / 'rf_simple_summary.json'

        camera_settings_json = camera_settings_dir / RF_CAMERA_SETTINGS_FILENAME
        extra_inputs = [camera_settings_json] if camera_settings_json.exists() else []

        if not should_process_task(
            input_paths=[csv_path] + extra_inputs,
            output_paths=[sentinel],
            force=force,
        ):
            print(f"[RF Simple] {session_id}: up-to-date, skipping.")
            produced.append(session_out / 'spike_positions.csv')
            continue

        print(f"[RF Simple] {session_id}: processing...")

        # --- Read aggregated CSV ---
        try:
            header_df = pd.read_csv(csv_path, nrows=0)
            missing = [c for c in _NEEDED_COLS if c not in header_df.columns]
            if missing:
                logger.warning(
                    "Skipping %s: missing columns %s", csv_path.name, missing
                )
                continue
            df = pd.read_csv(csv_path, usecols=_NEEDED_COLS)
        except Exception:
            logger.exception("Error reading %s", csv_path.name)
            continue

        if df.empty:
            logger.warning("Aggregated CSV is empty, skipping: %s", csv_path.name)
            continue

        # --- Forward-fill contact_points within each touch group (30Hz -> 1kHz) ---
        df = df.copy()
        df['contact_points'] = df.groupby(_KEY_COLS)['contact_points'].ffill()

        # --- Single-pass parse: collect spike-row contacts (raw_points) and the
        # mm-rounded unique cloud across ALL rows (neuron_contacts_xyz) for the
        # render-context centroid. Mirrors _parse_contacts_for_keys() in
        # rf_cluster_pipeline.py so spike_set ⊆ neuron_set holds.
        raw_points: List[Tuple[float, float, float]] = []
        all_pts_mm: set = set()
        dedup_mm = 1.0
        for row in df.itertuples(index=False):
            parsed = parse_contact_points(row.contact_points)
            for pt in parsed:
                all_pts_mm.add(tuple(round(v / dedup_mm) * dedup_mm for v in pt))
            if row.Nerve_spike == 1:
                raw_points.extend(parsed)

        if all_pts_mm:
            neuron_contacts_xyz = np.array(sorted(all_pts_mm), dtype=float)
        else:
            neuron_contacts_xyz = np.empty((0, 3))

        neuron_touches = int(df[_KEY_COLS].drop_duplicates().shape[0])

        n_spikes = len(raw_points)
        print(f"[RF Simple] {session_id}: {n_spikes} spike contact vertices.")

        # --- Save spike_positions.csv ---
        session_out.mkdir(parents=True, exist_ok=True)
        positions_csv = session_out / 'spike_positions.csv'
        positions_df = pd.DataFrame(raw_points, columns=['x', 'y', 'z'])
        positions_df.to_csv(positions_csv, index=False)
        produced.append(positions_csv)

        # --- Resolve forearm PLY (used for aggregation and rendering) ---
        forearm_ply = resolve_forearm_ply(csv_path.parent, session_id)

        # --- Render heatmap PNG ---
        if raw_points:
            if len(neuron_contacts_xyz) == 0:
                raise ValueError(
                    f"[RF Simple] {session_id}: spike rows present but no contact "
                    "points parsed across the session — pipeline contract violation."
                )

            spike_counts_df = _aggregate_spike_counts(positions_df, forearm_ply, session_id)

            # Simple pipeline has no clustering: the "cluster" is the whole neuron,
            # so cluster-scoped fields equal neuron-scoped fields.
            render_context = RFRenderContext(
                neuron_touches=neuron_touches,
                neuron_cluster_touches=neuron_touches,
                neuron_contacts_xyz=neuron_contacts_xyz,
                neuron_cluster_contacts_xyz=neuron_contacts_xyz,
                feature_ranges={},
            )

            session_R = load_rf_camera_rotation(camera_settings_dir, session_id)
            suffix = f'_{projection_method}' if projection_method else ''
            png_path = session_out / f'{session_id}_rf_simple{suffix}.png'
            try:
                render_forearm_heatmap(
                    forearm_ply_path=forearm_ply,
                    spike_counts_df=spike_counts_df,
                    output_path=png_path,
                    session_id=session_id,
                    cluster_label='simple',
                    interactive=show_interactive,
                    projection_method=projection_method,
                    render_context=render_context,
                    rotation_matrix=session_R,
                )
                print(f"[RF Simple] {session_id}: heatmap saved -> {png_path.name}")
            except Exception:
                logger.exception(
                    "Failed to render heatmap for session %s", session_id
                )
        else:
            print(f"[RF Simple] {session_id}: no spikes — skipping PNG render.")

        # --- Save sentinel ---
        with open(sentinel, 'w') as f:
            json.dump(
                {
                    'session_id': session_id,
                    'n_spike_vertices': n_spikes,
                    'n_unique_positions': len(positions_df.drop_duplicates()) if not positions_df.empty else 0,
                },
                f,
                indent=2,
            )

    return produced
