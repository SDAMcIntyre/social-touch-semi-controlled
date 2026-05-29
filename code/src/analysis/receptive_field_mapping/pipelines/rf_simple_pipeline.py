"""Simple per-neuron receptive field mapping pipeline.

For each session, loads per-touch population data via ``load_population_data()``,
extracts spike-associated contact vertex indices, saves them as
``spike_positions.csv`` (x, y, z), and renders a forearm heatmap PNG.

Reads the series-augmented CSV produced by ``touch_compute_series``
(``4_analysed/series_transforms/<session_id>_series_augmented.csv``).

Runs early in the analysis workflow — no dependency on feature extraction
or clustering.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from analysis.receptive_field_mapping.rendering.rf_cluster_visualizer import (
    RFRenderContext,
    render_forearm_heatmap,
)
from analysis.receptive_field_mapping.data.rf_data_loader import (
    resolve_forearm_ply,
)
from analysis.receptive_field_mapping.data.rf_extraction_io import (
    RF_CAMERA_SETTINGS_FILENAME,
    load_rf_camera_settings,
)
from analysis.receptive_field_mapping.surface.rf_projection import (
    _compute_local_radius,
    fit_cylinder_axis,
    project_to_2d,
)
from analysis.receptive_field_mapping.rendering.rf_simple_diagnostics import (
    run_diagnostics,
)
from analysis.receptive_field_mapping.surface.tangent_plane_alignment import (
    camera_settings_to_rotation,
)
from analysis.receptive_field_mapping.data.touch_population_data import (
    load_population_data,
)
from analysis.pipeline.shared_constants import session_id_from_path
from utils.should_process_task import should_process_task

logger = logging.getLogger(__name__)


def run_simple_rf_mapping(
    input_items: List[Tuple[Path, Path]],
    output_dir: Path,
    force: bool = False,
    show_interactive: bool = False,
    projection_method: Optional[str] = None,
    save_diagnostics: bool = False,
) -> List[Path]:
    """Extract raw spike-associated contact positions and render forearm heatmaps.

    For each session in ``input_items``:
    1. Loads per-touch population data via ``load_population_data()`` from the
       series-augmented CSV (``4_analysed/series_transforms/<session_id>_series_augmented.csv``).
    2. Extracts spike-associated contact vertex indices (``cp_vertex_idx[cp_spike]``).
    3. Saves ``spike_positions.csv`` (columns: x, y, z) — one row per spike contact vertex.
    4. Aggregates positions by vertex index using ``np.bincount()`` and renders
       ``<session_id>_rf_simple.png`` overlaid on the forearm PLY.

    Parameters
    ----------
    input_items:
        List of (aggregated_csv_path, database_path) tuples from the DAG runner.
        ``aggregated_csv_path`` is used only to derive ``session_id`` and locate
        the forearm PLY; the series-augmented CSV is resolved from ``database_path``.
    output_dir:
        Root output directory. Session results go under
        ``output_dir / <session_id> /``.
    force:
        If True, reprocess even if outputs are up-to-date.
    show_interactive:
        If True, display an interactive 3D PyVista window per session
        (blocking — waits for user to close before processing the next session)
        in addition to saving the PNG.

    Returns
    -------
    List of paths to produced spike_positions.csv files.
    """
    produced: List[Path] = []
    camera_settings_dir = output_dir.parent / 'rf_camera_settings'

    for csv_path, database_path in input_items:
        session_id = session_id_from_path(csv_path)
        session_out = output_dir / session_id
        sentinel = session_out / 'rf_simple_summary.json'

        # --- Resolve series-augmented CSV ---
        series_csv_path = (
            database_path / '4_analysed' / 'series_transforms'
            / f'{session_id}_series_augmented.csv'
        )
        if not series_csv_path.exists():
            raise FileNotFoundError(
                f"[RF Simple] {session_id}: series-augmented CSV not found — "
                f"run 'touch_compute_series' first: {series_csv_path}"
            )

        # --- Resolve forearm PLY (mandatory — no fallback) ---
        forearm_ply_path = resolve_forearm_ply(csv_path.parent, session_id)
        if forearm_ply_path is None:
            raise ValueError(
                f"[RF Simple] {session_id}: forearm PLY not found in {csv_path.parent} — "
                "RF-centred PLY must exist before running simple RF mapping."
            )

        # --- Idempotency check ---
        camera_settings_json = camera_settings_dir / RF_CAMERA_SETTINGS_FILENAME
        extra_inputs = [camera_settings_json] if camera_settings_json.exists() else []

        if not should_process_task(
            input_paths=[series_csv_path] + extra_inputs,
            output_paths=[sentinel],
            force=force,
        ):
            print(f"[RF Simple] {session_id}: up-to-date, skipping.")
            produced.append(session_out / 'spike_positions.csv')
            continue

        print(f"[RF Simple] {session_id}: processing...")

        # --- Load population data ---
        pop_data = load_population_data(series_csv_path, forearm_ply_path)

        # --- Extract spike contact vertices ---
        spike_vertex_indices = pop_data.cp_vertex_idx[pop_data.cp_spike]
        forearm_vertices = pop_data.forearm_vertices

        n_spikes = len(spike_vertex_indices)
        print(f"[RF Simple] {session_id}: {n_spikes} spike contact vertices.")

        # --- Save spike_positions.csv ---
        session_out.mkdir(parents=True, exist_ok=True)
        positions_csv = session_out / 'spike_positions.csv'
        if n_spikes > 0:
            spike_xyz = forearm_vertices[spike_vertex_indices]
        else:
            spike_xyz = np.empty((0, 3), dtype=np.float64)
        positions_df = pd.DataFrame(spike_xyz, columns=['x', 'y', 'z'])
        positions_df.to_csv(positions_csv, index=False)
        produced.append(positions_csv)

        # --- Render heatmap PNG ---
        all_unique_vertex_indices = np.unique(pop_data.cp_vertex_idx)
        neuron_contacts_xyz = forearm_vertices[all_unique_vertex_indices]
        spike_counts_df = pd.DataFrame()
        rotation_matrix = None

        if n_spikes > 0:
            # Build spike_counts_df: aggregate spike frame counts per vertex.
            # spike_count — total spike-frame contact-point entries per vertex.
            # unique_touch_spike_count — number of distinct touches that produced
            #   a spike at that vertex.
            spike_touch_indices = pop_data.cp_touch_idx[pop_data.cp_spike]

            raw_spike_counts = np.bincount(
                spike_vertex_indices, minlength=len(forearm_vertices)
            )
            # Compute unique touch counts per vertex: count distinct touch indices
            # that produced a spike at each vertex.
            unique_pairs = np.unique(
                np.stack([spike_vertex_indices, spike_touch_indices], axis=1), axis=0
            )
            unique_touch_counts = np.bincount(
                unique_pairs[:, 0], minlength=len(forearm_vertices)
            )

            # Restrict to vertices with at least one spike.
            nonzero_mask = raw_spike_counts > 0
            nonzero_indices = np.where(nonzero_mask)[0]
            spike_counts_df = pd.DataFrame({
                'x': forearm_vertices[nonzero_indices, 0],
                'y': forearm_vertices[nonzero_indices, 1],
                'z': forearm_vertices[nonzero_indices, 2],
                'spike_count': raw_spike_counts[nonzero_indices],
                'unique_touch_spike_count': unique_touch_counts[nonzero_indices],
            })

            # neuron_touches: number of unique touches in this session.
            neuron_touches = int(pop_data.touch_triple_keys.shape[0])

            # Simple pipeline has no clustering: the "cluster" is the whole neuron,
            # so cluster-scoped fields equal neuron-scoped fields.
            render_context = RFRenderContext(
                neuron_touches=neuron_touches,
                neuron_cluster_touches=neuron_touches,
                neuron_contacts_xyz=neuron_contacts_xyz,
                neuron_cluster_contacts_xyz=neuron_contacts_xyz,
                feature_ranges={},
            )

            cameras = load_rf_camera_settings(camera_settings_dir)
            if session_id not in cameras:
                raise ValueError(
                    f"[RF Simple] {session_id}: no camera settings found. "
                    "Run 'spatial_set_camera' first."
                )
            session_cam = cameras[session_id]
            rotation_matrix = camera_settings_to_rotation(session_cam)
            suffix = f'_{projection_method}' if projection_method else ''
            png_path = session_out / f'{session_id}_rf_simple{suffix}.png'
            try:
                render_forearm_heatmap(
                    forearm_ply_path=forearm_ply_path,
                    spike_counts_df=spike_counts_df,
                    output_path=png_path,
                    session_id=session_id,
                    cluster_label='simple',
                    interactive=show_interactive,
                    projection_method=projection_method,
                    render_context=render_context,
                    camera_settings=session_cam,
                )
                print(f"[RF Simple] {session_id}: heatmap saved -> {png_path.name}")
            except Exception:
                logger.exception(
                    "Failed to render heatmap for session %s", session_id
                )
        else:
            print(f"[RF Simple] {session_id}: no spikes — skipping PNG render.")

        # --- Diagnostic figures ---
        if save_diagnostics:
            try:
                print(f"[RF Simple] {session_id}: computing diagnostics...")
                spike_mask = pop_data.spike_elicited
                proj_method = projection_method or 'cylindrical_unwrap'
                spike_unique_vtx = (
                    np.unique(spike_vertex_indices)
                    if len(spike_vertex_indices) > 0
                    else np.array([], dtype=np.intp)
                )
                centroid = (
                    forearm_vertices[spike_unique_vtx].mean(axis=0)
                    if len(spike_unique_vtx) > 0
                    else forearm_vertices.mean(axis=0)
                )
                print(f"[RF Simple] {session_id}: projecting UV (spikes)...")
                if len(spike_unique_vtx) > 0:
                    _slim_cache_path = None
                    if proj_method == "slim":
                        _slim_cache_path = (
                            database_path / '4_analysed' / 'forearm_slim_uv'
                            / session_id / f"{session_id}_slim_uv.npz"
                        )
                    uv_spikes = project_to_2d(
                        forearm_vertices[spike_unique_vtx], forearm_vertices, centroid,
                        method=proj_method, rotation_matrix=rotation_matrix,
                        slim_cache_path=_slim_cache_path,
                    )
                else:
                    uv_spikes = np.empty((0, 2), dtype=np.float64)
                if rotation_matrix is not None:
                    axis = rotation_matrix[0]
                    _axis_point, mean_radius = _compute_local_radius(
                        forearm_vertices, centroid, axis
                    )
                    axis_direction = axis
                else:
                    axis, _axis_point, mean_radius = fit_cylinder_axis(
                        forearm_vertices, centroid
                    )
                    axis_direction = axis
                projection_metadata = {
                    'centroid': centroid,
                    'rotation_matrix': (
                        rotation_matrix if rotation_matrix is not None else np.eye(3)
                    ),
                    'axis_direction': axis_direction,
                    'mean_radius': mean_radius,
                }
                print(f"[RF Simple] {session_id}: generating figures...")
                run_diagnostics(
                    population_data=pop_data,
                    spike_mask=spike_mask,
                    forearm_vertices=forearm_vertices,
                    spike_vertex_indices=spike_vertex_indices,
                    spike_counts_df=spike_counts_df,
                    spike_xyz=spike_xyz,
                    neuron_contacts_xyz=neuron_contacts_xyz,
                    uv_spikes=uv_spikes,
                    projection_metadata=projection_metadata,
                    output_dir=session_out,
                    save=True,
                    show=show_interactive,
                )
                print(f"[RF Simple] {session_id}: diagnostics saved.")
            except Exception:
                logger.warning(
                    "Diagnostics failed for session %s", session_id, exc_info=True
                )

        # --- Save sentinel ---
        with open(sentinel, 'w') as f:
            json.dump(
                {
                    'session_id': session_id,
                    'n_spike_vertices': n_spikes,
                    'n_unique_positions': int(np.count_nonzero(
                        np.bincount(spike_vertex_indices, minlength=len(forearm_vertices))
                    )) if n_spikes > 0 else 0,
                },
                f,
                indent=2,
            )

    return produced
