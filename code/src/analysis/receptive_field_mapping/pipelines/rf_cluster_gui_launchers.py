"""RF cluster GUI launcher functions.

Pre-compute explorer caches and launch interactive PyQt5 viewer windows
for the RF Feature-Space, Touch Playback, Single-Touch RF, Touch Population,
RF Gallery, and RF Camera Settings explorers.
"""

import sys
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

from analysis.receptive_field_mapping.data.rf_data_loader import resolve_forearm_ply
from analysis.touch_analytics.pipeline_shared import session_id_from_path

from .rf_cluster_pipeline import _resolve_explorer_session_paths

logger = logging.getLogger(__name__)


def precompute_explorer_caches(
    input_items: List[Tuple[Path, Path]],
    max_workers: int = 4,
) -> None:
    """Pre-compute .npz sidecar caches for the RF Feature-Space Explorer.

    Resolves the series-augmented CSV and forearm PLY for each session, then
    calls ``load_explorer_data()`` in a thread pool so the KDTree computation
    and CSV reads run concurrently.  ``load_explorer_data`` is internally
    idempotent: sessions with a fresh ``.npz`` sidecar are skipped
    automatically (cache hit, ~200 ms each).

    Parameters
    ----------
    input_items:
        List of ``(aggregated_csv_path, database_path)`` tuples, one per session.
    max_workers:
        Number of parallel worker threads.  Defaults to 4; keep ≤ 4 to avoid
        excessive peak memory (each session may allocate ~1–2 GB during CSV
        read and contact-point array construction).
    """
    from analysis.receptive_field_mapping.data.rf_explorer_data import load_explorer_data

    session_specs = _resolve_explorer_session_paths(input_items)
    n = len(session_specs)
    if n == 0:
        return

    print(
        f"[RF Explorer] Pre-computing caches for {n} session(s)"
        f" (max_workers={min(max_workers, n)})..."
    )

    def _compute_one(spec: Tuple[str, Path, Path]) -> None:
        session_id, series_csv, forearm_ply = spec
        print(f"[RF Explorer] → {session_id} ...", flush=True)
        load_explorer_data(series_csv, forearm_ply)
        print(f"[RF Explorer] ✓ {session_id}", flush=True)

    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as executor:
        list(executor.map(_compute_one, session_specs))

    print(f"[RF Explorer] Cache pre-computation complete ({n} session(s)).")


def launch_feature_space_explorer(
    input_items: List[Tuple[Path, Path]],
) -> None:
    """Launch the RF Feature-Space Explorer GUI for all sessions in input_items.

    Resolves the series-augmented CSV and forearm PLY for each session, loads
    ``ExplorerData`` via ``load_explorer_data()`` in a thread pool, then opens
    the ``RFFeatureSpaceExplorer`` window.  When caches are warm the load is
    near-instant; on a cache miss the KDTree computation runs in parallel
    across sessions.  Blocks until the user closes the window.

    Parameters
    ----------
    input_items:
        List of ``(aggregated_csv_path, database_path)`` tuples, one per
        session — the same format used throughout the analysis pipeline.
    """
    from analysis.receptive_field_mapping.data.rf_explorer_data import load_explorer_data
    from analysis.receptive_field_mapping.gui import RFFeatureSpaceExplorer
    from PyQt5.QtWidgets import QApplication

    session_specs = _resolve_explorer_session_paths(input_items)
    n = len(session_specs)
    if n == 0:
        raise ValueError("launch_feature_space_explorer: no sessions to display.")

    print(f"[RF Explorer] Loading {n} session(s)...")

    def _load_one(spec: Tuple[str, Path, Path]) -> Tuple[str, object]:
        session_id, series_csv, forearm_ply = spec
        return session_id, load_explorer_data(series_csv, forearm_ply)

    with ThreadPoolExecutor(max_workers=min(4, n)) as executor:
        sessions: List[Tuple[str, object]] = list(executor.map(_load_one, session_specs))

    first_label, first_data = sessions[0]
    first_database_path = input_items[0][1]
    settings_path = first_database_path / "4_analysed" / "rf_explorer_settings.json"

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = RFFeatureSpaceExplorer(
        explorer_data=first_data,
        sessions=sessions,
        settings_path=settings_path,
    )
    viewer.show()
    app.exec_()


def launch_touch_playback_explorer(
    input_items: List[Tuple[Path, Path]],
) -> None:
    """Launch the Touch Playback Explorer GUI for all sessions in input_items.

    Resolves the series-augmented CSV and forearm PLY for each session, loads
    ``PlaybackData`` via ``load_playback_data()`` in a thread pool, then opens
    the ``TouchPlaybackExplorer`` window.  When caches are warm the load is
    near-instant; on a cache miss the computation runs in parallel across
    sessions.  Blocks until the user closes the window.

    Parameters
    ----------
    input_items:
        List of ``(aggregated_csv_path, database_path)`` tuples, one per
        session — the same format used throughout the analysis pipeline.
    """
    from analysis.receptive_field_mapping.data.touch_playback_data import load_playback_data
    from analysis.receptive_field_mapping.gui import TouchPlaybackExplorer
    from PyQt5.QtWidgets import QApplication

    session_specs = _resolve_explorer_session_paths(input_items)
    n = len(session_specs)
    if n == 0:
        raise ValueError("launch_touch_playback_explorer: no sessions to display.")

    print(f"[Touch Playback] Loading {n} session(s)...")

    def _load_one(spec: Tuple[str, Path, Path]) -> Tuple[str, object]:
        session_id, series_csv, forearm_ply = spec
        return session_id, load_playback_data(series_csv, forearm_ply)

    with ThreadPoolExecutor(max_workers=min(4, n)) as executor:
        sessions: List[Tuple[str, object]] = list(executor.map(_load_one, session_specs))

    first_label, first_data = sessions[0]

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = TouchPlaybackExplorer(
        playback_data=first_data,
        sessions=sessions,
    )
    viewer.show()
    app.exec_()


def launch_single_touch_rf_explorer(
    input_items: List[Tuple[Path, Path]],
    neuron_mode: str = "iff",
) -> None:
    """Launch the Single-Touch RF Explorer GUI for all sessions in input_items.

    Resolves the per-session ``.npz`` file and forearm PLY, loads
    ``SingleTouchRFViewerData`` via ``load_single_touch_rf_data()`` in a thread
    pool, then opens the ``SingleTouchRFExplorer`` window.  Blocks until the
    user closes the window.

    Parameters
    ----------
    input_items:
        List of ``(aggregated_csv_path, database_path)`` tuples, one per
        session — the same format used throughout the analysis pipeline.
    neuron_mode:
        ``"iff"`` or ``"spike"`` — must match the mode used when
        ``run_single_touch_rf_mapping`` was run.  Only used to validate the
        loaded data; the actual mode is stored in the ``.npz`` file.
    """
    from analysis.receptive_field_mapping.gui.single_touch_rf_explorer import (
        SingleTouchRFExplorer,
        load_single_touch_rf_data,
    )
    from PyQt5.QtWidgets import QApplication

    n = len(input_items)
    if n == 0:
        raise ValueError("launch_single_touch_rf_explorer: no sessions to display.")

    def _resolve_and_load(item: Tuple[Path, Path]) -> Tuple[str, object]:
        csv_path, database_path = item
        session_id = session_id_from_path(csv_path)
        npz_path = (
            database_path / "4_analysed" / "single_touch_rf_maps"
            / session_id / "single_touch_rf_maps.npz"
        )
        if not npz_path.exists():
            raise ValueError(
                f"launch_single_touch_rf_explorer: npz not found for session "
                f"'{session_id}': {npz_path}"
            )
        forearm_ply_path = resolve_forearm_ply(csv_path.parent, session_id)
        if forearm_ply_path is None:
            raise ValueError(
                f"launch_single_touch_rf_explorer: forearm PLY not found for "
                f"session '{session_id}' in {csv_path.parent}"
            )
        return session_id, load_single_touch_rf_data(npz_path, forearm_ply_path, session_id)

    print(f"[Single-Touch RF Explorer] Loading {n} session(s)...")

    with ThreadPoolExecutor(max_workers=min(4, n)) as executor:
        sessions: List[Tuple[str, object]] = list(
            executor.map(_resolve_and_load, input_items)
        )

    first_label, first_data = sessions[0]

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = SingleTouchRFExplorer(
        data=first_data,
        sessions=sessions,
    )
    viewer.show()
    app.exec_()


def launch_touch_population_explorer(
    input_items: List[Tuple[Path, Path]],
    neuron_mode: str = "iff",
) -> None:
    """Launch the Touch Population Explorer GUI for all sessions in input_items.

    Resolves the series-augmented CSV and forearm PLY for each session, loads
    ``PopulationData`` via ``load_population_data()`` in a thread pool, then opens
    the ``TouchPopulationExplorer`` window.  When caches are warm the load is
    near-instant; on a cache miss the computation runs in parallel across
    sessions.  Blocks until the user closes the window.

    For each session, also attempts to load a ``PopulationRFData`` from the
    pre-computed ``single_touch_rf_maps.npz`` (produced by
    ``run_single_touch_rf_mapping``).  If the ``.npz`` is absent for a session,
    ``None`` is stored in ``rf_sessions`` for that position — the viewer
    gracefully disables RF heatmap mode for that session.  If the file exists
    but is corrupt or misaligned, the ``ValueError`` from
    ``load_population_rf_data`` propagates immediately (fail-fast).

    Parameters
    ----------
    input_items:
        List of ``(aggregated_csv_path, database_path)`` tuples, one per
        session — the same format used throughout the analysis pipeline.
    neuron_mode:
        ``"iff"`` or ``"spike"`` — must match the mode used when
        ``run_single_touch_rf_mapping`` was run.  Passed to the viewer so it
        can label the RF heatmap axis correctly.
    """
    from analysis.receptive_field_mapping.data.touch_population_data import (
        load_population_data,
        load_population_rf_data,
        PopulationRFData,
    )
    from analysis.receptive_field_mapping.gui import TouchPopulationExplorer
    from PyQt5.QtWidgets import QApplication

    session_specs = _resolve_explorer_session_paths(input_items)
    n = len(session_specs)
    if n == 0:
        raise ValueError("launch_touch_population_explorer: no sessions to display.")

    print(f"[Touch Population] Loading {n} session(s)...")

    def _load_one(spec: Tuple[str, Path, Path]) -> Tuple[str, object]:
        session_id, series_csv, forearm_ply = spec
        touch_features_dir = series_csv.parent.parent / 'touch_features'
        return session_id, load_population_data(series_csv, forearm_ply, touch_features_dir=touch_features_dir)

    with ThreadPoolExecutor(max_workers=min(4, n)) as executor:
        sessions: List[Tuple[str, object]] = list(executor.map(_load_one, session_specs))

    # Resolve RF data per session — None if .npz absent, ValueError propagates if corrupt.
    rf_sessions: List[Optional[PopulationRFData]] = []
    for (csv_path, database_path), (session_id, pop_data) in zip(input_items, sessions):
        npz_path = (
            database_path / "4_analysed" / "single_touch_rf_maps"
            / session_id / "single_touch_rf_maps.npz"
        )
        if not npz_path.exists():
            print(
                f"[Touch Population] {session_id}: RF .npz not found — "
                f"RF heatmap mode disabled for this session."
            )
            rf_sessions.append(None)
        else:
            n_vertices = len(pop_data.forearm_vertices)
            rf_sessions.append(
                load_population_rf_data(npz_path, pop_data.touch_triple_keys, n_vertices)
            )

    first_label, first_data = sessions[0]

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = TouchPopulationExplorer(
        population_data=first_data,
        sessions=sessions,
        rf_sessions=rf_sessions,
    )
    viewer.show()
    app.exec_()


def launch_gallery_viewer(
    output_dir: Path,
    combo_name: str,
    clusterer_name: str,
    gesture_type: str | None = None,
) -> None:
    """Launch the RF cluster gallery viewer for the given combo/clusterer pair.

    Opens a PyQt5 window with a thumbnail sidebar and interactive 3D PyVista
    view.  Blocks until the user closes the window.

    Parameters
    ----------
    output_dir:
        Root of the RF cluster artifact directory (same as extraction output_dir).
    combo_name:
        Feature combination / cluster group name (e.g. ``"pressure_velocity_mean"``).
    clusterer_name:
        Clusterer profile name (e.g. ``"binning"``).
    gesture_type:
        When the cluster group used ``per_type_clustering``, pass the gesture type
        (``"tap"``, ``"stroke_proximal"``, ``"stroke_distal"``) to load the correct
        per-type artifact tree.  ``None`` loads the flat artifact tree.
    """
    from analysis.receptive_field_mapping.data.rf_gallery_data import load_gallery_data
    from analysis.receptive_field_mapping.gui import RFClusterGalleryViewer
    from PyQt5.QtWidgets import QApplication

    gallery_data = load_gallery_data(output_dir, combo_name, clusterer_name, gesture_type)
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = RFClusterGalleryViewer(gallery_data)
    viewer.show()
    app.exec_()


def launch_rf_surface_viewer(
    input_items: List[Tuple[Path, Path]],
    neuron_mode: str = "iff",
) -> None:
    """Launch the RF Surface Viewer for all sessions in input_items.

    Resolves the per-session ``_population_response_fields.npz`` produced by
    ``run_population_response_field_extraction`` and opens the ``RFSurfaceViewer`` window.
    Raises ``ValueError`` if the NPZ is absent for any session (fail-fast).
    Blocks until the user closes the window.

    Parameters
    ----------
    input_items:
        List of ``(aggregated_csv_path, database_path)`` tuples.
    neuron_mode:
        Accepted for interface consistency; the NPZ is mode-agnostic.
    """
    from analysis.receptive_field_mapping.gui.rf_surface_viewer import RFSurfaceViewer
    from PyQt5.QtWidgets import QApplication

    n = len(input_items)
    if n == 0:
        raise ValueError("launch_rf_surface_viewer: no sessions to display.")

    sessions: List[Tuple[str, Path]] = []
    for csv_path, database_path in input_items:
        session_id = session_id_from_path(csv_path)
        npz_path = (
            database_path / "4_analysed" / "population_response_fields"
            / session_id / f"{session_id}_population_response_fields.npz"
        )
        if not npz_path.exists():
            raise ValueError(
                f"launch_rf_surface_viewer: vertex NPZ not found for session "
                f"'{session_id}': {npz_path}"
            )
        sessions.append((session_id, npz_path))

    print(f"[RF Surface Viewer] Launching viewer for {n} session(s)...")

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = RFSurfaceViewer(sessions=sessions)
    viewer.show()
    app.exec_()


def launch_rf_camera_settings_viewer(
    input_items: List[Tuple[Path, Path]],
) -> None:
    """Launch the RF Camera Settings GUI for all sessions in input_items.

    Passes session paths to the viewer which loads data on-demand per session.
    Blocks until the user closes the window.
    Camera settings are saved to 4_analysed/rf_camera_settings/rf_camera_settings.json.
    """
    from analysis.receptive_field_mapping.gui.rf_camera_settings_viewer import RFCameraSettingsViewer
    from PyQt5.QtWidgets import QApplication

    session_specs = _resolve_explorer_session_paths(input_items)
    n = len(session_specs)
    if n == 0:
        raise ValueError("launch_rf_camera_settings_viewer: no sessions to display.")

    print(f"[RF Camera Settings] Launching viewer for {n} session(s)...")

    database_path = input_items[0][1]
    output_dir = database_path / '4_analysed' / 'rf_camera_settings'

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = RFCameraSettingsViewer(
        sessions=session_specs,
        output_dir=output_dir,
    )
    viewer.show()
    app.exec_()


def launch_slim_uv_config_viewer(
    input_items: List[Tuple[Path, Path]],
    dag_defaults: dict,
) -> None:
    """Launch the SLIM UV per-session config GUI for all sessions in input_items.

    For each session, resolves the forearm PLY path, the single-touch RF maps
    NPZ path, and the per-session ``forearm_slim_uv`` output directory; then
    opens the ``SlimUvConfigViewer`` window.  Blocks until the user closes
    the window.  Per-session ``slim_uv_config.yaml`` files are written to
    ``4_analysed/forearm_slim_uv/<session_id>/`` on "Accept".

    Parameters
    ----------
    input_items:
        List of ``(aggregated_csv_path, database_path)`` tuples — the same
        shape used throughout the analysis pipeline.
    dag_defaults:
        DAG-level defaults forwarded to ``make_default_config`` when a session
        has no existing per-session config.  Expected keys: ``mesh_method``,
        ``max_edge_mm``, ``clean_steps``, ``n_iter``, ``save_diagnostics``.

    Notes
    -----
    Missing ``rf_maps_npz`` is **not** raised here — the path is still passed
    to the GUI so the user can configure the session.  The error surfaces at
    "Process" time inside the worker thread with a clear message.  Missing
    ``forearm_ply_path`` **is** raised eagerly: without the raw point cloud
    the GUI has nothing to display.
    """
    from analysis.receptive_field_mapping.gui.slim_uv_config_viewer import SlimUvConfigViewer
    from PyQt5.QtWidgets import QApplication

    n = len(input_items)
    if n == 0:
        raise ValueError("launch_slim_uv_config_viewer: no sessions to display.")

    sessions: list[dict] = []
    for csv_path, database_path in input_items:
        session_id = session_id_from_path(csv_path)
        forearm_ply_path = resolve_forearm_ply(csv_path.parent, session_id)
        if forearm_ply_path is None:
            raise FileNotFoundError(
                f"launch_slim_uv_config_viewer: forearm PLY not found "
                f"for session '{session_id}' in {csv_path.parent} — "
                "run forearm extraction first."
            )
        rf_maps_npz = (
            database_path / '4_analysed' / 'single_touch_rf_maps'
            / session_id / 'single_touch_rf_maps.npz'
        )
        output_dir = database_path / '4_analysed' / 'forearm_slim_uv'
        sessions.append({
            "session_id": session_id,
            "forearm_ply_path": forearm_ply_path,
            "rf_maps_npz": rf_maps_npz,
            "output_dir": output_dir,
        })

    print(f"[SLIM UV Config] Launching viewer for {n} session(s)...")

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = SlimUvConfigViewer(
        sessions=sessions,
        dag_defaults=dag_defaults,
    )
    viewer.show()
    app.exec_()
