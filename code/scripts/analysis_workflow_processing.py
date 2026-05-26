# analysis_workflow_processing.py
# Self-contained processing entry script. Holds only the 20 processing-slice
# flows, their imports, and dispatch logic.  Does not import from
# analysis_workflow_viewers.py.
import argparse
import logging
from multiprocessing import Queue, freeze_support
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from prefect import flow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from utils import path_tools
from utils import DagConfigHandler, PipelineMonitor, TaskExecutor
from utils.pipeline.session_config_resolver import resolve_session_configs
from primary_processing import KinectConfigFileHandler, KinectConfig

from analysis.touch_analytics import generate_ap_efficacy_matrix, generate_session_summary
from analysis.touch_analytics.preparation_pipeline import run_preparation
from analysis.touch_analytics.series_pipeline import run_series_transforms
from analysis.touch_analytics.extraction_pipeline import run_feature_extraction
from analysis.touch_analytics.clustering_pipeline import run_clustering
from analysis.touch_analytics.comparing_pipeline import run_comparing
from analysis.receptive_field_mapping import (
    run_cluster_rf_extraction,
    run_cluster_rf_metrics_computation,
    run_cluster_rf_mapping,
    run_cluster_rf_visualization,
    run_simple_rf_mapping,
    run_single_touch_rf_mapping,
    run_population_rf_grid,
    PopulationRFGridConfig,
    run_population_rf_grid_metrics,
    PopulationRFGridMetricsConfig,
    run_population_rf_grid_metrics_visualization,
    run_session_comparison_visualization,
    run_population_response_field_extraction,
    run_session_rf_boundary_comparison,
    run_proximal_distal_center_comparison,
    run_touch_feature_radar,
    launch_rf_camera_settings_viewer,
    launch_slim_uv_config_viewer,
)
from analysis.receptive_field_mapping.data.rf_data_loader import resolve_forearm_ply
from analysis.receptive_field_mapping.data.rf_extraction_io import load_rf_camera_settings
from analysis.receptive_field_mapping.surface.slim_uv_config_io import (
    config_path_for_session,
    load_slim_uv_config,
    make_default_config,
    config_hash as compute_config_hash,
)
from analysis.touch_analytics.pipeline_shared import session_id_from_path

from analysis.pipeline import collect_unique_session_dirs, discover_input_items, run_pipeline_stages

# --- Module-level helpers ---

def _collect_unified_files(input_items: List[Tuple[Path, Path]]) -> List[Path]:
    """
    Reconstruct expected paths of touch-summary CSVs written by touch_feature_extraction.
    Scans all extraction profile subdirectories under '4_analysed/unified_touches/'.
    Returns deduplicated list of existing paths.
    """
    seen: set = set()
    unified_files = []
    for input_file, database_path in input_items:
        unified_root = database_path / "4_analysed" / "touch_features"
        filename = input_file.name

        if "_semicontrolled_" in filename:
            prefix = filename.split("_semicontrolled_")[0]
            new_filename = f"{prefix}_semicontrolled_touch_summary.csv"
        else:
            new_filename = f"{input_file.stem}_touch_summary.csv"

        for profile_dir in sorted(unified_root.glob("*")):
            if not profile_dir.is_dir():
                continue
            candidate = profile_dir / new_filename
            if candidate.exists() and candidate not in seen:
                seen.add(candidate)
                unified_files.append(candidate)

    return unified_files


# --- Processing Flows ---

@flow(name="summarize_session_blocks")
def summarize_session_blocks_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
) -> List[Path]:
    """
    STEP 1: Session Block Summary.
    Reads all aggregated session CSVs and writes a single combined CSV
    (``4_analysed/session_block_summary.csv``) with one row per block,
    describing block IDs, trial counts, and trial ID lists.
    """
    print(f"[Batch Analysis] Generating session block summary for {len(input_items)} item(s)...")

    if not input_items:
        logging.warning("No input items provided for session block summary.")
        return []

    input_paths = [item[0] for item in input_items]
    anchor_db_path = input_items[0][1]

    output_dir = anchor_db_path / "4_analysed" / "session_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "session_block_summary.csv"

    try:
        result_path = generate_session_summary(
            input_paths=input_paths,
            output_path=output_path,
            force=force_processing,
        )
        return [result_path]
    except Exception as exc:
        logging.error(f"Failed to generate session block summary: {exc}")
        return []


@flow(name="map_receptive_fields_simple")
def map_receptive_fields_simple_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    show_interactive: bool = False,
    projection_method: str = None,
    save_diagnostics: bool = False,
) -> List[Path]:
    """
    Simple RF mapping: raw spike-position CSV + forearm heatmap per session.
    Runs before feature extraction — no dependency on clustering or extraction.
    Output: ``4_analysed/receptive_field_maps_simple/<session_id>/``
    """
    print(f"[Batch Analysis] Running simple RF mapping for {len(input_items)} item(s)...")
    if not input_items:
        return []
    output_dir = input_items[0][1] / '4_analysed' / 'receptive_field_maps_simple'
    return run_simple_rf_mapping(
        input_items=input_items,
        output_dir=output_dir,
        force=force_processing,
        show_interactive=show_interactive,
        projection_method=projection_method,
        save_diagnostics=save_diagnostics,
    )


@flow(name="configure_forearm_slim_uv")
def configure_forearm_slim_uv_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    interactive: bool = True,
    mesh_method: str = "bpa",
    max_edge_mm: float = 0.0,
    n_iter: int = 40,
    save_diagnostics: bool = True,
    clean_steps: dict | None = None,
) -> None:
    """Launch the per-session SLIM UV config GUI for sessions missing a config.

    For each session, checks whether ``4_analysed/forearm_slim_uv/<session_id>/
    slim_uv_config.yaml`` already exists. Sessions that already have a config
    are skipped (unless ``force_processing`` is True, which forces the GUI for
    every session). The GUI saves per-session YAML configs consumed by
    ``precompute_forearm_slim_uv_flow``.

    When ``interactive`` is False the flow is a no-op: sessions without a
    config will simply fall back to DAG defaults in the precompute step.
    """
    if not input_items:
        return

    if not interactive:
        print(
            "[SLIM UV Config] interactive=false — skipping GUI; sessions "
            "without a per-session YAML will use DAG defaults."
        )
        return

    database_path = input_items[0][1]
    output_dir = database_path / '4_analysed' / 'forearm_slim_uv'

    if force_processing:
        missing_items = list(input_items)
        print(
            f"[SLIM UV Config] force_processing=true — launching viewer for "
            f"all {len(missing_items)} session(s)."
        )
    else:
        missing_items = []
        for csv_path, db_path in input_items:
            session_id = session_id_from_path(csv_path)
            cfg_path = config_path_for_session(
                db_path / '4_analysed' / 'forearm_slim_uv',
                session_id,
            )
            if not cfg_path.exists():
                missing_items.append((csv_path, db_path))

        if not missing_items:
            print(
                f"[SLIM UV Config] All {len(input_items)} session(s) already "
                "have slim_uv_config.yaml — skipping. "
                "Set force_processing=true to re-open."
            )
            return

        print(
            f"[SLIM UV Config] {len(missing_items)}/{len(input_items)} "
            "session(s) missing per-session config — launching viewer."
        )

    dag_defaults = {
        "mesh_method": mesh_method,
        "max_edge_mm": float(max_edge_mm),
        "n_iter": int(n_iter),
        "save_diagnostics": bool(save_diagnostics),
        "clean_steps": clean_steps or {},
    }

    launch_slim_uv_config_viewer(missing_items, dag_defaults)


@flow(name="precompute_forearm_slim_uv")
def precompute_forearm_slim_uv_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    n_iter: int = 40,
    save_diagnostics: bool = False,
    interactive: bool = False,
    clean_steps: dict | None = None,
    mesh_method: str = "bpa",
    max_edge_mm: float | None = None,
) -> List[Path]:
    """Precompute and cache SLIM UV maps for all sessions.

    For each session, loads the forearm PLY (via load_or_build_forearm_mesh)
    and the single-touch RF maps NPZ (from map_single_touch_rf output), runs
    SLIM, and caches the UV map as
    ``4_analysed/forearm_slim_uv/<session_id>/<session_id>_slim_uv.npz``.

    The UV origin (center_vid) is placed at the IFF-weighted centroid of all
    single-touch RF maps, giving a neuroscientifically meaningful anchor point.

    Per-session ``slim_uv_config.yaml`` files written by
    ``configure_forearm_slim_uv_flow`` override the DAG-level params on a
    session-by-session basis.  Missing per-session YAML legitimately falls
    back to DAG defaults (designed behaviour per the plan).
    """
    from analysis.receptive_field_mapping.surface.forearm_slim_uv import (
        precompute_forearm_slim_uv as _precompute,
    )
    from analysis.receptive_field_mapping.data.rf_data_loader import resolve_forearm_ply
    from analysis.touch_analytics.pipeline_shared import session_id_from_path
    from utils.should_process_task import should_process_task

    print(f"[SLIM UV] Precomputing SLIM UV maps for {len(input_items)} item(s)...")
    if not input_items:
        return []

    results: List[Path] = []

    for csv_path, db_path in input_items:
        session_id = session_id_from_path(csv_path)

        forearm_ply_path = resolve_forearm_ply(csv_path.parent, session_id)
        if forearm_ply_path is None:
            raise ValueError(
                f"[SLIM UV] {session_id}: forearm PLY not found in {csv_path.parent} — "
                "run forearm extraction first."
            )

        rf_maps_npz = (
            db_path / '4_analysed' / 'single_touch_rf_maps'
            / session_id / 'single_touch_rf_maps.npz'
        )

        output_dir = db_path / '4_analysed' / 'forearm_slim_uv' / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_path = output_dir / f"{session_id}_slim_uv.npz"

        cfg_path = config_path_for_session(
            db_path / '4_analysed' / 'forearm_slim_uv', session_id,
        )
        if cfg_path.exists():
            cfg = load_slim_uv_config(cfg_path)
            logging.info(
                f"[SLIM UV] {session_id}: using per-session config from {cfg_path}"
            )
            session_mesh_method = cfg.mesh_method
            session_max_edge_mm = None if cfg.max_edge_mm == 0.0 else cfg.max_edge_mm
            session_clean_steps = cfg.clean_steps.to_dict()
            session_n_iter = cfg.n_iter
            session_save_diagnostics = cfg.save_diagnostics
        else:
            logging.info(
                f"[SLIM UV] {session_id}: no per-session config; using DAG defaults"
            )
            session_mesh_method = mesh_method
            session_max_edge_mm = (
                None if (max_edge_mm is None or max_edge_mm == 0.0) else max_edge_mm
            )
            session_clean_steps = clean_steps or None
            session_n_iter = n_iter
            session_save_diagnostics = save_diagnostics

        # Build the expected config hash for staleness comparison against the NPZ.
        make_default_kwargs: dict = {
            "mesh_method": session_mesh_method,
            "max_edge_mm": (
                0.0 if session_max_edge_mm is None else float(session_max_edge_mm)
            ),
            "n_iter": int(session_n_iter),
            "save_diagnostics": bool(session_save_diagnostics),
        }
        if session_clean_steps:
            make_default_kwargs["clean_steps"] = session_clean_steps
        expected_cfg = make_default_config(session_id, **make_default_kwargs)
        expected_hash = compute_config_hash(expected_cfg)

        if not should_process_task(
            input_paths=[forearm_ply_path, rf_maps_npz],
            output_paths=[cache_path],
            force=force_processing,
        ):
            _force_for_config_change = False
            if cache_path.exists():
                try:
                    import numpy as _np
                    _cached_data = _np.load(cache_path, allow_pickle=False)
                    _cached_hash = (
                        str(_cached_data['config_hash'])
                        if 'config_hash' in _cached_data.files else ""
                    )
                    if _cached_hash != expected_hash:
                        print(
                            f"[SLIM UV] {session_id}: config_hash changed "
                            f"({_cached_hash!r} → {expected_hash!r}) — forcing recompute."
                        )
                        _force_for_config_change = True
                except Exception as _exc:
                    print(
                        f"[SLIM UV] {session_id}: could not read cached config_hash "
                        f"({_exc}) — forcing recompute."
                    )
                    _force_for_config_change = True
            if not _force_for_config_change:
                print(f"[SLIM UV] {session_id}: up-to-date, skipping.")
                results.append(cache_path)
                continue

        camera_settings_dir = db_path / '4_analysed' / 'rf_camera_settings'
        print(
            f"[SLIM UV] {session_id}: computing SLIM UV map "
            f"(n_iter={session_n_iter}, mesh_method={session_mesh_method!r})..."
        )
        result = _precompute(
            forearm_ply_path=forearm_ply_path,
            rf_maps_npz=rf_maps_npz,
            cache_path=cache_path,
            n_iter=session_n_iter,
            save_diagnostics=session_save_diagnostics,
            camera_settings_dir=camera_settings_dir if session_save_diagnostics else None,
            interactive=interactive,
            clean_steps=session_clean_steps,
            mesh_method=session_mesh_method,
            max_edge_mm=session_max_edge_mm,
        )
        print(f"[SLIM UV] {session_id}: cached → {result.name}")
        results.append(result)

    return results


@flow(name="map_single_touch_rf")
def map_single_touch_rf_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    neuron_mode: str = "iff",
    preparation_dir: Optional[Path] = None,
) -> List[Path]:
    """Per-touch RF mapping: accumulate per-vertex neuron values for every single touch.

    Reads the prepared CSV produced by ``touch_preparation`` via
    ``load_playback_data()``, accumulates IFF or spike values per forearm vertex
    for each touch event, and saves results as a sparse ``.npz`` file.
    Output: ``4_analysed/single_touch_rf_maps/<session_id>/``
    """
    print(f"[Batch Analysis] Running single-touch RF mapping for {len(input_items)} item(s)...")
    if not input_items:
        return []
    output_dir = input_items[0][1] / '4_analysed' / 'single_touch_rf_maps'
    return run_single_touch_rf_mapping(
        input_items=input_items,
        output_dir=output_dir,
        force=force_processing,
        neuron_mode=neuron_mode,
        preparation_dir=preparation_dir,
    )


@flow(name="extract_population_rf_response_field_boundaries")
def extract_population_rf_response_field_boundaries_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    neuron_mode: str = "iff",
    min_overlap_pct: float = 25.0,
    median_filter_size: int | None = None,
    inflection_sigma: float | None = None,
    heatmap_space: str = "linear",
    cmap: str = "jet",
) -> None:
    """Render per-session 2D population RF heatmap PNGs projected via SLIM UV.

    For each session, produces one PNG per gesture subset (all, tap,
    stroke_proximal, stroke_distal) under
    ``4_analysed/population_response_fields/<session_id>/``.
    Idempotent via sentinel JSON.
    """
    print(f"[Batch Analysis] Extracting population response field boundaries for {len(input_items)} item(s)...")
    if not input_items:
        return

    run_population_response_field_extraction(
        session_configs=input_items,
        neuron_mode=neuron_mode,
        min_overlap_pct=min_overlap_pct,
        force_processing=force_processing,
        median_filter_size=median_filter_size,
        inflection_sigma=inflection_sigma,
        heatmap_space=heatmap_space,
        cmap=cmap,
    )


@flow(name="compare_session_rf_boundaries")
def compare_session_rf_boundaries_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
) -> None:
    """Aggregate RF boundary metrics across sessions and render comparison visuals.

    Reads per-session NPZ files produced by extract_population_rf_response_field_boundaries,
    builds a summary CSV, and renders UV contour overlays, per-gesture metric panels,
    and session-x-gesture heatmaps.
    Output: 4_analysed/session_rf_boundary_comparison/
    """
    print(f"[Batch Analysis] Comparing session RF boundaries for {len(input_items)} item(s)...")
    if not input_items:
        return

    run_session_rf_boundary_comparison(
        session_configs=input_items,
        force_processing=force_processing,
    )


@flow(name="compare_rf_center_proximal_distal")
def compare_rf_center_proximal_distal_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    heatmap_space: str = "linear",
    cmap: str = "jet",
) -> None:
    """Compare RF center positions between proximal and distal strokes across sessions.

    Reads per-session NPZ files produced by extract_population_rf_response_field_boundaries,
    renders per-session center-marked heatmap PNGs, and produces a cross-session
    aggregate scatter plot and summary CSV.
    Output: 4_analysed/rf_center_proximal_distal/
    """
    print(f"[Batch Analysis] Comparing RF centers for {len(input_items)} item(s)...")
    if not input_items:
        return

    run_proximal_distal_center_comparison(
        session_configs=input_items,
        force_processing=force_processing,
        heatmap_space=heatmap_space,
        cmap=cmap,
    )


@flow(name="map_population_rf_grid")
def map_population_rf_grid_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    neuron_mode: str = "iff",
    per_gesture_type: bool = True,
    vertex_threshold_ratio: float = 0.25,
    features: Optional[dict] = None,
    compute_baseline: bool = True,
    grid_groups: Optional[dict] = None,
) -> List[Path]:
    """Systematic RF population mapping via feature-space grid sweep.

    For each session, builds an N-dimensional grid over configured touch features,
    filters touches per cell, averages their single-touch RF maps, and saves
    one NPZ per cell grid (or per gesture type if configured).
    Output: ``4_analysed/population_rf_grid/<session_id>/`` (flat, backward compat)
    or ``4_analysed/population_rf_grid/<group_name>/<session_id>/`` (when grid_groups).
    """
    print(f"[Batch Analysis] Running population RF grid for {len(input_items)} item(s)...")
    if not input_items:
        return []

    # Resolve enabled groups or fall back to flat params for backward compat.
    if grid_groups:
        enabled_groups = {
            name: cfg
            for name, cfg in grid_groups.items()
            if cfg.get("enabled", True)
        }
        if not enabled_groups:
            logging.info("map_population_rf_grid_flow: all grid groups are disabled — skipping.")
            return []
    else:
        # Backward compat: wrap flat params into a single anonymous group.
        if features is None:
            raise ValueError(
                "map_population_rf_grid_flow: 'features' config is required — "
                "define at least one feature in the DAG config."
            )
        enabled_groups = {None: {
            "features": features,
            "neuron_mode": neuron_mode,
            "per_gesture_type": per_gesture_type,
            "vertex_threshold_ratio": vertex_threshold_ratio,
            "compute_baseline": compute_baseline,
        }}

    database_path = input_items[0][1]
    output_dir = database_path / '4_analysed'
    touch_features_dir = database_path / '4_analysed' / 'touch_features'

    resolved_items = []
    for csv_path, db_path in input_items:
        session_id = session_id_from_path(csv_path)
        series_csv_path = (
            db_path / '4_analysed' / 'series_transforms'
            / f'{session_id}_series_augmented.csv'
        )
        if not series_csv_path.exists():
            raise ValueError(
                f"map_population_rf_grid_flow: series-augmented CSV not found for "
                f"session '{session_id}': {series_csv_path}"
            )
        forearm_ply_path = resolve_forearm_ply(csv_path.parent, session_id)
        if forearm_ply_path is None:
            raise ValueError(
                f"map_population_rf_grid_flow: forearm PLY not found for "
                f"session '{session_id}' in {csv_path.parent}"
            )
        npz_path = (
            db_path / '4_analysed' / 'single_touch_rf_maps'
            / session_id / 'single_touch_rf_maps.npz'
        )
        if not npz_path.exists():
            raise ValueError(
                f"map_population_rf_grid_flow: single_touch_rf_maps.npz not found for "
                f"session '{session_id}': {npz_path}. Run map_single_touch_rf first."
            )
        resolved_items.append({
            "series_csv_path": series_csv_path,
            "npz_path": npz_path,
            "forearm_ply_path": forearm_ply_path,
            "session_id": session_id,
            "touch_features_dir": touch_features_dir if touch_features_dir.exists() else None,
        })

    all_results = []
    for group_name, group_cfg in enabled_groups.items():
        config = PopulationRFGridConfig(
            features=group_cfg["features"],
            neuron_mode=group_cfg.get("neuron_mode", neuron_mode),
            vertex_threshold_ratio=group_cfg.get("vertex_threshold_ratio", vertex_threshold_ratio),
            per_gesture_type=group_cfg.get("per_gesture_type", per_gesture_type),
            compute_baseline=group_cfg.get("compute_baseline", compute_baseline),
        )
        results = run_population_rf_grid(
            input_items=resolved_items,
            output_dir=output_dir,
            config=config,
            force=force_processing,
            group_name=group_name,
        )
        all_results.extend(results)
    return all_results


@flow(name="reduce_population_rf_grid")
def reduce_population_rf_grid_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    projection_method: str = "tangent_plane",
    grid_group_defs: dict = None,
) -> List[Path]:
    """Reduce per-gesture-type population RF grid NPZ files to scalar metric CSVs.

    Reads NPZ files produced by ``map_population_rf_grid`` and computes per-cell
    scalar descriptors (IFF intensity, topographic, distributional, shape metrics).
    Output: ``4_analysed/population_rf_grid_metrics/<session_id>/`` (flat, backward compat)
    or ``4_analysed/population_rf_grid_metrics/<group_name>/<session_id>/`` (when grid_group_defs).
    """
    print(f"[Batch Analysis] Running population RF grid metrics for {len(input_items)} item(s)...")
    if not input_items:
        return []

    # Resolve enabled group names or fall back to [None] for backward compat.
    if grid_group_defs:
        enabled_group_names = [
            name for name, cfg in grid_group_defs.items()
            if cfg.get("enabled", True)
        ]
    else:
        enabled_group_names = [None]

    database_path = input_items[0][1]
    output_dir = database_path / '4_analysed'
    slim_uv_cache_dir = (
        database_path / '4_analysed' / 'forearm_slim_uv'
        if projection_method == 'slim' else None
    )

    config = PopulationRFGridMetricsConfig(projection_method=projection_method)

    all_results = []
    for group_name in enabled_group_names:
        resolved_items = []
        for csv_path, db_path in input_items:
            session_id = session_id_from_path(csv_path)
            forearm_ply_path = resolve_forearm_ply(csv_path.parent, session_id)
            if forearm_ply_path is None:
                raise ValueError(
                    f"reduce_population_rf_grid_flow: forearm PLY not found for "
                    f"session '{session_id}' in {csv_path.parent}"
                )
            if group_name is not None:
                grid_dir = db_path / '4_analysed' / 'population_rf_grid' / group_name / session_id
            else:
                grid_dir = db_path / '4_analysed' / 'population_rf_grid' / session_id
            if not grid_dir.exists():
                raise ValueError(
                    f"reduce_population_rf_grid_flow: population_rf_grid directory not found "
                    f"for session '{session_id}': {grid_dir}. Run map_population_rf_grid first."
                )
            npz_files = list(grid_dir.glob("population_rf_grid_*.npz"))
            if not npz_files:
                raise ValueError(
                    f"reduce_population_rf_grid_flow: no NPZ files found for session "
                    f"'{session_id}' in {grid_dir}. Run map_population_rf_grid first."
                )
            resolved_items.append({
                "forearm_ply_path": forearm_ply_path,
                "grid_dir": grid_dir,
                "session_id": session_id,
            })
        results = run_population_rf_grid_metrics(
            input_items=resolved_items,
            output_dir=output_dir,
            config=config,
            force=force_processing,
            group_name=group_name,
            slim_uv_cache_dir=slim_uv_cache_dir,
        )
        all_results.extend(results)
    return all_results


@flow(name="visualize_population_rf_grid_metrics")
def visualize_population_rf_grid_metrics_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    extracted_features: list = None,
    grid_group_defs: dict = None,
) -> None:
    """Render per-metric heatmap PNGs from population RF grid metrics CSVs.

    Reads CSVs produced by ``reduce_population_rf_grid`` and renders one PNG
    per IFF metric per session+gesture type into a metric-organized folder.
    Output: ``4_analysed/population_rf_grid_metrics_heatmaps/<metric>/`` (flat, backward compat)
    or ``4_analysed/population_rf_grid_metrics_heatmaps/<group_name>/<metric>/`` (when grid_group_defs).
    """
    print(f"[Batch Analysis] Visualizing population RF grid metrics for {len(input_items)} item(s)...")
    if not input_items:
        return

    # Resolve enabled group names or fall back to [None] for backward compat.
    if grid_group_defs:
        enabled_group_names = [
            name for name, cfg in grid_group_defs.items()
            if cfg.get("enabled", True)
        ]
    else:
        enabled_group_names = [None]

    database_path = input_items[0][1]

    resolved_items = []
    for csv_path, db_path in input_items:
        session_id = session_id_from_path(csv_path)
        resolved_items.append({"session_id": session_id})

    for group_name in enabled_group_names:
        if group_name is not None:
            output_dir = database_path / '4_analysed' / 'population_rf_grid_metrics_heatmaps' / group_name
            metrics_base_dir = database_path / '4_analysed' / 'population_rf_grid_metrics' / group_name
        else:
            output_dir = database_path / '4_analysed' / 'population_rf_grid_metrics_heatmaps'
            metrics_base_dir = None
        run_population_rf_grid_metrics_visualization(
            input_items=resolved_items,
            output_dir=output_dir,
            force=force_processing,
            extracted_features=extracted_features,
            metrics_base_dir=metrics_base_dir,
        )


@flow(name="visualize_session_comparison")
def visualize_session_comparison_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    features: dict = None,
    neuron_mode: str = "iff",
    vertex_threshold_ratio: float = 0.25,
    metric_name: str = "mean_iff",
    projection_method: str = "tangent_plane",
) -> None:
    """Render cross-session comparison heatmaps from 1D population RF grids.

    Builds a 1D feature-space grid, computes RF metrics, and renders one PNG
    per gesture type with sessions on Y-axis and feature bins on X-axis.
    Output: ``4_analysed/session_comparison/session_comparison_heatmaps/<metric>/``
    """
    print(f"[Batch Analysis] Visualizing session comparison for {len(input_items)} item(s)...")
    if not input_items:
        return

    if features is None:
        raise ValueError(
            "visualize_session_comparison_flow: 'features' config is required — "
            "define exactly one feature in the DAG config."
        )

    database_path = input_items[0][1]
    touch_features_dir = database_path / '4_analysed' / 'touch_features'

    resolved_items = []
    for csv_path, db_path in input_items:
        session_id = session_id_from_path(csv_path)
        series_csv_path = (
            db_path / '4_analysed' / 'series_transforms'
            / f'{session_id}_series_augmented.csv'
        )
        if not series_csv_path.exists():
            raise ValueError(
                f"visualize_session_comparison_flow: series-augmented CSV not found for "
                f"session '{session_id}': {series_csv_path}"
            )
        forearm_ply_path = resolve_forearm_ply(csv_path.parent, session_id)
        if forearm_ply_path is None:
            raise ValueError(
                f"visualize_session_comparison_flow: forearm PLY not found for "
                f"session '{session_id}' in {csv_path.parent}"
            )
        npz_path = (
            db_path / '4_analysed' / 'single_touch_rf_maps'
            / session_id / 'single_touch_rf_maps.npz'
        )
        if not npz_path.exists():
            raise ValueError(
                f"visualize_session_comparison_flow: single_touch_rf_maps.npz not found for "
                f"session '{session_id}': {npz_path}. Run map_single_touch_rf first."
            )
        resolved_items.append({
            "series_csv_path": series_csv_path,
            "npz_path": npz_path,
            "forearm_ply_path": forearm_ply_path,
            "session_id": session_id,
            "touch_features_dir": touch_features_dir if touch_features_dir.exists() else None,
        })

    run_session_comparison_visualization(
        input_items=resolved_items,
        database_path=database_path,
        features=features,
        metric_name=metric_name,
        neuron_mode=neuron_mode,
        vertex_threshold_ratio=vertex_threshold_ratio,
        projection_method=projection_method,
        force=force_processing,
    )


@flow(name="touch_preparation")
def touch_preparation_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    preparation: dict = None,
) -> List[Path]:
    """
    Stage 1: Per-session data preparation (block-ID synthesis + NaN-gap interpolation).
    Writes ``4_analysed/preparation/<session_id>_prepared.csv``.
    """
    print(f"[Batch Analysis] Running touch preparation for {len(input_items)} item(s)...")
    if not input_items:
        return []
    output_dir = input_items[0][1] / '4_analysed' / 'preparation'
    return run_preparation(
        input_items=input_items,
        preparation_cfg=preparation or {},
        output_dir=output_dir,
        force=force_processing,
    )


@flow(name="touch_series_transforms")
def touch_series_transforms_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    transforms: dict = None,
    preparation_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Stage 2a: Per-session series-level transforms (kinematics, pressure, MoS).
    Writes ``4_analysed/series_transforms/<session>_series_augmented.csv``.
    """
    print(f"[Batch Analysis] Running touch series transforms for {len(input_items)} item(s)...")
    if not input_items:
        return []
    output_dir = input_items[0][1] / '4_analysed' / 'series_transforms'
    return run_series_transforms(
        input_items=input_items,
        transforms=transforms or {},
        output_dir=output_dir,
        force=force_processing,
        preparation_dir=preparation_dir,
    )


@flow(name="touch_feature_extraction")
def touch_feature_extraction_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    features: dict = None,
    series_dir: Optional[Path] = None,
    preparation_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Stage 2b: Per-session feature extraction.
    Writes one CSV per (session, feature) to
    ``4_analysed/touch_features/<feature>/<session>_touch_summary.csv``.
    """
    print(f"[Batch Analysis] Running touch feature extraction for {len(input_items)} item(s)...")
    if not input_items:
        return []
    feature_dict = features or {'max': {'enabled': True}}
    output_dir = input_items[0][1] / '4_analysed' / 'touch_features'
    per_feature = run_feature_extraction(
        input_items=input_items,
        features=feature_dict,
        output_dir=output_dir,
        force=force_processing,
        series_dir=series_dir,
        preparation_dir=preparation_dir,
    )
    return [path for paths in per_feature.values() for path in paths]


@flow(name="render_touch_feature_radar")
def render_touch_feature_radar_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    radar_groups: dict = None,
) -> None:
    """
    Stage 2c: Per-session radar plots of touch feature distributions.
    Writes one PNG per gesture type + composite to
    ``4_analysed/touch_feature_radar/<group_name>/<session_id>/``.
    """
    print(f"[Batch Analysis] Rendering touch feature radar plots for {len(input_items)} item(s)...")
    if not input_items:
        return
    run_touch_feature_radar(
        session_configs=input_items,
        radar_groups=radar_groups,
        force_processing=force_processing,
    )


@flow(name="touch_clustering")
def touch_clustering_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    cluster_groups: Optional[dict] = None,
    feature_combinations: Optional[dict] = None,
    clustering_profiles: Optional[dict] = None,
    reduction: Optional[dict] = None,
    evaluation: Optional[dict] = None,
) -> List[Path]:
    """
    Stage 2: Global clustering on pooled feature CSVs.
    Discovers CSVs written by touch_feature_extraction, merges per group,
    and writes
    ``4_analysed/touch_clusters/<group>/<clusterer>/pooled_touch_summary_clustered.csv``.
    """
    print(f"[Batch Analysis] Running touch clustering for {len(input_items)} item(s)...")
    if not input_items:
        return []
    extraction_dir = input_items[0][1] / '4_analysed' / 'touch_features'
    output_dir = input_items[0][1] / '4_analysed' / 'touch_clusters'
    per_key = run_clustering(
        output_dir=output_dir,
        cluster_groups=cluster_groups,
        feature_combinations=feature_combinations,
        clustering_profiles=clustering_profiles,
        force=force_processing,
        extraction_dir=extraction_dir,
        reduction=reduction,
        evaluation=evaluation,
    )
    return [path for paths in per_key.values() for path in paths]


@flow(name="touch_comparing")
def touch_comparing_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    cluster_groups: list = None,
    cluster_group_defs: dict = None,
    feature_combinations: dict = None,
    clustering_profiles: dict = None,
    comparing_profiles: dict = None,
    min_instances_per_sensor: int = 5,
    min_sensor_types: int = 2,
) -> List[Path]:
    """
    Stage 3: Statistical comparison of sensor measurements across strata.
    Discovers clustered CSVs written by touch_clustering and writes
    per-strategy result JSONs and a dispersion-weighted synthesis report.
    """
    print(f"[Batch Analysis] Running touch comparing for {len(input_items)} item(s)...")
    if not input_items:
        return []
    clustering_dir = input_items[0][1] / '4_analysed' / 'touch_clusters'
    output_dir = input_items[0][1] / '4_analysed' / 'touch_comparisons'
    return run_comparing(
        output_dir=output_dir,
        cluster_groups=cluster_groups,
        cluster_group_defs=cluster_group_defs,
        feature_combinations=feature_combinations,
        clustering_profiles=clustering_profiles,
        comparing_profiles=comparing_profiles,
        min_instances_per_sensor=min_instances_per_sensor,
        min_sensor_types=min_sensor_types,
        force=force_processing,
        clustering_dir=clustering_dir,
    )


@flow(name="analyse_ap_efficacy")
def analyse_ap_efficacy_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False
) -> List[Path]:
    """
    STEP 3: Matrix Generation (Efficacy).
    Generates a matrix of AP efficacy into '4_analysed/ap_efficacy'.
    """
    print(f"[Batch Analysis - AP Efficacy] Generating Efficacy Matrix...")

    unified_files = _collect_unified_files(input_items)

    if unified_files:
        try:
            anchor_db_path = input_items[0][1]

            # Define specific subfolder for this analysis flow
            output_dir = anchor_db_path / "4_analysed" / "ap_efficacy"
            output_dir.mkdir(parents=True, exist_ok=True)

            matrix_output_path = output_dir / "batch_ap_efficacy_matrix.csv"

            logging.info(f"Calling batch AP efficacy matrix generation. Output: {matrix_output_path}")
            # Delegate processing (and checks) to the function
            generate_ap_efficacy_matrix(
                unified_files,
                matrix_output_path,
                show=True,
                force=force_processing
            )
            return [matrix_output_path]
        except Exception as e:
            logging.error(f"Failed to generate AP matrix: {e}")
            return []
    else:
        logging.warning("No unified summary files found. Run 'unified_touch_analysis' first.")
        return []

@flow(name="map_receptive_fields_clustered")
def map_receptive_fields_clustered_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    cluster_groups: list = None,
    cluster_group_defs: dict = None,
    feature_combinations: dict = None,
    clustering_profiles: dict = None,
    projection_method: str = None,
    disjoint_mask_distance_mm: float = 8.0,
) -> List[Path]:
    """
    Cluster-based RF mapping: spike-count heatmaps per cluster from touch_clustering output.
    Reads pooled_touch_summary_clustered.csv, forward-fills contact_points (30Hz->1kHz),
    counts spikes per (x,y,z) point, and renders 3D forearm heatmap PNGs.
    Output: ``4_analysed/receptive_field_maps_clustered/<group>/<clusterer>/``
    """
    print(f"[Batch Analysis] Running cluster-based RF mapping for {len(input_items)} item(s)...")
    if not input_items:
        return []

    database_path = input_items[0][1]
    clustering_dir = database_path / '4_analysed' / 'touch_clusters'
    output_dir = database_path / '4_analysed' / 'receptive_field_maps_clustered'

    result = run_cluster_rf_mapping(
        clustering_dir=clustering_dir,
        input_items=input_items,
        output_dir=output_dir,
        cluster_groups=cluster_groups,
        cluster_group_defs=cluster_group_defs,
        feature_combinations=feature_combinations,
        clustering_profiles=clustering_profiles,
        force=force_processing,
        projection_method=projection_method,
        disjoint_mask_distance_mm=disjoint_mask_distance_mm,
    )

    return result


@flow(name="extract_receptive_fields_clustered")
def extract_receptive_fields_clustered_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    cluster_groups: list = None,
    cluster_group_defs: dict = None,
    feature_combinations: dict = None,
    clustering_profiles: dict = None,
) -> List[Path]:
    """
    Extraction-only step: load aggregated CSVs, count spikes per contact point,
    and write intermediate artifacts to disk.
    Output: ``4_analysed/receptive_field_maps_clustered/<group>/<clusterer>/``
    """
    print(f"[Batch Analysis] Running RF extraction for {len(input_items)} item(s)...")
    if not input_items:
        return []

    database_path = input_items[0][1]
    clustering_dir = database_path / '4_analysed' / 'touch_clusters'
    output_dir = database_path / '4_analysed' / 'receptive_field_maps_clustered'

    return run_cluster_rf_extraction(
        clustering_dir=clustering_dir,
        input_items=input_items,
        output_dir=output_dir,
        cluster_groups=cluster_groups,
        cluster_group_defs=cluster_group_defs,
        feature_combinations=feature_combinations,
        clustering_profiles=clustering_profiles,
        force=force_processing,
    )


@flow(name="compute_receptive_field_metrics")
def compute_receptive_field_metrics_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    cluster_groups: list = None,
    cluster_group_defs: dict = None,
    feature_combinations: dict = None,
    clustering_profiles: dict = None,
    projection_method: str = None,
    slim_uv_cache_dir: Optional[Path] = None,
) -> None:
    """
    Metrics-only step: load extraction artifacts and compute RF metrics
    (centroid, hull area, Gaussian fit) per cluster, writing rf_metrics.json
    and rf_metrics_summary.csv.
    Output: ``4_analysed/receptive_field_maps_clustered/<group>/<clusterer>/``
    """
    print(f"[Batch Analysis] Running RF metrics computation for {len(input_items)} item(s)...")
    if not input_items:
        return

    database_path = input_items[0][1]
    output_dir = database_path / '4_analysed' / 'receptive_field_maps_clustered'
    slim_uv_cache_dir = (
        database_path / '4_analysed' / 'forearm_slim_uv'
        if projection_method == 'slim' else None
    )

    run_cluster_rf_metrics_computation(
        output_dir=output_dir,
        cluster_groups=cluster_groups,
        cluster_group_defs=cluster_group_defs,
        feature_combinations=feature_combinations,
        clustering_profiles=clustering_profiles,
        projection_method=projection_method,
        force=force_processing,
        slim_uv_cache_dir=slim_uv_cache_dir,
    )


@flow(name="visualize_receptive_fields_clustered")
def visualize_receptive_fields_clustered_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    cluster_groups: list = None,
    cluster_group_defs: dict = None,
    feature_combinations: dict = None,
    clustering_profiles: dict = None,
    projection_method: str = None,
    disjoint_mask_distance_mm: float = 8.0,
    gallery_viewer: bool = False,
) -> List[Path]:
    """
    Visualization-only step: load extraction artifacts, compute RF metrics,
    and render per-session heatmap PNGs.
    Output: ``4_analysed/receptive_field_maps_clustered/<group>/<clusterer>/``
    """
    print(f"[Batch Analysis] Running RF visualization for {len(input_items)} item(s)...")
    if not input_items:
        return []

    database_path = input_items[0][1]
    output_dir = database_path / '4_analysed' / 'receptive_field_maps_clustered'
    slim_uv_cache_dir = (
        database_path / '4_analysed' / 'forearm_slim_uv'
        if projection_method == 'slim' else None
    )

    result = run_cluster_rf_visualization(
        output_dir=output_dir,
        cluster_groups=cluster_groups,
        cluster_group_defs=cluster_group_defs,
        feature_combinations=feature_combinations,
        clustering_profiles=clustering_profiles,
        projection_method=projection_method,
        disjoint_mask_distance_mm=disjoint_mask_distance_mm,
        force=force_processing,
        gallery_viewer=gallery_viewer,
        input_items=input_items,
        slim_uv_cache_dir=slim_uv_cache_dir,
    )

    return result


@flow(name="set_rf_camera_settings")
def set_rf_camera_settings_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
) -> None:
    """
    Launch the RF Camera Settings GUI for the given sessions.
    Reads series-augmented CSVs produced by ``touch_series_transforms``.
    Allows the researcher to interactively set the camera orientation per
    session and save it. Settings are consumed by all downstream RF rendering
    and projection tasks.

    Skipped when all sessions already have saved camera settings and
    ``force_processing`` is False.
    """
    if not input_items:
        return

    database_path = input_items[0][1]
    camera_settings_dir = database_path / '4_analysed' / 'rf_camera_settings'
    session_ids = [session_id_from_path(csv_path) for csv_path, _ in input_items]

    if not force_processing:
        existing = load_rf_camera_settings(camera_settings_dir)
        missing = [sid for sid in session_ids if sid not in existing]
        if not missing:
            print(
                f"[RF Camera Settings] All {len(session_ids)} session(s) already have "
                "saved camera settings — skipping. Set force_processing=true to re-open."
            )
            return
        print(
            f"[RF Camera Settings] {len(missing)}/{len(session_ids)} session(s) missing "
            "camera settings — launching viewer."
        )
    else:
        print(f"[RF Camera Settings] Launching viewer for {len(session_ids)} session(s) (force).")

    launch_rf_camera_settings_viewer(input_items)


# --- Dispatch ---

def main():
    freeze_support()
    parser = argparse.ArgumentParser(
        description=(
            "Analysis workflow — processing tasks only. "
            "Runs tasks defined in the given DAG config. "
            "Interactive GUI / viewer tasks are excluded; use "
            "analysis_workflow_viewers.py for those."
        )
    )
    parser.add_argument(
        "--dag-config",
        type=Path,
        default=Path("configs/analyse_workflow_processing_dag.yaml"),
        help="Path to the DAG config YAML (default: configs/analyse_workflow_processing_dag.yaml)",
    )
    args = parser.parse_args()

    dag_config_path: Path = args.dag_config
    project_data_root = path_tools.get_project_data_root()
    configs_dir = Path("configs")
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_file_path = reports_dir / "analysis_processing_status.xlsx"

    if not dag_config_path.exists():
        raise FileNotFoundError(
            f"DAG config not found at {dag_config_path}"
        )

    dag_handler = DagConfigHandler(dag_config_path)
    entries = dag_handler.get_parameter('kinect_configs')
    block_files = resolve_session_configs(entries, configs_dir / "kinect_configs")

    session_map = collect_unique_session_dirs(block_files, project_data_root)

    if not session_map:
        logging.warning("No valid session directories found. Exiting.")
        return

    items_to_process = discover_input_items(session_map)

    if not items_to_process:
        logging.warning("No input files found. Exiting.")
        return

    pipeline_stages = _build_pipeline_stages(dag_handler, items_to_process)
    task_names = [s["name"] for s in pipeline_stages]
    monitor = PipelineMonitor(report_path=report_file_path, stages=task_names, data_queue=Queue())

    logging.info(f"Starting analysis for {len(items_to_process)} collected items.")
    run_pipeline_stages(pipeline_stages, dag_handler, monitor, items_to_process, "batch_run_processing")
    logging.info("Batch analysis finished.")


def _build_pipeline_stages(dag_handler: DagConfigHandler, items_to_process) -> list:
    """Build the ordered pipeline_stages list for the processing slice."""

    def _task_enabled(name: str) -> bool:
        return bool(
            name in dag_handler.tasks
            and dag_handler.tasks[name].get("enabled", True)
        )

    database_path = items_to_process[0][1] if items_to_process else None

    def _preparation_dir() -> Optional[Path]:
        if database_path is not None and _task_enabled("touch_preparation"):
            return database_path / '4_analysed' / 'preparation'
        return None

    def _series_dir() -> Optional[Path]:
        if database_path is not None and _task_enabled("touch_series_transforms"):
            return database_path / '4_analysed' / 'series_transforms'
        return None

    return [
        {
            "name": "summarize_session_blocks",
            "func": summarize_session_blocks_flow,
            "params": lambda: {},
        },
        {
            "name": "touch_preparation",
            "func": touch_preparation_flow,
            "params": lambda: {
                "preparation": dag_handler.get_task_options("touch_preparation").get("preparation"),
            },
        },
        {
            "name": "map_single_touch_rf",
            "func": map_single_touch_rf_flow,
            "params": lambda: {
                "neuron_mode": dag_handler.get_task_options("map_single_touch_rf").get("neuron_mode", "iff"),
                "preparation_dir": _preparation_dir(),
            },
        },
        {
            "name": "touch_series_transforms",
            "func": touch_series_transforms_flow,
            "params": lambda: {
                "transforms": dag_handler.get_task_options("touch_series_transforms").get("transforms"),
                "preparation_dir": _preparation_dir(),
            },
        },
        {
            "name": "set_rf_camera_settings",
            "func": set_rf_camera_settings_flow,
            "params": lambda: {},
        },
        {
            "name": "map_receptive_fields_simple",
            "func": map_receptive_fields_simple_flow,
            "params": lambda: {
                "show_interactive": dag_handler.get_task_options("map_receptive_fields_simple").get("show_interactive", False),
                "projection_method": dag_handler.get_task_options("map_receptive_fields_simple").get("projection_method"),
                "save_diagnostics": bool(dag_handler.get_task_options("map_receptive_fields_simple").get("save_diagnostics", False)),
            },
        },
        {
            "name": "configure_forearm_slim_uv",
            "func": configure_forearm_slim_uv_flow,
            "params": lambda: {
                "interactive": bool(dag_handler.get_task_options("configure_forearm_slim_uv").get("interactive", True)),
                "mesh_method": str(dag_handler.get_task_options("configure_forearm_slim_uv").get("mesh_method", "bpa")),
                "max_edge_mm": float(dag_handler.get_task_options("configure_forearm_slim_uv").get("max_edge_mm", 0.0)),
                "n_iter": int(dag_handler.get_task_options("configure_forearm_slim_uv").get("n_iter", 40)),
                "save_diagnostics": bool(dag_handler.get_task_options("configure_forearm_slim_uv").get("save_diagnostics", True)),
                "clean_steps": dag_handler.get_task_options("configure_forearm_slim_uv").get("clean_steps"),
            },
        },
        {
            "name": "precompute_forearm_slim_uv",
            "func": precompute_forearm_slim_uv_flow,
            "params": lambda: {
                "n_iter": int(dag_handler.get_task_options("precompute_forearm_slim_uv").get("n_iter", 40)),
                "save_diagnostics": bool(dag_handler.get_task_options("precompute_forearm_slim_uv").get("save_diagnostics", False)),
                "interactive": bool(dag_handler.get_task_options("precompute_forearm_slim_uv").get("interactive", False)),
                "clean_steps": dag_handler.get_task_options("precompute_forearm_slim_uv").get("clean_steps"),
                "mesh_method": str(dag_handler.get_task_options("precompute_forearm_slim_uv").get("mesh_method", "bpa")),
                **(
                    {"max_edge_mm": float(dag_handler.get_task_options("precompute_forearm_slim_uv")["max_edge_mm"])}
                    if dag_handler.get_task_options("precompute_forearm_slim_uv").get("max_edge_mm") is not None
                    else {}
                ),
            },
        },
        {
            "name": "extract_population_rf_response_field_boundaries",
            "func": extract_population_rf_response_field_boundaries_flow,
            "params": lambda: {
                "neuron_mode": dag_handler.get_task_options("extract_population_rf_response_field_boundaries").get("neuron_mode", "iff"),
                "min_overlap_pct": float(dag_handler.get_task_options("extract_population_rf_response_field_boundaries").get("min_overlap_pct", 25.0)),
                "heatmap_space": dag_handler.get_task_options("extract_population_rf_response_field_boundaries").get("heatmap_space", "linear"),
                "cmap": dag_handler.get_task_options("extract_population_rf_response_field_boundaries").get("cmap", "jet"),
                **(
                    {"median_filter_size": int(dag_handler.get_task_options("extract_population_rf_response_field_boundaries")["median_filter_size"])}
                    if dag_handler.get_task_options("extract_population_rf_response_field_boundaries").get("median_filter_size") is not None
                    else {}
                ),
                **(
                    {"inflection_sigma": float(dag_handler.get_task_options("extract_population_rf_response_field_boundaries")["inflection_sigma"])}
                    if dag_handler.get_task_options("extract_population_rf_response_field_boundaries").get("inflection_sigma") is not None
                    else {}
                ),
            },
        },
        {
            "name": "compare_session_rf_boundaries",
            "func": compare_session_rf_boundaries_flow,
            "params": lambda: {},
        },
        {
            "name": "compare_rf_center_proximal_distal",
            "func": compare_rf_center_proximal_distal_flow,
            "params": lambda: {
                "heatmap_space": dag_handler.get_task_options("compare_rf_center_proximal_distal").get("heatmap_space", "linear"),
                "cmap": dag_handler.get_task_options("compare_rf_center_proximal_distal").get("cmap", "jet"),
            },
        },
        {
            "name": "touch_feature_extraction",
            "func": touch_feature_extraction_flow,
            "params": lambda: {
                "features": dag_handler.get_task_options("touch_feature_extraction").get("features"),
                "series_dir": _series_dir(),
                "preparation_dir": _preparation_dir(),
            },
        },
        {
            "name": "render_touch_feature_radar",
            "func": render_touch_feature_radar_flow,
            "params": lambda: {
                "radar_groups": dag_handler.get_task_options("render_touch_feature_radar").get("radar_groups"),
            },
        },
        {
            "name": "map_population_rf_grid",
            "func": map_population_rf_grid_flow,
            "params": lambda: {
                "grid_groups": dag_handler.get_task_options("map_population_rf_grid").get("grid_groups"),
                "neuron_mode": dag_handler.get_task_options("map_population_rf_grid").get("neuron_mode", "iff"),
                "per_gesture_type": bool(dag_handler.get_task_options("map_population_rf_grid").get("per_gesture_type", True)),
                "vertex_threshold_ratio": float(dag_handler.get_task_options("map_population_rf_grid").get("vertex_threshold_ratio", 0.25)),
                "compute_baseline": bool(dag_handler.get_task_options("map_population_rf_grid").get("compute_baseline", True)),
                **(
                    {"features": dag_handler.get_task_options("map_population_rf_grid")["features"]}
                    if "features" in dag_handler.get_task_options("map_population_rf_grid")
                    else {}
                ),
            },
        },
        {
            "name": "reduce_population_rf_grid",
            "func": reduce_population_rf_grid_flow,
            "params": lambda: {
                "projection_method": dag_handler.get_task_options("reduce_population_rf_grid").get("projection_method", "tangent_plane"),
                "grid_group_defs": dag_handler.get_task_options("map_population_rf_grid").get("grid_groups") or None,
            },
        },
        {
            "name": "visualize_population_rf_grid_metrics",
            "func": visualize_population_rf_grid_metrics_flow,
            "params": lambda: {
                "extracted_features": list(dag_handler.get_task_options("visualize_population_rf_grid_metrics")["extracted_features"])
                    if "extracted_features" in dag_handler.get_task_options("visualize_population_rf_grid_metrics")
                    else None,
                "grid_group_defs": dag_handler.get_task_options("map_population_rf_grid").get("grid_groups") or None,
            },
        },
        {
            "name": "visualize_session_comparison",
            "func": visualize_session_comparison_flow,
            "params": lambda: {
                "features": dag_handler.get_task_options("visualize_session_comparison").get("features"),
                "neuron_mode": dag_handler.get_task_options("visualize_session_comparison").get("neuron_mode", "iff"),
                "vertex_threshold_ratio": float(dag_handler.get_task_options("visualize_session_comparison").get("vertex_threshold_ratio", 0.25)),
                "metric_name": dag_handler.get_task_options("visualize_session_comparison").get("metric_name", "mean_iff"),
                "projection_method": dag_handler.get_task_options("visualize_session_comparison").get("projection_method", "tangent_plane"),
            },
        },
        {
            "name": "touch_clustering",
            "func": touch_clustering_flow,
            "params": lambda: {
                "cluster_groups": dag_handler.get_task_options("touch_clustering").get("cluster_groups"),
                "feature_combinations": dag_handler.get_task_options("touch_clustering").get("feature_combinations"),
                "clustering_profiles": dag_handler.get_task_options("touch_clustering").get("clustering_profiles"),
                "reduction": dag_handler.get_task_options("touch_clustering").get("reduction"),
                "evaluation": dag_handler.get_task_options("touch_clustering").get("evaluation"),
            },
        },
        {
            "name": "touch_comparing",
            "func": touch_comparing_flow,
            "params": lambda: {
                "cluster_groups": dag_handler.get_task_options("touch_comparing").get("cluster_groups"),
                "cluster_group_defs": dag_handler.get_task_options("touch_clustering").get("cluster_groups") or None,
                "feature_combinations": dag_handler.get_task_options("touch_comparing").get("feature_combinations"),
                "clustering_profiles": dag_handler.get_task_options("touch_comparing").get("clustering_profiles"),
                "comparing_profiles": dag_handler.get_task_options("touch_comparing").get("comparing_profiles"),
                "min_instances_per_sensor": dag_handler.get_task_options("touch_comparing").get("min_instances_per_sensor", 5),
                "min_sensor_types": dag_handler.get_task_options("touch_comparing").get("min_sensor_types", 2),
            },
        },
        {
            "name": "analyse_ap_efficacy",
            "func": analyse_ap_efficacy_flow,
            "params": lambda: {},
        },
        {
            "name": "extract_receptive_fields_clustered",
            "func": extract_receptive_fields_clustered_flow,
            "params": lambda: {
                "cluster_groups": dag_handler.get_task_options("extract_receptive_fields_clustered").get("cluster_groups"),
                "cluster_group_defs": dag_handler.get_task_options("touch_clustering").get("cluster_groups") or None,
                "feature_combinations": dag_handler.get_task_options("extract_receptive_fields_clustered").get("feature_combinations"),
                "clustering_profiles": dag_handler.get_task_options("extract_receptive_fields_clustered").get("clustering_profiles"),
            },
        },
        {
            "name": "compute_receptive_field_metrics",
            "func": compute_receptive_field_metrics_flow,
            "params": lambda: {
                "cluster_groups": dag_handler.get_task_options("compute_receptive_field_metrics").get("cluster_groups"),
                "cluster_group_defs": dag_handler.get_task_options("touch_clustering").get("cluster_groups") or None,
                "feature_combinations": dag_handler.get_task_options("compute_receptive_field_metrics").get("feature_combinations"),
                "clustering_profiles": dag_handler.get_task_options("compute_receptive_field_metrics").get("clustering_profiles"),
                "projection_method": dag_handler.get_task_options("compute_receptive_field_metrics").get("projection_method"),
            },
        },
        {
            "name": "visualize_receptive_fields_clustered",
            "func": visualize_receptive_fields_clustered_flow,
            "params": lambda: {
                "cluster_groups": dag_handler.get_task_options("visualize_receptive_fields_clustered").get("cluster_groups"),
                "cluster_group_defs": dag_handler.get_task_options("touch_clustering").get("cluster_groups") or None,
                "feature_combinations": dag_handler.get_task_options("visualize_receptive_fields_clustered").get("feature_combinations"),
                "clustering_profiles": dag_handler.get_task_options("visualize_receptive_fields_clustered").get("clustering_profiles"),
                "projection_method": dag_handler.get_task_options("visualize_receptive_fields_clustered").get("projection_method"),
                "disjoint_mask_distance_mm": float(dag_handler.get_task_options("visualize_receptive_fields_clustered").get("disjoint_mask_distance_mm", 8.0)),
                "gallery_viewer": bool(dag_handler.get_task_options("visualize_receptive_fields_clustered").get("gallery_viewer", False)),
            },
        },
        # Legacy — kept for back-compat with DAG configs that still reference this key.
        {
            "name": "map_receptive_fields_clustered",
            "func": map_receptive_fields_clustered_flow,
            "params": lambda: {
                "cluster_groups": dag_handler.get_task_options("map_receptive_fields_clustered").get("cluster_groups"),
                "cluster_group_defs": dag_handler.get_task_options("touch_clustering").get("cluster_groups") or None,
                "feature_combinations": dag_handler.get_task_options("map_receptive_fields_clustered").get("feature_combinations"),
                "clustering_profiles": dag_handler.get_task_options("map_receptive_fields_clustered").get("clustering_profiles"),
                "projection_method": dag_handler.get_task_options("map_receptive_fields_clustered").get("projection_method"),
                "disjoint_mask_distance_mm": float(dag_handler.get_task_options("map_receptive_fields_clustered").get("disjoint_mask_distance_mm", 8.0)),
            },
        },
    ]


if __name__ == "__main__":
    main()
