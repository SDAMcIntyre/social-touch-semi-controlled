"""
postprocess_visualization.py
----------------------------
Workflow script for the PostprocessedSceneViewer — a read-only visualization
tool for PCA-calibrated postprocessed data (contact points on forearm surface,
optional sticker spheres and hand mesh).

Paths are resolved entirely from the standard KinectConfig system.
Point the DAG config to the desired kinect_configs entry.

Usage:
    python code/scripts/postprocess_visualization.py --dag-config configs/postprocess_visualization_dag.yaml

DAG config: configs/postprocess_visualization_dag.yaml

Tasks:
  view_postprocessed_simple   — forearm PLY + contact points + neural panels
  view_postprocessed_advanced — adds sticker spheres + hand mesh (ICP+PCA)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from prefect import flow
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication

# ---------------------------------------------------------------------------
# CuPy early-init — MUST happen before any preprocessing import.
# See: docs/development/knowledge-base/note-cupy-import-order.md
# ---------------------------------------------------------------------------
try:
    import cupy  # noqa: F401  — early-init only
except Exception:
    pass

import utils.path_tools as path_tools
from utils.pipeline.pipeline_config_manager import DagConfigHandler
from utils.pipeline.session_config_resolver import resolve_session_configs
from primary_processing import (
    KinectConfigFileHandler,
    KinectConfig,
)

from preprocessing.forearm_extraction import (
    get_transform_schedule,
    ForearmFrameParametersFileHandler,
    ForearmCatalog,
    get_forearms_with_fallback,
)

from postprocessing.gui import (
    PostprocessedSceneViewer,
    BeforeAfterStepViewer,
    ForearmStageInspector,
    PostprocessingStageViewer,
    StagePaths,
    STAGE_LABELS,
)
from postprocessing.gui.forearm_stage_inspector import resolve_all_session_stage_paths
from postprocessing.xyz_reference_from_gestures.calibration_pca_engine import CalibrationResult


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_postprocessed_paths(config: KinectConfig) -> Dict[str, Optional[Path]]:
    """Derive all paths required by PostprocessedSceneViewer from a KinectConfig.

    Expected pipeline outputs:
      - contact_projected_csv : session_merged_output_dir/blocks_contact_projected/{merged_csv_name}
      - forearm_pca_ply       : session_merged_output_dir/forearm_pca_calibrated/{session_id}_forearm.ply
      - pca_calib_json        : session_merged_output_dir/blocks_pca_calibrated/pca-xyz_transformation-matrices.json
      - hand_motion_path      : video_processed_output_dir/kinematics_analysis/{stem}_handmodel_motion.npz
      - forearm_metadata_path : session_processed_output_dir/forearm_pointclouds/{session_id}_arm_roi_metadata.json
    """
    stem = config.source_video.stem
    merged_name = (
        f"{config.session_id}_semicontrolled_{config.block_id}_merged_data_pca-xyz.csv"
    )

    contact_projected_csv: Optional[Path] = None
    forearm_pca_ply: Optional[Path] = None
    pca_calib_json: Optional[Path] = None
    if config.session_merged_output_dir:
        contact_projected_csv = (
            config.session_merged_output_dir / "blocks_contact_projected" / merged_name
        )
        forearm_pca_ply = (
            config.session_merged_output_dir
            / "forearm_pca_calibrated"
            / f"{config.session_id}_forearm.ply"
        )
        pca_calib_json = (
            config.session_merged_output_dir
            / "blocks_pca_calibrated"
            / "pca-xyz_transformation-matrices.json"
        )

    forearm_dir = config.session_processed_output_dir / "forearm_pointclouds"

    return {
        "recording_name": stem,
        "contact_projected_csv": contact_projected_csv,
        "forearm_pca_ply": forearm_pca_ply,
        "pca_calib_json": pca_calib_json,
        "hand_motion_path": (
            config.video_processed_output_dir
            / "kinematics_analysis"
            / f"{stem}_handmodel_motion.npz"
        ),
        "forearm_metadata_path": forearm_dir / f"{config.session_id}_arm_roi_metadata.json",
        "forearm_pointcloud_dir": forearm_dir,
    }


def _load_per_video_forearm(config: KinectConfig):
    """Return the reference forearm (frame 0) as an Open3D PointCloud, or None.

    Loads forearm metadata for the current block via ForearmCatalog and returns
    the frame-0 reference pointcloud using get_forearms_with_fallback().
    Returns None when the metadata file is missing or loading fails.
    """
    forearm_dir = config.session_processed_output_dir / "forearm_pointclouds"
    metadata_path = forearm_dir / f"{config.session_id}_arm_roi_metadata.json"
    if not metadata_path.exists():
        print(f"  Warning: forearm metadata not found at {metadata_path} — step 1 before-forearm unavailable.")
        return None
    try:
        forearm_params = ForearmFrameParametersFileHandler.load(metadata_path)
        catalog = ForearmCatalog(forearm_params, forearm_dir)
        forearms = get_forearms_with_fallback(catalog, config.source_video.name)
        return forearms.get(0)
    except Exception as exc:
        print(f"  Warning: could not load per-video forearm: {exc}")
        return None


def _load_unified_forearm(config: KinectConfig) -> Optional[Path]:
    """Return the path to the unified registered PLY, or a single-forearm fallback.

    Mirrors the _find_forearm_ply() pattern from export_forearm_pca_calibrated.py.
    Returns None when no PLY is found.
    """
    forearm_dir = config.session_processed_output_dir / "forearm_pointclouds"
    unified = forearm_dir / f"{config.session_id}_unified_registered.ply"
    if unified.exists():
        return unified
    plies = sorted(forearm_dir.glob("*.ply"))
    if plies:
        print(f"  No unified registered PLY found; using fallback: {plies[0].name}")
        return plies[0]
    return None


def resolve_before_after_paths(config: KinectConfig) -> List[Dict]:
    """Return a list of step descriptors for the 4 postprocessing steps.

    Each descriptor is a dict with keys:
        step_label    : str
        before_csv    : Path
        after_csv     : Path
        before_forearm: Optional[Union[Path, Open3D PointCloud]]
        after_forearm : Optional[Union[Path, Open3D PointCloud]]

    Returns an empty list when config.session_merged_output_dir is None.
    """
    if config.session_merged_output_dir is None:
        return []

    base = config.session_merged_output_dir
    session_id = config.session_id
    block_id = config.block_id

    raw_name = f"{session_id}_semicontrolled_{block_id}_merged_data.csv"
    pca_name = f"{session_id}_semicontrolled_{block_id}_merged_data_pca-xyz.csv"
    forearm_ply_name = f"{session_id}_forearm.ply"

    forearm_pca_dir = base / "forearm_pca_calibrated"
    forearm_rf_dir = base / "forearm_rf_centered"

    # Forearms for steps 1 and 2 require loading from the preprocessing outputs.
    per_video_forearm = _load_per_video_forearm(config)   # Open3D PointCloud or None
    unified_forearm_ply = _load_unified_forearm(config)   # Path or None

    return [
        {
            "step_label": "Step 1: ICP Registration",
            "before_csv": base / "blocks_merged" / raw_name,
            "after_csv": base / "blocks_registered" / raw_name,
            "before_forearm": per_video_forearm,
            "after_forearm": unified_forearm_ply,
        },
        {
            "step_label": "Step 2: PCA Calibration",
            "before_csv": base / "blocks_registered" / raw_name,
            "after_csv": base / "blocks_pca_calibrated" / pca_name,
            "before_forearm": unified_forearm_ply,
            "after_forearm": forearm_pca_dir / forearm_ply_name,
        },
        {
            "step_label": "Step 3: Forearm Projection",
            "before_csv": base / "blocks_pca_calibrated" / pca_name,
            "after_csv": base / "blocks_contact_projected" / pca_name,
            "before_forearm": forearm_pca_dir / forearm_ply_name,
            "after_forearm": forearm_pca_dir / forearm_ply_name,
        },
        {
            "step_label": "Step 4: RF Centering",
            "before_csv": base / "blocks_contact_projected" / pca_name,
            "after_csv": base / "blocks_rf_centered" / pca_name,
            "before_forearm": forearm_pca_dir / forearm_ply_name,
            "after_forearm": forearm_rf_dir / forearm_ply_name,
        },
    ]


def resolve_stage_paths(config: KinectConfig) -> List[StagePaths]:
    """Return one StagePaths instance per postprocessing stage.

    Always returns a list of exactly 5 entries (one per label in STAGE_LABELS).
    CSV paths are set to None when config.session_merged_output_dir is None;
    otherwise the path is set regardless of whether the file exists yet —
    the viewer handles missing files gracefully.
    """
    base = config.session_merged_output_dir
    session_id = config.session_id
    block_id = config.block_id

    raw_name = f"{session_id}_semicontrolled_{block_id}_merged_data.csv"
    pca_name = f"{session_id}_semicontrolled_{block_id}_merged_data_pca-xyz.csv"

    per_video_forearm = _load_per_video_forearm(config)
    unified_forearm_ply = _load_unified_forearm(config)

    if base is not None:
        forearm_pca_ply = base / "forearm_pca_calibrated" / f"{session_id}_forearm.ply"
        forearm_rf_ply = base / "forearm_rf_centered" / f"{session_id}_forearm.ply"
    else:
        forearm_pca_ply = None
        forearm_rf_ply = None

    return [
        StagePaths(
            stage_label=STAGE_LABELS[0],
            csv_path=base / "blocks_merged" / raw_name if base is not None else None,
            forearm=per_video_forearm,
            coordinate_frame="camera",
        ),
        StagePaths(
            stage_label=STAGE_LABELS[1],
            csv_path=base / "blocks_registered" / raw_name if base is not None else None,
            forearm=unified_forearm_ply,
            coordinate_frame="camera",
        ),
        StagePaths(
            stage_label=STAGE_LABELS[2],
            csv_path=base / "blocks_pca_calibrated" / pca_name if base is not None else None,
            forearm=forearm_pca_ply,
            coordinate_frame="pca",
        ),
        StagePaths(
            stage_label=STAGE_LABELS[3],
            csv_path=base / "blocks_contact_projected" / pca_name if base is not None else None,
            forearm=forearm_pca_ply,
            coordinate_frame="pca",
        ),
        StagePaths(
            stage_label=STAGE_LABELS[4],
            csv_path=base / "blocks_rf_centered" / pca_name if base is not None else None,
            forearm=forearm_rf_ply,
            coordinate_frame="pca",
        ),
    ]


def _load_icp_schedule(
    config: KinectConfig,
    forearm_pointcloud_dir: Path,
) -> Optional[List[Tuple[int, np.ndarray]]]:
    """Load the ICP transform schedule for the current video block.

    Loads the registration transforms JSON file directly and passes the raw
    dict data to get_transform_schedule(), matching the pattern used by
    apply_icp_registration.py.

    Returns None if registration transforms are unavailable.
    """
    transforms_path = forearm_pointcloud_dir / f"{config.session_id}_registration_transforms.json"
    if not transforms_path.exists():
        return None
    try:
        with open(transforms_path) as fh:
            raw = json.load(fh)
    except Exception as exc:
        print(f"  Warning: could not load registration transforms: {exc}")
        return None

    schedule = get_transform_schedule(
        raw["transforms"],
        config.source_video.stem,
        max_frame=1_000_000,
    )
    return schedule if schedule else None


def _load_pca_calib(pca_calib_json: Optional[Path]) -> Optional[CalibrationResult]:
    """Load PCA calibration matrices from JSON.  Returns None on error."""
    if pca_calib_json is None or not pca_calib_json.exists():
        return None
    try:
        with open(pca_calib_json) as fh:
            data = json.load(fh)
        return CalibrationResult.from_dict(data)
    except Exception as exc:
        print(f"  Warning: could not load PCA calibration JSON: {exc}")
        return None


# ---------------------------------------------------------------------------
# Per-block viewer launches
# ---------------------------------------------------------------------------

def run_single_session_pipeline(
    config: KinectConfig,
    dag_handler: DagConfigHandler,
) -> None:
    """Launch PostprocessedSceneViewer in simple mode for one block."""
    task_name = "view_postprocessed_simple"
    block_name = config.source_video.name
    print(f"[{block_name}] ==> Checking task: {task_name}")

    if not dag_handler.can_run(task_name):
        print(f"[{block_name}] Task disabled — skipping.")
        return

    paths = resolve_postprocessed_paths(config)

    for key in ("contact_projected_csv", "forearm_pca_ply"):
        p = paths[key]
        if p is None or not p.exists():
            print(f"[{block_name}] Missing required file '{key}': {p}  — skipping.")
            return

    pca_calib = _load_pca_calib(paths["pca_calib_json"])
    if pca_calib is not None:
        print(f"[{block_name}] PCA calibration loaded.")
    else:
        print(f"[{block_name}] No PCA calibration — using default camera orientation.")

    print(f"[{block_name}] Launching PostprocessedSceneViewer (simple mode)...")
    app = QApplication.instance() or QApplication(sys.argv)

    viewer = PostprocessedSceneViewer(
        postprocessed_csv_path=paths["contact_projected_csv"],
        forearm_ply_path=paths["forearm_pca_ply"],
        recording_name=paths["recording_name"],
        mode="simple",
        pca_calib=pca_calib,
    )
    viewer.show()
    app.exec_()
    QCoreApplication.processEvents()

    dag_handler.mark_completed(task_name)


def run_single_session_pipeline_advanced(
    config: KinectConfig,
    dag_handler: DagConfigHandler,
) -> None:
    """Launch PostprocessedSceneViewer in advanced mode for one block.

    Skips (with a warning) when:
    - The task is disabled in the DAG config.
    - Required postprocessed CSV or forearm PLY are missing.
    Advanced features degrade gracefully:
    - Missing hand motion NPZ → hand mesh disabled.
    - Missing registration transforms → hand mesh rendered without ICP.
    - Missing PCA calib JSON → hand mesh rendered without PCA transform.
    """
    task_name = "view_postprocessed_advanced"
    block_name = config.source_video.name
    print(f"[{block_name}] ==> Checking task: {task_name}")

    if not dag_handler.can_run(task_name):
        print(f"[{block_name}] Task disabled — skipping.")
        return

    paths = resolve_postprocessed_paths(config)

    for key in ("contact_projected_csv", "forearm_pca_ply"):
        p = paths[key]
        if p is None or not p.exists():
            print(f"[{block_name}] Missing required file '{key}': {p}  — skipping.")
            return

    # Load optional advanced artifacts (graceful degradation on failure)
    icp_schedule = _load_icp_schedule(
        config,
        paths["forearm_pointcloud_dir"],
    )
    if icp_schedule:
        print(f"[{block_name}] ICP schedule loaded ({len(icp_schedule)} entries).")
    else:
        print(f"[{block_name}] No ICP schedule — hand mesh will not be ICP-transformed.")

    pca_calib = _load_pca_calib(paths["pca_calib_json"])
    if pca_calib is not None:
        print(f"[{block_name}] PCA calibration loaded.")
    else:
        print(f"[{block_name}] No PCA calibration — hand mesh will not be PCA-transformed.")

    hand_motion_path: Optional[Path] = paths["hand_motion_path"]
    if hand_motion_path is None or not hand_motion_path.exists():
        print(f"[{block_name}] No hand motion NPZ — hand mesh disabled.")
        hand_motion_path = None
    else:
        print(f"[{block_name}] Hand motion NPZ found.")

    print(f"[{block_name}] Launching PostprocessedSceneViewer (advanced mode)...")
    app = QApplication.instance() or QApplication(sys.argv)

    viewer = PostprocessedSceneViewer(
        postprocessed_csv_path=paths["contact_projected_csv"],
        forearm_ply_path=paths["forearm_pca_ply"],
        recording_name=paths["recording_name"],
        mode="advanced",
        hand_motion_path=hand_motion_path,
        icp_schedule=icp_schedule,
        pca_calib=pca_calib,
    )
    viewer.show()
    app.exec_()
    QCoreApplication.processEvents()

    dag_handler.mark_completed(task_name)


def run_single_session_pipeline_before_after(
    config: KinectConfig,
    dag_handler: DagConfigHandler,
) -> None:
    """Launch BeforeAfterStepViewer for each postprocessing step of one block."""
    task_name = "view_before_after_steps"
    if not dag_handler.can_run(task_name):
        return

    steps = resolve_before_after_paths(config)
    block_name = config.source_video.name

    for step in steps:
        before_csv: Path = step["before_csv"]
        after_csv: Path = step["after_csv"]
        step_label: str = step["step_label"]

        if not before_csv.exists():
            print(f"[{block_name}] {step_label}: missing before CSV {before_csv} — skipping.")
            continue
        if not after_csv.exists():
            print(f"[{block_name}] {step_label}: missing after CSV {after_csv} — skipping.")
            continue

        print(f"[{block_name}] Launching BeforeAfterStepViewer for {step_label}...")
        app = QApplication.instance() or QApplication(sys.argv)

        viewer = BeforeAfterStepViewer(
            before_csv_path=before_csv,
            after_csv_path=after_csv,
            before_forearm=step["before_forearm"],
            after_forearm=step["after_forearm"],
            step_label=step_label,
            recording_name=config.source_video.stem,
        )
        viewer.show()
        app.exec_()
        # Pump the event loop twice: first pass processes deferred deletions
        # (VTK render window Finalize calls from closeEvent), second pass
        # flushes any events those deletions may have posted.  This ensures
        # the OpenGL context is fully released before the next viewer opens.
        QCoreApplication.processEvents()
        QCoreApplication.processEvents()

    dag_handler.mark_completed(task_name)


def run_single_session_pipeline_stage_viewer(
    config: KinectConfig,
    dag_handler: DagConfigHandler,
) -> None:
    """Launch PostprocessingStageViewer for one block."""
    task_name = "view_postprocessing_stages"
    block_name = config.source_video.name
    print(f"[{block_name}] ==> Checking task: {task_name}")
    if not dag_handler.can_run(task_name):
        print(f"[{block_name}] Task disabled — skipping.")
        return
    stage_paths = resolve_stage_paths(config)
    available = [sp for sp in stage_paths if sp.csv_path is not None and sp.csv_path.exists()]
    if not available:
        print(f"[{block_name}] No stage CSV files found — skipping.")
        return
    recording_name = config.source_video.stem
    print(f"[{block_name}] Launching PostprocessingStageViewer ({len(available)}/5 stages with data)...")
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = PostprocessingStageViewer(stage_paths, recording_name=recording_name)
    viewer.show()
    app.exec_()
    QCoreApplication.processEvents()
    dag_handler.mark_completed(task_name)


# ---------------------------------------------------------------------------
# Session-level viewers
# ---------------------------------------------------------------------------


def run_forearm_stage_inspector(
    session_map: Dict[str, List[KinectConfig]],
    dag_handler: DagConfigHandler,
) -> None:
    """Launch ForearmStageInspector for all sessions in *session_map*.

    Checks can_run("view_forearm_stage_inspector") before opening the viewer.
    Resolves all PLY paths from the session map and opens a single window
    that lets the user navigate sessions and stages via dropdowns.
    """
    task_name = "view_forearm_stage_inspector"
    if not dag_handler.can_run(task_name):
        print(f"Task '{task_name}' disabled — skipping.")
        return

    stage_index = resolve_all_session_stage_paths(session_map)
    print(
        f"Launching ForearmStageInspector for {len(stage_index)} session(s)..."
    )
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = ForearmStageInspector(stage_index)
    viewer.show()
    app.exec_()
    QCoreApplication.processEvents()
    dag_handler.mark_completed(task_name)


# ---------------------------------------------------------------------------
# Batch dispatcher
# ---------------------------------------------------------------------------

@flow(name="Run Postprocessed Viewer Batch Sequentially", log_prints=True)
def run_batch_sequentially(
    block_files: list[Path],
    project_data_root: Path,
    dag_config_path: Path,
) -> None:
    """Run the viewer for each block config file sequentially."""
    from collections import defaultdict

    dag_handler_template = DagConfigHandler(dag_config_path)

    # First pass: load all configs and group by session_id for session-level tasks.
    session_map: Dict[str, List[KinectConfig]] = defaultdict(list)
    loaded_configs: List[tuple] = []
    for block_file in block_files:
        try:
            config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
            config = KinectConfig(config_data=config_data, database_path=project_data_root)
            session_map[config.session_id].append(config)
            loaded_configs.append((block_file, config))
        except Exception as exc:
            print(f"Failed to initialise session {block_file.name}: {exc}")

    # Session-level stage inspector (one window for all sessions)
    run_forearm_stage_inspector(dict(session_map), dag_handler_template)

    # Per-block viewers
    for block_file, config in loaded_configs:
        print(f"--- Opening block: {block_file.name} ---")
        dag_handler_instance = dag_handler_template.copy()
        run_single_session_pipeline(config, dag_handler_instance)
        run_single_session_pipeline_advanced(config, dag_handler_instance)
        run_single_session_pipeline_before_after(config, dag_handler_instance)
        run_single_session_pipeline_stage_viewer(config, dag_handler_instance)

    print("All postprocessed viewer sessions completed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--dag-config", type=Path, required=True)
    _args = _parser.parse_args()

    project_data_root = path_tools.get_project_data_root()
    configs_dir = Path("configs")
    dag_config_path = _args.dag_config

    try:
        main_dag_handler = DagConfigHandler(dag_config_path)
        entries = main_dag_handler.get_parameter("kinect_configs")
        block_files = resolve_session_configs(entries, configs_dir / "kinect_configs")
    except FileNotFoundError:
        print(f"DAG config not found at '{dag_config_path}'.")
        sys.exit(1)

    print("Launching Postprocessed Viewer batch (sequential).")
    run_batch_sequentially(
        block_files=block_files,
        project_data_root=project_data_root,
        dag_config_path=dag_config_path,
    )
