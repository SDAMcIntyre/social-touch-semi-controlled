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

from preprocessing.forearm_extraction import get_transform_schedule

from postprocessing.gui import PostprocessedSceneViewer
from postprocessing.xyz_reference_from_gestures.calibration_pca_engine import CalibrationResult


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_postprocessed_paths(config: KinectConfig) -> Dict[str, Optional[Path]]:
    """Derive all paths required by PostprocessedSceneViewer from a KinectConfig.

    Expected pipeline outputs:
      - contact_projected_csv : session_merged_output_dir/blocks_contact_projected/{merged_csv_name}
      - forearm_pca_ply       : session_merged_output_dir/forearm_pca_calibrated/{session_id}_forearm_pca_calibrated.ply
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
            / f"{config.session_id}_forearm_pca_calibrated.ply"
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

    print(f"[{block_name}] Launching PostprocessedSceneViewer (simple mode)...")
    app = QApplication.instance() or QApplication(sys.argv)

    viewer = PostprocessedSceneViewer(
        postprocessed_csv_path=paths["contact_projected_csv"],
        forearm_ply_path=paths["forearm_pca_ply"],
        recording_name=paths["recording_name"],
        mode="simple",
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
    dag_handler_template = DagConfigHandler(dag_config_path)

    for block_file in block_files:
        print(f"--- Opening block: {block_file.name} ---")
        try:
            config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
            config = KinectConfig(config_data=config_data, database_path=project_data_root)
            dag_handler_instance = dag_handler_template.copy()

            run_single_session_pipeline(config, dag_handler_instance)
            run_single_session_pipeline_advanced(config, dag_handler_instance)
        except Exception as exc:
            print(f"Failed to initialise session {block_file.name}: {exc}")
            continue

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
