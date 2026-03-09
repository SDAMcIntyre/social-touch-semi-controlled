"""
view_merged_neural_kinect.py
----------------------------
Workflow script for the NeuralKinectViewer — the high-performance 3D visualization
tool for merged neural + Kinect recordings.

Paths are resolved entirely from the standard KinectConfig system; no hardcoded
paths are required. Point the DAG config to the desired kinect_configs entry.

Usage:
    python code/scripts/view_merged_neural_kinect.py

DAG config: configs/view_merged_neural_kinect_dag.yaml

Features:
  - GPU-cropped point cloud (CuPy, ±crop_half_size_mm AABB around contact centroid)
  - Background MKV pre-loading (8-frame ring buffer)
  - Per-sticker velocity compass widgets
  - Live Nerve_freq / contact_depth / contact_area time-series panel
  - Camera centered on contact region
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict

from prefect import flow
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QCoreApplication

# ---------------------------------------------------------------------------
# CuPy early-init — MUST happen before any preprocessing import.
#
# Several preprocessing subpackages load compiled C-extensions (Open3D,
# pyk4a, …) whose initialisation mutates NumPy's internal dtype registry.
# If CuPy is imported *after* that mutation, its Cython _dtype init crashes:
#     TypeError: Alias 'bool8' was removed in NumPy 2.0
# Importing CuPy here, while NumPy's state is still pristine, avoids the
# conflict.  The result is cached in sys.modules so later `import cupy`
# calls in any module are free.
# See: docs/bugs/cupy-bool8-import-order.md
# ---------------------------------------------------------------------------
try:
    import cupy  # noqa: F401  — early-init only, used later via _CUPY_AVAILABLE
except Exception:
    pass  # GPU unavailable — viewer falls back to NumPy automatically

import utils.path_tools as path_tools
from utils.pipeline.pipeline_config_manager import DagConfigHandler

from utils.pipeline.session_config_resolver import resolve_session_configs
from primary_processing import (
    KinectConfigFileHandler,
    KinectConfig,
)

from merging.gui.neural_kinect_scene_viewer import NeuralKinectViewer


# ---------------------------------------------------------------------------
# Path resolution (mirrors resolve_filenames from merging_pipeline)
# ---------------------------------------------------------------------------

def resolve_viewer_paths(config: KinectConfig) -> Dict[str, Optional[Path]]:
    """
    Derives all paths required by NeuralKinectViewer from a KinectConfig.

    Naming conventions follow the rest of the pipeline:
      - Sticker XYZ CSV  : video_processed_output_dir/handstickers/<stem>_handstickers_xyz_tracked.csv
      - Hand motion      : video_processed_output_dir/kinematics_analysis/<stem>_handmodel_motion.npz
      - Forearm dir      : session_processed_output_dir/forearm_pointclouds/
      - Forearm metadata : <forearm_dir>/<session_id>_arm_roi_metadata.json
      - Merged CSV       : session_merged_output_dir/sessions/<session_id>_semicontrolled_<block_id>_merged_data.csv
    """
    stem = config.source_video.stem
    forearm_dir = config.session_processed_output_dir / "forearm_pointclouds"

    merged_csv: Optional[Path] = None
    if config.session_merged_output_dir:
        merged_name = f"{config.session_id}_semicontrolled_{config.block_id}_merged_data.csv"
        candidate = config.session_merged_output_dir / "sessions" / merged_name
        merged_csv = candidate if candidate.exists() else None

    return {
        "recording_name": stem,
        "xyz_csv_path": (
            config.video_processed_output_dir / "handstickers"
            / f"{stem}_handstickers_xyz_tracked.csv"
        ),
        "kinect_mkv_path": config.source_video,
        "forearm_pointcloud_dir": forearm_dir,
        "forearm_metadata_path": forearm_dir / f"{config.session_id}_arm_roi_metadata.json",
        # Passed as a plain filename — ForearmCatalog uses it as a lookup key
        "rgb_video_path": Path(f"{stem}.mp4"),
        "hand_motion_path": (
            config.video_processed_output_dir / "kinematics_analysis"
            / f"{stem}_handmodel_motion.npz"
        ),
        "merged_csv_path": merged_csv,
    }


# ---------------------------------------------------------------------------
# Per-block viewer launch
# ---------------------------------------------------------------------------

def run_single_session_pipeline(
    config: KinectConfig,
    dag_handler: DagConfigHandler,
) -> None:
    """Launches NeuralKinectViewer for a single block if the task is enabled."""
    block_name = config.source_video.name
    print(f"[{block_name}] ==> Checking task: view_neural_kinect_scene")

    if not dag_handler.can_run("view_neural_kinect_scene"):
        print(f"[{block_name}] Task disabled — skipping.")
        return

    paths = resolve_viewer_paths(config)

    # Validate the minimum required inputs exist
    for key in ("xyz_csv_path", "kinect_mkv_path", "forearm_metadata_path"):
        p = paths[key]
        if not p.exists():
            print(f"[{block_name}] Missing required file '{key}': {p}  — skipping.")
            return

    options = dag_handler.get_task_options("view_neural_kinect_scene")
    crop_half_size = float(options.get("crop_half_size_mm", 400.0))

    if paths["merged_csv_path"] is None:
        print(f"[{block_name}] No merged CSV found — launching in pure-3D mode.")
    else:
        print(f"[{block_name}] Merged CSV found — neural overlay enabled.")

    print(f"[{block_name}] Launching NeuralKinectViewer...")
    app = QApplication.instance() or QApplication(sys.argv)

    viewer = NeuralKinectViewer(
        xyz_csv_path=paths["xyz_csv_path"],
        kinect_mkv_path=paths["kinect_mkv_path"],
        forearm_pointcloud_dir=paths["forearm_pointcloud_dir"],
        forearm_metadata_path=paths["forearm_metadata_path"],
        rgb_video_path=paths["rgb_video_path"],
        hand_motion_path=paths["hand_motion_path"],
        recording_name=paths["recording_name"],
        merged_csv_path=paths["merged_csv_path"],
        crop_half_size_mm=crop_half_size,
    )
    viewer.show()
    app.exec_()

    # Flush any deferred Qt destruction events (widget teardown, OpenGL context
    # release) that were posted but not yet processed when exec_() returned.
    # Without this, the next iteration's NeuralKinectViewer may start VTK
    # initialisation while the previous render window's context is still live,
    # causing wglMakeCurrent to fail.
    QCoreApplication.processEvents()

    dag_handler.mark_completed("view_neural_kinect_scene")


# ---------------------------------------------------------------------------
# Batch dispatcher (sequential — one viewer at a time, matching visualisation workflow)
# ---------------------------------------------------------------------------

@flow(name="Run Neural-Kinect Viewer Batch Sequentially", log_prints=True)
def run_batch_sequentially(
    block_files: list[Path],
    project_data_root: Path,
    dag_config_path: Path,
) -> None:
    """Runs the viewer for each block config file in block_files."""
    dag_handler_template = DagConfigHandler(dag_config_path)

    for block_file in block_files:
        print(f"--- Opening block: {block_file.name} ---")
        try:
            config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
            config = KinectConfig(config_data=config_data, database_path=project_data_root)
            dag_handler_instance = dag_handler_template.copy()

            run_single_session_pipeline(config, dag_handler_instance)
        except Exception as exc:
            print(f"Failed to initialise session {block_file.name}: {exc}")
            continue

    print("All viewer sessions completed.")


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

    print("Launching Neural-Kinect Viewer batch (sequential).")
    run_batch_sequentially(
        block_files=block_files,
        project_data_root=project_data_root,
        dag_config_path=dag_config_path,
    )
