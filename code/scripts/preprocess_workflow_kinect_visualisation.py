import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# CuPy import guard — must precede any preprocessing imports (NumPy 2.0 removed the
# bool8 alias CuPy needs; C-extensions imported first corrupt the dtype state).
try:
    import cupy  # noqa: F401
except Exception:
    pass

from prefect import flow
import utils.path_tools as path_tools
from utils.pipeline.pipeline_config_manager import DagConfigHandler

from primary_processing import (
    KinectConfigFileHandler,
    KinectConfig,
)

from utils.pipeline.session_config_resolver import resolve_session_configs

from _3_preprocessing._1_sticker_tracking import (
    view_ellipse_tracking,
    view_ellipse_tracking_adjusted,
    view_summary_stickers_on_rgb_data,
    view_xyz_stickers_on_depth_data,
    view_xyz_depth_aggregation,
)

from _3_preprocessing._3_forearm_extraction import (
    is_forearm_valid
)

from _3_preprocessing._4_somatosensory_quantification import (
    view_somatosensory_3d_scene,
    build_somatosensory_scene_factory,
)

from _3_preprocessing._2_hand_tracking.view_hand_mesh_comparison import (
    build_hand_mesh_comparison_factory,
)

from preprocessing.motion_analysis.hand_tracking.handmesh_overlay_renderer import (
    resolve_hand_motion_npz_path,
)

import logging

logger = logging.getLogger(__name__)


# --- Sub-Flows (Visualization Tasks) ---

def view_ellipse_tracking_flow(
    rgb_video_path: Path,
    sticker_dir: Path
) -> Path:
    """Visualize the 3D sticker data on the depth point cloud."""
    name_baseline = rgb_video_path.stem + "_handstickers"
    print(f"[{rgb_video_path.name}] Viewing ellipse tracking...")

    metadata_colorspace_path = sticker_dir / (name_baseline + "_colorspace_metadata.json")
    binary_video_base_path = sticker_dir / (name_baseline + "_corrmap.mp4")
    fit_ellipses_path = sticker_dir / (name_baseline + "_ellipses.csv")

    view_ellipse_tracking(binary_video_base_path, metadata_colorspace_path, fit_ellipses_path)
    return True


def view_ellipse_adjusted_tracking_flow(
    rgb_video_path: Path,
    sticker_dir: Path
) -> Path:
    """Visualize the adjusted ellipse data."""
    name_baseline = rgb_video_path.stem + "_handstickers"
    print(f"[{rgb_video_path.name}] Viewing adjusted ellipse tracking...")
    fit_ellipses_path = sticker_dir / (name_baseline + "_ellipses_center_adjusted.csv")
    view_ellipse_tracking_adjusted(rgb_video_path, fit_ellipses_path)
    return True


def view_consolidated_2d_tracking_data(
    rgb_video_path: Path,
    sticker_dir: Path
) -> Path:
    """Visualize the 2D sticker summaries."""
    print(f"[{rgb_video_path.name}] Viewing summary 2d sticker data...")
    xy_csv_path = sticker_dir / (rgb_video_path.stem + "_handstickers_summary_2d_coordinates.csv")

    view_summary_stickers_on_rgb_data(
        xy_csv_path,
        rgb_video_path
    )
    return True


def view_xyz_stickers(
    source_video: Path,
    sticker_dir: Path,
    rgb_video_path: Path,
    session_common_dir: Path,
    session_id: str
) -> Path:
    """Visualize the 3D sticker data on the depth point cloud."""
    print(f"[{rgb_video_path.name}] Validating xyz stickers extraction...")
    forearm_pointcloud_dir = session_common_dir / "forearm_pointclouds"
    metadata_filaname = session_id + "_arm_roi_metadata.json"
    forearm_metadata_path = forearm_pointcloud_dir / metadata_filaname

    if not is_forearm_valid(forearm_pointcloud_dir):
        print("Forearm data invalid.")
        return False

    xyz_csv_path = sticker_dir / (rgb_video_path.stem + '_handstickers_xyz_tracked.csv')

    view_xyz_stickers_on_depth_data(
        xyz_csv_path,
        source_video,
        forearm_pointcloud_dir,
        forearm_metadata_path,
        rgb_video_path.name
    )
    return True


def view_xyz_depth_aggregation_flow(
    source_video: Path,
    sticker_dir: Path,
    rgb_video_path: Path,
) -> bool:
    """Launch the depth-aggregation diagnostics viewer for one session block."""
    print(f"[{rgb_video_path.name}] Launching depth-aggregation diagnostics viewer...")
    xy_csv_path = sticker_dir / (
        rgb_video_path.stem + "_handstickers_summary_2d_coordinates.csv"
    )
    view_xyz_depth_aggregation(
        xy_csv_path=xy_csv_path,
        mkv_path=source_video,
        rgb_video_path=rgb_video_path,
    )
    return True


def build_somatosensory_factory(
    source_video: Path,
    sticker_dir: Path,
    rgb_video_path: Path,
    kinematics_dir: Path,
    session_common_dir: Path,
    session_id: str,
    block_id: str,
) -> Optional[Callable[[], List]]:
    """
    Build and return a scene-object factory for one somatosensory block.

    Returns ``None`` when forearm data is invalid or any required input is
    missing (fail-fast: loading errors inside ``build_somatosensory_scene_factory``
    propagate immediately as exceptions).
    """
    print(f"[{rgb_video_path.name}] Building somatosensory scene factory...")
    forearm_pointcloud_dir = session_common_dir / "forearm_pointclouds"
    metadata_filaname = session_id + "_arm_roi_metadata.json"
    forearm_metadata_path = forearm_pointcloud_dir / metadata_filaname

    if not is_forearm_valid(forearm_pointcloud_dir):
        print("Forearm data invalid — skipping block.")
        return None

    xyz_csv_path = sticker_dir / (rgb_video_path.stem + "_handstickers_xyz_tracked.csv")
    raw_motion_npz_path = kinematics_dir / (rgb_video_path.stem + "_handmodel_motion.npz")
    hand_motion_path = resolve_hand_motion_npz_path(raw_motion_npz_path)

    return build_somatosensory_scene_factory(
        xyz_csv_path=xyz_csv_path,
        kinect_video_path=source_video,
        forearm_pointcloud_dir=forearm_pointcloud_dir,
        forearm_metadata_path=forearm_metadata_path,
        rgb_video_path=rgb_video_path.name,
        hand_motion_path=hand_motion_path,
    )


def build_hand_mesh_comparison(
    source_video: Path,
    sticker_dir: Path,
    rgb_video_path: Path,
    kinematics_dir: Path,
    session_common_dir: Path,
    session_id: str,
    block_id: str,
    session_merged_output_dir: Optional[Path] = None,
) -> Optional[Callable[[], List]]:
    """
    Build and return a scene-object factory for one hand-mesh comparison block.

    Returns ``None`` when forearm data is invalid or the stabilised NPZ does
    not exist (no comparison possible).
    """
    forearm_pointcloud_dir = session_common_dir / "forearm_pointclouds"
    forearm_metadata_path = forearm_pointcloud_dir / (session_id + "_arm_roi_metadata.json")

    if not is_forearm_valid(forearm_pointcloud_dir):
        logger.info("[%s/%s] Forearm data invalid — skipping.", session_id, block_id)
        return None

    raw_npz_path = kinematics_dir / (rgb_video_path.stem + "_handmodel_motion.npz")
    stabilised_npz_path = kinematics_dir / (
        rgb_video_path.stem + "_handmodel_motion_stabilised.npz"
    )

    if not raw_npz_path.exists():
        logger.info("[%s/%s] Raw NPZ not found — skipping.", session_id, block_id)
        return None
    if not stabilised_npz_path.exists():
        logger.info("[%s/%s] Stabilised NPZ not found — skipping.", session_id, block_id)
        return None

    xyz_csv_path = sticker_dir / (rgb_video_path.stem + "_handstickers_xyz_tracked.csv")

    merged_csv_path: Optional[Path] = None
    if session_merged_output_dir is not None:
        merged_name = f"{session_id}_semicontrolled_{block_id}_merged_data.csv"
        candidate = session_merged_output_dir / "blocks_merged" / merged_name
        merged_csv_path = candidate if candidate.exists() else None

    return build_hand_mesh_comparison_factory(
        raw_npz_path=raw_npz_path,
        stabilised_npz_path=stabilised_npz_path,
        kinect_video_path=source_video,
        forearm_pointcloud_dir=forearm_pointcloud_dir,
        forearm_metadata_path=forearm_metadata_path,
        rgb_video_path=rgb_video_path.name,
        xyz_csv_path=xyz_csv_path,
        merged_csv_path=merged_csv_path,
    )


# --- The "Worker" Flow ---
@flow(name="Run Single Session Visualization")
def run_single_session_visualization(
    config: KinectConfig,
    dag_handler: DagConfigHandler
):
    """
    Processes a single dataset by calling visualization sub-routines.

    Note: the ``view_somatosensory_assessement`` task is intentionally NOT
    handled here.  All somatosensory factories are collected in
    ``run_batch_sequentially`` and shown in a single multi-block viewer.
    """
    block_name = config.source_video.name
    print(f"Starting visualization pipeline for block: {block_name}")

    rgb_video_path = config.video_primary_output_dir / f"{config.source_video.stem}.mp4"
    if not rgb_video_path.exists():
        print(f"Critical Error: RGB video not found at {rgb_video_path}.")
        return {"status": "failed", "error": "RGB video not found"}

    try:
        stickers_dir = config.video_processed_output_dir / "handstickers"
        kin_dir = config.video_processed_output_dir / "kinematics_analysis"

        if dag_handler.can_run('view_ellipse_tracking'):
            print(f"[{block_name}] ==> Running task: view_ellipse_tracking")
            view_ellipse_tracking_flow(
                rgb_video_path=rgb_video_path,
                sticker_dir=stickers_dir
            )
            dag_handler.mark_completed('view_ellipse_tracking')

        if dag_handler.can_run('view_ellipse_tracking_adjusted'):
            print(f"[{block_name}] ==> Running task: view_ellipse_tracking_adjusted")
            view_ellipse_adjusted_tracking_flow(
                rgb_video_path=rgb_video_path,
                sticker_dir=stickers_dir
            )
            dag_handler.mark_completed('view_ellipse_tracking_adjusted')

        if dag_handler.can_run('view_consolidated_2d_tracking_data'):
            print(f"[{block_name}] ==> Running task: view_consolidated_2d_tracking_data")
            view_consolidated_2d_tracking_data(
                rgb_video_path=rgb_video_path,
                sticker_dir=stickers_dir
            )
            dag_handler.mark_completed('view_consolidated_2d_tracking_data')

        if dag_handler.can_run('view_xyz_stickers'):
            print(f"[{block_name}] ==> Running task: view_xyz_stickers")
            view_xyz_stickers(
                source_video=config.source_video,
                sticker_dir=stickers_dir,
                rgb_video_path=rgb_video_path,
                session_common_dir=config.session_processed_output_dir,
                session_id=config.session_id
            )
            dag_handler.mark_completed('view_xyz_stickers')

        if dag_handler.can_run('view_xyz_depth_aggregation'):
            print(f"[{block_name}] ==> Running task: view_xyz_depth_aggregation")
            view_xyz_depth_aggregation_flow(
                source_video=config.source_video,
                sticker_dir=stickers_dir,
                rgb_video_path=rgb_video_path,
            )
            dag_handler.mark_completed('view_xyz_depth_aggregation')

    except Exception as e:
        print(f"Pipeline failed during visualization. Error: {e}")
        return {"status": "failed", "error": str(e)}

    print(f"Visualization finished for session: {block_name}")
    return {"status": "success", "completed_tasks": list(dag_handler.completed_tasks)}


# --- The "Dispatcher" Flow ---
@flow(name="Run Visualization Batch Sequentially", log_prints=True)
def run_batch_sequentially(block_files: list[Path], project_data_root: Path, dag_config_path: Path):
    """
    Runs all session pipelines one by one.

    For tasks other than ``view_somatosensory_assessement``, each block is
    processed sequentially by ``run_single_session_visualization``.

    For ``view_somatosensory_assessement`` the factories from ALL blocks are
    collected first and then presented in a single ``SceneViewerVideoMaker``
    window, allowing the user to navigate between blocks without reopening the
    viewer.
    """
    dag_handler_template = DagConfigHandler(dag_config_path)

    # -----------------------------------------------------------------------
    # Pass 1 — per-block non-somatosensory tasks (ellipse, xyz, etc.)
    # -----------------------------------------------------------------------
    for block_file in block_files:
        print(f"--- Running session: {block_file.name} ---")
        try:
            config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
            validated_config = KinectConfig(config_data=config_data, database_path=project_data_root)
            dag_handler_instance = dag_handler_template.copy()

            result = run_single_session_visualization(
                config=validated_config,
                dag_handler=dag_handler_instance
            )
            print(f"--- Completed session: {block_file.name} | Status: {result.get('status', 'unknown')} ---")
        except Exception as e:
            print(f"Failed to initialize session {block_file.name}. Error: {e}")
            continue

    # -----------------------------------------------------------------------
    # Pass 2 — collect somatosensory factories, open ONE viewer for all blocks
    # -----------------------------------------------------------------------
    if dag_handler_template.can_run('view_somatosensory_assessement'):
        session_blocks: Dict[Tuple[str, str], Callable[[], List]] = {}

        for block_file in block_files:
            try:
                config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
                config = KinectConfig(config_data=config_data, database_path=project_data_root)

                rgb_video_path = config.video_primary_output_dir / f"{config.source_video.stem}.mp4"
                if not rgb_video_path.exists():
                    print(f"[{block_file.name}] RGB video not found — skipping somatosensory task.")
                    continue

                stickers_dir = config.video_processed_output_dir / "handstickers"
                kin_dir = config.video_processed_output_dir / "kinematics_analysis"

                factory = build_somatosensory_factory(
                    source_video=config.source_video,
                    sticker_dir=stickers_dir,
                    rgb_video_path=rgb_video_path,
                    kinematics_dir=kin_dir,
                    session_common_dir=config.session_processed_output_dir,
                    session_id=config.session_id,
                    block_id=config.block_id,
                )
                if factory is not None:
                    session_blocks[(config.session_id, config.block_id)] = factory
                    print(f"[{block_file.name}] Somatosensory factory built for ({config.session_id}, {config.block_id}).")
            except Exception as e:
                print(f"Failed to build somatosensory factory for {block_file.name}. Error: {e}")
                continue

        if session_blocks:
            from PyQt5.QtWidgets import QApplication
            from preprocessing.common import SceneViewerVideoMaker

            app = QApplication.instance() or QApplication(sys.argv)
            viewer = SceneViewerVideoMaker(session_blocks=session_blocks)
            viewer.show()
            app.exec_()

            dag_handler_template.mark_completed('view_somatosensory_assessement')
            print("Somatosensory viewer closed.")
        else:
            print("No valid somatosensory blocks found — skipping viewer.")

    # -----------------------------------------------------------------------
    # Pass 3 — hand mesh comparison (raw vs stabilised)
    # -----------------------------------------------------------------------
    if dag_handler_template.can_run('view_hand_mesh_comparison'):
        comparison_blocks: Dict[Tuple[str, str], Callable[[], List]] = {}

        for block_file in block_files:
            try:
                config_data = KinectConfigFileHandler.load_and_resolve_config(block_file)
                config = KinectConfig(config_data=config_data, database_path=project_data_root)

                rgb_video_path = config.video_primary_output_dir / f"{config.source_video.stem}.mp4"
                if not rgb_video_path.exists():
                    continue

                stickers_dir = config.video_processed_output_dir / "handstickers"
                kin_dir = config.video_processed_output_dir / "kinematics_analysis"

                factory = build_hand_mesh_comparison(
                    source_video=config.source_video,
                    sticker_dir=stickers_dir,
                    rgb_video_path=rgb_video_path,
                    kinematics_dir=kin_dir,
                    session_common_dir=config.session_processed_output_dir,
                    session_id=config.session_id,
                    block_id=config.block_id,
                    session_merged_output_dir=config.session_merged_output_dir,
                )
                if factory is not None:
                    comparison_blocks[(config.session_id, config.block_id)] = factory
                    print(f"[{block_file.name}] Hand mesh comparison factory built.")
            except Exception as e:
                print(f"Failed to build hand mesh comparison for {block_file.name}. Error: {e}")
                continue

        if comparison_blocks:
            from PyQt5.QtWidgets import QApplication
            from preprocessing.common import SceneViewerVideoMaker

            app = QApplication.instance() or QApplication(sys.argv)
            viewer = SceneViewerVideoMaker(session_blocks=comparison_blocks)
            viewer.show()
            app.exec_()

            dag_handler_template.mark_completed('view_hand_mesh_comparison')
            print("Hand mesh comparison viewer closed.")
        else:
            print("No blocks with both raw and stabilised NPZs — skipping comparison viewer.")

    print("All sequential visualization runs have completed.")


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--dag-config", type=Path, required=True)
    _args = _parser.parse_args()
    print("Setting up files for visualization...")
    project_data_root = path_tools.get_project_data_root()
    configs_dir = Path("configs")
    dag_config_path = _args.dag_config

    try:
        main_dag_handler = DagConfigHandler(dag_config_path)
        entries = main_dag_handler.get_parameter('kinect_configs')
        block_files = resolve_session_configs(entries, configs_dir / "kinect_configs")
    except FileNotFoundError:
        print(f"Error: '{dag_config_path}' not found.")
        exit(1)

    if main_dag_handler.can_run('view_hand_model_overlay'):
        _overlay_configs = []
        for _block_file in block_files:
            _config_data = KinectConfigFileHandler.load_and_resolve_config(_block_file)
            _overlay_configs.append(KinectConfig(config_data=_config_data, database_path=project_data_root))
        from preprocessing.motion_analysis.hand_tracking.gui.handmesh_overlay_exporter import (
            launch_handmesh_overlay_exporter,
        )
        launch_handmesh_overlay_exporter(_overlay_configs)
        main_dag_handler.mark_completed('view_hand_model_overlay')

    print("Launching visualization batch processing SEQUENTIALLY.")
    run_batch_sequentially(
        block_files=block_files,
        project_data_root=project_data_root,
        dag_config_path=dag_config_path
    )
