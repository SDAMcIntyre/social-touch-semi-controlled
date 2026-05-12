import sys
from pathlib import Path
from typing import Callable, List
import traceback
from PyQt5.QtWidgets import QApplication
import numpy as np


from preprocessing.common import (
    KinectMKV,
    KinectPointCloudView,
    SceneViewerVideoMaker,
    SceneViewer,
    LazyPointCloudSequence,
    PersistentOpen3DPointCloudSequence,
    Open3DTriangleMeshSequence,
    Trajectory,
    define_custom_colors,
    LineSpec,
    SubplotSpec,
    TimeSeriesPanel,
)

from preprocessing.stickers_analysis import (
    XYZDataFileHandler
)

from preprocessing.forearm_extraction import (
    ForearmFrameParametersFileHandler,
    ForearmParameters,
    ForearmCatalog,
    get_forearms_with_fallback
)

from preprocessing.motion_analysis import (
    HandMotionManager
)


def build_somatosensory_scene_factory(
    xyz_csv_path: Path,
    kinect_video_path: Path,
    forearm_pointcloud_dir: Path,
    forearm_metadata_path: Path,
    rgb_video_path: Path,
    hand_motion_path: Path,
) -> Callable[[], List]:
    """
    Build a factory closure for the somatosensory 3D scene.

    Eagerly loads all non-MKV data (stickers, forearms, hand meshes) so that
    repeated factory calls are cheap.  The ``LazyPointCloudSequence`` opens its
    own MKV handle on first frame access and closes it when
    ``SceneViewer.clear_objects()`` is called.

    Returns
    -------
    Callable[[], List[SceneObject]]
        A zero-argument callable that, when invoked, returns a fresh list of
        scene objects for the block.  May be called multiple times (each call
        creates a new ``LazyPointCloudSequence`` with a fresh MKV handle).

    Raises
    ------
    Exception
        Propagates any loading error immediately (fail-fast).
    """

    # 1. Load Sticker Data
    stickers_df_dict = XYZDataFileHandler.load(xyz_csv_path)
    custom_colors = define_custom_colors(stickers_df_dict.keys())
    stickers_xyz_dict = {
        key: df[['x_mm', 'y_mm', 'z_mm']].to_numpy()
        for key, df in stickers_df_dict.items()
    }

    # 2. Load Forearm Data
    forearm_params: List[ForearmParameters] = ForearmFrameParametersFileHandler.load(forearm_metadata_path)
    catalog = ForearmCatalog(forearm_params, forearm_pointcloud_dir)
    forearms_dict = get_forearms_with_fallback(catalog, rgb_video_path)

    # 3. Load Hand Motion Data
    print(f"Initializing HandMotionManager for: {hand_motion_path}")
    hand_manager = HandMotionManager()
    hand_manager.load(str(hand_motion_path))

    handmeshes_dict = {}
    frame_count = len(hand_manager)
    print(f"Processing {frame_count} frames into Open3D meshes...")
    for i in range(frame_count):
        handmeshes_dict[i] = hand_manager[i]
    print(f"Successfully loaded hand motion sequence.")

    # Return a factory closure; captures the pre-loaded data by reference.
    # The MKV is opened lazily by LazyPointCloudSequence on first frame access.
    def factory() -> List:
        objects: List = [
            LazyPointCloudSequence(
                name="kinect_point_cloud",
                mkv_path=kinect_video_path,
            ),
            PersistentOpen3DPointCloudSequence(
                name="forearms",
                frame_data=forearms_dict,
                point_size=10,
            ),
            Open3DTriangleMeshSequence(
                name="hand_meshes",
                frame_data=handmeshes_dict,
                point_size=10,
            ),
        ]

        for sticker_name, all_positions in stickers_xyz_dict.items():
            trajectory_frame_data = {
                frame_index: position
                for frame_index, position in enumerate(all_positions)
            }
            sticker_color = custom_colors.get(sticker_name, 'magenta')
            objects.append(Trajectory(
                name=sticker_name,
                frame_data=trajectory_frame_data,
                color=sticker_color,
                radius=4.0,
            ))

        return objects

    return factory


def view_somatosensory_3d_scene(
    xyz_csv_path: Path,
    kinect_video_path: Path,
    forearm_pointcloud_dir: Path,
    forearm_metadata_path: Path,
    rgb_video_path: Path,
    hand_motion_path: Path,
    session_id: str = "",
    block_id: str = "",
):
    """
    Launch a single-block somatosensory 3D viewer.

    Builds the scene factory, wraps it in a single-entry ``session_blocks``
    dict, and opens one ``SceneViewerVideoMaker`` window.  This function is
    kept for backward compatibility with the per-block call site in
    ``preprocess_workflow_kinect_visualisation.py``.
    """
    try:
        factory = build_somatosensory_scene_factory(
            xyz_csv_path=xyz_csv_path,
            kinect_video_path=kinect_video_path,
            forearm_pointcloud_dir=forearm_pointcloud_dir,
            forearm_metadata_path=forearm_metadata_path,
            rgb_video_path=rgb_video_path,
            hand_motion_path=hand_motion_path,
        )
    except Exception as e:
        print(f"An error occurred building the somatosensory scene: {e}")
        traceback.print_exc()
        return

    app = QApplication.instance() or QApplication(sys.argv)

    session_blocks = {(session_id, block_id): factory}
    viewer = SceneViewerVideoMaker(session_blocks=session_blocks)
    viewer.show()
    app.exec_()


if __name__ == "__main__":
    # Define the file and directory paths
    xyz_csv = Path('F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled/2_processed/kinect/2022-06-15_ST14-01/block-order-07/handstickers/2022-06-15_ST14-01_semicontrolled_block-order07_kinect_handstickers_xyz_tracked.csv')

    # NOTE: Inferred the metadata path from the xyz_csv_path
    xyz_md = xyz_csv.with_suffix('.json')

    # The rgb_video_path is full path object
    kinect_video_path = Path('F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled/1_primary/kinect/2022-06-15_ST14-01/block-order-07/2022-06-15_ST14-01_semicontrolled_block-order07_kinect.mkv')

    forearm_pointcloud_dir = Path('F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled/2_processed/kinect/2022-06-15_ST14-01/forearm_pointclouds')

    # Assuming this is relative or needs to be adapted to the environment
    rgb_video_path = Path('2022-06-15_ST14-01_semicontrolled_block-order07_kinect.mp4')

    forearm_metadata_path = Path('F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled/2_processed/kinect/2022-06-15_ST14-01/forearm_pointclouds/2022-06-15_ST14-01_arm_roi_metadata.json')

    hand_motion_glb_path = Path('F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled/2_processed/kinect/2022-06-15_ST14-01/hand_motion.glb')

    view_somatosensory_3d_scene(
        xyz_csv_path=xyz_csv,
        kinect_video_path=kinect_video_path,
        forearm_pointcloud_dir=forearm_pointcloud_dir,
        forearm_metadata_path=forearm_metadata_path,
        rgb_video_path=rgb_video_path,
        hand_motion_path=hand_motion_glb_path
    )
