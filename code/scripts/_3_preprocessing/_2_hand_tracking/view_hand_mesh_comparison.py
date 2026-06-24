from pathlib import Path
from typing import Callable, Dict, List, Optional
import logging

import numpy as np
import open3d as o3d
import pandas as pd

from preprocessing.common import (
    ContactPointsSequence,
    LazyPointCloudSequence,
    PersistentOpen3DPointCloudSequence,
    Open3DTriangleMeshSequence,
    Trajectory,
    define_custom_colors,
)

from preprocessing.stickers_analysis import XYZDataFileHandler

from preprocessing.forearm_extraction import (
    ForearmFrameParametersFileHandler,
    ForearmParameters,
    ForearmCatalog,
    get_forearms_with_fallback,
)

from preprocessing.motion_analysis import HandMotionManager

logger = logging.getLogger(__name__)

_COLOR_RAW = (0.0, 1.0, 1.0)       # cyan
_COLOR_STABILISED = (1.0, 1.0, 0.0) # yellow


def _parse_contact_points_cell(cell: str) -> Optional[np.ndarray]:
    """Return (N, 3) float32 array, or None for empty / NaN cells."""
    if not isinstance(cell, str) or cell.strip() in ('', '[]', 'nan'):
        return None
    try:
        nums = np.fromstring(
            cell.replace('[', '').replace(']', ''), sep=' ', dtype=np.float32
        )
        if nums.size == 0 or nums.size % 3 != 0:
            return None
        return nums.reshape(-1, 3)
    except ValueError:
        return None


def _tint_mesh(mesh: o3d.geometry.TriangleMesh, rgb: tuple) -> None:
    n_vertices = len(mesh.vertices)
    colors = np.tile(rgb, (n_vertices, 1))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)


def _load_tinted_meshes(
    npz_path: Path, color: tuple,
) -> Dict[int, o3d.geometry.TriangleMesh]:
    manager = HandMotionManager()
    manager.load(str(npz_path))
    meshes: Dict[int, o3d.geometry.TriangleMesh] = {}
    for i in range(len(manager)):
        mesh = manager[i]
        _tint_mesh(mesh, color)
        meshes[i] = mesh
    return meshes


def build_hand_mesh_comparison_factory(
    raw_npz_path: Path,
    stabilised_npz_path: Path,
    kinect_video_path: Path,
    forearm_pointcloud_dir: Path,
    forearm_metadata_path: Path,
    rgb_video_path: Path,
    xyz_csv_path: Path,
    merged_csv_path: Optional[Path] = None,
) -> Callable[[], List]:
    """
    Build a factory closure that returns scene objects for comparing raw and
    stabilised hand meshes.

    The raw mesh is rendered in cyan, the stabilised in yellow.  Both appear
    as coloured wireframes in the same viewport; the user can toggle each
    independently via the visibility checkbox.

    When *merged_csv_path* is provided and the file contains both
    ``contact_points`` and ``time_kinect`` columns, a ``ContactPointsSequence``
    is appended to the object list.
    """

    # 1. Load Sticker Data
    stickers_df_dict = XYZDataFileHandler.load(xyz_csv_path)
    custom_colors = define_custom_colors(stickers_df_dict.keys())
    stickers_xyz_dict = {
        key: df[['x_mm', 'y_mm', 'z_mm']].to_numpy()
        for key, df in stickers_df_dict.items()
    }

    # 2. Load Forearm Data
    forearm_params: List[ForearmParameters] = (
        ForearmFrameParametersFileHandler.load(forearm_metadata_path)
    )
    catalog = ForearmCatalog(forearm_params, forearm_pointcloud_dir)
    forearms_dict = get_forearms_with_fallback(catalog, rgb_video_path)

    # 3. Load Both Hand Motion Sequences
    logger.info("Loading raw hand meshes from %s", raw_npz_path)
    raw_meshes = _load_tinted_meshes(raw_npz_path, _COLOR_RAW)
    logger.info("Loading stabilised hand meshes from %s", stabilised_npz_path)
    stab_meshes = _load_tinted_meshes(stabilised_npz_path, _COLOR_STABILISED)

    # 4. Optionally load contact points from merged CSV
    contact_pts_by_frame: Dict[int, Optional[np.ndarray]] = {}
    if merged_csv_path is not None:
        df = pd.read_csv(merged_csv_path)
        if 'contact_points' not in df.columns or 'time_kinect' not in df.columns:
            raise ValueError(
                f"merged_csv_path {merged_csv_path} is missing 'contact_points' "
                "or 'time_kinect' column"
            )
        kinect_rows = df.dropna(subset=['time_kinect'])
        contact_pts_by_frame = {
            i: _parse_contact_points_cell(str(cell))
            for i, cell in enumerate(kinect_rows['contact_points'])
        }
        logger.info(
            "Loaded contact points for %d kinect frames from %s",
            len(contact_pts_by_frame), merged_csv_path.name,
        )

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
                name="hand_mesh_raw",
                frame_data=raw_meshes,
                point_size=10,
            ),
            Open3DTriangleMeshSequence(
                name="hand_mesh_stabilised",
                frame_data=stab_meshes,
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

        if contact_pts_by_frame:
            objects.append(
                ContactPointsSequence(
                    name="contact_points",
                    frame_data=contact_pts_by_frame,
                )
            )

        return objects

    return factory
