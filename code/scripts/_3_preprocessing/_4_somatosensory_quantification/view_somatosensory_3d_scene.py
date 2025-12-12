import sys
from pathlib import Path
from typing import Iterable, List
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
    Trajectory
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

def define_custom_colors(string_list: Iterable[str]) -> dict[str, str]:
    """
    Searches an iterable of strings for standard color keywords.

    Args:
        string_list: An iterable (e.g., list, dict_keys) of strings to search through.

    Returns:
        A list of unique color names found in the strings.
    """
    # Define the set of standard color keywords to search for
    STANDARD_COLORS = {
        "red", "green", "blue", "yellow", "orange", "purple", "pink",
        "black", "white", "brown", "gray", "grey", "cyan", "magenta", "violet"
    }
    
    found_colors = {}
    
    # Iterate through each string in the input list
    for item in string_list:
        # Convert the string to lowercase for case-insensitive matching
        item_lower = item.lower()
        # Check if any of the standard colors are a substring of the item
        for color in STANDARD_COLORS:
            if color in item_lower:
                found_colors[item] = color
    
    return found_colors


def view_somatosensory_3d_scene(
    xyz_csv_path: Path,
    kinect_video_path: Path,
    forearm_pointcloud_dir: Path,
    forearm_metadata_path: Path,
    rgb_video_path: Path,
    hand_motion_path: Path
):
    
    # 1. Load Sticker Data
    try:
        stickers_df_dict = XYZDataFileHandler.load(xyz_csv_path)
    except Exception as e:
        print(f"An error occurred during XYZ stickers data loading: {e}")
        return

    custom_colors = define_custom_colors(stickers_df_dict.keys())
    stickers_xyz_dict = {
        key: df[['x_mm', 'y_mm', 'z_mm']].to_numpy()
        for key, df in stickers_df_dict.items()
    }

    # 2. Load Forearm Data
    try:
        forearm_params: List[ForearmParameters] = ForearmFrameParametersFileHandler.load(forearm_metadata_path)
    except Exception as e:
        print(f"An error occurred during forearm data loading: {e}")
        return None
    
    # Collect the forearms pointclouds for the specific video
    catalog = ForearmCatalog(forearm_params, forearm_pointcloud_dir)
    forearms_dict = get_forearms_with_fallback(catalog, rgb_video_path)

    # 3. Load Hand Motion Data (Refactored to use HandMotionManager)
    try:
        print(f"Initializing HandMotionManager for: {hand_motion_path}")
        hand_manager = HandMotionManager()
        hand_manager.load(str(hand_motion_path))
        
        # Open3DTriangleMeshSequence expects a Dict[int, Mesh].
        # We eagerly calculate transformations from the manager to populate this.
        handmeshes_dict = {}
        frame_count = len(hand_manager)
        
        print(f"Processing {frame_count} frames into Open3D meshes...")
        for i in range(frame_count):
            # hand_manager[i] returns the transformed o3d.geometry.TriangleMesh
            handmeshes_dict[i] = hand_manager[i]
            
        print(f"Successfully loaded hand motion sequence.")

    except Exception as e:
        print(f"An error occurred during hand motion loading: {e}")
        traceback.print_exc()
        return None

    # 4. Initialize Visualization
    app = QApplication.instance() or QApplication(sys.argv)
    
    with KinectMKV(kinect_video_path) as mkv:
        # Instantiate the SceneViewer
        viewer = SceneViewerVideoMaker()

        point_cloud_view = KinectPointCloudView(mkv)
        
        # Create the LazyPointCloudSequence
        main_cloud = LazyPointCloudSequence(
            name="kinect_point_cloud",
            data_source=point_cloud_view 
        )
        viewer.add_object(main_cloud)

        # Add Forearms
        forearms_cloud = PersistentOpen3DPointCloudSequence(
            name="forearms", 
            frame_data=forearms_dict, 
            point_size=10
        )
        viewer.add_object(forearms_cloud)

        # Add Hand Meshes
        hand_meshes_seq = Open3DTriangleMeshSequence(
            name="hand_meshes", 
            frame_data=handmeshes_dict, 
            point_size=10
        )
        viewer.add_object(hand_meshes_seq)
        

        # 5. Prepare the sticker trajectory data
        for sticker_name, all_positions in stickers_xyz_dict.items():
            # The all_positions array is (num_frames, 3).
            trajectory_frame_data = {
                frame_index: position
                for frame_index, position in enumerate(all_positions)
            }
            
            # Get the color for this sticker
            sticker_color = custom_colors.get(sticker_name, 'magenta')
            
            # Create a Trajectory SceneObject
            sticker_object = Trajectory(
                name=sticker_name, 
                frame_data=trajectory_frame_data, 
                color=sticker_color, 
                radius=4.0
            )
            viewer.add_object(sticker_object)

        # 6. Show the viewer and run the application
        viewer.show()
        app.exec_()

    return


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
    
    # Path for the GLB file (Must be defined to pass to the function)
    # Assuming it resides in a similar directory or needs a placeholder path. 
    # Based on the previous imports, we'll assume a standard location or the user provides it.
    # For this execution, we will define a placeholder path.
    hand_motion_glb_path = Path('F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled/2_processed/kinect/2022-06-15_ST14-01/hand_motion.glb')

    # Run the review process (Corrected function name)
    view_somatosensory_3d_scene(
        xyz_csv_path=xyz_csv,
        kinect_video_path=kinect_video_path,
        forearm_pointcloud_dir=forearm_pointcloud_dir,
        forearm_metadata_path=forearm_metadata_path,
        rgb_video_path=rgb_video_path,
        hand_motion_glb_path=hand_motion_glb_path
    )