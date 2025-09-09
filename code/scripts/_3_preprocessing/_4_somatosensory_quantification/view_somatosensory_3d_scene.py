import sys
from pathlib import Path
from typing import Iterable, List
from PyQt5.QtWidgets import QApplication

from preprocessing.common import (
    GLBDataHandler,
    KinectMKV,
    KinectPointCloudView,

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
    HandMotion
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
    hand_motion_glb_path: Path,
    *,
    force_processing: bool = False
):
    
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

    try:
        forearm_params: List[ForearmParameters] = ForearmFrameParametersFileHandler.load(forearm_metadata_path)
    except Exception as e:
        print(f"An error occurred during forearm data loading: {e}")
        return None
    # Collect the forearms pointclouds for the specific video
    catalog = ForearmCatalog(forearm_params, forearm_pointcloud_dir)
    forearms_dict = get_forearms_with_fallback(catalog, rgb_video_path)

    try:
        loader = GLBDataHandler()
        loader.load(hand_motion_glb_path)
        hand_motion_data = loader.get_data()
        if hand_motion_data:
            print(f"Successfully loaded hand motion dictionary.")
    except Exception as e:
        print(f"An error occurred during forearm data loading: {e}")
        return None
    hand_motion_manager = HandMotion(hand_motion_data)
    handmeshes_dict = hand_motion_manager.get_all_transformed_meshes()

    app = QApplication.instance() or QApplication(sys.argv)
    
    with KinectMKV(kinect_video_path) as mkv:
        # 2. Instantiate the SceneViewer
        viewer = SceneViewer()

        point_cloud_view = KinectPointCloudView(mkv)
        # 3. Create the LazyPointCloudSequence and pass the wrapper directly.
        #    No pre-loading loop is needed!
        main_cloud = LazyPointCloudSequence(
            name="kinect_point_cloud",
            data_source=point_cloud_view 
        )
        viewer.add_object(main_cloud)

        forearms_cloud = PersistentOpen3DPointCloudSequence(name="forearms", frame_data=forearms_dict, point_size=10)
        viewer.add_object(forearms_cloud)

        hand_meshes_seq = Open3DTriangleMeshSequence(name="hand_meshes", frame_data=handmeshes_dict, point_size=10)
        viewer.add_object(hand_meshes_seq)
        

        # 4. Prepare the sticker trajectory data
        #    Iterate through each sticker and convert its data into the
        #    Dict[int, np.ndarray] format required by Trajectory.
        for sticker_name, all_positions in stickers_xyz_dict.items():
            # The all_positions array is (num_frames, 3). We need to create a
            # dictionary mapping each frame index to its corresponding (3,) position.
            trajectory_frame_data = {
                frame_index: position
                for frame_index, position in enumerate(all_positions)
            }
            
            # Get the color for this sticker, defaulting to a visible color like 'magenta'
            sticker_color = custom_colors.get(sticker_name, 'magenta')
            
            # Create a Trajectory SceneObject for the sticker and add it
            sticker_object = Trajectory(
                name=sticker_name, 
                frame_data=trajectory_frame_data, 
                color=sticker_color, 
                radius=4.0  # Adjust radius as needed
            )
            viewer.add_object(sticker_object)

        # 5. Show the viewer and run the application
        #    This replaces the old 'window = PointCloudViewer(...)' call.
        viewer.show()
        app.exec_()

    return # This line will not be reached, which is expected for a GUI app.


if __name__ == "__main__":
    from pathlib import Path

    # Define the file and directory paths
    xyz_csv = Path('F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled/2_processed/kinect/2022-06-15_ST14-01/block-order-07/handstickers/2022-06-15_ST14-01_semicontrolled_block-order07_kinect_handstickers_xyz_tracked.csv')
    
    # NOTE: Inferred the metadata path from the xyz_csv_path
    xyz_md = xyz_csv.with_suffix('.json')
    
    # The rgb_video_path is full path object
    kinect_video_path = Path('F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled/1_primary/kinect/2022-06-15_ST14-01/block-order-07/2022-06-15_ST14-01_semicontrolled_block-order07_kinect.mkv')
    
    forearm_pointcloud_dir = Path('F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled/2_processed/kinect/2022-06-15_ST14-01/forearm_pointclouds')
    rgb_video_path = Path('2022-06-15_ST14-01_semicontrolled_block-order07_kinect.mp4')
    forearm_metadata_path = Path('F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled/2_processed/kinect/2022-06-15_ST14-01/forearm_pointclouds/2022-06-15_ST14-01_arm_roi_metadata.json')

    # Run the review process
    review_xyz_stickers_on_depth_data(
        xyz_csv_path=xyz_csv,
        kinect_video_path=kinect_video_path,
        forearm_pointcloud_dir=forearm_pointcloud_dir,
        forearm_metadata_path=forearm_metadata_path,
        rgb_video_path=rgb_video_path
    )