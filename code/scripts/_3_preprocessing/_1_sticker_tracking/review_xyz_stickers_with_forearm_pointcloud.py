from pathlib import Path
from PyQt5.QtWidgets import QApplication
from typing import List, Iterable
import sys

from preprocessing.forearm_extraction import (
    ForearmFrameParametersFileHandler,
    ForearmParameters,
    ForearmCatalog,
    get_forearms_with_fallback
)

from preprocessing.stickers_analysis import (
    XYZDataFileHandler,
    XYZReviewWithForearmGUI
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



def review_xyz_stickers_with_forearm_pointcloud(
    xyz_csv_path: Path,
    xyz_md_path: Path,
    forearm_pointcloud_dir: Path,
    forearm_metadata_path: Path,
    rgb_video_path: Path
):
    
    try:
        stickers_df_dict = XYZDataFileHandler.load(xyz_csv_path)
        forearm_params: List[ForearmParameters] = ForearmFrameParametersFileHandler.load(forearm_metadata_path)
        # Collect the forearms pointclouds for the specific video
        catalog = ForearmCatalog(forearm_params, forearm_pointcloud_dir)
        forearms_dict = get_forearms_with_fallback(catalog, rgb_video_path)
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None
    
    
    custom_colors = define_custom_colors(stickers_df_dict.keys())
    forearm = forearms_dict[0]


    app = QApplication.instance() or QApplication(sys.argv)
    # 2. Instantiate the new visualizer class with the data
    visualizer = XYZReviewWithForearmGUI(
        ref_pcd=forearm,
        trajectories=stickers_df_dict,
        trajectory_colors=custom_colors
    )

    # 3. Run the visualizer (this calls setup_scene internally)
    visualizer.run()

    # 4. Execute the Qt application event loop
    sys.exit(app.exec_())
    
    return


if __name__ == "__main__":
    from pathlib import Path

    # Define the file and directory paths
    xyz_csv = Path('F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled/2_processed/kinect/2022-06-15_ST14-01/block-order-07/handstickers/2022-06-15_ST14-01_semicontrolled_block-order07_kinect_handstickers_xyz_tracked.csv')
    
    # NOTE: Inferred the metadata path from the xyz_csv_path
    xyz_md = xyz_csv.with_suffix('.json')
    
    forearm_pcd_dir = Path('F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled/2_processed/kinect/2022-06-15_ST14-01/forearm_pointclouds')
    
    forearm_md = Path('F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled/2_processed/kinect/2022-06-15_ST14-01/forearm_pointclouds/2022-06-15_ST14-01_arm_roi_metadata.json')
    
    # The rgb_video_path is likely just the filename, not a full path object
    rgb_video = '2022-06-15_ST14-01_semicontrolled_block-order07_kinect.mp4'

    # Run the review process
    review_xyz_stickers_with_forearm_pointcloud(
        xyz_csv_path=xyz_csv,
        xyz_md_path=xyz_md,
        forearm_pointcloud_dir=forearm_pcd_dir,
        forearm_metadata_path=forearm_md,
        rgb_video_path=rgb_video
    )