import re
import logging
import open3d as o3d
from pathlib import Path 
from typing import List, Dict, Optional


from preprocessing.common import (
    GLBDataHandler
)

from preprocessing.motion_analysis import (
    ObjectsInteractionController
)

from preprocessing.forearm_extraction import (
    ForearmFrameParametersFileHandler,
    ForearmParameters,
    ForearmCatalog
)


# Configure a basic logger instead of using print()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_forearms_with_fallback(
    catalog: ForearmCatalog, 
    current_video_filename: str
) -> Dict[int, o3d.geometry.PointCloud]:
    """
    Gets forearm point clouds for a video, with intelligent fallback.

    This function first attempts to find all forearms that exactly match the
    video filename. If none are found, it uses the catalog's reference-finding
    logic to locate and load the single closest forearm from another video
    based on the 'block-order' number.

    Args:
        catalog: An initialized ForearmCatalog instance.
        current_video_filename: The filename of the video to process.

    Returns:
        A dictionary mapping frame IDs to o3d.geometry.PointCloud objects. This will
        contain all forearms for the video, a single reference forearm,
        or be empty if no data can be found.
    """
    # 1. Attempt the primary, explicit search first.
    forearms = catalog.get_pointclouds_for_video(current_video_filename)
    if not forearms:
        # 2. If the primary search returned nothing, trigger the fallback.
        logging.info(
            f"No direct forearm match for '{current_video_filename}'. "
            "Attempting to find a fallback reference."
        )
        reference_data = catalog.find_closest_reference(current_video_filename)

        # 3. If a reference was found, format it.
        if reference_data:
            frame_id, pointcloud = reference_data
            forearms = {frame_id: pointcloud}

    if forearms:
        # Find the lowest key and rebuild the dict, replacing that key with 0.
        min_key = min(forearms.keys())
        forearms_adjusted = {0 if k == min_key else k: v for k, v in forearms.items()}
        return forearms_adjusted
    
    # 4. If nothing was found, return an empty dictionary.
    logging.warning(f"Could not find any data or suitable reference for '{current_video_filename}'.")
    return {}



def compute_somatosensory_characteristics(
        hand_motion_glb_path: Path, 
        metadata_path: Path,
        forearm_pointcloud_dir: Path,
        current_video_filename: str,
        output_csv_path: Path,
        *,
        monitor: bool = True,
        fps: bool = 30
) -> str:
    loader = GLBDataHandler()
    loader.load(hand_motion_glb_path)
    hand_motion_data = loader.get_data()
    if hand_motion_data:
        print(f"Successfully loaded hand motion dictionary.")

    forearm_params: List[ForearmParameters] = ForearmFrameParametersFileHandler.load(metadata_path)

    # Collect the forearms pointclouds for the specific video
    catalog = ForearmCatalog(forearm_params, forearm_pointcloud_dir)
    forearms_dict = get_forearms_with_fallback(catalog, current_video_filename)

    # 3. INITIALIZE AND RUN THE ORCHESTRATOR
    controller_with_vis = ObjectsInteractionController(
        hand_motion_data,
        forearms_dict,
        visualize=monitor,
        fps=fps
    )
    results_df = controller_with_vis.run()
    print("Results from visualized run:")
    print(results_df.head())

    # Save the DataFrame to the CSV file
    # index=False prevents pandas from writing the row index to the file
    results_df.to_csv(output_csv_path, index=False)

    return output_csv_path

