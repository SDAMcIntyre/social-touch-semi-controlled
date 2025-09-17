import re
import logging
import open3d as o3d
from pathlib import Path 
from typing import List, Dict, Optional


from utils.should_process_task import should_process_task

from preprocessing.common import (
    GLBDataHandler
)

from preprocessing.motion_analysis import (
    ObjectsInteractionController
)

from preprocessing.forearm_extraction import (
    ForearmFrameParametersFileHandler,
    ForearmParameters,
    ForearmCatalog,
    get_forearms_with_fallback
)


# Configure a basic logger instead of using print()
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def compute_somatosensory_characteristics(
        hand_motion_glb_path: Path, 
        metadata_path: Path,
        forearm_pointcloud_dir: Path,
        current_video_filename: str,
        output_csv_path: Path,
        *,
        force_processing: bool = False,
        monitor: bool = False,
        fps: bool = 30
) -> str:
    if not should_process_task(
        output_paths=output_csv_path, 
        input_paths=[hand_motion_glb_path, metadata_path], 
        force=force_processing):
        print(f"✅ Output file '{output_csv_path}' already exists. Use --force to overwrite.")
        return
    
    loader = GLBDataHandler()
    loader.load(hand_motion_glb_path)
    hand_motion_data = loader.get_data()
    if hand_motion_data:
        print(f"Successfully loaded hand motion dictionary.")

    # Collect the forearms pointclouds for the specific video
    forearm_params: List[ForearmParameters] = ForearmFrameParametersFileHandler.load(metadata_path)
    catalog = ForearmCatalog(forearm_params, forearm_pointcloud_dir)
    forearms_dict = get_forearms_with_fallback(catalog, current_video_filename)

    # 3. INITIALIZE AND RUN THE ORCHESTRATOR
    controller = ObjectsInteractionController(
        hand_motion_data,
        forearms_dict,
        visualize=monitor,
        fps=fps
    )
    results_df = controller.run()
    print("Results from visualized run:")
    print(results_df.head())

    # Save the DataFrame to the CSV file
    # index=False prevents pandas from writing the row index to the file
    results_df.to_csv(output_csv_path, index=False)

    return output_csv_path

