import re
import logging
import sys
import open3d as o3d
from pathlib import Path 
from typing import List, Dict, Optional

from PyQt5.QtWidgets import QApplication

from utils.should_process_task import should_process_task

from preprocessing.common import (
    GLBDataHandler
)

from preprocessing.motion_analysis import (
    ObjectsInteractionController,
    ObjectsInteractionVisualizer,
    HandMetadataFileHandler,
    HandMetadataManager
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
        hand_metadata_path: Path, 
        forearm_metadata_path: Path,
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
        input_paths=[hand_motion_glb_path, forearm_metadata_path], 
        force=force_processing):
        print(f"✅ Output file '{output_csv_path}' already exists. Use force_processing to overwrite.")
        return
    
    loader = GLBDataHandler()
    loader.load(hand_motion_glb_path)
    hand_motion_data = loader.get_data()
    if hand_motion_data:
        print(f"Successfully loaded hand motion dictionary.")

    hand_metadata: HandMetadataManager = HandMetadataFileHandler.load(hand_metadata_path)
    
    # Collect the forearms pointclouds for the specific video
    forearm_params: List[ForearmParameters] = ForearmFrameParametersFileHandler.load(forearm_metadata_path)
    catalog = ForearmCatalog(forearm_params, forearm_pointcloud_dir)
    forearms_dict = get_forearms_with_fallback(catalog, current_video_filename, use_mesh=False)

    # 3. INITIALIZE AND RUN THE ORCHESTRATOR
    controller = ObjectsInteractionController(
        hand_motion_data,
        forearms_dict,
        selected_points=hand_metadata.selected_points,
        fps=fps
    )
    
    # Modified call: unpack tuple (DataFrame, Visualization Artifacts)
    results_df, vis_artifacts = controller.run()
    
    print("Results from run:")
    print(results_df.head())

    # --- MONITORING LOGIC ---
    if monitor and vis_artifacts:
        print("Starting interactive monitoring session...")
        if QApplication.instance() is None:
            app = QApplication(sys.argv)
        
        # Create a dummy PointCloud for contact points visualization (Red)
        contact_pcd = o3d.geometry.PointCloud()
        contact_pcd.points = o3d.utility.Vector3dVector([[0, 0, -9999]])
        contact_pcd.paint_uniform_color([1.0, 0.0, 0.0]) 

        view = ObjectsInteractionVisualizer(width=vis_artifacts['view_width'])
        
        # Add geometries from the returned artifacts
        view.add_geometry('forearm', vis_artifacts['reference_pcd'])
        view.add_geometry('hand', vis_artifacts['base_mesh'])
        view.add_geometry('contacts', contact_pcd, point_size=10.0)
        
        # Set animation data
        view.set_frame_data(vis_artifacts['frames'])

        # Prepare Plot Data from the Results DataFrame
        records = results_df.to_dict('records')
        contact_depth_data = [r.get('contact_depth', 0) for r in records]
        contact_area_data = [r.get('contact_area', 0) for r in records]
        
        view.add_plot(
            title="Contact Properties",
            data_vector=contact_depth_data,
            color='g',
            y_label='Depth (NA)'
        )
        view.add_plot(
            title="Contact Properties",
            data_vector=contact_area_data,
            color='m',
            y_label='Contact Area (cm^2)'
        )

        center_point = vis_artifacts['reference_pcd'].get_center()
        view.recenter_view_on_point(center_point)

        view.run()
        
        # Execute App if this is the top-level trigger
        app = QApplication.instance()
        app.exec()
    # ------------------------

    # Save the DataFrame to the CSV file
    results_df.to_csv(output_csv_path, index=False)

    return output_csv_path