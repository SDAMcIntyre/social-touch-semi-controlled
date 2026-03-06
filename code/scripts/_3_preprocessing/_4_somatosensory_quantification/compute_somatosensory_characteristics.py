import re
import logging
import sys
import open3d as o3d
import numpy as np
from pathlib import Path 
from typing import List, Dict, Optional
import pandas as pd 

from PyQt5.QtWidgets import QApplication

from utils.should_process_task import should_process_task

from preprocessing.motion_analysis import (
    HandMotionManager,
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

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def _style_open3d_mesh(mesh: o3d.geometry.TriangleMesh, color: List[float]) -> o3d.geometry.TriangleMesh:
    """
    Applies visual styling (color and normals) to an existing Open3D mesh.
    
    Args:
        mesh: The Open3D TriangleMesh instance (modified in-place).
        color: RGB list [r, g, b] (0.0 - 1.0).
        
    Returns:
        The styled Open3D mesh.
    """
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh

def compute_somatosensory_characteristics(
        hand_motion_path: Path, 
        hand_metadata_path: Path, 
        forearm_metadata_path: Path,
        forearm_pointcloud_dir: Path,
        current_video_filename: str,
        output_csv_path: Path,
        *,
        force_processing: bool = False,
        monitor: bool = True,
        fps: int = 30
) -> Optional[str]:
    if not should_process_task(
        output_paths=output_csv_path, 
        input_paths=[hand_motion_path, forearm_metadata_path], 
        force=force_processing):
        print(f"✅ Output file '{output_csv_path}' already exists. Use force_processing to overwrite.")
        return None
    
    # 1. LOAD MOTION DATA via HandMotionManager
    print(f"Loading Hand Motion Data from: {hand_motion_path}")
    motion_manager = HandMotionManager(fps=float(fps))
    try:
        motion_manager.load(str(hand_motion_path))
    except Exception as e:
        logging.error(f"Failed to load GLB: {e}")
        return None

    if len(motion_manager) == 0:
        logging.error("HandMotionManager loaded 0 frames.")
        return None

    print(f"Successfully loaded hand motion data. Frames: {len(motion_manager)}")

    # 2. LOAD METADATA & FOREARMS
    hand_metadata: HandMetadataManager = HandMetadataFileHandler.load(hand_metadata_path)
    
    forearm_params: List[ForearmParameters] = ForearmFrameParametersFileHandler.load(forearm_metadata_path)
    catalog = ForearmCatalog(forearm_params, forearm_pointcloud_dir)
    forearms_dict = get_forearms_with_fallback(catalog, current_video_filename, use_mesh=True)

    # 3. PREPARE MESH SEQUENCE (ADAPTER LAYER)
    # Convert Manager (Sequence of Trimesh) -> List of Open3D Meshes
    hand_color = [228/255, 178/255, 148/255]
    
    # Pre-generate the Open3D mesh sequence
    # Note: HandMotionManager.__getitem__ applies the World Transform and returns an o3d.geometry.TriangleMesh.
    # We apply styling directly to these objects.
    print("Converting motion frames to Open3D geometries...")
    o3d_hand_sequence = [
        _style_open3d_mesh(motion_manager[i], hand_color) 
        for i in range(len(motion_manager))
    ]

    # 4. INITIALIZE AND RUN THE ORCHESTRATOR
    # Now injecting the explicit sequence of meshes and timestamps
    controller = ObjectsInteractionController(
        hand_meshes=o3d_hand_sequence,
        timestamps=motion_manager.timestamps,
        references_mesh=forearms_dict,
        selected_points=hand_metadata.selected_points,
        excluded_vertex_ids=hand_metadata.excluded_vertex_ids,
        fps=fps
    )
    
    results_df, vis_artifacts = controller.run()
    
    print("Results from run:")
    print(results_df.head())

    # --- MONITORING LOGIC ---
    if monitor and vis_artifacts:
        print("Starting interactive monitoring session...")
        if QApplication.instance() is None:
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()
        
        contact_pcd = o3d.geometry.PointCloud()
        contact_pcd.points = o3d.utility.Vector3dVector([[0, 0, -9999]])
        contact_pcd.paint_uniform_color([1.0, 0.0, 0.0]) 

        view = ObjectsInteractionVisualizer(width=vis_artifacts['view_width'])
        
        view.add_geometry('forearm', vis_artifacts['reference_mesh'])
        view.add_geometry('hand', vis_artifacts['base_mesh'])
        view.add_geometry('contacts', contact_pcd, point_size=10.0)
        
        view.set_frame_data(vis_artifacts['frames'])

        records = results_df.to_dict('records')
        contact_depth_data = [r.get('contact_depth', 0) for r in records]
        contact_area_data = [r.get('contact_area', 0) for r in records]
        
        view.add_plot(
            title="Contact Depth",
            data_vector=contact_depth_data,
            color='g',
            y_label='Depth (NA)'
        )
        view.add_plot(
            title="Contact Area",
            data_vector=contact_area_data,
            color='m',
            y_label='Area (cm^2)'
        )

        center_point = vis_artifacts['reference_mesh'].get_center()
        view.recenter_view_on_point(center_point)

        view.run()
        
        #app = QApplication.instance()
        # --- CRITICAL FIX: Start the Qt Event Loop ---
        # This line blocks execution and hands control to the Qt Event System.
        # The script will now wait here until the window is closed.
        try:
            # PySide6 / PyQt6 syntax
            sys.exit(app.exec())
        except AttributeError:
            # PyQt5 syntax fallback
            sys.exit(app.exec_())
        except SystemExit:
            # Handles the exception raised by sys.exit() cleanly
            pass

    # Save the DataFrame to the CSV file
    results_df.to_csv(output_csv_path, index=False)

    return str(output_csv_path)