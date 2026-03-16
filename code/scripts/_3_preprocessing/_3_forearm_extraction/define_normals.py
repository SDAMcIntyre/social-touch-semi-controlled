import os

from utils.should_process_task import should_process_task
from preprocessing.forearm_extraction import (
    PointCloudController,
    PointCloudModel,
    PointCloudVisualizer
)

def define_normals(
        input_ply_path: str,
        output_ply_path: str,
        output_metadata_path: str,
        *,
        force_processing: bool = False,
):
    """
    Main function to set up and run the application.
    """
    if not should_process_task(
        output_paths=[output_ply_path, output_metadata_path],
        input_paths=[input_ply_path],
        force=force_processing,
    ):
        return output_ply_path, output_metadata_path
    
    # --- MVC Setup ---
    # 1. Create the Model
    model = PointCloudModel()
    
    # 2. Create the Controller and link it to the Model
    controller = PointCloudController(model)
    
    # 3. Create the View and link it to the Controller
    visualizer = PointCloudVisualizer(controller, source_file=input_ply_path)
    
    # 4. Link the Controller back to the View
    controller.set_visualizer(visualizer)

    # --- Application Start ---
    # Load data, which triggers the first computation and plot
    controller.load_point_cloud(input_ply_path, output_ply_path, output_metadata_path)

    controller.run()


    return output_ply_path, output_metadata_path
