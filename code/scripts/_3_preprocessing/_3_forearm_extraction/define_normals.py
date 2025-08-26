import os

from preprocessing.forearm_analysis.normals_estimation.point_cloud_controller import PointCloudController
from preprocessing.forearm_analysis.normals_estimation.point_cloud_visualizer import PointCloudVisualizer
from preprocessing.forearm_analysis.normals_estimation.point_cloud_model import PointCloudModel

def define_normals(
        input_ply_path: str,
        output_ply_path: str,
        output_metadata_path:str
):
    """
    Main function to set up and run the application.
    """
    if os.path.exists(output_ply_path) and  os.path.exists(output_metadata_path):
        print(f"Normals have already been estimated for file {input_ply_path}.")
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

    # Launch the visualizer's event loop
    visualizer.launch()


    return output_ply_path, output_metadata_path
