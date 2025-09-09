import os

from preprocessing.forearm_extraction import (
    PointCloudController,
    PointCloudModel,
    PointCloudVisualizer
)

def define_normals(
        input_ply_path: str,
        output_ply_path: str,
        output_metadata_path:str,
        *,
        force_processing: bool = True
):
    """
    Main function to set up and run the application.
    """
    if not os.path.exists(input_ply_path): 
        print(f"Input PLY couldn't be found: {input_ply_path}. Run automatic pipeline first.")
        return None
    
    if not force_processing:
        if os.path.exists(output_ply_path) and os.path.exists(output_metadata_path):
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

    controller.run()


    return output_ply_path, output_metadata_path
