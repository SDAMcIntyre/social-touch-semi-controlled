import json
import os
from pathlib import Path

import numpy as np

from utils.should_process_task import refresh_output_mtimes, should_process_task
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
    output_metadata_path = Path(output_metadata_path)
    output_ply_path_obj = Path(output_ply_path)

    outputs_existed = output_metadata_path.exists() and output_ply_path_obj.exists()

    if not should_process_task(
        output_paths=[output_ply_path, output_metadata_path],
        input_paths=[input_ply_path],
        force=force_processing,
    ):
        return str(output_ply_path_obj), str(output_metadata_path)

    # --- Seed model parameters from prior outputs if they exist ---
    model_kwargs = {}
    if output_metadata_path.exists():
        try:
            with open(output_metadata_path, 'r') as f:
                prior = json.load(f)
            proc = prior.get("processing_parameters", {})
            trans = prior.get("final_transformations", {})
            model_kwargs = {
                "k_neighbors":         proc.get("k_neighbors_for_normals", 100),
                "radius":              proc.get("radius_for_hybrid", 0.1),
                "hybrid_tree":         proc.get("used_hybrid_tree", False),
                "align_with_viewpoint": proc.get("aligned_with_viewpoint", False),
                "viewpoint":           np.array(proc.get("viewpoint_vector", [0.0, 0.0, 0.0])),
                "is_centered":         trans.get("is_centered", False),
                "scale_factor":        trans.get("scale_factor", 1.0),
                "normals_flipped":     trans.get("normals_flipped", False),
            }
            print(f"Pre-filling normals GUI from prior metadata: {output_metadata_path}")
        except Exception as e:
            print(f"Warning: could not read prior metadata for pre-fill ({e}). Using defaults.")

    # --- MVC Setup ---
    # 1. Create the Model (seeded with prior values when available)
    model = PointCloudModel(**model_kwargs)

    # 2. Create the Controller and link it to the Model
    controller = PointCloudController(model)

    # 3. Create the View and link it to the Controller
    visualizer = PointCloudVisualizer(controller, source_file=input_ply_path)

    # 4. Link the Controller back to the View
    controller.set_visualizer(visualizer)

    # --- Application Start ---
    # Load data, which triggers the first computation and plot
    controller.load_point_cloud(input_ply_path, str(output_ply_path_obj), str(output_metadata_path))

    controller.run()

    # If outputs already existed and the operator closed without saving,
    # refresh their mtimes so the staleness check does not refire.
    if outputs_existed and not controller.saved:
        refresh_output_mtimes([output_ply_path_obj, output_metadata_path])

    return str(output_ply_path_obj), str(output_metadata_path)
