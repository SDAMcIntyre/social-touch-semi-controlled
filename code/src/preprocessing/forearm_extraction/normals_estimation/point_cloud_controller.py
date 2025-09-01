# point_cloud_controller.py

from .point_cloud_model import PointCloudModel
from .point_cloud_visualizer import PointCloudVisualizer


class PointCloudController:
    """
    Connects the PointCloudModel (data) with the PointCloudVisualizer (view).
    It is the "Controller" in MVC.
    """
    def __init__(self, model: PointCloudModel):
        self.model = model
        self.visualizer: PointCloudVisualizer | None = None
        self.output_path = "output.ply"  # Default save paths
        self.metadata_path = "output_metadata.json"

    def set_visualizer(self, visualizer: PointCloudVisualizer):
        """Connects the controller to the visualizer."""
        self.visualizer = visualizer

    def load_point_cloud(self, filepath: str, output_path: str, metadata_path: str):
        """Loads data, updates UI, and performs initial computation."""
        self.output_path = output_path
        self.metadata_path = metadata_path
        
        if self.model.load(filepath) and self.visualizer:
            # 1. Update slider ranges dynamically based on data bounds
            bounds = self.model.get_bounds()
            if bounds:
                self.visualizer.update_viewpoint_slider_ranges(bounds)
            
            # 2. Perform the initial normal computation
            self.recompute_normals()

    def recompute_normals(self):
        """Triggers normal computation and tells the visualizer to update."""
        if not self.visualizer: return
        print("\nRe-computing normals...")
        self.model.compute_normals()
        self.visualizer.update_plot()
        print("Computation and display update complete.")
    
    def save_and_exit(self):
        """Saves all data and closes the application."""
        self.model.save(self.output_path, self.metadata_path)
        if self.visualizer:
            self.visualizer.close()

    # --- Methods called by Visualizer Callbacks ---

    def set_k_neighbors(self, k: int):
        self.model.k_neighbors = k

    def set_radius(self, radius: float):
        self.model.radius = radius

    def toggle_hybrid(self, state: bool):
        self.model.hybrid_tree = state
    
    def toggle_viewpoint_align(self, state: bool):
        self.model.align_with_viewpoint = state

    def set_viewpoint(self, axis_index: int, value: float):
        self.model.viewpoint[axis_index] = value

    def toggle_centering(self, state: bool):
        self.model.is_centered = state
        if self.visualizer:
            self.visualizer.update_plot()
            if self.model.get_transformed_points().size > 0:
                self.visualizer.plotter.set_focus(self.model.get_transformed_points().mean(axis=0))

    def set_scale_factor(self, scale: float):
        self.model.scale_factor = scale
        if self.visualizer:
            self.visualizer.update_plot()
            
    def flip_normals(self, state: bool):
        # The state from the checkbox is the source of truth
        if self.model.normals_flipped != state:
            self.model.flip_normals()
        if self.visualizer:
            self.visualizer.update_glyphs()