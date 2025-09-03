# point_cloud_controller.py
import time
import sys 

from .point_cloud_model import PointCloudModel
from .point_cloud_visualizer import PointCloudVisualizer

# 🧠 KEY CHANGE: Import the function to get the Qt App instance
from qtpy.QtWidgets import QApplication

class PointCloudController:
    """
    Connects the PointCloudModel (data) with the PointCloudVisualizer (view).
    It is the "Controller" in MVC.
    """
    def __init__(self, model: PointCloudModel):
        self.model = model
        self.view: PointCloudVisualizer | None = None
        self.output_path = "output.ply"  # Default save paths
        self.metadata_path = "output_metadata.json"
        self._running = True

    def set_visualizer(self, visualizer: PointCloudVisualizer):
        """Connects the controller to the visualizer."""
        self.view = visualizer

    def load_point_cloud(self, filepath: str, output_path: str, metadata_path: str):
        """Loads data, updates UI, and performs initial computation."""
        self.output_path = output_path
        self.metadata_path = metadata_path
        
        if self.model.load(filepath) and self.view:
            # 1. Update slider ranges dynamically based on data bounds
            bounds = self.model.get_bounds()
            if bounds:
                self.view.update_viewpoint_slider_ranges(bounds)
            
            # 2. Perform the initial normal computation
            self.recompute_normals()

    def recompute_normals(self):
        """Triggers normal computation and tells the visualizer to update."""
        if not self.view: return
        print("\nRe-computing normals...")
        self.model.compute_normals()
        self.view.update_plot()
        print("Computation and display update complete.")
    
    def request_save_and_close(self):
        """Handles saving and signals the main loop to terminate."""
        if not self._running: # Prevent double-execution
            return
            
        print("Controller received save and close request.")
        try:
            # 1. --- SAVE THE STATE FROM THE MODEL ---
            self.model.save(self.output_path, self.metadata_path)
        except Exception as e:
            # Log any errors that occur during the save process
            print(f"❌ Error during save operation: {e}")
        finally:
            self._running = False
            # 2. --- CLOSE THE VIEW ---
            # This ensures the window is always closed, even if saving fails.
            self.view.close()
            print("view closed!")

    def show(self):
        # Launch the visualizer's event loop
        print("Launching interactive visualizer...")
        self.view._setup_plotter()
        self.view._add_widgets()
        # The show() method blocks until the window is closed.
        self.view.plotter.show()
        print("Visualizer closed.")

    def run(self):
        """
        Sets up the UI and enters a controlled loop that waits for the
        visualizer to close.
        """
        self._running = True

        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        # 1. Setup the scene (non-blocking)
        self.view.setup_scene()
        
        print("Main script is now running. Visualizer is active in the background.")
        print("Press 'q' in the visualizer window or use the 'Save & Exit' button.")

        # 2. 🧠 KEY CHANGE: This is our new, controlled "event loop".
        #    We keep the main script alive as long as the window is open.
        try:
            while self._running:
                # Tell the Qt App to process events like button clicks and mouse moves
                app.processEvents()
                # A very short sleep is still good to prevent this loop from
                # needlessly consuming CPU cycles.
                time.sleep(0.01)

                if not self.view.is_active():
                    self._running = False
        except:
            pass
        # 3. This code will now reliably execute after the window closes.
        print("Visualizer window has been closed. Controller is shutting down.")

    
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
        if self.view:
            self.view.update_plot()
            if self.model.get_transformed_points().size > 0:
                self.view.plotter.set_focus(self.model.get_transformed_points().mean(axis=0))

    def set_scale_factor(self, scale: float):
        self.model.scale_factor = scale
        if self.view:
            self.view.update_plot()
            
    def flip_normals(self, state: bool):
        # The state from the checkbox is the source of truth
        if self.model.normals_flipped != state:
            self.model.flip_normals()
        if self.view:
            self.view.update_glyphs()