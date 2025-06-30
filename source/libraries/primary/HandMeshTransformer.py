import open3d as o3d
import numpy as np
import os
import json
import tkinter as tk
from tkinter import ttk
import copy

class HandMeshTransformer:
    """
    Interactively transforms a list of 3D meshes using Open3D's legacy 
    visualizer and a Tkinter control panel.

    This class provides a workaround for systems where the modern Open3D GUI
    (open3d.visualization.gui) is not compatible.
    """
    def __init__(self, meshes, reference_pcd=None, 
                 tk_root=None,
                 rotation_point=np.array([0.0, 0.0, 0.0]), 
                 save_path="transformations.json"):
        """
        Initializes the application, creating a 3D visualization window and a
        separate control panel.

        Args:
            meshes (list[o3d.geometry.Geometry3D]): A list of meshes or geometries to transform.
            reference_pcd (o3d.geometry.PointCloud, optional): A reference point cloud that will not be transformed. Defaults to None.
            tk_root (tk.Tk, optional): An existing Tkinter root window. If None, a new one is created.
            rotation_point (np.ndarray): The initial point around which to rotate the meshes.
            save_path (str): The file path where to save the transformation data.
        """
        if not isinstance(meshes, list):
            self.meshes = [meshes]
        else:
            self.meshes = meshes
        
        # Keep a deep copy of original meshes for clean transformations
        self.original_meshes = [copy.deepcopy(m) for m in self.meshes]
        self.reference_pcd = reference_pcd
        self.initial_rotation_point = np.copy(rotation_point) # Keep a copy of the initial rotation point
        self.save_path = save_path

        # --- Calculate combined bounding box for all meshes ---
        if self.meshes:
            combined_bbox = self.meshes[0].get_axis_aligned_bounding_box()
            for i in range(1, len(self.meshes)):
                combined_bbox += self.meshes[i].get_axis_aligned_bounding_box()
            self.model_extent = max(combined_bbox.get_max_bound() - combined_bbox.get_min_bound())
        else:
            self.model_extent = 1.0

        # --- Visual markers for the rotation point ---
        # Create a small sphere marker
        sphere_radius = self.model_extent * 0.035 if self.model_extent > 0 else 0.035
        self.rotation_point_marker = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        self.rotation_point_marker.paint_uniform_color([0.0, 0.0, 0.0]) # Black color
        self.original_rotation_marker = copy.deepcopy(self.rotation_point_marker)
        self.rotation_point_marker.translate(self.initial_rotation_point)
        
        # Create a coordinate frame marker
        axis_marker_size = self.model_extent * 0.1 if self.model_extent > 0 else 0.1
        self.rotation_point_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_marker_size, origin=[0, 0, 0])
        self.original_rotation_point_axes = copy.deepcopy(self.rotation_point_axes)
        self.rotation_point_axes.translate(self.initial_rotation_point)

        # --- Transformation State Variables (managed by Tkinter) ---
        self.rotation_vars = []
        self.translation_vars = []

        # --- Application State ---
        self.is_running = True

        # --- Setup the Legacy Open3D Visualizer ---
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="3D Model View", width=1024, height=768)
        
        # Add a coordinate frame at the world origin
        axis_size = self.model_extent * 0.5 if self.model_extent > 0 else 0.5
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_size, origin=[0, 0, 0]
        )
        self.vis.add_geometry(coordinate_frame)
        
        # Add all transformable meshes
        for mesh in self.meshes:
            self.vis.add_geometry(mesh)
        
        # Add the rotation point markers
        self.vis.add_geometry(self.rotation_point_marker)
        self.vis.add_geometry(self.rotation_point_axes)
        
        if self.reference_pcd:
            self.vis.add_geometry(self.reference_pcd)
        
        # --- Setup the Tkinter Controls Window ---
        self._setup_controls_window(tk_root)

    def _setup_controls_window(self, tk_root):
        """Creates the Tkinter window with sliders and buttons."""
        if tk_root is None:
            self.tk_root = tk.Tk()
        else:
            self.tk_root = tk_root
        self.tk_root.title("Transform Controls")
        self.tk_root.protocol("WM_DELETE_WINDOW", self._on_close)
        # Set a minimum size for the control window
        self.tk_root.minsize(350, 450)

        style = ttk.Style(self.tk_root)
        style.theme_use('clam')

        main_frame = ttk.Frame(self.tk_root, padding="10")
        main_frame.pack(expand=True, fill='both')

        # --- Rotation Controls ---
        rot_frame = ttk.LabelFrame(main_frame, text="Rotation (°)", padding="10")
        rot_frame.pack(fill='x', expand=True, pady=5)
        for i, axis in enumerate(["X", "Y", "Z"]):
            self.rotation_vars.append(self._create_slider(rot_frame, f"Rotate {axis}", -180, 180, 0))

        # --- Translation Controls ---
        trans_frame = ttk.LabelFrame(main_frame, text="Translation", padding="10")
        trans_frame.pack(fill='x', expand=True, pady=5)
        limit = self.model_extent if self.model_extent > 0 else 1.0
        for i, axis in enumerate(["X", "Y", "Z"]):
            self.translation_vars.append(self._create_slider(trans_frame, f"Shift {axis}", -limit, limit, 0, resolution=limit/1000))

        # --- Save Button ---
        save_button = ttk.Button(main_frame, text="Save Transformations", command=self._on_save)
        save_button.pack(pady=10, fill='x', expand=True)

        # --- Status Label ---
        self.status_var = tk.StringVar()
        status_label = ttk.Label(main_frame, textvariable=self.status_var, anchor='center')
        status_label.pack(pady=5, fill='x', expand=True)

    def _create_slider(self, parent, label, from_, to, initial_value, resolution=0.1):
        """Helper to create a label, slider, and value display."""
        var = tk.DoubleVar(value=initial_value)
        
        frame = ttk.Frame(parent)
        frame.pack(fill='x', expand=True, pady=2)

        ttk.Label(frame, text=label, width=10).pack(side='left')
        
        slider = ttk.Scale(
            frame,
            from_=from_,
            to=to,
            orient='horizontal',
            variable=var,
            command=self._update_transform
        )
        slider.pack(side='left', fill='x', expand=True, padx=5)

        value_label = ttk.Label(frame, textvariable=var, width=6)
        # Format the label to show 1 decimal place
        var.trace_add("write", lambda *args: value_label.config(text=f"{var.get():.1f}"))
        value_label.config(text=f"{var.get():.1f}") # Set initial text
        value_label.pack(side='left')

        return var

    def _update_transform(self, _=None):
        """
        Applies the current rotation and translation to all meshes and markers.
        This is called whenever a slider value changes.
        """
        if not self.is_running:
            return

        # 1. Get values from Tkinter variables
        rotation_xyz = [var.get() for var in self.rotation_vars]
        translation_xyz = [var.get() for var in self.translation_vars]

        # 2. Get rotation matrix from angles (degrees to radians)
        rot_rad = np.radians(rotation_xyz)
        # Use a dummy mesh to calculate rotation matrix; it's a class method but this is safe
        R = self.original_meshes[0].get_rotation_matrix_from_xyz(rot_rad)

        # 3. Get translation vector
        T = np.array(translation_xyz)

        # 4. Update each transformable mesh
        for i, mesh in enumerate(self.meshes):
            original_mesh = self.original_meshes[i]
            
            # Apply rotation around the initial rotation point
            rotated_vertices = (R @ (np.asarray(original_mesh.vertices) - self.initial_rotation_point).T).T + self.initial_rotation_point
            
            # Apply translation
            transformed_vertices = rotated_vertices + T
            
            # Update mesh vertices and normals
            mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
            mesh.compute_vertex_normals()

            # Update geometry in the 3D scene
            self.vis.update_geometry(mesh)
            
        # 5. Update the rotation point markers' position
        new_marker_center = self.initial_rotation_point + T
        
        # Update sphere marker
        self.rotation_point_marker.vertices = o3d.utility.Vector3dVector(
            np.asarray(self.original_rotation_marker.vertices) + new_marker_center
        )
        self.rotation_point_marker.compute_vertex_normals()
        self.vis.update_geometry(self.rotation_point_marker)
        
        # Update axes marker
        self.rotation_point_axes.vertices = o3d.utility.Vector3dVector(
            np.asarray(self.original_rotation_point_axes.vertices) + new_marker_center
        )
        self.rotation_point_axes.compute_vertex_normals()
        self.vis.update_geometry(self.rotation_point_axes)
        
        # The main `run` loop will handle rendering updates.

    def _on_save(self):
        """Saves the current transformations to a JSON file."""
        # Calculate the final position of the rotation point
        T = np.array([var.get() for var in self.translation_vars])
        final_rotation_point = self.initial_rotation_point + T

        transformations = {
            "rotation_xyz_degrees": [var.get() for var in self.rotation_vars],
            "translation_xyz": [var.get() for var in self.translation_vars],
            "final_rotation_point": final_rotation_point.tolist()
        }

        try:
            with open(self.save_path, 'w') as f:
                json.dump(transformations, f, indent=4)
            feedback_text = f"Saved to {os.path.basename(self.save_path)}"
            print(f"Transformations saved to {self.save_path}")
        except IOError as e:
            feedback_text = "Error: Could not save file."
            print(f"Error saving file: {e}")
        
        # Provide visual feedback and hide it after 2 seconds
        self.status_var.set(feedback_text)
        self.tk_root.after(2000, lambda: self.status_var.set(""))

    def _on_close(self):
        """Sets the flag to stop the main loop when the Tkinter window is closed."""
        self.is_running = False

    def run(self):
        """
        Runs the main application loop, updating both the Open3D visualizer
        and the Tkinter control window.
        """
        while self.is_running:
            # Update the Open3D window and check if it has been closed
            if not self.vis.poll_events():
                self._on_close()
                break
            self.vis.update_renderer()

            # Update the Tkinter window
            try:
                self.tk_root.update()
            except tk.TclError:
                # This can happen if the window is destroyed
                self._on_close()
                break
        
        # --- Cleanup ---
        print("Closing application...")
        self.vis.destroy_window()
        try:
            self.tk_root.destroy()
        except tk.TclError:
            pass # Window might already be gone

# --- Example Usage ---
if __name__ == '__main__':
    # 1. Create a sample list of meshes to transform
    print("Creating sample meshes...")
    box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
    box.compute_vertex_normals()
    box.paint_uniform_color([0.8, 0.8, 0.2]) # Make it yellowish

    # Create a second object
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.4)
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color([0.2, 0.8, 0.8]) # Make it cyan
    sphere.translate(np.array([1.5, 0.0, 0.0])) # Move it next to the box

    list_of_meshes_to_transform = [box, sphere]

    # 2. Center the collection of meshes to make transformations more intuitive
    if list_of_meshes_to_transform:
        combined_bbox = list_of_meshes_to_transform[0].get_axis_aligned_bounding_box()
        for m in list_of_meshes_to_transform[1:]:
            combined_bbox += m.get_axis_aligned_bounding_box()
        center = combined_bbox.get_center()
        for m in list_of_meshes_to_transform:
            m.translate(-center)

    # 3. Create a reference point cloud (a larger sphere) that won't move
    print("Creating a reference point cloud...")
    reference_pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    reference_pcd = reference_pcd.sample_points_uniformly(number_of_points=500)
    reference_pcd.paint_uniform_color([0.2, 0.2, 0.8]) # Make it blue
    reference_pcd.translate(np.array([0.0, 2.0, 0.0])) # Position it above the scene
    
    # 4. Define the initial point for rotation (the origin in this case)
    rotation_center_point = np.array([0.0, 0.0, 0.0])

    # 5. Initialize and run the application
    print("Starting HandMeshTransformer...")
    try:
        transformer_app = HandMeshTransformer(
            meshes=list_of_meshes_to_transform,
            reference_pcd=reference_pcd,
            rotation_point=rotation_center_point,
            save_path="multi_mesh_transformations.json"
        )
        transformer_app.run()
    except Exception as e:
        print(f"An error occurred: {e}")
        print("This might happen if you don't have a display environment (e.g., running on a remote server without X11).")
