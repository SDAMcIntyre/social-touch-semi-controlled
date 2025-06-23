import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import os
import sys
import json
import time
import threading


class HandMeshTransformer:
    def __init__(self, mesh, reference_pcd=None, rotation_point=np.array([0.0, 0.0, 0.0]), save_path="transformations.json"):
        """
        Initializes the application, creating two separate windows for the 3D
        visualization and the transformation controls.

        Args:
            mesh (o3d.geometry.TriangleMesh): The mesh to transform.
            reference_pcd (o3d.geometry.PointCloud, optional): A reference point cloud that will not be transformed. Defaults to None.
            rotation_point (np.ndarray): The initial point around which to rotate the mesh.
            save_path (str): The file path where to save the transformation data.
        """
        self.mesh = mesh
        self.original_mesh = o3d.geometry.TriangleMesh(mesh)  # Keep a copy for clean transforms
        self.reference_pcd = reference_pcd
        self.rotation_point = rotation_point
        self.initial_rotation_point = np.copy(rotation_point) # Keep a copy of the initial rotation point
        self.save_path = save_path

        # --- Transformation State Variables ---
        self.rotation_xyz = [0.0, 0.0, 0.0]  # [X, Y, Z] angles in degrees
        self.translation_xyz = [0.0, 0.0, 0.0]  # [X, Y, Z] shift values

        # --- Application and Window Setup ---
        self.app = gui.Application.instance
        self.app.initialize()

        # --- Create the Main 3D View Window ---
        self.window = self.app.create_window("3D Model View", 1024, 768)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)

        # --- 3D Scene Widget ---
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.widget3d)

        # --- Add mesh to the scene ---
        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultLit"
        self.widget3d.scene.add_geometry("mesh", self.mesh, self.material)

        # --- Add reference point cloud to the scene if it exists ---
        if self.reference_pcd:
            pcd_material = rendering.MaterialRecord()
            pcd_material.shader = "defaultUnlit"
            pcd_material.point_size = 3.0
            self.widget3d.scene.add_geometry("reference_pcd", self.reference_pcd, pcd_material)


        # Set camera view and determine model size for sliders
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(60, bounds, bounds.get_center())
        self.model_extent = max(bounds.get_extent())

        # --- Create the Controls Window ---
        self.controls_window = self.app.create_window("Transform Controls", 320, 500)
        self.controls_window.set_on_close(self._on_close)

        em = self.controls_window.theme.font_size
        self.panel = gui.Vert(0.5 * em, gui.Margins(em))
        self.controls_window.add_child(self.panel)

        # --- Rotation Controls ---
        self.panel.add_child(gui.Label("Rotation (°)"))
        for i, axis in enumerate(["X", "Y", "Z"]):
            self.panel.add_child(gui.Label(f"Rotate {axis}-axis"))
            self.panel.add_child(self._create_rotation_controls(i))

        # --- Translation Controls ---
        self.panel.add_child(gui.Label("Translation"))
        for i, axis in enumerate(["X", "Y", "Z"]):
            self.panel.add_child(gui.Label(f"Shift {axis}-axis"))
            self.panel.add_child(self._create_translation_controls(i))

        # --- Save Button ---
        save_button = gui.Button("Save Transformations")
        save_button.set_on_clicked(self._on_save)
        self.panel.add_child(save_button)

        # --- Visual Feedback Label ---
        self.status_label = gui.Label("")
        self.status_label.visible = False
        self.panel.add_child(self.status_label)


    def _create_linked_controls(self, on_value_changed):
        """Helper to create a linked slider and number editor."""
        h_layout = gui.Horiz(0.2 * self.controls_window.theme.font_size)
        slider = gui.Slider(gui.Slider.DOUBLE)
        num_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)

        # Link slider and number editor to each other
        slider.set_on_value_changed(lambda val: num_edit.set_value(val))
        num_edit.set_on_value_changed(lambda val: slider.set_double_value(val))

        # Link both to the main update callback
        slider.set_on_value_changed(on_value_changed)
        num_edit.set_on_value_changed(on_value_changed)

        h_layout.add_child(slider)
        h_layout.add_child(num_edit)
        return h_layout, slider, num_edit

    def _create_rotation_controls(self, axis_index):
        """Creates controls for rotation."""
        on_change = lambda value: self._on_rotation_change(axis_index, value)
        layout, slider, num_edit = self._create_linked_controls(on_change)
        slider.set_limits(0, 360)
        num_edit.set_value(0)
        return layout

    def _create_translation_controls(self, axis_index):
        """Creates controls for translation."""
        on_change = lambda value: self._on_translation_change(axis_index, value)
        layout, slider, num_edit = self._create_linked_controls(on_change)
        # Set slider limits based on the model's size for intuitive shifting
        limit = self.model_extent if self.model_extent > 0 else 1.0
        slider.set_limits(-limit, limit)
        num_edit.set_value(0)
        return layout

    def _on_rotation_change(self, axis_index, value):
        self.rotation_xyz[axis_index] = value
        self._update_mesh_transform()

    def _on_translation_change(self, axis_index, value):
        self.translation_xyz[axis_index] = value
        self._update_mesh_transform()

    def _update_mesh_transform(self):
        """
        Applies the current rotation and translation to the mesh and provides visual feedback.
        """
        # 1. Get rotation matrix from angles (degrees to radians)
        rot_rad = np.radians(self.rotation_xyz)
        R = self.original_mesh.get_rotation_matrix_from_xyz(rot_rad)

        # 2. Get translation vector
        T = np.array(self.translation_xyz)
        
        # 3. Update the rotation point based on the translation
        self.rotation_point = self.initial_rotation_point + T

        # 4. Apply rotation around the updated rotation point
        original_vertices = np.asarray(self.original_mesh.vertices)
        rotated_vertices = (R @ (original_vertices - self.initial_rotation_point).T).T + self.initial_rotation_point

        # 5. Apply translation
        transformed_vertices = rotated_vertices + T

        # 6. Update mesh vertices and normals
        self.mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
        self.mesh.compute_vertex_normals()

        # 7. Update geometry in the 3D scene
        self.widget3d.scene.remove_geometry("mesh")
        self.widget3d.scene.add_geometry("mesh", self.mesh, self.material)

        # 8. Provide visual feedback
        self.status_label.text = "Transformation Refreshed"
        self.status_label.visible = True

        # Hide the label after 2 seconds
        def hide_label():
            time.sleep(2.0)
            self.app.post_to_main_thread(self.window, self._hide_status_label)

        # Run the timer in a separate thread to avoid blocking the GUI
        threading.Thread(target=hide_label).start()

    def _hide_status_label(self):
        """Hides the status label."""
        self.status_label.visible = False

    def _on_save(self):
        """Saves the current rotation and translation to the specified JSON file."""
        transformations = {
            "rotation_xyz_degrees": self.rotation_xyz,
            "translation_xyz": self.translation_xyz,
            "final_rotation_point": self.rotation_point.tolist()
        }

        try:
            with open(self.save_path, 'w') as f:
                json.dump(transformations, f, indent=4)

            feedback_text = f"Saved to {os.path.basename(self.save_path)}"
            print(f"Transformations saved to {self.save_path}")

        except IOError as e:
            feedback_text = "Error: Could not save file."
            print(f"Error saving file: {e}")

        # Provide visual feedback on save action
        self.status_label.text = feedback_text
        self.status_label.visible = True
        # Hide the label after 2 seconds
        def hide_label():
            time.sleep(2.0)
            self.app.post_to_main_thread(self.window, self._hide_status_label)

        threading.Thread(target=hide_label).start()

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.widget3d.frame = r

    def _on_close(self):
        self.app.quit()

    def run(self):
        self.app.run()