import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import os
import sys
import json
import time

# Ensure this path is correct for your project structure
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'imported'))
from imported.hand_mesh import HandMesh



class MeshTransformer:
    def __init__(self, mesh, save_path="transformations.json"):
        """
        Initializes the application, creating two separate windows for the 3D
        visualization and the transformation controls.

        Args:
            mesh (o3d.geometry.TriangleMesh): The mesh to transform.
            save_path (str): The file path where to save the transformation data.
        """
        self.mesh = mesh
        self.original_mesh = o3d.geometry.TriangleMesh(mesh) # Keep a copy for clean transforms
        self.rotation_center = np.array([0.0, 0.0, 0.0]) # Default rotation origin
        self.save_path = save_path

        # --- Transformation State Variables ---
        self.rotation_xyz = [0.0, 0.0, 0.0] # [X, Y, Z] angles in degrees
        self.translation_xyz = [0.0, 0.0, 0.0] # [X, Y, Z] shift values

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
        R = self.mesh.get_rotation_matrix_from_xyz(rot_rad)

        # 2. Get translation vector
        T = np.array(self.translation_xyz)

        # 3. Apply rotation around the center point
        original_vertices = np.asarray(self.original_mesh.vertices)
        rotated_vertices = (R @ (original_vertices - self.rotation_center).T).T + self.rotation_center
        
        # 4. Apply translation
        transformed_vertices = rotated_vertices + T
        
        # 5. Update mesh vertices and normals
        self.mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
        self.mesh.compute_vertex_normals()

        # 6. Update geometry in the 3D scene
        self.widget3d.scene.remove_geometry("mesh")
        self.widget3d.scene.add_geometry("mesh", self.mesh, self.material)
        
        # 7. Provide visual feedback
        self.status_label.text = "Transformation Refreshed"
        self.status_label.visible = True
        
        # Hide the label after 2 seconds
        def hide_label():
            time.sleep(2.0)
            self.app.post_to_main_thread(self.window, self._hide_status_label)

        # Run the timer in a separate thread to avoid blocking the GUI
        import threading
        threading.Thread(target=hide_label).start()

    def _hide_status_label(self):
        """Hides the status label."""
        self.status_label.visible = False

    def _on_save(self):
        """Saves the current rotation and translation to the specified JSON file."""
        transformations = {
            "rotation_xyz_degrees": self.rotation_xyz,
            "translation_xyz": self.translation_xyz
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
            
        import threading
        threading.Thread(target=hide_label).start()

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.widget3d.frame = r

    def _on_close(self):
        self.app.quit()

    def run(self):
        self.app.run()


def main():
    """
    Loads a mesh and displays it with GUI sliders for rotation and translation.
    """
    # --- Path setup ---
    # !! IMPORTANT !!: Update these paths to point to your actual file locations.
    handmesh_path = "F:\\OneDrive - Linköpings universitet\\_Teams\\Social touch Kinect MNG\\data\\semi-controlled\\handmesh_models\\singleFingerVertices0.txt"
    handmesh_model_path = "F:\\GitHub\\social-touch-semi-controlled\\source\\imported\\model\\hand_mesh\\hand_mesh_model.pkl"
    
    # Define the output path for the JSON file.
    # This file will be saved in the same directory where the script is run.
    save_file_path = "hand_transformations.json"


    # --- Vertex and Triangle Loading ---
    if not os.path.exists(handmesh_path):
        print(f"Error: Vertex data file not found at '{handmesh_path}'")
        print("Creating a simple cube as a fallback mesh.")
        v_handMesh = np.array([
            [0,0,0], [1,0,0], [1,1,0], [0,1,0],
            [0,0,1], [1,0,1], [1,1,1], [0,1,1]
        ])
    else:
        print(f"Loading vertices from: {handmesh_path}")
        v_handMesh = np.loadtxt(handmesh_path)

    if not os.path.exists(handmesh_model_path):
        print(f"Warning: Hand mesh model not found at '{handmesh_model_path}'")
        hand_mesh_model = HandMesh() # Will be empty, fallback below will be used
    else:
        print(f"Loading hand model from: {handmesh_model_path}")
        hand_mesh_model = HandMesh(handmesh_model_path)

    t_handMesh = np.asarray(hand_mesh_model.faces)

    # Fallback to cube triangles if model loading failed
    if t_handMesh.size == 0:
        print("Using placeholder cube faces because model faces could not be loaded.")
        t_handMesh = np.array([
            [0, 1, 2], [0, 2, 3], [4, 5, 1], [4, 1, 0],
            [7, 6, 5], [7, 5, 4], [3, 2, 6], [3, 6, 7],
            [1, 5, 6], [1, 6, 2], [4, 0, 3], [4, 3, 7]
        ])

    # --- Mesh Creation ---
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v_handMesh)
    mesh.triangles = o3d.utility.Vector3iVector(t_handMesh)
    mesh.paint_uniform_color([0.8, 0.2, 0.2])
    mesh.compute_vertex_normals()

    # --- Run the application ---
    # Pass the mesh and the desired save path to the transformer
    app = MeshTransformer(mesh, save_path=save_file_path)
    app.run()


if __name__ == "__main__":
    main()