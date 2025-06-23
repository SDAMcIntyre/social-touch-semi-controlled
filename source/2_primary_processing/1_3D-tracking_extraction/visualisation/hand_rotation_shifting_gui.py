import open3d as o3d
import numpy as np
import os
import sys

# Adjusting path to import the HandMesh class
# This assumes the script is run from a location where this relative path is valid.
try:
    # Attempt to add the necessary paths to import HandMesh
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'imported'))
    from imported.hand_mesh import HandMesh
except (ImportError, ModuleNotFoundError, NameError):
    print("Warning: Could not import HandMesh class. A placeholder will be used.")
    # Define a placeholder class with simple cube faces if the import fails
    class HandMesh:
        def __init__(self, model_path=None):
            self.faces = np.array([
                [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
                [0, 4, 7], [0, 7, 3], [1, 5, 6], [1, 6, 2],
                [0, 1, 5], [0, 5, 4], [3, 2, 6], [3, 6, 7]
            ])

def main():
    """
    Loads a mesh from specified files (or creates a fallback cube)
    and displays it in an interactive Open3D window with keyboard callbacks
    to flip the mesh along the X, Y, and Z axes.
    """
    # --- IMPORTANT ---
    # You MUST replace the placeholder paths below with the correct absolute
    # paths to your data files on your system.
    handmesh_path = "C:\\Users\\basdu83\\OneDrive - Linköpings universitet\\_Teams\\Social touch Kinect MNG\\data\\semi-controlled\\handmesh_models\\singleFingerVertices0.txt"
    handmesh_model_path = "C:\\Users\\basdu83\\Documents\\GitHub\\social-touch-semi-controlled\\source\\imported\\model\\hand_mesh\\hand_mesh_model.pkl"

    # --- Vertex Loading ---
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

    # --- Triangle/Face Loading ---
    if not os.path.exists(handmesh_model_path):
        print(f"Warning: Hand mesh model not found at '{handmesh_model_path}'")
        print("Using placeholder cube faces.")
        hand_mesh_model = HandMesh()
    else:
        print(f"Loading hand model from: {handmesh_model_path}")
        hand_mesh_model = HandMesh(handmesh_model_path)

    t_handMesh = hand_mesh_model.faces

    # --- Mesh Creation ---
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v_handMesh)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(t_handMesh))
    mesh.paint_uniform_color([0.8, 0.2, 0.2])
    mesh.compute_vertex_normals()

    # --- Interactive Visualization with Flipping ---

    # State variables to track the flip status for each axis
    flip_status = {'x': False, 'y': False, 'z': False}

    def flip_mesh(axis_index):
        """Generic function to flip the mesh along a given axis."""
        vertices = np.asarray(mesh.vertices)
        vertices[:, axis_index] *= -1
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        # Re-compute normals for correct lighting after flipping
        mesh.compute_vertex_normals()

    def toggle_flip_x(vis):
        """Callback function for flipping along the X-axis."""
        flip_status['x'] = not flip_status['x']
        state = "ON" if flip_status['x'] else "OFF"
        print(f"Flipping X-axis: {state}")
        flip_mesh(0)
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()

    def toggle_flip_y(vis):
        """Callback function for flipping along the Y-axis."""
        flip_status['y'] = not flip_status['y']
        state = "ON" if flip_status['y'] else "OFF"
        print(f"Flipping Y-axis: {state}")
        flip_mesh(1)
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()

    def toggle_flip_z(vis):
        """Callback function for flipping along the Z-axis."""
        flip_status['z'] = not flip_status['z']
        state = "ON" if flip_status['z'] else "OFF"
        print(f"Flipping Z-axis: {state}")
        flip_mesh(2)
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()


    print("\n--- Interactive Controls ---")
    print("Press 'X' to flip the mesh along the X-axis.")
    print("Press 'Y' to flip the mesh along the Y-axis.")
    print("Press 'Z' to flip the mesh along the Z-axis.")
    print("Close the window to exit.")

    # Setup the visualizer with key callbacks
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Interactive Mesh Viewer", width=1024, height=768)
    vis.add_geometry(mesh)
    
    # Register the callback functions
    # The key codes for 'x', 'y', 'z' are 88, 89, 90 respectively.
    vis.register_key_callback(88, toggle_flip_x)  # 'X' key
    vis.register_key_callback(89, toggle_flip_y)  # 'Y' key
    vis.register_key_callback(90, toggle_flip_z)  # 'Z' key
    
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()