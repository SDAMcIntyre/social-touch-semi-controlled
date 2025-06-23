import open3d as o3d
import numpy as np
import os
import sys

# Attempt to add the necessary paths to import HandMesh
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'imported'))
from imported.hand_mesh import HandMesh



def main():
    """
    Loads a mesh and displays it with keyboard callbacks to correctly flip
    the mesh along the X, Y, and Z axes.
    """
    # --- Path setup (remains the same) ---
    handmesh_path = "F:\\OneDrive - Linköpings universitet\\_Teams\\Social touch Kinect MNG\\data\\semi-controlled\\handmesh_models\\singleFingerVertices0.txt"
    handmesh_model_path = "F:\\GitHub\\social-touch-semi-controlled\\source\\imported\\model\\hand_mesh\\hand_mesh_model.pkl"

    # --- Vertex and Triangle Loading (remains the same) ---
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
        print("Using placeholder cube faces.")
        hand_mesh_model = HandMesh()
    else:
        print(f"Loading hand model from: {handmesh_model_path}")
        hand_mesh_model = HandMesh(handmesh_model_path)

    t_handMesh = np.asarray(hand_mesh_model.faces)

    # --- Mesh Creation ---
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(v_handMesh)
    mesh.triangles = o3d.utility.Vector3iVector(t_handMesh)
    mesh.paint_uniform_color([0.8, 0.2, 0.2])
    mesh.compute_vertex_normals()

    # --- Interactive Visualization with Flipping ---
    flip_status = {'x': False, 'y': False, 'z': False}

    # ================================================================= #
    # ==================== CORRECTED FLIP FUNCTION ==================== #
    # ================================================================= #
    def flip_mesh(mesh_to_flip, axis_index):
        """
        Flips the mesh along a given axis and corrects the triangle winding order.
        """
        # 1. Flip the vertices
        vertices = np.asarray(mesh_to_flip.vertices)
        vertices[:, axis_index] *= -1
        mesh_to_flip.vertices = o3d.utility.Vector3dVector(vertices)

        # 2. Flip the triangle winding order to correct the normals
        triangles = np.asarray(mesh_to_flip.triangles)
        # Swaps the second and third vertex of each triangle (e.g., [0,1,2] -> [0,2,1])
        triangles = triangles[:, [0, 2, 1]]
        mesh_to_flip.triangles = o3d.utility.Vector3iVector(triangles)
        
        # 3. Re-compute normals which will now be correct
        mesh_to_flip.compute_vertex_normals()


    def toggle_flip_x(vis):
        flip_status['x'] = not flip_status['x']
        print(f"Flipping X-axis: {'ON' if flip_status['x'] else 'OFF'}")
        flip_mesh(mesh, 0)
        vis.update_geometry(mesh)

    def toggle_flip_y(vis):
        flip_status['y'] = not flip_status['y']
        print(f"Flipping Y-axis: {'ON' if flip_status['y'] else 'OFF'}")
        flip_mesh(mesh, 1)
        vis.update_geometry(mesh)

    def toggle_flip_z(vis):
        flip_status['z'] = not flip_status['z']
        print(f"Flipping Z-axis: {'ON' if flip_status['z'] else 'OFF'}")
        flip_mesh(mesh, 2)
        vis.update_geometry(mesh)
    
    print("\n--- Interactive Controls ---")
    print("Press 'X' to flip the mesh along the X-axis.")
    print("Press 'Y' to flip the mesh along the Y-axis.")
    print("Press 'Z' to flip the mesh along the Z-axis.")
    print("Close the window to exit.")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Interactive Mesh Viewer", width=1024, height=768)
    vis.add_geometry(mesh)
    
    vis.register_key_callback(ord('X'), toggle_flip_x)
    vis.register_key_callback(ord('Y'), toggle_flip_y)
    vis.register_key_callback(ord('Z'), toggle_flip_z)
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()