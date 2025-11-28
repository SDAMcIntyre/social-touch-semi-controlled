import numpy as np
from scipy.spatial import Delaunay
import trimesh
import open3d as o3d
import os
from pathlib import Path
from typing import Union, Optional
import sys

# Assume it exists
import utils.path_tools as path_tools

def trimesh_to_open3d(src_mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """
    Helper to convert a Trimesh object to an Open3D TriangleMesh for visualization.
    """
    dst_mesh = o3d.geometry.TriangleMesh()
    dst_mesh.vertices = o3d.utility.Vector3dVector(src_mesh.vertices)
    dst_mesh.triangles = o3d.utility.Vector3iVector(src_mesh.faces)
    
    # Transfer vertex colors if they exist
    if src_mesh.visual.kind == 'vertex' and src_mesh.visual.vertex_colors is not None:
        # Trimesh colors are usually uint8 (0-255), Open3D expects float (0.0-1.0)
        colors = src_mesh.visual.vertex_colors[:, :3] / 255.0
        dst_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        
    # Transfer normals if they exist
    if 'vertex_normals' in src_mesh._cache:
         dst_mesh.vertex_normals = o3d.utility.Vector3dVector(src_mesh.vertex_normals)
    else:
        dst_mesh.compute_vertex_normals()
        
    return dst_mesh

def define_forearm_mesh(source: Union[np.ndarray, str, Path], 
                                output_path: Optional[Union[str, Path]] = None, 
                                show: bool = False) -> trimesh.Trimesh:
    """
    Converts a topographical point cloud to a mesh using 2.5D Delaunay Triangulation.
    
    Architectural Updates:
    1. Loads data using Open3D.
    2. Uses SciPy for Delaunay Triangulation.
    3. Uses Trimesh for mesh repair and normal consistency.
    4. Visualization logic migrated to Open3D to fix COM MTA errors and blank GUI on Windows.
    5. Added persistence logic to save the mesh to `output_path`.
    """
    # 1. Input Parsing and Data Loading via Open3D
    points = None
    input_normals = None
    input_colors = None
    
    if isinstance(source, (str, Path)):
        file_path = str(source)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at path {file_path} does not exist.")
        
        print(f"Loading point cloud using Open3D from: {file_path}")
        pcd = o3d.io.read_point_cloud(file_path)
        
        if pcd.is_empty():
            raise ValueError(f"Open3D failed to load data or file is empty: {file_path}")
            
        points = np.asarray(pcd.points)
        if pcd.has_normals():
            input_normals = np.asarray(pcd.normals)
        if pcd.has_colors():
            print("Color data detected in point cloud. Propagating to mesh...")
            input_colors = np.asarray(pcd.colors)
            
    elif isinstance(source, np.ndarray):
        points = source
    else:
        raise TypeError(f"Source must be a file path (str/Path) or a numpy array. Received: {type(source)}")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must be an (N, 3) array.")

    # 2. Project to 2D (XY plane) for 2.5D Triangulation
    xy_points = points[:, 0:2]
    
    # 3. Perform Delaunay Triangulation
    print("Performing Delaunay Triangulation...")
    tri = Delaunay(xy_points)
    
    # 4. Create the Mesh using Trimesh
    mesh = trimesh.Trimesh(vertices=points,
                           faces=tri.simplices,
                           vertex_colors=input_colors)
    
    # 5. Normal-Based Auto-Correction
    if input_normals is not None and len(input_normals) == len(points):
        print("Input normals detected. Attempting to align mesh orientation...")
        generated_face_normals = mesh.face_normals
        faces = mesh.faces
        face_vertex_normals = input_normals[faces]
        avg_input_normals = face_vertex_normals.mean(axis=1)
        dots = np.einsum('ij,ij->i', generated_face_normals, avg_input_normals)
        
        if np.sum(dots < 0) > (len(dots) / 2):
            print("Detected inverted winding. Flipping mesh...")
            mesh.invert()
    else:
        # Fallback: Z-up assumption
        if np.mean(mesh.face_normals[:, 2]) < 0:
             print("Mesh appears to face downwards (Z-). Flipping to Z+ assumption.")
             mesh.invert()

    mesh.fix_normals()

    # 6. Save Mesh (Persistence)
    if output_path:
        out = Path(output_path)
        try:
            # Ensure the parent directory exists
            out.parent.mkdir(parents=True, exist_ok=True)
            print(f"Saving generated mesh to: {out.resolve()}")
            # Trimesh infers format from extension (e.g., .ply, .obj, .stl)
            mesh.export(str(out))
        except Exception as e:
            print(f"Failed to save mesh to {out}: {e}")
    
    # 7. Visualization (Swapped to Open3D to fix Blank Screen/COM issues)
    if show:
        print("\n--- Viewer Controls ---")
        print("Press 'F' to Flip mesh orientation (invert normals).")
        print("Press 'W' to toggle Wireframe (if supported by shader).")
        print("Close window to continue.")
        print("-----------------------")

        # Convert Trimesh to Open3D Mesh for visualization
        o3d_mesh = trimesh_to_open3d(mesh)
        o3d_mesh.compute_vertex_normals()

        # Define callbacks
        def flip_mesh(vis):
            print("User requested Flip via 'F' key...")
            # Invert triangles winding order
            tris = np.asarray(o3d_mesh.triangles)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(tris[:, [0, 2, 1]])
            o3d_mesh.compute_vertex_normals()
            vis.update_geometry(o3d_mesh)
            return False

        # Initialize Visualizer with Key Callbacks
        key_to_callback = {}
        # GLFW key code for 'F' is 70
        key_to_callback[70] = flip_mesh

        try:
            o3d.visualization.draw_geometries_with_key_callbacks(
                [o3d_mesh], 
                key_to_callback,
                window_name="Terrain Mesh Viewer",
                width=1280,
                height=720,
                left=50,
                top=50
            )
        except Exception as e:
            print(f"Open3D visualization failed: {e}")
    
    return mesh

# --- Verification Logic ---

if __name__ == "__main__":
    project_data_root = path_tools.get_project_data_root()
    data_ply_dir = project_data_root / "2_processed/kinect/2022-06-17_ST16-05/forearm_pointclouds"
    data_ply_path = data_ply_dir / "2022-06-17_ST16-05_semicontrolled_block-order01_kinect_frame_0007_with_normals.ply"
    
    # Define an output path for verification
    output_mesh_path = data_ply_dir / "generated_terrain_mesh.obj"

    print(f"Processing file: {data_ply_path}")
    # Call function with output_path parameter
    terrain_mesh = point_cloud_to_terrain_mesh(data_ply_path, output_path=output_mesh_path, show=True)
    
    print(f"Final Mesh Stats: {len(terrain_mesh.vertices)} vertices, {len(terrain_mesh.faces)} faces.")
