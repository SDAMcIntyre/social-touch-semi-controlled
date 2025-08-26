import os
import pandas as pd
import numpy as np
import trimesh
import json
from pathlib import Path

from preprocessing.motion_analysis.hand_tracking.hand_mesh_processor import HandMeshProcessor
from preprocessing.motion_analysis.common.glb_data_handler import GLBDataHandler

def _calculate_rotation_from_planes(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """
    Calculates the 3x3 rotation matrix to align the plane of source_points to target_points.

    It creates a unique coordinate system (orthonormal basis) from the three points
    for both the source and target, and then finds the rotation between these two systems.
    """
    def create_basis(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Helper to create an orthonormal basis from three points."""
        v1 = p1 - p0
        v2 = p2 - p0
        
        # Normalize the first vector to get the x-axis
        x_axis = v1 / np.linalg.norm(v1)
        
        # Get the z-axis from the cross product, which is orthogonal to the plane
        z_axis_unnorm = np.cross(x_axis, v2)
        if np.linalg.norm(z_axis_unnorm) < 1e-6: # Fallback for collinear points
            temp_vec = np.array([0, 1, 0]) if np.allclose(np.abs(x_axis), [1, 0, 0]) else np.array([1, 0, 0])
            z_axis_unnorm = np.cross(x_axis, temp_vec)
        z_axis = z_axis_unnorm / np.linalg.norm(z_axis_unnorm)
        
        # Get the y-axis to complete the right-handed coordinate system
        y_axis = np.cross(z_axis, x_axis)
        
        return np.column_stack([x_axis, y_axis, z_axis])

    # Create a basis for both the source and target point configurations
    source_basis = create_basis(source_points[0], source_points[1], source_points[2])
    target_basis = create_basis(target_points[0], target_points[1], target_points[2])
    
    # The rotation R that maps the source basis to the target basis is R = target_basis @ source_basis.T
    rotation_matrix = target_basis @ source_basis.T
    return rotation_matrix

def _calculate_translation_for_origin(source_origin: np.ndarray, target_origin: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Calculates the translation vector to align a rotated source origin to a target origin.
    
    The goal is to find the translation 't' such that: target_origin = (R @ source_origin) + t.
    """
    # First, apply the rotation to the source origin point
    rotated_source_origin = rotation_matrix @ source_origin
    # The required translation is the vector from the rotated point to the target point
    translation_vector = target_origin - rotated_source_origin
    return translation_vector

def calculate_prioritized_rigid_transformation(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """
    Calculates a 4x4 rigid transformation matrix with an explicit, prioritized alignment.

    This process is broken down into two distinct steps:
    1.  **Planar Alignment (Rotation):** The rotation is calculated first by aligning the
        orientation of the plane defined by the three source points to the target plane.
    2.  **Origin Alignment (Translation):** The translation is then calculated to precisely
        move the first source point (the "origin") to the position of the first target point.
    """
    if source_points.shape != (3, 3) or target_points.shape != (3, 3):
        raise ValueError("Input point clouds must be of shape (3, 3)")

    # --- Step 1: Calculate rotation from the planar alignment of all three points ---
    rotation = _calculate_rotation_from_planes(source_points, target_points)

    # --- Step 2: Calculate translation to shift the rotated origin to the target origin ---
    # Explicitly define which points serve which purpose
    source_origin = source_points[0]
    target_origin = target_points[0]
    translation = _calculate_translation_for_origin(source_origin, target_origin, rotation)

    # --- Step 3: Assemble the final 4x4 transformation matrix ---
    transform_matrix = np.eye(4)
    transform_matrix[0:3, 0:3] = rotation
    transform_matrix[0:3, 3] = translation
    
    return transform_matrix

# This function remains unchanged.
def parse_coordinates_from_dataframe(df: pd.DataFrame, sticker_names: list[str]) -> np.ndarray:
    """
    Parses a DataFrame and extracts coordinate data for three specified stickers,
    formatting it into a NumPy array suitable for the animation function.
    """
    if len(sticker_names) != 3:
        raise ValueError("Please provide exactly three sticker names.")
    
    df = df.sort_values(by='frame').reset_index(drop=True)
    
    all_coords = []
    for name in sticker_names:
        cols = [f"sticker_{name}_x_mm", f"sticker_{name}_y_mm", f"sticker_{name}_z_mm"]
        if not all(col in df.columns for col in cols):
            raise ValueError(f"DataFrame is missing one or more required columns for sticker '{name}': {cols}")
        
        sticker_coords = df[cols].to_numpy()
        all_coords.append(sticker_coords)

    # Shape: (num_frames, 3 points, 3 coords)
    return np.stack(all_coords, axis=1)


def glb_input_is_clean(data_to_check):
    is_clean = True
    print("Checking for invalid float values (NaN or infinity)...")
    for name, arr in data_to_check.items():
        if not np.all(np.isfinite(arr)):
            print(f"❌ Found invalid values in '{name}' array!")
            is_clean = False

    return is_clean


# This function remains unchanged except for the call to the new transformation function.
def generate_hand_motion(
    stickers_path: str,
    hand_models_dir: str,
    metadata_path: str,
    output_glb_path: str,
    output_csv_path: str,
    *,
    fps: int = 30
):
    """
    Generates and stores a moving 3D hand model as an animated glTF/GLB file
    and saves the motion data to a CSV file.

    Args:
        stickers_path (str): Path to the CSV file with sticker coordinates over time.
        hand_models_dir (str): Path to the directory containing hand model files.
        metadata_path (str): Path to the JSON file containing vertex IDs for the stickers.
        output_glb_path (str): Path to save the output animated .glb file.
        output_csv_path (str): Path to save the output translation and rotation data as a .csv file.
        fps (int): Frames per second for the animation.
    """
    if os.path.exists(output_glb_path) and os.path.exists(output_csv_path):
        print("hand motion has been processed already, skipping...")
        return output_glb_path, output_csv_path
    
    
    # 1. Load All Input Data
    print("Step 1: Loading input data...")
    try:
        df = pd.read_csv(stickers_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        template_mesh_path = Path(hand_models_dir) / metadata["selected_hand_model_name"]
        mesh_params = {'left': (metadata["hand_orientation"] == "left")}
        
        # Note: HandMeshProcessor seems to be a custom utility. Assuming it works as intended.
        # This part remains dependent on your local 'HandMeshProcessor' implementation.
        o3d_mesh = HandMeshProcessor.create_mesh(str(template_mesh_path), mesh_params, SCALE_M_TO_MM=True)
        mesh = trimesh.Trimesh(np.asarray(o3d_mesh.vertices), np.asarray(o3d_mesh.triangles))
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return
    
    # 2. Extract and Organize Data
    print("Step 2: Organizing tracking data...")
    sticker_info = sorted(metadata['selected_points'], key=lambda x: x['label'])
    sticker_names = [s['label'].replace('sticker_', '') for s in sticker_info]
    vertex_indices = [s['vertex_id'] for s in sticker_info]
    source_points = mesh.vertices[vertex_indices]
    coordinates_over_time = parse_coordinates_from_dataframe(df, sticker_names)
    num_frames = coordinates_over_time.shape[0]

    # 3. Generate Animation Keyframes
    print(f"Step 3: Calculating transformations for {num_frames} frames...")
    timestamps = np.array([i / fps for i in range(num_frames)], dtype=np.float32)
    translations = np.zeros((num_frames, 3), dtype=np.float32)
    rotations = np.zeros((num_frames, 4), dtype=np.float32) # For quaternions (x, y, z, w)

    for i in range(num_frames):
        target_points = coordinates_over_time[i]
        transform_matrix = calculate_prioritized_rigid_transformation(source_points, target_points)
        
        # Decompose matrix to get translation and quaternion rotation
        scale, shear, angles, trans, persp = trimesh.transformations.decompose_matrix(transform_matrix)
        quat = trimesh.transformations.quaternion_from_euler(*angles) # Returns (w, x, y, z)
        
        translations[i] = trans
        # glTF expects quaternions in (x, y, z, w) format. np.roll shifts 'w' to the end.
        rotations[i] = np.roll(quat, -1) 
    
    # 4. Save Transformation Data to CSV (NEW STEP)
    print(f"Step 4: Saving transformation data to {output_csv_path}...")
    try:
        motion_df = pd.DataFrame({
            'frames': range(num_frames),
            'timestamp': timestamps,
            'translation_x': translations[:, 0],
            'translation_y': translations[:, 1],
            'translation_z': translations[:, 2],
            'rotation_x': rotations[:, 0],
            'rotation_y': rotations[:, 1],
            'rotation_z': rotations[:, 2],
            'rotation_w': rotations[:, 3]
        })
        motion_df.to_csv(output_csv_path, index=False)
        print("✅ Successfully saved transformation data.")
    except Exception as e:
        print(f"An error occurred while saving the CSV file: {e}")
        # Continue to glb creation even if CSV saving fails
    
    # 5. Prepare Raw Binary Data for the glTF Buffer
    print("Step 5: Building glTF data buffers...")

    vertex_positions = np.array(mesh.vertices, dtype=np.float32)
    face_indices = np.array(mesh.faces, dtype=np.uint32)
    
    # Your data arrays
    data_to_check = {
        "vertices": vertex_positions,
        "timestamps": timestamps,
        "translations": translations,
        "rotations": rotations
    }

    if not glb_input_is_clean(data_to_check):
        # --- Clean the data before saving ---
        # Apply this to all float arrays that might be affected.
        vertex_positions = np.nan_to_num(vertex_positions)
        translations = np.nan_to_num(translations)
        rotations = np.nan_to_num(rotations)
        timestamps = np.nan_to_num(timestamps)

    saver = GLBDataHandler()
    saver.save(
        output_path=output_glb_path,
        vertices=vertex_positions,
        faces=face_indices,
        time_points=timestamps,
        translations=translations,
        rotations=rotations
    )
    
    print("✅ Successfully saved animated hand model.")