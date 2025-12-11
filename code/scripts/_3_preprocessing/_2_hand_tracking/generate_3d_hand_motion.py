import os
import pandas as pd
import numpy as np
import json
from pathlib import Path

from utils.should_process_task import should_process_task
from preprocessing.motion_analysis import MeshSequenceLoader, HandMotionManager

def parse_coordinates_from_dataframe(
        df: pd.DataFrame, 
        sticker_names: list[str]) -> np.ndarray:
    """
    Parses a DataFrame and extracts coordinate data for three specified stickers.
    Returns shape: (num_frames, 3 points, 3 coords)
    """
    if len(sticker_names) != 3:
        raise ValueError("Please provide exactly three sticker names.")
    
    df = df.sort_values(by='frame').reset_index(drop=True)
    
    all_coords = []
    for name in sticker_names:
        cols = [f"sticker_{name}_x_mm", f"sticker_{name}_y_mm", f"sticker_{name}_z_mm"]
        if not all(col in df.columns for col in cols):
            raise ValueError(f"DataFrame is missing columns for sticker '{name}'")
        
        sticker_coords = df[cols].to_numpy()
        all_coords.append(sticker_coords)

    return np.stack(all_coords, axis=1)

def generate_hand_motion(
    stickers_path: Path,
    hands_curated_path: Path,
    metadata_path: Path,
    output_glb_path: Path,
    output_csv_path: Path,
    *,
    fps: float = 30.0,
    force_processing: bool = False
):
    """
    Generates and stores a moving 3D hand model.
    Orchestrates data loading and delegates domain logic to HandMotionManager.
    """
    if not should_process_task(
        input_paths=[stickers_path, hands_curated_path, metadata_path], 
        output_paths=[output_csv_path, output_glb_path], 
        force=force_processing):
        print(f"✅ Output files already exist. Use --force to overwrite.")
        return
    
    print("Step 1: Loading input data...")
    try:
        df = pd.read_csv(stickers_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        mesh_loader = MeshSequenceLoader(hands_curated_path)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Prepare Metadata
    sticker_info = sorted(metadata['selected_points'], key=lambda x: x['label'])
    sticker_names = [s['label'].replace('sticker_', '') for s in sticker_info]
    vertex_indices = [s['vertex_id'] for s in sticker_info]
    
    # Extract tracking points
    coordinates_over_time = parse_coordinates_from_dataframe(df, sticker_names)
    num_frames = coordinates_over_time.shape[0]

    # Initialize Manager
    manager = HandMotionManager(fps=fps)
    
    print(f"Step 2: Processing {num_frames} frames via HandMotionManager...")

    for i in range(num_frames):
        timestamp = i / fps
        
        # 1. Get Mesh
        mesh_data = mesh_loader[i]
        if mesh_data is None:
            # If mesh is missing, use the previous valid mesh (or fail if index 0)
            if i == 0:
                raise ValueError("Frame 0 mesh is missing.")
            # Note: HandMotionManager will handle vertex history, 
            # so we could pass explicit None or handle duplication here.
            # Simulating 'hold' by fetching previous from loader is safer if loader supports it,
            # otherwise we rely on loader returning None and handling it here:
            # Implementation Choice: Pass last known good mesh
            mesh_data = mesh_loader[i-1] 
            
        verts = mesh_data['vertices']
        faces = mesh_data['faces']

        # 2. Get Targets
        target_stickers = coordinates_over_time[i]

        # 3. Delegate to Manager
        # The manager handles alignment calculation, NaN checks, and storage
        manager.process_frame(
            mesh_vertices=verts,
            mesh_faces=faces,
            target_sticker_coords=target_stickers,
            sticker_vertex_indices=vertex_indices,
            timestamp=timestamp
        )

    print("Step 3: Saving data...")
    
    # Save GLB
    manager.save(str(output_glb_path))
    
    # Save CSV (Redistribute/Export capabilities from the manager)
    # We access the calculated data back from the manager to ensure consistency
    try:
        motion_df = pd.DataFrame({
            'frames': range(len(manager.timestamps)),
            'timestamp': manager.timestamps,
            'translation_x': [t[0] for t in manager.translations],
            'translation_y': [t[1] for t in manager.translations],
            'translation_z': [t[2] for t in manager.translations],
            'rotation_x': [r[0] for r in manager.rotations],
            'rotation_y': [r[1] for r in manager.rotations],
            'rotation_z': [r[2] for r in manager.rotations],
            'rotation_w': [r[3] for r in manager.rotations]
        })
        motion_df.to_csv(output_csv_path, index=False)
    except Exception as e:
        print(f"⚠️ Error saving CSV: {e}")

    print("✅ Successfully generated animated hand model.")