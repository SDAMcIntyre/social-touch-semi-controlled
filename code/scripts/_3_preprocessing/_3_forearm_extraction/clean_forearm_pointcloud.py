import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Optional, Union

from .visualize_point_cloud_comparison import visualize_point_cloud_comparison

# --- Main Processing Function ---

def clean_forearm_pointcloud(
    input_ply_path: Union[str, Path], 
    output_ply_path: Union[str, Path], 
    output_metadata_path: Optional[Union[str, Path]] = None, 
    *,
    interactive: bool = False
) -> None:
    """
    Loads a pointcloud, filters points with duplicate X,Y coordinates by keeping 
    the one with the lowest Z value, and saves the result. 
    Delegates visualization to `visualize_point_cloud_comparison`.

    Args:
        input_ply_path (Path): Path to the source .ply file.
        output_ply_path (Path): Path where the cleaned .ply file will be saved.
        output_metadata_path (Path, optional): Path to save cleaning statistics.
        interactive (bool): If True, opens a synchronized split-window comparison.
    """
    input_path = Path(input_ply_path)
    output_path = Path(output_ply_path)
    
    # 1. Load the Point Cloud
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    pcd = o3d.io.read_point_cloud(str(input_path))
    
    if not pcd.has_points():
        print(f"⚠️ Warning: Point cloud at {input_path} is empty.")
        o3d.io.write_point_cloud(str(output_path), pcd)
        return

    # 2. Convert to DataFrame for efficient GroupBy operations
    points = np.asarray(pcd.points)
    
    # Create DataFrame
    data = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
    df = pd.DataFrame(data)
    
    # 3. Filtering Logic
    # Round coordinates to handle floating point jitter
    df['x_round'] = df['x'].round(6)
    df['y_round'] = df['y'].round(6)

    initial_count = len(df)
    
    # Sort by Z (ascending), then drop duplicates keeping the first (lowest Z)
    # This logic assumes the 'forearm' scan surface is best represented by the 'lowest' Z 
    # relative to the scanner origin.
    cleaned_df_indices = df.sort_values('z').drop_duplicates(subset=['x_round', 'y_round'], keep='first').index
    
    # 4. Select points from original PCD to preserve color/normals
    cleaned_pcd = pcd.select_by_index(cleaned_df_indices)
    final_count = len(cleaned_df_indices)
    reduction = initial_count - final_count
    
    # 5. Save Output
    o3d.io.write_point_cloud(str(output_path), cleaned_pcd)
    print(f"✅ Cleaned Point Cloud saved to {output_path}")
    print(f"   Points reduced from {initial_count} to {final_count} (Removed {reduction})")
    
    # 6. Save Metadata (Optional)
    if output_metadata_path:
        stats = {
            "input_file": str(input_path.name),
            "initial_points": int(initial_count),
            "final_points": int(final_count),
            "points_removed": int(reduction),
            "reduction_percentage": round((reduction / initial_count) * 100, 2) if initial_count > 0 else 0
        }
        with open(output_metadata_path, 'w') as f:
            json.dump(stats, f, indent=4)

    # 7. Interactive Visualization (Delegated)
    if interactive:
        visualize_point_cloud_comparison(
            pcd_left=pcd,
            pcd_right=cleaned_pcd,
            title="Forearm Cleaning Result",
            left_label="Raw Input",
            right_label="Cleaned Output"
        )
