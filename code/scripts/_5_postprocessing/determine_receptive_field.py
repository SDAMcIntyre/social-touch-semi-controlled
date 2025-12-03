import json
import logging
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d  # Added Open3D dependency
from matplotlib.colors import LogNorm
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional, Any
from dataclasses import dataclass, asdict
from collections import Counter

from utils.should_process_task import should_process_task
from preprocessing.forearm_extraction import (
    ForearmFrameParametersFileHandler,
    ForearmParameters,
    ForearmCatalog
)
from primary_processing import (
    KinectConfig
)
from utils.gui.visualize_point_cloud_comparison import visualize_point_cloud_comparison

# Setup module-level logger
logger = logging.getLogger(__name__)

# --- 1. Configuration Layer ---

@dataclass
class ReceptiveFieldConfig:
    """Configuration for receptive field determination."""
    cols_input: dict[str, str] = None
    
    def __post_init__(self):
        if self.cols_input is None:
            self.cols_input = {
                'touch_id': 'single_touch_id',
                'points': 'contact_points',
                'spike': 'Nerve_spike',
            }

    col_output_flag: str = 'on_receptive_field'
    
    output_csv_suffix: str = "_rf_tagged.csv"
    stats_filename: str = "active_point_counts.csv"

# --- 2. Helper Functions ---

def parse_contact_points(point_str: str) -> List[Tuple[float, float, float]]:
    """
    Parses a string representation of 3D points.
    Expected format: "[[x1 y1 z1] [x2 y2 z2]]" or "[]"
    Returns a list of (x, y, z) tuples.
    """
    if pd.isna(point_str) or point_str.strip() == "[]" or not isinstance(point_str, str):
        return []

    points = []
    # Find content inside inner brackets [ ... ]
    matches = re.findall(r'\[([^\]]+)\]', point_str)
    
    for match in matches:
        try:
            # Assuming space-separated values inside the brackets: "4.3 3 34.1"
            parts = match.strip().split()
            if len(parts) == 3:
                # Convert to float tuple for hashability in Sets/Counters
                pt = (float(parts[0]), float(parts[1]), float(parts[2]))
                points.append(pt)
        except ValueError:
            continue
            
    return points

def plot_comparison(point_counts: Counter, forearm_pcd: Any = None, log_scale: bool = True):
    """
    Generates a side-by-side 3D comparison using Open3D:
    Left: The Forearm Point Cloud (Reference).
    Right: The Receptive Field Heatmap (Converted to PointCloud).
    """
    if not point_counts:
        logger.warning("No data available to plot heatmap.")
        return

    # --- 1. Process Receptive Field Data (Right Side) ---
    points = []
    counts = []
    for point, count in point_counts.items():
        points.append([point[0], point[1], point[2]])
        counts.append(count)
    
    if not points:
        return

    points_np = np.asarray(points)
    counts_np = np.asarray(counts)

    # Generate Colors based on Counts (Heatmap)
    # Use Matplotlib to calculate color values, then transfer to Open3D
    norm = LogNorm() if log_scale else plt.Normalize()
    norm.autoscale(counts_np)
    cmap = plt.get_cmap('viridis')
    
    # cmap returns RGBA, we only need RGB. Shape: (N, 3)
    mapped_colors = cmap(norm(counts_np))[:, :3]

    # Create Open3D PointCloud for the Heatmap
    rf_pcd = o3d.geometry.PointCloud()
    rf_pcd.points = o3d.utility.Vector3dVector(points_np)
    rf_pcd.colors = o3d.utility.Vector3dVector(mapped_colors)

    # --- 2. Process Forearm Data (Left Side) ---
    if forearm_pcd:
        # Check if it needs conversion or is already an open3d geometry
        # Assuming forearm_pcd is already an Open3D geometry based on context.
        # If it's a wrapper class, extract the geometry.
        if hasattr(forearm_pcd, 'pcd'): 
             # generic catch if ForearmCatalog returns a wrapper
            left_geometry = forearm_pcd.pcd
        else:
            left_geometry = forearm_pcd
    else:
        # Create an empty placeholder if no forearm is loaded
        left_geometry = o3d.geometry.PointCloud()
        logger.warning("No Forearm Point Cloud Loaded for comparison.")

    # --- 3. Launch Visualization ---
    visualize_point_cloud_comparison(
        pcd_left=left_geometry,
        pcd_right=rf_pcd,
        title="Forearm vs Receptive Field Distribution",
        left_label="Forearm Reference",
        right_label=f"Global Point Distribution (N={len(points)})"
    )

# --- 3. Core Logic ---

def determine_receptive_field(
    input_files: List[Path], 
    arm_roi_metadata_path: Optional[Path], 
    output_dir: Path,
    *,
    force_processing: bool = False,
    monitor: bool = True
) -> Tuple[List[Path], Path]:
    """
    Orchestrates the Receptive Field analysis pipeline.
    1. Aggregates all contact points associated with spikes (Nerve_spike == 1),
       filtering out invalid touches (touch_id == 0).
    2. Aggregates ALL valid contact points to ensure comprehensive statistics.
    3. Tags rows in all files if their contact points intersect with the active set.
    4. Outputs a frequency map of all points, detailing active counts.
    
    Args:
        input_files: List of paths to input CSVs.
        output_dir: Directory to save results.
        arm_roi_metadata_path: Path to the specific *_arm_roi_metadata.json file.
        force_processing: If True, re-runs even if outputs exist.
        monitor: If True, visualizes the result.
    """
    logger.info(f"[{output_dir.name}] Starting Receptive Field analysis on {len(input_files)} files.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    config_rf = ReceptiveFieldConfig()
    
    # Prepare expected outputs for idempotency
    stats_output_path = output_dir / config_rf.stats_filename
    expected_output_files = [output_dir / f"{p.stem}{config_rf.output_csv_suffix}" for p in input_files]
    
    all_check_outputs = expected_output_files + [stats_output_path]
    
    if not should_process_task(
        input_paths=input_files,
        output_paths=all_check_outputs,
        force=force_processing
    ):
        logger.info(f"[{output_dir.name}] Task up-to-date. Skipping.")
        return expected_output_files, output_dir

    # --- Phase 1: Global Aggregation ---
    logger.info("Phase 1: Aggregating contact points...")
    
    active_points_counter = Counter()
    global_points_counter = Counter()
    
    for input_path in input_files:
        try:
            # Load necessary columns
            # Corrected syntax for usecols to list(values())
            df = pd.read_csv(input_path, usecols=list(config_rf.cols_input.values()))
            
            # Extract columns to numpy arrays for performance/indexing
            is_spike = df[config_rf.cols_input["spike"]].to_numpy()
            touch_ids = df[config_rf.cols_input["touch_id"]].to_numpy()
            raw_points_col = df[config_rf.cols_input["points"]]
            
            for idx, raw_points_str in enumerate(raw_points_col):
                # Filter: Skip if touch_id is 0 (Background/Noise)
                # This check happens BEFORE parsing points or checking spikes
                if touch_ids[idx] == 0:
                    continue

                parsed_points = parse_contact_points(raw_points_str)
                if not parsed_points:
                    continue
                
                # Update global counter (valid touches only)
                global_points_counter.update(parsed_points)
                
                # Update active counter (spikes only)
                if is_spike[idx] == 1:
                    active_points_counter.update(parsed_points)
                
        except Exception as e:
            logger.error(f"Error reading {input_path.name} during aggregation: {e}")
            continue

    active_points_set = set(active_points_counter.keys())
    logger.info(f"Found {len(active_points_set)} unique active spatial points out of {len(global_points_counter)} total points.")

    # --- Visualization Block ---
    if monitor:
        logger.info("Monitor enabled: Preparing visualization...")
        
        try:
            # Load parameters
            if arm_roi_metadata_path and arm_roi_metadata_path.exists():
                forearm_params = ForearmFrameParametersFileHandler.load(arm_roi_metadata_path)
                # Instantiate Catalog, the pointcloud directory is assumed to be the parent directory of the metadata file          
                catalog = ForearmCatalog(forearm_params, arm_roi_metadata_path.parent)
                # Use get_first_pointcloud instead of specific video lookup
                forearm_pcd = catalog.get_first_pointcloud()
            else:
                logger.warning("Metadata path invalid or missing. Skipping background loading.")
                forearm_pcd = None
            
            # Plot comparison using the new Open3D implementation
            plot_comparison(global_points_counter, forearm_pcd=forearm_pcd, log_scale=True)
        except Exception as e:
            logger.error(f"Visualization failed: {e}")

    # --- Save Statistics ---
    try:
        stats_data = []
        for pt in global_points_counter.keys():
            stats_data.append({
                'x': pt[0],
                'y': pt[1],
                'z': pt[2],
                'occurrences': active_points_counter[pt],
                'total_occurrences': global_points_counter[pt]
            })
        df_stats = pd.DataFrame(stats_data)
        df_stats.to_csv(stats_output_path, index=False)
        logger.info(f"Saved comprehensive point statistics to {stats_output_path.name}")
    except Exception as e:
        logger.error(f"Failed to save statistics: {e}")

    # --- Phase 2: Tagging and File Generation ---
    logger.info("Phase 2: Tagging files with 'on_receptive_field'...")
    generated_files = []

    for input_path in input_files:
        output_path = output_dir / f"{input_path.stem}{config_rf.output_csv_suffix}"
        
        try:
            df = pd.read_csv(input_path)
            points_col = config_rf.cols_input["points"]
            
            if points_col not in df.columns:
                continue

            def check_intersection(row_str):
                points = parse_contact_points(row_str)
                # Checks if any point in this row exists in the identified active set
                return 1 if any(pt in active_points_set for pt in points) else 0

            # 1. Calculate the receptive field value (0 or 1) based on geometry
            rf_values = df[points_col].apply(check_intersection)

            # 2. Apply validation mask: check for frame_index existence
            if 'frame_index' in df.columns:
                # Convert to object type to support 'None' alongside integers
                rf_values = rf_values.astype(object)
                
                # Identify potential rows: non-NaN frame_index
                # If frame_index is NaN (None), output should be None
                invalid_rows_mask = df['frame_index'].isna()
                rf_values.loc[invalid_rows_mask] = None

            # Assign to the dataframe
            df[config_rf.col_output_flag] = rf_values
            
            df.to_csv(output_path, index=False)
            generated_files.append(output_path)
            
        except Exception as e:
            logger.error(f"Failed to process and save {input_path.name}: {e}")

    return generated_files, output_dir