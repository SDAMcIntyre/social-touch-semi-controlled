import json
import logging
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass
from collections import Counter

# Import the idempotency check utility
from utils.should_process_task import should_process_task

# Setup module-level logger
logger = logging.getLogger(__name__)

# --- 1. Configuration Layer ---

@dataclass(frozen=True)
class ReceptiveFieldConfig:
    """Immutable configuration for receptive field determination."""
    col_contact_points: str = 'contact_points'
    col_nerve_spike: str = 'Nerve_spike'
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
    # This regex looks for literal '[' followed by non-']' chars, then ']'
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

def plot_3d_heatmap(point_counts: Counter, log_scale: bool = True):
    """
    Generates a 3D scatter plot of points colored by their frequency.
    """
    if not point_counts:
        logger.warning("No data available to plot.")
        return

    # Unpack data
    # points keys are (x, y, z), values are counts
    xs, ys, zs = [], [], []
    counts = []

    for point, count in point_counts.items():
        xs.append(point[0])
        ys.append(point[1])
        zs.append(point[2])
        counts.append(count)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Determine normalization (Log or Linear)
    norm = LogNorm() if log_scale else None

    # Plot
    img = ax.scatter(xs, ys, zs, c=counts, cmap='viridis', norm=norm, marker='o', s=10, alpha=0.6)
    
    # Labels and Colorbar
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title(f'Global Point Distribution (N={sum(counts)})')
    
    cbar = plt.colorbar(img, ax=ax, pad=0.1)
    cbar.set_label('Count (Log Scale)' if log_scale else 'Count')

    plt.show(block=True)

# --- 3. Core Logic ---

def determine_receptive_field(
    input_files: List[Path], 
    output_dir: Path,
    *,
    force_processing: bool = False,
    monitor: bool = True
) -> Tuple[List[Path], Path]:
    """
    Orchestrates the Receptive Field analysis pipeline.
    1. Aggregates all contact points associated with spikes (Nerve_spike == 1).
    2. Aggregates ALL contact points to ensure comprehensive statistics.
    3. Tags rows in all files if their contact points intersect with the active set.
    4. Outputs a frequency map of all points, detailing active counts.
    """
    logger.info(f"[{output_dir.name}] Starting Receptive Field analysis on {len(input_files)} files.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ReceptiveFieldConfig()
    
    # Prepare expected outputs for idempotency
    stats_output_path = output_dir / config.stats_filename
    expected_output_files = [output_dir / f"{p.stem}{config.output_csv_suffix}" for p in input_files]
    
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
    
    # Counter for ONLY spikes (used for receptive field definition)
    active_points_counter = Counter()
    
    # Counter for ALL points (used for visualization and universe definition)
    # We now populate this regardless of monitor status to ensure the stats file is complete
    global_points_counter = Counter()
    
    for input_path in input_files:
        try:
            # Load necessary columns
            df = pd.read_csv(input_path, usecols=[config.col_contact_points, config.col_nerve_spike])
            
            # Optimization: Parse points once per row
            # Extract spike status as a numpy array for faster indexing
            is_spike = df[config.col_nerve_spike].to_numpy()
            
            # Iterate through rows and update counters
            # Using zip allows us to iterate the parsed lists alongside the spike status
            # This ensures we identify 'all points' even if they never spike.
            for idx, raw_points_str in enumerate(df[config.col_contact_points]):
                parsed_points = parse_contact_points(raw_points_str)
                
                if not parsed_points:
                    continue
                
                # Update Global Counter (All points seen)
                global_points_counter.update(parsed_points)
                
                # Update Active Counter (Only if spike == 1)
                if is_spike[idx] == 1:
                    active_points_counter.update(parsed_points)
                
        except Exception as e:
            logger.error(f"Error reading {input_path.name} during aggregation: {e}")
            continue

    # Create a Set for O(1) lookup during the tagging phase (Active points only)
    active_points_set = set(active_points_counter.keys())
    logger.info(f"Found {len(active_points_set)} unique active spatial points out of {len(global_points_counter)} total points.")

    # --- Visualization Block ---
    if monitor:
        logger.info("Monitor enabled: Generating 3D distribution plot of all points...")
        try:
            plot_3d_heatmap(global_points_counter, log_scale=True)
        except Exception as e:
            logger.error(f"Failed to generate monitor plot: {e}")

    # Save Statistics (Point Occurrences for Spikes + Non-spiking points)
    # Modification: Iterate over global_points_counter keys to include all points
    try:
        stats_data = []
        for pt in global_points_counter.keys():
            stats_data.append({
                'x': pt[0],
                'y': pt[1],
                'z': pt[2],
                'occurrences': active_points_counter[pt],      # Active/Spike Count (0 if never spiked)
                'total_occurrences': global_points_counter[pt] # Total times point appeared in data
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
        output_path = output_dir / f"{input_path.stem}{config.output_csv_suffix}"
        
        try:
            # Read full file
            df = pd.read_csv(input_path)
            
            if config.col_contact_points not in df.columns:
                logger.warning(f"Skipping {input_path.name}: '{config.col_contact_points}' column missing.")
                continue

            # Function to determine if a row is on the receptive field
            def check_receptive_field(row_str):
                points = parse_contact_points(row_str)
                # Check intersection: if any point in this row is in the active set
                return 1 if any(pt in active_points_set for pt in points) else 0

            # Apply logic
            df[config.col_output_flag] = df[config.col_contact_points].apply(check_receptive_field)
            
            # Save
            df.to_csv(output_path, index=False)
            generated_files.append(output_path)
            
        except Exception as e:
            logger.error(f"Failed to process and save {input_path.name}: {e}")

    return generated_files, output_dir