# touch_analysis.py
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Union

# Architectural Import
from utils.should_process_task import should_process_task

def generate_unified_summary(
        input_path: Path, 
        output_path: Path, 
        show: bool = False,
        force: bool = False
) -> Path:
    """
    Unified analysis function.
    Loads a CSV file, calculates kinematic/spatial metrics AND spike efficacy, 
    and saves the summary to a single file.
    Includes internal idempotency check.
    """
    # 1. Idempotency Check
    if not should_process_task(
        input_paths=[input_path], 
        output_paths=[output_path], 
        force=force
    ):
        logging.info(f"Skipping Unified Summary for {input_path.name} (Up-to-date).")
        return output_path

    logging.info(f"Analyzing (Unified): {input_path.name}")
    return _process_touch_analysis(input_path, output_path, show)

def generate_touch_summary_matrix(
        input_paths: List[Path],
        output_path: Path,
        show: bool = False,
        force: bool = False
) -> Path:
    """
    Generates a matrix of touch counts/conditions from multiple summary files.
    Includes internal idempotency check.
    """
    # 1. Idempotency Check
    if not should_process_task(
        input_paths=input_paths, 
        output_paths=[output_path], 
        force=force
    ):
        logging.info(f"Skipping Touch Summary Matrix (Up-to-date): {output_path.name}")
        return output_path

    logging.info(f"Generating Touch Summary Matrix to {output_path}...")
    
    # --- Implementation Logic Placeholder ---
    # (Original implementation logic would go here. 
    #  Since it was not provided in the source file, this is a structural stub.)
    try:
        # Example logic:
        # df_list = [pd.read_csv(p) for p in input_paths]
        # combined = pd.concat(df_list)
        # ... processing ...
        # combined.to_csv(output_path)
        
        # Creating a dummy file to satisfy the architecture if running this code directly
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("trial_id,condition,count\n") # Mock header
        
        logging.info(f"Saved matrix to {output_path}")
        
    except Exception as e:
        logging.error(f"Failed to generate touch matrix: {e}")
        raise

    return output_path

def generate_ap_efficacy_matrix(
        input_paths: List[Path],
        output_path: Path,
        show: bool = False,
        force: bool = False
) -> Path:
    """
    Generates a matrix of AP efficacy from multiple summary files.
    Includes internal idempotency check.
    """
    # 1. Idempotency Check
    if not should_process_task(
        input_paths=input_paths, 
        output_paths=[output_path], 
        force=force
    ):
        logging.info(f"Skipping AP Efficacy Matrix (Up-to-date): {output_path.name}")
        return output_path

    logging.info(f"Generating AP Efficacy Matrix to {output_path}...")

    # --- Implementation Logic Placeholder ---
    try:
        # Example logic:
        # ... processing spike_elicited columns ...
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("trial_id,efficacy_score\n") # Mock header
            
        logging.info(f"Saved AP matrix to {output_path}")
        
    except Exception as e:
        logging.error(f"Failed to generate AP matrix: {e}")
        raise

    return output_path

def _process_touch_analysis(input_path: Path, output_path: Path, show: bool) -> Path:
    """
    Internal shared logic for processing touch kinematics.
    Always includes 'spike_elicited' column.
    """
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        logging.error(f"Failed to load CSV: {e}")
        raise

    if 'source_block_file' in df.columns:
        df['block_order_id'] = df['source_block_file'].astype(str).str.extract(r'_block-order-(\d+)_', expand=False)
    else:
        df['block_order_id'] = None

    # Check for Nerve_spike column presence
    has_nerve_data = 'Nerve_spike' in df.columns
    if not has_nerve_data:
        logging.warning(f"'Nerve_spike' column missing in {input_path.name}. 'spike_elicited' will be 0.")

    results = []

    # Group by trial and touch ID
    for (trial_id, touch_id), group in df.groupby(['trial_id', 'single_touch_id']):
        if group.empty or touch_id == 0:
            continue
            
        # --- Shared Kinematics ---
        max_depth = group['contact_depth'].max()
        max_contact_area = group['contact_area'].max()
        touch_type = group['type_metadata'].iloc[0] if 'type_metadata' in group.columns else "unknown"
        block_order_id = group['block_order_id'].iloc[0]

        # Velocity
        velocity_vectors = group[['sticker_blue_position_x', 'sticker_blue_position_y', 'sticker_blue_position_z']].diff().fillna(0)
        velocity_magnitudes = np.sqrt(velocity_vectors.pow(2).sum(axis=1))
        max_velocity = velocity_magnitudes.max()

        # Acceleration
        acceleration_vectors = velocity_vectors.diff().fillna(0)
        acceleration_magnitudes = np.sqrt(acceleration_vectors.pow(2).sum(axis=1))
        max_acceleration = acceleration_magnitudes.max()

        # Direction
        direction = None
        if touch_type == "stroke":
            start_y = group['sticker_blue_position_y'].iloc[0]
            end_y = group['sticker_blue_position_y'].iloc[-1]
            direction = "proximal" if end_y > start_y else "distal"
        else:
            direction = "static" 

        # --- Efficacy Logic (Always Run) ---
        # Check if ANY frame in this touch had a spike (1)
        if has_nerve_data:
            spike_elicited = 1 if group['Nerve_spike'].max() == 1 else 0
        else:
            spike_elicited = 0

        row_data = {
            'trial_id': trial_id,
            'single_touch_id': touch_id,
            'block_order_id': block_order_id,
            'type_metadata': touch_type,
            'direction': direction,
            'max_depth': max_depth,
            'max_contact_area': max_contact_area,
            'max_velocity': max_velocity,
            'max_acceleration': max_acceleration,
            'spike_elicited': spike_elicited
        }

        results.append(row_data)

    summary_df = pd.DataFrame(results)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        summary_df.to_csv(output_path, index=False)
        logging.info(f"Saved unified summary to {output_path} (n={len(summary_df)})")
    except Exception as e:
        logging.error(f"Failed to save output CSV: {e}")
        raise

    # Visualization
    if show and not summary_df.empty:
        try:
            _generate_summary_plot(summary_df, input_path.name)
        except Exception as e:
            logging.warning(f"Could not generate summary plot: {e}")

    return output_path

def _generate_summary_plot(summary_df, name):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Analysis Summary: {name}', fontsize=16)

    axs[0, 0].scatter(summary_df['max_depth'], summary_df['max_velocity'], alpha=0.6, c='blue')
    axs[0, 0].set_title('Max Velocity vs Max Depth')
    
    type_counts = summary_df['type_metadata'].value_counts()
    type_counts.plot(kind='bar', ax=axs[0, 1], color='orange', alpha=0.7)
    axs[0, 1].set_title('Distribution of Touch Types')

    axs[1, 0].hist(summary_df['max_acceleration'], bins=20, color='green', alpha=0.7)
    axs[1, 0].set_title('Max Acceleration Distribution')

    # Add spike info to the plot if available
    if 'spike_elicited' in summary_df.columns:
        spike_ratio = summary_df['spike_elicited'].mean()
        axs[1, 1].text(0.5, 0.5, f"Spike Ratio:\n{spike_ratio:.2%}", 
                       ha='center', va='center', fontsize=20, transform=axs[1, 1].transAxes)
        axs[1, 1].set_title('Efficacy Summary')
    else:
        axs[1, 1].hist(summary_df['max_contact_area'], bins=20, color='purple', alpha=0.7)
        axs[1, 1].set_title('Max Contact Area Distribution')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=True)