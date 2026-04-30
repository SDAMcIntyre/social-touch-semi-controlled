# touch_analysis.py
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .preparation.gesture_type import classify_gesture_type
from .representation.series_level.kinematics import compute_velocity, compute_acceleration

# Architectural Import
from utils.should_process_task import should_process_task, clean_task_outputs

def generate_unified_summary(
        input_path: Path,
        output_path: Path,
        show: bool = False,
        force: bool = False,
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
    clean_task_outputs(output_path)
    logging.info(f"Analyzing (Unified): {input_path.name}")
    return _process_touch_analysis(input_path, output_path, show)

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

    if 'block_order_id' not in df.columns:
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

        vel_df = compute_velocity(group)
        accel_df = compute_acceleration(vel_df)
        max_velocity = np.sqrt(vel_df.pow(2).sum(axis=1)).max()
        max_acceleration = np.sqrt(accel_df.pow(2).sum(axis=1)).max()

        gesture_type = classify_gesture_type(group)

        # --- Contact Location (mean per touch) ---
        mean_contact_x = group['contact_location_x'].mean() if 'contact_location_x' in group.columns else None
        mean_contact_y = group['contact_location_y'].mean() if 'contact_location_y' in group.columns else None
        mean_contact_z = group['contact_location_z'].mean() if 'contact_location_z' in group.columns else None

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
            'gesture_type': gesture_type,
            'depth_max': max_depth,
            'area_max': max_contact_area,
            'velocity_max': max_velocity,
            'acceleration_max': max_acceleration,
            'mean_contact_x': mean_contact_x,
            'mean_contact_y': mean_contact_y,
            'mean_contact_z': mean_contact_z,
            'spike_elicited': spike_elicited,
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

    axs[0, 0].scatter(summary_df['depth_max'], summary_df['velocity_max'], alpha=0.6, c='blue')
    axs[0, 0].set_title('Max Velocity vs Max Depth')
    
    type_counts = summary_df['type_metadata'].value_counts()
    type_counts.plot(kind='bar', ax=axs[0, 1], color='orange', alpha=0.7)
    axs[0, 1].set_title('Distribution of Touch Types')

    axs[1, 0].hist(summary_df['acceleration_max'], bins=20, color='green', alpha=0.7)
    axs[1, 0].set_title('Max Acceleration Distribution')

    # Add spike info to the plot if available
    if 'spike_elicited' in summary_df.columns:
        spike_ratio = summary_df['spike_elicited'].mean()
        axs[1, 1].text(0.5, 0.5, f"Spike Ratio:\n{spike_ratio:.2%}", 
                       ha='center', va='center', fontsize=20, transform=axs[1, 1].transAxes)
        axs[1, 1].set_title('Efficacy Summary')
    else:
        axs[1, 1].hist(summary_df['area_max'], bins=20, color='purple', alpha=0.7)
        axs[1, 1].set_title('Max Contact Area Distribution')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show(block=True)