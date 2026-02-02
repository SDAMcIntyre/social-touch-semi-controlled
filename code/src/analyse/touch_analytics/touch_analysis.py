# touch_analysis.py
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyse_number_single_touches(
        input_path: Path, 
        output_path: Path, 
        show: bool = False
) -> Path:
    """
    Loads a CSV file, calculates kinematic/spatial metrics, and saves the summary.
    """
    logging.info(f"Analyzing: {input_path.name}")

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        logging.error(f"Failed to load CSV: {e}")
        raise

    if 'source_block_file' in df.columns:
        df['block_order_id'] = df['source_block_file'].astype(str).str.extract(r'_block-order-(\d+)_', expand=False)
    else:
        logging.warning("'source_block_file' column missing. block_order_id will be None.")
        df['block_order_id'] = None

    results = []

    for (trial_id, touch_id), group in df.groupby(['trial_id', 'single_touch_id']):
        if group.empty or touch_id == 0:
            continue
            
        max_depth = group['contact_depth'].max()
        max_contact_area = group['contact_area'].max()
        
        touch_type = group['type_metadata'].iloc[0] if 'type_metadata' in group.columns else "unknown"
        block_order_id = group['block_order_id'].iloc[0]

        # Calculate Velocity
        velocity_vectors = group[[
            'sticker_blue_position_x', 
            'sticker_blue_position_y', 
            'sticker_blue_position_z'
        ]].diff().fillna(0)
        
        velocity_magnitudes = np.sqrt(
            velocity_vectors['sticker_blue_position_x']**2 + 
            velocity_vectors['sticker_blue_position_y']**2 + 
            velocity_vectors['sticker_blue_position_z']**2
        )
        max_velocity = velocity_magnitudes.max()

        # Calculate Acceleration
        acceleration_vectors = velocity_vectors.diff().fillna(0)
        
        acceleration_magnitudes = np.sqrt(
            acceleration_vectors['sticker_blue_position_x']**2 + 
            acceleration_vectors['sticker_blue_position_y']**2 + 
            acceleration_vectors['sticker_blue_position_z']**2
        )
        max_acceleration = acceleration_magnitudes.max()

        # Determine Direction
        direction = None
        if touch_type == "stroke":
            start_y = group['sticker_blue_position_y'].iloc[0]
            end_y = group['sticker_blue_position_y'].iloc[-1]
            if end_y > start_y:
                direction = "proximal"
            else:
                direction = "distal"
        else:
            direction = "static" 

        results.append({
            'trial_id': trial_id,
            'single_touch_id': touch_id,
            'block_order_id': block_order_id,
            'type_metadata': touch_type,
            'direction': direction,
            'max_depth': max_depth,
            'max_contact_area': max_contact_area,
            'max_velocity': max_velocity,
            'max_acceleration': max_acceleration
        })

    summary_df = pd.DataFrame(results)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        summary_df.to_csv(output_path, index=False)
        logging.info(f"Summary saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save output CSV: {e}")
        raise

    if show and not summary_df.empty:
        try:
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Analysis Summary: {input_path.name}', fontsize=16)

            axs[0, 0].scatter(summary_df['max_depth'], summary_df['max_velocity'], alpha=0.6, c='blue')
            axs[0, 0].set_title('Max Velocity vs Max Depth')
            axs[0, 0].set_xlabel('Max Depth')
            axs[0, 0].set_ylabel('Max Velocity')
            axs[0, 0].grid(True, linestyle='--', alpha=0.5)

            type_counts = summary_df['type_metadata'].value_counts()
            type_counts.plot(kind='bar', ax=axs[0, 1], color='orange', alpha=0.7)
            axs[0, 1].set_title('Distribution of Touch Types')
            axs[0, 1].set_ylabel('Count')
            axs[0, 1].tick_params(axis='x', rotation=45)

            axs[1, 0].hist(summary_df['max_acceleration'], bins=20, color='green', alpha=0.7)
            axs[1, 0].set_title('Max Acceleration Distribution')
            axs[1, 0].set_xlabel('Max Acceleration')
            axs[1, 0].set_ylabel('Frequency')

            axs[1, 1].hist(summary_df['max_contact_area'], bins=20, color='purple', alpha=0.7)
            axs[1, 1].set_title('Max Contact Area Distribution')
            axs[1, 1].set_xlabel('Max Contact Area')
            axs[1, 1].set_ylabel('Frequency')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show(block=True)
            
        except Exception as e:
            logging.warning(f"Could not generate summary plot: {e}")

    return output_path