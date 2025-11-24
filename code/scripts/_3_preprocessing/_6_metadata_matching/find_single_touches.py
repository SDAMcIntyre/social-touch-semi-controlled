import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, List
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Configure basic logging to print to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Assuming utils.should_process_task is available in the user's environment
# If this module is missing in the specific execution context, this import may need mocking
from utils.should_process_task import should_process_task

def find_single_touches(
        stickers_xyz_path: Path,
        trial_data_path: Path,
        stimuli_metadata_path: Path,
        output_path: Path,
        *,
        trial_col: str = "trial_on",
        xyz_cols: List[str] = ["sticker_blue_x_mm", "sticker_blue_y_mm", "sticker_blue_z_mm"],
        force_processing: bool = False,
        interpolation_method: str = "cubic",
        enable_visualization: bool = True  # MODIFICATION: Default is now True
) -> bool:
    """
    Generates a separate CSV containing a 'single_touch_id' column based on 
    signal transitions from the source data file.
    
    The output file is a sidecar file; it does not contain the original unified data.

    Logic:
    1. Loads the binary touch signal (defined by trial_col) from trial_data_path.
    2. Identifies consecutive groups of 1s as discrete touch events.
    3. Extracts XYZ coordinates for each group.
    4. Interpolates missing values (NaNs) to ensure PCA stability.
    5. Runs PCA on the reconstructed data.
    6. Saves a CSV with N rows and a 'single_touch_id' column (0 = no touch, >0 = touch ID).

    Args:
        stickers_xyz_path: Path to the CSV containing the raw XYZ sticker data.
        trial_data_path: Path to the CSV containing the binary trial signal.
        output_path: Path for the output CSV file containing only touch metadata.
        trial_col: The name of the column representing binary touch state (default: 'trial_on').
        xyz_cols: List of column names representing the 3D coordinates.
        force_processing: If True, overwrites the output file even if it exists.
        interpolation_method: The method used to fill NaN values (default: 'cubic').
                              Supported values: 'linear', 'cubic', 'spline', etc.
        enable_visualization: If True, plots the data projected onto the 1st Principal Component.
                              WARNING: This blocks execution until the plot window is closed.

    Returns:
        True if the operation was successful (or skipped correctly), False otherwise.
    """
    if not should_process_task(
        input_paths=[stickers_xyz_path, trial_data_path, stimuli_metadata_path],
        output_paths=[output_path],
        force=force_processing
    ):
        logging.info(f"✅ Skipping task: Output file '{output_path}' already exists.")
        return True 

    try:
        # Load Data
        source_df = pd.read_csv(stickers_xyz_path)
        trial_df = pd.read_csv(trial_data_path, usecols=[trial_col])
        stimuli_df = pd.read_csv(stimuli_metadata_path)
        
        # Validate Data Alignment
        if len(source_df) != len(trial_df):
            logging.warning(f"⚠️ Row count mismatch: Source ({len(source_df)}) vs Trial ({len(trial_df)}). Truncating to minimum length.")
            min_len = min(len(source_df), len(trial_df))
            source_df = source_df.iloc[:min_len]
            trial_df = trial_df.iloc[:min_len]

        # extract the gesture types for the current dataset
        gesture_type = stimuli_df["type_metadata"]

        # Ensure signal is integer (0/1)
        trial_signal = trial_df[trial_col].fillna(0).astype(int)
        
        # --- Logic: Identify Consecutive Groups ---
        # Detect changes in the signal (rising or falling edges)
        raw_groups = (trial_signal != trial_signal.shift()).cumsum()
        
        # Initialize the output series with 0
        touch_ids = pd.Series(0, index=trial_df.index, name="single_touch_id")
        
        # Assign raw groups only where the signal is 1
        active_touches_mask = (trial_signal == 1)
        temp_ids = raw_groups[active_touches_mask]
        
        # Normalize IDs to be 1, 2, 3... strictly sequential
        if not temp_ids.empty:
            normalized_ids = temp_ids.rank(method='dense').astype(int)
            touch_ids.loc[active_touches_mask] = normalized_ids
        
        # --- Logic: PCA on Chunks ---
        unique_ids = touch_ids.unique()
        unique_ids = unique_ids[unique_ids != 0]
        
        logging.info(f"ℹ️  Found {len(unique_ids)} discrete touch events. Processing PCA with {interpolation_method} interpolation...")
        
        for uid in unique_ids:
            current_gesture_type = gesture_type[uid]

            # Chunk the source DF based on the mask for this specific ID
            chunk_mask = (touch_ids == uid)
            chunk_xyz = source_df.loc[chunk_mask, xyz_cols]
            
            # Check length before processing
            if len(chunk_xyz) < 2:
                continue

            # --- Interpolation Logic ---
            chunk_xyz_clean = chunk_xyz.copy()
            
            if chunk_xyz_clean.isnull().values.any():
                try:
                    if 'cubic' in interpolation_method and len(chunk_xyz_clean) < 4:
                         chunk_xyz_clean = chunk_xyz_clean.interpolate(method='linear')
                    else:
                         chunk_xyz_clean = chunk_xyz_clean.interpolate(method=interpolation_method)
                    
                    chunk_xyz_clean = chunk_xyz_clean.ffill().bfill()
                    
                except Exception as e:
                    logging.warning(f"⚠️ Interpolation failed for ID {uid}: {e}. Retrying with 'linear'.")
                    chunk_xyz_clean = chunk_xyz_clean.interpolate(method='linear').ffill().bfill()

            if chunk_xyz_clean.isnull().values.any():
                 logging.warning(f"⚠️ Touch ID {uid}: Skipped. Data could not be reconstructed.")
                 continue

            # Apply PCA (Still 3D for calculation accuracy)
            pca = PCA(n_components=3)
            pca.fit(chunk_xyz_clean)
            _ = pca.explained_variance_ratio_
            # Transform the original 3D data into the Principal Component space
            # Result shape is (n_samples, 3)
            transformed_data = pca.transform(chunk_xyz_clean)
            
            # Extract the 1st dimension (PC1). This corresponds to the "main dimension"
            # requested by the user (like the x-axis, but in PC space).
            pc1_projection = transformed_data[:, 0]
            

            # TODO return the code to handle either 'tap' or 'stroke' for finding the single touches
            
            # --- VISUALIZATION LOGIC (1D Projection on PC1) ---
            if enable_visualization:
                logging.info(f"📊 Visualizing Touch ID {uid} (PC1 Projection)...")
                
                # Create a plot
                fig, ax = plt.subplots(figsize=(10, 5))
                
                # Create an index array for the X-axis (Time/Sequence)
                # We reset the index to 0..N for this specific touch event
                time_steps = np.arange(len(pc1_projection))
                
                # Plot the single dimension
                ax.plot(time_steps, pc1_projection, marker='o', markersize=4, linestyle='-', linewidth=1.5, color='blue', label='PC1 Projection')
                
                # Visual Polish
                explained_var = pca.explained_variance_ratio_[0] * 100
                ax.set_title(f'Touch ID {uid}: Projection along Main Principal Component (PC1)\nPC1 Explained Variance: {explained_var:.2f}%')
                ax.set_xlabel('Sample Index (Relative Time)')
                ax.set_ylabel('Displacement along PC1 (mm)')
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend()
                
                # Ensure tight layout
                plt.tight_layout()
                
                # Show plot
                plt.show(block=True)

        # --- Save Output ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df = pd.DataFrame(touch_ids)
        output_df.to_csv(output_path, index=False)
        logging.info(f"✅ Successfully created separate single_touch_id file in '{output_path}'")
        
        return True

    except Exception as e:
        logging.error(f"❌ An unexpected error occurred: {e}", exc_info=True)
        return False