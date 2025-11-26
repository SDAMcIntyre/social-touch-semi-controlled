import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
import tkinter as tk
import multiprocessing
import time

# External utility imports
from utils.should_process_task import should_process_task
from utils import get_pca1_signal_configurable
from preprocessing.trial_segmentation.adjust_chunks_viewer import AdjustChunksViewer
from preprocessing.common.data_access.video_mp4_manager import VideoMP4Manager, ColorFormat
# Import the new persistent runner
from preprocessing.common.gui.video_chunk_viewer import run_persistent_viewer

# -----------------------------------------------------------------------------
# Main Review Function
# -----------------------------------------------------------------------------

def review_single_touches(
    rgb_video_path: Path,
    trial_data_path: Path,
    stickers_xyz_path: Path,
    stimuli_metadata_path: Path,
    input_touches_path: Path,
    output_path: Path,
    force_processing: bool = False,
    xyz_cols: List[str] = ["sticker_blue_x_mm", "sticker_blue_y_mm", "sticker_blue_z_mm"]
):
    """
    Review touches grouped by Trial ID using AdjustChunksViewer.
    Allows interactive modification of start/end times and rejection/selection of touches based on the PCA signal.
    """
    # Normalize paths immediately
    input_touches_path = Path(input_touches_path)
    output_path = Path(output_path)
    trial_data_path = Path(trial_data_path)
    stickers_xyz_path = Path(stickers_xyz_path)
    stimuli_metadata_path = Path(stimuli_metadata_path)
    rgb_video_path = Path(rgb_video_path)

    # Check dependencies to decide if processing is needed
    if not should_process_task(
        input_paths=[trial_data_path, input_touches_path, stickers_xyz_path, stimuli_metadata_path],
        output_paths=[output_path],
        force=force_processing
    ):
        return

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. Load Data
    logging.info("Loading data...")
    touches_df = pd.read_csv(input_touches_path)
    source_df = pd.read_csv(stickers_xyz_path)
    trial_df = pd.read_csv(trial_data_path)
    stimuli_df = pd.read_csv(stimuli_metadata_path)
    
    # Detect Trial Boundaries
    single_touch_arr = touches_df["single_touch_id"].values
    trial_ids = trial_df["trial_id"].values
    unique_trial_ids = np.unique(trial_ids[trial_ids != 0])
    logging.info(f"Starting review. Found {len(unique_trial_ids)} potential trials.")

    # --- Initialize Persistent Video Viewer Process ---
    # We spawn ONE process for the entire duration of the loop.
    # This prevents Python interpreter reload overhead (approx 0.5-1s per trial on Windows).
    viewer_queue = multiprocessing.Queue()
    video_process = multiprocessing.Process(
        target=run_persistent_viewer,
        args=(viewer_queue, str(rgb_video_path))
    )
    video_process.start()
    
    # Give the viewer a moment to initialize VideoManager
    time.sleep(1)
    

    # TODO initialise an output data frame with the same number of rows as touches_df and a column named single_touch_id
    output_touches_df = pd.DataFrame(index=touches_df.index)
    output_touches_df["single_touch_id"] = 0 # Initialize all to 0 (no touch)
    chunks_id = 0 # global counter.

    try:
        # 2. Iterate Through Trials
        for uid, trial_id in enumerate(unique_trial_ids):
            trial_mask = (trial_ids == trial_id)
            # fetch the locations of the first and last True value in trial_mask
            t_start = np.where(trial_mask)[0][0]
            t_end = np.where(trial_mask)[0][-1]

            # --- A. Prepare Data for Viewer ---
            
            # 1. Extract Signal
            # Pandas .loc is inclusive, so t_end is included.
            # Ensure xyz_cols follows the requirement: X, Y, Z in order for PCA logic.
            # CRITICAL FIX: Convert DataFrame slice to NumPy array explicitly.
            # The PCA function uses array slicing (data[:, i]) which fails on DataFrames.
            chunk_xyz = source_df.loc[t_start:t_end, xyz_cols].to_numpy()
            
            # Now passing a raw (N, 3) numpy array to the configurable function
            pc1_signal = get_pca1_signal_configurable(chunk_xyz, enable_z_correction=True, z_correction_mode='negative')

            # 2. Extract Existing Touches
            # NumPy slicing is EXCLUSIVE. We must use t_end + 1 to include the last frame.
            trial_touches_slice = single_touch_arr[t_start : t_end + 1]
            valid_mask = trial_touches_slice != 0
            existing_ids = np.unique(trial_touches_slice[valid_mask])
            
            if not np.size(existing_ids):
                continue

            initial_chunks = []
            initial_selected_indices = [] 
            
            for local_idx, touch_id in enumerate(existing_ids):
                indices = np.where(trial_touches_slice == touch_id)[0]
                
                if len(indices) == 0: continue
                
                # Convert global indices to trial-relative indices
                rel_start = indices[0]
                rel_end = indices[-1]
                
                initial_chunks.append((rel_start, rel_end))
                initial_selected_indices.append(local_idx) 

            if len(pc1_signal) < 5:
                continue

            subset = stimuli_df.loc[stimuli_df["trial_id"] == trial_id, "type_metadata"]
            gesture_type = str(subset.item()).upper()
            logging.info(f"Reviewing Trial {uid} ({gesture_type}) Frames: {t_start}-{t_end}")
            
            # --- B. Update Video Viewer (Non-Blocking) ---
            # Instead of reading frames here, we send a command to the persistent process.
            # The worker process handles I/O parallel to us initializing the Signal Viewer.
            title = f"Video Review: Trial {uid} | {gesture_type}"
            viewer_queue.put((t_start, t_end, title))

            # --- C. Launch Signal Viewer (Blocking) ---
            # We assume AdjustChunksViewer creates its own Toplevel or uses a hidden root.
            
            viewer = AdjustChunksViewer(
                signals=[pc1_signal],
                initial_split_indices=initial_chunks,
                initial_selected_indices=initial_selected_indices,
                title=f"Signal Review: Trial {uid} | Type: {gesture_type} (Space=Create, Click=Select)",
                labels_list=["PCA Position Proxy"]
            )
            
            # --- D. Process Results ---
            for chunk in viewer.get_selected_chunks():
                chunks_id += 1
                global_idx_start = chunk[0] + t_start
                global_idx_end = chunk[1] + t_start
                
                output_touches_df.loc[global_idx_start:global_idx_end, "single_touch_id"] = chunks_id
            
            logging.info(f"Trial {uid}: Saved {len(viewer.get_selected_chunks())} chunks.")

    finally:
        # --- Cleanup ---
        logging.info("Cleaning up persistent video viewer...")
        viewer_queue.put("EXIT")
        video_process.join(timeout=2.0)
        if video_process.is_alive():
            video_process.terminate()

    # 3. Save Final Output
    logging.info(f"Saving modified touches to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure the output directory exists
    output_touches_df.to_csv(output_path, index=False)
    
    logging.info("✅ Review Complete.")