# define_trial_chunks.py

import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Iterable, Dict

# Setup a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# External utility imports
# user-defined modules assumed to exist in the environment
from utils.should_process_task import should_process_task

# Importing necessary managers from the existing codebase structure
from preprocessing.common import VideoMP4Manager
from preprocessing.stickers_analysis import (
    ConsolidatedTracksFileHandler,
    ConsolidatedTracksManager
)
from preprocessing.trial_segmentation import TrialSegmenterGUI


def define_custom_colors(string_list: Iterable[str]) -> Dict[str, str]:
    """
    Assigns a standard color keyword based on substrings in a list of names.

    Args:
        string_list: An iterable (e.g., list) of object names.

    Returns:
        A dictionary mapping each object name to a found color string.
    """
    STANDARD_COLORS = {
        "red", "green", "blue", "yellow", "orange", "purple", "pink",
        "black", "white", "brown", "gray", "grey", "cyan", "magenta", "violet"
    }
    
    found_colors = {}
    
    for item in string_list:
        item_lower = item.lower()
        for color in STANDARD_COLORS:
            if color in item_lower:
                found_colors[item] = color
                break  # Assign the first color found and move to the next item
    
    return found_colors


def load_existing_chunks_from_csv(csv_path: Path) -> List[Tuple[int, int]]:
    """
    Loads previously recorded chunks from a CSV file containing a 'trial_on' binary column.
    
    Args:
        csv_path: Path to the CSV file.
        
    Returns:
        A list of (start, end) tuples derived from the binary mask. 
        Returns an empty list if file is missing or invalid.
    """
    if not csv_path.exists():
        return []
    
    try:
        df = pd.read_csv(csv_path)
        if 'trial_on' not in df.columns:
            logging.warning(f"CSV found but 'trial_on' column is missing in {csv_path}.")
            return []
            
        # Convert binary mask back to chunks (start, end)
        mask = df['trial_on'].to_numpy(dtype=int)
        
        # Pad with 0 at the start and end to detect transitions at boundaries
        padded_mask = np.pad(mask, (1, 1), mode='constant', constant_values=0)
        
        # distinct changes: 1 indicates 0->1 (start), -1 indicates 1->0 (end)
        diff = np.diff(padded_mask)
        
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        # Zip starts and ends together. 
        # Adjusting to inclusive ranges: (s, e - 1)
        chunks = []
        for s, e in zip(starts, ends):
            chunks.append((int(s), int(e - 1)))
            
        return chunks

    except Exception as e:
        logging.warning(f"Could not load existing chunks from {csv_path}: {e}")
        return []

def load_reference_led_mask(csv_path: Path) -> Optional[np.ndarray]:
    """
    Loads the external LED reference data, replacing NaNs with 0.

    Args:
        csv_path: Path to the CSV containing 'led_on'.

    Returns:
        Numpy array of the binary mask (int), or None if loading fails.
    """
    if not csv_path.exists():
        logging.warning(f"Reference LED file not found at: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        # Architectural fix: log message aligned with actual logic ('led_on')
        if 'led_on' not in df.columns:
            logging.warning(f"'led_on' column missing in reference file: {csv_path}")
            return None
        
        # Optimization: fillna(0) handles NaN floats before casting to int.
        # This prevents IntCastingNaNError and ensures valid binary mask generation.
        return df['led_on'].fillna(0).to_numpy(dtype=int)

    except Exception as e:
        logging.error(f"Failed to load reference LED mask: {e}")
        return None
    
def save_chunks_to_csv(chunks: List[Tuple[int, int]], output_csv_path: Path, total_frames: int):
    """
    Creates a new CSV file with a 'trial_on' mask.
    
    Args:
        chunks: List of (start, end) tuples (inclusive).
        output_csv_path: Path where the updated CSV will be saved.
        total_frames: The total number of frames in the video (defines row count).
    """
    try:
        # Ensure parent directory exists
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize DataFrame with zeros for the specified number of frames
        df = pd.DataFrame(0, index=np.arange(total_frames), columns=['trial_on'])
        
        for start, end in chunks:
            # Clip values to ensure we don't exceed bounds
            s = max(0, start)
            e = min(total_frames - 1, end)
            
            # Set mask to 1 for the range [s, e] inclusive
            if s <= e:
                df.iloc[s : e + 1, 0] = 1
            
        df.to_csv(output_csv_path, index=False)
        logging.info(f"Successfully saved binary mask in: {output_csv_path.name}")
        
    except Exception as e:
        logging.error(f"Critical Error: Failed to save chunks to {output_csv_path}: {e}")


def define_trial_chunks(
    xy_csv_path: Path,
    led_on_path: Path,
    rgb_video_path: Path,
    output_csv_path: Path,
    *,
    force_processing: bool = False
):
    """
    Loads data, launches the TrialSegmenterGUI, and saves the resulting annotations to an independent CSV.

    Args:
        xy_csv_path: Path to the tracking data CSV (for visualization).
        led_on_path: Path to the reference LED CSV (visualization).
        rgb_video_path: Path to the MP4 video file.
        output_csv_path: Final destination CSV containing only 'trial_on'.
        force_processing: If True, overrides the task processing check.
    """
    # 1. Check if processing is needed
    if not should_process_task(
         input_paths=[xy_csv_path, rgb_video_path], 
         output_paths=[output_csv_path], 
         force=force_processing):
        logging.info(f"Output files already exist. Skipping analysis for '{rgb_video_path.name}'.")
        return

    # 2. Load Tracking Data (For GUI Visualization)
    logging.info(f"Loading tracking data from: {xy_csv_path.name}...")
    try:
        tracked_data: ConsolidatedTracksManager = ConsolidatedTracksFileHandler.load(xy_csv_path)
        sticker_names = tracked_data.object_names
        logging.info(f"Data loaded. Objects found: {tracked_data.object_names}")
    except Exception as e:
        logging.error(f"Error loading tracking data: {e}")
        return
    
    # Automatically determine colors for each sticker based on its name
    sticker_colors = define_custom_colors(sticker_names)

    # 3. Load Video
    logging.info(f"Loading video from: {rgb_video_path.name}...")
    try:
        video_manager = VideoMP4Manager(rgb_video_path)
        logging.info(f"Video loaded successfully. FPS: {video_manager.fps}")
    except Exception as e:
        logging.error(f"Error loading video: {e}")
        return

    # 4. Load Reference LED Data (New Integration)
    logging.info(f"Loading reference LED data from: {led_on_path.name}...")
    reference_mask = load_reference_led_mask(led_on_path)
    if reference_mask is not None:
        logging.info(f"Reference mask loaded. Length: {len(reference_mask)}")
    else:
        logging.warning("Proceeding without reference LED mask.")

    # 5. Load Existing Annotations
    # We load from output_csv_path if it exists to support resuming
    initial_chunks = load_existing_chunks_from_csv(output_csv_path)
    if initial_chunks:
        logging.info(f"Resuming session with {len(initial_chunks)} pre-existing chunks.")
    else:
        logging.info("Starting a new recording session (No active chunks found).")

    # 6. Initialize and Start GUI
    logging.info("Launching Chunk Recording Interface...")
    logging.info("  - Controls: [Space] to Toggle Record, [Arrows] to Seek")
    
    gui = TrialSegmenterGUI(
        video_manager=video_manager,
        tracks_manager=tracked_data,
        object_colors=sticker_colors,
        initial_chunks=initial_chunks,
        reference_mask=reference_mask,
        title=f"Annotating: {rgb_video_path.name}",
        windowState='maximized'
    )
    
    gui.start()

    # 7. Retrieve and Save Data
    final_chunks = gui.chunks
    
    logging.info("-" * 40)
    logging.info("Session Ended.")
    
    if final_chunks != initial_chunks:
        logging.info(f"Changes detected. Updating mask for {len(final_chunks)} chunks...")
        
        # Correctly retrieve total frames from the manager property.
        total_frames = video_manager.total_frames

        save_chunks_to_csv(final_chunks, output_csv_path, total_frames)
    else:
        logging.info("No changes detected. File not updated.")