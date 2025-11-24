import logging
from pathlib import Path
import pandas as pd
from typing import Optional

# Configure basic logging to print to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from utils.should_process_task import should_process_task

def calculate_trial_id(
        trial_chunk_path: Path,
        output_path: Path,
        *,
        force_processing: bool = False
) -> bool:
    """
    Generates a separate CSV containing 'trial_id' and 'trial_on' columns based on 
    signal transitions from an external chunk file.
    
    The output file is a sidecar file; it does not contain the original unified data.

    Logic:
    1. Loads 'trial_on' from trial_chunk_path.
    2. Validates length against unified_data_path to ensure synchronization.
    3. Increments 'trial_id' on every rising edge (0 -> 1) of the 'trial_on' signal.
    4. Saves ONLY the 'trial_on' and 'trial_id' columns to output_path.

    Args:
        trial_chunk_path: Path to the CSV containing the 'trial_on' column.
        output_path: Path for the output CSV file containing only trial metadata.
        force_processing: If True, overwrites the output file even if it exists.

    Returns:
        True if the operation was successful (or skipped correctly), False otherwise.
    """
    if not should_process_task(
        input_paths=[unified_data_path, trial_chunk_path],
        output_paths=[output_path],
        force=force_processing
    ):
        logging.info(f"✅ Skipping task: Output file '{output_path}' already exists.")
        return True 

    try:
        # Load trial chunk data
        if not trial_chunk_path.exists():
            logging.error(f"❌ Trial chunk file not found: {trial_chunk_path}")
            return False
            
        chunk_df = pd.read_csv(trial_chunk_path)

        if "trial_on" not in chunk_df.columns:
            logging.error(f"❌ Column 'trial_on' missing in {trial_chunk_path}")
            return False

        # Create a new DataFrame for output to ensure separation
        output_df = pd.DataFrame()

        # Ensure trial_on is integer (0/1) using the data from chunk_df
        trial_signal = chunk_df["trial_on"].fillna(0).astype(int)
        output_df["trial_on"] = trial_signal

        # --- Generate Trial ID ---
        # Detect rising edges: Current is 1, Previous was 0
        rising_edges = (trial_signal == 1) & (trial_signal.shift(1).fillna(0) == 0)

        # Cumulative sum creates the ID (1, 2, 3...)
        cumulative_ids = rising_edges.cumsum()

        # --- Restrict to Trial Chunk ---
        # Apply mask: trial_id should be 0 if trial_on is 0. 
        # Otherwise, it takes the value of the cumulative ID.
        output_df["trial_id"] = cumulative_ids * trial_signal

        # --- Save Output ---
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save only the trial metadata, not the full dataset
        output_df.to_csv(output_path, index=False)
        logging.info(f"✅ Successfully created separate trial_ids file in '{output_path}'")
        
        return True

    except FileNotFoundError as e:
        logging.error(f"❌ File not found during processing: {e}")
        return False
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred: {e}", exc_info=True)
        return False