import logging
from pathlib import Path
import pandas as pd
from typing import Optional

# Configure basic logging to print to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from utils.should_process_task import should_process_task

def add_trial_id(
        input_path: Path,
        output_path: Path,
        *,
        min_duration: int = 60,
        force_processing: bool = False
) -> bool:
    """
    Generates a 'trial_id' column based on 'led_on' signal transitions.

    Logic:
    1. Filters 'led_on' high states (1) that are shorter than min_duration.
    2. Increments 'trial_id' on every rising edge (0 -> 1) of the filtered signal.

    Args:
        input_path: Path to the input CSV containing 'led_on' column.
        output_path: Path for the output CSV file with added 'trial_id'.
        min_duration: Minimum number of consecutive samples for a '1' state to be considered valid.
        force_processing: If True, overwrites the output file even if it exists.

    Returns:
        True if the operation was successful (or skipped correctly), False otherwise.
    """
    if not should_process_task(
        input_paths=[input_path],
        output_paths=[output_path],
        force=force_processing
    ):
        logging.info(f"✅ Skipping task: Output file '{output_path}' already exists.")
        return True # Task is successfully skipped, so we return True.

    try:
        # Load data
        contact_chars_df = pd.read_csv(input_path)

        if "led_on" not in contact_chars_df.columns:
            logging.error(f"❌ Column 'led_on' missing in {input_path}")
            return False

        # Ensure led_on is integer/boolean and handle potential NaNs (assume 0)
        led_signal = contact_chars_df["led_on"].fillna(0).astype(int)

        # --- Step 1: Filter short pulses ---
        # Identify contiguous groups of values
        # (signal != signal.shift()) is True at transition points
        # .cumsum() creates a unique ID for each contiguous group
        groups = (led_signal != led_signal.shift()).cumsum()

        # Calculate the size of each group
        group_sizes = led_signal.groupby(groups).transform('count')

        # Create mask: True where signal is 1 BUT the group is shorter than threshold
        mask_short_pulses = (led_signal == 1) & (group_sizes < min_duration)

        # Apply mask: Set short pulses to 0
        # We use a copy to avoid SettingWithCopy warnings if applicable, though here it's a Series
        led_signal_filtered = led_signal.mask(mask_short_pulses, 0)

        # --- Step 2: Generate Trial ID ---
        # Detect rising edges on the FILTERED signal
        # Rising edge: Current is 1, Previous was 0
        # shift(1) gets previous value. fillna(0) ensures first sample is treated correctly if it's 1
        rising_edges = (led_signal_filtered == 1) & (led_signal_filtered.shift(1).fillna(0) == 0)

        # Cumulative sum of rising edges creates the ID
        # 0 0 1 1 0 0 1 1 -> Rising edges at idx 2 and 6 -> Cumsum: 0 0 1 1 1 1 2 2
        contact_chars_df["trial_id"] = rising_edges.cumsum()

        # --- Step 3: Save Output ---
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        contact_chars_df.to_csv(output_path, index=False)
        logging.info(f"✅ Successfully created trial_ids in '{output_path}'")
        
        return True

    except FileNotFoundError as e:
        logging.error(f"❌ File not found during processing: {e}")
        return False
    except Exception as e:
        logging.error(f"❌ An unexpected error occurred: {e}", exc_info=True)
        return False