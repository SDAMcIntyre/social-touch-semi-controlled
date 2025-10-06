import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from typing import List, Optional, Tuple, NamedTuple

import logging
from pathlib import Path

# Setup a basic logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from utils.should_process_task import should_process_task


from preprocessing.led_analysis import (
    LedSignalValidator
)

# ===================================================================
#  STEP 1: Data Handling Functions
# ===================================================================
class DataIOException(Exception):
    """Custom exception for data I/O errors."""
    pass

def _load_data(
    csv_led_path: Path, stimulus_metadata_path: Path
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Loads LED and stimulus data, returning sanitized data structures.
    _ (underscore) indicates an internal helper function.
    """
    try:
        stimuli_df = pd.read_csv(stimulus_metadata_path)
        led_df = pd.read_csv(csv_led_path)

        if "LED on" not in led_df.columns:
            raise KeyError("'LED on' column not found.")

        led_on_signal = led_df["LED on"].values
        led_on_signal[np.isnan(led_on_signal)] = 0
        return led_df, led_on_signal, stimuli_df
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
        raise DataIOException(f"Failed to load or parse data. Reason: {e}")


def _save_corrected_data(
    output_path: Path, led_df: pd.DataFrame, corrected_signal: np.ndarray
) -> None:
    """Saves the DataFrame with the corrected signal."""
    output_df = led_df.copy()
    output_df["LED on"] = corrected_signal

    # Ensure signal length matches the original DataFrame's length
    if len(output_df) != len(corrected_signal):
        new_signal = np.zeros(len(output_df))
        size_to_copy = min(len(output_df), len(corrected_signal))
        new_signal[:size_to_copy] = corrected_signal[:size_to_copy]
        output_df["LED on"] = new_signal
        warnings.warn(f"Corrected signal length mismatch in {output_path.name}. "
                      f"Signal has been resized from {len(corrected_signal)} "
                      f"to {len(output_df)} frames.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)



# ===================================================================
#  STEP 2: Validation and Visualization Functions
# ===================================================================
class ValidationResult(NamedTuple):
    """A structured result for a validation check."""
    passed: bool
    message: str

def _validate_and_plot_results(
    file_name: str,
    validator: LedSignalValidator,
    expected_block_id: int,
    expected_ntrials: int,
    show_warning_plots: bool,
    show_final_plot: bool
) -> bool:
    """
    Performs validation and triggers plotting based on results and flags.
    Returns True if validation passes, False otherwise.
    """
    validation_passed = True
    
    # --- Block ID Validation ---
    if validator.led_block_id != expected_block_id:
        msg = (f"Block ID mismatch for '{file_name}': "
               f"Expected {expected_block_id}, found {validator.led_block_id}.")
        warnings.warn(msg)
        validation_passed = False
        if show_warning_plots:
            _plot_validation_failure(
                validator.raw_signal,
                f"{file_name}\nBlock ID Mismatch",
                expected_block_id,
                validator.led_block_id
            )

    # --- Trial Count Validation ---
    if validator.led_ntrials != expected_ntrials:
        msg = (f"Trial count mismatch for '{file_name}': "
               f"Expected {expected_ntrials}, found {validator.led_ntrials}.")
        warnings.warn(msg)
        validation_passed = False
        if show_warning_plots:
            trial_signal_start = validator.block_id_end_frame + 1
            signal_segment = validator.raw_signal[trial_signal_start:]
            _plot_validation_failure(
                signal_segment,
                f"{file_name}\nTrial Count Mismatch",
                expected_ntrials,
                validator.led_ntrials
            )

    # --- Final Comparison Plot ---
    if validation_passed and show_final_plot:
        _plot_final_comparison(
            file_name, validator.raw_signal, validator.corrected_signal
        )

    return validation_passed


def _plot_validation_failure(signal: np.ndarray, title: str, expected: int, found: int) -> None:
    """Displays a plot specifically for a validation failure."""
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title(f"{title}\nExpected: {expected} vs Found: {found}")
    plt.xlabel("Frame")
    plt.ylabel("Signal")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show(block=True)


def _plot_final_comparison(file_name: str, original: np.ndarray, corrected: np.ndarray) -> None:
    """Shows a side-by-side comparison of the original and corrected signals."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"Signal Correction for {file_name}", fontsize=16)

    ax1.plot(original, label='Original Signal', color='blue')
    ax1.set_title('Input Signal')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(corrected, label='Corrected Signal', color='green')
    ax2.set_title('Corrected Output')
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show(block=True)


# ===================================================================
#  STEP 3: Orchestrator (Ties everything together)
# ===================================================================
def validate_and_correct_led_timing_from_stimuli(
    csv_led_path: Path,
    stimulus_metadata_path: Path,
    output_path: Path,
    *,
    force_processing: bool = False,
    blinking_id_nframe_max: int = 4,
    merge_threshold: int = 10,
    show_warning_plots: bool = False,
    show_final_plot: bool = False
) -> None:
    """
    Orchestrates the validation and correction of LED timing data.

    This function coordinates the process:
    1. Checks preconditions (file skipping).
    2. Loads data using a helper function.
    3. Processes the signal using the LedSignalValidator.
    4. Validates and plots results using helper functions.
    5. Saves the corrected data if validation passes.
    """
    file_name = csv_led_path.name
    if not should_process_task(
         input_paths=[csv_led_path, stimulus_metadata_path], 
         output_paths=[output_path],
         force=force_processing):
        logging.info(f"Output file already exists. Skipping ROI definition for '{file_name}'.")
        return True

    print(f"INFO: Processing file: {file_name}")

    # --- Load, Process, Validate, Save ---
    try:
        # Step 1: Load Data
        led_df, led_on_signal, stimuli_df = _load_data(csv_led_path, stimulus_metadata_path)

        # Step 2: Process Signal
        validator = LedSignalValidator(
            led_on_signal=led_on_signal,
            blinking_id_nframe_max=blinking_id_nframe_max,
            merge_threshold=merge_threshold,
        )
        validator.process()

        # Step 3: Validate and Plot
        expected_block_id = stimuli_df["run_block_id"].iloc[0]
        expected_ntrials = len(stimuli_df)
        is_valid = _validate_and_plot_results(
            file_name, validator, expected_block_id, expected_ntrials,
            show_warning_plots, show_final_plot
        )

        # Step 4: Save Output
        if is_valid:
            _save_corrected_data(output_path, led_df, validator.corrected_signal)
            print(f"INFO: Successfully saved corrected file to '{output_path}'")
            return True
        else:
            print(f"WARN: Validation failed for '{file_name}'. File not saved.")
            return False

    except (DataIOException, Exception) as e:
        print(f"ERROR: An unexpected error occurred while processing '{file_name}'. Reason: {e}")
