import numpy as np
import warnings
from typing import Tuple

# --- Utility Functions (Could be in a separate utils.py) ---

def group_lengths(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the lengths of contiguous groups of identical values in an array.
    This is a form of run-length encoding.
    
    Args:
        arr (np.ndarray): The 1D input array.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing (lengths, labels).
    """
    if arr.size == 0:
        return np.array([]), np.array([])

    change_indices = np.where(arr[:-1] != arr[1:])[0] + 1
    indices = np.concatenate(([0], change_indices, [arr.size]))
    lengths = np.diff(indices)
    labels = arr[indices[:-1]]
    return lengths, labels


def merge_small_groups(lengths: np.ndarray, labels: np.ndarray, threshold: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merges groups smaller than a threshold with their neighbors.
    This implementation avoids modifying an array while iterating over it.

    Args:
        lengths (np.ndarray): Array of group lengths.
        labels (np.ndarray): Array of group labels (0 or 1).
        threshold (int): The minimum length for a group to be kept.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The new, merged lengths and labels.
    """
    if len(lengths) == 0:
        return lengths, labels

    new_lengths = []
    new_labels = []
    
    # Make copies to not modify original inputs
    temp_lengths = list(lengths)
    temp_labels = list(labels)

    i = 0
    while i < len(temp_lengths):
        if temp_lengths[i] < threshold:
            # If a small group is found, merge it with its right neighbor.
            # If it's the last element, merge it with its left neighbor.
            if i < len(temp_lengths) - 1:
                # Merge with the right neighbor
                temp_lengths[i+1] += temp_lengths[i]
            elif i > 0:
                # Last element is small, merge with the left neighbor (which is already in new_*)
                new_lengths[-1] += temp_lengths[i]
            # Delete the small group and its label
            temp_lengths.pop(i)
            temp_labels.pop(i)
            # Do not increment i, as the list has shifted
        else:
            # The group is large enough, move it to the new list
            new_lengths.append(temp_lengths.pop(i))
            new_labels.append(temp_labels.pop(i))
            
    return np.array(new_lengths), np.array(new_labels)


# --- Main Processor Class ---

class LedSignalValidator:
    """
    Encapsulates the logic for validating and correcting LED timing data.
    """
    def __init__(self, led_on_signal: np.ndarray, blinking_id_nframe_max: int = 4, merge_threshold: int = 10):
        """
        Initializes the processor with the signal and configuration.

        Args:
            led_on_signal (np.ndarray): The raw 1D numpy array of the LED signal (0s and 1s).
            blinking_id_nframe_max (int): Max frame duration for a blink to be part of the block ID.
            merge_threshold (int): Threshold below which signal groups are merged.
        """
        if not isinstance(led_on_signal, np.ndarray) or led_on_signal.ndim != 1:
            raise ValueError("led_on_signal must be a 1D numpy array.")
            
        self.raw_signal = led_on_signal
        self.blinking_id_nframe_max = blinking_id_nframe_max
        self.merge_threshold = merge_threshold

        # --- Results attributes ---
        self.is_processed = False
        self.led_block_id: int = 0
        self.led_ntrials: int = 0
        self.block_id_end_frame: int = -1
        self.corrected_signal: np.ndarray = np.array([])

    def process(self) -> None:
        """
        Executes the full processing pipeline for the signal.
        """
        self._find_block_id()
        if self.block_id_end_frame == -1:
            warnings.warn("Could not determine block ID from signal. Aborting processing.")
            self.is_processed = True
            return

        self._clean_and_count_trials()
        self._generate_corrected_signal()
        self.is_processed = True

    def _find_block_id(self) -> None:
        """Identifies the block ID from the initial blinks in the signal."""
        lengths, labels = group_lengths(self.raw_signal)
        
        # We only care about the 'ON' signals (label == 1) for block ID
        on_lengths = lengths[labels == 1]
        
        block_id = 0
        for length in on_lengths:
            if length < self.blinking_id_nframe_max:
                block_id += 1
            else:
                # First long signal marks the end of the block ID sequence
                break
        
        self.led_block_id = block_id
        
        if self.led_block_id > 0:
            # Find the end frame of the last blink
            on_indices = np.where(labels == 1)[0]
            if self.led_block_id <= len(on_indices):
                last_blink_index_in_groups = on_indices[self.led_block_id - 1]
                self.block_id_end_frame = np.sum(lengths[:last_blink_index_in_groups + 1]) - 1

    def _clean_and_count_trials(self) -> None:
        """Cleans the trial portion of the signal and counts the trials."""
        trial_signal = self.raw_signal[self.block_id_end_frame + 1:].copy()
        trial_signal[np.isnan(trial_signal)] = 0 # Sanitize NaNs
        
        lengths, labels = group_lengths(trial_signal)
        
        # Merge small spurious signals
        self.cleaned_lengths, self.cleaned_labels = merge_small_groups(lengths, labels, self.merge_threshold)
        
        # Count trials (sum of 'ON' labels)
        self.led_ntrials = int(np.sum(self.cleaned_labels))

    def _generate_corrected_signal(self) -> None:
        """Reconstructs the full signal from the processed parts."""
        block_id_part = self.raw_signal[:self.block_id_end_frame + 1]
        
        # Ensure cleaned_lengths and labels are available
        if not hasattr(self, 'cleaned_lengths') or not hasattr(self, 'cleaned_labels'):
             warnings.warn("Cleaning step was not run. Corrected signal will be incomplete.")
             self.corrected_signal = block_id_part
             return

        trial_part = np.repeat(self.cleaned_labels, self.cleaned_lengths)
        
        self.corrected_signal = np.concatenate([block_id_part, trial_part])