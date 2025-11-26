import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Union, Literal
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter, windows
from utils.should_process_task import should_process_task

# 
from utils.should_process_task import should_process_task
from utils import get_pca1_signal_configurable

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _load_and_validate_data(
    stickers_path: Path, 
    trial_path: Path, 
    stimuli_path: Path, 
    trial_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads CSVs, aligns row counts, and extracts gesture metadata.
    """
    logging.info(f"📂 Loading data from {stickers_path.name}...")
    source_df = pd.read_csv(stickers_path)
    trial_df = pd.read_csv(trial_path, usecols=[trial_col])
    stimuli_df = pd.read_csv(stimuli_path)
    
    # Validate Data Alignment
    if len(source_df) != len(trial_df):
        logging.warning(f"⚠️ Row count mismatch: Source ({len(source_df)}) vs Trial ({len(trial_df)}). Truncating to minimum length.")
        min_len = min(len(source_df), len(trial_df))
        source_df = source_df.iloc[:min_len]
        trial_df = trial_df.iloc[:min_len]

    # Extract gesture types mapping
    gesture_type_map = stimuli_df["type_metadata"] if "type_metadata" in stimuli_df.columns else pd.Series(["tap"] * (len(trial_df) // 100 + 1))
    if "type_metadata" not in stimuli_df.columns:
        logging.warning("⚠️ 'type_metadata' column missing. Defaulting to 'tap' for all trials.")

    speed_metadata_map = stimuli_df["speed_metadata"] if "speed_metadata" in stimuli_df.columns else pd.Series(["normal"] * (len(trial_df) // 100 + 1))
    if "speed_metadata" not in stimuli_df.columns:
        logging.warning("⚠️ 'speed_metadata' column missing. Defaulting to 'normal' for all trials.")

    return source_df, trial_df, gesture_type_map, speed_metadata_map

def _correct_segments(
    segments: List[Tuple[int, int]], 
    total_len: int, 
    hp_duration_samples: int = 50
) -> List[Tuple[int, int]]:
    """
    Replicates the 'correct' method from semicontrolled_data_splitter.py.
    Merges segments that are too short (less than hp_duration_samples).
    """
    if not segments:
        return []

    # Use a while loop to allow dynamic modification of the list
    idx = 0
    # Create a mutable list of segments
    current_segments = segments.copy()

    while len(current_segments) > 1 and idx < len(current_segments):
        start, end = current_segments[idx]
        duration = end - start
        
        # If segment is too short, merge it
        if duration < hp_duration_samples:
            if idx == 0:
                # Merge with right neighbor
                next_start, next_end = current_segments[idx + 1]
                new_seg = (start, next_end)
                current_segments[0] = new_seg
                del current_segments[1]
                # Do not increment idx, re-evaluate index 0
            elif idx == len(current_segments) - 1:
                # Merge with left neighbor
                prev_start, prev_end = current_segments[idx - 1]
                new_seg = (prev_start, end)
                current_segments[idx - 1] = new_seg
                del current_segments[idx]
                # Index is now out of bounds, loop will terminate or check previous
                idx -= 1
            else:
                # Merge with the smaller neighbor
                prev_idx = idx - 1
                next_idx = idx + 1
                len_prev = current_segments[prev_idx][1] - current_segments[prev_idx][0]
                len_next = current_segments[next_idx][1] - current_segments[next_idx][0]
                
                if len_prev > len_next: 
                    # Note: Following reference logic strictly
                    # if scd_list[_prev].md.nsample > scd_list[_next].md.nsample:
                    #    scd_list[_prev].append(scd_list[idx]) -> extend left
                    
                    prev_start, prev_end = current_segments[prev_idx]
                    new_seg = (prev_start, end) # Extend left to include current
                    current_segments[prev_idx] = new_seg
                    del current_segments[idx]
                    # Idx points to next element now, but we removed current. 
                    # So idx now points to what was next. Re-evaluate loop.
                    idx -= 1
                else:
                    # scd_list[idx].append(scd_list[_next]) -> extend current to include right
                    next_start, next_end = current_segments[next_idx]
                    new_seg = (start, next_end)
                    current_segments[idx] = new_seg
                    del current_segments[next_idx]
                    # Do not increment, re-evaluate merged segment
        else:
            idx += 1
            
    return current_segments

def _apply_edge_dampening(signal_arr: np.ndarray, edge_ratio: float = 0.1) -> np.ndarray:
    """
    Applies a Gaussian window to the first and last 'edge_ratio' percentage of the signal.
    The dampening is applied 'around the mean', meaning the signal is pulled towards its mean value.
    """
    nsample = len(signal_arr)
    edge_len = int(nsample * edge_ratio)
    
    if edge_len < 1:
        return signal_arr.copy()


    # Create Gaussian Taper
    # We want a half-gaussian that goes from 0 (at index 0) to 1 (at index edge_len).
    # Standard gaussian std dev (sigma). 3 sigma covers 99.7%. 
    # So if we set 3*sigma = edge_len, index 0 will be ~0.01 (relative to peak).
    sigma = edge_len / 3.0
    
    # Generate full gaussian and take the left half
    # A window of length 2*edge_len + 1, peak at middle.
    gauss_window = windows.gaussian(2 * edge_len + 1, std=sigma)
    # Taking the rising edge (0 to peak)
    rise_edge = gauss_window[:edge_len]
    
    # Construct the full mask: [Rise] + [Ones] + [Fall]
    mask = np.ones(nsample)
    
    # Apply Rise to start
    mask[:edge_len] = rise_edge
    
    # Apply Fall to end (reverse of rise)
    # Ensure we don't overflow if 2*edge_len > nsample, though logical 20% limit prevents this.
    end_start_idx = nsample - edge_len
    mask[end_start_idx:] = rise_edge[::-1]
    
    # Calculate Mean
    sig_mean = np.mean(signal_arr)
    centered_sig = signal_arr - sig_mean
    
    # Apply mask to centered signal and add mean back
    dampened_sig = (centered_sig * mask) + sig_mean
    
    return dampened_sig

def _signal_segmentation(
    signal_arr: np.ndarray, 
    global_offset: int,
    strategy: Literal['strokes', 'taps'],
    expected_nsample_per_segment: int = 1,
    max_diff_ratio: float = 0.3,
    edge_dampening_ratio: float = 0.1,
    retries_remaining: int = 3,
    low_bound: float = 0.3,
    plot_results: bool = False
) -> List[Tuple[int, int]]:
    """
    Unified kernel for splitting signals based on strokes or taps strategies.
    Handles peak detection, plotting, adaptive recursion, and boundary generation.
    """
    nsample = len(signal_arr)
    # Reference: min_dist_peaks = .5 * scd.md.nsample/nb_period_expected
    min_dist_peaks = max(5, int(0.5 * expected_nsample_per_segment))
    
    # --- Strategy Execution ---
    peaks = np.array([], dtype=int)
    actual_segments = 0
    
    if strategy == 'stroke':
        # Find peaks (Participant's hand)
        pos_peaks_a, _ = find_peaks(signal_arr, distance=min_dist_peaks, prominence=(low_bound, None))
        # Find valleys (Participant's elbow / return motion)
        pos_peaks_b, _ = find_peaks(-1 * signal_arr, distance=min_dist_peaks, prominence=(low_bound, None))
        # Merge and sort
        peaks = np.sort(np.concatenate((pos_peaks_a, pos_peaks_b)))
        # Peaks define boundaries roughly 1:1 with segments in this logic
        actual_segments = len(peaks)
    
    elif strategy == 'tap':
        # Find peaks with prominence
        peaks, _ = find_peaks(signal_arr, distance=min_dist_peaks, prominence=(low_bound, None))
        # Taps logic usually results in N+1 segments for N peaks
        actual_segments = len(peaks) + 1

    # --- Visualization ---
    if plot_results:
        plt.figure(figsize=(12, 6))
        plt.plot(signal_arr, label='Signal', color='blue')
        
        # Plot peaks overlay
        if len(peaks) > 0:
            plt.plot(peaks, signal_arr[peaks], "x", label='Detected Peaks', color='red', markersize=10)
            
        plt.title(f"Signal Analysis ({strategy.capitalize()}): {len(peaks)} Peaks Detected")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show(block=True)

    # --- Adaptive Check Logic ---
    if retries_remaining > 0 and expected_nsample_per_segment > 0:
        expected_n_segments = nsample / expected_nsample_per_segment
        
        # Avoid division by zero
        if expected_n_segments > 0:
            diff = abs(expected_n_segments - actual_segments) / expected_n_segments
            
            if diff > max_diff_ratio:
                logging.debug(f"{strategy.capitalize()}: Mismatch detected (Diff: {diff:.2f}). Retrying with Gaussian dampening. Retries left: {retries_remaining - 1}")
                
                # Apply dampening (Assumed to exist in scope)
                dampened_signal = _apply_edge_dampening(signal_arr, edge_ratio=edge_dampening_ratio)
                
                # Recursive retry calling this core function
                return _signal_segmentation(
                    signal_arr=dampened_signal, 
                    global_offset=global_offset,
                    strategy=strategy,
                    expected_nsample_per_segment=expected_nsample_per_segment, 
                    max_diff_ratio=max_diff_ratio, 
                    edge_dampening_ratio=edge_dampening_ratio, 
                    retries_remaining=retries_remaining - 1,
                    low_bound=low_bound/2,
                    plot_results=plot_results
                )

    # --- Boundary Construction ---
    all_boundaries = [
        np.array([element]) if isinstance(element, int) else element 
        for element in [0, peaks, nsample]
    ]
    all_boundaries = np.concatenate(all_boundaries)

    endpoints_list = []
    for i in range(len(all_boundaries) - 1):
        # Reference: endpoints = (pos_peaks[i], pos_peaks[i+1] - 1)
        start = all_boundaries[i]
        end = all_boundaries[i+1] - 1
        endpoints_list.append((start + global_offset, end + global_offset))
        
    return endpoints_list


def find_single_touches(
        stickers_xyz_path: Path,
        trial_data_path: Path,
        stimuli_metadata_path: Path,
        output_path: Path,
        *,
        force_processing: bool = False,
        enable_visualization: bool = False,
        trial_col: str = "trial_on",
        xyz_cols: List[str] = ["sticker_blue_x_mm", "sticker_blue_y_mm", "sticker_blue_z_mm"],
        Fs: float = 30.0, # Hz, camera refresh rate
        path_length: float = 3.0 # quick-fix hardcoded path length (in cm)
) -> bool:
    """
    Generates 'single_touch_id' by replicating the segmentation logic of 
    semicontrolled_data_splitter.py (Peak/Valley splitting and merging),
    using PCA of sticker coordinates as the signal source.
    """
    if not should_process_task(
        input_paths=[stickers_xyz_path, trial_data_path, stimuli_metadata_path],
        output_paths=[output_path],
        force=force_processing
    ):
        logging.info(f"✅ Skipping task: Output file '{output_path}' already exists.")
        return True 

    # 1. Load Data
    source_df, trial_df, gesture_type_map, speed_metadata_map = _load_and_validate_data(
        stickers_xyz_path, trial_data_path, stimuli_metadata_path, trial_col
    )
    
    # 2. Identify Coarse Regions (Trial ON)
    # We use the original 'coarse' logic just to find the windows where data is valid.
    trial_signal = trial_df[trial_col].fillna(0).astype(int)
    #plt.plot(trial_signal); plt.show(block=True)
    # Find contiguous regions of 1s
    trial_signal_diff = np.diff(np.concatenate(([0], trial_signal.values, [0])))
    starts = np.where(trial_signal_diff == 1)[0]
    ends = np.where(trial_signal_diff == -1)[0]
    
    refined_ids = pd.Series(0, index=trial_df.index, name="single_touch_id")
    touch_counter = 1
    
    logging.info(f"ℹ️  Found {len(starts)} coarse trial regions. Applying reference splitting logic...")
    
    # 3. Iterate over coarse regions and split them internally
    for i, (start_idx, end_idx) in enumerate(zip(starts, ends)):
        # Get Gesture Type for this region
        # Map coarse region index to gesture type index (approximate if metadata isn't 1:1)
        meta_idx = min(i, len(gesture_type_map) - 1)
        gesture_type = str(gesture_type_map.iloc[meta_idx]).lower()
        speed_cm_s = float(speed_metadata_map.iloc[meta_idx])
        
        logging.debug(f"Processing Trial {i}: gesture={gesture_type}, speed={speed_cm_s} cm/s")
        
        # Extract Signal (PCA of XYZ) and Data Chunk
        chunk_xyz = source_df.loc[start_idx:end_idx, xyz_cols].values
        
        # --- MODIFICATION: Use standardized generic_signal_processing logic ---
        # The generic function handles interpolation, robust fitting, z-correction, and smoothing.
        # It expects a numpy array.
        pc1_signal = get_pca1_signal_configurable(chunk_xyz, enable_z_correction=True, z_correction_mode='negative')
        # Determine Splitting Strategy
        segments = []
        
        if "stroke" in gesture_type:
            expected_nsample_per_segment = int(Fs*(path_length/speed_cm_s))
        else:
            # Default to Tap logic for taps
            expected_nsample_per_segment = int(2.0 * Fs*(path_length/speed_cm_s))
                
        segments = _signal_segmentation(
            signal_arr=pc1_signal, 
            global_offset=start_idx, 
            strategy=gesture_type,
            expected_nsample_per_segment=expected_nsample_per_segment,
            plot_results=enable_visualization)

        # Correction Phase (Merge short segments)
        # Dynamic HP duration: 25% of the expected single period duration
        hp_limit = int(expected_nsample_per_segment * 0.25)
        
        segments = _correct_segments(segments, len(pc1_signal), hp_duration_samples=hp_limit)
        
        # 4. Assign IDs and Prepare Visualization Data
        for (seg_start, seg_end) in segments:
            # Ensure bounds
            seg_start = max(start_idx, seg_start)
            seg_end = min(end_idx, seg_end)
            
            refined_ids.loc[seg_start:seg_end] = touch_counter
            touch_counter += 1

        # Debug Visualization (Aggregated per Trial UID)
        # Limit to the first 5 trials to prevent flooding
        if enable_visualization:
            plt.figure(figsize=(10, 3))
            plt.plot(pc1_signal, label='PCA Signal')
            
            # Iterate through all segments found in this trial to plot them on the same figure
            for (seg_start, seg_end) in segments:
                # Calculate local indices relative to the PCA signal chunk
                local_s = max(0, seg_start - start_idx)
                local_e = min(len(pc1_signal) - 1, seg_end - start_idx)
                plt.axvspan(local_s, local_e, color='green', alpha=0.3)
                
            plt.title(f"Trial {i}: {gesture_type} (Speed: {speed_cm_s}cm/s, Exp number of samples per segment: {expected_nsample_per_segment})")
            plt.xlabel("Local Samples")
            plt.ylabel("Normalized Amplitude")
            plt.legend()
            plt.show(block=True)

        pass

    # 5. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df = pd.DataFrame(refined_ids)
    output_df.to_csv(output_path, index=False)
    
    logging.info(f"✅ Successfully created reference-matched single_touch_id file in '{output_path}'")
    
    return True