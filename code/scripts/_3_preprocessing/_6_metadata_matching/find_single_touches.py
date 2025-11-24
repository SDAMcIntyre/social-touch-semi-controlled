import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict, Union
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from utils.should_process_task import should_process_task

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

def _normalize_signal(sig: np.ndarray) -> np.ndarray:
    """Min-Max normalization to [0, 1] range."""
    denom = np.max(sig) - np.min(sig)
    if denom == 0:
        return np.zeros_like(sig)
    return (sig - np.min(sig)) / denom

def _get_pca_signal(chunk_xyz: pd.DataFrame) -> np.ndarray:
    """
    Computes 1D projection of 3D data using PCA. 
    Acts as a proxy for 'pos_1D' or 'depth'.
    """
    if len(chunk_xyz) < 5:
        return np.zeros(len(chunk_xyz))

    # Interpolate missing values
    chunk_clean = chunk_xyz.interpolate(method='linear').ffill().bfill()
    if chunk_clean.isnull().values.any():
        return np.zeros(len(chunk_xyz))

    pca = PCA(n_components=1)
    # Fit transform
    transformed = pca.fit_transform(chunk_clean)
    sig = transformed[:, 0]
    
    # Smooth signal (blind smoothing approximation)
    window_len = max(5, int(len(sig) * 0.05))
    if window_len % 2 == 0: window_len += 1
    if window_len < 3: window_len = 3
    
    sig_smooth = savgol_filter(sig, window_length=window_len, polyorder=2)
    sig_norm = _normalize_signal(sig_smooth)
    
    return sig_norm

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

def _split_strokes_reference(
    signal_arr: np.ndarray, 
    global_offset: int,
    expected_nsample_per_segment: int = 1
) -> List[Tuple[int, int]]:
    """
    Replicates 'get_single_strokes' logic.
    Defines segments based on intervals between extrema (peaks and valleys).
    """
    nsample = len(signal_arr)
    # Reference: min_dist_peaks = .5 * scd.md.nsample/nb_period_expected
    # Use max(1, ...) to avoid division by zero
    min_dist_peaks = max(10, int(0.5 * expected_nsample_per_segment))

    # Find peaks (Participant's hand)
    pos_peaks_a, _ = find_peaks(signal_arr, distance=min_dist_peaks)
    # Find valleys (Participant's elbow / return motion)
    pos_peaks_b, _ = find_peaks(-1 * signal_arr, distance=min_dist_peaks)
    
    # Merge and sort
    pos_peaks = np.sort(np.concatenate((pos_peaks_a, pos_peaks_b)))
    
    all_boundaries = [
        np.array([element]) if isinstance(element, int) else element 
        for element in [0, pos_peaks, nsample]
    ]
    all_boundaries=  np.concatenate(all_boundaries)

    endpoints_list = []
    for i in range(len(all_boundaries) - 1):
        # Reference: endpoints = (pos_peaks[i], pos_peaks[i+1] - 1)
        start = all_boundaries[i]
        end = all_boundaries[i+1] - 1
        endpoints_list.append((start + global_offset, end + global_offset))
        
    return endpoints_list

def _split_taps_reference(
    signal_arr: np.ndarray, 
    global_offset: int,
    expected_nsample_per_segment: int = 1
) -> List[Tuple[int, int]]:
    """
    Replicates 'get_single_taps' logic.
    Defines segments based on midpoints between peaks.
    """
    nsample = len(signal_arr)
    min_dist_peaks = max(10, int(0.5 *expected_nsample_per_segment))
    
    # Find peaks with prominence (as per reference)
    peaks, _ = find_peaks(signal_arr, distance=min_dist_peaks, prominence=(0.3, None))
    

    all_boundaries = [
        np.array([element]) if isinstance(element, int) else element 
        for element in [0, peaks, nsample]
    ]
    all_boundaries=  np.concatenate(all_boundaries)
    
    endpoints_list = []
    for i in range(len(all_boundaries) - 1):
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
        chunk_xyz = source_df.loc[start_idx:end_idx, xyz_cols]
        pc1_signal = _get_pca_signal(chunk_xyz)
        
        # Determine Splitting Strategy
        segments = []
        
        if "stroke" in gesture_type:
            expected_nsample_per_segment = int(Fs*(path_length/speed_cm_s))
            segments = _split_strokes_reference(pc1_signal, start_idx, expected_nsample_per_segment)
        else:
            # Default to Tap logic for taps or unknown
            expected_nsample_per_segment = int(2.0 * Fs*(path_length/speed_cm_s))
            
            # it is important to split taps from where the hand is at the highest
            # Low risk hypothesis here is made, where the xyz position of the hand is 
            # at the highest at t=0. If the normalised pc1 as its first value closer 
            # to 0, we flip it (mimick the hand being higher than the skin)
            if pc1_signal[0] < 0.5:
                pc1_signal = np.abs(1-pc1_signal)
            
            segments = _split_taps_reference(pc1_signal, start_idx, expected_nsample_per_segment)
            
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
        if enable_visualization and i < 5:
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

    # 5. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df = pd.DataFrame(refined_ids)
    output_df.to_csv(output_path, index=False)
    
    logging.info(f"✅ Successfully created reference-matched single_touch_id file in '{output_path}'")
    
    return True