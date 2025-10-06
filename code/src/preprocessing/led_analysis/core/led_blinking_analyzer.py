# Standard library imports
import logging
from pathlib import Path
from typing import Union, Tuple, List, Optional, Dict, Any
import warnings

# Third-party imports
import cv2
import numpy as np
from sklearn import mixture
from sklearn.exceptions import ConvergenceWarning

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Local application/library specific imports
from preprocessing.common.data_access.video_mp4_manager import VideoMP4Manager

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --- Helper Function (Unchanged) ---
def normalize_signal(signal: Union[List[float], np.ndarray]) -> np.ndarray:
    """Normalizes a signal to the range [0, 1]. Handles NaN values."""
    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return signal
    
    try:
        min_val = np.nanmin(signal)
        max_val = np.nanmax(signal)
        if min_val == max_val:
            return np.full_like(signal, 0.5)
        normalized_signal = (signal - min_val) / (max_val - min_val)
    except (ValueError, FloatingPointError):
        return np.full_like(signal, np.nan)
        
    return normalized_signal


class LEDBlinkingAnalyzer:
    """
    Analyzes a video to detect and timestamp the blinking of an LED.
    
    This class uses memory-efficient algorithms to identify blinking pixels,
    extract their signal, classify the LED's on/off state, and detect potential
    occlusions or corrupted frames, making it suitable for large video files.
    """

    def __init__(self, video_path: Union[str, Path],
                 bic_diff_threshold: float = 10.0,
                 min_separation_factor: float = 0.1,
                 min_weight: float = 0.1,
                 occlusion_threshold: float = 40.0,
                 discard_black_frames: bool = True,
                 black_frame_threshold: float = 1.0):
        """
        Initializes the analyzer with configuration parameters.

        Args:
            video_path (Union[str, Path]): Path to the video file.
            bic_diff_threshold (float): BIC score difference to confirm bimodality.
            min_separation_factor (float): Minimum separation between GMM means.
            min_weight (float): Minimum weight for a GMM component.
            occlusion_threshold (float): Threshold for detecting occluded frames.
            discard_black_frames (bool): If True, detects and ignores all-black frames.
            black_frame_threshold (float): Mean pixel intensity below which a frame is
                                           considered black/corrupted.
        """
        self.video_manager = VideoMP4Manager(video_path)
        
        # Store configuration parameters
        self.bic_diff_threshold = bic_diff_threshold
        self.min_separation_factor = min_separation_factor
        self.min_weight = min_weight
        self.occlusion_threshold = occlusion_threshold
        self.discard_black_frames = discard_black_frames
        self.black_frame_threshold = black_frame_threshold

        # --- Results Attributes ---
        self.results: Dict[str, Any] = {
            "time_series_data": {},
            "metadata": {}
        }
        self.blinking_pixels: Optional[List[Tuple[int, int]]] = None
        
    def _is_frame_black(self, frame: np.ndarray) -> bool:
        """Checks if a frame is considered black based on the mean intensity."""
        return frame.mean() < self.black_frame_threshold

    def run_analysis(self, led_on_method="bimodal", update_on_occlusion=True, show=False):
        """Executes the full, memory-efficient analysis pipeline."""
        logging.info(f"Starting analysis for '{self.video_manager.video_path.name}'...")
        
        self.locate_blinking_pixels()
        self.extract_led_signal() # Now also detects corrupted frames
        self.classify_led_state(method=led_on_method)
        self.detect_frame_occlusions()
        
        if update_on_occlusion:
            self.apply_occlusion_mask()
            
        self._populate_results_dictionary()

        self.time_series = self.results['time_series_data']
        self.metadata = self.results['metadata']

        if show:
            import matplotlib.pyplot as plt
            # Create a figure and a set of subplots with 2 rows and 1 column
            # sharex=True links the x-axes of both plots for easier comparison
            fig, axs = plt.subplots(2, 1, sharex=True)
            # --- Top Plot ---
            axs[0].plot(self.results['time_series_data']['led_on'])
            axs[0].set_title('LED Status Over Time')
            axs[0].set_ylabel('Status (On/Off)')
            # --- Bottom Plot ---
            axs[1].plot(self.results['time_series_data']['green_levels'])
            axs[1].set_title('Green Channel Intensity')
            axs[1].set_ylabel('Intensity')
            axs[1].set_xlabel('Time (frames/samples)')
            # Adjust layout to prevent titles/labels from overlapping
            plt.tight_layout()
            # Show the single window with both plots
            plt.show(block=True)
    
        logging.info("Analysis complete.")
        return self.time_series, self.metadata

    def locate_blinking_pixels(self, num_bins: int = 64):
        """
        Finds blinking pixels by building and analyzing histograms of pixel intensities.
        Skips corrupted (black) frames if configured to do so.
        """
        if self.video_manager.total_frames < 10:
            logging.warning("Insufficient frames for reliable analysis.")
            self.blinking_pixels = []
            return

        logging.info("Locating blinking pixels using memory-efficient histogram method...")
        nframes, height, width, _ = self.video_manager.shape
        pixel_histograms = np.zeros((height, width, num_bins), dtype=np.int32)
        bin_edges = np.linspace(0, 256, num_bins + 1)

        # Single pass through the video to build histograms
        for frame in self.video_manager:
            # **MODIFICATION**: Skip black frames from histogram calculation
            if self.discard_black_frames and self._is_frame_black(frame):
                continue
            
            green_channel = frame[:, :, 1]
            binned_data = np.digitize(green_channel, bins=bin_edges[:-1]) - 1
            for r in range(height):
                for c in range(width):
                    pixel_histograms[r, c, binned_data[r, c]] += 1

        # Analyze histograms to find bimodal ones
        self.blinking_pixels = []
        min_separation = self.min_separation_factor * 255.0
        
        pixel_iterator = range(height * width)
        if tqdm:
            pixel_iterator = tqdm(pixel_iterator, desc="Analyzing Pixel Histograms")

        for i in pixel_iterator:
            r, c = i // width, i % width
            hist = pixel_histograms[r, c, :]
            if np.sum(hist) < 10: continue

            data = np.repeat(bin_edges[:-1], hist).reshape(-1, 1)

            try:
                gmm1 = mixture.GaussianMixture(n_components=1, n_init=3).fit(data)
                gmm2 = mixture.GaussianMixture(n_components=2, n_init=3).fit(data)
                is_bic_lower = (gmm1.bic(data) - gmm2.bic(data)) > self.bic_diff_threshold
                means_separated = np.abs(gmm2.means_[0, 0] - gmm2.means_[1, 0]) > min_separation
                weights_valid = np.all(gmm2.weights_ > self.min_weight)

                if is_bic_lower and means_separated and weights_valid:
                    self.blinking_pixels.append((r, c))
            except ValueError:
                continue
        
        logging.info(f"Found {len(self.blinking_pixels)} blinking pixels.")

    def extract_led_signal(self):
        """
        Calculates green channel intensity from blinking pixels for each frame.
        Also identifies and flags corrupted frames.
        """
        ts_data = self.results["time_series_data"]
        num_frames = self.video_manager.total_frames
        
        if not self.blinking_pixels:
            logging.warning("No blinking pixels found. Green level signal will be empty.")
            ts_data['green_levels'] = np.full(num_frames, np.nan)
            ts_data['corrupted'] = np.zeros(num_frames, dtype=bool)
            return

        # **MODIFICATION**: Initialize with NaN to handle missing values from corrupted frames
        ts_data['green_levels'] = np.full(num_frames, np.nan, dtype=np.float64)
        ts_data['corrupted'] = np.zeros(num_frames, dtype=bool)
        
        rows, cols = zip(*self.blinking_pixels)
        rows_idx, cols_idx = np.array(rows), np.array(cols)

        logging.info(f"Extracting signal and identifying corrupted frames...")
        for idx, frame in enumerate(self.video_manager):
            # **MODIFICATION**: Check for black frames and flag them
            if self.discard_black_frames and self._is_frame_black(frame):
                ts_data['corrupted'][idx] = True
                continue  # green_level for this index remains NaN

            green_values = frame[rows_idx, cols_idx, 1]
            ts_data['green_levels'][idx] = np.max(green_values) if green_values.size > 0 else 0
        
        # Normalization function correctly ignores NaN values
        ts_data['green_levels'] = normalize_signal(ts_data['green_levels'])
        ts_data['time'] = np.linspace(0, (num_frames - 1) / self.video_manager.fps, num_frames)

    def classify_led_state(self, method: str = "bimodal", threshold: float = 0.5):
        """
        Classifies LED state (on/off). Handles NaN values from corrupted frames.
        """
        ts_data = self.results["time_series_data"]
        if 'green_levels' not in ts_data:
            raise RuntimeError("Run extract_led_signal() before classifying LED state.")

        green_levels = ts_data['green_levels']
        # Create a mask for valid (non-NaN, non-corrupted) data points
        valid_mask = ~np.isnan(green_levels)
        
        # Initialize led_on array with NaNs
        led_on = np.full(self.video_manager.total_frames, np.nan)

        if not np.any(valid_mask):
            logging.warning("No valid green level data to classify.")
            ts_data['led_on'] = led_on
            return

        valid_data = green_levels[valid_mask].reshape(-1, 1)

        if method == "bimodal":
            if len(valid_data) < 10:
                logging.warning("Not enough valid data points for bimodal classification.")
                # Fallback to thresholding on the few valid points
                predictions = (valid_data.flatten() > threshold).astype(int)
            else:
                gmm = mixture.GaussianMixture(n_components=2, n_init=3).fit(valid_data)
                on_component = np.argmax(gmm.means_)
                predictions = (gmm.predict(valid_data) == on_component).astype(int)
        elif method == "threshold":
            predictions = (valid_data.flatten() > threshold).astype(int)
        else:
            raise ValueError("Method must be 'bimodal' or 'threshold'")
        
        # Place predictions back into the full-size array
        led_on[valid_mask] = predictions
        ts_data['led_on'] = led_on


    def detect_frame_occlusions(self, z_score_threshold: float = 5.0):
        """
        Identifies occluded frames using a vectorized, CPU-only NumPy approach.
        """
        ts_data = self.results["time_series_data"]
        if 'led_on' not in ts_data:
            raise RuntimeError("Run classify_led_state() before detecting occlusions.")

        ts_data['occluded'] = np.zeros(self.video_manager.total_frames, dtype=bool)
        led_on_signal = ts_data['led_on']

        valid_mask = ~ts_data.get('corrupted', np.zeros_like(led_on_signal, dtype=bool))
        all_valid_indices = np.where(valid_mask)[0]

        # Create a map to find a frame's position within our loaded array
        index_map = {original_idx: new_idx for new_idx, original_idx in enumerate(all_valid_indices)}

        off_idx_original = np.where((led_on_signal == 0) & valid_mask)[0]
        on_idx_original = np.where((led_on_signal == 1) & valid_mask)[0]

        if len(off_idx_original) < 3 or len(on_idx_original) < 3:
            logging.warning("Cannot detect occlusions; insufficient valid frames.")
            return

        # --- Step 1: Bulk Load All Valid Frames into RAM ---
        # This is the single biggest performance improvement.
        logging.info(f"Bulk loading {len(all_valid_indices)} valid frames into memory...")
        # Using float32 is more memory and computationally efficient than the default float64
        all_valid_frames = np.stack(
            [self.video_manager[int(i)] for i in all_valid_indices], axis=0
        ).astype(np.float32)

        # --- Step 2: Get Subsets and Calculate Templates ---
        # Use fancy indexing instead of loops to get frame subsets.
        on_indices_in_array = [index_map[i] for i in on_idx_original]
        off_indices_in_array = [index_map[i] for i in off_idx_original]
        
        on_frames = all_valid_frames[on_indices_in_array]
        off_frames = all_valid_frames[off_indices_in_array]
        
        # Vectorized calculation of template frames
        frame_avg_off = np.mean(off_frames, axis=0)
        frame_avg_on = np.mean(on_frames, axis=0)

        # --- Step 3: Vectorized Stats & Threshold Calculation ---
        logging.info("Calculating statistical distribution with NumPy...")
        # Calculate Mean Absolute Difference for all frames in each group at once
        diffs_off = np.mean(np.abs(off_frames - frame_avg_off), axis=(1, 2, 3))
        diffs_on = np.mean(np.abs(on_frames - frame_avg_on), axis=(1, 2, 3))
        
        threshold_off = np.mean(diffs_off) + z_score_threshold * np.std(diffs_off)
        threshold_on = np.mean(diffs_on) + z_score_threshold * np.std(diffs_on)

        logging.info(f"Dynamic Occlusion Thresholds | OFF: {threshold_off:.2f}, ON: {threshold_on:.2f}")

        # --- Step 4: Fully Vectorized Outlier Detection ---
        # Perform the comparison on the ENTIRE array of frames in one operation.
        logging.info("Detecting occlusions with a single vectorized comparison...")
        all_diffs_off = np.mean(np.abs(all_valid_frames - frame_avg_off), axis=(1, 2, 3))
        all_diffs_on = np.mean(np.abs(all_valid_frames - frame_avg_on), axis=(1, 2, 3))
        
        # The core boolean mask calculation, done on all frames simultaneously
        is_occluded_mask = (all_diffs_off > threshold_off) & (all_diffs_on > threshold_on)
        
        # Use the boolean mask to get the original frame indices that are occluded
        occluded_original_indices = all_valid_indices[is_occluded_mask]
        
        # Update the final result array in one go
        ts_data['occluded'][occluded_original_indices] = True
    
    def apply_occlusion_mask(self):
        """Applies occlusion results to the led_on data, setting occluded frames to NaN."""
        ts_data = self.results["time_series_data"]
        if 'occluded' not in ts_data or 'led_on' not in ts_data:
            raise RuntimeError("Run detect_frame_occlusions() and classify_led_state() first.")
        
        occluded_idx = np.where(ts_data['occluded'])[0]
        if occluded_idx.size > 0:
            # led_on is already float and has NaNs from corrupted frames
            ts_data['led_on'][occluded_idx] = np.nan

    def _populate_results_dictionary(self):
        """Gathers all analysis results and metadata into the main results dictionary."""
        self.results["metadata"] = {
            "video_path": str(self.video_manager.video_path),
            "total_frames": self.video_manager.total_frames,
            "fps": self.video_manager.fps,
            "shape": self.video_manager.shape,
            "analysis_parameters": {
                "bic_diff_threshold": self.bic_diff_threshold,
                "min_separation_factor": self.min_separation_factor,
                "min_weight": self.min_weight,
                "occlusion_threshold": self.occlusion_threshold,
                "discard_black_frames": self.discard_black_frames,
                "black_frame_threshold": self.black_frame_threshold,
            },
            "blinking_pixel_count": len(self.blinking_pixels) if self.blinking_pixels else 0,
            # **MODIFICATION**: Add counts for corrupted and occluded frames
            "corrupted_frame_count": int(np.sum(self.results["time_series_data"].get('corrupted', []))),
            "occluded_frame_count": int(np.sum(self.results["time_series_data"].get('occluded', [])))
        }
