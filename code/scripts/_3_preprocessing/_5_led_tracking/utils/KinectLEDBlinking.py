import os
import platform
# Check if running on Windows before setting the variable
if platform.system() == 'Windows':
    # Set the environment variable OMP_NUM_THREADS
    # The warning suggested '10' for your specific case, try that first.
    # If issues persist or for a more conservative approach, try '1'.
    num_threads_to_set = '4' # Or '4', '2', '1' - Experiment if needed
    os.environ['OMP_NUM_THREADS'] = num_threads_to_set
    print(f"INFO: Setting OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']} to mitigate potential MKL memory leak on Windows.")

import cv2
import json
import csv
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import platform
from sklearn import mixture
import subprocess
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning

from .time_cost_function import time_it
from .waitforbuttonpress_popup import WaitForButtonPressPopup


class KinectLEDBlinking:
    def __init__(self, video_path, result_dir_path=None, result_file_name=None):
        self.video_path = video_path
        self.result_dir_path = result_dir_path
        self.result_file_name = result_file_name

        self.lower_green = np.array([40, 40, 40])
        self.upper_green = np.array([80, 255, 255])
        self.threshold_value = None

        self.roi_frames = None
        self.fps = None
        self.nframes = None

        self.time = []
        self.bimodal_pixels = None         # Initialize attribute to store bimodal indices
        self.green_levels = []
        self.led_on = []
        self.occluded = []

    # Method to check if the results have already been processed and saved
    def is_already_processed(self):
        csv = os.path.join(self.result_dir_path, self.result_file_name + ".csv")
        txt = os.path.join(self.result_dir_path, self.result_file_name + "_metadata.txt")
        return os.path.isfile(csv) and os.path.isfile(txt)

    def create_phantom_result_file(self, metadata_filename_abs):
        metadata_file = os.path.join(metadata_filename_abs)
        # Read the content of the file
        with open(metadata_file, 'r', encoding='utf-8') as file:
            data = file.read()

        # Parse the JSON content
        json_data = json.loads(data)

        # Extract the required values
        self.nframes = json_data['nframes']
        self.fps = json_data['fps']

        # generate nan values over the entire video as the data were occluded (out of bounds)
        self.green_levels = np.full(self.nframes, np.nan)
        self.led_on = np.full(self.nframes, np.nan)
        self.time = np.linspace(0, 1 / self.fps * (self.nframes - 1), self.nframes)

        self.video_path = "None"

    def load_video(self):
        # Clean area of interest variable
        self.roi_frames = []

        # COLOR/RGB channels (or stream) in Kinect videos are located in 0
        info = subprocess.run([
            *"ffprobe -v quiet -print_format json -show_format -show_streams".split(),
            self.video_path
        ], capture_output=True)
        info.check_returncode()
        stream_rgb = json.loads(info.stdout)["streams"][0]
        if stream_rgb["codec_type"] != "video":
            warnings.warn("PROBLEM: Expected stream for RGB video is not a video")
        # get mp4 channel's rgb
        index = stream_rgb["index"]
        # get frame X.Y
        frame_height = stream_rgb["height"]
        frame_width = stream_rgb["width"]
        # get fps
        frame_rate = stream_rgb['avg_frame_rate']
        num, denom = map(int, frame_rate.split('/'))
        self.fps = num / denom

        frame_shape = np.array([frame_height, frame_width, 3])
        nelem = frame_shape.prod()
        nframes = 0

        cmd = ["ffmpeg", "-i", self.video_path]
        cmd += "-map", f"0:{index}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
        with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
            while True:
                data = proc.stdout.read(nelem)  # One byte per each element
                if not data:
                    break
                nframes += 1
                frame = np.frombuffer(data, dtype=np.uint8).reshape(frame_shape)
                self.roi_frames.append(frame)

        self.roi_frames = np.array(self.roi_frames)
        self.nframes = self.roi_frames.shape[0]

    @time_it
    def find_bimodal_green_pixels(self, min_frames_threshold=10, min_unique_values=5, bic_diff_threshold=10, min_separation_factor=0.1, min_weight=0.1):
        """
        Analyzes green channel intensity of each pixel over time to find bimodal pixels.

        Uses Gaussian Mixture Models (GMM) to compare a 1-component vs 2-component
        model fit for each pixel's green intensity timeseries.

        Args:
            min_frames_threshold (int): Minimum number of frames required for analysis.
                                        Default: 10.
            min_unique_values (int): Minimum unique green intensity values a pixel needs
                                     to have over time to be considered. Helps filter out
                                     static or near-static pixels. Default: 5.
            bic_diff_threshold (float): Minimum BIC improvement needed for the 2-component
                                        model over the 1-component model to indicate
                                        bimodality. Lower BIC is better. Default: 10.
            min_separation_factor (float): Minimum required separation between the means
                                           of the two Gaussian components, expressed as a
                                           factor of the full intensity range (0-255).
                                           Helps ensure the modes are distinct. Default: 0.1.
            min_weight (float): Minimum weight (proportion) for each component in the
                                2-component GMM. Ensures both modes are substantially
                                represented. Default: 0.1.

        Returns:
            list: A list of tuples (row, col) representing the indices of pixels
                  identified as having bimodal green intensity distribution.
            None: If input conditions (e.g., number of frames) are not met.
        """
        if not hasattr(self, 'roi_frames') or self.roi_frames is None:
            print("Error: self.roi_frames is empty or not initialized.")
            return None

        num_frames = len(self.roi_frames)

        if num_frames < min_frames_threshold:
            print(f"Warning: Insufficient frames ({num_frames} found, {min_frames_threshold} required) for reliable bimodality analysis.")
            return [] # Return empty list if not enough frames

        # --- Input Validation ---
        try:
            # Get dimensions from the first frame, assume all are the same
            first_frame = self.roi_frames[0]
            if not isinstance(first_frame, np.ndarray):
                 raise TypeError("Frames must be NumPy arrays.")
            height, width, channels = first_frame.shape

            if channels < 3: # Need at least B, G, R
                 print(f"Error: Frames need at least 3 channels (BGR expected), but found {channels}.")
                 return None
            if height <= 0 or width <= 0:
                print(f"Error: Invalid frame dimensions ({height}x{width}).")
                return None

            # Stack frames into a single NumPy array for efficient slicing
            # This requires significant memory if frames are large or numerous
            print("Stacking frames into array (requires memory)...")
            frames_array = np.stack(self.roi_frames, axis=0) # Shape: (num_frames, height, width, channels)
            print("Stacking complete.")

        except (AttributeError, TypeError, ValueError) as e:
             print(f"Error processing input frames: {e}. Ensure self.roi_frames is a list of consistent NumPy arrays.")
             return None

        self.bimodal_pixels = []
        min_separation = min_separation_factor * 255.0 # Absolute separation threshold

        print(f"Analyzing {height}x{width} pixels over {num_frames} frames for green channel bimodality...")

        # Suppress ConvergenceWarning from GMM for cleaner output
        # These warnings indicate that the GMM fit might not be optimal for some pixels
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning, module='sklearn')

            for r in range(height):
                # Optional: Add progress indicator every 10% of rows
                if height > 10 and r > 0 and (r + 1) % max(1, height // 10) == 0:
                    progress = (r + 1) / height * 100
                    print(f"  Processing... {progress:.0f}% complete (Row {r+1}/{height})")

                for c in range(width):
                    # Extract the green channel time series for pixel (r, c)
                    # Green channel is index 1 in BGR format
                    pixel_timeseries = frames_array[:, r, c, 1].astype(np.float64)

                    # --- Pixel Pre-checks ---
                    # 1. Check for sufficient unique values
                    unique_values = np.unique(pixel_timeseries)
                    if len(unique_values) < min_unique_values:
                        continue # Skip pixels with too little variation

                    # --- GMM Fitting and Comparison ---
                    # Reshape for GMM input (n_samples, n_features=1)
                    data = pixel_timeseries.reshape(-1, 1)

                    try:
                        # Fit 1-component GMM
                        gmm1 = GaussianMixture(n_components=1, random_state=0, n_init=3, covariance_type='full').fit(data)
                        # Fit 2-component GMM
                        gmm2 = GaussianMixture(n_components=2, random_state=0, n_init=3, covariance_type='full').fit(data)

                        # Compare models using Bayesian Information Criterion (BIC)
                        # Lower BIC generally indicates a better model fit, penalizing complexity.
                        bic1 = gmm1.bic(data)
                        bic2 = gmm2.bic(data)

                        # --- Bimodality Criteria ---
                        # 1. BIC for GMM2 is significantly lower than GMM1
                        is_bic_lower = (bic1 - bic2) > bic_diff_threshold
                        # 2. The means of the two components are sufficiently separated
                        means_separated = np.abs(gmm2.means_[0, 0] - gmm2.means_[1, 0]) > min_separation
                        # 3. Both components have a minimum weight (proportion)
                        weights_valid = np.all(gmm2.weights_ > min_weight) and np.all(gmm2.weights_ < (1.0 - min_weight))

                        if is_bic_lower and means_separated and weights_valid:
                            self.bimodal_pixels.append((r, c))

                    except ValueError:
                        # GMM fitting can fail on degenerate data, even after checks.
                        # Silently skip these pixels, or add a print statement for debugging:
                        # print(f" GMM failed for pixel ({r},{c}) - Skipping")
                        continue

        print(f"\nAnalysis complete. Found {len(self.bimodal_pixels)} pixels with bimodal green channel distribution.")
        return self.bimodal_pixels


    # --- New Method for Monitoring Green Levels of Bimodal Pixels ---
    def monitor_bimodal_pixels_green(self):
        """
        Calculates the average green channel intensity per frame, considering
        only the pixels previously identified as bimodal.

        Assumes `self.find_bimodal_pixels()` has been run successfully and its
        results are stored in `self.bimodal_pixels`.

        Stores the results in `self.green_levels`.

        Returns:
            np.ndarray: An array containing the average green level of bimodal
                        pixels for each frame.
            None: If prerequisites aren't met (no frames, or bimodal pixels
                  not found).
        """
        if not hasattr(self, 'roi_frames') or self.roi_frames is None:
            print("Error: self.roi_frames is empty or not initialized. Cannot monitor green levels.")
            return None

        # Check if bimodal pixels have been identified and stored
        if not hasattr(self, 'bimodal_pixels') or self.bimodal_pixels is None:
            print("Error: Bimodal pixel indices not found in self.bimodal_pixels. "
                  "Run `find_bimodal_pixels()` first.")
            return None

        bimodal_indices = self.bimodal_pixels
        num_frames = len(self.roi_frames)
        # Initialize results array (or overwrite if run previously)
        self.green_levels = np.zeros(num_frames, dtype=np.float64)

        if not bimodal_indices: # Check if the list of bimodal indices is empty
            print("Warning: No bimodal pixels were identified. Average green level based on bimodal pixels will be zero.")
            # self.green_levels is already all zeros, so just return it.
            return self.green_levels

        # --- Use NumPy fancy indexing for efficiency ---
        try:
            # Separate row and column indices from the list of tuples
            # Ensure indices are integers
            rows = np.array([int(idx[0]) for idx in bimodal_indices])
            cols = np.array([int(idx[1]) for idx in bimodal_indices])

            print(f"Monitoring green levels for {len(bimodal_indices)} bimodal pixels across {num_frames} frames...")

            for idx, frame in enumerate(self.roi_frames):
                 # Ensure frame is a NumPy array (should be from __init__, but double-check)
                 current_frame = np.asarray(frame)

                 # Check frame dimensions consistency (optional but good practice)
                 if current_frame.shape[0] <= np.max(rows) or current_frame.shape[1] <= np.max(cols):
                     print(f"Error: Frame {idx} dimensions ({current_frame.shape[:2]}) are smaller than "
                           f"required by bimodal indices (max row={np.max(rows)}, max col={np.max(cols)}).")
                     self.green_levels.fill(np.nan) # Mark results as invalid
                     return None # Abort

                 # Extract green values (channel 1) at ALL bimodal pixel locations
                 # for the CURRENT frame using advanced indexing
                 green_values_at_bimodal_pixels = current_frame[rows, cols, 1]

                 # Calculate the mean for the current frame's bimodal pixels
                 if green_values_at_bimodal_pixels.size > 0:
                      # Calculate max, ensuring it's float64 for precision
                      avg_green = np.max(green_values_at_bimodal_pixels).astype(float)
                      self.green_levels[idx] = avg_green
                 # else:
                      # This case should not happen if bimodal_indices is not empty
                      # self.green_levels[idx] = 0 # Or np.nan if preferred

        except IndexError as e:
             # This error occurs if a row/col index is out of bounds for the frame dimensions
             print(f"Error: Bimodal pixel index out of bounds for frame {idx}. "
                   f"Details: {e}. Ensure frames used for monitoring are the same dimensions "
                   "as those used for finding bimodal pixels.")
             self.green_levels.fill(np.nan) # Mark results as invalid
             return None
        except Exception as e:
             print(f"An unexpected error occurred during monitoring: {e}")
             self.green_levels.fill(np.nan) # Mark results as invalid
             return None

        self.green_levels = normalize_signal(self.green_levels, dtype=np.ndarray)
        self.time = np.linspace(0, 1 / self.fps * (self.nframes - 1), self.nframes)
        
        print("Monitoring of bimodal pixels complete. Results stored in self.green_levels.")
        return self.green_levels


    @time_it
    def monitor_green_levels(self, show=False):
        if self.roi_frames is None:
            raise Exception("Error: Area of Interest was not initialized.")
        self.green_levels = np.zeros(self.nframes)

        bimodal_pixel_indices = self.find_bimodal_green_pixels()

        # Lists to store results needed for visualization
        processed_display_frames = [] # Store original frames for display
        processed_roi_greens = []     # Store the green channel of the masked ROI

        print("Processing frames...")
        # --- First Loop: Processing ---
        for idx, frame in enumerate(self.roi_frames):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
            # Apply mask: roi contains original colors where mask is white, black otherwise
            roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # Calculate average green intensity ONLY in the masked area
            green_channel_masked = roi_frame[:, :, 1][mask > 0]
            if green_channel_masked.size > 0:
                avg_green = np.mean(green_channel_masked)
            else:
                avg_green = 0
            self.green_levels[idx] = avg_green

            # Store data needed *only* if showing later
            if show:
                processed_display_frames.append(frame) # Store the original frame
                processed_roi_greens.append(roi_frame[:, :, 1].flatten()) # Store green channel of ROI

            # Optional: Add progress indicator for long processing
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{self.nframes} frames")
        print("Processing complete.")

        # --- Second Loop: Visualization (only if show is True) ---
        if show:
            print("Starting visualization...")
            fig = None
            ax = None
            try:
                # Initialize plot window
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                plt.ion() # Turn on interactive mode

                # Target delay for 30 Hz (in seconds)
                target_delay = 1.0 / 30.0

                for idx, (display_frame, intensity_values) in enumerate(zip(processed_display_frames, processed_roi_greens)):
                    start_time = time.time() # Record start time for rate limiting

                    # --- Plotting Logic ---
                    # 1. Clear previous plots
                    ax[0].cla()
                    ax[1].cla()

                    # 2. Display current frame on the left subplot
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    ax[0].imshow(frame_rgb)
                    ax[0].set_title(f'Frame {idx}')
                    ax[0].axis('off')

                    # 3. Plot intensity values from the stored roi's green channel
                    element_indices = np.arange(intensity_values.size)

                    ax[1].plot(element_indices, intensity_values)
                    ax[1].set_title('ROI Green Channel Intensity')
                    ax[1].set_xlabel('Pixel Index (Flattened)')
                    ax[1].set_ylabel('Intensity')
                    ax[1].set_ylim(0, 256) # Set Y-axis limits (0-255 for 8-bit intensity)

                    # Adjust layout
                    plt.tight_layout()

                    # Redraw the plot and process GUI events
                    plt.pause(0.001)

                    # --- Rate Limiting Logic ---
                    display_time = time.time() - start_time
                    wait_time = target_delay - display_time
                    if wait_time > 0:
                        time.sleep(wait_time)

                    # Check if the plot window was closed
                    if not plt.fignum_exists(fig.number):
                        print("Plot window closed by user.")
                        break
            finally:
                # Clean up plot window if it was created
                if fig and plt.fignum_exists(fig.number):
                     plt.ioff() # Turn off interactive mode
                     plt.close(fig) # Close the figure window
                print("Visualization finished.")

        self.green_levels = normalize_signal(self.green_levels, dtype=np.ndarray)
        self.time = np.linspace(0, 1 / self.fps * (self.nframes - 1), self.nframes)


    def process_led_on(self, method="bimodal", threshold=0.25):
        if method == "bimodal":
            self.process_led_on_bimodal()
        elif method == "threshold":
            self.process_led_on_threshold(threshold=threshold)

    def process_led_on_bimodal(self):
        x = self.green_levels
        # reshape 1D data to ensure gmm to function correctly --> X.shape = (n_samples, 1).
        x = x.reshape(-1, 1)

        # create GMM model object
        gmm = mixture.GaussianMixture(n_components=2, max_iter=1000, random_state=10, covariance_type='full')

        try:
            mean = gmm.fit(x).means_
            print("GMM fitting successful with regularization.")
            # Note: The results might not be very meaningful if all data is identical.
            print("Means:", gmm.means_)
            print("Covariances:", gmm.covariances_)
        except Exception as e:
            print(f"GMM fitting failed even with regularization: {e}. This means there is very little variance between the two components. Process with standard thresholding instead.")
            self.process_led_on_threshold()
            return

        # find useful parameters
        gauss_idx_predicted = gmm.fit(x).predict(x)

        # if the first gaussian is the low green intensity gaussian,
        # label prediction works perfectly (0 or 1)
        if mean[0][0] < mean[1][0]:
            self.led_on = gauss_idx_predicted
        else:  # otherwise, reverse ON/OFF
            self.led_on = 1 - gauss_idx_predicted

        # The treshold could be estimated to the intersection rather than putting it to NaN
        # https://stats.stackexchange.com/questions/311592/how-to-find-the-point-where-two-normal-distributions-intersect
        self.threshold_value = np.nan

    def process_led_on_threshold(self, threshold=0.25):
        if self.green_levels is None:
            raise Exception("Error: green_levels was not initialized.")
        self.led_on = np.zeros(self.nframes)
        # put to one when green level is above the threshold
        self.threshold_value = threshold
        self.green_levels = normalize_signal(self.green_levels, dtype=np.ndarray)
        led_on_idx = np.where(self.green_levels > self.threshold_value)[0]
        self.led_on[led_on_idx] = 1

    def define_occlusion(self, threshold=40, show=False):
        if self.roi_frames is None or self.led_on is None:
            raise Exception("Error: led_on was not initialized.")
        self.occluded = np.full(self.nframes, False, dtype=bool)
        frame_avg_off_idx = np.where(self.led_on == 0)[0]
        frame_avg_on_idx = np.where(self.led_on == 1)[0]
        frame_avg_off = np.round(np.mean(self.roi_frames[frame_avg_off_idx, :, :, :], axis=0)).astype(int)
        frame_avg_on = np.round(np.mean(self.roi_frames[frame_avg_on_idx, :, :, :], axis=0)).astype(int)
        means_off = []
        means_on = []
        for idx, frame in enumerate(self.roi_frames):
            frame_off = np.subtract(frame, frame_avg_off)
            frame_on = np.subtract(frame, frame_avg_on)
            mean_off = abs(np.mean(frame_off))
            mean_on = abs(np.mean(frame_on))
            means_off.append(mean_off)
            means_on.append(mean_on)
            if not (mean_off < threshold) or not (mean_on < threshold):
                self.occluded[idx] = True
        if show:
            fig, axs = plt.subplots(3, 1, sharex=True)
            axs[0].plot(means_off, label='LED OFF')
            axs[0].axhline(threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
            axs[0].set_ylabel('average pixels value of the frame')
            axs[0].legend()
            axs[1].plot(means_on, label='LED OFF')
            axs[1].axhline(threshold, color='g', linestyle=':', label=f'Threshold={threshold}')
            axs[1].set_xlabel('frame')
            axs[1].set_ylabel('average pixels value of the frame')
            axs[1].legend()
            axs[2].plot(self.occluded, label='TRUE/FALSE OCCLUDED VECTOR')
            plt.tight_layout()
            plt.ion()
            plt.show()
            plt.pause(0.001)
            WaitForButtonPressPopup()
            plt.close()
            for idx, occlusion in enumerate(self.occluded):
                if occlusion:
                    plt.imshow(self.roi_frames[idx, :, :, :])
                    plt.ion()
                    plt.draw()
                    plt.pause(0.001)
                    WaitForButtonPressPopup()
            plt.close('all')
        if np.any(self.occluded):
            print("Some occlusions have been detected.")
            if show:
                update_result = None
                confirmation_window = np.zeros((200, 500, 3), dtype=np.uint8)
                cv2.putText(confirmation_window, "Update led_on?", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)
                cv2.rectangle(confirmation_window, (50, 100), (150, 150), (0, 255, 0), -1)
                cv2.putText(confirmation_window, "Yes", (70, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.rectangle(confirmation_window, (250, 100), (350, 150), (0, 0, 255), -1)
                cv2.putText(confirmation_window, "No", (270, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.imshow('Confirmation', confirmation_window)

                def confirm_square(event, x, y, flags, param):
                    nonlocal update_result
                    if event == cv2.EVENT_LBUTTONUP:
                        if 50 < x < 150 and 100 < y < 150:
                            update_result = True
                            cv2.destroyAllWindows()
                        elif 250 < x < 350 and 100 < y < 150:
                            update_result = False
                            cv2.destroyAllWindows()

                cv2.setMouseCallback('Confirmation', confirm_square)
                while update_result is None:
                    cv2.waitKey(1)
                cv2.destroyAllWindows()
            else:
                update_result = True
            if update_result:
                self.update_led_on()

    def update_led_on(self):
        if self.occluded.size == 0:
            raise Exception("Error: occluded was not initialized.")
        occluded_idx = np.where(self.occluded == 1)[0]
        if len(occluded_idx):
            # convert to float array to allow nan values
            self.led_on = np.array(self.led_on, dtype=float)
            self.led_on[occluded_idx] = np.nan

    def save_results(self):
        if not os.path.exists(self.result_dir_path):
            os.makedirs(self.result_dir_path)
        filename_abs = os.path.join(self.result_dir_path, self.result_file_name + ".csv")
        self.save_result_csv(filename_abs)
        filename_abs = os.path.join(self.result_dir_path, self.result_file_name + "_metadata.txt")
        self.save_result_metadata(filename_abs)

    def save_result_csv(self, file_path_abs):
        with open(file_path_abs, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["time (second)", "green level", "LED on"])
            for t_value, green_value, led_value in zip(self.time, self.green_levels, self.led_on):
                writer.writerow([t_value, green_value, led_value])

    def save_result_metadata(self, file_path_abs):
        metadata = {
            "video_path": self.video_path,
            "lower_green": self.lower_green.tolist(),
            "upper_green": self.upper_green.tolist(),
            "threshold_value": self.threshold_value,
            "fps": self.fps,
            "nframes": self.nframes
        }
        with open(file_path_abs, 'w') as f:
            json.dump(metadata, f, indent=4)




def normalize_signal(signal, dtype=list):
    if not len(signal):
        if dtype == np.ndarray:
            return np.array(signal)
        else:
            return signal
    min_val = np.nanmin(signal)
    max_val = np.nanmax(signal)
    normalized_signal = [(x - min_val) / (max_val - min_val) for x in signal]
    
    if dtype == np.ndarray:
        normalized_signal = np.array(normalized_signal)

    return normalized_signal
