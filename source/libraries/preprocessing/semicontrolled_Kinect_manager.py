import csv
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from ..misc.time_cost_function import time_it
from ..misc.waitforbuttonpress_popup import WaitForButtonPressPopup
from .semicontrolled_data_cleaning import normalize_signal


# Class to extract the LED green value of videos
class ProcessKinectLED:
    def __init__(self, video_path, result_dir_path, result_file_name):
        self.cap = None
        self.square_center = None
        self.square_size = None
        self.drawing = False
        self.confirmation_result = None
        self.lower_green = np.array([40, 40, 40])
        self.upper_green = np.array([80, 255, 255])

        # Reference frame and square mask frame
        self.reference_frame_idx = None
        self.reference_frame = None
        self.background_mask = None

        # Video info
        self.frame_width = None
        self.frame_height = None
        self.fps = None
        self.nframes = None
        self.fourcc_str = None

        # Results
        self.led_in_frame = None
        self.threshold_value = None
        self.aoi = None  # Area of Interest
        self.occluded = []
        self.green_levels = []
        self.led_on = []
        self.time = []

        # Save location and filename
        self.result_dir_path = result_dir_path
        self.result_file_name = result_file_name

        # Initialise video
        self.video_path = video_path

    # Method to initialise the video capture
    def initialise_video(self):
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            raise Exception("Error: Could not open video file.")

        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        # Convert fourcc to a string
        self.fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    # Method to define the reference frame by frame index
    def define_reference_frame(self, frame_idx):
        self.reference_frame_idx = frame_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, self.reference_frame = self.cap.read()
        if not ret:
            raise Exception("Error: Could not read the first frame.")

    # Method to select a good frame from the video
    def select_good_frame(self):
        if self.cap is None:
            raise Exception("Error: Video not initialized. Please call initialise() first.")

        def display_frame(frame, window_name="Frame"):
            cv2.imshow(window_name, frame)
            cv2.waitKey(1)

        # Display the first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = self.cap.read()
        if not ret:
            raise Exception("Error: Could not read the first frame.")

        display_frame(first_frame, "First Frame")

        # Confirmation window for user input
        confirmation_result = None
        confirmation_window = np.zeros((200, 500, 3), dtype=np.uint8)
        cv2.putText(confirmation_window, "Use this frame?", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(confirmation_window, (50, 100), (150, 150), (0, 255, 0), -1)
        cv2.putText(confirmation_window, "Yes", (70, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(confirmation_window, (250, 100), (350, 150), (0, 0, 255), -1)
        cv2.putText(confirmation_window, "No", (270, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow('Confirmation', confirmation_window)

        # Callback function for confirmation window
        def confirm_frame(event, x, y, flags, param):
            nonlocal confirmation_result
            if event == cv2.EVENT_LBUTTONUP:
                if 50 < x < 150 and 100 < y < 150:
                    confirmation_result = True
                    cv2.destroyAllWindows()
                elif 250 < x < 350 and 100 < y < 150:
                    confirmation_result = False
                    cv2.destroyAllWindows()

        cv2.setMouseCallback('Confirmation', confirm_frame)

        while confirmation_result is None:
            cv2.waitKey(1)

        if confirmation_result:
            return 0  # The first frame was chosen

        # Display 19 frames linearly spread over the entire video
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames / 2 - 1, 19, dtype=int)

        miniatures = []
        for i, frame_idx in enumerate(frame_indices):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                continue
            thumbnail = cv2.resize(frame, (160 * 4, 90 * 4))  # Create thumbnail
            miniatures.append((frame_idx, thumbnail))

        # Create a montage of thumbnails
        miniature_height = 90 * 4
        miniature_width = 160 * 4
        montage = np.zeros((miniature_height * 4, miniature_width * 5, 3), dtype=np.uint8)

        for i, (_, thumbnail) in enumerate(miniatures):
            row = i // 5
            col = i % 5
            montage[row * miniature_height:(row + 1) * miniature_height,
            col * miniature_width:(col + 1) * miniature_width] = thumbnail

        # Add "None" area for user to select
        none_area = np.zeros((miniature_height, miniature_width, 3), dtype=np.uint8)
        cv2.putText(none_area, "None of them", (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        montage[3 * miniature_height:(4) * miniature_height, 4 * miniature_width:(5) * miniature_width] = none_area

        # Display the montage and handle user selection
        cv2.namedWindow('Select Frame', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Select Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Select Frame', montage)

        selected_frame_idx = None

        def select_miniature(event, x, y, flags, param):
            nonlocal selected_frame_idx
            if event == cv2.EVENT_LBUTTONUP:
                frame_number = (y // miniature_height) * 5 + (x // miniature_width)
                if frame_number < len(miniatures):
                    selected_frame_idx = miniatures[frame_number][0]
                    cv2.destroyAllWindows()
                elif frame_number == 19:  # None area selected
                    selected_frame_idx = -1
                    cv2.destroyAllWindows()

        cv2.setMouseCallback('Select Frame', select_miniature)

        while selected_frame_idx is None:
            cv2.waitKey(1)

        if selected_frame_idx == -1:
            return None

        # Final confirmation for selected frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, selected_frame_idx)
        ret, selected_frame = self.cap.read()
        if not ret:
            return False

        display_frame(selected_frame, "Selected Frame")

        confirmation_result = None
        cv2.imshow('Confirmation', confirmation_window)
        cv2.setMouseCallback('Confirmation', confirm_frame)

        while confirmation_result is None:
            cv2.waitKey(1)

        if confirmation_result:
            return selected_frame_idx
        else:
            return None

    # Method to draw a square using mouse events
    def draw_square(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.square_center = (x, y)
            self.square_size = 0
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.square_size = int(max(abs(x - self.square_center[0]), abs(y - self.square_center[1])) * 2)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.square_size = int(max(abs(x - self.square_center[0]), abs(y - self.square_center[1])) * 2)

    # Method to select LED location with a square
    def select_led_location(self, square_center=None, square_size=None):
        if square_center is not None and square_size is not None:
            self.square_center = square_center
            self.square_size = square_size
            return [self.square_center, self.square_size]

        if self.reference_frame is None:
            raise Exception("Error: First frame not available. Please initialize the video first.")

        while True:
            cv2.namedWindow('First Frame', cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback('First Frame', self.draw_square)

            while True:
                frame_copy = self.reference_frame.copy()
                if self.square_center and self.square_size:
                    top_left = (
                    self.square_center[0] - self.square_size // 2, self.square_center[1] - self.square_size // 2)
                    bottom_right = (
                    self.square_center[0] + self.square_size // 2, self.square_center[1] + self.square_size // 2)
                    cv2.rectangle(frame_copy, top_left, bottom_right, (0, 255, 0), 2)
                cv2.imshow('First Frame', frame_copy)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or (not self.drawing and self.square_center and self.square_size):
                    break

            cv2.destroyAllWindows()

            if not self.square_center or not self.square_size:
                raise Exception("Error: No square was drawn.")

            # Display confirmation window
            self.confirmation_result = None
            confirmation_window = np.zeros((200, 500, 3), dtype=np.uint8)
            cv2.putText(confirmation_window, "Is the square set correctly?", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
            cv2.rectangle(confirmation_window, (50, 100), (150, 150), (0, 255, 0), -1)
            cv2.putText(confirmation_window, "Yes", (70, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.rectangle(confirmation_window, (250, 100), (350, 150), (0, 0, 255), -1)
            cv2.putText(confirmation_window, "No", (270, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow('Confirmation', confirmation_window)

            def confirm_square(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONUP:
                    if 50 < x < 150 and 100 < y < 150:
                        self.confirmation_result = True
                        cv2.destroyAllWindows()
                    elif 250 < x < 350 and 100 < y < 150:
                        self.confirmation_result = False
                        cv2.destroyAllWindows()

            cv2.setMouseCallback('Confirmation', confirm_square)

            while self.confirmation_result is None:
                cv2.waitKey(1)

            if self.confirmation_result:
                break
            else:
                self.square_center = None
                self.square_size = None

        return [self.square_center, self.square_size]

    @time_it
    def extract_aoi(self):
        if self.cap is None or self.reference_frame is None:
            raise Exception("Error: Video not initialized. Please call initialise() first.")

        # Reset the area of interest
        self.aoi = []

        # Define the square region
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, frame = self.cap.read()
        half_size = self.square_size // 2
        x_start = max(self.square_center[0] - half_size, 0)
        x_end = min(self.square_center[0] + half_size, frame.shape[1])
        y_start = max(self.square_center[1] - half_size, 0)
        y_end = min(self.square_center[1] + half_size, frame.shape[0])

        # Variables for progression and frame counting
        nframes_predicted = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progression_list = np.linspace(0, 100, 20)
        progression_idx = 0
        nframes = 0

        # Reset to the first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Extract the square region
        while self.cap.isOpened():
            correctly_read, frame = self.cap.read()
            if not correctly_read:
                break
            nframes += 1
            if (100 * nframes / nframes_predicted) > progression_list[progression_idx]:
                print("Create Area of Interest> Progression:", int(progression_list[progression_idx]), "% ( Frame:",
                      nframes, "/",
                      nframes_predicted, ")")
                progression_idx += 1
            self.aoi.append(frame[y_start:y_end, x_start:x_end])

        # convert into array
        self.aoi = np.array(self.aoi)

        # take advantage to store the correct number of frames
        self.nframes = nframes

        self.cap.release()

    # Method to monitor green levels in the selected area
    @time_it
    def monitor_green_levels(self, show=False):
        if self.aoi is None:
            raise Exception("Error: Area of Interest was not initialized. Please call extract_aoi() first.")

        self.green_levels = np.zeros(self.nframes)

        if show:
            cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

        for idx, frame in enumerate(self.aoi):
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # Create a mask for the green color
            mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
            # Calculate the average green level in the square AOI
            AOI = cv2.bitwise_and(frame, frame, mask=mask)
            avg_green = np.mean(AOI[:, :, 1])
            # Store the average green level
            self.green_levels[idx] = avg_green

            if show:
                # Display the frame (optional)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Normalize the green levels between 0 and 1
        self.green_levels = normalize_signal(self.green_levels, dtype=np.ndarray)
        # Create the vector time
        self.time = np.linspace(0, 1 / self.fps * (self.nframes - 1), self.nframes)

    # Method to process LED on/off based on a threshold value
    def process_led_on(self, threshold=0.25):
        if self.green_levels is None:
            raise Exception("Error: green_levels was not initialized. Please call monitor_green_levels().")

        # initialise occluded variable
        self.led_on = np.zeros(self.nframes)

        self.threshold_value = threshold

        self.green_levels = normalize_signal(self.green_levels, dtype=np.ndarray)

        led_on_idx = np.where(self.green_levels > self.threshold_value)[0]

        self.led_on[led_on_idx] = 1

    def define_occlusion(self, threshold=40, show=False):
        if self.aoi is None or self.led_on is None:
            raise Exception("Error: led_on was not initialized. Please call monitor_green_levels().")

        # initialise occluded variable
        self.occluded = np.full(self.nframes, False, dtype=bool)

        frame_avg_off_idx = np.where(self.led_on == False)[0]
        frame_avg_on_idx = np.where(self.led_on == True)[0]

        frame_avg_off = np.round(np.mean(self.aoi[frame_avg_off_idx, :, :, :], axis=0)).astype(int)
        frame_avg_on = np.round(np.mean(self.aoi[frame_avg_on_idx, :, :, :], axis=0)).astype(int)

        means_off = []
        means_on = []

        for idx, frame in enumerate(self.aoi):
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
            # Plot ax1
            axs[0].plot(means_off, label='LED OFF')
            # Add horizontal line at threshold on ax1
            axs[0].axhline(threshold, color='r', linestyle='--', label=f'Threshold={threshold}')
            axs[0].set_ylabel('average pixels value of the frame')
            axs[0].legend()
            # Plot ax2
            axs[1].plot(means_on, label='LED OFF')
            # Add horizontal line at threshold on ax2
            axs[1].axhline(threshold, color='g', linestyle=':', label=f'Threshold={threshold}')
            axs[1].set_xlabel('frame')
            axs[1].set_ylabel('average pixels value of the frame')
            axs[1].legend()
            # Plot ax3
            axs[2].plot(self.occluded, label='TRUE/FALSE OCCLUDED VECTOR')
            # Adjust layout to prevent overlap of labels
            plt.tight_layout()
            # Display the plot
            plt.ion()
            plt.show()
            plt.pause(0.001)
            WaitForButtonPressPopup()
            plt.close()

            for idx, occlusion in enumerate(self.occluded):
                if occlusion:
                    plt.imshow(self.aoi[idx, :, :, :])
                    plt.ion()
                    plt.draw()
                    plt.pause(0.001)
                    WaitForButtonPressPopup()
            plt.close('all')

        # if there is at least one occluded value do update process
        # If show is True, ask for update
        if np.any(self.occluded):
            print("some occlusions have been detected.")
            if show:
                # Display confirmation window
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
                    nonlocal update_result  # Use nonlocal to modify update_result from the enclosing scope
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
            raise Exception("Error: occluded was not initialized. Please call define_occlusion().")
        occluded_idx = np.where(self.occluded == True)[0]
        self.led_on[occluded_idx] = np.nan

    # Method to check if the results have already been processed and saved
    def is_already_processed(self):
        csv = os.path.join(self.result_dir_path, self.result_file_name + ".csv")
        txt = os.path.join(self.result_dir_path, self.result_file_name + "_metadata.txt")
        return os.path.isfile(csv) and os.path.isfile(txt)

    # Method to load previously saved results
    def load_results(self, dropna=False):
        csv_file = os.path.join(self.result_dir_path, self.result_file_name + ".csv")
        df = pd.read_csv(csv_file)
        if dropna:
            df.dropna(inplace=True)
        self.time = [round(num, 5) for num in df["time (second)"].values]
        self.green_levels = [round(num, 5) for num in df["green level"].values]
        self.led_on = df["LED on"].values

        metadata_file = os.path.join(self.result_dir_path, self.result_file_name + "_metadata.txt")
        with open(metadata_file, 'r') as jsonfile:
            data = json.load(jsonfile)
            self.video_path = data.get("video_path", "")
            self.reference_frame_idx = data.get("reference_frame_idx", "")
            self.led_in_frame = data.get("led_in_frame", "")
            self.square_center = data.get("square_center", [])
            self.square_size = data.get("square_size", 0)
            self.lower_green = np.array(data.get("lower_green", []))
            self.upper_green = np.array(data.get("upper_green", []))
            self.threshold_value = data.get("threshold_value", 0.0)
            self.frame_width = data.get("frame_width", 0)
            self.frame_height = data.get("frame_height", 0)
            self.fps = data.get("fps", 0.0)
            self.nframes = data.get("nframes", 0)
            self.fourcc_str = data.get("fourcc_str", "")

    def save_results(self):
        if not os.path.exists(self.result_dir_path):
            os.makedirs(self.result_dir_path)
            print(f"Directory '{self.result_file_name}' created.")
        else:
            print(f"Directory '{self.result_file_name}' already exists.")

        self.save_result_csv(self.result_dir_path, self.result_file_name + ".csv")
        self.save_result_metadata(self.result_dir_path, self.result_file_name + "_metadata.txt")

    def save_result_csv(self, file_path, file_name):
        full_path = os.path.join(file_path, file_name)
        with open(full_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["time (second)", "green level", "LED on"])
            for t_value, green_value, led_value in zip(self.time, self.green_levels, self.led_on):
                writer.writerow([t_value, green_value, led_value])

    def save_result_metadata(self, file_path, file_name):
        full_path = os.path.join(file_path, file_name)

        metadata = {
            "video_path": self.video_path,
            "led_in_frame": self.led_in_frame,
            "reference_frame_idx": int(self.reference_frame_idx),
            "square_center": self.square_center,
            "square_size": self.square_size,
            "lower_green": self.lower_green.tolist(),
            "upper_green": self.upper_green.tolist(),
            "threshold_value": self.threshold_value,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "fps": self.fps,
            "nframes": self.nframes,
            "fourcc_str": self.fourcc_str
        }

        with open(full_path, 'w') as f:
            json.dump(metadata, f, indent=4)
