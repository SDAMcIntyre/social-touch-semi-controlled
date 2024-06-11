import csv
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.interpolate import interp1d

from ..misc.time_cost_function import time_it
from ..misc.waitforbuttonpress_popup import WaitForButtonPressPopup


# class of the LED green value of videos for the preprocessing
class SemiControlledDataLED:
    def __init__(self):
        self.session = []
        self.block_id = []
        self.file_path = []
        self.timeseries_filename = []
        self.metadata_filename = []

        self.time = []
        self.green_levels = []
        self.led_on = []

    def load_timeseries(self, led_files_info):
        self.session = led_files_info["session"]
        self.block_id = led_files_info["block_id"]
        self.file_path = led_files_info["file_path"]
        self.timeseries_filename = led_files_info["timeseries_filename"]
        self.metadata_filename = led_files_info["metadata_filename"]

        df = pd.read_csv(os.path.join(led_files_info["file_path"], led_files_info["timeseries_filename"]))
        df.dropna(inplace=True)  # remove lines that contains NaN values
        self.time = [round(num, 5) for num in df["time (second)"].values]
        self.green_levels = [round(num, 5) for num in df["green level"].values]
        self.led_on = df["LED on"].values

    def load_class_list_from_infos(self, led_files_info_list):
        data_led_list = []

        for led_files_info in led_files_info_list:
            scdl = SemiControlledDataLED()
            scdl.load_timeseries(led_files_info)
            data_led_list.append(scdl)

        return data_led_list

    def resample(self, new_time, show=False):
        if show:
            plt.figure()
            plt.plot(self.time, self.green_levels)
            plt.plot(self.time, self.led_on)

        # reset potential non start to zero.
        new_time = new_time - new_time[0]

        # Find the index of the first 1
        start = 0
        while start < len(self.led_on) and self.led_on[start] == 0:
            start += 1
        # Find the index of the last 1
        end = len(self.led_on) - 1
        while end >= 0 and self.led_on[end] == 0:
            end -= 1
        time_essential = self.time[start:end] - self.time[start]

        # display basic info
        print("start/end: original:({:.2f}, {:.2f}), essential:({:.2f}, {:.2f}), target:({:.2f}, {:.2f})"
              .format(self.time[0], self.time[-1], time_essential[0], time_essential[-1], new_time[0], new_time[-1]))
        print("nb. element ---> original:{:.2f}, target:{:.2f}, ratio:{:.2f} (expected ~0.03)".format(len(self.time), len(new_time), len(self.time)/len(new_time)))

        # /!\
        # /!\ FOR SOME REASON, TIME DON'T MATCH BETWEEN SHAN'S CSV AND KINECT VIDEO (MY EXTRACTION)
        # /!\
        # Artificially make them match
        # could include a PB bc the rows with NaN values have been removed during load_dataframe()
        new_time = np.linspace(self.time[0], self.time[-1], len(new_time))

        # Create interpolation functions
        interp_func_led_on = interp1d(self.time, self.led_on, kind='nearest')  # 'linear', 'quadratic', 'cubic'
        interp_func_green_levels = interp1d(self.time, self.green_levels, kind='linear')  # 'linear', 'quadratic', 'cubic'

        # Interpolate the values at the new time points
        self.led_on = interp_func_led_on(new_time)
        self.green_levels = interp_func_green_levels(new_time)

        # replace the old time vector by the new one
        self.time = new_time

        if show:
            plt.figure()
            plt.plot(self.time, self.green_levels)
            plt.plot(self.time, self.led_on)
            plt.show()
            WaitForButtonPressPopup()



# class to extract the LED green value of videos
class KinectLED:
    def __init__(self, video_path=None):
        self.cap = None
        self.circle_center = None
        self.circle_radius = None
        self.drawing = False
        self.confirmation_result = None
        self.lower_green = np.array([40, 40, 40])
        self.upper_green = np.array([80, 255, 255])
        self.first_frame = None
        self.background_mask = None
        self.threshold_value = None

        # video info
        self.frame_width = None
        self.frame_height = None
        self.fps = None
        self.nframes = None
        self.fourcc_str = None

        # results
        self.green_levels = []
        self.led_on = []
        self.time = []

        # initialise directly
        if video_path is not None:
            self.initialise(video_path)

    def initialise(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise Exception("Error: Could not open video file.")

        ret, self.first_frame = self.cap.read()
        if not ret:
            raise Exception("Error: Could not read the first frame.")

        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        # Convert fourcc to a string
        self.fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    def draw_circle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.circle_center = (x, y)
            self.circle_radius = 0

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.circle_radius = int(np.sqrt((x - self.circle_center[0])**2 + (y - self.circle_center[1])**2))

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.circle_radius = int(np.sqrt((x - self.circle_center[0])**2 + (y - self.circle_center[1])**2))

    def select_led(self):
        if self.first_frame is None:
            raise Exception("Error: First frame not available. Please initialize the video first.")

        while True:
            cv2.namedWindow('First Frame', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('First Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback('First Frame', self.draw_circle)

            while True:
                frame_copy = self.first_frame.copy()
                if self.circle_center and self.circle_radius:
                    cv2.circle(frame_copy, self.circle_center, self.circle_radius, (0, 255, 0), 2)
                cv2.imshow('First Frame', frame_copy)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or (not self.drawing and self.circle_center and self.circle_radius):
                    break

            cv2.destroyAllWindows()

            if not self.circle_center or not self.circle_radius:
                raise Exception("Error: No circle was drawn.")

            # Display the confirmation window with buttons
            self.confirmation_result = None
            confirmation_window = np.zeros((200, 500, 3), dtype=np.uint8)
            cv2.putText(confirmation_window, "Is the circle set correctly?", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
            cv2.rectangle(confirmation_window, (50, 100), (150, 150), (0, 255, 0), -1)
            cv2.putText(confirmation_window, "Yes", (70, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.rectangle(confirmation_window, (250, 100), (350, 150), (0, 0, 255), -1)
            cv2.putText(confirmation_window, "No", (270, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow('Confirmation', confirmation_window)

            def confirm_circle(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    if 50 < x < 150 and 100 < y < 150:
                        self.confirmation_result = True
                    elif 250 < x < 350 and 100 < y < 150:
                        self.confirmation_result = False
                    cv2.destroyAllWindows()

            cv2.setMouseCallback('Confirmation', confirm_circle)

            while self.confirmation_result is None:
                cv2.waitKey(1)

            if self.confirmation_result:
                break
            else:
                self.circle_center = None
                self.circle_radius = None

        # Create the background mask based on the selected circle
        self.background_mask = np.zeros_like(self.first_frame)
        cv2.circle(self.background_mask, self.circle_center, self.circle_radius, (255, 255, 255), -1)
        self.background_mask = cv2.bitwise_and(self.first_frame, self.background_mask)

    @time_it
    def monitor_green_levels(self, threshold, show=False):
        if self.cap is None or self.first_frame is None:
            raise Exception("Error: Video not initialized. Please call initialise() first.")

        # Reset to the first frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if show:
            cv2.namedWindow('Frame', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        nframes_predicted = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        nframes = 0
        while True:
            correctly_read, frame = self.cap.read()
            if not correctly_read:
                break
            nframes += 1
            print("Frame:", nframes, "/", nframes_predicted)

            # Subtract the background
            foreground = cv2.absdiff(frame, self.background_mask)

            # Convert the frame to the HSV color space
            hsv = cv2.cvtColor(foreground, cv2.COLOR_BGR2HSV)

            # Create a mask for the green color
            mask = cv2.inRange(hsv, self.lower_green, self.upper_green)

            # Create a mask for the circular ROI
            circle_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.circle(circle_mask, self.circle_center, self.circle_radius, 255, -1)

            # Combine the green mask and the circle mask
            combined_mask = cv2.bitwise_and(mask, mask, mask=circle_mask)

            # Calculate the average green level in the circular AOI
            AOI = cv2.bitwise_and(frame, frame, mask=combined_mask)
            avg_green = np.mean(AOI[:, :, 1])  # Average of the green channel

            # Store the average green level
            self.green_levels.append(avg_green)

            if show:
                # Draw the circle on the frame (optional)
                cv2.circle(frame, self.circle_center, self.circle_radius, (0, 255, 0), 2)
                # Display the frame (optional)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cap.release()
        cv2.destroyAllWindows()

        # define the ON/OFF time series
        self.threshold_value = threshold
        self.led_on = [green > self.threshold_value for green in self.green_levels]

        self.nframes = nframes
        # Create the vector dt of delta times
        self.time = np.linspace(0, 1/self.fps * (self.nframes - 1), self.nframes)

    def get_green_levels(self):
        return self.green_levels

    def save(self, file_path, file_name):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            print(f"Directory '{file_path}' created.")
        else:
            print(f"Directory '{file_path}' already exists.")

        self.save_csv(file_path, file_name+".csv")
        self.save_metadata(file_path, file_name+"_metadata.txt")

    def save_csv(self, file_path, file_name):
        full_path = f"{file_path}/{file_name}"
        with open(full_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["time (second)", "green level", "LED on"])
            for t_value, green_value, led_value in zip(self.time, self.green_levels, self.led_on):
                writer.writerow([t_value, green_value, led_value])

    def save_metadata(self, file_path, file_name):
        full_path = f"{file_path}/{file_name}"

        # Convert numpy arrays to lists, as they are not JSON serializable
        metadata = {
            "video_path": self.video_path,
            "circle_center": self.circle_center,
            "circle_radius": self.circle_radius,
            "lower_green": self.lower_green.tolist(),
            "upper_green": self.upper_green.tolist(),
            "threshold_value": self.threshold_value,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "fps": self.fps,
            "nframes": self.nframes,
            "fourcc_str": self.fourcc_str
        }

        # Write the metadata to a JSON file
        with open(full_path, 'w') as f:
            json.dump(metadata, f, indent=4)
