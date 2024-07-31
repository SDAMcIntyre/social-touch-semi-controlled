import os
import cv2
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import mixture
import subprocess
import warnings

from ..misc.time_cost_function import time_it
from ..misc.waitforbuttonpress_popup import WaitForButtonPressPopup
from ..processing.semicontrolled_data_cleaning import normalize_signal


class KinectLEDBlinking:
    def __init__(self, video_path, result_dir_path, result_file_name):
        self.video_path = video_path
        self.result_dir_path = result_dir_path
        self.result_file_name = result_file_name

        self.lower_green = np.array([40, 40, 40])
        self.upper_green = np.array([80, 255, 255])
        self.threshold_value = None

        self.roi = None
        self.fps = None
        self.nframes = None

        self.time = []
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
        self.roi = []

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
                self.roi.append(frame)

        self.roi = np.array(self.roi)
        self.nframes = self.roi.shape[0]

    @time_it
    def monitor_green_levels(self, show=False):
        if self.roi is None:
            raise Exception("Error: Area of Interest was not initialized.")
        self.green_levels = np.zeros(self.nframes)
        if show:
            cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        for idx, frame in enumerate(self.roi):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
            roi = cv2.bitwise_and(frame, frame, mask=mask)
            avg_green = np.mean(roi[:, :, 1])
            self.green_levels[idx] = avg_green
            if show:
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        self.green_levels = normalize_signal(self.green_levels, dtype=np.ndarray)
        self.time = np.linspace(0, 1 / self.fps * (self.nframes - 1), self.nframes)

    def process_led_on(self, method="bimodal", threshold=0.25):
        match method:
            case "bimodal":
                self.process_led_on_bimodal()
            case "threshold":
                self.process_led_on_threshold(threshold=threshold)

    def process_led_on_bimodal(self):
        x = self.green_levels
        # reshape 1D data to ensure gmm to function correctly --> X.shape = (n_samples, 1).
        x = x.reshape(-1, 1)

        # create GMM model object
        gmm = mixture.GaussianMixture(n_components=2, max_iter=1000, random_state=10, covariance_type='full')

        # find useful parameters
        mean = gmm.fit(x).means_
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
        if self.roi is None or self.led_on is None:
            raise Exception("Error: led_on was not initialized.")
        self.occluded = np.full(self.nframes, False, dtype=bool)
        frame_avg_off_idx = np.where(self.led_on == 0)[0]
        frame_avg_on_idx = np.where(self.led_on == 1)[0]
        frame_avg_off = np.round(np.mean(self.roi[frame_avg_off_idx, :, :, :], axis=0)).astype(int)
        frame_avg_on = np.round(np.mean(self.roi[frame_avg_on_idx, :, :, :], axis=0)).astype(int)
        means_off = []
        means_on = []
        for idx, frame in enumerate(self.roi):
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
                    plt.imshow(self.roi[idx, :, :, :])
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
            "lower_green": self.lower_green.tolist(),
            "upper_green": self.upper_green.tolist(),
            "threshold_value": self.threshold_value,
            "fps": self.fps,
            "nframes": self.nframes
        }
        with open(full_path, 'w') as f:
            json.dump(metadata, f, indent=4)
