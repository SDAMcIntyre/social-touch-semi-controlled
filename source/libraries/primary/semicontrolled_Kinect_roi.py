import os
import cv2
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import warnings

from ..misc.time_cost_function import time_it
from ..misc.waitforbuttonpress_popup import WaitForButtonPressPopup
from ..processing.semicontrolled_data_cleaning import normalize_signal


class KinectRegionOfInterest:
    def __init__(self, video_path_abs, output_filename_abs):
        self.video_path = video_path_abs
        self.result_filename_abs = output_filename_abs
        self.drawing = False
        self.confirmation_result = None

        self.cap = None
        self.frame_width = None
        self.frame_height = None
        self.fps = None
        self.nframes = None
        self.fourcc_str = None

        self.rectangle_top_left = (None, None)
        self.rectangle_bottom_right = (None, None)

        self.reference_frame_idx = None
        self.reference_frame = None

        # results
        self.target_in_frame = False
        self.roi = None

    def is_already_processed(self):
        return os.path.isfile(self.result_filename_abs)

    def load_target_location(self):
        # Read the content of the file
        with open(self.result_filename_abs, 'r', encoding='utf-8') as file:
            data = file.read()

        # Parse the JSON content
        json_data = json.loads(data)

        # Extract the required values
        self.reference_frame_idx = json_data['reference_frame_idx']
        self.rectangle_top_left = (json_data['region_of_interest']['top_left_corner']['x'],
                                   json_data['region_of_interest']['top_left_corner']['y'])
        self.rectangle_bottom_right = (json_data['region_of_interest']['bottom_right_corner']['x'],
                                       json_data['region_of_interest']['bottom_right_corner']['y'])

        # if the file exists, then target is in frame.
        self.target_in_frame = True

    def initialise_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise Exception("Error: Could not open video file.")
        # Get video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        # Convert fourcc to a string (i.e. "mpeg4")
        self.fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    def select_good_frame(self, frame_range='first-half'):
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
            self.target_in_frame = True
            return 0  # The first frame was chosen

        # Display 19 frames linearly spread over the entire video
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_range == 'first-half':
            frame_indices = np.linspace(0, total_frames / 2 - 1, 19, dtype=int)
        if frame_range == 'second-half':
            frame_indices = np.linspace(total_frames / 2 , total_frames - 1, 19, dtype=int)
        elif frame_range == 'full':
            frame_indices = np.linspace(0, total_frames - 1, 19, dtype=int)
        elif frame_range == 'random':
            frame_indices = np.sort(np.random.randint(0, total_frames-1, 19))

        else:
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

        # if the selected frame is the "None" box, it means the target is not in the video
        if selected_frame_idx == -1:
            self.target_in_frame = False
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
            self.target_in_frame = True
            return selected_frame_idx
        else:
            self.target_in_frame = False
            return None

    def set_reference_frame(self, frame_idx):
        self.reference_frame_idx = frame_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, self.reference_frame = self.cap.read()
        if not ret:
            raise Exception("Error: Could not read the reference frame.")

    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.rectangle_top_left = (x, y)
            self.rectangle_bottom_right = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.rectangle_bottom_right = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.rectangle_bottom_right = (x, y)

    def select_target_location(self, rectangle_top_left=None, rectangle_bottom_right=None):
        if rectangle_top_left is not None and rectangle_bottom_right is not None:
            self.rectangle_top_left = rectangle_top_left
            self.rectangle_bottom_right = rectangle_bottom_right
            return [self.rectangle_top_left, self.rectangle_bottom_right]

        if self.reference_frame is None:
            raise Exception("Error: First frame not available. Please initialize the video first.")

        while True:
            cv2.namedWindow('First Frame', cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback('First Frame', self.draw_rectangle)

            while True:
                frame_copy = self.reference_frame.copy()
                if self.rectangle_top_left and self.rectangle_bottom_right:
                    cv2.rectangle(frame_copy, self.rectangle_top_left, self.rectangle_bottom_right, (0, 255, 0), 2)
                    cv2.rectangle(frame_copy, self.rectangle_top_left, self.rectangle_bottom_right, (0, 255, 0), 2)
                cv2.imshow('First Frame', frame_copy)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or (not self.drawing and self.rectangle_top_left and self.rectangle_bottom_right):
                    break
            cv2.destroyAllWindows()

            if not self.rectangle_top_left or not self.rectangle_bottom_right:
                raise Exception("Error: No rectangle was drawn.")

            # Display confirmation window
            self.confirmation_result = None
            confirmation_window = np.zeros((200, 500, 3), dtype=np.uint8)
            cv2.putText(confirmation_window, "Is the rectangle set correctly?", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
            cv2.rectangle(confirmation_window, (50, 100), (150, 150), (0, 255, 0), -1)
            cv2.putText(confirmation_window, "Yes", (70, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.rectangle(confirmation_window, (250, 100), (350, 150), (0, 0, 255), -1)
            cv2.putText(confirmation_window, "No", (270, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.imshow('Confirmation', confirmation_window)

            def confirm_rectangle(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONUP:
                    if 50 < x < 150 and 100 < y < 150:
                        self.confirmation_result = True
                        cv2.destroyAllWindows()
                    elif 250 < x < 350 and 100 < y < 150:
                        self.confirmation_result = False
                        cv2.destroyAllWindows()
            cv2.setMouseCallback('Confirmation', confirm_rectangle)

            while self.confirmation_result is None:
                cv2.waitKey(1)

            if self.confirmation_result:
                break
            else:
                self.rectangle_top_left = None
                self.rectangle_bottom_right = None

        return [self.rectangle_top_left, self.rectangle_bottom_right]

    @time_it
    def extract_metadata_video(self):
        if self.cap is None:
            raise Exception("Error: Video not initialized. Please call initialise_video() first.")

        # Variables for progression and frame counting
        nframes_predicted = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progression_list = np.linspace(0, 100, 20)
        progression_idx = 0

        # CV2 read tends to sometimes not fully scan the video
        # It can stop at 50%, which will give wrong results.
        # Hence, we will use FFMPEG window library with the terminal to read frames after frames
        self.cap.release()

        # Check if the video exists
        if not os.path.exists(self.video_path):
            print(f"The video_path does not point on an existing file.")

        cmd = ["ffmpeg", "-i", self.video_path]
        # COLOR/RGB channels (or stream) in Kinect videos are located in 0
        probe = subprocess.run([
            *"ffprobe -v quiet -print_format json -show_format -show_streams".split(),
            self.video_path
        ], capture_output=True)
        probe.check_returncode()
        stream_rgb = json.loads(probe.stdout)["streams"][0]

        index = stream_rgb["index"]
        if stream_rgb["codec_type"] != "video":
            warnings.warn("PROBLEM: Expected stream for RGB video is not a video")
        cmd += "-map", f"0:{index}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"

        frame_shape = np.array([self.frame_height, self.frame_width, 3])
        nelem = frame_shape.prod()
        nframes = 0
        with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
            while True:
                data = proc.stdout.read(nelem)  # One byte per each element
                if not data:
                    break
                nframes += 1
                if (100 * nframes / nframes_predicted) > progression_list[progression_idx]:
                    print("Create Area of Interest> Progression:", int(progression_list[progression_idx]),
                          "% ( Frame:",
                          nframes, "/",
                          nframes_predicted, ")")
                    progression_idx += 1
        # take advantage to store the correct number of frames
        self.nframes = nframes

    def save_result(self, verbose=False):
        if self.reference_frame_idx is None:
            self.reference_frame_idx = -1

        metadata = {
            "video_path": self.video_path,
            "reference_frame_idx": int(self.reference_frame_idx),
            "region_of_interest": {
                "top_left_corner": {
                    "x": self.rectangle_top_left[0],
                    "y": self.rectangle_top_left[1]
                },
                "bottom_right_corner": {
                    "x": self.rectangle_bottom_right[0],
                    "y": self.rectangle_bottom_right[1]
                }
            },
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "fps": self.fps,
            "nframes": self.nframes,
            "fourcc_str": self.fourcc_str
        }
        with open(self.result_filename_abs, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)

        with open(self.result_filename_abs, 'r', encoding='utf-8') as f:
            content = f.read()
        print(content)

        if verbose:
            print(f"Results saved as {self.result_filename_abs}*.")