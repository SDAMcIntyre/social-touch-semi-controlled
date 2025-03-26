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


class KinectLEDRegionOfInterest:
    def __init__(self, video_path=None, output_dirname=None, output_filename=None):
        self.video_path = video_path
        self.result_dir_path = output_dirname
        self.result_filename = output_filename

        self.cap = None
        self.frame_width = None
        self.frame_height = None
        self.fps = None
        self.nframes = None
        self.fourcc_str = None

        self.square_center = None
        self.square_size = None

        self.reference_frame_idx = None
        self.reference_frame = None

        # results
        self.led_in_frame = False
        self.roi = None

    def is_already_processed(self):
        metadata_file = os.path.join(self.result_dir_path, self.result_filename + "_metadata.txt")
        return os.path.isfile(metadata_file)

    def load_metadata(self, metadata_path_abs=None):
        if metadata_path_abs is None:
           metadata_path_abs = os.path.join(self.result_dir_path, self.result_filename + "_metadata.txt")
        else:
            self.result_dir_path = os.path.dirname(metadata_path_abs)
            self.result_filename = os.path.basename(metadata_path_abs).replace("_metadata.txt", "")

        # Read the content of the file
        with open(metadata_path_abs, 'r', encoding='utf-8') as file:
            data = file.read()

        # Parse the JSON content
        json_data = json.loads(data)

        # Extract the required values
        self.video_path = json_data["video_path"]
        self.reference_frame_idx = json_data['reference_frame_idx']
        self.square_center = json_data['square_center']
        self.square_size = json_data['square_size']
        # if the file exists, then the LED is in frame.
        if 0 <= self.reference_frame_idx:
            self.led_in_frame = True
        else:
            self.led_in_frame = False
        

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


    def __show_confirmation_blocking_window(self, question_str="Use this frame?"):
        confirmation_result = None
        # Callback function for confirmation window
        def confirm_frame(event, x, y, flags, param):
            nonlocal confirmation_result
            if event == cv2.EVENT_LBUTTONUP:
                if 50 < x < 350 and 100 < y < 150:
                    confirmation_result = True
                elif 450 < x < 750 and 100 < y < 150:
                    confirmation_result = False
        # window parameters

        confirmation_window = np.zeros((200, 800, 3), dtype=np.uint8)
        cv2.putText(confirmation_window, question_str, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(confirmation_window, (50, 100), (350, 150), (0, 255, 0), -1)
        cv2.putText(confirmation_window, "Yes <ENTER>", (100, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(confirmation_window, (450, 100), (750, 150), (0, 0, 255), -1)
        cv2.putText(confirmation_window, "No <ESC>", (520, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # Confirmation window for user input
        window_name_confirmation = 'Confirmation'
        cv2.namedWindow(window_name_confirmation)
        cv2.setMouseCallback(window_name_confirmation, confirm_frame)
        cv2.imshow(window_name_confirmation, confirmation_window)

        while confirmation_result is None:
            key = cv2.waitKey(1)
            if key == 13:  # ENTER key
                confirmation_result = True
            elif key == 27:  # ESC key
                confirmation_result = False
        cv2.destroyWindow(window_name_confirmation)

        return confirmation_result


    def _confirm_frame_selection(self, frame_id):
        # Set the frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = self.cap.read()
        if not ret:
            raise Exception(f"Error: Could not read the frame at position {frame_id}.")

        window_name_frame = f"Frame {frame_id}"
        cv2.imshow(window_name_frame, frame)
        confirmation_result = self.__show_confirmation_blocking_window()
        cv2.destroyWindow(window_name_frame)

        return confirmation_result


    def _choose_frame_from_montage(self):
        selected_frame_idx = None
        frame_is_selected = False
        def select_miniature(event, x, y, flags, param):
            nonlocal selected_frame_idx, frame_is_selected
            if event == cv2.EVENT_LBUTTONUP:
                frame_number = (y // miniature_height) * 5 + (x // miniature_width)
                if frame_number < len(miniatures):
                    selected_frame_idx = miniatures[frame_number][0]
                    frame_is_selected = True
                elif frame_number == 19:  # None area selected
                    selected_frame_idx = None
                    frame_is_selected = True
        
        # Display 19 frames linearly spread over the entire video
        nframe = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, nframe / 2 - 1, 19, dtype=int)

        # Create 19 thumbnails
        miniature_height = 90 * 4
        miniature_width = 160 * 4
        miniatures = []
        for i, frame_idx in enumerate(frame_indices):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                continue
            thumbnail = cv2.resize(frame, (miniature_width, miniature_height))  # Create thumbnail
            miniatures.append((frame_idx, thumbnail))
        
        # create 1 thumbnails "None" for user to select if led is not visible
        none_area = np.zeros((miniature_height, miniature_width, 3), dtype=np.uint8)
        cv2.putText(none_area, "None of them", (50, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Create a montage of thumbnails
        montage = np.zeros((miniature_height * 4, miniature_width * 5, 3), dtype=np.uint8)
        for i, (_, thumbnail) in enumerate(miniatures):
            row = i // 5
            col = i % 5
            montage[row * miniature_height:(row + 1) * miniature_height,
                    col * miniature_width:(col + 1) * miniature_width] = thumbnail
        montage[3 * miniature_height:(4) * miniature_height, 4 * miniature_width:(5) * miniature_width] = none_area

        # Display the montage and handle user selection
        window_name_montage = 'Select Frame'
        cv2.namedWindow(window_name_montage, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name_montage, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback(window_name_montage, select_miniature)
        cv2.imshow(window_name_montage, montage)

        while not frame_is_selected:
            cv2.waitKey(1)
        cv2.destroyWindow(window_name_montage)

        return selected_frame_idx


    def get_frame_id_with_led(self):
        if self.cap is None:
            raise Exception("Error: Video not initialized. Please call initialise() first.")
        frame_id = 0
        while not self._confirm_frame_selection(frame_id):
            frame_id = self._choose_frame_from_montage()
            if frame_id is None:  # LED is not visible on any of the frames
                self.led_in_frame = False
                return None
        self.led_in_frame = True
        return frame_id  

    def set_reference_frame(self, frame_idx):
        self.reference_frame_idx = frame_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, self.reference_frame = self.cap.read()
        if not ret:
            raise Exception("Error: Could not read the reference frame.")
        
    
    def _set_square(self, window_name='First Frame', destroy=True):
        square_is_done = False
        drawing = False
        origin = None
        side_length = None

        def draw_square(event, x, y, flags, param):
            nonlocal window_name, square_is_done, drawing, origin, side_length
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                origin = (x, y)
            
            if drawing:
                if event == cv2.EVENT_MOUSEMOVE:
                    width = int(abs(x - origin[0]))
                    height = int(abs(y - origin[1]))
                    side_length = max(width, height)
                    top_left = (origin[0] - side_length // 2, origin[1] - side_length // 2)
                    bottom_right = (origin[0] + side_length // 2, origin[1] + side_length // 2)
                    frame_copy = self.reference_frame.copy()
                    cv2.rectangle(frame_copy, top_left, bottom_right, (0, 255, 0), 2)
                    cv2.imshow(window_name, frame_copy)

                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                    square_is_done = True

        # Generate window to draw rectangle (to become square)
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, draw_square)
        cv2.imshow(window_name, self.reference_frame)

        while not square_is_done:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        if destroy:
            cv2.destroyWindow(window_name)
        
        return origin, side_length


    # Method to select LED location with a square  
    def draw_led_location(self):
        if self.reference_frame is None:
            raise Exception("Error: First frame not available. Please initialize the video first.")

        destroy_in_function = False
        window_name = 'First Frame'

        origin = None
        side_length = None
        is_valid_square = False
        while not is_valid_square:
            origin, side_length = self._set_square(window_name=window_name, destroy=destroy_in_function)
            is_valid_square = self.__show_confirmation_blocking_window(question_str="Is square valid?")
        
        if not destroy_in_function:
            cv2.destroyWindow(window_name)

        self.square_center = origin
        self.square_size = side_length

        return self.square_center, self.square_size


    # Method to select LED location with a square  
    def set_led_location(self, square_center, square_size):
        if self.reference_frame is None:
            raise Exception("Error: First frame not available. Please initialize the video first.")
        if square_center is None or square_size is None:
            raise Exception("Error: square_center or square_size is None.")
        
        self.square_center = square_center
        self.square_size = square_size
        return self.square_center, self.square_size


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

    @time_it
    def extract_roi(self):
        if self.cap is None or self.reference_frame is None:
            raise Exception("Error: Video not initialized. Please call initialise_video() first.")

        # Variables for progression and frame counting
        nframes_predicted = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progression_list = np.linspace(0, 100, 20)
        progression_idx = 0
        nframes = 0

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

        # Clean area of interest variable
        self.roi = []

        # Define the square region
        half_size = self.square_size // 2
        x_start = max(self.square_center[0] - half_size, 0)
        x_end = min(self.square_center[0] + half_size, self.frame_width)
        y_start = max(self.square_center[1] - half_size, 0)
        y_end = min(self.square_center[1] + half_size, self.frame_height)

        frame_shape = np.array([self.frame_height, self.frame_width, 3])
        nelem = frame_shape.prod()
        nframes = 0
        with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
            while True:
                data = proc.stdout.read(nelem)  # One byte per each element
                if not data:
                    break
                nframes += 1
                frame = np.frombuffer(data, dtype=np.uint8).reshape(frame_shape)

                if (100 * nframes / nframes_predicted) > progression_list[progression_idx]:
                    print("Create Area of Interest> Progression:", int(progression_list[progression_idx]), "% ( Frame:",
                          nframes, "/",
                          nframes_predicted, ")")
                    progression_idx += 1
                self.roi.append(frame[y_start:y_end, x_start:x_end])

        # convert into array
        self.roi = np.array(self.roi)
        # take advantage to store the correct number of frames
        self.nframes = nframes

    def save_results(self, verbose=False):
        if not os.path.exists(self.result_dir_path):
            os.makedirs(self.result_dir_path)
        filename_abspath = os.path.join(self.result_dir_path, self.result_filename + ".mp4")
        self.save_roi_as_video(filename_abspath)
        filename_abspath = os.path.join(self.result_dir_path, self.result_filename + "_metadata.txt")
        self.save_result_metadata(filename_abspath)
        if verbose:
            print(f"Results saved as {self.result_dir_path}{self.result_filename}*.")
    
    def save_roi_as_video(self, filename_abspath, show=False):
        if self.roi is None:
            if not self.led_in_frame:
                return
            else:
                raise Exception("Error: roi not extracted.")


        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        size = (self.roi.shape[2], self.roi.shape[1])
        out = cv2.VideoWriter(filename_abspath, fourcc, self.fps, size)
        for frame in self.roi:
            out.write(frame)

            if show:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        out.release()
        cv2.destroyAllWindows()

    def save_result_metadata(self, filename_abspath):
        if self.reference_frame_idx is None:
            self.reference_frame_idx = -1

        metadata = {
            "video_path": self.video_path,
            "reference_frame_idx": int(self.reference_frame_idx),
            "square_center": self.square_center,
            "square_size": self.square_size,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "fps": self.fps,
            "nframes": self.nframes,
            "fourcc_str": self.fourcc_str
        }
        with open(filename_abspath, 'w') as f:
            json.dump(metadata, f, indent=4)
