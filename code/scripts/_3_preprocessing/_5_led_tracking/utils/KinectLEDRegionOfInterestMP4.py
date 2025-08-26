import os
import cv2
import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from .KinectLEDRegionOfInterest import KinectLEDRegionOfInterest
from .time_cost_function import time_it


class KinectLEDRegionOfInterestMP4(KinectLEDRegionOfInterest):

    def extract_metadata_video(self):
        if self.cap is None:
            raise Exception("Error: Video not initialized. Please call initialise_video() first.")

        # Variables for progression and frame counting
        nframes_predicted = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progression_list = np.linspace(0, 100, 20)
        progression_idx = 0

        # Check if the video exists
        if not os.path.exists(self.video_path):
            print("The video_path does not point to an existing file.")
            return

        # Re-initialize video capture
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise Exception("Error: Cannot open video file.")

        # Read frames using OpenCV
        frame_shape = (self.frame_height, self.frame_width, 3)
        nframes = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break  # Stop if the frame could not be read (end of video)

            nframes += 1

            if (100 * nframes / nframes_predicted) > progression_list[progression_idx]:
                print("Create Area of Interest> Progression:", int(progression_list[progression_idx]),
                      "% ( Frame:", nframes, "/", nframes_predicted, ")")
                progression_idx += 1

        # Release the capture after reading all frames
        self.cap.release()

        # Store the correct number of frames
        self.nframes = nframes

    @time_it
    def extract_roi(self):
        # Check if the video is initialized
        if self.cap is None or self.reference_frame is None:
            raise Exception("Error: Video not initialized. Please call initialise_video() first.")

        # Check if the video exists
        if not os.path.exists(self.video_path):
            raise Exception(f"The video_path does not point to an existing file.")

        # Set the frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Variables for progression and frame counting
        nframes_predicted = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progression_list = np.linspace(0, 100, 20)
        progression_idx = 0

        # Clean area of interest variable
        self.roi = []

        # Define the square region
        half_size = self.square_size // 2
        x_start = max(self.square_center[0] - half_size, 0)
        x_end = min(self.square_center[0] + half_size, self.frame_width)
        y_start = max(self.square_center[1] - half_size, 0)
        y_end = min(self.square_center[1] + half_size, self.frame_height)

        # Start reading frames
        has_frame = True
        first_frame = True
        frame_idx = None
        while has_frame:
            # I. get the formatted frame
            has_frame, frame = self.cap.read()
            if not has_frame:
                if self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    print("End of video file reached.")
                    continue
                else:
                    # Append an mock image to the output DataFrame using loc
                    print(f"frame {frame_idx}: capture read failed.")
                    roi_frame = np.zeros((y_end - y_start, x_end - x_start, 3))
                    # try to go to next frame
                    old_frame_idx = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, old_frame_idx+1)
                    # if fails to go to next frame, there is an issue with the video file.
                    if old_frame_idx == self.cap.get(cv2.CAP_PROP_POS_FRAMES):
                        has_frame = False
                        print("End of video file reached.")
                        continue
                    else:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, old_frame_idx)
                        has_frame = True
            else:
                # Extract the region of interest
                roi_frame = frame[y_start:y_end, x_start:x_end]
            
            if first_frame:
                frame_idx = 0
                first_frame = False
            else:
                frame_idx += 1
            
            self.roi.append(roi_frame)

            # Display progress
            if (100 * frame_idx / nframes_predicted) > progression_list[progression_idx]:
                print("Create Area of Interest> Progression:", int(progression_list[progression_idx]), "% ( Frame:",
                      frame_idx, "/", nframes_predicted, ")")
                progression_idx += 1

        # Convert the list of ROIs to a numpy array
        self.roi = np.array(self.roi)

        # Store the number of frames processed
        self.nframes = frame_idx + 1
        print(f'Final number of frame read vs expected:{self.nframes}/{nframes_predicted}')

        # Release the video capture object
        self.cap.release()
