import cv2
import numpy as np

from .semicontrolled_Kinect_led_blinking import KinectLEDBlinking


class KinectLEDBlinkingMP4(KinectLEDBlinking):
    def load_video(self):
        # Clean area of interest variable
        self.roi = []

        # Open the video file with OpenCV
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open the video file: {self.video_path}")

        # Get frame properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)

        frame_shape = np.array([frame_height, frame_width, 3])

        nframes = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame from BGR (default in OpenCV) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.roi.append(frame_rgb)
            nframes += 1

        cap.release()

        self.roi = np.array(self.roi)
        self.nframes = self.roi.shape[0]
