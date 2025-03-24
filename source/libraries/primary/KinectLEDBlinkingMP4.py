import cv2
import numpy as np

from .KinectLEDBlinking import KinectLEDBlinking


class KinectLEDBlinkingMP4(KinectLEDBlinking):
    def load_video(self):
        # Clean area of interest variable
        self.roi = []

        capture = cv2.VideoCapture(self.video_path)
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if not capture.isOpened():
            print(f"video cannot be opened: {self.video_path}")
            return None
        capture_nframe = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get frame properties
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = capture.get(cv2.CAP_PROP_FPS)

        frame_shape = np.array([frame_height, frame_width, 3])

        nframes = 0
        first_frame = True
        end_of_video = False
        while not end_of_video:
            if first_frame:
                frame_idx = 0
            else:
                frame_idx += 1
            
            has_frame, frame = capture.read()
            if not has_frame:
                if capture.get(cv2.CAP_PROP_POS_FRAMES) >= capture.get(cv2.CAP_PROP_FRAME_COUNT):
                    print("End of video file reached.")
                    end_of_video = True
                    continue
                else:
                    frame_rgb = []
            else:
                # Convert frame from BGR (default in OpenCV) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.roi.append(frame_rgb)
            nframes += 1

        capture.release()

        self.roi = np.array(self.roi)
        self.nframes = self.roi.shape[0]
