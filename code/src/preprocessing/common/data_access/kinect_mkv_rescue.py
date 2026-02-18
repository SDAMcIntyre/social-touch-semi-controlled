import cv2
import numpy as np
from pathlib import Path

class KinectRescue:
    def __init__(self, color_path, depth_path, width=640, height=576):
        self.color_cap = cv2.VideoCapture(str(color_path))
        self.depth_path = Path(depth_path)
        self.width = width
        self.height = height
        self.frame_len = width * height * 2  # 16-bit = 2 bytes per pixel
        self._depth_file = None
        
    def __enter__(self):
        if not self.depth_path.exists():
            raise FileNotFoundError(f"Raw depth file not found: {self.depth_path}")
        self._depth_file = open(self.depth_path, 'rb')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._depth_file:
            self._depth_file.close()
        self.color_cap.release()

    def __iter__(self):
        while True:
            # 1. Read Color
            ret, color_frame = self.color_cap.read()
            if not ret:
                break # End of video

            # 2. Read Depth (Raw Bytes)
            raw_bytes = self._depth_file.read(self.frame_len)
            if len(raw_bytes) != self.frame_len:
                break # End of depth stream

            # 3. Convert Bytes to Image (16-bit)
            depth_frame = np.frombuffer(raw_bytes, dtype=np.uint16).reshape((self.height, self.width))

            yield RescueFrame(color_frame, depth_frame)

class RescueFrame:
    def __init__(self, color, depth):
        self.color = color
        self.depth = depth
        # SDK features are unavailable
        self.transformed_depth = None 
        self.transformed_depth_point_cloud = None
