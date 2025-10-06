import numpy as np
import logging

from preprocessing.common.data_access.video_mp4_manager import VideoMP4Manager, ColorFormat
from .KinectLEDBlinking import KinectLEDBlinking


class KinectLEDBlinkingMP4(KinectLEDBlinking):
    """
    Analyzes Kinect LED blinking by loading video data from an MP4 file
    using the robust VideoMP4Manager.
    """
    def load_video(self):
        """
        Loads the video frames and metadata using VideoMP4Manager.

        This method initializes the video manager, sets the required color format to RGB,
        and then efficiently loads all frames into a NumPy array.
        """
        try:
            # 1. Instantiate the manager. It handles file path validation,
            #    opens the video, and reads all metadata in its constructor.
            #    We request RGB format directly, so conversion is handled automatically.
            video_manager = VideoMP4Manager(self.video_path, color_format=ColorFormat.RGB)

            # 2. Assign metadata directly from the manager's properties.
            self.fps = video_manager.fps
            self.nframes = video_manager.total_frames
            
            # 3. Use the manager's NumPy protocol support to convert the entire
            #    video into a NumPy array with a single, clear command. This
            #    replaces the manual frame-by-frame reading loop.
            logging.info(f"Loading {self.nframes} frames from '{self.video_path}'...")
            self.roi_frames = np.array(video_manager)
            
            # Verify the loaded shape for consistency.
            if self.roi_frames.shape[0] != self.nframes:
                logging.warning(f"Number of loaded frames ({self.roi_frames.shape[0]}) does not match metadata ({self.nframes}).")
                self.nframes = self.roi_frames.shape[0]

            logging.info(f"Successfully loaded video with shape {self.roi_frames.shape}.")

        except (FileNotFoundError, IOError) as e:
            # Catch exceptions raised by VideoMP4Manager for cleaner error handling.
            logging.error(f"Failed to load video: {e}")
            self.roi_frames = np.array([]) # Ensure roi_frames is an empty array on failure
            self.nframes = 0
            self.fps = 0.0