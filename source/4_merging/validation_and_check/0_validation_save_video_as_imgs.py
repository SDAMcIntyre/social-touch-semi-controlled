import subprocess
import os
import json
import sys
import warnings
import numpy as np
from PIL import Image

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libraries.misc.path_tools as path_tools  # noqa: E402

def extract_frames(video_path, output_folder, frame_rate=1):
    """
    Extract frames from a video and save them to a specific folder.

    :param video_path: Path to the input video file.
    :param output_folder: Path to the folder where images will be saved.
    :param frame_rate: The number of frames to extract per second (default is 1).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # COLOR/RGB channels (or stream) in Kinect videos are located in 0
    info = subprocess.run([
        *"ffprobe -v quiet -print_format json -show_format -show_streams".split(),
        video_path
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

    frame_shape = np.array([frame_height, frame_width, 3])
    nelem = frame_shape.prod()
    nframes = 0

    cmd = ["ffmpeg", "-i", video_path]
    cmd += "-map", f"0:{index}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        while True:
            data = proc.stdout.read(nelem)  # One byte per each element
            if not data:
                break
            nframes += 1
            frame = np.frombuffer(data, dtype=np.uint8).reshape(frame_shape)
            image_array = frame.astype(np.uint8)  # Ensure the array is in the correct type
            # Convert the NumPy array to an Image object
            image = Image.fromarray(image_array)
            # Save as PNG
            image.save(os.path.join(output_folder, f"frame_{nframes:04}.png"))


# Example usage
# get database directory
db_path = path_tools.get_database_path()
db_path_input = os.path.join(db_path, "semi-controlled", "primary", "kinect")
db_path_input_mkv = os.path.join(db_path_input, "2_roi_led", "2022-06-14_ST13-01")
vid_fname = "2022-06-14_ST13-01_semicontrolled_block-order01_kinect_LED_roi.mp4"
vid_fname_abs = os.path.join(db_path_input_mkv, vid_fname)

output_folder = 'C:\\Users\\cestl\\Desktop\\img_kinect_led_roi'
extract_frames(vid_fname_abs, output_folder)
