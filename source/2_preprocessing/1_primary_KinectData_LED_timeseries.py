import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libraries.preprocessing.semicontrolled_Kinect_manager import ProcessKinectLED  # noqa: E402
import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.misc.waitforbuttonpress_popup import WaitForButtonPressPopup

def find_mkv_files(sessions):
    """
    Given a dictionary of directory names and their absolute paths,
    find all .mkv files nested inside these directories.

    Args:
    directories (dict): A dictionary where keys are directory names and
                        values are their absolute paths.

    Returns:
    list: A list of absolute paths to all .mkv files found.
    """
    data_dir_base = path_tools.get_path_root_abs()
    input_dir = "primary"
    datatype_str = "semi-controlled\kinect"

    # destination
    dir_abs = os.path.join(data_dir_base, input_dir)
    # target the current experiment, postprocessing and datatype
    dir_abs = os.path.join(dir_abs, datatype_str)

    mkv_files_session = []
    mkv_files = []
    mkv_files_abs = []

    for session in sessions:
        dir_path = os.path.join(dir_abs, session)
        # Walk through the directory recursively
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.mkv'):
                    mkv_files.append(file)
                    mkv_files_abs.append(os.path.join(root, file))
                    mkv_files_session.append(session)

    return mkv_files_abs, mkv_files, mkv_files_session


def get_output_dir():
    data_dir_base = path_tools.get_path_root_abs()
    output_dir = "processed"
    datatype_str = "semi-controlled\kinect"
    # destination
    dir_abs = os.path.join(data_dir_base, output_dir)
    # target the current experiment, postprocessing and datatype
    dir_abs = os.path.join(dir_abs, datatype_str)

    if not os.path.exists(dir_abs):
        os.makedirs(dir_abs)
        print(f"Directory '{dir_abs}' created.")
    else:
        print(f"Directory '{dir_abs}' already exists.")

    return dir_abs


if __name__ == "__main__":
    force_processing = False
    show = False  # put True if user wants to monitor what's happening
    save_results = True

    # Session names
    sessions = ['2022-06-15_ST13-01',
                '2022-06-15_ST13-02',
                '2022-06-15_ST13-03',
                '2022-06-15_ST14-01',
                '2022-06-15_ST14-02',
                '2022-06-15_ST14-03',
                '2022-06-15_ST14-04',
                '2022-06-16_ST15-01',
                '2022-06-16_ST15-02',
                '2022-06-17_ST16-02',
                '2022-06-17_ST16-03',
                '2022-06-17_ST16-04',
                '2022-06-17_ST16-05',
                '2022-06-22_ST18-01',
                '2022-06-22_ST18-02',
                '2022-06-22_ST18-04']
    mkv_files_abs, mkv_files, mkv_files_session = find_mkv_files(sessions)
    output_dir_abs = get_output_dir()

    # 1. Determine:
    #   a. If the file contains the LED (not out of frame)
    #   b. If there is a good frame (no occlusion)
    #   c. Then extract LED location if both are fulfilled
    led_in_frames = []
    frame_idx = []
    square_center = []
    square_radius = []

    print("Step 1: Determine manually the location of the LED.")
    for mkv_filename_abs, mkv_filename, session in zip(mkv_files_abs, mkv_files, mkv_files_session):
        print(f"File '{mkv_filename}'")

        # Saving info
        output_dirname = os.path.join(output_dir_abs, session)
        output_filename = mkv_filename.replace(".mkv", "") + "_LED_timeseries"
        # create Kinect processing manager
        kinect_led = ProcessKinectLED(mkv_filename_abs, output_dirname, output_filename)

        kinect_led.led_in_frame = False
        frame_number = None
        center = None
        radius = None

        # force processing or if the video hasn't been processed yet
        if force_processing or not kinect_led.is_already_processed():
            kinect_led.initialise_video()
            frame_number = kinect_led.select_good_frame()

            # if the LED is not occluded or not out of bounds
            if frame_number is not None:
                kinect_led.led_in_frame = True
                kinect_led.define_reference_frame(frame_number)
                [center, radius] = kinect_led.select_led_location()

        # keep the variables
        led_in_frames.append(kinect_led.led_in_frame)
        frame_idx.append(frame_number)
        square_center.append(center)
        square_radius.append(radius)

    print("Step 2: Automatic processing...")
    # 2. Process LED blinking
    for idx, (mkv_filename_abs, mkv_filename, session) in enumerate(zip(mkv_files_abs, mkv_files, mkv_files_session)):
        print(f"File '{mkv_filename}'")
        # Saving info
        output_dirname = os.path.join(output_dir_abs, session)
        output_filename = mkv_filename.replace(".mkv", "") + "_LED_timeseries"
        # create Kinect processing manager
        kinect_led = ProcessKinectLED(mkv_filename_abs, output_dirname, output_filename)

        kinect_led.led_in_frame = led_in_frames[idx]

        if not force_processing and kinect_led.is_already_processed():
            kinect_led.load_results()
        elif kinect_led.led_in_frame:
            # if the video hasn't been processed yet or redo the processing (load results = False)
            kinect_led.initialise_video()
            kinect_led.led_is_visible = True
            kinect_led.define_reference_frame(frame_idx[idx])
            kinect_led.select_led_location(square_center[idx], square_radius[idx])
            kinect_led.extract_aoi()
            kinect_led.monitor_green_levels(show=show)
            kinect_led.process_led_on(threshold=.20)
            # correct for any occlusion
            kinect_led.define_occlusion(threshold=40, show=show)

            if save_results:
                kinect_led.save_results()

        if show:
            plt.plot(kinect_led.time, kinect_led.led_on)
            plt.show()
