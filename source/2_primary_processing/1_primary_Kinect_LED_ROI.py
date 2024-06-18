import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libraries.primary.semicontrolled_Kinect_led_roi import KinectLEDRegionOfInterest  # noqa: E402
import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.misc.waitforbuttonpress_popup import WaitForButtonPressPopup


def find_mkv_files(input_path, sessions):
    """
    Given a dictionary of directory names and their absolute paths,
    find all .mkv files nested inside these directories.

    Args:
    directories (dict): A dictionary where keys are directory names and
                        values are their absolute paths.

    Returns:
    list: A list of absolute paths to all .mkv files found.
    """
    mkv_files_session = []
    mkv_files = []
    mkv_files_abs = []

    for session in sessions:
        dir_path = os.path.join(input_path, session)
        # Walk through the directory recursively
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.mkv'):
                    mkv_files.append(file)
                    mkv_files_abs.append(os.path.join(root, file))
                    mkv_files_session.append(session)

    return mkv_files_abs, mkv_files, mkv_files_session


if __name__ == "__main__":
    force_processing = True  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening
    save_results = True

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    database_path = path_tools.get_database_path()
    # get input base directory
    database_path_input = os.path.join(database_path, "semi-controlled", "primary", "kinect", "raw_noIR")
    # get output base directory
    database_path_output = os.path.join(database_path, "semi-controlled", "primary", "kinect", "roi_led")
    if not os.path.exists(database_path_output):
        os.makedirs(database_path_output)
        print(f"Directory '{database_path_output}' created.")
    # Session names
    sessions = ['2022-06-14_ST13-01',
                '2022-06-14_ST13-02',
                '2022-06-14_ST13-03',
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

    mkv_files_abs, mkv_files, mkv_files_session = find_mkv_files(database_path_input, sessions)

    print("Step 1: Determine manually the location of the LED.")
    # 1. Determine:
    #   a. If the file contains the LED (not out of frame)
    #   b. If there is a good frame (no occlusion)
    #   c. Then extract LED location if both are fulfilled
    led_in_frames = []
    frame_idx = []
    square_center = []
    square_radius = []
    for mkv_filename_abs, mkv_filename, session in zip(mkv_files_abs, mkv_files, mkv_files_session):
        print(f"File '{mkv_filename}'")

        # create Kinect processing manager
        led_roi = KinectLEDRegionOfInterest(mkv_filename_abs, "", "")

        frame_number = None
        center = None
        radius = None

        # force processing or if the video hasn't been processed yet
        if force_processing or not led_roi.is_already_processed():
            led_roi.initialise_video()
            frame_number = led_roi.select_good_frame()

            # if the LED is not occluded or not out of bounds
            if frame_number is not None:
                led_roi.set_reference_frame(frame_number)
                led_roi.select_led_location()

        # keep the variables
        led_in_frames.append(led_roi.led_in_frame)
        frame_idx.append(led_roi.reference_frame_idx)
        square_center.append(led_roi.square_center)
        square_radius.append(led_roi.square_size)

    print("Step 2: Automatic processing...")
    # 2. Process LED blinking
    for idx, (mkv_filename_abs, mkv_filename, session) in enumerate(zip(mkv_files_abs, mkv_files, mkv_files_session)):
        print(f"File '{mkv_filename}'")
        # Saving info
        output_dirname = os.path.join(database_path_output, session)
        output_filename = mkv_filename.replace(".mkv", "") + "_LED_roi"
        # create Kinect processing manager
        led_roi = KinectLEDRegionOfInterest(mkv_filename_abs, output_dirname, output_filename)

        led_roi.led_in_frame = led_in_frames[idx]

        if (force_processing or not led_roi.is_already_processed()) and led_roi.led_in_frame:
            # if the video hasn't been processed yet or redo the processing (load results = False)
            led_roi.initialise_video()
            led_roi.set_reference_frame(frame_idx[idx])
            led_roi.select_led_location(square_center[idx], square_radius[idx])
            led_roi.extract_roi()

            if save_results:
                led_roi.save_results()
