import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libraries.primary.semicontrolled_Kinect_led_blinking import KinectLEDBlinking  # noqa: E402
import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.misc.waitforbuttonpress_popup import WaitForButtonPressPopup


def find_mp4_files(input_path, sessions):
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
                if file.endswith('.mp4'):
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
    database_path_input = os.path.join(database_path, "semi-controlled", "primary", "kinect", "roi_led")
    # get output base directory
    database_path_output = os.path.join(database_path, "semi-controlled", "processed", "kinect", "led")
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
    sessions = ['2022-06-14_ST13-02']

    mkv_files_abs, mkv_files, mkv_files_session = find_mp4_files(database_path_input, sessions)

    print("Step 2: Automatic processing...")
    # 2. Process LED blinking
    for idx, (mkv_filename_abs, mkv_filename, session) in enumerate(zip(mkv_files_abs, mkv_files, mkv_files_session)):
        print(f"File '{mkv_filename}'")
        # Saving info
        output_dirname = os.path.join(database_path_output, session)
        output_filename = mkv_filename.replace(".mp4", "") + "_LED_timeseries"
        # create Kinect processing manager
        led_blink = KinectLEDBlinking(mkv_filename_abs, output_dirname, output_filename)

        if force_processing or not led_blink.is_already_processed():
            # if the video hasn't been processed yet or redo the processing (load results = False)
            led_blink.load_video()

            led_blink.monitor_green_levels(show=show)
            led_blink.process_led_on(threshold=.20)
            # correct for any occlusion
            led_blink.define_occlusion(threshold=40, show=show)

            if save_results:
                led_blink.save_results()

            if show:
                plt.plot(led_blink.time, led_blink.led_on)
                plt.ion()
                plt.show()
                WaitForButtonPressPopup()
                plt.close()
