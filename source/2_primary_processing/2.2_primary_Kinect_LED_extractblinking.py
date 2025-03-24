import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from source.libraries.primary.KinectLEDBlinking import KinectLEDBlinking  # noqa: E402
from source.libraries.primary.KinectLEDBlinkingMP4 import KinectLEDBlinkingMP4  # noqa: E402
import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.misc.waitforbuttonpress_popup import WaitForButtonPressPopup


def find_metadata_files(input_path, sessions):
    """
    Given a dictionary of directory names and their absolute paths,
    find all _metadata.txt files nested inside these directories.

    Args:
    directories (dict): A dictionary where keys are directory names and
                        values are their absolute paths.

    Returns:
    list: A list of absolute paths to all .md files found.
    """
    if not isinstance(sessions, list):
        sessions = [sessions]
    md_files_session = []
    md_files = []
    md_files_abs = []

    for session in sessions:
        dir_path = os.path.join(input_path, session)
        # Walk through the directory recursively
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('_metadata.txt'):
                    md_files.append(file)
                    md_files_abs.append(os.path.join(root, file))
                    md_files_session.append(session)

    return md_files_abs, md_files, md_files_session


if __name__ == "__main__":

    force_processing = True  # If user wants to force data processing even if results already exist
    save_results = True

    show_video_by_frames = False  # If user wants to monitor what's happening
    show_results = False  # If user wants to monitor what's happening

    use_mp4 = True

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get input base directory
    if use_mp4:
        db_path_input = os.path.join(db_path, "1_primary", "kinect", "2_roi_led_mp4")
        extension = '.mp4'
    else:
        db_path_input = os.path.join(db_path, "1_primary", "kinect", "2_roi_led")
        extension = '.mkv'
    # get output base directory
    db_path_output = os.path.join(db_path, "2_processed", "kinect", "led", "0_block-order")
    if not os.path.exists(db_path_output):
        os.makedirs(db_path_output)
        print(f"Directory '{db_path_output}' created.")
    # Session names
    sessions_ST13 = ['2022-06-14_ST13-01',
                     '2022-06-14_ST13-02',
                     '2022-06-14_ST13-03']

    sessions_ST14 = ['2022-06-15_ST14-01',
                     '2022-06-15_ST14-02',
                     '2022-06-15_ST14-03',
                     '2022-06-15_ST14-04']

    sessions_ST15 = ['2022-06-16_ST15-01',
                     '2022-06-16_ST15-02']

    sessions_ST16 = ['2022-06-17_ST16-02',
                     '2022-06-17_ST16-03',
                     '2022-06-17_ST16-04',
                     '2022-06-17_ST16-05']

    sessions_ST18 = ['2022-06-22_ST18-01',
                     '2022-06-22_ST18-02',
                     '2022-06-22_ST18-04']

    sessions = []
    sessions = sessions + sessions_ST13
    sessions = sessions + sessions_ST14
    sessions = sessions + sessions_ST15
    sessions = sessions + sessions_ST16
    sessions = sessions + sessions_ST18

    for session in sessions:
        md_files_abs, md_files, _ = find_metadata_files(db_path_input, session)

        # Process LED blinking
        for idx, (md_filename_abs, md_filename) in enumerate(zip(md_files_abs, md_files)):
            print(f"File '{md_filename}'")

            # output directory
            output_dirname = os.path.join(db_path_output, session)
            if not os.path.exists(output_dirname):
                os.makedirs(output_dirname)

            # create video filename from metadata filename
            video_filename_abs = md_filename_abs.replace("_metadata.txt", ".mp4")

            # create output filename from metadata filename
            output_filename = md_filename.replace("_kinect_LED_roi_metadata.txt", "_LED")

            # create Kinect processing manager
            if "mp4" in extension:
                led_blink = KinectLEDBlinkingMP4(video_filename_abs, output_dirname, output_filename)
            else:
                led_blink = KinectLEDBlinking(video_filename_abs, output_dirname, output_filename)

            if led_blink.is_already_processed() and not force_processing:
                continue

            # Check if the video exists
            if not os.path.exists(video_filename_abs):
                print(f"The video_path does not point on an existing file. Creating a phantom result file...")
                led_blink.create_phantom_result_file(md_filename_abs)
            else:
                # if the video hasn't been processed yet or redo the processing (load results = False)
                led_blink.load_video()

                led_blink.monitor_green_levels(show=show_video_by_frames)
                led_blink.process_led_on(threshold=.20)
                # correct for any occlusion
                led_blink.define_occlusion(threshold=40, show=show_video_by_frames)

            if save_results:
                led_blink.save_results()

            if show_results:
                plt.plot(led_blink.time, led_blink.led_on)
                plt.ion()
                plt.show()
                WaitForButtonPressPopup()
                plt.close()

    if save_results:
        p = db_path_output.replace("\\", "/")
        print(f"Results saved in:\n{p}")
