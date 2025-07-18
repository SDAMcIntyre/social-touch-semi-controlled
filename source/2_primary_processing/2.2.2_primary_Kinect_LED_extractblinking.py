import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libraries.primary.KinectLEDBlinking import KinectLEDBlinking  # noqa: E402
from libraries.primary.KinectLEDBlinkingMP4 import KinectLEDBlinkingMP4  # noqa: E402
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
    if os.path.sep == '/':  # Running on a Unix-like system
        OS_linux = True
    elif os.path.sep == '\\':  # Running on Windows
        OS_linux = False

    force_processing = True  # If user wants to force data processing even if results already exist
    save_results = True
    save_figures = True

    show = False  # If user wants to monitor what's happening
    show_video_by_frames = False  # If user wants to monitor what's happening

    video_extension = '_LED_roi.mp4'

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path_linux = os.path.expanduser('~/Documents/datasets/semi-controlled')
    if OS_linux:
        db_path = db_path_linux
    else:
        db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get input base directory
    db_path_input = os.path.join(db_path, "2_processed", "kinect")
    # set output base directory
    db_path_output = os.path.join(db_path, "2_processed", "kinect")
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

    use_specific_sessions = True
    if not use_specific_sessions:
        sessions = []
        sessions = sessions + sessions_ST13
        sessions = sessions + sessions_ST14
        sessions = sessions + sessions_ST15
        sessions = sessions + sessions_ST16
        sessions = sessions + sessions_ST18
    else:
        sessions = ['2022-06-17_ST16-02']
    
    use_specific_blocks = True
    specific_blocks = ['block-order16']


    for session in sessions:
        # files_abs, files, _ = find_metadata_files(db_path_input, session)
        curr_dir = os.path.join(db_path_input, session)
        files_abs, files = path_tools.find_files_in_directory(curr_dir, ending=video_extension)

        # Process LED blinking
        for idx, (file_abs, file) in enumerate(zip(files_abs, files)):
            if use_specific_blocks :
                is_not_specific_block = True
                for block in specific_blocks:
                    if block in file:
                        is_not_specific_block = False
                if is_not_specific_block:
                    continue
            print("--- --- --- --- ---")
            print(f"Current file is: '{file}'")
            
            # Results filename and location
            output_dirname = os.path.dirname(file_abs).replace(db_path_input, db_path_output)
            output_filename_csv = file.replace(video_extension, "_LED.csv")
            output_filename_md = file.replace(video_extension, "_LED_metadata.txt")
            output_filename_csv_abs = os.path.join(output_dirname, output_filename_csv)
            output_filename_md_abs = os.path.join(output_dirname, output_filename_md)

            if not force_processing and os.path.exists(output_filename_csv_abs):
                print(f"File '{file}' already exists and the processing is not forced. Move to next file...")
                continue
            
            # create Kinect processing manager
            led_blink = KinectLEDBlinkingMP4(file_abs)
        
            # if the video hasn't been processed yet or redo the processing (load results = False)
            led_blink.load_video()
            # Find pixels that are most likely to be the LED in the video
            led_blink.find_bimodal_green_pixels()
            # Run the monitoring (results are stored in processor.bimodal_green_levels)
            bimodal_levels = led_blink.monitor_bimodal_pixels_green()
            led_blink.process_led_on()
            # correct for any occlusion
            #led_blink.define_occlusion(threshold=40, show=show_video_by_frames)

            if save_results:
                if not os.path.exists(output_dirname):
                    os.makedirs(output_dirname)
                led_blink.save_result_csv(output_filename_csv_abs)
                led_blink.save_result_metadata(output_filename_md_abs)

            if show:
                plt.plot(led_blink.time, led_blink.green_levels)
                plt.plot(led_blink.time, led_blink.led_on*255)
                plt.ion()
                plt.show()
                WaitForButtonPressPopup()
                plt.close()

    if save_results:
        p = db_path_output.replace("\\", "/")
        print(f"Results saved in:\n{p}")
