import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libraries.primary.semicontrolled_Kinect_roi import KinectRegionOfInterest  # noqa: E402
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
    if not isinstance(sessions, list):
        sessions = [sessions]

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
    show = True  # If user wants to monitor what's happening

    force_processing = True  # If user wants to force data processing even if results already exist
    save_results = True

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled", "1_primary", "kinect")
    db_path = 'C:\\Users\\basil\\tactile-sign-languages_kinect\\2025-03-03'
    # get input base directory
    db_path_input = os.path.join(db_path, "1_block-order")
    # get output base directory
    db_path_output = os.path.join(db_path, "2_handstickers_roi")
    if save_results and not os.path.exists(db_path_output):
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
    print("Selected sessions:")
    print(np.transpose(sessions))
    print("--- --- --- --- ---")

    print("Step 1: Determine manually the location of the LED.")
    # 1. Determine:
    #   a. If the file contains the region of interest (not out of frame)
    #   b. If there is a good frame (no occlusion)
    #   c. If a and b are fulfilled, extract target location
    frame_idx = []
    square_center = []
    square_radius = []
    
    for session in sessions:
        curr_kinect_dir = os.path.join(db_path_input, session)
        mkv_files_abs, mkv_files = path_tools.find_files_in_directory(curr_kinect_dir, ending='.mkv')

        # Results filename and location
        output_filename_abs = os.path.join(db_path_output, session + "_kinect_handstickers_roi_metadata.txt")

        for mkv_filename_abs, mkv_filename in zip(mkv_files_abs, mkv_files):
            print(f"File '{mkv_filename}'")

            # create Kinect processing manager
            handstickers_roi = KinectRegionOfInterest(mkv_filename_abs, output_filename_abs)

            if not force_processing and handstickers_roi.is_already_processed():
                print(f"Results already exist and user decided to not force processing.")
                print(f"Go to next file --->")
            else:
                handstickers_roi.initialise_video()
                frame_number = handstickers_roi.select_good_frame(frame_range = 'second-half')

                # if the target is not occluded or not out of bounds, proceed to draw a rectangle around the target
                if frame_number is not None:
                    handstickers_roi.set_reference_frame(frame_number)
                    handstickers_roi.select_target_location()
                    if save_results and (force_processing or not(os.path.exists(output_filename_abs))):
                        handstickers_roi.save_result(verbose=True)
                    break  # Exit the inner loop immediately, go to next neuron

    if save_results:
        p = db_path_output.replace("\\", "/")
        print(f"Results saved in:\n{p}")
    print(f"All processings are done.")
