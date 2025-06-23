import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libraries.primary.KinectLEDRegionOfInterest import KinectLEDRegionOfInterest  # noqa: E402
from libraries.primary.KinectLEDRegionOfInterestMP4 import KinectLEDRegionOfInterestMP4  # noqa: E402
import libraries.misc.path_tools as path_tools  # noqa: E402


if __name__ == "__main__":
    force_processing = True  # If user wants to force data processing even if results already exist
    save_results = True

    show = False  # If user wants to monitor what's happening

    video_extension = '.mp4'

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    OS_linux = sys.platform.startswith('linux')
    db_path_linux = os.path.expanduser('~/Documents/datasets/semi-controlled')
    if OS_linux:
        db_path = db_path_linux
    else:
        db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get input base directory
    db_path_input = os.path.join(db_path, "1_primary", "kinect")
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

    print("Selected sessions:")
    print(np.transpose(sessions))
    print("--- --- --- --- ---")
    
    print("Step 1: Determine manually the location of the LED.")
    # 1. Determine:
    #   a. If the file contains the LED (not out of frame)
    #   b. If there is a good frame (no occlusion)
    #   c. Then extract LED location if both are fulfilled
    for session in sessions:
        curr_dir = os.path.join(db_path_input, session)
        files_abs, files = path_tools.find_files_in_directory(curr_dir, ending=video_extension)

        for file_abs, file in zip(files_abs, files):
            print("--- --- --- --- ---")
            print(f"Current file is: '{file}'")
            if use_specific_blocks :
                is_not_specific_block = True
                for block in specific_blocks:
                    if block in file:
                        is_not_specific_block = False
                if is_not_specific_block:
                    continue

            # Results filename and location
            output_dirname = os.path.dirname(file_abs).replace(db_path_input, db_path_output)
            output_filename = file.replace(video_extension, "_LED_roi_metadata.txt")
            output_filename_abs = os.path.join(output_dirname, output_filename)

            if not force_processing and os.path.exists(output_filename_abs):
                print(f"File '{file}' already exists and the processing is not forced. Move to next file...")
                continue

            # create Kinect processing manager
            led_roi = KinectLEDRegionOfInterestMP4(file_abs)
            led_roi.initialise_video()
            
            is_done = False
            while not is_done:
                frame_number = led_roi.get_frame_id_with_led()
                # if the LED is occluded or out of bounds
                if frame_number is None:
                    is_done = True
                    continue

                led_roi.set_reference_frame(frame_number)
                location_set = led_roi.draw_led_location()
                if location_set is not None:
                    is_done = True

            if save_results:
                if not os.path.exists(output_dirname):
                    os.makedirs(output_dirname)
                led_roi.save_result_metadata(output_filename_abs)

    if save_results:
        p = db_path_output.replace("\\", "/")
        print(f"Results saved in:\n{p}")
    print(f"All processings are done.")

