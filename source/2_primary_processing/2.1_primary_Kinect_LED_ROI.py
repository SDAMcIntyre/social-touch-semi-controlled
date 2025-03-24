import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from source.libraries.primary.KinectLEDRegionOfInterest import KinectLEDRegionOfInterest  # noqa: E402
from source.libraries.primary.KinectLEDRegionOfInterestMP4 import KinectLEDRegionOfInterestMP4  # noqa: E402
import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.misc.waitforbuttonpress_popup import WaitForButtonPressPopup


if __name__ == "__main__":
    use_mp4 = True

    force_processing = False  # If user wants to force data processing even if results already exist
    load_led_location = False  # If user wants to load already defined LED location if it already exists
    show = False  # If user wants to monitor what's happening

    save_results = True

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled", "1_primary", "kinect")
    # get input base directory
    if use_mp4:
        db_path_input = os.path.join(db_path, "1_block-order-mp4")
        extension = '.mp4'
        # get output base directory
        db_path_output = os.path.join(db_path, "2_roi_led_mp4")
    else:
        db_path_input = os.path.join(db_path, "1_block-order")
        extension = '.mkv'
        # get output base directory
        db_path_output = os.path.join(db_path, "2_roi_led")
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

    print("Selected sessions:")
    print(np.transpose(sessions))
    print("--- --- --- --- ---")

    print("Step 1: Determine manually the location of the LED.")
    # 1. Determine:
    #   a. If the file contains the LED (not out of frame)
    #   b. If there is a good frame (no occlusion)
    #   c. Then extract LED location if both are fulfilled
    led_in_frames = []
    frame_idx = []
    square_center = []
    square_radius = []
    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:
        curr_dir = os.path.join(db_path_input, session)
        files_abs, files = path_tools.find_files_in_directory(curr_dir, ending=extension)

        for file_abs, file in zip(files_abs, files):
            print(f"File '{file}'")

            # Results filename and location
            output_dirname = os.path.join(db_path_output, session)
            output_filename = file.replace("_kinect"+extension, "") + "_kinect_LED_roi"

            # create Kinect processing manager
            if "mp4" in extension:
                led_roi = KinectLEDRegionOfInterestMP4(file_abs, output_dirname, output_filename)
            else:
                led_roi = KinectLEDRegionOfInterest(file_abs, output_dirname, output_filename)

            if not force_processing and led_roi.is_already_processed():
                print(f"Results already exist and user decided to not force processing.")
                print(f"Go to next file --->")
            else:
                # force processing or if the video hasn't been processed yet
                if load_led_location and led_roi.is_already_processed():
                    led_roi.load_led_location()
                else:
                    led_roi.initialise_video()
                    frame_number = led_roi.select_good_frame()

                    # if the LED is not occluded or not out of bounds
                    if frame_number is not None:
                        led_roi.set_reference_frame(frame_number)
                        led_roi.select_led_location()

            # keep the variables, even if the results already exist (to not mess up idx in step 2)
            led_in_frames.append(led_roi.led_in_frame)
            frame_idx.append(led_roi.reference_frame_idx)
            square_center.append(led_roi.square_center)
            square_radius.append(led_roi.square_size)

    print("Step 2: Automatic processing...")
    idx = -1
    # 2. Process LED blinking
    for session in sessions:
        curr_dir = os.path.join(db_path_input, session)
        files_abs, files = path_tools.find_files_in_directory(curr_dir, ending=extension)

        output_session_abs = os.path.join(db_path_output, session)
        if save_results and not os.path.exists(output_session_abs):
            os.makedirs(output_session_abs)
            print(f"Directory '{output_session_abs}' created.")

        for file_abs, file in zip(files_abs, files):
            print(f"File '{file}'")

            idx = idx + 1

            # Results filename and location
            output_dirname = os.path.join(db_path_output, session)
            output_filename = file.replace("_kinect"+extension, "") + "_kinect_LED_roi"

            # create Kinect processing manager
            if "mp4" in extension:
                led_roi = KinectLEDRegionOfInterestMP4(file_abs, output_dirname, output_filename)
            else:
                led_roi = KinectLEDRegionOfInterest(file_abs, output_dirname, output_filename)

            if not force_processing and led_roi.is_already_processed():
                print(f"Results already exist and user decided to not force processing.")
                print(f"Go to next file --->")
                continue


            if led_in_frames[idx]:
                # if the video hasn't been processed yet or redo the processing (load results = False)
                led_roi.led_in_frame = led_in_frames[idx]
                led_roi.initialise_video()
                led_roi.set_reference_frame(frame_idx[idx])
                led_roi.select_led_location(square_center[idx], square_radius[idx])
                led_roi.extract_roi()
            else:
                led_roi.initialise_video()
                led_roi.extract_metadata_video()
                print(f"LED is not in frame, extract and save metadata only.")

            if save_results:
                led_roi.save_results(verbose=True)

        if save_results:
            p = db_path_output.replace("\\", "/")
            print(f"Results saved in:\n{p}")
        print(f"All processings are done.")

