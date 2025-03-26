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
    OS_linux = True

    force_processing = False  # If user wants to force data processing even if results already exist
    save_results = True

    video_extension = '.mp4'
    metadata_filename_end = '_LED_roi_metadata.txt'
    show = False  # If user wants to monitor what's happening

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

    sessions = []
    sessions = sessions + sessions_ST13
    sessions = sessions + sessions_ST14
    sessions = sessions + sessions_ST15
    sessions = sessions + sessions_ST16
    sessions = sessions + sessions_ST18

    print("Selected sessions:")
    print(np.transpose(sessions))
    print("--- --- --- --- ---")
    
    idx = -1
    # 2. Process LED blinking
    for session in sessions:
        curr_dir = os.path.join(db_path_input, session)
        files_abs, files = path_tools.find_files_in_directory(curr_dir, ending=metadata_filename_end)

        for file_abs, file in zip(files_abs, files):
            idx = idx + 1
            print("--- --- --- --- ---")
            print(f"Current file is: '{file}'")

            # Results filename and location
            output_dirname = os.path.dirname(file_abs).replace(db_path_input, db_path_output)
            output_filename = file.replace(metadata_filename_end, "_LED_roi.mp4")
            output_filename_abs = os.path.join(output_dirname, output_filename)

            if not force_processing and os.path.exists(output_filename_abs):
                print(f"File '{file}' already exists and the processing is not forced. Move to next file...")
                continue

            led_roi = KinectLEDRegionOfInterestMP4()
            led_roi.load_metadata(file_abs)
            if not led_roi.led_in_frame:
                continue
            
            led_roi.initialise_video()
            led_roi.set_reference_frame(led_roi.reference_frame_idx)
            led_roi.extract_roi()

            if save_results:
                if not os.path.exists(output_dirname):
                    os.makedirs(output_dirname)
                led_roi.save_roi_as_video(output_filename_abs)

        if save_results:
            p = db_path_output.replace("\\", "/")
            print(f"Results saved in:\n{p}")
        print(f"All processings are done.")

