from datetime import datetime
import numpy as np
import pandas as pd
import re

import os
import sys
import warnings

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libraries.misc.path_tools as path_tools  # noqa: E402


def find_mp4_files(input_path, sessions):
    if not isinstance(sessions, list):
        sessions = [sessions]

    stim_files_session = []
    stim_files = []
    stim_files_abs = []

    for session in sessions:
        dir_path = os.path.join(input_path, session)
        # Walk through the directory recursively
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.mp4'):
                    stim_files.append(file)
                    stim_files_abs.append(os.path.join(root, file))
                    stim_files_session.append(session)

    return stim_files_abs, stim_files, stim_files_session


# extract the stimulus characteristics from log files
# this script mainly does two things:
# 1. save the results into separate files for each block to match the Kinect videos format
# 2. redefine the block ids as repeated block's reset block ids to 1 every time (1, 2, 1, 2, 3 instead of 1, 2, 3, 4, 5)
if __name__ == "__main__":
    force_processing = False  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening
    save_results = True

    print("Step 0: Extract the selected sessions.")
    # get database directory
    database_path = path_tools.get_database_path()
    # get input base directory
    database_path_input = os.path.join(database_path, "semi-controlled", "primary", "kinect", "2_roi_led")
    # get output base directory
    database_path_output = os.path.join(database_path, "semi-controlled", "primary", "kinect", "2_roi_led")
    if not os.path.exists(database_path_output):
        os.makedirs(database_path_output)
        print(f"Directory '{database_path_output}' created.")
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
    print(sessions)

    # it is important to split by MNG files / neuron recordings to get the correct block order.
    for neuron_sessions in sessions:
        stim_files_abs, stim_files, stim_files_session = find_mp4_files(database_path_input, neuron_sessions)

        # ensure that the extracted files are in the time order (to set the correct block order)
        # Step 1: Extract and convert strings to datetime objects
        date_start_idx = len("NoIR_")
        date_end_idx = date_start_idx + len("YYYY-MM-DD_HH-MM-SS")
        date_objects = [datetime.strptime(date_str[date_start_idx:date_end_idx], "%Y-%m-%d_%H-%M-%S") for date_str in stim_files]
        # Step 2: Create a list of indices and sort them based on date objects
        sorted_indices = sorted(range(len(date_objects)), key=lambda k: date_objects[k])
        # Step 3: Use sorted indices to reorder original list
        stim_files_abs = [stim_files_abs[i] for i in sorted_indices]
        stim_files = [stim_files[i] for i in sorted_indices]
        stim_files_session = [stim_files_session[i] for i in sorted_indices]

        print(np.transpose(stim_files))
        print(sorted_indices)
        print("---\n")

        global_block_id = 0
        for filename_input_abs, filename_input, session in zip(stim_files_abs, stim_files, stim_files_session):
            # output directory
            output_dirname = os.path.join(database_path_output, session)
            if not os.path.exists(output_dirname):
                os.makedirs(output_dirname)
            # Prepare output filename to match standard
            fname = filename_input.replace("NoIR_", "").replace(".mp4", "")
            substrings = fname.split("_")
            date = substrings[0]
            neuron_id = substrings[3] + "-0" + substrings[4]

            # extract the block id for the current set of stimuli
            global_block_id += 1
            global_block_id_str = str(global_block_id)
            if global_block_id < 10:
                global_block_id_str = "0" + global_block_id_str

            # create the standard filename
            filename_output = date + "_" + neuron_id + "_semicontrolled_block" + global_block_id_str + "_LED_roi.mp4"
            filename_output_abs = os.path.join(output_dirname, filename_output)

            filename_input_abs = os.path.join(output_dirname, filename_input)
            print(output_dirname)
            print("old name:")
            print(filename_input)
            print("new name:")
            print(filename_output)
            hehe = 1
            try:
                os.rename(filename_input_abs, filename_output_abs)
                os.rename(filename_input_abs.replace(".mp4", "_metadata.txt"), filename_output_abs.replace(".mp4", "_metadata.txt"))
            except:
                pass


