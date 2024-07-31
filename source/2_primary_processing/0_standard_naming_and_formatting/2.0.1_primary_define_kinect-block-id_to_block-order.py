import csv
from datetime import datetime
import numpy as np
import pandas as pd
import re
import os
import sys
import warnings

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libraries.misc.path_tools as path_tools  # noqa: E402


def find_stimuli_files(input_path, sessions):
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
                if file.endswith('_stimuli.csv'):
                    stim_files.append(file)
                    stim_files_abs.append(os.path.join(root, file))
                    stim_files_session.append(session)

    return stim_files_abs, stim_files, stim_files_session


# extract the stimulus characteristics from log files
# this script mainly does two things:
# 1. save the results into separate files for each block to match the Kinect videos format
# 2. Define global block id, as several runs could have been done in one neuron, and block id is always reset to 1
if __name__ == "__main__":
    force_processing = False  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening

    save_results = True

    print("Step 0: Extract the selected sessions.")
    # get database directory
    database_path = path_tools.get_database_path()
    # get input base directory
    database_path_input = os.path.join(database_path, "semi-controlled", "1_primary", "logs", "0_raw")
    # get output base directory
    database_path_output = os.path.join(database_path, "semi-controlled", "1_primary", "logs", "1_kinect-name_to_block-order")
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
    for neuron_session in sessions:
        # output filename
        filename_output_abs = os.path.join(database_path_output, neuron_session + "_kinect-name_to_block-id.csv")

        stim_files_abs, stim_files, stim_files_session = find_stimuli_files(database_path_input, neuron_session)

        # ensure that the extracted files are in the time order (to set the correct block order)
        # Step 1: Extract and convert strings to datetime objects
        date_end_idx = len("YYYY-MM-DD_HH-MM-SS")
        date_objects = [datetime.strptime(date_str[:date_end_idx], "%Y-%m-%d_%H-%M-%S") for date_str in stim_files]
        # Step 2: Create a list of indices and sort them based on date objects
        sorted_indices = sorted(range(len(date_objects)), key=lambda k: date_objects[k])
        # Step 3: Use sorted indices to reorder original list
        stim_files_abs = [stim_files_abs[i] for i in sorted_indices]
        stim_files = [stim_files[i] for i in sorted_indices]
        stim_files_session = [stim_files_session[i] for i in sorted_indices]

        print("stim_files:")
        print(np.transpose(stim_files))
        print("sorted_indices:")
        print(sorted_indices)
        print("---\n")

        kinect_filenames = []
        for filename_input_abs, filename_input, session in zip(stim_files_abs, stim_files, stim_files_session):
            # load stimulus characteristics
            df = pd.read_csv(filename_input_abs)
            # if the log file is empty, the run has been canceled before doing anything
            if df.empty:
                continue

            kinect_filenames_current = df.kinect_recording.values

            # Extract only the filename from each path
            kinect_filenames_current = [os.path.basename(filepath) for filepath in kinect_filenames_current]
            # save them
            kinect_filenames.extend(kinect_filenames_current)

        # Keep only unique filenames
        unique_kinect_filenames = list(set(kinect_filenames))

        # Sort the list alphabetically and numerically
        unique_kinect_filenames = sorted(unique_kinect_filenames, key=lambda x: (re.split(r'(\d+)', x.lower()), x))

        block_ids = [int(re.search(r'block(\d+)', filename).group(1)) for filename in unique_kinect_filenames]

        # Create a list of tuples where each tuple contains (filename, block-order)

        filename_block_tuples = []
        for idx, (filename, block_id) in enumerate(zip(unique_kinect_filenames, block_ids)):
            filename_block_tuples.append((filename, block_id, idx + 1))

        # Write to CSV
        with open(filename_output_abs, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['kinect_filenames', 'block_id', 'block_order'])
            csvwriter.writerows(filename_block_tuples)

    if save_results:
        p = database_path_output.replace("\\", "/")
        print(f"Results saved in:\n{p}")
