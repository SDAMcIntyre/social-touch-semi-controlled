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
    database_path_input = os.path.join(database_path, "semi-controlled", "primary", "logs", "stimuli_by_blocks")
    # get output base directory
    database_path_output = os.path.join(database_path, "semi-controlled", "processed")
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

    # generate path to the output csv file
    output_filename_abs = os.path.join(database_path_output, "block_usability.csv")
    # Create an empty DataFrame to store combined data
    combined_blocks = []

    # it is important to split by MNG files / neuron recordings to get the correct block order.
    for neuron_sessions in sessions:
        _, stim_files, _ = find_stimuli_files(database_path_input, neuron_sessions)

        for filename_input in stim_files:
            print(f"read filename{filename_input}")

            # Extract values from filename
            substrings = filename_input.split("_")
            date = substrings[0]
            unit_name = substrings[1]
            block_id = substrings[3].replace("block", "")

            # Create a dictionary with the extracted values and default values
            current_block = {
                'date': [date],
                'unit_name': [unit_name],
                'block_id': [block_id],
                'unit_type': ["N/A"],
                'zoom_file': ["N/A"],
                'usable_kinect': [1],
                'usable_neuron': ["N/A"],
                'good_for_analysis': ["N/A"]
            }
            current_block_df = pd.DataFrame(current_block)

            # Append the DataFrame to the list of blocks
            combined_blocks.append(current_block_df)

    final_df = pd.concat(combined_blocks, ignore_index=True)
    # Save the combined DataFrame to a new CSV file
    final_df.to_csv(output_filename_abs, index=False)
