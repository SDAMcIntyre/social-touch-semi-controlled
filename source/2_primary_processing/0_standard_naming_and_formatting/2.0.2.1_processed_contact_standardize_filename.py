import csv
from datetime import datetime
import numpy as np
import pandas as pd
import re
import shutil

import os
import sys
import warnings

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libraries.misc.path_tools as path_tools  # noqa: E402


def find_csv_files(input_path, sessions):
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
                if file.endswith('.csv'):
                    stim_files.append(file)
                    stim_files_abs.append(os.path.join(root, file))
                    stim_files_session.append(session)

    return stim_files_abs, stim_files, stim_files_session


# Extract the file mapping standard from the kinect videos and redo it on data shared by Shan
if __name__ == "__main__":
    force_processing = True  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening

    save_results = True

    print("Step 0: Extract the selected sessions.")
    # get database directory
    database_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get input base directory
    database_path_input = os.path.join(database_path, "2_processed", "kinect", "contact", "0_raw_date-adjusted")
    # get output base directory
    database_path_output = os.path.join(database_path, "2_processed", "kinect", "contact", "1_block-order")
    if not os.path.exists(database_path_output):
        os.makedirs(database_path_output)
        print(f"Directory '{database_path_output}' created.")

    # get metadata dataframe
    metadata_path = os.path.join(database_path, "1_primary", "logs", "1_kinect-name_to_block-order")

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


    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for neuron_session in sessions:
        # output directory
        output_dirname = os.path.join(database_path_output, neuron_session)
        if not os.path.exists(output_dirname):
            os.makedirs(output_dirname)

        stim_files_abs, stim_files, stim_files_session = find_csv_files(database_path_input, neuron_session)
        print(np.transpose(stim_files))
        print("---\n")

        # load the kinect filename - to - block-order csv file of the session
        filename_to_blockorder_path_abs = os.path.join(metadata_path, neuron_session + "_kinect-name_to_block-id.csv")
        df_blockorder = pd.read_csv(filename_to_blockorder_path_abs)

        for filename_input_abs, filename_input, session in zip(stim_files_abs, stim_files, stim_files_session):
            # Prepare output filename to match standard
            fname = filename_input.replace("NoIR_", "")
            substrings = fname.split("_")
            date = substrings[0]
            neuron_id = f"{substrings[3]}-{int(substrings[4]):02d}"

            supposed_fname_kinect = fname.replace("-ContQuantAll.csv", ".mkv")
            matched_row = df_blockorder[df_blockorder["kinect_filenames"] == supposed_fname_kinect]
            if not matched_row.empty:
                block_order = matched_row["block_order"].values[0]
            else:
                warnings.warn(f"No match found for '{fname}'")
                block_order = np.nan

            # create the standard filename
            filename_output = date + "_" + neuron_id + "_semicontrolled_block-order" + f"{block_order:02d}" + "_contact.csv"
            filename_output_abs = os.path.join(output_dirname, filename_output)
            print(output_dirname)
            print("old name:")
            print(filename_input)
            print("new name:")
            print(filename_output)

            if save_results:
                shutil.copy(filename_input_abs, filename_output_abs)
