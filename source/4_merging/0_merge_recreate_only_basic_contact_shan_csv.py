from collections import defaultdict, Counter
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import re
import sys
import tkinter as tk
import warnings

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libraries.misc.path_tools as path_tools  # noqa: E402


def find_shan_files(input_path, sessions):
    if not isinstance(sessions, list):
        sessions = [sessions]

    shan_files_session = []
    shan_files = []
    shan_files_abs = []

    for session in sessions:
        # Walk through the directory recursively
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.startswith(session):
                    shan_files.append(file)
                    shan_files_abs.append(os.path.join(root, file))
                    shan_files_session.append(session)

    return shan_files_abs, shan_files, shan_files_session


if __name__ == "__main__":
    """
    Load the CSV data: preprocess, split into single touch event, and save the generated variable with pickle.dump
    """
    force_processing = False  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening
    save_results = True

    print("Step 0: Extract the selected sessions.")
    # get database directory
    database_path = path_tools.get_database_path()
    # get input base directory
    database_path_input = os.path.join(database_path, "semi-controlled", "merged", "archive", "contact_and_neural", "new_axes_3Dposition")
    # get output base directory
    database_path_output = os.path.join(database_path, "semi-controlled", "processed", "kinect", "contact")
    if not os.path.exists(database_path_output):
        os.makedirs(database_path_output)
        print(f"Directory '{database_path_output}' created.")
    # Session names
    sessions_ST13 = ['2022-06-14-ST13-unit1',
                     '2022-06-14-ST13-unit2',
                     '2022-06-14-ST13-unit3']

    sessions_ST14 = ['2022-06-15-ST14-unit1',
                     '2022-06-15-ST14-unit2',
                     '2022-06-15-ST14-unit3',
                     '2022-06-15-ST14-unit4']

    sessions_ST15 = ['2022-06-16-ST15-unit1',
                     '2022-06-16-ST15-unit2']

    sessions_ST16 = ['2022-06-17-ST16-unit2',
                     '2022-06-17-ST16-unit3',
                     '2022-06-17-ST16-unit4',
                     '2022-06-17-ST16-unit5']

    sessions_ST18 = ['2022-06-22-ST18-unit1',
                     '2022-06-22-ST18-unit2',
                     '2022-06-22-ST18-unit4']

    sessions = []
    sessions = sessions + sessions_ST13
    sessions = sessions + sessions_ST14
    sessions = sessions + sessions_ST15
    sessions = sessions + sessions_ST16
    sessions = sessions + sessions_ST18

    stim_files_abs, stim_files, stim_files_session = find_shan_files(database_path_input, sessions)

    scd_list = []
    nsessions = len(sessions)
    for idx, (filename_input_abs, filename_input, session) in enumerate(zip(stim_files_abs, stim_files, stim_files_session)):
        print(f"session ({(idx+1)}/{nsessions}): {filename_input}")

        # Prepare output directory to match standard
        output_directory = session.replace("-ST", "_ST").replace("-unit", "-0")
        output_directory_abs = os.path.join(database_path_output, output_directory)
        if not os.path.exists(output_directory_abs):
            os.makedirs(output_directory_abs)
        # Prepare output filename standard base
        substrings = output_directory.split("_")
        date = substrings[0]
        neuron_id = substrings[1]

        df = pd.read_csv(filename_input_abs)

        if df.isna().any().any():
            warnings.warn("loaded csv file contains nan values")

        # if dataframe is empty, the run has been canceled before doing anything
        if df.empty:
            warnings.warn("ISSUE: dataframe is empty (recreate only basic contact shan csv.py")

        # for each block:
        # - renamed trial into trial_id
        # - select the columns of interest
        # - adjust for the trial ids
        # - save into a separate file
        for block_id, group in df.groupby('block_id'):
            if not pd.notna(block_id):
                warnings.warn("The extracted block doesn't contain any data!")
                continue

            # make a hard copy to avoid issues with dataframe slices
            group = group.copy()

            # adjust time vector to 0
            group["t"] = group["t"] - np.min(group["t"])
            group = group.copy()

            # Keep only certain columns based on the label
            desired_columns = ['t', 'block_id', 'trial_id']
            desired_columns = desired_columns + ['Position_x', 'Position_y', 'Position_z']
            desired_columns = desired_columns + ['Position_index_x', 'Position_index_y', 'Position_index_z']
            filtered_group = group[desired_columns]

            # create the standard filename
            filename_output = date + "_" + neuron_id + "_semicontrolled_block" + str(block_id) + "_contact.csv"
            filename_output_abs = os.path.join(output_directory_abs, filename_output)
            filtered_group.to_csv(filename_output_abs, index=False)

    print("done.")
