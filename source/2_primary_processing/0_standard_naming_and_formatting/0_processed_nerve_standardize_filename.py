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
    force_processing = False  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening
    save_results = True

    print("Step 0: Extract the selected sessions.")
    # get database directory
    database_path = path_tools.get_database_path()
    # filename mapping location
    filename_mapping = os.path.join(database_path, "semi-controlled", "primary", "kinect", "1_standard_name", "standard_filename_mapping.csv")
    # get input base directory
    database_path_input = os.path.join(database_path, "semi-controlled", "processed", "nerve", "0_named_by_runs")
    # get output base directory
    database_path_output = os.path.join(database_path, "semi-controlled", "processed", "nerve", "1_standard_name")
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
    # Session names
    sessions_ST13 = ['ST13_unit1',
                     'ST13_unit2',
                     'ST13_unit3']

    sessions_ST14 = ['ST14_unit1',
                     'ST14_unit2',
                     'ST14_unit3',
                     'ST14_unit4']

    sessions_ST15 = ['ST15_unit1',
                     'ST15_unit2']

    sessions_ST16 = ['ST16_unit2',
                     'ST16_unit3',
                     'ST16_unit4',
                     'ST16_unit5']

    sessions_ST18 = ['ST18_unit1',
                     'ST18_unit2',
                     'ST18_unit4']

    sessions = []
    sessions = sessions + sessions_ST13
    sessions = sessions + sessions_ST14
    sessions = sessions + sessions_ST15
    sessions = sessions + sessions_ST16
    sessions = sessions + sessions_ST18
    print(sessions)

    # load the filename mapping csv as a dictionary
    mapping = {}
    with open(filename_mapping, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            old_filename, new_filename = row
            mapping[old_filename] = new_filename

    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for neuron_sessions in sessions:
        stim_files_abs, stim_files, stim_files_session = find_csv_files(database_path_input, neuron_sessions)
        print(np.transpose(stim_files))
        print("---\n")

        for filename_input_abs, filename_input, session in zip(stim_files_abs, stim_files, stim_files_session):
            # Iterate through the dictionary
            substring_cutoff = "controlled-touch"
            substring_dirname_cutoff = "_semicontrolled"
            standard_kinect_filename = ""
            standard_foldername = ""
            for key in mapping:
                start_index = key.find(substring_cutoff)
                key_short = key[start_index:].replace(".mkv", "")
                if key_short in filename_input:
                    standard_kinect_filename = mapping[key]
                    # get the base filename to create the standard folder name
                    start_index = standard_kinect_filename.find(substring_dirname_cutoff)
                    standard_foldername = standard_kinect_filename[:start_index]
                    break
            if standard_kinect_filename == "":
                warnings.warn("filename not found as a key!")

            # create output directory standard name
            output_dirname = os.path.join(database_path_output, standard_foldername)
            if not os.path.exists(output_dirname):
                os.makedirs(output_dirname)

            # create the filename using the standard
            filename_output = standard_kinect_filename.replace("kinect.mkv", "nerve.csv")
            filename_output_abs = os.path.join(output_dirname, filename_output)

            print("current directory:")
            print(output_dirname)
            print("old name:")
            print(filename_input)
            print("new name:")
            print(filename_output)
            try:
                os.rename(filename_input_abs, filename_output_abs)
            except:
                pass


