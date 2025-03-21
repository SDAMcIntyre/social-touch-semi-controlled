import csv
from datetime import datetime
import numpy as np
import pandas as pd
import re
import os
import shutil
import sys
import warnings

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libraries.misc.path_tools as path_tools  # noqa: E402


def find_mkv_files(input_path):
    mkv_files = []
    mkv_files_abs = []

    # Walk through the directory recursively
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith('.mkv'):
                if not file.startswith('.'):
                    mkv_files.append(file)
                    mkv_files_abs.append(os.path.join(root, file))

    mkv_files_abs = np.array(mkv_files_abs)
    mkv_files = np.array(mkv_files)

    return mkv_files_abs, mkv_files


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
    database_path = os.path.join(path_tools.get_database_path(), "semi-controlled", "1_primary")

    # get input base directory
    external_HDD = "G:"
    database_path_input = os.path.join(external_HDD, "social_touch_MNG_kinect", "semi-controlled")
    # get metadata dataframe
    metadata_path = os.path.join(database_path, "logs", "1_kinect-name_to_block-order")

    # get output base directory
    database_path_output = os.path.join(database_path, "kinect", "1_block-order")
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

    # load all mkv filenames in the input directory
    mkv_files_abs, mkv_files = find_mkv_files(database_path_input)

    # List to store the mapping of old and new filenames
    filename_mapping = []
    # it is important to split by MNG files / neuron recordings to get the correct block order.
    for neuron_session in sessions:
        # output directory
        output_dirname = os.path.join(database_path_output, neuron_session)
        if not os.path.exists(output_dirname):
            os.makedirs(output_dirname)

        # load the kinect filename - to - block-order csv file of the session
        filename_to_blockorder_path_abs = os.path.join(metadata_path, neuron_session + "_kinect-name_to_block-id.csv")
        df_blockorder = pd.read_csv(filename_to_blockorder_path_abs)

        # create current neuron id
        session_split = neuron_session.split("_")
        neuron_id_split = session_split[1].split("-")
        neuron_id = neuron_id_split[0] + f"_{int(neuron_id_split[1])}"

        # List to store indices where the substring "neuron_id" is found
        neuron_id_idx = [index for index, string in enumerate(mkv_files) if neuron_id in string]

        for filename_input_abs, filename_input in zip(mkv_files_abs[neuron_id_idx], mkv_files[neuron_id_idx]):
            # Prepare output filename to match standard
            fname = filename_input.replace("NoIR_", "").replace("noIR_", "")
            substrings = fname.split("_")
            date = substrings[0]
            neuron_id = f"{substrings[3]}-{int(substrings[4]):02d}"

            matched_row = df_blockorder[df_blockorder["kinect_filenames"] == fname]
            if not matched_row.empty:
                block_order = matched_row["block_order"].values[0]
            else:
                print(f"No match found for '{fname}'")

            # create the standard filename
            filename_output = date + "_" + neuron_id + "_semicontrolled_block-order" + f"{block_order:02d}" + "_kinect.mkv"
            filename_output_abs = os.path.join(output_dirname, filename_output)
            print(output_dirname)
            print("old name:")
            print(filename_input)
            print("new name:")
            print(filename_output)

            # Append the mapping to the list
            filename_mapping.append((filename_input, filename_output))

            if save_results:
                if not force_processing and os.path.exists(filename_output_abs):
                    continue
                print(f"copying file with standard name into the output directory...")
                shutil.copy2(filename_input_abs, filename_output_abs)

    filename_mapping_csvfilename = os.path.join(database_path_output, 'standard_filename_mapping.csv')
    # Write the mapping to a CSV file
    with open(filename_mapping_csvfilename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Old Filename', 'New Filename'])
        writer.writerows(filename_mapping)

    if save_results:
        p = database_path_output.replace("\\", "/")
        print(f"Results saved in:\n{p}")
