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


def find_files_in_directory(input_path, sessions, ending='.csv'):
    if not isinstance(sessions, list):
        sessions = [sessions]

    files_session = []
    files = []
    files_abs = []

    for session in sessions:
        dir_path = os.path.join(input_path, session)
        # Walk through the directory recursively
        for root, _, f in os.walk(dir_path):
            for file in f:
                if file.endswith(ending):
                    files.append(file)
                    files_abs.append(os.path.join(root, file))
                    files_session.append(session)

    return files_abs, files, files_session


# Extract the file mapping standard from the kinect videos and redo it on data shared by Shan
if __name__ == "__main__":
    show = False  # If user wants to monitor what's happening
    save_results = True

    print("Step 0: Extract the selected sessions.")
    # load base directories
    md_path = path_tools.get_metadata_path()
    db_path = path_tools.get_database_path()
    db_path = os.path.join(db_path, "semi-controlled")

    # get metadata file
    df_nerve_kinect_filename = os.path.join(md_path, 'semicontrolled_data-collection_quality-check.xlsx')
    df_quality_control = pd.read_excel(df_nerve_kinect_filename)

    # get input directory
    db_path_input = os.path.join(db_path, "processed", "nerve", "1_csv_files")
    # get output directory
    db_path_output = os.path.join(db_path, "processed", "nerve", "2_block-order")
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
    print(np.transpose(sessions))

    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for neuron_sessions in sessions:
        files_abs, files, files_session = find_files_in_directory(db_path_input, neuron_sessions, ending='.csv')
        print(np.transpose(files))
        print("---\n")

        for filename_input_abs, filename_input, session in zip(files_abs, files, files_session):
            print(f"current session: {session}")
            print(f"current filename: {filename_input}")
            # output directory
            output_dirname = os.path.join(db_path_output, session)
            if not os.path.exists(output_dirname):
                os.makedirs(output_dirname)

            # extract zoom id and block id of the current csv file
            filename_chunks = filename_input.split("_")
            zoom_id = int(filename_chunks[3])
            block_id = int(filename_chunks[5].replace("block", ""))

            # find block order value of the current file/block
            try:
                result = df_quality_control.loc[(df_quality_control['Zoom'] == zoom_id) & (df_quality_control['Zoom Block ID'] == block_id), 'Block order']
                if result.empty:
                    warnings.warn("Zoom file + block id combination couldn't be found!")
                block_order_id = int(result.values[0])
            except:
                pass

            # create the filename using the standard
            filename_output = f"{session}_semicontrolled_block-order{block_order_id:02}_nerve.csv"
            filename_output_abs = os.path.join(output_dirname, filename_output)

            print(f"new filename: {filename_output}")
            print("---\n")
            try:
                os.rename(filename_input_abs, filename_output_abs)
            except:
                pass


