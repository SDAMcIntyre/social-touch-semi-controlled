import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import shutil
import sys
import warnings

# homemade libraries
# current_dir = Path(__file__).resolve()
sys.path.append(str(Path(__file__).resolve().parent.parent))
import libraries.misc.path_tools as path_tools  # noqa: E402


def find_groups_of_ones(arr):
    groups = []
    in_group = False
    start_index = 0

    for i, value in enumerate(arr):
        if value == 1 and not in_group:
            # Start of a new group
            in_group = True
            start_index = i
        elif value == 0 and in_group:  
            # End of the current group
            in_group = False
            groups.append(list(range(start_index, i)))  # range is right boundary exclusive

    # If the array ends with a group of 1s
    if in_group:
        groups.append(list(range(start_index, len(arr))))

    return groups


if __name__ == "__main__":
    # parameters
    force_processing = True  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening

    save_results = False

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled", "3_merged", "1_kinect_and_nerve_shandata")

    # get input base directory
    db_path_input = os.path.join(db_path, "0_by-units_renamed_trial-corrected_block-corrected")
    # get output base directory
    db_path_output = os.path.join(db_path, "1_by-trials")

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
    #print(sessions)
    for session in sessions:
        file = [f.name for f in Path(db_path_input).iterdir() if session in f.name]
        if len(file) != 1:
            warnings.warn(f"Issue detected: not exactly 1 csv file found for {session}.")
            continue
        file = file[0]
        file_abs = os.path.join(db_path_input, file)
        print(f"current file: {file}")
        
        output_session_abs = os.path.join(db_path_output, session)
        if not os.path.exists(output_session_abs):
            os.makedirs(output_session_abs)
            print(f"Directory '{output_session_abs}' created.")

        # load current data
        data_unit = pd.read_csv(file_abs)
        for block_order in pd.unique(data_unit["block_order"]):
            if np.isnan(block_order):
                continue
            block_order_str = f"block-order{round(block_order):02}"
            output_dir_abs = os.path.join(output_session_abs, block_order_str)
            data_block = data_unit[data_unit['block_order'] == block_order]

            for trial_id in pd.unique(data_block["trial_id"]):
                if math.isnan(trial_id):
                    continue
                print(f"[{file}] block_order = {block_order}, trial_id = {trial_id}")

                data_trial = data_block[data_block['trial_id'] == trial_id]

                output_filename = file.replace(".csv", f"_block-order{round(block_order):02}_trial{round(trial_id):02}.csv")
                output_filename_abs = os.path.join(output_dir_abs, output_filename)
                if not force_processing:
                    try:
                        with open(output_filename_abs, 'r'):
                            print("Result file exists, jump to the next dataset.")
                            continue
                    except FileNotFoundError:
                        pass

                if trial_id > 3 and len(data_trial) < 100:
                    pass

                if show:
                    plt.figure(figsize=(10, 12))  # Increase height for two subplots
                    plt.plot(data_trial["Nerve_TTL"].values, label='adjusted')
                    plt.plot(data_trial["LED on"].values, label='TTL_kinect_rescale', alpha=0.6, linestyle='--')
                    plt.legend()
                    plt.title('TTL_kinect_rescale')
                    plt.show()

                # save data on the hard drive ?
                if save_results:
                    if not os.path.exists(output_dir_abs):
                        os.makedirs(output_dir_abs)
                        print(f"Directory '{output_dir_abs}' created.")

                    # https://answers.microsoft.com/en-us/msoffice/forum/all/excel-file-open-the-file-name-is-too-long-rename/ef736fec-0bd4-42a9-806d-5b22dbfdda81#:~:text=To%20resolve%20this%20issue%2C%20you,structure%2C%20is%20still%20too%20long.
                    #  Excel indicates that the total path length,
                    #  including the filename and its directory structure,
                    #  exceeds the Windows maximum limit of 260 characters.
                    data_trial.to_csv(output_filename_abs, index=False)
                
            print("done.")

























