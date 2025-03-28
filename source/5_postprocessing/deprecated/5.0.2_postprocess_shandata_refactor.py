import os
# os.environ["MPLBACKEND"] = "Agg"  # Use a non-interactive backend

import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'

import math
import matplotlib.pyplot as plt
import numpy as np
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


if __name__ == "__main__":
    # parameters
    force_processing = True  # If user wants to force data processing even if results already exist
    save_results = True

    use_latest_dataset = True

    show = False  # If user wants to monitor what's happening

    print("Step 0: Extract the videos embedded in the selected sessions.")
    if use_latest_dataset:
        # get database directory
        db_path = os.path.join(path_tools.get_database_path(), "semi-controlled", "3_merged", "1_kinect_and_nerve_shandata")
        # get input base directory
        db_path_input = os.path.join(db_path, "0_1_by-units_jan-and-sept")
    else:
        db_path = os.path.join(path_tools.get_database_path(), "semi-controlled", "3_merged", "archive", "contact_and_neural")
        db_path_input = os.path.join(db_path, "new_axes_3Dposition")  # manual_axes, manual_axes_3Dposition_RF, new_axes, new_axes_3Dposition

    # get output base directory
    db_path_output = os.path.join(db_path, "0_2_by-units_jan-and-sept_standard-cols")

    if save_results and not os.path.exists(db_path_output):
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
    
    for session in sessions:
        file = [f.name for f in Path(db_path_input).iterdir() if session in f.name]
        if len(file) != 1:
            warnings.warn(f"Issue detected: not exactly 1 csv file found for {session}.")
            continue
        file = file[0]
        file_abs = os.path.join(db_path_input, file)
        
        # load current data
        data = pd.read_csv(file_abs)
        
        # modify column names to standard
        data = data.rename(columns={'spike': 'Nerve_spike', 
                                    'IFF': 'Nerve_freq', 
                                    'depth': 'contact_depth', 
                                    'area': 'contact_area'})

        print(f"--------------------------------")
        print(f"current file: {file}")
        print(f'{session}: data has {len(data)} samples.')

        if show:
            plt.plot(data["velAbsRaw"], label='velAbsRaw')
            plt.plot(data["Contact_Flag"], label='Contact_Flag Data')
            plt.legend()
            plt.title(session)
            plt.show(block=True)
        
        # save data on the hard drive ?
        if save_results:
            output_filename = f"{session}_semicontrolled.csv"
            output_filename_abs = os.path.join(db_path_output, output_filename)
            # https://answers.microsoft.com/en-us/msoffice/forum/all/excel-file-open-the-file-name-is-too-long-rename/ef736fec-0bd4-42a9-806d-5b22dbfdda81#:~:text=To%20resolve%20this%20issue%2C%20you,structure%2C%20is%20still%20too%20long.
            #  Excel indicates that the total path length,
            #  including the filename and its directory structure,
            #  exceeds the Windows maximum limit of 260 characters.
            data.to_csv(output_filename_abs, index=False)
        
        print("done.")

























