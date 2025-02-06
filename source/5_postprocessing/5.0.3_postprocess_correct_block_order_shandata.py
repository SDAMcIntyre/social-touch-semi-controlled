import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
import re
import shutil
import sys
import warnings

# homemade libraries
# current_dir = Path(__file__).resolve()
sys.path.append(str(Path(__file__).resolve().parent.parent))
import libraries.misc.path_tools as path_tools  # noqa: E402



if __name__ == "__main__":
    '''

        Transform the block_id (that resets after every trial)
        to block_order (that creates a unique ID for each block of a neuron)
        
    '''
    # parameters
    force_processing = True  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening
    
    save_results = False

    print("Step 0: Extract the videos embedded in the selected sessions.")

    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled", "3_merged", "1_kinect_and_nerve_shandata")

    # get metadata file to get block-id to block-order
    metadata_filename =  os.path.join(path_tools.get_metadata_path(), "semicontrolled_data-collection_quality-check.xlsx")
    metadata = pd.read_excel(metadata_filename)

    # get input base directory
    db_path_input = os.path.join(db_path, "0_by-units_renamed_trial-corrected")
    # get output base directory
    db_path_output = os.path.join(db_path, "0_by-units_renamed_trial-corrected_block-corrected")

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
    
    for session in sessions:
        unit_id = session.split("_")[1]
        data_session = metadata[metadata["Unit"] == unit_id]

        file = [f.name for f in Path(db_path_input).iterdir() if session in f.name]
        if len(file) > 1:
            warnings.warn(f"Issue detected: not exactly 1 csv file found for {session}.")
            continue
        file = file[0]
        file_abs = os.path.join(db_path_input, file)
        print(f"current file: {file} and unit number: {unit_id}")
        
        # load current data
        data_unit = pd.read_csv(file_abs)
        
        # transform the block_id (that has reset after every trial) to block_order (that creates a unique ID for each block of a neuron)
        df_transition = pd.DataFrame({
            'block_id': data_session["Block_id"],
            'block_order': data_session["Block order"]
        })
        df_transition = df_transition.dropna(subset=['block_id'])
        print(df_transition)

        df_merged = data_unit.merge(df_transition, on='block_id', how='left')
        df_merged = df_merged.drop(columns=['block_id'])
        print("done.")

        # save data on the hard drive ?
        if save_results:
            output_filename_abs = os.path.join(db_path_output, file)
            # https://answers.microsoft.com/en-us/msoffice/forum/all/excel-file-open-the-file-name-is-too-long-rename/ef736fec-0bd4-42a9-806d-5b22dbfdda81#:~:text=To%20resolve%20this%20issue%2C%20you,structure%2C%20is%20still%20too%20long.
            #  Excel indicates that the total path length,
            #  including the filename and its directory structure,
            #  exceeds the Windows maximum limit of 260 characters.
            df_merged.to_csv(output_filename_abs, index=False)
        
        print("done.")




















