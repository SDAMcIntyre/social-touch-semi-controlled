import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys
import warnings
import time

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libraries.preprocessing.semicontrolled_data_manager import SemiControlledDataManager  # noqa: E402
import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.misc.waitforbuttonpress_popup import WaitForButtonPressPopup
from libraries.plot.semicontrolled_data_visualizer import SemiControlledDataVisualizer  # noqa: E402
import libraries.plot.semicontrolled_data_visualizer_unitcontact as scdata_visualizer_neur  # noqa: E402


if __name__ == "__main__":
    force_processing = True  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening
    save_results = True

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get input data directory
    db_path_input = os.path.join(db_path, "merged", "kinect_and_nerve", "1_by-trials")
    # get metadata paths
    md_stimuli_path = os.path.join(db_path, "primary", "logs", "stimuli_by_blocks")
    md_neuron_filename_abs = os.path.join(db_path, "primary", "nerve", "semicontrol_unit-name_to_unit-type.csv")
    # get output directory
    result_path = os.path.join(path_tools.get_result_path(), "semi-controlled")
    db_path_output = os.path.join(result_path, "kinect_and_nerve", "0_by-trials")
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
    #sessions = sessions + sessions_ST14
    #sessions = sessions + sessions_ST15
    #sessions = sessions + sessions_ST16
    #sessions = sessions + sessions_ST18
    print(sessions)

    scd_list = []
    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:
        curr_dir = os.path.join(db_path_input, session)
        files_abs, files = path_tools.find_files_in_directory(curr_dir, ending='.csv')

        output_session_abs = db_path_output

        for data_filename_abs, data_filename in zip(files_abs, files):
            # 1. extract metadata related to the current stimulus set and check if exists
            md_stimuli_filename = re.sub(r'_trial\d{2}\.csv', '_stimuli.csv', data_filename)
            md_stimuli_filename_abs = os.path.join(md_stimuli_path, session, md_stimuli_filename)
            if not os.path.exists(md_stimuli_filename_abs):
                warnings.warn(f'The file {md_stimuli_filename_abs} exists.', Warning)
                continue

            # 2. check if neuron metadata file exists
            if not os.path.exists(md_neuron_filename_abs):
                warnings.warn(f'The file {md_neuron_filename_abs} exists.', Warning)
                continue

            # 3. create a SemiControlledData's list of TOUCH EVENT:
            scdm = SemiControlledDataManager()
            data_sliced = scdm.preprocess_data_file(data_filename_abs, md_stimuli_filename_abs, md_neuron_filename_abs,
                                                    correction=True, show=False, verbose=True)

            # store the calculated data
            scd_list.append(data_sliced)


    scd_visualiser = SemiControlledDataVisualizer()
    flattened_scd_list = list(itertools.chain.from_iterable(scd_list))
    for scd in flattened_scd_list:
        scd_visualiser.update(scd)
        WaitForButtonPressPopup()

    print("done.")

























