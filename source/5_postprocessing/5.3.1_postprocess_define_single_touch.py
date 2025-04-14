import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import re
import sys
import warnings
import time

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libraries.materials.semicontrolled_data import SemiControlledData  # noqa: E402
from libraries.processing.semicontrolled_data_manager import SemiControlledDataManager  # noqa: E402
from libraries.processing.semicontrolled_data_splitter import SemiControlledDataSplitter  # noqa: E402
from libraries.plot.semicontrolled_data_visualizer import SemiControlledDataVisualizer  # noqa: E402
import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.misc.waitforbuttonpress_popup import WaitForButtonPressPopup
import libraries.plot.semicontrolled_data_visualizer_unitcontact as scdata_visualizer_neur  # noqa: E402


if __name__ == "__main__":
    # ----------------------
    # User control variables
    # ----------------------
    force_processing = True  # If user wants to force data processing even if results already exist
    save_figures = False
    save_results = False


    input_filename_pattern_r = r"semicontrolled_block-order(0[1-9]|1[0-8])_trial(0[1-9]|1[0-9]).csv"
    # choose the method to split single touches:
    #  - method_1: Stroking trials are split with position, Taping using only IFF
    #  - method_2: Stroking trials are split with position, Taping using only depth
    #  - method_3: Stroking trials are split with position, Taping using depth and IFF
    split_method = "method_2"
    output_filename_end = f"_single-touch-endpoints_{split_method}.txt"

    show = True  # If user wants to monitor what's happening
    show_single_touches = False  # If user wants to visualise single touches, one by one
    manual_check = True  # If user wants to take the time to check the trial and how it has been split


    # ----------------------
    # ----------------------
    # ----------------------

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get metadata paths
    md_stimuli_path = os.path.join(db_path, "1_primary", "logs", "2_stimuli_by_blocks")
    md_neuron_filename_abs = os.path.join(db_path, "1_primary", "nerve", "semicontrol_unit-name_to_unit-type.csv")
    # 2. check if neuron metadata file exists
    if not os.path.exists(md_neuron_filename_abs):
        s = f'The file {md_neuron_filename_abs} doesn''t  exist.'
        warnings.warn(s, Warning)

    # get input data directory
    db_path_input = os.path.join(db_path, "3_merged", "sorted_by_trial")
    # get output directories
    db_path_output = os.path.join(db_path, "3_merged", "sorted_by_trial")
    output_figure_path = os.path.join(db_path, "3_merged", "sorted_by_trial")
    if save_results and not os.path.exists(db_path_output):
        os.makedirs(db_path_output)
    if save_figures and not os.path.exists(output_figure_path):
        os.makedirs(output_figure_path)

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
    #sessions = ['2022-06-14_ST13-03']
    print(sessions)

    if show:
        scd_list = []
        viz = SemiControlledDataVisualizer()
    else:
        viz = None
    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:
        curr_dir = os.path.join(db_path_input, session)
        files_abs, files = path_tools.find_files_in_directory(curr_dir, ending=input_filename_pattern_r)

        for data_filename_abs, data_filename in zip(files_abs, files):
            print(f"current file: {data_filename}")
            # Output filenames
            output_dir_abs = os.path.dirname(data_filename_abs).replace(db_path_input, db_path_output)
            filename_output = data_filename.replace(".csv", output_filename_end)
            filename_output_abs = os.path.join(output_dir_abs, filename_output)
            filename_image_output = filename_output.replace(".txt", ".png")
            filename_image_output_abs = os.path.join(output_dir_abs, filename_image_output)
            if not force_processing and os.path.exists(filename_output_abs):
                s = f'The file {filename_output_abs} exists.'
                print(s, Warning)
                continue

            # 1. extract metadata related to the current stimulus set and check if exists
            md_stimuli_filename = re.sub(r'_trial\d{2}\.csv', '_stimuli.csv', data_filename)
            md_stimuli_filename_abs = os.path.join(md_stimuli_path, session, md_stimuli_filename)
            if not os.path.exists(md_stimuli_filename_abs):
                s = f'The file {md_stimuli_filename_abs} doesn''t exist.'
                warnings.warn(s, Warning)
                continue

            # 2. create a SemiControlledData's list of TOUCH EVENT:
            # 2.1 load the data
            scd = SemiControlledData(data_filename_abs, md_stimuli_filename_abs, md_neuron_filename_abs)  # resources
            scd.set_variables(dropna=False)
            # 2.2 split by touch event
            splitter = SemiControlledDataSplitter(viz=viz,
                                                  show=show,
                                                  show_single_touches=show_single_touches,
                                                  manual_check=manual_check,
                                                  save_visualiser=save_figures,
                                                  save_visualiser_fname=filename_image_output_abs)  # tools
            data_sliced, endpoints = splitter.split_by_touch_event(scd, method=split_method, correction=True)
            
            # 3. store the calculated data
            if show:
                scd_list.append(data_sliced)

            # 4. save endpoints results
            if save_results:
                if not os.path.exists(output_dir_abs):
                    os.mkdir(output_dir_abs)
                # Open file for writing
                with open(filename_output_abs, 'w') as f:
                    for endpoint in endpoints:
                        # Convert tuple to string format if needed
                        endpoint_str = ','.join(map(str, endpoint))
                        # Write each tuple on a new line
                        f.write(endpoint_str + '\n')


    if show:
        flattened_scd_list = list(itertools.chain.from_iterable(scd_list))
        scd_visualiser = SemiControlledDataVisualizer()
        for scd in flattened_scd_list:
            scd_visualiser.update(scd)
            WaitForButtonPressPopup()

    print("done.")
