import matplotlib.pyplot as plt
from matplotlib.widgets import Button
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
from libraries.plot.semicontrolled_data_visualizer import DataVisualizer3D  # noqa: E402
import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.misc.waitforbuttonpress_popup import WaitForButtonPressPopup
import libraries.plot.semicontrolled_data_visualizer_unitcontact as scdata_visualizer_neur  # noqa: E402


if __name__ == "__main__":
    # ----------------------
    # User control variables
    # ----------------------
    force_processing = True  # If user wants to force data processing even if results already exist
    show = True  # If user wants to monitor what's happening

    # choose the method used to split single touches:
    #  - method_1: Stroking trials are split with position, Taping using only IFF
    #  - method_2: Stroking trials are split with position, Taping using only depth
    #  - method_3: Stroking trials are split with position, Taping using only depth and IFF
    split_method = "method_1"

    # ----------------------
    save_results = True
    # ----------------------
    # ----------------------
    # ----------------------

    print("Step 0: Extract the videos embedded in the selected neurons.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get metadata paths
    md_stimuli_path = os.path.join(db_path, "1_primary", "logs", "2_stimuli_by_blocks")
    md_neuron_filename_abs = os.path.join(db_path, "1_primary", "nerve", "semicontrol_unit-name_to_unit-type.csv")

    # get input data directories
    db_path_input = os.path.join(db_path, "3_merged", "1_kinect_and_nerve", "2_by-trials")
    db_path_input_touch_idx = os.path.join(db_path, "3_merged", "1_kinect_and_nerve", "2_by-trials_single-touches-index")

    # get output directories
    db_path_output = os.path.join(db_path, "3_merged", "1_kinect_and_nerve", "3_by-single-touches")
    if not os.path.exists(db_path_output):
        os.makedirs(db_path_output)
        print(f"Directory '{db_path_output}' created.")

    # neuron names
    neurons_ST13 = ['2022-06-14_ST13-01',
                    '2022-06-14_ST13-02',
                     '2022-06-14_ST13-03']

    neurons_ST14 = ['2022-06-15_ST14-01',
                     '2022-06-15_ST14-02',
                     '2022-06-15_ST14-03',
                     '2022-06-15_ST14-04']

    neurons_ST15 = ['2022-06-16_ST15-01',
                     '2022-06-16_ST15-02']

    neurons_ST16 = ['2022-06-17_ST16-02',
                     '2022-06-17_ST16-03',
                     '2022-06-17_ST16-04',
                     '2022-06-17_ST16-05']

    neurons_ST18 = ['2022-06-22_ST18-01',
                     '2022-06-22_ST18-02',
                     '2022-06-22_ST18-04']
    neurons = []
    neurons = neurons + neurons_ST13
    neurons = neurons + neurons_ST14
    neurons = neurons + neurons_ST15
    neurons = neurons + neurons_ST16
    neurons = neurons + neurons_ST18
    print(neurons)

    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for neuron in neurons:
        curr_dir = os.path.join(db_path_input, neuron)
        files_abs, files = path_tools.find_files_in_directory(curr_dir, ending=r'_trial\d{2}\.csv')

        xyz = None
        for data_filename_abs, data_filename in zip(files_abs, files):
            print(f"current file: {data_filename}")

            # 1. extract endpoints data
            block_id = data_filename_abs.split("\\")[-2]
            endpoints_filename = data_filename.replace(".csv", f"_single-touch-endpoints_{split_method}_correct.txt")
            endpoints_filename_abs = os.path.join(db_path_input_touch_idx, neuron, block_id, endpoints_filename)
            # ensure window character path limitation of 260 is ignored
            endpoints_filename_abs = path_tools.winapi_path(endpoints_filename_abs)
            if os.path.exists(endpoints_filename_abs):
                # Initialize an empty list to store tuples
                loaded_endpoints = []
                # Open file and read lines
                with open(endpoints_filename_abs, 'r') as f:
                    for line in f:
                        # Remove newline character and split by commas
                        parts = line.strip().split(',')
                        # Convert parts to integers or floats as needed
                        endpoint = tuple(map(int, parts))  # Assuming integers; use float() if floats are expected
                        # Append the tuple to the list
                        loaded_endpoints.append(endpoint)
            else:
                print(f'The file {endpoints_filename_abs} does not exist.', Warning)
                continue

            # 2. extract metadata related to the current stimulus set and check if exists
            md_stimuli_filename = re.sub(r'_trial\d{2}\.csv', '_stimuli.csv', data_filename)
            md_stimuli_filename_abs = os.path.join(md_stimuli_path, neuron, md_stimuli_filename)
            if not os.path.exists(md_stimuli_filename_abs):
                warnings.warn(f'The file {md_stimuli_filename_abs} doesn''t exist.', Warning)
                continue

            # 3. check if neuron metadata file exists
            if not os.path.exists(md_neuron_filename_abs):
                warnings.warn(f'The file {md_neuron_filename_abs} doesn''t  exist.', Warning)
                continue

            # 4. create a SemiControlledData's list of TOUCH EVENT:
            # 4.1 load the data
            scd = SemiControlledData(data_filename_abs, md_stimuli_filename_abs, md_neuron_filename_abs)  # resources
            scd.set_variables(dropna=False)
            # extract chosen single touches
            indices = None
            for tpl in loaded_endpoints:
                if indices is None:
                    indices = np.arange(tpl[0], tpl[1])
                else:
                    tmp = np.arange(tpl[0], tpl[1])
                    indices = np.concatenate((indices, tmp))

            if xyz is None:
                # Initialize with the first array
                xyz = scd.contact.pos[:, indices]
            else:
                xyz = np.concatenate((xyz, scd.contact.pos[:, indices]), axis=1)

        viz = DataVisualizer3D(neuron)
        viz.update(np.arange(np.shape(xyz)[1]), xyz)
        WaitForButtonPressPopup()







    print("done.")
