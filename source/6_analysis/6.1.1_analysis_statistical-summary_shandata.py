import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import os
import pandas as pd
from PIL import Image
import re
from scipy.ndimage import gaussian_filter1d
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


def calculate_max(d):
    return calculate(d, "max")

def calculate_mean(d):
    return calculate(d, "mean")

def calculate(d, method="mean"):
    # Remove zeros
    d = d[d != 0]
    # Check if there are any non-zero values
    if len(d):
        # Get the maximum, ignoring NaNs
        if method == "mean":
            res = np.nanmean(d)
        elif method == "max":
            res = np.nanmax(d)
        # Set res to 0 if result is NaN
        if np.isnan(res):
            res = 0
    else:
        # If no valid values, set res to 0
        res = np.nan
    return res


if __name__ == "__main__":
    # ----------------------
    # User control variables
    # ----------------------
    force_processing = True  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening

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

    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get metadata paths
    md_stimuli_path = os.path.join(db_path, "1_primary", "logs", "2_stimuli_by_blocks")
    md_neuron_filename_abs = os.path.join(db_path, "1_primary", "nerve", "semicontrol_unit-name_to_unit-type.csv")

    # get input data directories
    db_path_input = os.path.join(db_path, "3_merged", "1_kinect_and_nerve_shandata", "1_by-trials")
    # get output directories
    db_path_input_singletouch_idx = os.path.join(db_path, "3_merged", "1_kinect_and_nerve_shandata", "2_by-trials_single-touches-index")
    # get output directories
    db_path_output = os.path.join(db_path, "3_merged", "1_kinect_and_nerve_shandata", "3_global_result")
    if not os.path.exists(db_path_output) and save_results:
        os.makedirs(db_path_output)
        print(f"Directory '{db_path_output}' created.")
    # get output filenames
    filename_output = "semicontrolled_single-touch_summary-statistics.csv"

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

    stat_summary = pd.DataFrame({
        'stimulus_type': [],
        'stimulus_vel': [],
        'stimulus_size': [],
        'stimulus_force': [],
        'contact_vel': [],
        'contact_area': [],
        'contact_depth': [],
        'neuron_type': [],
        'neuron_ID': [],
        'neuron_IFF_max': [],
        'neuron_IFF_mean': []})

    for session in sessions:
        curr_dir = os.path.join(db_path_input, session)
        files_abs, files = path_tools.find_files_in_directory(curr_dir, ending=r'_trial\d{2}\.csv')

        for data_filename_abs, data_filename in zip(files_abs, files):
            print(f"current file: {data_filename}")
            # single touch endpoint results will be saved close to the .csv file.
            filename_output_abs = data_filename_abs.replace(".csv", f"_single-touch-endpoints_{split_method}_correct.txt")
            # ensure window character path limitation of 260 is ignored
            filename_output_abs = path_tools.winapi_path(filename_output_abs)
            if not force_processing and os.path.exists(filename_output_abs):
                print(f'The file {filename_output_abs} exists.', Warning)
                continue

            # 1. extract endpoints data
            endpoints_filename = data_filename.replace(".csv", f"_single-touch-endpoints_{split_method}.txt")
            endpoints_filename_abs = os.path.join(db_path_input_singletouch_idx, session, endpoints_filename)
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

            # 2. check if neuron metadata file exists
            if not os.path.exists(md_neuron_filename_abs):
                warnings.warn(f'The file {md_neuron_filename_abs} doesn''t  exist.', Warning)
                continue

            # 3. create a SemiControlledData's list of TOUCH EVENT:
            # 3.1 load the data
            scd = SemiControlledData(data_filename_abs, "", md_neuron_filename_abs)  # resources
            scd.set_variables(dropna=False)
            # Shan's storing method override trial_id defined during the experiment (remove bad trials and recount from there)
            data = pd.read_csv(data_filename_abs)
            scd.stim.type = data.stimulus[0]
            scd.stim.vel = data.vel[0]
            scd.stim.size = data.finger[0]
            scd.stim.force = data.force[0]


            for idx, (start, end) in enumerate(loaded_endpoints):
                scd_single_touch = scd.get_data_idx(np.arange(start, end))

                if show:
                    scd_visualiser = SemiControlledDataVisualizer(scd_single_touch)
                    WaitForButtonPressPopup()
                
                # velocity
                position = scd_single_touch.contact.pos_1D # mm
                if len(position):
                    position = position / 10  # cm
                    sampling_rate = scd.md.data_Fs  # Hz
                    sigma = 2  # Standard deviation for Gaussian kernel
                    smoothed_position = gaussian_filter1d(position, sigma=sigma)
                    velocity = np.gradient(smoothed_position, 1 / sampling_rate)
                    res_velocity = np.mean(np.abs(velocity))
                else:
                    res_velocity = np.nan
                # area
                res_area = calculate_max(scd_single_touch.contact.area)
                # depth
                res_depth = calculate_max(scd_single_touch.contact.depth)

                # spike IFF
                res_iff_max = calculate_max(scd_single_touch.neural.iff)
                res_iff_mean = calculate_mean(scd_single_touch.neural.iff)

                # 4.2 save generated results:
                new_row = pd.DataFrame({
                    'stimulus_type': [scd.stim.type],
                    'stimulus_vel': [scd.stim.vel],
                    'stimulus_size': [scd.stim.size],
                    'stimulus_force': [scd.stim.force],
                    'contact_vel': [res_velocity],
                    'contact_area': [res_area],
                    'contact_depth': [res_depth],
                    'neuron_type': [scd_single_touch.neural.unit_type],
                    'neuron_ID': [scd_single_touch.neural.unit_id],
                    'neuron_IFF_max': [res_iff_max],
                    'neuron_IFF_mean': [res_iff_mean]
                })
                # res_mean = res_mean.append(new_row, ignore_index=True)
                stat_summary = pd.concat([stat_summary, new_row], ignore_index=True)

    # 5. save endpoints results
    if save_results:
        if not os.path.exists(db_path_output):
            os.makedirs(db_path_output)
        filename_output_abs = os.path.join(db_path_output, filename_output)
        # ensure window character path limitation of 260 is ignored
        filename_output_abs = path_tools.winapi_path(filename_output_abs)
        stat_summary.to_csv(filename_output_abs, index=False)  # index=False prevents writing row indices to the file

    print("done.")
