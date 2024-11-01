import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import re
from sklearn.decomposition import PCA
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
from libraries.misc.interpolate_nan_values import interpolate_nan_values  # noqa: E402

if __name__ == "__main__":
    # ----------------------
    # User control variables
    # ----------------------
    force_processing = True  # If user wants to force data processing even if results already exist
    show = True  # If user wants to monitor what's happening

    # ----------------------
    save_results = False
    # ----------------------
    # ----------------------
    # ----------------------

    print("Step 0: Extract the videos embedded in the selected neurons.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get metadata paths
    md_stimuli_path = os.path.join(db_path, "1_primary", "logs", "2_stimuli_by_blocks")
    md_neuron_filename_abs = os.path.join(db_path, "1_primary", "nerve", "semicontrol_unit-name_to_unit-type.csv")

    # get input data directory
    db_path_input = os.path.join(db_path, "3_merged", "1_kinect_and_nerve", "2_by-trials")
    # get output directories
    db_path_output = os.path.join(db_path, "3_merged", "1_kinect_and_nerve", "2_by-trials_XYZ-normal")
    if save_results and not os.path.exists(db_path_output):
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
    for neuron_id in neurons:
        curr_dir = os.path.join(db_path_input, neuron_id)
        files_abs, files = path_tools.find_files_in_directory(curr_dir, ending=r'_trial\d{2}\.csv')

        # I. get the list of scd
        scd_list = []
        for data_filename_abs, data_filename in zip(files_abs, files):
            print(f"current file: {data_filename}")

            # 1. output filename
            filename_output_abs = data_filename_abs  # data_filename_abs.replace(".csv", f"_XYZ-normal.csv")
            if not force_processing and os.path.exists(filename_output_abs):
                s = f'The file {filename_output_abs} exists.'
                print(s, Warning)
                continue

            # 1. extract metadata related to the current stimulus set and check if exists
            md_stimuli_filename = re.sub(r'_trial\d{2}\.csv', '_stimuli.csv', data_filename)
            md_stimuli_filename_abs = os.path.join(md_stimuli_path, neuron_id, md_stimuli_filename)
            if not os.path.exists(md_stimuli_filename_abs):
                s = f'The file {md_stimuli_filename_abs} doesn''t exist.'
                warnings.warn(s, Warning)
                continue

            # 2. create a SemiControlledData's list of TOUCH EVENT:
            # 2.1 load the data
            scd = SemiControlledData(data_filename_abs, md_stimuli_filename_abs, md_neuron_filename_abs)  # resources'rfv
            scd.set_variables(dropna=False)
            scd.contact.interpolate_missing_values()
            scd_list.append(scd)

        # remove the global mean to center the data
        xyz = None
        for scd in scd_list:
            if xyz is None:
                xyz = scd.contact.pos
            else:
                xyz = np.concatenate((xyz, scd.contact.pos), axis=1)
        xyz_mean = [np.mean(xyz[0]), np.mean(xyz[1]), np.mean(xyz[2])]
        for scd in scd_list:
            scd.contact.pos[0] = scd.contact.pos[0] - xyz_mean[0]
            scd.contact.pos[1] = scd.contact.pos[1] - xyz_mean[1]
            scd.contact.pos[2] = scd.contact.pos[2] - xyz_mean[2]

        # save the initial scds
        scd_list_raw = []
        for scd in scd_list:
            scd_copy = copy.deepcopy(scd)
            scd_list_raw.append(scd_copy)

        # II. Get data train for PC for X-axis
        #     clean start and end of the trials to keep only meaningful XYZ data and merge them
        stroke_pos = None
        for scd in scd_list:
            # first PCA is on the stroke dataset to extract X
            if scd.stim.type != "stroke":
                continue

            # chunk the signal to get only good content
            TTL = scd.neural.TTL
            TTL_on_indices = np.where(TTL == 1)[0]
            # Get the first and last index
            first_index = TTL_on_indices[0]
            last_index = TTL_on_indices[-1]
            # define the safe zone
            trial_length = last_index - first_index
            safe_margin = int(trial_length * .25)
            # set the safe data to use
            p = scd.contact.pos[:, np.arange(first_index+safe_margin, last_index-safe_margin)]
            # keep the data of interest
            if stroke_pos is None:
                stroke_pos = p  # Initialize with the first array
            else:
                stroke_pos = np.concatenate((stroke_pos, p), axis=1)

        # III. fit PCA over 3 dimensions
        # transpose the position array for PCA
        if np.shape(stroke_pos)[1] != 3:
            stroke_pos = np.transpose(stroke_pos)
        if np.isnan(stroke_pos).any():
            nsamples, _ = np.shape(stroke_pos)
            # detrend the axis
            M = np.nanmean(stroke_pos, axis=0)
            pos3D_centered = stroke_pos - np.matlib.repmat(M, nsamples, 1)
            # interpolate the nan values
            stroke_pos = interpolate_nan_values(pos3D_centered)
        # get the first PCA
        pca = PCA(n_components=3)
        # Main PC is X dimension
        pca.fit(stroke_pos)

        # IV. Apply PCA
        for scd in scd_list:
            # transpose the position array for PCA
            if np.shape(scd.contact.pos)[1] != 3:
                pos = np.transpose(scd.contact.pos)
            else:
                pos = scd.contact.pos
            # First PC is X dimension
            xyz = pca.transform(pos)
            # transpose the position array for PCA
            if np.shape(xyz)[1] == 3:
                xyz = np.transpose(xyz)
            scd.contact.pos = xyz



        idx_stroke = []
        for idx, scd in enumerate(scd_list):
            # first PCA is on the stroke dataset to extract X
            if scd.stim.type == "stroke":
                idx_stroke.append(idx)



        idx = idx_stroke[0]
        pca = PCA(n_components=3)
        pos_raw = np.transpose(scd_list_raw[idx].contact.pos)
        pos_raw[:, 0] = pos_raw[:, 0] - np.mean(pos_raw[:, 0])
        pos_raw[:, 1] = pos_raw[:, 1] - np.mean(pos_raw[:, 1])
        pos_raw[:, 2] = pos_raw[:, 2] - np.mean(pos_raw[:, 2])

        plt.figure()
        plt.plot(pos_raw)
        plt.title("raw")
        plt.legend(['X', 'Y', 'Z'])
        plt.figure()
        plt.plot(pca.fit_transform(pos_raw))
        plt.title("modified")
        plt.legend(['X', 'Y', 'Z'])




        pos = np.transpose(scd_list[idx].contact.pos)






        if show:
            viz = SemiControlledDataVisualizer(title="Modified")
            viz_raw = SemiControlledDataVisualizer(title="Raw")

            for idx in np.arange(len(scd_list)):

                scd = scd_list[idx]
                scd_raw = scd_list_raw[idx]

                # plt.figure(); plt.plot(np.transpose(scd_raw.contact.pos))
                # first PCA is on the stroke dataset to extract X
                if scd.stim.type != "stroke":
                    continue
                viz.update(scd)
                viz_raw.update(scd_raw)
                WaitForButtonPressPopup()

        # V. Get data train for PC for Z-axis
        #    clean start and end of the trials to keep only meaningful XYZ data and merge them
        tap_pos = None
        for scd in scd_list:
            # first PCA is on the stroke dataset to extract X
            if scd.stim.type != "tap":
                continue

            # chunk the signal to get only good content
            TTL = scd.neural.TTL
            TTL_on_indices = np.where(TTL == 1)[0]
            # Get the first and last index
            first_index = TTL_on_indices[0]
            last_index = TTL_on_indices[-1]
            # define the safe zone
            trial_length = last_index - first_index
            safe_margin = int(trial_length * .25)
            # set the safe data to use
            p = scd.contact.pos[:, np.arange(first_index + safe_margin, last_index - safe_margin)]
            # keep the data of interest
            if tap_pos is None:
                tap_pos = p  # Initialize with the first array
            else:
                tap_pos = np.concatenate((tap_pos, p), axis=1)

        # ignore X-axis
        tap_pos = tap_pos[1:, :]

        # VI. fit PCA over 2 dimensions
        # transpose the position array for PCA
        if np.shape(tap_pos)[1] != 2:
            tap_pos = np.transpose(tap_pos)
        if np.isnan(tap_pos).any():
            nsamples, _ = np.shape(tap_pos)
            # detrend the axis
            M = np.nanmean(tap_pos, axis=0)
            pos3D_centered = tap_pos - np.matlib.repmat(M, nsamples, 1)
            # interpolate the nan values
            tap_pos = interpolate_nan_values(pos3D_centered)
        # get the first PCA
        pca = PCA(n_components=2)
        # Main PC is X dimension
        pca.fit(tap_pos)

        # VII. Apply PCA over Y and Z
        for scd in scd_list:
            # transpose the position array for PCA
            if np.shape(scd.contact.pos)[1] != 3:
                pos = np.transpose(scd.contact.pos)
            else:
                pos = scd.contact.pos
            # First PC is X dimension
            zy = pca.transform(pos[:, 1:])
            scd.contact.pos[1, :] = zy[:, 1]
            scd.contact.pos[2, :] = zy[:, 0]

        if show:
            viz = SemiControlledDataVisualizer()
            for scd in scd_list:
                viz.update(scd)
                WaitForButtonPressPopup()

        # VIII. save endpoints results
        if save_results:
            for scd in scd_list:
                # 1. output filename
                chunks = scd.md.data_filename.split("\\")
                filename_output = chunks[-1]
                session = chunks[-2]
                filename_output_abs = os.path.join(db_path_output, session, filename_output)
                if not force_processing and os.path.exists(filename_output_abs):
                    s = f'The file {filename_output_abs} exists.'
                    print(s, Warning)
                    continue

                df = scd.asDataFrame()
                df.save()
    print("done.")
