import itertools
from typing import List, Any
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import os
import pandas as pd
import pyglet
import re
from scipy.ndimage import gaussian_filter
import seaborn as sns
from sklearn.decomposition import PCA
import sys
import warnings

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libraries.materials.semicontrolled_data import SemiControlledData  # noqa: E402
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
    show = True  # If user wants to monitor what's happening
    save_results = False
    # ----------------------
    # ----------------------
    # ----------------------

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get input data directory
    db_path_input = os.path.join(db_path, "3_merged", "2_kinect_and_nerve", "1_by-trials")
    # get metadata paths
    md_stimuli_path = os.path.join(db_path, "1_primary", "logs", "stimuli_by_blocks")
    md_neuron_filename_abs = os.path.join(db_path, "1_primary", "nerve", "semicontrol_unit-name_to_unit-type.csv")


    # get output directories
    output_figure_path = os.path.join(path_tools.get_result_path(), "semi-controlled", "kinect_and_nerve",
                                      "receptive_field_mapping")
    if not os.path.exists(output_figure_path):
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
    #sessions = sessions + sessions_ST13
    #sessions = sessions + sessions_ST14
    sessions = sessions + sessions_ST15
    #sessions = sessions + sessions_ST16
    #sessions = sessions + sessions_ST18
    print(sessions)

    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:

        # define filename's output path
        filename_output = session + "_RFM.png"
        filename_output_abs = os.path.join(output_figure_path, filename_output)
        # ensure window character path limitation of 260 is ignored
        filename_output_abs = path_tools.winapi_path(filename_output_abs)
        if not force_processing and os.path.exists(filename_output_abs):
            print(f'The file {filename_output_abs} exists.', Warning)
            continue

        curr_dir = os.path.join(db_path_input, session)
        data_filenames_abs, data_filenames = path_tools.find_files_in_directory(curr_dir, ending='.csv')

        scd_current_neuron: list[Any] = []
        for data_filename_abs, data_filename in zip(data_filenames_abs, data_filenames):
            print(f"current file: {data_filename}")

            # 1. extract endpoints data
            endpoints_filename_abs = data_filename_abs.replace(".csv", "_single-touch-endpoints_correct.txt")
            # ensure window character path limitation of 260 is ignored
            endpoints_filename_abs = path_tools.winapi_path(endpoints_filename_abs)
            if not os.path.exists(endpoints_filename_abs):
                print(f'The file {data_filename.replace(".csv", "_single-touch-endpoints_correct.txt")} doesn''t exist.', Warning)
                continue
            endpoints = []
            with open(endpoints_filename_abs, 'r') as f:
                for line in f:
                    # Remove newline character and split by commas
                    parts = line.strip().split(',')
                    # Convert parts to integers or floats as needed
                    endpoint = tuple(map(int, parts))  # Assuming integers; use float() if floats are expected
                    # Append the tuple to the list
                    endpoints.append(endpoint)

            # 2. extract metadata related to the current stimulus set and check if exists
            md_stimuli_filename = re.sub(r'_trial\d{2}\.csv', '_stimuli.csv', data_filename)
            md_stimuli_filename_abs = os.path.join(md_stimuli_path, session, md_stimuli_filename)
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
            # 4.2 consider only stroking motion
            if scd.stim.type == "tap":
                print(f"current trial is of type ''tap'', ignore...")
                continue
            # 4.3 split into single touches
            splitter = SemiControlledDataSplitter()
            scd_valid_touches = splitter.get_single_touches(scd, endpoints)
            scd_current_neuron.extend(scd_valid_touches)

        # 0. if no stroking motion has been detected, jump to the next neuron
        if len(scd_current_neuron) == 0:
            continue
        # 1. interpolate contact values to avoid getting nans (as 30 Hz for contact and 1000 Hz for neural data)
        for scd in scd_current_neuron:
            scd.contact.interpolate_missing_values(method="linear")

        # 2. extract 2 principal components of positions, and neuron spike timings
        XYZ = []
        spikes = []
        for idx, scd in enumerate(scd_current_neuron):
            print(f"merging progress = {idx}/{len(scd_current_neuron)}")
            XYZ.extend(scd.contact.pos.T)
            spikes.extend(scd.neural.spike)
        spikes = np.array(spikes)
        XYZ = np.array(XYZ)
        XYZ = XYZ.T  # reverse to have [XYZ, time] format

        #  2.1 preprocess a bit the signal
        XYZ_mm = XYZ / 10

        # 2.2 Initialize PCA to reduce to 2 components
        pca = PCA(n_components=2)
        XY = pca.fit_transform(XYZ_mm.T)
        XY = XY.T  # reverse to have [XY, time] format

        # 3. extract contact position when spikes occur
        XY_spikes = XY[:, spikes.astype(bool)]

        # 4. Create a 2D histogram
        x = XY_spikes[0, :]
        y = XY_spikes[1, :]
        # Define the size of the grid
        x_bins = np.linspace(min(XY[0, :]) - 0.5, max(XY[0, :]) + 1.5, 256)
        y_bins = np.linspace(min(XY[1, :]) - 0.5, max(XY[1, :]) + 1.5, 256)
        # Create a 2D histogram
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=[x_bins, y_bins])
        # smooth a bit the heatmap
        heatmap = gaussian_filter(heatmap, sigma=0)
        # Plot the heatmap using Seaborn
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap.T, cmap="YlGnBu")
        plt.xlabel('PC1 (mm)')
        plt.ylabel('PC2 (mm)')
        plt.title(f'Receptive Field Mapping (log function):\n{scd_current_neuron[0].stim.print()}')
        plt.show(block=False)

        if save_results:
            # set correct dimensions
            display = pyglet.canvas.Display()
            screen = display.get_default_screen()
            screenratio = screen.width / screen.height
            dpi = 100
            height = 1080 / dpi
            width = 1920 / dpi
            fig.set_size_inches(width, height)
            # save the figure
            fig.savefig(filename_output_abs)

        plt.close(fig)

    print("done.")
