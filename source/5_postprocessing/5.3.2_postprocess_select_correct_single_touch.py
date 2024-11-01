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
import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.misc.waitforbuttonpress_popup import WaitForButtonPressPopup
import libraries.plot.semicontrolled_data_visualizer_unitcontact as scdata_visualizer_neur  # noqa: E402


def select_signal_chunks(signals, split_indices):
    """
    Function to interactively select chunks from signals, handling NaN values.

    Parameters:
    - signals (list of np.ndarray): List of 1D signals. Signals may contain NaN values.
    - split_indices (list of tuples): List of tuples specifying chunks.
                                     Each tuple: (signal_index, start_index, end_index).

    Returns:
    - list of tuples: Selected chunks in the format (signal_index, start_index, end_index).
    """

    # Function to remove NaNs and synchronize arrays
    def remove_nans_and_sync(signal, associated_array):
        valid = ~np.isnan(signal)
        return signal[valid], associated_array[valid]

    # Store selected chunks
    selected_chunks = split_indices

    # Plot the signals and chunks
    fig, axes = plt.subplots(len(signals), 1, figsize=(15, 10))
    for i, signal in enumerate(signals):
        clean_signal, clean_time = remove_nans_and_sync(signal,
                                                        np.arange(len(signal)))  # Remove NaNs and synchronize time
        axes[i].plot(clean_time, clean_signal, label=f'Signal {i}')
        axes[i].legend()

    # Highlight chunks on the first signal only
    chunk_areas = {}
    for idx, (start, end) in enumerate(split_indices):
        chunk_signal = signals[0][start:end]
        chunk_time = np.arange(start, end)
        _, clean_chunk_time = remove_nans_and_sync(chunk_signal, chunk_time)
        if idx % 2 == 0:
            color = 'red'
        else:
            color = 'green'
        chunk = axes[0].axvspan(clean_chunk_time[0], clean_chunk_time[-1], color=color, alpha=0.8)
        chunk_areas[chunk] = (start, end)

    # Button event handling
    def on_chunk_click(event):
        if event.inaxes == axes[0]:  # Only respond to clicks in the first subplot
            for chunk, (start, end) in chunk_areas.items():
                if chunk.contains(event)[0]:
                    if (start, end) in selected_chunks:
                        selected_chunks.remove((start, end))
                        chunk.set_alpha(0.5)
                    else:
                        selected_chunks.append((start, end))
                        chunk.set_alpha(0.8)
                    fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_chunk_click)

    # OK button functionality
    def on_ok_button_clicked(event):
        plt.close()

    # Add OK button
    ok_ax = plt.axes([0.85, 0.01, 0.1, 0.05])
    ok_button = Button(ok_ax, 'OK')
    ok_button.on_clicked(on_ok_button_clicked)

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

    plt.show(block=True)

    # Return selected chunks after closing the plot
    return selected_chunks


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

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get metadata paths
    md_stimuli_path = os.path.join(db_path, "1_primary", "logs", "2_stimuli_by_blocks")
    md_neuron_filename_abs = os.path.join(db_path, "1_primary", "nerve", "semicontrol_unit-name_to_unit-type.csv")

    # get input data directories
    db_path_input = os.path.join(db_path, "3_merged", "1_kinect_and_nerve", "2_by-trials")
    # get output directories
    db_path_output = os.path.join(db_path, "3_merged", "1_kinect_and_nerve", "3_by-single-touches")
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
    print(sessions)

    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
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
            endpoints_filename_abs = data_filename_abs.replace(".csv", f"_single-touch-endpoints_{split_method}.txt")
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
            # 4.2 create a list of signals to display:
            signals = []
            signals.append(scd.contact.pos_1D)
            signals.append(scd.contact.depth)
            signals.append(scd.contact.area)
            signals.append(scd.neural.iff)
            signals.append(scd.neural.spike)

            # 5. make a decision
            selected = select_signal_chunks(signals, loaded_endpoints)
            for tpl in selected:
                print(tpl)

            # 5. save endpoints results
            if save_results:
                # Open file for writing
                with open(filename_output_abs, 'w') as f:
                    for endpoint in selected:
                        # Convert tuple to string format if needed
                        endpoint_str = ','.join(map(str, endpoint))
                        # Write each tuple on a new line
                        f.write(endpoint_str + '\n')

    print("done.")
