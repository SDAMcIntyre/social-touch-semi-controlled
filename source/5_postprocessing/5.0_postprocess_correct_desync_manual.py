import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import shutil
from scipy import signal
import tkinter as tk
import sys
import warnings

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.materials.semicontrolled_data import SemiControlledData  # noqa: E402
from libraries.plot.semicontrolled_data_visualizer import SemiControlledDataVisualizer  # noqa: E402
import libraries.processing.semicontrolled_data_cleaning as scd_cleaning  # noqa: E402
from libraries.processing.semicontrolled_data_correct_lag_manual import TimeSeriesLagGUI



if __name__ == "__main__":
    # parameters
    force_processing = True  # If user wants to force data processing even if results already exist
    save_results = True

    # result saving parameters
    generate_report = True

    show = True  # If user wants to monitor what's happening

    input_ending = '.csv'
    output_ending = '_manual-resync.csv'

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get input base directory
    db_path_input = os.path.join(db_path, "3_merged", "sorted_by_block")
    # get output base directory
    db_path_output = os.path.join(db_path, "3_merged", "sorted_by_block")
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
    print(sessions)

    lag_list = []
    ratio_list = []
    file_list = []
    comment_list = []
    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:
        curr_dir = os.path.join(db_path_input, session)
        files_abs, files = path_tools.find_files_in_directory(curr_dir, ending=input_ending)

        for file_abs, file in zip(files_abs, files):
            print(f"---------------")
            print(f"current dataset: {file}")
            print(f"current dataset: {file_abs}")
            
            output_dir_abs = os.path.join(db_path_output, session)
            output_filename = file.replace(input_ending, output_ending)
            output_filename_abs = os.path.join(output_dir_abs, output_filename)
            if not force_processing and os.path.exists(output_filename_abs):
                continue

            # load data
            scd = SemiControlledData(file_abs, load_instant=True)
            scd.contact.update_pos_1D(pca_range=(int((1/4)*scd.contact.nsample), int((3/4)*scd.contact.nsample)))

            # interpolate the contact nan values to match neural Fs and contact Fs
            scd.contact.interpolate_missing_values(method="linear")

            neuron_iff = scd.neural.iff
            contact_depth = scd.contact.depth
            contact_hand_pos = scd.contact.pos_1D
            contact_hand_pos[100<np.abs(contact_hand_pos)] = 0

            # smooth signals
            data_Fs = 1 / np.nanmean(np.diff(scd.neural.time))
            window_sec = 0.2
            window_size = int(window_sec*data_Fs)
            neuron_iff = scd_cleaning.smooth_signal(neuron_iff, window_size=window_size)
            contact_depth = scd_cleaning.smooth_signal(contact_depth, window_size=window_size)
            

            # Create the main Tkinter window (it needs to exist for Toplevel)
            root_app = tk.Tk()
            root_app.withdraw() # Hide the main root window
            # Launch the GUI
            ref_signals = [contact_depth, contact_hand_pos]
            ref_labels = ["contact Depth", "1D hand position"]
            gui = TimeSeriesLagGUI(ref_signals, neuron_iff,
                                ref_signal_labels=ref_labels,
                                shift_signal_label="neuron IFF") # Optional: start with an initial lag guess
            # --- Execution resumes here after the GUI window is closed ---
            print("GUI closed.")
            lag, lag_seconds = gui.get_lag()
            if lag is not None:
                print(f"The selected lag from the GUI is: {lag} samples")
            else:
                print("GUI was closed without saving the lag.")
            # Explicitly destroy the hidden root window when done
            root_app.destroy()
            print("Script finished.")
            
            if lag is None:
                lag = 0
            # align signals by shifting nerve data
            if lag > 0:
                zeros = np.zeros(lag)
                # to the right
                spike_shifted = np.concatenate((zeros, scd.neural.spike[:-lag]))
                iff_shifted = np.concatenate((zeros, scd.neural.iff[:-lag]))
            elif lag < 0:
                lag_abs = abs(lag)
                zeros = np.zeros(lag_abs)
                # to the left
                spike_shifted = np.concatenate((scd.neural.spike[lag_abs:], zeros))
                iff_shifted = np.concatenate((scd.neural.iff[lag_abs:], zeros))
            else:
                # Do nothing
                spike_shifted = scd.neural.spike
                iff_shifted = scd.neural.iff
            
            # create the shifted dataset
            df_output = pd.read_csv(file_abs)
            df_output['Nerve_spike'] = spike_shifted
            df_output['Nerve_freq'] = iff_shifted

            # keep the lag for traceability file
            lag_list.append(lag)
            ratio_list.append(abs(lag) / len(contact_depth))
            file_list.append(file_list)

            # save data on the hard drive ?
            if save_results:
                if not os.path.exists(output_dir_abs):
                    os.makedirs(output_dir_abs)
                    print(f"Directory '{output_dir_abs}' created.")
                df_output.to_csv(output_filename_abs, index=False)

    if generate_report:
        report_filename = os.path.join(db_path_output, "corrected_lag_report.csv")
        report_data = []
        for filename, lag, ratio, comment in zip(file_list, lag_list, ratio_list, comment_list):
            report_data.append({"filename": filename, "lag_sample": lag, "ratio": ratio, "comment": comment})

        with open(report_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["filename", "lag_sample", "ratio", "comment"])
            writer.writeheader()
            for row in report_data:
                writer.writerow(row)

    print("done.")
