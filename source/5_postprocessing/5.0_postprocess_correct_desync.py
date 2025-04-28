import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import shutil
from scipy import signal
import sys
import warnings

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.materials.semicontrolled_data import SemiControlledData  # noqa: E402
from libraries.plot.semicontrolled_data_visualizer import SemiControlledDataVisualizer  # noqa: E402
import libraries.processing.semicontrolled_data_cleaning as scd_cleaning  # noqa: E402


def get_correlation(sig1, sig2, *, downsampling=0.1, show=False):
    if downsampling < 0 or downsampling > 1:
        warnings.warn("downsampling has to be a float between 0 and 1.")
        return

    # just in case, remove temporarily any nan value for correlation
    # for some reason, np.nan_to_num doesn't work.
    with pd.option_context('future.no_silent_downcasting', True):
        sig1 = pd.Series(sig1).fillna(0).values
        sig2 = pd.Series(sig2).fillna(0).values

    # normalise signals
    sig1 = scd_cleaning.normalize_signal(sig1, dtype=np.ndarray)
    sig2 = scd_cleaning.normalize_signal(sig2, dtype=np.ndarray)

    # signals can be downsampled for a faster correlation
    sig1_corr = sig1[np.linspace(0, len(sig1) - 1, int(downsampling * len(sig1)), dtype=int)]
    sig2_corr = sig2[np.linspace(0, len(sig2) - 1, int(downsampling * len(sig2)), dtype=int)]

    # remove the mean for a better estimation of the correlation
    #sig1_corr = sig1_corr - np.mean(sig1_corr)
    #sig2_corr = sig2_corr - np.mean(sig2_corr)

    # lag estimation
    correlation = signal.correlate(sig1_corr, sig2_corr, mode="full")
    lags = signal.correlation_lags(sig1_corr.size, sig2_corr.size, mode="full")
    lag = int(lags[np.argmax(correlation)] / downsampling)

    if show:
        if len(sig1_corr) > len(sig2_corr):
            x = np.linspace(0, len(sig1_corr) - 1, len(sig1_corr))
            y1 = sig1_corr
            y2 = np.pad(sig2_corr, (0, len(sig1_corr) - len(sig2_corr)), 'constant')
        else:
            x = np.linspace(0, len(sig2_corr) - 1, len(sig2_corr))
            y1 = np.pad(sig1_corr, (0, len(sig2_corr) - len(sig1_corr)), 'constant')
            y2 = sig2_corr

        # Create a figure with two subplots
        fig1, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

        # Plot the first signal on the first subplot
        ax1.plot(x, y1, label='Contact signal')
        ax1.set_title('Contact signal')
        ax1.set_ylabel('Amplitude')
        ax1.legend()

        # Plot the second signal on the second subplot
        ax2.plot(x, y2, label='Nerve signal', color='orange')
        ax2.set_title(f"Nerve signal")
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Amplitude')
        ax2.legend()

        # Set the main title using the file names
        # Adjust the layout
        plt.tight_layout()

    return lag


def find_groups_of_ones(arr):
    groups = []
    in_group = False
    start_index = 0

    for i, value in enumerate(arr):
        if value == 1 and not in_group:
            # Start of a new group
            in_group = True
            start_index = i
        elif value == 0 and in_group:
            # End of the current group
            in_group = False
            groups.append(list(range(start_index, i)))  # range is right boundary exclusive

    # If the array ends with a group of 1s
    if in_group:
        groups.append(list(range(start_index, len(arr))))

    return groups


if __name__ == "__main__":
    # parameters
    force_processing = True  # If user wants to force data processing even if results already exist
    save_results = False

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
            output_filename_abs = os.path.join(output_dir_abs, file)
            if not force_processing and os.path.exists(output_filename_abs):
                continue

            # load data
            scd = SemiControlledData(file_abs, load_instant=True)

            # interpolate the contact nan values to match neural Fs and contact Fs
            scd.contact.interpolate_missing_values(method="linear")

            neuron_iff = scd.neural.iff
            contact_depth = scd.contact.depth

            # smooth signals
            data_Fs = 1 / np.nanmean(np.diff(scd.neural.time))
            window_sec = 0.2
            window_size = int(window_sec*data_Fs)
            neuron_iff = scd_cleaning.smooth_signal(neuron_iff, window_size=window_size)
            contact_depth = scd_cleaning.smooth_signal(contact_depth, window_size=window_size)

            # lag estimation
            lag = get_correlation(contact_depth, neuron_iff, downsampling=0.1, show=show)
            if show:
                plt.show(block=True)

            print(f"lag/TTL_kinect length (ratio): {lag} / {len(contact_depth)} ({abs(lag)/len(contact_depth):.3f})")

            if abs(lag)/len(contact_depth) > .30:
                w = f"Most likely, there is a problem: lag/length signal over 30%\nReset the lag to zero."
                warnings.warn(w)
                comment = f"Lag ratio was too high ({abs(lag)/len(contact_depth):.3f}); any shift has been ignored."
                lag = 0
            else:
                comment = "Nothing to report"

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
            comment_list.append(comment)

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
