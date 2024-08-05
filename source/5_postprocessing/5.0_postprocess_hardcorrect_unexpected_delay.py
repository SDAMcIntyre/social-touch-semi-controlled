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
    show = False  # If user wants to monitor what's happening

    # result saving parameters
    save_results = True
    generate_report = True

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get input base directory
    db_path_input = os.path.join(db_path, "3_merged", "2_kinect_and_nerve", "0_block-order")
    # get output base directory
    db_path_output = os.path.join(db_path, "3_merged", "2_kinect_and_nerve", "1_block-order_corrected-delay")
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

    lag_list = []
    ratio_list = []
    file_list = []
    comment_list = []
    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:
        curr_dir = os.path.join(db_path_input, session)
        files_abs, files = path_tools.find_files_in_directory(curr_dir, ending='_kinect_and_nerve.csv')

        output_session_abs = os.path.join(db_path_output, session)
        if not os.path.exists(output_session_abs):
            os.makedirs(output_session_abs)
            print(f"Directory '{output_session_abs}' created.")

        for file_abs, file in zip(files_abs, files):
            print(f"---------------")
            print(f"current dataset: {file}")
            print(f"current dataset: {file_abs}")

            output_filename_abs = os.path.join(output_session_abs, file)

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

            # normalise signals
            neuron_iff = scd_cleaning.normalize_signal(neuron_iff, dtype=np.ndarray)
            contact_depth = scd_cleaning.normalize_signal(contact_depth, dtype=np.ndarray)

            # if necessary, down sample for a faster correlation
            downsampling = 1
            nsample_corr = int(downsampling*len(contact_depth))
            indexes = np.linspace(0, len(contact_depth)-1, nsample_corr, dtype=int)
            neuron_iff_corr = neuron_iff[indexes]
            contact_depth_corr = contact_depth[indexes]

            # remove the mean for a better estimation of the correlation
            neuron_iff_corr = neuron_iff_corr - np.mean(neuron_iff_corr)
            contact_depth_corr = contact_depth_corr - np.mean(contact_depth_corr)

            # lag estimation
            correlation = signal.correlate(contact_depth_corr, neuron_iff_corr, mode="full")
            lags = signal.correlation_lags(contact_depth_corr.size, neuron_iff_corr.size, mode="full")
            lag = int(lags[np.argmax(correlation)] / downsampling)
            del correlation, lags

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

            # show data?
            if show:
                #viz = SemiControlledDataVisualizer(scd)
                # Create a figure and subplots
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
                s1 = scd.contact.depth
                s2 = scd.neural.iff
                s3 = scd.contact.depth
                s4 = iff_shifted
                # Plot data
                ax1.plot(s1, 'b', label='RAW (depth)')
                ax2.plot(s2, 'g', label='RAW (iff)')
                ax3.plot(s3, 'b', label='RAW (depth)')
                ax4.plot(s4, 'g', label='shifted (iff)')
                # Show legend
                ax1.legend()
                ax2.legend()
                ax3.legend()
                ax4.legend()
                main_title = f'Lag: {lag} samples'
                fig.suptitle(main_title, fontsize=16)
                # Adjust layout
                plt.tight_layout()

                # Show plot
                plt.show()

            # save data on the hard drive ?
            if save_results:
                if not force_processing:
                    try:
                        with open(output_filename_abs, 'r'):
                            print("Result file exists, jump to the next dataset.")
                            continue
                    except FileNotFoundError:
                        pass
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

