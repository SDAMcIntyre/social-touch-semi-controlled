import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys
from scipy import signal
import warnings

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libraries.misc.path_tools as path_tools  # noqa: E402


def get_correlation(sig1, sig2, downsampling=0.1, show=False):
    if downsampling < 0 or downsampling > 1:
        warnings.warn("downsampling has to be a float between 0 and 1.")
        return

    # just in case, remove temporarily any nan value for correlation
    # for some reason, np.nan_to_num doesn't work.
    with pd.option_context('future.no_silent_downcasting', True):
        sig1 = pd.Series(sig1).fillna(0).values
        sig2 = pd.Series(sig2).fillna(0).values

    # signals can be downsampled for a faster correlation
    sig1_corr = sig1[np.linspace(0, len(sig1) - 1, int(downsampling * len(sig1)), dtype=int)]
    sig2_corr = sig2[np.linspace(0, len(sig2) - 1, int(downsampling * len(sig2)), dtype=int)]

    # remove the mean for a better estimation of the correlation
    sig1_corr = sig1_corr - np.mean(sig1_corr)
    sig2_corr = sig2_corr - np.mean(sig2_corr)

    # lag estimation
    correlation = signal.correlate(sig1_corr, sig2_corr, mode="full")
    lags = signal.correlation_lags(sig1_corr.size, sig2_corr.size, mode="full")
    lag = int(lags[np.argmax(correlation)] / downsampling)

    # --- Normalize the Peak Correlation ---
    # 1. Find the absolute peak of the correlation
    peak_corr_value = np.max(np.abs(correlation))
    peak_lag_index = np.argmax(np.abs(correlation))
    peak_lag = lags[peak_lag_index]

    # 2. Calculate the energy (sum of squares) of each signal
    # This is equivalent to the zero-lag autocorrelation value
    energy1 = np.sum(sig1_corr**2)
    energy2 = np.sum(sig2_corr**2)

    # 3. Normalize the peak value
    # Avoid division by zero if either signal has zero energy
    if energy1 > 1e-10 and energy2 > 1e-10: # Use a small threshold
        normalized_peak = peak_corr_value / np.sqrt(energy1 * energy2)
    else:
        normalized_peak = 0.0 # Or handle as appropriate

    # --- Convert to Percentage ---
    # This normalized_peak is now typically between 0 and 1 (since we took abs).
    # If you didn't take abs(), it would be between -1 and 1.
    # You can directly interpret this as a match score, or multiply by 100.

    percentage_match = normalized_peak * 100

    print(f"Raw correlation array length: {len(correlation)}")
    print(f"Maximum absolute raw correlation: {peak_corr_value:.4f}")
    print(f"Lag at maximum correlation: {peak_lag}")
    print(f"Energy of Signal 1: {energy1:.4f}")
    print(f"Energy of Signal 2: {energy2:.4f}")
    print(f"Normalized Peak Correlation Score: {normalized_peak:.4f}")
    print(f"Percentage Match (based on normalized peak): {percentage_match:.2f}%")

    if show:
        plt.figure(figsize=(10, 6))

        plt.subplot(3, 1, 1)
        plt.plot(sig1_corr)
        plt.title('Signal 1')
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(sig2_corr, color='orange')
        plt.title('Signal 2', color='orange')
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(lags, correlation)
        plt.axvline(peak_lag, color='r', linestyle='--', label=f'Peak Lag ({peak_lag})')
        plt.title(f'Cross-correlation (Raw) - Max Abs: {peak_corr_value:.2f}')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    return percentage_match, lag


if __name__ == "__main__":

    force_processing = True  # If user wants to force data processing even if results already exist
    save_results = True
    save_figure = True
    generate_report = True
    
    show = False  # If user wants to monitor what's happening
    show_correlation_result = False

    kinect_input_filename_ending = '_kinect.csv'
    nerve_input_filename_ending = '_nerve.csv'
    output_filename_ending = ".csv"
    scaling_nofilling = True

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get input base directory
    db_path_input_kinect = os.path.join(db_path, "2_processed", "kinect")
    db_path_input_nerve = os.path.join(db_path, "2_processed", "nerve", "3_cond-velocity-adj")
    # set output base directory
    db_path_output = os.path.join(db_path, "3_merged", "sorted_by_block")
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
    
    success_list = []
    alignment_technique_list = []
    lag_list = []
    match_score_list = []
    ratio_kinect_list = []
    ratio_nerve_list = []
    file_kinect_list = []
    file_nerve_list = []
    comment_list = []
    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:
        curr_kinect_dir = os.path.join(db_path_input_kinect, session)
        curr_nerve_dir = os.path.join(db_path_input_nerve, session)

        files_contact_abs, files_contact = path_tools.find_files_in_directory(curr_kinect_dir, ending=kinect_input_filename_ending)

        for file_contact_abs, file_contact in zip(files_contact_abs, files_contact):
            print(f"current file: {file_contact}")
            #f not ('block-order08' in file_contact_abs): continue
            #if 2 != int(re.search("block-order\d{2}", file_contact).group().replace("block-order", "")): continue
            # check if led and nerve files exist for this contact file
            input_dirname = os.path.dirname(file_contact_abs)
            file_nerve = file_contact.replace(kinect_input_filename_ending, nerve_input_filename_ending)
            file_nerve_abs = os.path.join(curr_nerve_dir, file_nerve)
            if not os.path.exists(file_nerve_abs):
                print(f"File '{file_nerve_abs}'  file does not exist. Move to next file...")
                continue

            # check if already exist, if not forced to process
            output_filename = file_contact.replace(kinect_input_filename_ending, output_filename_ending)
            output_dir_abs = os.path.join(db_path_output, session)
            output_filename_abs = os.path.join(output_dir_abs, output_filename)
            if not force_processing and os.path.exists(output_filename_abs):
                print(f"File '{file_contact}' already exists and the processing is not forced. Move to next file...")
                continue
            if save_results or save_figure:
                if not os.path.exists(output_dir_abs):
                    os.makedirs(output_dir_abs)
                    print(f"Directory '{output_dir_abs}' created.")

            # load current data
            kinect = pd.read_csv(file_contact_abs)
            nerve = pd.read_csv(file_nerve_abs)

            # 0. scale up kinect data to match nerve sampling rate
            nerve_Fs = 1/np.mean(np.diff(nerve.Sec_FromStart))
            kinect_Fs = 1/np.mean(np.diff(kinect.time))
            estimated_scale = nerve_Fs / kinect_Fs
            nsample_scaled = int(estimated_scale * len(kinect))
            # create the scaled dataframe
            kinect_scaled = pd.DataFrame(columns=kinect.columns)
            kinect_scaled = kinect_scaled.reindex(np.linspace(0, nsample_scaled-1, nsample_scaled, dtype=int))
            # put each row to its expected location
            for old_index, row in kinect.iterrows():
                kinect_scaled.loc[int(old_index * estimated_scale)] = row

            # 0.1 Fill nan values generated by scaling with previous valid value
            with pd.option_context('future.no_silent_downcasting', True):
                # fill only the LED to do the correlation
                if scaling_nofilling:
                    kinect_scaled["LED on"] = kinect_scaled["LED on"].ffill()
                else:
                    kinect_scaled = kinect_scaled.ffill()

            # 1. synchronise kinect and nerve data using TTL signal (nerve_Volt + kinect_LED)
            TTL_kinect = kinect_scaled["LED on"].values
            TTL_nerve = nerve["TTL_Aut"].values
            match_score, lag = get_correlation(TTL_kinect, TTL_nerve, downsampling=0.1, show=show_correlation_result)
            print(f"lag/TTL_kinect length (ratio): {lag} / {len(TTL_kinect)} ({abs(lag)/len(TTL_kinect):.3f})")

            if abs(lag)/len(TTL_kinect) > .30:
                w = f"\nMost likely, there is a problem: lag/length signal over 30%\nTry with Contact Depth and Neuron IFF instead."
                warnings.warn(w)
                comment = f"Lag ratio was too high ({abs(lag) / len(TTL_kinect):.3f}); Use Contact Depth and Neuron IFF instead."
                with pd.option_context('future.no_silent_downcasting', True):
                    k = kinect_scaled["contact_area"].ffill().values
                n = nerve["Freq"].values
                match_score, lag = get_correlation(k, n, downsampling=0.1)

                print(f"lag/Contact_Area length (ratio): {lag} / {len(TTL_kinect)} ({abs(lag) / len(TTL_kinect):.3f})")
                if abs(lag) / len(TTL_kinect) > .30:
                    w = f"\nMost likely, there is a problem: lag/length signal over 30%\nReset lag to 0 and ignore sync."
                    comment = f"Lag ratio was too high ({abs(lag) / len(TTL_kinect):.3f}); any shift has been ignored."
                    warnings.warn(w)
                    lag = 0
                    success = False
                    alignment_technique = 'none'
                else:
                    success = True
                    alignment_technique = 'IFF and contact area'
            else:
                alignment_technique = 'TTLs'
                comment = "Nothing to report"
                success = True

            # align signals by shifting nerve data
            if lag >= 0:
                # to the right
                zeros = np.zeros(lag)
                # Concatenate the zeros and the original column
                nerve_shifted = np.concatenate((zeros, nerve["Nervespike1"].values))
                freq_shifted = np.concatenate((zeros, nerve["Freq"].values))
                TTL_Aut = np.concatenate((zeros, nerve["TTL_Aut"].values))
                nerve_t = np.concatenate((zeros, nerve["Sec_FromStart"].values))
            else:
                # to the left
                nerve_shifted = nerve["Nervespike1"].iloc[abs(lag):].values
                freq_shifted = nerve["Freq"].iloc[abs(lag):].values
                TTL_Aut = nerve["TTL_Aut"].iloc[abs(lag):].values
                nerve_t = nerve["Sec_FromStart"].iloc[abs(lag):].values

            # if necessary, modify the nerve signal to match kinect data length
            nsample_kinect = len(kinect_scaled)
            nsample_nerve = len(nerve_shifted)
            if nsample_nerve > nsample_kinect:
                nerve_shifted = nerve_shifted[:nsample_kinect]
                freq_shifted = freq_shifted[:nsample_kinect]
                TTL_Aut = TTL_Aut[:nsample_kinect]
                nerve_t = nerve_t[:nsample_kinect]
            elif nsample_nerve < nsample_kinect:
                # add nan values rather than 0 to avoid adding non-real data
                nan_vector = np.full(nsample_kinect-nsample_nerve, np.nan)
                nerve_shifted = np.concatenate((nerve_shifted, nan_vector))
                freq_shifted = np.concatenate((freq_shifted, nan_vector))
                TTL_Aut = np.concatenate((TTL_Aut, nan_vector))
                nerve_t = np.concatenate((nerve_t, nan_vector))

            # create output dataframe
            kinect_scaled.rename(columns={'t': 't_Kinect'})
            df_output = kinect_scaled.copy()
            df_output["Nerve_spike"] = nerve_shifted
            df_output["Nerve_freq"] = freq_shifted
            df_output["Nerve_TTL"] = TTL_Aut
            df_output["time"] = nerve_t

            # keep the lag for traceability file
            alignment_technique_list.append(alignment_technique)
            success_list.append(success)
            lag_list.append(lag)
            match_score_list.append(match_score)
            ratio_kinect_list.append(abs(lag) / len(TTL_kinect))
            ratio_nerve_list.append(abs(lag) / len(TTL_nerve))
            file_kinect_list.append(file_contact)
            file_nerve_list.append(file_nerve)
            comment_list.append(comment)

            if show or save_figure:
                # Create a figure with two subplots
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(10, 6))

                x = df_output["time"].values
                sig1 = df_output["Nerve_TTL"].values
                sig2 = df_output["LED on"].values

                # Plot the first signal on the first subplot
                ax1.plot(x, sig1, label='Nerve_TTL signal')
                ax1.set_title('Nerve_TTL signal')
                ax1.set_ylabel('Amplitude')
                ax1.legend()

                # Plot the second signal on the second subplot
                ax2.plot(x, sig2, label='Kinect_TTL signal', color='orange')
                ax2.set_title(f"Kinect_TTL signal")
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Amplitude')
                ax2.legend()

                # Plot the second signal on the second subplot
                ax3.plot(x, sig1, label='Nerve_TTL')
                ax3.plot(x, sig2, label='TTL_kinect_rescale', alpha=0.6, linestyle='--')
                ax3.set_title("Imperposed")
                ax3.set_xlabel('Time')
                ax3.set_ylabel('Amplitude')
                ax3.legend()

                # Plot the second signal on the second subplot
                ax4.plot(x, sig1 - sig2, label='sig1 - sig2', color='red')
                ax4.set_title("Difference")
                ax4.set_xlabel('Time')
                ax4.set_ylabel('Amplitude')
                ax4.legend()

                # Set the main title using the file names
                main_title = f'Contact File: {file_contact}\n\nNerve File: {file_nerve}'
                fig.suptitle(main_title, fontsize=16)
                # Adjust the layout
                plt.tight_layout()
                
                if save_figure:
                    plt.savefig(output_filename_abs.replace(output_filename_ending, "_TTLs_alignement.png"))
                if show:
                    plt.show(block=True)
                plt.close('all')

            # save data on the hard drive ?
            if save_results:    
                if not os.path.exists(output_dir_abs):
                    os.makedirs(output_dir_abs)
                    print(f"Directory '{output_dir_abs}' created.")
                df_output.to_csv(path_tools.winapi_path(output_filename_abs), index=False)

            print("done.")

    if generate_report:
        report_filename = os.path.join(db_path_output, "shift_lag_report.csv")
        report_data = []
        # Define the headers/column names for the CSV report.
        # Using a constant makes it easier to keep dictionary keys and CSV headers consistent.
        REPORT_FIELDNAMES = [
            "Kinect Filename",
            "Nerve Filename",
            "Success",
            "Lag (sample)",
            "Match score (%)", # Added missing field
            "Ratio sample/kinect length",
            "Ratio sample/neuron length",
            "Alignment Technique",
            "Comments"
        ]
        # Iterate through all the related data lists simultaneously
        zipped_data = zip(
            file_kinect_list,
            file_nerve_list,
            success_list,
            lag_list,
            match_score_list,
            ratio_kinect_list,
            ratio_nerve_list,
            alignment_technique_list,
            comment_list
        )
        for (kinect_file, nerve_file, success_status, lag_value, score,
            kinect_ratio, neuron_ratio, align_tech, comment_text) in zipped_data:

            # Create a dictionary for the current row, using the predefined fieldnames as keys
            row_data = {
                REPORT_FIELDNAMES[0]: kinect_file,
                REPORT_FIELDNAMES[1]: nerve_file,
                REPORT_FIELDNAMES[2]: success_status,
                REPORT_FIELDNAMES[3]: lag_value,
                REPORT_FIELDNAMES[4]: f"{score:.2f}",
                REPORT_FIELDNAMES[5]: kinect_ratio,
                REPORT_FIELDNAMES[6]: neuron_ratio,
                REPORT_FIELDNAMES[7]: align_tech,
                REPORT_FIELDNAMES[8]: comment_text
            }
            report_data.append(row_data)

        with open(report_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=REPORT_FIELDNAMES)
            writer.writeheader()
            for row in report_data:
                writer.writerow(row)























