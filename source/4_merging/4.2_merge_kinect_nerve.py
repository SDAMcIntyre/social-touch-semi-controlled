import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
from scipy.interpolate import interp1d
from scipy.signal import correlate
from scipy import signal
import warnings

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libraries.misc.path_tools as path_tools  # noqa: E402


def cross_correlation_shift(signal1, signal2):
    correlation = correlate(signal1, signal2, mode='full')
    shift = np.argmax(correlation) - (len(signal2) - 1)
    return shift


def cost_function(params, signal1, signal2):
    shift, scale = params
    shifted_indices = (np.arange(len(signal2)) * scale) + shift
    interpolator = interp1d(shifted_indices, signal2, bounds_error=False, fill_value=0)
    shifted_scaled_signal2 = interpolator(np.arange(len(signal1)))
    return np.nansum((signal1 - shifted_scaled_signal2) ** 2)


def reconstruct_original_signal(shrinked_signal, estimated_shift, estimated_scale, original_length):
    # Generate indices for the original signal length
    original_indices = np.arange(original_length)

    # Reverse the scaling and shifting
    shrinked_indices = (original_indices - estimated_shift) / estimated_scale

    # Interpolate the shrinked signal to reconstruct the original signal
    interpolator = interp1d(np.arange(len(shrinked_signal)), shrinked_signal, bounds_error=False, fill_value=np.nan)
    reconstructed_signal = interpolator(shrinked_indices)

    return reconstructed_signal


if __name__ == "__main__":
    force_processing = True  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening
    save_results = True

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get input base directory
    db_path_input_kinect = os.path.join(db_path, "merged", "kinect", "1_block-order")
    db_path_input_nerve = os.path.join(db_path, "processed", "nerve", "2_block-order")
    # get output base directory
    db_path_output = os.path.join(db_path, "merged", "kinect_and_nerve", "0_block-order")
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

    diff_ms_all = []
    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:
        curr_kinect_dir = os.path.join(db_path_input_kinect, session)
        curr_nerve_dir = os.path.join(db_path_input_nerve, session)

        files_contact_abs, files_contact = path_tools.find_files_in_directory(curr_kinect_dir, ending='_kinect.csv')

        for file_contact_abs, file_contact in zip(files_contact_abs, files_contact):
            print(f"current file: {file_contact}")
            # check if led and nerve files exist for this contact file
            file_nerve = file_contact.replace("kinect.csv", "nerve.csv")
            file_nerve_abs = os.path.join(curr_nerve_dir, file_nerve)
            try:
                with open(file_nerve_abs, 'r'):
                    pass
            except FileNotFoundError:
                print("Matching nerve file does not exist.")
                continue

            # check if already exist, if not forced to process
            output_filename = file_contact.replace("_kinect.csv", "_kinect_and_nerve.csv")
            output_dir_abs = os.path.join(db_path_output, session)
            if not os.path.exists(output_dir_abs):
                os.makedirs(output_dir_abs)
            output_filename_abs = os.path.join(output_dir_abs, output_filename)
            if not force_processing:
                try:
                    with open(output_filename_abs, 'r'):
                        print("Result file exists, jump to the next dataset.")
                        continue
                except FileNotFoundError:
                    pass

            # load current data
            kinect = pd.read_csv(file_contact_abs)
            nerve = pd.read_csv(file_nerve_abs)

            # 0. scale up kinect data to match nerve sampling rate
            nerve_Fs = 1/np.mean(np.diff(nerve.Sec_FromStart))
            kinect_Fs = 1/np.mean(np.diff(kinect.t))
            estimated_scale = nerve_Fs / kinect_Fs
            nsample_scaled = int(estimated_scale * len(kinect))
            # create the scaled dataframe
            kinect_scaled = pd.DataFrame(columns=kinect.columns)
            kinect_scaled = kinect_scaled.reindex(np.linspace(0, nsample_scaled-1, nsample_scaled, dtype=int))
            # put each row to its expected location
            for old_index, row in kinect.iterrows():
                kinect_scaled.loc[int(old_index * estimated_scale)] = row
            # Fill nan values generated by scaling with previous valid value
            with pd.option_context('future.no_silent_downcasting', True):
                kinect_scaled = kinect_scaled.ffill()

            # 1. synchronise kinect and nerve data using TTL signal (nerve_Volt + kinect_LED)
            TTL_kinect = kinect_scaled["LED on"]
            TTL_nerve = nerve["TTL_Aut"]
            # remove temporarily any nan value for correlation
            TTL_kinect = np.nan_to_num(TTL_kinect, nan=0.0)
            TTL_nerve = np.nan_to_num(TTL_nerve, nan=0.0)
            # down sample for a faster correlation
            downsampling = .1
            TTL_kinect_corr = TTL_kinect[np.linspace(0, len(TTL_kinect)-1, int(downsampling * len(TTL_kinect)), dtype=int)]
            TTL_nerve_corr = TTL_nerve[np.linspace(0, len(TTL_nerve)-1, int(downsampling * len(TTL_nerve)), dtype=int)]
            # remove the mean for a better estimation of the correlation
            TTL_kinect_corr = TTL_kinect_corr - np.mean(TTL_kinect_corr)
            TTL_nerve_corr = TTL_nerve_corr - np.mean(TTL_nerve_corr)
            # lag estimation
            correlation = signal.correlate(TTL_kinect_corr, TTL_nerve_corr, mode="full")
            lags = signal.correlation_lags(TTL_kinect_corr.size, TTL_nerve_corr.size, mode="full")
            lag = int(lags[np.argmax(correlation)] / downsampling)
            del correlation, lags

            # align signals by shifting nerve data
            if lag > 0:
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
                zeros = np.zeros(nsample_kinect-nsample_nerve)
                nerve_shifted = np.concatenate((nerve_shifted, zeros))
                freq_shifted = np.concatenate((freq_shifted, zeros))
                TTL_Aut = np.concatenate((TTL_Aut, zeros))
                nerve_t = np.concatenate((nerve_t, zeros))

            # create output dataframe
            kinect_scaled.rename(columns={'t': 't_Kinect'})
            df_output = kinect_scaled
            df_output["Nerve_spike"] = nerve_shifted
            df_output["Nerve_freq"] = freq_shifted
            df_output["Nerve_TTL"] = TTL_Aut
            df_output["t"] = nerve_t

            if show:
                plt.figure(figsize=(10, 12))  # Increase height for two subplots
                plt.plot(df_output["Nerve_TTL"].values, label='adjusted')
                plt.plot(df_output["LED on"].values, label='TTL_kinect_rescale', alpha=0.6, linestyle='--')
                plt.legend()
                plt.title('TTL_kinect_rescale')
                plt.show()

            # save data on the hard drive ?
            if save_results:
                df_output.to_csv(output_filename_abs, index=False)

            print("done.")

























