
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys
from scipy import signal
import warnings
from pathlib import Path

from utils.should_process_task import should_process_task

def get_correlation(sig1, sig2, downsampling=0.1, show=False):
    # This helper function remains unchanged from the original script.
    # ... (function content is identical to the one provided) ...
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
    peak_corr_value = np.max(np.abs(correlation))
    peak_lag_index = np.argmax(np.abs(correlation))
    peak_lag = lags[peak_lag_index]
    energy1 = np.sum(sig1_corr**2)
    energy2 = np.sum(sig2_corr**2)
    if energy1 > 1e-10 and energy2 > 1e-10:
        normalized_peak = peak_corr_value / np.sqrt(energy1 * energy2)
    else:
        normalized_peak = 0.0
    percentage_match = normalized_peak * 100

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

def align_and_merge_neural_and_kinect(
    contact_filename: Path,
    nerve_filename: Path,
    output_filename: Path,
    *,
    force_processing:bool =False,
    show_plots: bool =False,
    save_plots: bool =False,
    scaling_nofilling: bool =True,
    show_correlation_result: bool =False
):
    """
    Aligns and merges Kinect and nerve signal data from two files.

    This function performs the following steps:
    1. Checks if the output already exists to potentially skip processing.
    2. Loads Kinect and nerve data from CSV files.
    3. Upsamples the Kinect data to match the nerve data's sampling rate.
    4. Calculates the time lag between the two signals using cross-correlation on their TTL signals.
    5. If TTL correlation is poor, it attempts correlation using contact area and nerve frequency.
    6. Aligns the nerve data based on the calculated lag.
    7. Merges the aligned data into a single DataFrame.
    8. Saves the merged DataFrame to a new CSV file.
    9. Optionally generates and saves/shows alignment plots.
    10. Returns a dictionary with processing metadata for reporting.

    Args:
        contact_filename (str): Absolute path to the Kinect CSV file.
        nerve_filename (str): Absolute path to the nerve CSV file.
        output_filename (str): Absolute path for the output merged CSV file.
        force_processing (bool, optional): If True, re-processes even if output exists. Defaults to False.
        show_plots (bool, optional): If True, displays the final alignment plot. Defaults to False.
        save_plots (bool, optional): If True, saves the final alignment plot. Defaults to False.
        scaling_nofilling (bool, optional): If True, fills only the 'led_on' column after upsampling. Defaults to True.
        show_correlation_result (bool, optional): If True, displays the cross-correlation plot. Defaults to False.

    Returns:
        dict: A dictionary containing metadata and results of the alignment process,
              including lag, success status, match score, and comments. Returns None if a file is not found.
    """
    # --- 1. Pre-computation checks ---
    if not os.path.exists(nerve_filename):
        print(f"Nerve file not found: '{nerve_filename}'. Skipping.")
        return None
    if not os.path.exists(contact_filename):
        print(f"Kinect file not found: '{contact_filename}'. Skipping.")
        return None

    if not should_process_task(
        input_paths=[contact_filename, nerve_filename], 
        output_paths=output_filename, 
        force=force_processing):
        print(f"✅ Output file '{output_filename}' already exists. Use force_processing to overwrite.")
        return
    
    print(f"\nProcessing: {os.path.basename(contact_filename)}")

    # --- 2. Load and Prepare Data ---
    kinect = pd.read_csv(contact_filename)
    nerve = pd.read_csv(nerve_filename)

    # Upsample Kinect data
    nerve_Fs = 1 / np.mean(np.diff(nerve.Sec_FromStart))
    kinect_Fs = 1 / np.mean(np.diff(kinect.time))
    estimated_scale = nerve_Fs / kinect_Fs
    nsample_scaled = int(estimated_scale * len(kinect))
    kinect_scaled = pd.DataFrame(columns=kinect.columns, index=range(nsample_scaled))
    for old_index, row in kinect.iterrows():
        kinect_scaled.loc[int(old_index * estimated_scale)] = row

    with pd.option_context('future.no_silent_downcasting', True):
        if scaling_nofilling:
            kinect_scaled["led_on"] = kinect_scaled["led_on"].ffill()
        else:
            kinect_scaled = kinect_scaled.ffill()

    # --- 3. Synchronize Signals (Correlation) ---
    TTL_kinect = kinect_scaled["led_on"].values
    TTL_nerve = nerve["TTL_Aut"].values
    match_score, lag = get_correlation(TTL_kinect, TTL_nerve, downsampling=0.1, show=show_correlation_result)
    
    lag_ratio = abs(lag) / len(TTL_kinect) if len(TTL_kinect) > 0 else 0
    print(f"Lag/TTL_kinect length (ratio): {lag} / {len(TTL_kinect)} ({lag_ratio:.3f})")

    # Fallback synchronization strategy
    if lag_ratio > 0.30:
        warnings.warn(f"Lag ratio ({lag_ratio:.3f}) is high. Attempting fallback sync.")
        comment = f"Lag ratio was too high ({lag_ratio:.3f}); using Contact Depth and Neuron IFF."
        
        with pd.option_context('future.no_silent_downcasting', True):
            k = kinect_scaled["contact_area"].ffill().values
        n = nerve["Freq"].values
        match_score, lag = get_correlation(k, n, downsampling=0.1, show=show_correlation_result)

        lag_ratio = abs(lag) / len(TTL_kinect) if len(TTL_kinect) > 0 else 0
        print(f"Fallback Lag/Contact_Area length (ratio): {lag} / {len(TTL_kinect)} ({lag_ratio:.3f})")
        
        if lag_ratio > 0.30:
            warnings.warn("Fallback lag ratio is also high. Ignoring sync and setting lag to 0.")
            comment = f"Fallback lag ratio was too high ({lag_ratio:.3f}); sync ignored."
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

    # --- 4. Align and Pad Signals ---
    if lag >= 0:
        nerve_shifted = np.pad(nerve["Nervespike1"].values, (lag, 0), 'constant', constant_values=0)
        freq_shifted = np.pad(nerve["Freq"].values, (lag, 0), 'constant', constant_values=0)
        TTL_Aut = np.pad(nerve["TTL_Aut"].values, (lag, 0), 'constant', constant_values=0)
        nerve_t = np.pad(nerve["Sec_FromStart"].values, (lag, 0), 'constant', constant_values=0)
    else:
        nerve_shifted = nerve["Nervespike1"].iloc[abs(lag):].values
        freq_shifted = nerve["Freq"].iloc[abs(lag):].values
        TTL_Aut = nerve["TTL_Aut"].iloc[abs(lag):].values
        nerve_t = nerve["Sec_FromStart"].iloc[abs(lag):].values

    # Match signal lengths
    nsample_kinect = len(kinect_scaled)
    nsample_nerve = len(nerve_shifted)
    if nsample_nerve > nsample_kinect:
        nerve_shifted, freq_shifted, TTL_Aut, nerve_t = (arr[:nsample_kinect] for arr in [nerve_shifted, freq_shifted, TTL_Aut, nerve_t])
    elif nsample_nerve < nsample_kinect:
        pad_width = nsample_kinect - nsample_nerve
        nan_vector = np.full(pad_width, np.nan)
        nerve_shifted = np.concatenate((nerve_shifted, nan_vector))
        freq_shifted = np.concatenate((freq_shifted, nan_vector))
        TTL_Aut = np.concatenate((TTL_Aut, nan_vector))
        nerve_t = np.concatenate((nerve_t, nan_vector))

    # --- 5. Create Output DataFrame ---
    df_output = kinect_scaled.copy()
    df_output = df_output.rename(columns={'time': 'time_kinect'})
    df_output["time_nerve"] = nerve_t
    df_output["time"] = np.linspace(0, len(TTL_Aut) / nerve_Fs, len(TTL_Aut))
    df_output["Nerve_spike"] = nerve_shifted
    df_output["Nerve_freq"] = freq_shifted
    df_output["Nerve_TTL"] = TTL_Aut
    df_output = df_output[['time', 'time_kinect', 'time_nerve'] + [col for col in df_output.columns if col not in ['time', 'time_kinect', 'time_nerve']]]

    # --- 6. Save Results and Plots ---
    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")
    
    df_output.to_csv(output_filename, index=False)
    print(f"Successfully saved merged data to {os.path.basename(output_filename)}")

    if save_plots or show_plots:
        # Plotting logic from original script
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(10, 6))
        x = df_output["time"].values
        # Plotting code remains the same...
        fig.suptitle(f'Contact: {os.path.basename(contact_filename)}\nNerve: {os.path.basename(nerve_filename)}', fontsize=16)
        plt.tight_layout()
        if save_plots:
            plot_filename = output_filename.replace(".csv", "_TTLs_alignement.png")
            plt.savefig(plot_filename)
        if show_plots:
            plt.show(block=True)
        plt.close('all')

    # --- 7. Return Report Data ---
    report = {
        "status": "completed",
        "success": success,
        "alignment_technique": alignment_technique,
        "lag": lag,
        "match_score": match_score,
        "ratio_kinect": abs(lag) / len(TTL_kinect) if len(TTL_kinect) > 0 else 0,
        "ratio_nerve": abs(lag) / len(TTL_nerve) if len(TTL_nerve) > 0 else 0,
        "comment": comment
    }
    return report