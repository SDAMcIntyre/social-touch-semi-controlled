import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
# import shutil # No longer needed
from scipy import signal
import tkinter as tk
import sys
import warnings
from datetime import datetime # For timestamping

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.materials.semicontrolled_data import SemiControlledData  # noqa: E402
from libraries.plot.semicontrolled_data_visualizer import SemiControlledDataVisualizer  # noqa: E402
import libraries.processing.semicontrolled_data_cleaning as scd_cleaning  # noqa: E402
from libraries.processing.semicontrolled_data_correct_lag_manual import TimeSeriesLagGUI


if __name__ == "__main__":
    # parameters
    force_processing = True  # If user wants to force data processing even if results already exist in the log
    save_results = True      # If True, lag results will be saved to the log CSV file

    show = True  # If user wants to monitor what's happening (affects print statements, not core logic here)
    show_visualizer = True

    input_ending = r"\d{2}\.csv$"

    print("Script to manually correct lag between neural and kinematic data.")

    # Get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get Metadata base directory
    db_path_md = os.path.join(db_path, "1_primary", "logs", "2_stimuli_by_blocks")
    # Get input base directory
    db_path_input = os.path.join(db_path, "3_merged", "5.1.0_sorted_by_trial")
    # Get output base directory (for the log file)
    db_path_output = os.path.join(db_path, "3_merged", "5.1.1_sorted_by_trial")

    # Define log file path and headers
    log_file_name = "manual_lag_processing_log.csv"
    log_file_path = os.path.join(db_path_output, log_file_name)
    log_headers = ["file_path", "session", "filename", "lag_samples", "lag_seconds",
                   "ratio_lag_to_length", "status", "processing_timestamp", "comment"]

    # Create output directory for the log file if it doesn't exist and save_results is True
    if save_results and not os.path.exists(db_path_output):
        os.makedirs(db_path_output)
        print(f"Directory '{db_path_output}' created for the log file.")

    # Load existing log file or initialize an empty DataFrame
    # This DataFrame is used to check if a file has been processed
    # and will be updated if save_results is True.
    try:
        if os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0:
            log_df = pd.read_csv(log_file_path)
            # Ensure all defined headers are present
            for header in log_headers:
                if header not in log_df.columns:
                    log_df[header] = pd.NA
            log_df = log_df[log_headers] # Enforce column order and selection
        else:
            log_df = pd.DataFrame(columns=log_headers)
    except pd.errors.EmptyDataError:
        print(f"Log file {log_file_path} is empty. Initializing new log.")
        log_df = pd.DataFrame(columns=log_headers)
    except Exception as e:
        print(f"Error loading log file {log_file_path}: {e}. Initializing new log.")
        log_df = pd.DataFrame(columns=log_headers)


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
    
    use_specific_sessions = True
    if not use_specific_sessions:
        sessions = []
        sessions = sessions + sessions_ST13
        sessions = sessions + sessions_ST14
        sessions = sessions + sessions_ST15
        sessions = sessions + sessions_ST16
        sessions = sessions + sessions_ST18
    else:
        sessions = ['2022-06-14_ST13-01']

    use_specific_blocks = False
    specific_blocks = ['block-order01',
                       'block-order02']

    print(f"Processing sessions: {sessions}")

    for session_name in sessions:
        curr_dir = os.path.join(db_path_input, session_name)
        if not os.path.isdir(curr_dir):
            print(f"Warning: Session directory not found: {curr_dir}. Skipping.")
            continue
        files_abs, files_short = path_tools.find_files_in_directory(curr_dir, ending=input_ending)

        for file_abs_path, file_short_name in zip(files_abs, files_short):
            print(f"---------------\nChecking dataset: {file_short_name} in session {session_name}")
            print(f"Full path: {file_abs_path}")
            if use_specific_blocks :
                is_not_specific_block = True
                for block in specific_blocks:
                    if block in file_short_name:
                        is_not_specific_block = False
                if is_not_specific_block:
                    continue

            # Check if already processed
            processed_already = False
            existing_entry = log_df[log_df['file_path'] == file_abs_path]

            if not existing_entry.empty:
                processed_already = True
                status_in_log = existing_entry.iloc[0]['status']
                lag_in_log = existing_entry.iloc[0]['lag_samples']
                print(f"File found in log. Status: '{status_in_log}', Lag: {lag_in_log} samples.")

            if processed_already and not force_processing:
                print(f"Skipping (force_processing=False).")
                continue
            
            if processed_already and force_processing:
                print("Reprocessing (force_processing=True).")

            # Load data
            try:
                stimuli_filename = re.sub(r"trial\d{2}\.csv$", "stimuli.csv", file_short_name)
                filename_md_path_abs = os.path.join(db_path_md, session_name, stimuli_filename)
                scd = SemiControlledData(file_abs_path, md_stim_filename=filename_md_path_abs, load_instant=True)
            except Exception as e:
                print(f"Error loading data for {file_abs_path}: {e}. Skipping.")
                if save_results: # Log error if we are saving results
                    error_entry = {
                        "file_path": file_abs_path, "session": session_name, "filename": file_short_name,
                        "lag_samples": pd.NA, "lag_seconds": pd.NA, "ratio_lag_to_length": pd.NA,
                        "status": "error_loading_data", "processing_timestamp": datetime.now().isoformat(),
                        "comment": str(e)
                    }
                    if processed_already:
                        idx_to_update = existing_entry.index[0]
                        for key, value in error_entry.items(): log_df.loc[idx_to_update, key] = value
                    else:
                        log_df = pd.concat([log_df, pd.DataFrame([error_entry])], ignore_index=True)
                    try:
                        log_df.to_csv(log_file_path, index=False)
                    except Exception as e_save:
                        print(f"CRITICAL: Could not save error state to log: {e_save}")
                continue


            # Interpolate the contact nan values
            scd.contact.interpolate_missing_values(method="linear")

            neuron_iff = scd.neural.iff
            contact_depth = scd.contact.depth
            contact_hand_pos = scd.contact.pos_1D
            contact_hand_pos[100 < np.abs(contact_hand_pos)] = 0 # Basic outlier handling

            # Calculate the velocity
            data_Fs_contact = 1 / np.nanmean(np.diff(scd.contact.time))
            if np.isnan(data_Fs_contact) or data_Fs_contact == 0: data_Fs_contact = 100 # default Fs
            window_sec_vel = 0.1
            window_size_vel = int(window_sec_vel * data_Fs_contact)
            if window_size_vel < 1: window_size_vel = 1

            pos_x = scd_cleaning.smooth_signal(scd.contact.pos[0, :], window_size=window_size_vel)
            pos_y = scd_cleaning.smooth_signal(scd.contact.pos[1, :], window_size=window_size_vel)
            pos_z = scd_cleaning.smooth_signal(scd.contact.pos[2, :], window_size=window_size_vel)
            pos_xyz = np.array([pos_x, pos_y, pos_z])
            hand_vel_abs = np.linalg.norm(np.diff(pos_xyz, axis=1) / (1/data_Fs_contact), axis=0) # diff reduces length by 1
            # Pad hand_vel_abs to match original length for GUI consistency if needed, or use with care
            if len(hand_vel_abs) < len(pos_x): # typically len(hand_vel_abs) == len(pos_x) -1
                 hand_vel_abs = np.pad(hand_vel_abs, (0, len(pos_x) - len(hand_vel_abs)), 'edge')


            # Smooth signals for GUI display
            data_Fs_neural = 1 / np.nanmean(np.diff(scd.neural.time))
            if np.isnan(data_Fs_neural) or data_Fs_neural == 0: data_Fs_neural = 1000 # default Fs
            window_sec_smooth = 0.2
            window_size_smooth_contact = int(window_sec_smooth * data_Fs_contact)
            window_size_smooth_neural = int(window_sec_smooth * data_Fs_neural) # if smoothing neuron_iff
            if window_size_smooth_contact < 1: window_size_smooth_contact = 1
            
            # neuron_iff_smoothed = scd_cleaning.smooth_signal(neuron_iff, window_size=window_size_smooth_neural)
            contact_depth_smoothed = scd_cleaning.smooth_signal(contact_depth, window_size=window_size_smooth_contact)

            if show_visualizer:
                scd_viz = SemiControlledDataVisualizer(scd) # Not used directly for lag GUI
            
            root_app = tk.Tk()
            root_app.withdraw()
            
            ref_signals = [contact_depth_smoothed, contact_hand_pos, hand_vel_abs, pos_x, pos_y, pos_z]
            ref_labels = ["Contact Depth", "1D Hand Pos", "Hand Vel", 'Pos X', 'Pos Y', 'Pos Z']
            
            gui = TimeSeriesLagGUI(ref_signals, neuron_iff, # Pass original or smoothed IFF as needed
                                   ref_signal_labels=ref_labels,
                                   shift_signal_label="Neuron IFF",
                                   title=f"{file_short_name}\n{scd.stim.print(enriched_text=False)}",
                                   offset_references=True)
            #root_app.mainloop() # This will block until GUI is closed

            print("GUI closed.")
            lag_samples, lag_seconds_val = gui.get_lag()
            # comment_from_gui = gui.get_comment() # If your GUI had a comment feature

            root_app.destroy()
            print("Tkinter root app destroyed.")

            if save_results:
                current_timestamp = datetime.now().isoformat()
                log_status = ""
                calculated_ratio = pd.NA

                if lag_samples is not None:
                    log_status = "reprocessed" if processed_already else "processed"
                    if len(contact_depth_smoothed) > 0: # Use length of a reference signal
                        calculated_ratio = abs(lag_samples) / len(contact_depth_smoothed)
                    else:
                        calculated_ratio = pd.NA
                    print(f"Selected lag: {lag_samples} samples ({lag_seconds_val:.3f} s). Ratio: {calculated_ratio:.4f}")
                else:
                    log_status = "gui_closed_no_lag"
                    lag_samples = pd.NA # Ensure NA for log
                    lag_seconds_val = pd.NA # Ensure NA for log
                    print("GUI closed without saving lag.")

                current_log_entry = {
                    "file_path": file_abs_path,
                    "session": session_name,
                    "filename": file_short_name,
                    "lag_samples": lag_samples,
                    "lag_seconds": lag_seconds_val,
                    "ratio_lag_to_length": calculated_ratio,
                    "status": log_status,
                    "processing_timestamp": current_timestamp,
                    "comment": "" # Placeholder for comment_from_gui or manual entry later
                }

                if processed_already: # Update existing entry
                    idx_to_update = existing_entry.index[0]
                    for key, value in current_log_entry.items():
                        log_df.loc[idx_to_update, key] = value
                    print(f"Updated log for: {file_short_name}")
                else: # Append new entry
                    # log_df = log_df.append(current_log_entry, ignore_index=True) # Deprecated
                    new_row_df = pd.DataFrame([current_log_entry])
                    log_df = pd.concat([log_df, new_row_df], ignore_index=True)
                    print(f"Added new log entry for: {file_short_name}")
                
                # Save the updated log DataFrame to CSV
                try:
                    # Ensure all defined columns are present before saving, and in correct order
                    for header in log_headers:
                        if header not in log_df.columns:
                            log_df[header] = pd.NA
                    log_df = log_df[log_headers]
                    log_df.to_csv(log_file_path, index=False)
                    print(f"Log file saved: {log_file_path}")
                except Exception as e_save:
                    print(f"CRITICAL: Error saving log file {log_file_path}: {e_save}")
            
            # No saving of modified dataframe (df_output)
            
            if show_visualizer:
                del(scd_viz) # scd_viz was not used for GUI, but good practice if it were.
            del(scd) # Clean up data object

    # Removed: generate_report section

    print("-------------------------------------------")
    print("All specified sessions and files processed according to parameters.")
    if save_results:
        print(f"Processing log updated at: {log_file_path}")
    else:
        print("save_results was False. No log file was written or updated.")
    print("done.")