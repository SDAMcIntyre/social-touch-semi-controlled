import csv
import matplotlib.pyplot as plt # Kept from original, though not directly used in this logic
import numpy as np
import os
import pandas as pd
import re
import shutil # Added for file copying
from scipy import signal # Kept from original
# import tkinter as tk # No longer needed for this script's purpose
import sys
import warnings
from datetime import datetime

# homemade libraries
# Ensure this path is correct relative to where you run the script, or add to PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.materials.semicontrolled_data import SemiControlledData  # noqa: E402
# from libraries.plot.semicontrolled_data_visualizer import SemiControlledDataVisualizer # Not needed
# import libraries.processing.semicontrolled_data_cleaning as scd_cleaning # Not needed directly
# from libraries.processing.semicontrolled_data_correct_lag_manual import TimeSeriesLagGUI # Not needed



if __name__ == "__main__":
    # --- Configuration ---
    try:
        base_db_path = path_tools.get_database_path()
    except Exception as e:
        print(f"Error getting database path: {e}")
        print("Please ensure 'path_tools.get_database_path()' is correctly configured or set 'base_db_path' manually.")
        sys.exit(1)

    # Input data path (as in original script)
    db_path_input = os.path.join(base_db_path, "semi-controlled", "3_merged", "sorted_by_trial")
    # Log file path
    log_file_dir = db_path_input # Log file is in the same directory as input sorted_by_trial
    log_file_name = "manual_lag_processing_log.csv"
    log_file_path = os.path.join(log_file_dir, log_file_name)

    # New output directory for corrected or copied files
    output_base_dir = os.path.join(base_db_path, "semi-controlled", "3_merged", "sorted_by_trial_lag_corrected")
    
    # Path for metadata (stimuli files)
    db_path_md_base = os.path.join(base_db_path, "semi-controlled", "1_primary", "logs", "2_stimuli_by_blocks")
    
    input_ending = r"\d{2}\.csv$" # From original script
    # --- End Configuration ---

    print(f"Starting script. Output will be in: {output_base_dir}")
    print(f"Reading log file from: {log_file_path}")

    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        print(f"Created base output directory: {output_base_dir}")

    # Load the log file
    if not os.path.exists(log_file_path):
        print(f"Warning: Log file not found at {log_file_path}. All files will be copied as-is.")
        log_df = pd.DataFrame() # Empty DataFrame if log not found
    else:
        try:
            log_df = pd.read_csv(log_file_path)
            print(f"Successfully loaded log file with {len(log_df)} entries.")
        except Exception as e:
            print(f"Error reading log file {log_file_path}: {e}. All files will be copied as-is.")
            log_df = pd.DataFrame()

    # Session names (from original script)
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
    sessions.extend(sessions_ST13)
    sessions.extend(sessions_ST14)
    sessions.extend(sessions_ST15)
    sessions.extend(sessions_ST16)
    sessions.extend(sessions_ST18)
    # sessions = ['2022-06-15_ST14-03'] # For testing, as in original script. Comment out for full run.
    print(f"Processing sessions: {sessions}")

    files_corrected_count = 0
    files_copied_count = 0
    error_processing_count = 0
    skipped_input_count = 0

    for session_name in sessions:
        current_input_session_dir = os.path.join(db_path_input, session_name)
        if not os.path.isdir(current_input_session_dir):
            print(f"Warning: Session directory not found: {current_input_session_dir}. Skipping.")
            skipped_input_count +=1
            continue

        current_output_session_dir = os.path.join(output_base_dir, session_name)
        if not os.path.exists(current_output_session_dir):
            try:
                os.makedirs(current_output_session_dir)
            except OSError as e:
                print(f"Error creating output session directory {current_output_session_dir}: {e}. Skipping session.")
                skipped_input_count += (len(path_tools.find_files_in_directory(current_input_session_dir, ending=input_ending)[0])
                                       if os.path.isdir(current_input_session_dir) else 1)
                continue
        
        files_abs, files_short = path_tools.find_files_in_directory(current_input_session_dir, ending=input_ending)

        for file_abs_path_original, file_short_name in zip(files_abs, files_short):
            print(f"\n--- File: {file_short_name} in session {session_name} ---")
            destination_file_path = os.path.join(current_output_session_dir, file_short_name)
            apply_correction_flag = False
            lag_to_apply = pd.NA

            if not log_df.empty:
                entry_in_log_df = log_df[log_df['file_path'] == file_abs_path_original]
                if not entry_in_log_df.empty:
                    log_row = entry_in_log_df.iloc[0]
                    lag_samples_val_from_log = log_row.get('lag_samples')
                    status_from_log = log_row.get('status')

                    if not (pd.isna(lag_samples_val_from_log) or \
                            str(lag_samples_val_from_log).strip() == 'NA' or \
                            str(lag_samples_val_from_log).strip() == ''):
                        try:
                            lag_to_apply = int(float(str(lag_samples_val_from_log)))
                            if status_from_log in ["processed", "reprocessed"]:
                                apply_correction_flag = True
                                print(f"Log entry found: Status '{status_from_log}', Lag {lag_to_apply} samples. Will apply correction.")
                            else:
                                print(f"Log entry found: Status '{status_from_log}'. Not 'processed' or 'reprocessed'. Will copy original.")
                        except ValueError:
                            print(f"Log entry found: lag_samples '{lag_samples_val_from_log}' is not a valid number. Will copy original.")
                    else:
                        print(f"Log entry found: No valid lag_samples (value: '{lag_samples_val_from_log}'). Will copy original.")
                else:
                    print("No entry found in log for this file. Will copy original.")
            else:
                print("Log file was empty or not found. Will copy original.")


            if apply_correction_flag:
                spike_shifted = pd.NA
                iff_shifted = pd.NA
                try:
                    # load data
                    scd = SemiControlledData(file_abs_path_original, load_instant=True)
                    # align signals by shifting nerve data
                    if lag_to_apply > 0:
                        zeros = np.zeros(lag_to_apply)
                        # to the right
                        spike_shifted = np.concatenate((zeros, scd.neural.spike[:-lag_to_apply]))
                        iff_shifted = np.concatenate((zeros, scd.neural.iff[:-lag_to_apply]))
                    elif lag_to_apply < 0:
                        lag_abs = abs(lag_to_apply)
                        zeros = np.zeros(lag_abs)
                        # to the left
                        spike_shifted = np.concatenate((scd.neural.spike[lag_abs:], zeros))
                        iff_shifted = np.concatenate((scd.neural.iff[lag_abs:], zeros))
                    else:
                        # Do nothing
                        spike_shifted = scd.neural.spike
                        iff_shifted = scd.neural.iff

                    # create the shifted dataset
                    df_output = pd.read_csv(file_abs_path_original)
                    df_output['Nerve_spike'] = spike_shifted
                    df_output['Nerve_freq'] = iff_shifted
                    df_output.to_csv(destination_file_path, index=False)

                    files_corrected_count += 1

                except FileNotFoundError:
                     print(f"Error during correction: Original data file not found: {file_abs_path_original}. Skipping.")
                     error_processing_count += 1
                except Exception as e_correct:
                    print(f"Error processing and correcting file {file_short_name}: {e_correct}")
                    error_processing_count += 1
                finally:
                    if scd: del scd
            else:
                # Copy the original file
                try:
                    shutil.copy2(file_abs_path_original, destination_file_path)
                    print(f"Successfully copied original file to: {destination_file_path}")
                    files_copied_count += 1
                except Exception as e_copy:
                    print(f"Error copying file {file_abs_path_original} to {destination_file_path}: {e_copy}")
                    error_processing_count += 1
    
    print("\n-------------------------------------------")
    print("File processing finished.")
    print(f"Files corrected and saved: {files_corrected_count}")
    print(f"Files copied as-is: {files_copied_count}")
    print(f"Files skipped due to input filters or missing session folders: {skipped_input_count}")
    print(f"Errors during processing/copying: {error_processing_count}")
    print(f"Output directory: {output_base_dir}")
    print("Done.")