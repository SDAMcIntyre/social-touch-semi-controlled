import glob
import os
import re
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# homemade libraries
# Add project root to sys.path to allow importing local libraries
sys.path.append(str(Path(__file__).resolve().parent.parent))
import libraries.misc.path_tools as path_tools  # noqa: E402


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


if __name__ == "__main__":    # parameters
    force_processing = True  # If user wants to force data processing even if results already exist
    save_results = True
    # generate_report = True
    
    # result saving parameters
    input_filename_pattern_r = r"semicontrolled_block-order(0[1-9]|1[0-8]).csv"
    show = False  # If user wants to monitor what's happening
    
    # choose the method to split trials:
    #  - soft: will find the middle point between the blocks to separate them, else chunk by TTL
    #  - hard: ?
    #  - with_following_rest_time: will include the signal when it is OFF too as the gesture can continue
    #                              after the TTL goes off (internal "lag" of the expert signing)
    split_type = "with_following_rest_time"  # soft, hard, TTL_beginning

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled", "3_merged")
    # get input base directory
    db_path_input = os.path.join(db_path, "sorted_by_block")
    # get output base directory
    db_path_output = os.path.join(db_path, "sorted_by_trial")
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

    
    sessions = ['2022-06-15_ST14-04']
    use_specific_blocks = True
    specific_blocks = ['block-order01']

    diff_ms_all = []
    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:
        curr_dir = os.path.join(db_path_input, session)
        files_abs, files = path_tools.find_files_in_directory(curr_dir, ending=input_filename_pattern_r)

        output_session_abs = os.path.join(db_path_output, session)
        if save_results and not os.path.exists(output_session_abs):
            os.makedirs(output_session_abs)
            print(f"Directory '{output_session_abs}' created.")

        for file_abs, file in zip(files_abs, files):
            print(f"\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
            print(f"current file: {file}")
            if use_specific_blocks :
                is_not_specific_block = True
                for block in specific_blocks:
                    if block in file:
                        is_not_specific_block = False
                if is_not_specific_block:
                    continue

            output_dir_abs = os.path.dirname(file_abs).replace(db_path_input, db_path_output)
            if not force_processing and glob.glob(os.path.join(output_dir_abs, "*.csv")):
                print(f"output folder already contains csv file(s), most likely output data, and not force processing. Skipping...")
                continue

            # load current data
            data = pd.read_csv(file_abs)
            # use Nerve_TTL as only the good nerve signal have been kept
            on_signal_idx = find_groups_of_ones(data["Nerve_TTL"].values)

            # extract the identifier indexes of the TTL (1, 2, 3, or 4)
            block_id_blinking_dur_sec = .100
            block_id_blinking_nsample = block_id_blinking_dur_sec * 1000
            on_signal_lengths = [len(sublist) for sublist in on_signal_idx]
            is_block_id = [length < block_id_blinking_nsample for length in on_signal_lengths]
            blocks_id_idx = [index for index, value in enumerate(is_block_id) if value]
            if len(blocks_id_idx) == 0:
                warnings.warn("Issue detected: Value of the block id is equal to zero.")
                continue

            # remove the identifier blocks to keep only the trial blocks.
            trials_idx = [trial_indexes for idx, trial_indexes in enumerate(on_signal_idx) if idx not in blocks_id_idx]

            # modify the trials idx in function of split_type
            # find middle point between trials as split locations
            if split_type == "with_following_rest_time":
                trials_idx_new = []
                ntrials = len(trials_idx)
                for trial_id in np.arange(ntrials):
                    curr_trial_indexes = trials_idx[trial_id]
                    start_idx = curr_trial_indexes[0]
                    if trial_id+1 == ntrials:
                        # get the last sample of the file as it is the last trial
                        end_idx = len(data["Nerve_TTL"].values)
                    else:
                        next_trial_indexes = trials_idx[trial_id+1]
                        end_idx = next_trial_indexes[0]
                    trials_idx_new.append(start_idx + np.arange(end_idx-start_idx))
                trials_idx = trials_idx_new

            elif split_type ==  "soft":
                trials_idx_initial = trials_idx
                ntrials = len(trials_idx_initial)
                middles = []
                avg_rest_time = []
                for x in range(0, ntrials-1):
                    last = trials_idx_initial[x][-1]
                    first = trials_idx_initial[x+1][1]
                    rest_time = (first-last)
                    avg_rest_time.append(rest_time)
                    middle = int(last+rest_time/2)
                    middles.append(middle)
                # add starting point of the first trial block based on avg extension
                middles.insert(0, int(trials_idx_initial[0][0]-np.mean(avg_rest_time)/2))
                # add end point of the last trial block
                middles.append(int(trials_idx_initial[-1][-1]+np.mean(avg_rest_time)/2))
                # redefine trials_idx
                trials_idx = []
                for m_idx in range(0, len(middles)-1):
                    trials_idx.append(np.arange(middles[m_idx], middles[m_idx+1]))
            elif split_type == "hard":
                pass
            

            # split by trials
            data_trials = []
            for trial_indexes in trials_idx:
                # get trials
                data_curr = data.iloc[trial_indexes].reset_index(drop=True)
                data_trials.append(data_curr)

            ntrials = len(data_trials)
            if ntrials == 0:
                warnings.warn("Issue detected: Number of trials is equal to zeros.")

            for (trial_id, data_trial) in enumerate(data_trials, 1):  # starts at id = 1
                output_filename = file.replace(".csv", f"_trial{trial_id:02}.csv")
                output_filename_abs = os.path.join(output_dir_abs, output_filename)

                if show:
                    plt.figure(figsize=(10, 12))  # Increase height for two subplots
                    plt.plot(data_trial["Nerve_TTL"].values, label='adjusted')
                    plt.plot(data_trial["LED on"].values, label='TTL_kinect_rescale', alpha=0.6, linestyle='--')
                    plt.legend()
                    plt.title('TTL_kinect_rescale')

                    # --- Attempt to maximize plot window ---
                    try:
                        fig_manager = plt.get_current_fig_manager()
                        # Try the 'zoomed' state first (common for TkAgg on Windows)
                        if hasattr(fig_manager, 'window') and hasattr(fig_manager.window, 'state'):
                             fig_manager.window.state('zoomed')
                             print("Attempted to maximize plot window using 'zoomed' state.")
                        # Fallback checks for other backends
                        elif hasattr(fig_manager, 'window') and hasattr(fig_manager.window, 'showMaximized'): # Qt
                            fig_manager.window.showMaximized()
                            print("Attempted to maximize plot window using 'showMaximized'.")
                        elif hasattr(fig_manager, 'window') and hasattr(fig_manager.window, 'Maximize'): # Wx
                            fig_manager.window.Maximize(True)
                            print("Attempted to maximize plot window using 'Maximize'.")
                        else:
                            print("Note: Could not automatically maximize plot window for the current backend.")
                    except Exception as e:
                        print(f"Note: Error trying to maximize plot window: {e}")
                    # --- End of maximization attempt ---

                    plt.show(block=True)

                # save data on the hard drive ?
                if save_results:
                    if not os.path.exists(output_dir_abs):
                        os.makedirs(output_dir_abs)
                        print(f"Directory '{output_dir_abs}' created.")

                    # https://answers.microsoft.com/en-us/msoffice/forum/all/excel-file-open-the-file-name-is-too-long-rename/ef736fec-0bd4-42a9-806d-5b22dbfdda81#:~:text=To%20resolve%20this%20issue%2C%20you,structure%2C%20is%20still%20too%20long.
                    #  Excel indicates that the total path length,
                    #  including the filename and its directory structure,
                    #  exceeds the Windows maximum limit of 260 characters.
                    data_trial.to_csv(output_filename_abs, index=False)
            
            print("done.")
