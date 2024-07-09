import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import warnings

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
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


if __name__ == "__main__":
    force_processing = True  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening
    save_results = True

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get input base directory
    db_path_input = os.path.join(db_path, "merged", "kinect_and_nerve", "0_block-order")
    # get output base directory
    db_path_output = os.path.join(db_path, "merged", "kinect_and_nerve", "1_by-trials")
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
        curr_dir = os.path.join(db_path_input, session)
        files_abs, files = path_tools.find_files_in_directory(curr_dir, ending='_kinect_and_nerve.csv')

        output_session_abs = os.path.join(db_path_output, session)
        if not os.path.exists(output_session_abs):
            os.makedirs(output_session_abs)
            print(f"Directory '{output_session_abs}' created.")

        for file_abs, file in zip(files_abs, files):
            print(f"current file: {file}")

            block_order_str = file.replace("_kinect_and_nerve.csv", "")
            output_dir_abs = os.path.join(output_session_abs, block_order_str)
            if not os.path.exists(output_dir_abs):
                os.makedirs(output_dir_abs)
                print(f"Directory '{output_dir_abs}' created.")

            # load current data
            data = pd.read_csv(file_abs)
            # use Nerve_TTL as only the good nerve signal have been kept
            trials_idx = find_groups_of_ones(data["Nerve_TTL"].values)

            # extract the block id value and index
            block_id_blinking_dur_sec = .100
            block_id_blinking_nsample = block_id_blinking_dur_sec * 1000

            trial_lengths = [len(sublist) for sublist in trials_idx]
            is_block_id = [length < block_id_blinking_nsample for length in trial_lengths]
            block_id_idx = [index for index, value in enumerate(is_block_id) if value]
            block_id_val = len(block_id_idx)

            if block_id_val == 0:
                warnings.warn("Issue detected: Value of the block id is equal to zero.")

            # get trials
            data_trials = []
            for idx, trial_indexes in enumerate(trials_idx):
                # if it is a block id group, jump to next
                if idx in block_id_idx:
                    continue
                data_curr = data.iloc[trial_indexes].reset_index(drop=True)
                data_trials.append(data_curr)

            ntrials = len(data_trials)
            if ntrials == 0:
                warnings.warn("Issue detected: Number of trials is equal to zeros.")

            for (trial_id, data_trial) in enumerate(data_trials, 1):  # starts at id = 1
                output_filename = file.replace("_kinect_and_nerve.csv", f"_trial{trial_id:02}.csv")
                output_filename_abs = os.path.join(output_dir_abs, output_filename)
                if not force_processing:
                    try:
                        with open(output_filename_abs, 'r'):
                            print("Result file exists, jump to the next dataset.")
                            continue
                    except FileNotFoundError:
                        pass

                if show:
                    plt.figure(figsize=(10, 12))  # Increase height for two subplots
                    plt.plot(data_trial["Nerve_TTL"].values, label='adjusted')
                    plt.plot(data_trial["LED on"].values, label='TTL_kinect_rescale', alpha=0.6, linestyle='--')
                    plt.legend()
                    plt.title('TTL_kinect_rescale')
                    plt.show()

                # save data on the hard drive ?
                if save_results:
                    data_trial.to_csv(output_filename_abs, index=False)
            
            print("done.")

























