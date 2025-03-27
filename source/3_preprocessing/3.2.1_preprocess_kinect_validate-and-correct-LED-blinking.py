import csv
import numpy as np
import os
import pandas as pd
import re
import sys
import warnings
import matplotlib.pyplot as plt

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libraries.misc.path_tools as path_tools  # noqa: E402


def create_log_filename(input_string):
    # Extract date
    date_pattern = r'\d{4}-\d{2}-\d{2}'
    date_match = re.search(date_pattern, input_string)
    date = date_match.group() if date_match else None

    # Extract integers from the input string without date
    integer_pattern = r'\d+'
    integers = re.findall(integer_pattern, input_string.replace(date, ""))

    # Construct new string with the extracted date and integers
    output_string = f"{date}_ST{integers[0]}-{integers[1]}_semicontrolled_block-order{integers[2]}_stimuli.csv"

    return output_string


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



def group_lengths(arr):
    if arr.size == 0:
        return []

    # Find the indices where the value changes
    change_indices = np.where(arr[:-1] != arr[1:])[0] + 1

    # Include the start and end of the array
    indices = np.concatenate(([0], change_indices, [arr.size]))

    # Calculate lengths and corresponding values
    lengths = np.diff(indices)
    labels = arr[indices[:-1]]

    return lengths, labels



# merge neighbouring values to the small number
def merge_small_group_with_neighbour(values, labels, val_max=10):
    output_values = values
    output_labels = labels

    i = 0
    while i < len(output_values):
        if output_values[i] < val_max:
            # Sum neighbors if they exist
            if i > 0:
                # add left element to the current
                output_values[i] += output_values[i - 1]
                output_labels[i] = output_labels[i - 1]
                # delete left element
                output_values = np.delete(output_values, i - 1)
                output_labels = np.delete(output_labels, i - 1)
                # Move one step back after deletion
                i -= 1

            if i < len(output_values) - 1:
                # add right element to the current
                output_values[i] += output_values[i + 1]
                output_labels[i] = output_labels[i + 1]
                # delete right element
                output_values = np.delete(output_values, i + 1)
                output_labels = np.delete(output_labels, i + 1)

        i += 1  # Move to the next element

    print(f"values input:  {values}")
    print(f"values output: {output_values}")

    print(f"labels input:  {labels}")
    print(f"labels output: {output_labels}")
    return output_values, output_labels


if __name__ == "__main__":
    OS_linux = False

    force_processing = True  # If user wants to force data processing even if results already exist
    save_results = True
    generate_report = False

    input_filename_ending = '_kinect_LED.csv'
    # usually, blinking is 100ms maximum, which is equivalent to 4 frames at 30 Hz (Kinect Fs).
    blinking_id_nframe_max = 4
    show_warning = False  # If user wants to monitor what's happening
    show = False  # If user wants to monitor what's happening

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path_linux = os.path.expanduser('~/Documents/datasets/semi-controlled')
    if OS_linux:
        db_path = db_path_linux
    else:
        db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get input base directory
    db_path_md = os.path.join(db_path, "1_primary", "logs", "2_stimuli_by_blocks")
    db_path_input = os.path.join(db_path, "2_processed", "kinect")
    # set output base directory
    db_path_output = os.path.join(db_path, "2_processed", "kinect")
    if not os.path.exists(db_path_output):
        os.makedirs(db_path_output)
        print(f"Directory '{db_path_output}' created.")

    # neuron names
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
    names_contact = []
    names_led = []
    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:
        # files_abs, files, _ = find_metadata_files(db_path_input, session)
        curr_dir = os.path.join(db_path_input, session)
        files_abs, files = path_tools.find_files_in_directory(curr_dir, ending=input_filename_ending)
        
        for file_abs, file in zip(files_abs, files):
            print(f"current file: {file}")
            output_dirname = os.path.dirname(file_abs).replace(db_path_input, db_path_output)
            output_filename = file.replace(".csv", "_corrected.csv")
            output_filename_abs = os.path.join(output_dirname, output_filename)

            stimuli_info_filename = create_log_filename(file)
            stimuli_info_filename_abs = os.path.join(db_path_md, session, stimuli_info_filename)
            
            # 2. check if led and nerve files exist for this contact file
            if not force_processing and os.path.exists(output_filename_abs):
                print(f"File '{file}' already exists and the processing is not forced. Move to next file...")
                continue
            if not os.path.exists(stimuli_info_filename_abs):
                print("Matching stimuli file does not exist.")
                continue
            
            # extract block id and number of trials
            stimuli = pd.read_csv(stimuli_info_filename_abs)
            block_id = stimuli["run_block_id"].values[0]
            ntrials = len(stimuli)

            # 3.2 from LED info
            led = pd.read_csv(file_abs)
            led_on = led["LED on"].values

            # 4. work
            # use Nerve_TTL as only the good nerve signal have been kept
            on_signal_idx = find_groups_of_ones(led_on)

            # extract the identifier indexes of the TTL (1, 2, 3, or 4)
            on_signal_lengths = [len(sublist) for sublist in on_signal_idx]
            is_potential_block_id = [length < blinking_id_nframe_max for length in on_signal_lengths]
            # 4.2 count the number of True in a row at the start of the block
            led_block_id = 0
            for value in is_potential_block_id:
                if value:
                    led_block_id += 1
                else:
                    break
            if led_block_id != block_id:
                w = (f"Issue detected: Value of the block id tracked in the FRAMES is not what is expected:\n"
                     f"(Expected){block_id} vs {led_block_id}(Tracked)")
                warnings.warn(w)
                if show_warning: 
                    plt.plot(led_on)
                    plt.title(f"{file}\nError in Block ID blinking\nExpected vs LED block id = {block_id} vs {led_block_id}")
                    plt.show(block=True)
                continue

            # 4.2 count the number of trials
            block_id_end_loc = on_signal_idx[led_block_id-1][-1]
            led_on_trials_only = led_on[(block_id_end_loc+1):]
            # Replace NaN values with zero
            nan_mask = np.isnan(led_on_trials_only)
            led_on_trials_only[nan_mask] = 0
            # get length of continuous labels (0 or 1)
            groups_length, groups_label = group_lengths(led_on_trials_only)
            # remove potential small undetected LED on to have consistent trial groups
            groups_length, groups_label = merge_small_group_with_neighbour(groups_length, groups_label, val_max=10)
            # sum all groups that are 1 (meaning it's a trial)
            led_ntrials = sum(groups_label)
            if led_ntrials != ntrials:
                w = (f"Issue detected: Value of the number of trials in the FRAMES is not what is expected:\n"
                     f"(Expected){ntrials} vs {led_ntrials}(Tracked)")
                warnings.warn(w)
                if show_warning: 
                    plt.plot(led_on_trials_only)
                    plt.title(f"{file}\nError in the number of trials\nExpected vs LED trials number = {ntrials} vs {led_ntrials}")
                    plt.show(block=True)
                continue

            # 5. create output data with corrected block trials and initial block id blinking
            arr_output = np.concatenate([led_on[:block_id_end_loc+1], np.repeat(groups_label, groups_length)])
            
            if show:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
                ax1.plot(led_on)
                ax1.set_title('Input')
                ax2.plot(arr_output)
                ax2.set_title('Output')
                plt.show(block=True)

            # save data on the hard drive ?
            if save_results:
                if not os.path.exists(output_dirname):
                    os.makedirs(output_dirname)
                led["LED on"] = arr_output
                led.to_csv(output_filename_abs, index=False)

    print("done.")

























