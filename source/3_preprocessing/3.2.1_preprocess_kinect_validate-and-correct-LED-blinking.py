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


def group_lengths2(arr):
    if not arr:
        return []

    group_length = []
    group_label = []
    current_length = 1
    current_value = arr[0]

    for i in range(1, len(arr)):
        if arr[i] == current_value:
            current_length += 1
        else:
            # save previous group length and label/value
            group_length.append(current_length)
            group_label.append(current_value)
            current_value = arr[i]
            current_length = 1

    # save the last group length and label/value
    group_length.append(current_length)
    group_label.append(current_value)

    return group_length, group_label


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
    force_processing = True  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening

    save_results = False
    generate_report = False

    print("Step 0: Extract the videos embedded in the selected neurons.")
    # get metadata dataframe
    md_path = os.path.join(path_tools.get_database_path(), "semi-controlled", "1_primary", "logs", "2_stimuli_by_blocks")

    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled", "2_processed", "kinect", "led")
    # get input base directory
    db_path_input = os.path.join(db_path, "0_block-order")
    # get output base directory
    db_path_output = os.path.join(db_path, "1_block-order_corrected")

    if save_results and not os.path.exists(db_path_output):
        os.makedirs(db_path_output)
        print(f"Directory '{db_path_output}' created.")

    # neuron names
    neurons_ST13 = ['2022-06-14_ST13-01',
                    '2022-06-14_ST13-02',
                    '2022-06-14_ST13-03']

    neurons_ST14 = ['2022-06-15_ST14-01',
                    '2022-06-15_ST14-02',
                    '2022-06-15_ST14-03',
                    '2022-06-15_ST14-04']

    neurons_ST15 = ['2022-06-16_ST15-01',
                    '2022-06-16_ST15-02']

    neurons_ST16 = ['2022-06-17_ST16-02',
                    '2022-06-17_ST16-03',
                    '2022-06-17_ST16-04',
                    '2022-06-17_ST16-05']

    neurons_ST18 = ['2022-06-22_ST18-01',
                    '2022-06-22_ST18-02',
                    '2022-06-22_ST18-04']
    neurons = []
    neurons = neurons + neurons_ST13
    neurons = neurons + neurons_ST14
    neurons = neurons + neurons_ST15
    neurons = neurons + neurons_ST16
    neurons = neurons + neurons_ST18
    print(neurons)

    diff_ms_all = []
    names_contact = []
    names_led = []
    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for neuron in neurons:
        curr_led_data = os.path.join(db_path_input, neuron)

        files_led_abs, files_led = path_tools.find_files_in_directory(curr_led_data, ending='_LED.csv')
        
        for file_led_abs, file_led in zip(files_led_abs, files_led):
            print(f"current file: {file_led}")
            # 1. check if results already exists
            output_filename = file_led  # file_led.replace(".csv", "_corrected.csv")
            output_dir_abs = os.path.join(db_path_output, neuron)
            output_filename_abs = os.path.join(output_dir_abs, output_filename)
            if not force_processing:
                try:
                    with open(output_filename_abs, 'r'):
                        print("Result file exists, jump to the next dataset.")
                        continue
                except FileNotFoundError:
                    pass

            # 2. check if led and nerve files exist for this contact file
            stimuli_info_file = os.path.join(md_path, neuron, file_led.replace("_LED.csv", "_stimuli.csv"))
            try:
                with open(stimuli_info_file, 'r'):
                    pass
            except FileNotFoundError:
                print("Matching stimuli file does not exist.")
                continue

            # 3. Load meaningful data
            # 3.1 from stimulus info
            stimuli = pd.read_csv(stimuli_info_file)
            # numbers of initial expected blinking
            run_block_id = stimuli["run_block_id"].values[0]
            # usually, blinking is 100ms maximum, which is equivalent to 4 frames at 30 Hz (Kinect Fs).
            blinking_id_nframe_max = 4
            # number of trials
            n_trials = len(stimuli["run_block_id"])

            # 3.2 from LED info
            led = pd.read_csv(file_led_abs)
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
            if led_block_id != run_block_id:
                w = (f"Issue detected: Value of the block id tracked in the FRAMES is not what is expected:\n"
                     f"(Expected){run_block_id} vs {led_block_id}(Tracked)")
                warnings.warn(w)
                plt.plot(led_on)
                plt.title(f"True block id = {run_block_id}")
                plt.show()
                continue
                plt.close()

            # 4.2 count the number of trials
            without_block_id = on_signal_idx[led_block_id-1][-1] + 1
            led_on_trials = led_on[without_block_id:]
            # Replace NaN values with zero
            nan_mask = np.isnan(led_on_trials)
            led_on_trials[nan_mask] = 0
            # get length of continuous labels (0 or 1)
            groups_length, groups_label = group_lengths(led_on_trials)
            # remove potential small undetected LED on to have consistent trial groups
            groups_length, groups_label = merge_small_group_with_neighbour(groups_length, groups_label, val_max=10)
            # sum all groups that are 1 (meaning it's a trial)
            led_n_trials = sum(groups_label)
            if led_n_trials != n_trials:
                w = (f"Issue detected: Value of the number of trials in the FRAMES is not what is expected:\n"
                     f"(Expected){n_trials} vs {led_n_trials}(Tracked)")
                warnings.warn(w)
                plt.plot(led_on_trials)
                plt.title(f"True of trials = {n_trials}")
                plt.show()
                continue
                plt.close()

            # 5. create output data
            df_output = []

            # keep variables for the report
            if generate_report:
                pass

            if show:
                pass

            # save data on the hard drive ?
            if save_results:
                if not os.path.exists(output_dir_abs):
                    os.makedirs(output_dir_abs)
                df_output.to_csv(output_filename_abs, index=False)

    if generate_report:
        report_filename = os.path.join(db_path_output, "frame_differences_report.csv")
        report_data = []
        for name_contact, name_led, diff_ms in zip(names_contact, names_led, diff_ms_all):
            report_data.append({"filename_contact": name_contact, "filename_led": name_led, "frame_difference": diff_ms})
        with open(report_filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["filename_contact", "filename_led", "frame_difference"])
            writer.writeheader()
            for row in report_data:
                writer.writerow(row)

    print("done.")

























