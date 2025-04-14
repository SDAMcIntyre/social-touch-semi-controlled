import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import os
import pandas as pd
from PIL import Image
import re
import sys
import warnings
import time

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libraries.materials.semicontrolled_data import SemiControlledData  # noqa: E402
from libraries.processing.semicontrolled_data_adjust_single_touch_chunks import AdjustChunksViewer  # noqa: E402
import libraries.misc.path_tools as path_tools  # noqa: E402


def adjust_start_indices(chunks):
  """
  Adjusts the start index of a tuple in a list of tuples by adding 1
  if the start index is equal to the end index of the previous tuple.

  Args:
    chunks: A list of tuples, where each tuple represents a start and end index.
            Example: [(2384, 4381), (4382, 7486), (7486, 8659), (8660, 10939)]

  Returns:
    A new list of tuples with adjusted start indices.
  """
  adjusted_chunks = []
  previous_end = None
  for start, end in chunks:
    if previous_end is not None and start == previous_end:
      adjusted_chunks.append((start + 1, end))
    else:
      adjusted_chunks.append((start, end))
    previous_end = end
  return adjusted_chunks


if __name__ == "__main__":
    # ----------------------
    # User control variables
    # ----------------------
    force_processing = False  # If user wants to force data processing even if results already exist
    save_figures = True
    save_results = True

    input_filename_pattern_r = r"semicontrolled_block-order(0[1-9]|1[0-8])_trial(0[1-9]|1[0-9]).csv"
    # choose the method to split single touches:
    #  - method_1: Stroking trials are split with position, Taping using only IFF
    #  - method_2: Stroking trials are split with position, Taping using only depth
    #  - method_3: Stroking trials are split with position, Taping using depth and IFF
    split_method = "method_2"
    input_single_touch_filename_end = f"_single-touch-endpoints_{split_method}.txt"
    output_single_touch_filename_end = f"_single-touch-endpoints_{split_method}_curated.txt"
    
    show = False  # If user wants to monitor what's happening
    show_single_touches = False  # If user wants to visualise single touches, one by one
    manual_check = True  # If user wants to take the time to check the trial and how it has been split
    # ----------------------
    # ----------------------
    # ----------------------

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get metadata paths
    md_stimuli_path = os.path.join(db_path, "1_primary", "logs", "2_stimuli_by_blocks")
    md_neuron_filename_abs = os.path.join(db_path, "1_primary", "nerve", "semicontrol_unit-name_to_unit-type.csv")
    # 2. check if neuron metadata file exists
    if not os.path.exists(md_neuron_filename_abs):
        s = f'The file {md_neuron_filename_abs} doesn''t  exist.'
        warnings.warn(s, Warning)

    # get input data directory
    db_path_input = os.path.join(db_path, "3_merged", "sorted_by_trial")
    # get output directories
    db_path_output = os.path.join(db_path, "3_merged", "sorted_by_single-touches")
    output_figure_path = os.path.join(db_path, "3_merged", "sorted_by_single-touches")
    if save_results and not os.path.exists(db_path_output):
        os.makedirs(db_path_output)
    if save_figures and not os.path.exists(output_figure_path):
        os.makedirs(output_figure_path)

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

    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:
        curr_dir = os.path.join(db_path_input, session)
        files_abs, files = path_tools.find_files_in_directory(curr_dir, ending=input_filename_pattern_r)

        for data_filename_abs, data_filename in zip(files_abs, files):
            print(f"current file: {data_filename}")
            # single touch endpoint results will be saved close to the .csv file.
            filename_output_abs = data_filename_abs.replace(".csv", output_single_touch_filename_end)
            # ensure window character path limitation of 260 is ignored
            filename_output_abs = path_tools.winapi_path(filename_output_abs)
            if not force_processing and os.path.exists(filename_output_abs):
                print(f'The file {filename_output_abs} exists. Skipping processing...', Warning)
                continue

            # 1. extract endpoints data
            endpoints_filename_abs = data_filename_abs.replace(".csv", input_single_touch_filename_end)
            # ensure window character path limitation of 260 is ignored
            endpoints_filename_abs = path_tools.winapi_path(endpoints_filename_abs)
            if os.path.exists(endpoints_filename_abs):
                # Initialize an empty list to store tuples
                loaded_endpoints = []
                # Open file and read lines
                with open(endpoints_filename_abs, 'r') as f:
                    for line in f:
                        # Remove newline character and split by commas
                        parts = line.strip().split(',')
                        # Convert parts to integers or floats as needed
                        endpoint = tuple(map(int, parts))  # Assuming integers; use float() if floats are expected
                        # Append the tuple to the list
                        loaded_endpoints.append(endpoint)
            else:
                print(f'The file {endpoints_filename_abs} does not exist.', Warning)
                continue
            # make sure that the endpoints have the correct format
            loaded_endpoints_adjusted = adjust_start_indices(loaded_endpoints)
            print("\nStart/End Chunks:", loaded_endpoints)
            print("Adjusted Start/End Chunks:", loaded_endpoints_adjusted)
            if not loaded_endpoints_adjusted:
                loaded_endpoints_adjusted = [(0, len(scd.contact.pos_1D))]

            # 2. extract metadata related to the current stimulus set and check if exists
            md_stimuli_filename = re.sub(r'_trial\d{2}\.csv', '_stimuli.csv', data_filename)
            md_stimuli_filename_abs = os.path.join(md_stimuli_path, session, md_stimuli_filename)
            if not os.path.exists(md_stimuli_filename_abs):
                warnings.warn(f'The file {md_stimuli_filename_abs} doesn''t exist.', Warning)
                continue

            # 3. check if neuron metadata file exists
            if not os.path.exists(md_neuron_filename_abs):
                warnings.warn(f'The file {md_neuron_filename_abs} doesn''t  exist.', Warning)
                continue

            # 4. create a SemiControlledData's list of TOUCH EVENT:
            # 4.1 load the data
            scd = SemiControlledData(data_filename_abs, md_stimuli_filename_abs, md_neuron_filename_abs)  # resources
            scd.set_variables(dropna=False)
            scd.contact.update_pos_1D(pca_range=(int((1/4)*scd.contact.nsample), int((3/4)*scd.contact.nsample)))
            # 4.2 create a list of signals to display:
            signals_list = [];  
            signals_list.append(scd.contact.pos_1D)
            signals_list.append(scd.contact.depth)
            signals_list.append(scd.contact.area)
            signals_list.append(scd.neural.iff)
            signals_list.append(scd.neural.spike)
            
            labels_list = []
            labels_list.append("position_1D (3D contact PCA)")
            labels_list.append("depth")
            labels_list.append("contact area")
            labels_list.append("iff")
            labels_list.append("spike")
            
            # create title
            info_str = ("Neuron Info: "
                        f"ID: {scd.neural.unit_id}\t"
                        f"Type: {scd.neural.unit_type}\n"
                        "Stimulus Info: "
                        f"Type: {scd.stim.type}\t"
                        f"Force: {scd.stim.force}\t"
                        f"Size: {scd.stim.size}\t"
                        f"Velocity: {scd.stim.vel} cm/s")
            title = data_filename + "\n" + info_str
            # 5. make a decision
            viewer = AdjustChunksViewer(signals_list, loaded_endpoints_adjusted, labels_list=labels_list, title=title)

            final_chunks = viewer.get_final_chunks()
            selected_chunks = viewer.get_selected_chunks()
            print("\nFinal (All) Chunks:", final_chunks)
            print("Selected Chunks:", selected_chunks)

            for tpl in selected_chunks:
                print(tpl)

            # 5. save endpoints results
            if save_results:
                # Open file for writing
                with open(filename_output_abs, 'w') as f:
                    for endpoint in selected_chunks:
                        # Convert tuple to string format if needed
                        endpoint_str = ','.join(map(str, endpoint))
                        # Write each tuple on a new line
                        f.write(endpoint_str + '\n')

    print("done.")
