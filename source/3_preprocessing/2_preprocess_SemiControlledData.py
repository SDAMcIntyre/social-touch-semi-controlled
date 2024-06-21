from collections import defaultdict, Counter
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import sys
import tkinter as tk

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libraries.preprocessing.semicontrolled_data_manager import SemiControlledDataManager  # noqa: E402
import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.misc.waitforbuttonpress_popup import WaitForButtonPressPopup
import libraries.plot.semicontrolled_data_visualizer as scdata_visualizer  # noqa: E402
import libraries.plot.semicontrolled_data_visualizer_unitcontact as scdata_visualizer_neur  # noqa: E402


def categorize_values(values, num_bins=10):
    """
    Categorize the values into bins and return the bin ranges.
    """
    bins = np.linspace(min(values), max(values), num_bins+1)
    categories = np.digitize(values, bins, right=True)
    bin_ranges = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
    return categories, bin_ranges


def display_single_touch_per_unit_type(scdm):
    list_single_touches_all = scdm.data

    # get a dictionary of key=unit_types, values=single touches
    dict_single_touches, unit_types = scdm.sort_per_unit_type()

    for unit_type, single_touches in dict_single_touches.items():
        # count the number of unit of this type
        unit_ids = [single_t.neural.unit_id for single_t in single_touches]
        nunit = Counter(unit_ids).keys()

        # set up the manager with the specific dataset
        scdm.set_data(single_touches)

        # extract the contact
        [velocity_cm, depth_cm, area_cm] = scdm.estimate_contact_averaging()
        [contact_types, velocity_cm_exp, depth_cm_exp, area_cm_exp] = scdm.get_contact_expected()

        # Use Shan's plot system
        data = {
            'estimated_velocity': velocity_cm,
            'estimated_depth': depth_cm,
            'estimated_area': area_cm,
            'expected_velocity': velocity_cm_exp,
            'expected_depth': depth_cm_exp,
            'expected_area': area_cm_exp,
            'contact_type': contact_types
        }
        df = pd.DataFrame(data)
        scdata_visualizer.display_attribute(df, selection=0)
        scdata_visualizer.display_attribute(df, selection=1)
        scdata_visualizer.display_attribute(df, selection=2)

        # Categorize each variable into 10 bins and get the bin ranges
        contact_attr_names = ["velocity", "depth", "area"]
        vel_cat, vel_ranges = categorize_values(velocity_cm, num_bins=7)
        depth_cat, depth_ranges = categorize_values(depth_cm, num_bins=3)
        area_cat, area_ranges = categorize_values(area_cm, num_bins=3)
        categorized_contact_attr = list(zip(vel_cat, depth_cat, area_cat))
        categorized_contact_ranges = [vel_ranges, depth_ranges, area_ranges]

        # create a window to display the information relative to the contact
        # characteristics for this unit type
        root = tk.Tk()
        scd_viz_neur = scdata_visualizer_neur.SemiControlledData_VisualizerNeuralContact(root)
        scd_viz_neur.set_vars(unit_type, nunit, contact_attr_names, categorized_contact_attr, categorized_contact_ranges)
        scd_viz_neur.update_label()
        root.mainloop()

    scdm.set_data(list_single_touches_all)


def find_csv_files(sessions):
    """
    Given a list of session names,
    find all CSV files nested inside these directories.

    Args:
    sessions (list): A list where of directory names.

    Returns:
    session_led_dict: A dictionary of paths to all CSV files found.
    """

    data_dir_base = path_tools.get_path_root_abs()
    input_dir = "processed"
    experiment = "semi-controlled"
    datatype_str = os.path.join("contact_and_neural", "new_axes_3Dposition")
    neural_name2type_filename = "semicontrol_unit-name_to_unit-type.csv"

    # destination
    dir_abs = os.path.join(data_dir_base, input_dir)
    # destination
    dir_abs = os.path.join(dir_abs, experiment)
    # target the current experiment, postprocessing and datatype
    dir_abs = os.path.join(dir_abs, datatype_str)

    session_contactneur_dict = {}
    for session in sessions:
        # Create the search pattern
        file_path = os.path.join(dir_abs, session+"*")
        # Use glob to find files matching the pattern
        files_list = glob.glob(file_path)

        if len(files_list) != 1:
            continue

        filename = files_list[0]
        if filename.endswith('.csv'):
            csv_file_info = {"session": session,
                             "file_path": dir_abs,
                             "filename": Path(filename).name,
                             "filename_abs": filename,
                             "neuron_name2type_filename_abs": os.path.join(dir_abs, neural_name2type_filename)
                             }
            session_contactneur_dict[session] = csv_file_info

    return session_contactneur_dict


def find_led_files(sessions):
    """
    Given a list of session names,
    find all led files nested inside these directories.

    Args:
    sessions (list): A list where of directory names.

    Returns:
    session_led_dict: A dictionary of paths to all led files found.
    """
    data_dir_base = path_tools.get_path_root_abs()
    input_dir = "processed"
    experiment = "semi-controlled"
    datatype_str = "kinect"

    # destination
    dir_abs = os.path.join(data_dir_base, input_dir)
    # destination
    dir_abs = os.path.join(dir_abs, experiment)
    # target the current experiment, postprocessing and datatype
    dir_abs = os.path.join(dir_abs, datatype_str)

    session_led_dict = {}

    for session in sessions:
        # Create the search pattern
        session_4led = session.replace("-ST", "_ST").replace("unit", "0")
        dir_path = os.path.join(dir_abs, session_4led)
        if not os.path.exists(dir_path):
            continue

        led_files_info = []
        block_id = 0
        for file_path, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.csv'):
                    block_id += 1
                    led_file_info = {"session": session,
                                     "block_id": block_id,
                                     "file_path": file_path,
                                     "timeseries_filename": file,
                                     "metadata_filename": file.replace(".csv", "_metadata.txt")
                                     }
                    led_files_info.append(led_file_info)

        session_led_dict[session] = led_files_info

    return session_led_dict


if __name__ == "__main__":
    """
    Load the CSV data: preprocess, split into single touch event, and save the generated variable with pickle.dump
    """

    force_processing = False  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening
    save_results = True

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    database_path = path_tools.get_database_path()
    # get input base directory
    database_path_input = os.path.join(database_path, "semi-controlled", "processed", "kinect", "led")
    # get output base directory
    database_path_output = os.path.join(database_path, "semi-controlled", "primary", "kinect", "roi_led")
    if not os.path.exists(database_path_output):
        os.makedirs(database_path_output)
        print(f"Directory '{database_path_output}' created.")
    # Session names
    sessions_ST13 = ['2022-06-14-ST13-unit1',
                     '2022-06-14-ST13-unit2',
                     '2022-06-14-ST13-unit3']

    sessions_ST14 = ['2022-06-15-ST14-unit1',
                     '2022-06-15-ST14-unit2',
                     '2022-06-15-ST14-unit3',
                     '2022-06-15-ST14-unit4']

    sessions_ST15 = ['2022-06-16-ST15-unit1',
                     '2022-06-16-ST15-unit2']

    sessions_ST16 = ['2022-06-17-ST16-unit2',
                     '2022-06-17-ST16-unit3',
                     '2022-06-17-ST16-unit4',
                     '2022-06-17-ST16-unit5']

    sessions_ST18 = ['2022-06-22-ST18-unit1',
                     '2022-06-22-ST18-unit2',
                     '2022-06-22-ST18-unit4']

    sessions = []
    sessions = sessions + sessions_ST13
    #sessions = sessions + sessions_ST14
    #sessions = sessions + sessions_ST15
    #sessions = sessions + sessions_ST16
    #sessions = sessions + sessions_ST18
    print(sessions)



    load_mode = "automatic"  # "manual", "automatic"
    save_data = False

    # Session names
    sessions = ['2022-06-14-ST13-unit1',
                '2022-06-14-ST13-unit2',
                '2022-06-14-ST13-unit3',
                '2022-06-15-ST14-unit1',
                '2022-06-15-ST14-unit2',
                '2022-06-15-ST14-unit3',
                '2022-06-15-ST14-unit4',
                '2022-06-15-ST15-unit1',
                '2022-06-15-ST15-unit2',
                '2022-06-17-ST16-unit2',
                '2022-06-17-ST16-unit3',
                '2022-06-17-ST16-unit4',
                '2022-06-17-ST16-unit5',
                '2022-06-22-ST18-unit1',
                '2022-06-22-ST18-unit2',
                '2022-06-22-ST18-unit4']
    # data not fully download on the hard drive, needs to create a sub selection
    sessions = ['2022-06-14-ST13-unit1',
                '2022-06-14-ST13-unit2',
                '2022-06-14-ST13-unit3',
                '2022-06-15-ST14-unit1',
                '2022-06-15-ST14-unit2',
                '2022-06-15-ST14-unit3',
                '2022-06-22-ST18-unit1',
                '2022-06-22-ST18-unit4']
    session_contactneur_dict = find_csv_files(sessions)
    session_led_dict = find_led_files(sessions)

    scd_list = []
    for session in sessions:
        led_files_info = session_led_dict[session]
        data_filename = session_contactneur_dict[session]["filename_abs"]
        neuron_name2type_filename = session_contactneur_dict[session]["neuron_name2type_filename_abs"]

        # create a list of SemiControlledData of touch event
        scdm = SemiControlledDataManager()
        data_sliced = scdm.preprocess_data_file(data_filename, neuron_name2type_filename, led_files_info,
                                                correction=True, show=False, verbose=True)

        # store the calculated data
        scd_list.append(data_sliced)

    # save data on the hard drive ?
    if save_data:
        with open(os.path.join(output_dir, 'semicontrolleddata_period_list.pkl'), 'wb') as file:
            pickle.dump(scdm, file)

    scdata_visualizer.display_scd_one_by_one(scdm.data)
    # local function: display the stimulus content/landscape for each unit type
    display_single_touch_per_unit_type(scdm)
    scdm.define_trust_scores()
    #plt.plot(scdm.get_ratio_durations())
    scdata_visualizer.display_scd_one_by_one(scdm.data)

    print("done.")





