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
    # target the current experiment, postprocessing and datatype
    dir_abs = os.path.join(data_dir_base, input_dir, experiment, datatype_str)

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


if __name__ == "__main__":
    force_processing = False  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening
    save_results = False


    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = path_tools.get_database_path()

    # get input base directory
    db_path_input = os.path.join(db_path, "semi-controlled", "processed")
    db_path_input_contact = os.path.join(db_path_input, "kinect", "contact", "1_block-order")
    db_path_input_led = os.path.join(db_path_input, "kinect", "led")
    db_path_input_nerve = os.path.join(db_path_input, "nerve", "2_block-order")

    # get output base directory
    db_path_output = os.path.join(db_path, "semi-controlled", "merged", "tracking_and_neural", "1_block-order")
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

    scd_list = []
    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:

        curr_contact_path_dir = os.path.join(db_path_input_contact, session)
        curr_contact_led_dir = os.path.join(db_path_input_led, session)
        curr_contact_nerve_dir = os.path.join(db_path_input_nerve, session)

        files_contact_abs, files_contact = path_tools.find_files_in_directory(curr_contact_path_dir, ending='_contact.csv')

        for file_contact_abs, file_contact in zip(files_contact_abs, files_contact):
            valid = True
            # check if led and nerve files exist for this contact file
            file_led = file_contact.replace("contact.csv", "LED.csv")
            file_led_abs = os.path.join(curr_contact_led_dir, file_led)
            try:
                with open(file_led_abs, 'r'):
                    print("Matching LED file exists.")
            except FileNotFoundError:
                print("Matching LED file does not exist.")
                valid = False

            file_nerve = file_contact.replace("contact.csv", "nerve.csv")
            file_nerve_abs = os.path.join(curr_contact_nerve_dir, file_nerve)
            try:
                with open(file_nerve_abs, 'r'):
                    print("Matching nerve file exists.")
            except FileNotFoundError:
                print("Matching nerve file does not exist.")
                valid = False

            # if either of the LED or nerve file doesn't exist, go to next contact file
            if not valid:
                continue

            contact = pd.read_csv(file_contact_abs)
            led = pd.read_csv(file_led_abs)
            nerve = pd.read_csv(file_nerve_abs)

            print(f"current filename = {file_contact.replace('_contact.csv', '')}:")
            print(f"contact = {contact['t'].iloc[-1]} seconds.")
            print(f"led = {led['time (second)'].iloc[-1]} seconds.")
            print(f"nerve = {nerve['Sec_FromStart'].iloc[-1]} seconds.")
            print(f"------------------\n")
            # save data on the hard drive ?
            if save_results:
                pass

            print("done.")





