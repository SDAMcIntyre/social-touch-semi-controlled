from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
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


if __name__ == "__main__":

    load_mode = "automatic"  # "manual", "automatic"
    save_data = False

    """
    Load the CSV data: preprocess, split into single touch event, and save the generated variable with pickle.dump
    """
    [input_dir, output_dir] = path_tools.select_files_processed_data(input_dir="processed", output_dir="analysed")
    data_files = path_tools.select_files(input_dir, mode=load_mode)
    fname_neur_name2type = os.path.join(input_dir, "semicontrol_unit-name_to_unit-type.csv")

    # create a list of SemiControlledData
    scdm = SemiControlledDataManager()
    scdm.preprocess_data_files(data_files, fname_neur_name2type, correction=True, show=False)

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





