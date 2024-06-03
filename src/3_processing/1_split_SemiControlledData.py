import numpy as np
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'libraries'))

import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.processing.semicontrolled_data_manager import SemiControlledDataManager  # noqa: E402
import libraries.misc.semicontrolled_data_visualizer as scdata_visualizer  # noqa: E402
import libraries.misc.time_cost_function as time_cost
from libraries.materials.semicontrolled_data import SemiControlledData  # noqa: E402


if __name__ == "__main__":

    load_mode = "automatic"  # "manual", "automatic"
    save_data = False

    [input_dir, output_dir] = path_tools.get_path_abs(input_dir="processed", output_dir="analysed")
    selected_files = path_tools.select_files(input_dir, mode=load_mode)

    # create a list of SemiControlledData
    scdm = SemiControlledDataManager()
    scdm.load_by_single_touch_event(selected_files, correction=True, show=False)

    [velocity_cm, depth_cm, area_cm] = scdm.estimate_contact_averaging()
    [contact_types, velocity_cm_exp, depth_cm_exp, area_cm_exp] = scdm.get_contact_expected()
    idx_tap = [index for index, value in enumerate(contact_types) if value == 'tap']
    idx_stroke = [index for index, value in enumerate(contact_types) if value == 'stroke']

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

    scdm.define_trust_scores()
    #plt.plot(scdm.get_ratio_durations())

    # save the data on the hard drive ?
    if save_data:
        with open(os.path.join(output_dir, 'semicontrolleddata_period_list.pkl'), 'wb') as file:
            pickle.dump(scdm, file)

    semicontrolled_data_visualizer.display_one_by_one(scdm.data)

    print("done.")
