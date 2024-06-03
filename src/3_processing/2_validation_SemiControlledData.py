
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'libraries'))
from libraries.misc.path_tools import get_path_abs  # noqa: E402
from libraries.processing.semicontrolled_data_splitter import SemiControlledDataManager  # noqa: E402
from libraries.misc.semicontrolled_data_visualizer import SemiControlledDataVisualizer  # noqa: E402
from libraries.processing.semicontrolled_data_validation import SemiControlledValidatorApp  # noqa: E402


if __name__ == "__main__":

    save_data = True

    # define file and directory names
    [input_dir, output_dir] = get_path_abs(input_dir="analysed", output_dir="analysed")
    input_data_filename = 'semicontrolleddata_period_list.pkl'
    output_data_filename = 'semicontrolleddata_period_validation.pkl'

    # load the pickle variable SemiControlledDataManager
    with open(os.path.join(input_dir, input_data_filename), 'rb') as file:
        # Load the variable from the file
        scdm = pickle.load(file)
        if not isinstance(scdm, SemiControlledDataManager):
            raise TypeError(f"Invalid type: {type(scdm)}. my_var must be a SemiControlledDataManager")

    signal_valid_list = np.array([True] * len(scdm.data_periods))

    # automatic filtering
    # 1. Based on the duration
    duration_recorded = []
    duration_expected = []
    for scd in scdm.data_periods:
        duration_recorded.append(1000 * (scd.md.time[-1] - scd.md.time[0]))
        duration_expected.append(1000 * scd.stim.get_singular_contact_duration_expected())
    # Extract the indices of elements that is below the boundary
    incorrect_idx = [idx for idx, dur in enumerate(duration_recorded) if dur < 50]
    signal_valid_list[incorrect_idx] = False

    plt.hist(duration_recorded)
    plt.hist(duration_recorded, bins=200, edgecolor='black')


    # manual validation
    scd_visualiser = SemiControlledDataVisualizer()
    for index, scd in enumerate(scdm.data_periods):
        app = SemiControlledValidatorApp(scd, scd_visualiser)
        app.root.mainloop()
        signal_valid_list[index] = app.signal_valid is True

    # save the data on the hard drive ?
    if save_data:
        with open(os.path.join(output_dir, output_data_filename), 'wb') as file:
            pickle.dump(signal_valid_list, file)


