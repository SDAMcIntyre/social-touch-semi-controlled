import matplotlib.pyplot as plt
import os
import pandas as pd
import socket
import tkinter as tk
from tkinter import filedialog

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'libraries'))

from libraries.processing.contactdata import ContactData, ContactDataPlot  # noqa: E402
from libraries.processing.semicontrolled_data import SemiControlledData  # noqa: E402


def get_path_abs():
    computer_name = socket.gethostname()
    print("Computer Name:", computer_name)

    if computer_name == "baz":
        data_dir_base = "E:\\"
    else:
        data_dir_base = 'C:\\Users\\basdu83'
    data_dir_base = os.path.join(data_dir_base, 'OneDrive - Linköpings universitet', '_Teams', 'touch comm MNG Kinect',
                                 'basil_tmp', 'data')
    data_dir = os.path.join(data_dir_base, 'processed')
    output_dir = os.path.join(data_dir_base, 'analysed')

    return data_dir, output_dir


def select_files(initial_folder):
    # Create the root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # File dialog to select multiple files, starting from the specified folder
    file_paths = filedialog.askopenfilenames(
        title="Select Files",
        initialdir=initial_folder,
        filetypes=[("All Files", "*.*")]
    )
    # Print the absolute paths of the selected files
    for file_path in file_paths:
        print(file_path)

    return list(file_paths)


def process_contact_data(data_dir, per_repeat=True):
    if per_repeat:
        contact_data_filename = os.path.join(data_dir, 'contact_features_per_repeat.csv')
    else:
        contact_data_filename = os.path.join(data_dir, 'contact_features.csv')
    cd = ContactData(contact_data_filename)

    return cd


if __name__ == "__main__":
    data_dir = get_path_abs()[0]
    # target the current experiment
    data_dir = os.path.join(data_dir, 'semi-controlled')
    # target the current postprocessing
    data_dir = os.path.join(data_dir, 'tracking_and_neural')
    # target the datatype
    data_dir = os.path.join(data_dir, 'new_axes')
    selected_files = select_files(data_dir)

    #scd = SemiControlledData(selected_files[0], automatic_load=True)
    scd = SemiControlledData(selected_files[0], split_by_trial=False)

    scd.split_by_trial()

    #cd = process_contact_data(data_dir, per_repeat=True)
    cd = process_contact_data(data_dir, per_repeat=False)

    # cd.redefine_stimulus_groups()

    cd_plot = ContactDataPlot()
    print("yeah boi")
    fig_area = cd_plot.plot_contact_hist(cd, 'area')
    fig_depth = cd_plot.plot_contact_hist(cd, 'depth')
    fig_vel = cd_plot.plot_contact_hist(cd, 'velocity')

    plt.draw()
    plt.show()
    print("done.")
