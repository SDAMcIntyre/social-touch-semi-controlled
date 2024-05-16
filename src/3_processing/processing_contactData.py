import os
import pandas as pd
import socket

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'libraries'))

from libraries.processing.contactdata import ContactData, ContactDataPlot  # noqa: E402


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
    plot_dir = os.path.join(data_dir_base, 'analysed')

    return data_dir, plot_dir


def process_contact_data(data_dir, per_repeat=True):
    if per_repeat:
        contact_data_filename = os.path.join(data_dir, 'contact_features_per_repeat.csv')
    else:
        contact_data_filename = os.path.join(data_dir, 'contact_features.csv')
    cd = ContactData(contact_data_filename)

    return cd


if __name__ == "__main__":
    data_dir, plot_dir = get_path_abs()
    cd = process_contact_data(data_dir, per_repeat=True)
    #cd.redefine_stimulus_groups()

    cd_plot = ContactDataPlot()
    cd_plot.plot_contact_hist(cd, 'area')
    #cd_plot.plot_contact_hist(cd, 'depth')
    #cd_plot.plot_contact_hist(cd, 'velocity')
