import csv
import numpy as np
import os
import pandas as pd
import re
import sys
import warnings
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.plot.semicontrolled_data_visualizer import DataVisualizer3D  # noqa: E402



if __name__ == "__main__":
    force_processing = True  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening

    save_results = False
    generate_report = False

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")

    # get input base directory
    db_path_input = os.path.join(db_path, "2_processed", "kinect")
    # get output base directory
    db_path_output = os.path.join(db_path, "2_processed", "kinect")
    if not os.path.exists(db_path_output):
        os.makedirs(db_path_output)
        print(f"Directory '{db_path_output}' created.")

    # session names
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
    # it is important to split by MNG files / session recordings to create the correct subfolders.
    for session in sessions:
        curr_contact_path_dir = os.path.join(db_path_input, session)

        files_contact_abs, files_contact = path_tools.find_files_in_directory(curr_contact_path_dir, ending='somatosensory_data.csv')

        # load all data of the current session into contacts
        contacts = []
        for file_contact_abs, file_contact in zip(files_contact_abs, files_contact):
            print(f"current file: {file_contact}")
            contacts.append(pd.read_csv(file_contact_abs))

        # load all data of the current session into contacts
        file_id = None
        xyz = None
        xyz_index = None
        xyz_green = None
        xyz_red = None
        xyz_yellow = None
        for idx, c in enumerate(contacts):
            if xyz is None:
                # Initialize with the first array
                tmp = [c["Position_x"].values, c["Position_y"].values, c["Position_z"].values]
                xyz = np.array(tmp)[:, 100:-100]
                tmp = [c["Position_index_x"].values, c["Position_index_y"].values, c["Position_index_z"].values]
                xyz_index = np.array(tmp)[:, 100:-100]
                tmp = [c["Position_green_x"].values, c["Position_green_y"].values, c["Position_green_z"].values]
                xyz_green = np.array(tmp)[:, 100:-100]
                tmp = [c["Position_red_x"].values, c["Position_red_y"].values, c["Position_red_z"].values]
                xyz_red = np.array(tmp)[:, 100:-100]
                tmp = [c["Position_yellow_x"].values, c["Position_yellow_y"].values, c["Position_yellow_z"].values]
                xyz_yellow = np.array(tmp)[:, 100:-100]

                file_id = np.ones(len(np.array(tmp)[0, 100:-100])) * idx
            else:
                tmp = c["Position_x"].values, c["Position_y"].values, c["Position_z"].values
                xyz = np.concatenate((xyz, np.array(tmp)[:, 100:-100]), axis=1)
                tmp = c["Position_index_x"].values, c["Position_index_y"].values, c["Position_index_z"].values
                xyz_index = np.concatenate((xyz_index, np.array(tmp)[:, 100:-100]), axis=1)
                tmp = c["Position_green_x"].values, c["Position_green_y"].values, c["Position_green_z"].values
                xyz_green = np.concatenate((xyz_green, np.array(tmp)[:, 100:-100]), axis=1)
                tmp = c["Position_red_x"].values, c["Position_red_y"].values, c["Position_red_z"].values
                xyz_red = np.concatenate((xyz_red, np.array(tmp)[:, 100:-100]), axis=1)
                tmp = c["Position_yellow_x"].values, c["Position_yellow_y"].values, c["Position_yellow_z"].values
                xyz_yellow = np.concatenate((xyz_yellow, np.array(tmp)[:, 100:-100]), axis=1)

                file_id = np.concatenate((file_id, np.ones(len(np.array(tmp)[0, 100:-100])) * idx))

        # concatenate the different locations for processing
        positions = [xyz, xyz_index, xyz_green, xyz_red, xyz_yellow]

        viz = DataVisualizer3D(session)
        viz.update(np.arange(np.shape(xyz)[1]), xyz)

        continue
        # get rid of outliers
        theoretical_acc_limit = 100  # cm/sec^2
        positions_no_outliers = []
        for pos in positions:
            acceleration = np.diff(np.diff(pos))
            acc_global = np.nansum(acceleration, axis=0)
            indices = np.where(acc_global > theoretical_acc_limit)[0]
            for idx in indices:
                pos[:, idx] = [np.nan, np.nan, np.nan]
            positions_no_outliers.append(pos)
        positions = positions_no_outliers

        # recover from concatenation before including back the processing data into respective dataframe
        xyz = positions[0]
        xyz_index = positions[1]
        xyz_green = positions[2]
        xyz_red = positions[3]
        xyz_yellow = positions[4]

        viz = DataVisualizer3D("xyz")
        viz.update(np.arange(np.shape(xyz)[1]), xyz)

        for file_contact_abs, file_contact in zip(files_contact_abs, files_contact):
            print(f"current file: {file_contact}")

            output_filename = file_contact.replace(".csv", "_no-outlier.csv")
            output_dir_abs = os.path.join(db_path_output, session)
            output_filename_abs = os.path.join(output_dir_abs, output_filename)
            if not force_processing:
                try:
                    with open(output_filename_abs, 'r'):
                        print("Result file exists, jump to the next dataset.")
                        continue
                except FileNotFoundError:
                    pass
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

























