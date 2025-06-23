import csv
import math
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



def calculate_acceleration_from_position(position_xyz, fs):
  """
  Calculates acceleration vectors from a time series of position vectors.

  Args:
    position_xyz: A NumPy array of shape (N, 3) where N is the number
                  of time samples and columns represent X, Y, Z coordinates.
    fs: The sampling frequency in Hertz (Hz).

  Returns:
    A NumPy array of shape (N, 3) representing the acceleration vectors
    (Ax, Ay, Az) corresponding to each time sample. Returns None if input
    is invalid.
  """
  if not isinstance(position_xyz, np.ndarray) or position_xyz.ndim != 2 or position_xyz.shape[1] != 3:
      print("Error: position_xyz must be a NumPy array of shape (N, 3)")
      return None
  if position_xyz.shape[0] < 3:
      print("Error: Need at least 3 data points to calculate acceleration reliably.")
      return None
  if not isinstance(fs, (int, float)) or fs <= 0:
      print("Error: fs (sampling frequency) must be a positive number.")
      return None

  # Calculate the time step (delta t) between samples
  dt = 1.0 / fs

  # Calculate velocity (first derivative of position)
  # np.gradient calculates the gradient along the specified axis (axis=0 for time)
  # It uses central differences for interior points and forward/backward
  # differences for boundaries, providing second-order accuracy.
  # The 'edge_order=2' argument ensures second-order accuracy at the boundaries as well.
  velocity_xyz = np.gradient(position_xyz, dt, axis=0, edge_order=2)

  # Calculate acceleration (first derivative of velocity)
  acceleration_xyz = np.gradient(velocity_xyz, dt, axis=0, edge_order=2)

  return acceleration_xyz



if __name__ == "__main__":
    force_processing = True  # If user wants to force data processing even if results already exist
    save_results = True

    target_filename = 'somatosensory_data.csv'
    show = False  # If user wants to monitor what's happening
    generate_report = False

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")

    # get input base directory
    db_path_input_contact = os.path.join(db_path, "2_processed", "kinect")
    db_path_input_config = os.path.join(db_path, "1_primary", "kinect")
    # get output base directory
    db_path_output = os.path.join(db_path, "2_processed", "kinect")
    if not os.path.exists(db_path_output):
        os.makedirs(db_path_output)
        print(f"Directory '{db_path_output}' created.")
    
    # extract config files location
    files_config_abs, files_config = path_tools.find_files_in_directory(db_path_input_config, ending='somatosensory-features_extraction-config.json')
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
    use_specific_sessions = True
    if not use_specific_sessions:
        sessions = []
        sessions = sessions + sessions_ST13
        sessions = sessions + sessions_ST14
        sessions = sessions + sessions_ST15
        sessions = sessions + sessions_ST16
        sessions = sessions + sessions_ST18
    else:
        sessions = ['2022-06-17_ST16-02']
    
    use_specific_blocks = True
    specific_blocks = ['block-order-16']

    print(sessions)

    diff_ms_all = []
    names_contact = []
    names_led = []
    # it is important to split by MNG files / session recordings to create the correct subfolders.
    for session in sessions:
        curr_contact_path_dir = os.path.join(db_path_input_contact, session)
        files_contact_abs, files_contact = path_tools.find_files_in_directory(curr_contact_path_dir, ending=target_filename)
        # load all data of the current session into contacts
        contacts = []
        for file_contact_abs, file_contact in zip(files_contact_abs, files_contact):
            if use_specific_blocks:
                is_not_specific_block = True
                for block in specific_blocks:
                    if block in file_contact_abs:
                        is_not_specific_block = False
                if is_not_specific_block:
                    continue
            
            print(f"current file: {file_contact_abs}")
            #if not ('block-order-08' in file_contact_abs): continue
            
            output_filename = file_contact.replace(".csv", "_no-outlier.csv")
            output_dir_abs = os.path.dirname(file_contact_abs).replace(db_path_input_contact, db_path_output)
            output_filename_abs = os.path.join(output_dir_abs, output_filename)
            if not force_processing and os.path.exists(output_filename_abs):
                print(f"result has been found and not force processing. Skip this data...")
                continue

            data = pd.read_csv(file_contact_abs)
            # concatenate the different locations for processingl
            xyz_palm = np.array([data["palm_position_x"].values, data["palm_position_y"].values, data["palm_position_z"].values])
            xyz_index = np.array([data["index_position_x"].values, data["index_position_y"].values, data["index_position_z"].values])

            if show:
                viz = DataVisualizer3D(f'Raw {file_contact}: xyz_index')
                viz.update(np.arange(np.shape(xyz_index)[1]), xyz_index)
            
            # processing steps
            # 1/2 get rid of outliers using beginner peak acceleration from https://doi.org/10.1080/17461391.2013.775  350
            log_value_m_s2 = 1.70  # Input value: log base 10 of acceleration in m/s^2
            acceleration_limit_mm_s2 = (10 ** log_value_m_s2) * 1000
            positions_no_outliers = []
            for pos in [xyz_palm, xyz_index]:
                acceleration = calculate_acceleration_from_position(pos.T, 30)
                acc_magnitude = np.linalg.norm(acceleration, axis=1)
                indices = np.where(acc_magnitude > acceleration_limit_mm_s2)[0]
                for idx in indices:
                    pos[:, idx] = [np.nan, np.nan, np.nan]
                positions_no_outliers.append(pos)
            xyz_palm = positions_no_outliers[0]
            xyz_index = positions_no_outliers[1]
            # 2/2 get rid of (0,0,0) positions as:
            #  - it doesn't make sense from a Kinect reference frame point of view
            #  - it is most likely an overwrite from previous pre-processing
            eps = 1e-6  
            zero_rows_mask = np.all(np.abs(xyz_palm) < eps, axis=0)
            xyz_palm[:, zero_rows_mask] = np.nan
            zero_rows_mask = np.all(np.abs(xyz_index) < eps, axis=0)
            xyz_index[:, zero_rows_mask] = np.nan
            
            data["palm_position_x"] = xyz_palm[0, :]
            data["palm_position_y"] = xyz_palm[1, :]
            data["palm_position_z"] = xyz_palm[2, :]
            data["index_position_x"] = xyz_index[0, :]
            data["index_position_y"] = xyz_index[1, :]
            data["index_position_z"] = xyz_index[2, :]

            if show:
                viz_after = DataVisualizer3D(f'outliers removed\n {file_contact}: no outlier xyz_index', block_on_update=True)
                viz_after.update(np.arange(np.shape(xyz_index)[1]), xyz_index)

            # keep variables for the report
            if generate_report:
                pass

            # save data on the hard drive ?
            if save_results:
                if not os.path.exists(output_dir_abs):
                    os.makedirs(output_dir_abs)
                data.to_csv(output_filename_abs, index=False)

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

























