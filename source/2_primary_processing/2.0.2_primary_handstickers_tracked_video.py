import open3d as o3d
import csv
import json
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np
import os
import sys
import pandas as pd 
import cv2
import numpy as np

import tkinter as tk
from tkinter import ttk
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tkinter as tk
from tkinter import Toplevel


# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from imported.hand_mesh import HandMesh
from imported.utils import imresize  # OneEuroFilter, imresize
from imported.kinematics import mpii_to_mano
# from imported.wrappers import ModelPipeline
import imported.config as config

import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.primary.StickersVideoGenerator import StickersVideoGenerator


def load_variables_from_json(json_file_path):
    # Load the JSON file
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Extract variables from the JSON data
    files_path = {
        "video_fname": data["video_fname"],
        "sticker_fname": data["sticker_fname"],
        "arm_ply_path": data["arm-ply_rel-path"],
        "arm_normal_path": data["arm-normal-vector_rel-path"],
        "handmesh_model_path": data["hand-mesh-model_rel-path"]
        }

    parameters = data["parameters"]
    parameters["hand_mesh"]["hardcoded_shift"]["general"] = np.array(parameters["hand_mesh"]["hardcoded_shift"]["general"])
    parameters["hand_mesh"]["hardcoded_shift"]["specific_time_sections"] = [
        [np.array(shift[0]), np.array(shift[1])] for shift in parameters["hand_mesh"]["hardcoded_shift"]["specific_time_sections"]
    ]
    parameters["viewer"]["matrix_rotation_viewer"] = np.array(parameters["viewer"]["matrix_rotation_viewer"])

    return files_path, parameters


import cv2
import numpy as np

def video_to_rgb_frame_array(video_path: str) -> np.ndarray:
    """
    Reads a video file and returns a single NumPy array containing all frames
    as 2D RGB images, while displaying the processing progress.

    Args:
        video_path: The full path to the video file.

    Returns:
        A NumPy array with a shape of (num_frames, height, width, 3),
        where each element along the first axis is a single RGB frame.
        Returns an empty array if the video cannot be opened or read.
    """
    # Open the video file with OpenCV
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        # Return an empty NumPy array with a valid but empty shape
        return np.array([])

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle case where video has no frames
    if total_frames == 0:
        print(f"Warning: Video at {video_path} has no frames or metadata is unreadable.")
        cap.release()
        return np.array([])
        
    processed_frames = 0

    frames = []
    while True:
        # Read one frame from the video
        ret, frame = cap.read()

        # If 'ret' is False, it means there are no more frames to read,
        # so we break the loop
        if not ret:
            break

        # OpenCV reads frames in BGR format by default.
        # Convert the frame from BGR to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Add the RGB frame to our list of frames
        frames.append(rgb_frame)

        # Increment the counter for processed frames
        processed_frames += 1
        # Calculate and display the progress percentage
        progress = (processed_frames / total_frames) * 100
        # Use \r to move the cursor to the beginning of the line, and end='' to prevent a new line
        print(f"\rProcessing... {progress:.2f}% complete", end="")
    # Print a newline character at the end to move to the next line after the loop
    print()

    # Release the video capture object to free up resources
    cap.release()
    
    # Convert the list of frames into a single NumPy array and return it
    return np.array(frames)



def interpolate_single_point(data: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    Transforms an (N, 3) array by replacing (0,0,0) rows with NaNs and
    then interpolates these NaN values along the time axis.

    Args:
        data (np.ndarray): The input array with shape (N, 3).
                           Represents N time samples for a single point's
                           (x, y, z) coordinates.
        method (str): The interpolation method to use.
                      Examples: 'linear', 'quadratic', 'cubic'.
                      Passed directly to pandas.DataFrame.interpolate.

    Returns:
        np.ndarray: The processed array with shape (N, 3) where
                    (0,0,0) rows have been interpolated.
    """
    # Validate the input shape
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError("Input array must have shape (N, 3)")

    processed_data = data.copy().astype(float) # Use float for NaN compatibility

    # Step 1: Replace (0,0,0) rows with (NaN, NaN, NaN)
    # Create a boolean mask for time samples where the point is (0,0,0)
    zero_mask = np.all(processed_data == 0, axis=1)
    
    # Use the mask to set these rows to NaN
    processed_data[zero_mask] = np.nan

    # Step 2: Interpolate the NaN values using pandas
    # Create a DataFrame; columns represent x, y, and z time-series
    df = pd.DataFrame(processed_data, columns=['x', 'y', 'z'])

    # Interpolate the data.
    # limit_direction='both' fills NaNs at the start/end of the series
    # by extending the first/last valid observation.
    df_interpolated = df.interpolate(method=method, axis=0, limit_direction='both')

    # Step 3: Convert the interpolated DataFrame back to a NumPy array
    final_output = df_interpolated.to_numpy()

    return final_output



def show_stickers_path(stickers_pos):
    """
    Visualizes the 3D trajectories of a specified number of points over a
    given number of samples.

    Args:
        corrected_joints_color (np.array): A numpy array with shape (n, p*3)
                                           containing the coordinates of the points.
        n (int): The number of samples (timesteps).
        p (int): The number of points to plot.
    """
    stickers_pos = np.array(stickers_pos)

    # If the input is 2D (N, 3), reshape it to (N, 1, 3) to treat it as a single point.
    if stickers_pos.ndim == 2:
        if stickers_pos.shape[1] != 3:
            raise ValueError("Input array with 2 dimensions must have shape (N, 3).")
        stickers_pos = stickers_pos.reshape(stickers_pos.shape[0], 1, 3)
    elif stickers_pos.ndim != 3 or stickers_pos.shape[2] != 3:
        raise ValueError("Input array must have shape (N, 3) or (N, P, 3).")

    # Get the dimensions N (samples) and P (points)
    (n, p, _) = stickers_pos.shape

    # --- Dynamic Plotting Definitions ---
    # Generate a list of markers. If p > len(available_markers), it will cycle through them.
    available_markers = ['o', '^', 's', 'p', '*', '+', 'x', 'D', 'v', '<', '>']
    markers = [available_markers[i % len(available_markers)] for i in range(p)]

    # Generate a list of linestyles. Cycles if p is large.
    available_linestyles = ['-', '--', ':', '-.']
    linestyles = [available_linestyles[i % len(available_linestyles)] for i in range(p)]

    # Generate a colormap for the lines to ensure each has a distinct color.
    line_colors = cm.jet(np.linspace(0, 1, p))


    # --- Plotting Setup ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Create a color map for the scatter points to show the progression of time
    scatter_colors = cm.viridis(np.linspace(0, 1, n))

    # --- Plot Trajectories and Points ---
    for j in range(p):  # Iterate through each of the 'p' points
        # Extract the trajectory (path) of the point over all 'n' samples
        x_path = stickers_pos[:, j, 0]
        y_path = stickers_pos[:, j, 1]
        z_path = stickers_pos[:, j, 2]

        # Plot the trajectory line for the current point
        ax.plot(x_path, y_path, z_path,
                color=line_colors[j],
                linestyle=linestyles[j],
                label=f'Point {j+1} Trajectory')

        # Plot the individual points along the trajectory
        for i in range(n):  # Iterate through each sample to plot the point
            ax.scatter(x_path[i], y_path[i], z_path[i],
                       c=[scatter_colors[i]], # Color by sample index (time)
                       marker=markers[j],
                       s=100,
                       edgecolor=line_colors[j], # Edge color matches the trajectory line
                       linewidth=1.5)

    # --- Labels, Title, and Legend ---
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title(f'3D Trajectories of {p} Points Over Time')

    # Add a color bar to indicate time/sample for the scatter points
    mappable = cm.ScalarMappable(cmap=cm.viridis)
    mappable.set_array(np.arange(n))
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, aspect=8)
    cbar.set_label('Sample Index (Time)')

    # Create a legend for the trajectories
    ax.legend(title="Stickers")

    # Display the plot
    plt.show(block=True)



def generate_video(files_path, curr_session_path, block_dir_abs, save_file_path):
    video_fname_abs = os.path.join(block_dir_abs, files_path["video_fname"])
    
    # 1/2 get the XYZ location of the sticker from the Kinect RGB+depth
    sticker_fname_abs = os.path.join(block_dir_abs, files_path["sticker_fname"])
    corrected_joints_color = np.genfromtxt(sticker_fname_abs, delimiter=',')
    nsample = corrected_joints_color.shape[0]
    stickers_pos = corrected_joints_color.reshape(nsample, 3, 3)    
    sticker_red_ps = interpolate_single_point(stickers_pos[:, 0, :], method='quadratic')
    sticker_green_ps = interpolate_single_point(stickers_pos[:, 1, :], method='quadratic')
    sticker_yellow_ps = interpolate_single_point(stickers_pos[:, 2, :], method='quadratic')

    stickers_pos = [sticker_red_ps, sticker_green_ps, sticker_yellow_ps]
    # rearrange into (nsample, XYZ, points)
    stickers_pos = np.asarray(stickers_pos).transpose(1, 2, 0)

    # --- 2/2 forearm point cloud Creation ---
    arm_ply_fname_abs = files_path["arm_ply_path"].replace("<session_dir>", curr_session_path) # if necessary, create the long-path-aware version for the Windows API: "\\\\?\\"+    
    arm_pcd = o3d.io.read_point_cloud(arm_ply_fname_abs)
    arm_pcd.estimate_normals()
    arm_pcd.orient_normals_to_align_with_direction(([0., 0., -1.]))

    # Pass the mesh and the desired save path to the transformer
    video_generator = StickersVideoGenerator(time_series_xyz=stickers_pos,
                                             images=video_fname_abs,
                                             reference_pcd=arm_pcd,
                                             colors=["red", "green", "yellow"],
                                             fps=30,
                                             save_path=save_file_path)
    
    view_param_filename_abs = os.path.join(curr_session_path, "3d_view_param.json")
    view_param_window_filename_abs = os.path.join(curr_session_path, "3d_view_param_window_dimensions.json")
    if os.path.exists(view_param_filename_abs) and os.path.exists(view_param_window_filename_abs):
        video_generator.view_params = o3d.io.read_pinhole_camera_parameters(view_param_filename_abs)
        
        with open(view_param_window_filename_abs, 'r') as f:
            data = json.load(f)
        w_width = data.get('width')
        w_height = data.get('height')
    else:
        video_generator.adjust_view_parameters_interactively()
        o3d.io.write_pinhole_camera_parameters(view_param_filename_abs, video_generator.view_params)
        w_witdh = None
        w_height = None

    video_generator.create_video(width3d=w_width, height3d=w_height)


if __name__ == "__main__":
    force_processing = True  # If user wants to force data processing even if results already exist
    save_results = True
    generate_report = True

    show = True  # If user wants to monitor what's happening

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    OS_linux = sys.platform.startswith('linux')
    if OS_linux:
        db_path = os.path.expanduser('~/Documents/datasets/semi-controlled')
    else:
        db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")

    # get input base directory
    db_path_input = os.path.join(db_path, "1_primary", "kinect")
    db_path_handmesh = os.path.join(db_path, "handmesh_models")

    # get output base directory
    db_path_output = os.path.join(db_path, "1_primary", "kinect")
    if save_results and not os.path.exists(db_path_output):
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
    

    use_specific_sessions = False
    if not use_specific_sessions:
        sessions = []
        sessions = sessions + sessions_ST13
        sessions = sessions + sessions_ST14
        sessions = sessions + sessions_ST15
        sessions = sessions + sessions_ST16
        sessions = sessions + sessions_ST18
    else:
        sessions = ['2022-06-15_ST14-02']
    
    use_specific_blocks = False
    specific_blocks = ['block-order-01']

    
    print(sessions)

    diff_ms_all = []
    names_contact = []
    names_led = []
    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:
        curr_session_path = os.path.join(db_path_input, session)
        config_files_abs, config_files = path_tools.find_files_in_directory(curr_session_path, ending='_extraction-config.json')
        print(config_files_abs)

        for config_file_abs, config_file in zip(config_files_abs, config_files):
            if use_specific_blocks:
                is_not_specific_block = True
                for block in specific_blocks:
                    if block in config_file_abs:
                        is_not_specific_block = False
                if is_not_specific_block:
                    continue

            block_dir_abs = os.path.dirname(config_file_abs)
            print(f"----------- {block_dir_abs} -----------")
            output_path_abs = block_dir_abs.replace(db_path_input, db_path_output)
            output_filename_abs = os.path.join(output_path_abs, "handstickers_tracking.mp4")
            if not force_processing and os.path.exists(output_filename_abs):
                continue
            
            block_dir_abs = os.path.dirname(config_file_abs)
            files_path, parameters = load_variables_from_json(config_file_abs)

            generate_video(files_path, curr_session_path, block_dir_abs,
                           output_filename_abs)
