import itertools
import json
import math # Required for float('nan')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os
import pandas as pd
from PIL import Image
import re
import sys
import warnings
import time
from typing import Optional # Import Optional for type hinting

from matplotlib.animation import FuncAnimation
from matplotlib import __version__ as mpl_version
from packaging.version import parse as parse_version
import tempfile

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libraries.materials.semicontrolled_data import SemiControlledData  # noqa: E402
from libraries.processing.semicontrolled_data_manager import SemiControlledDataManager  # noqa: E402
from libraries.processing.semicontrolled_data_splitter import SemiControlledDataSplitter  # noqa: E402
from libraries.plot.semicontrolled_data_visualizer import SemiControlledDataVisualizer  # noqa: E402
import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.misc.waitforbuttonpress_popup import WaitForButtonPressPopup
import libraries.plot.semicontrolled_data_visualizer_unitcontact as scdata_visualizer_neur  # noqa: E402



def duplicate_lists_in_array(array_of_lists):
    output_array_of_lists = []
    current_list = np.nan
    for potential_list in array_of_lists:
        if potential_list is not np.nan:
            current_list = potential_list
        output_array_of_lists.append(current_list)
    
    return np.array(output_array_of_lists)



def extract_xyz_from_string(data_string):
    """
    Extracts a list of [X, Y, Z] float coordinates from a formatted string.

    Args:
        data_string (str): The input string containing coordinates.
                           Example: '[[-48.333332 185.       620.      ]\n [-47.925926 186.42592  616.85187 ]]'

    Returns:
        list: A list of lists, where each inner list contains three floats [X, Y, Z].
              Returns an empty list if no coordinates are found or in case of an error.
    """
    xyz_coordinates = []
    try:
        # Remove outer brackets and split by the pattern "]\n [" or "]\n["
        # This handles variations in spacing around the newline.
        # We also strip any leading/trailing whitespace from the main string first.
        cleaned_string = data_string.strip()[1:-1].strip()

        # Find all occurrences of numbers within brackets
        # This regex looks for patterns like "[ number number number ]"
        # It captures the three numbers.
        matches = re.findall(r'\[\s*(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s*\]', cleaned_string)

        for match in matches:
            # Convert the captured strings to floats
            x = float(match[0])
            y = float(match[1])
            z = float(match[2])
            xyz_coordinates.append([x, y, z])
            
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
        
    return xyz_coordinates



def format_pc_str_to_xyz(list_of_lists_str):
    xyz_contact_per_spike = []

    for list_of_str in list_of_lists_str:
        for str_val in list_of_str:
            xyz_contact_from_str = extract_xyz_from_string(str_val)
            for xyz_contact in xyz_contact_from_str:
                xyz_contact_per_spike.append(xyz_contact)

    return xyz_contact_per_spike



def generate_oriented_3d_scatter(hist_points, title_str):
    """
    Generates a 3D scatter plot from hist_points, orienting the view
    based on the normal vector of the best-fit plane to the data points.

    Args:
        hist_points (dict): A dictionary where keys are (x, y, z) tuples (coordinates)
                            and values are the corresponding color_values for the scatter plot.
        show_plot (bool): If True, displays the plot. Defaults to True.
        save_plot (bool): If True, saves the plot to a file. Defaults to False.
        figure_filename (str): The filename to use if save_plot is True.
                               Defaults to "3d_scatter_orient.png".

    Returns:
        tuple: A tuple (fig, ax) containing the matplotlib figure and axes objects if a plot
               is generated, otherwise (None, None).
    """
    # 1. Prepare data for plotting
    # Extract X, Y, Z coordinates and the values (colors)
    x_coords_list = []
    y_coords_list = []
    z_coords_list = []
    color_values_list = []

    if not hist_points:
        return None, None

    for key, value in hist_points.items():
        try:
            x, y, z = key  # Unpack the tuple key
            x_coords_list.append(x)
            y_coords_list.append(y)
            z_coords_list.append(z)
            color_values_list.append(value)
        except (TypeError, ValueError):
            print(f"Warning: Skipping invalid key in hist_points: {key}")
            continue

    # Convert to NumPy arrays for convenience with matplotlib
    x_coords = np.array(x_coords_list)
    y_coords = np.array(y_coords_list)
    z_coords = np.array(z_coords_list)
    color_values = np.array(color_values_list)

    if x_coords.size == 0: # Check if after processing, there's no valid data
        print("Warning: No valid data points extracted from hist_points. Plot will be empty or near empty.")
        # Allow to proceed to plot an empty graph with titles if show_plot/save_plot is True

    # --- 2. Calculate the Normal Vector of the Best-Fit Plane ---
    def calculate_plane_normal(x_param, y_param, z_param):
        if len(x_param) < 3:
            print("Warning: Need at least 3 points to define a plane robustly. Using default normal [0,0,1].")
            return np.array([0, 0, 1])

        points = np.vstack((x_param, y_param, z_param)).T  # Create an N_points x 3 array
        
        # Ensure there are enough unique points for SVD if len(x_param) >= 3
        if points.shape[0] < 3: # Should be caught by len(x_param) but good for robustness
             print("Warning: Less than 3 valid points after stacking for plane normal. Using default normal [0,0,1].")
             return np.array([0,0,1])
             
        # Calculate the centroid of the points for plane fitting
        centroid_plane = np.mean(points, axis=0)
        
        # Center the points (subtract the centroid)
        centered_points = points - centroid_plane
        
        # Perform Singular Value Decomposition (SVD)
        try:
            # The normal to the plane is the singular vector corresponding to the smallest singular value.
            # This is the last column of V (or last row of V.T)
            _, _, vh = np.linalg.svd(centered_points)
            plane_normal_vec = vh[-1, :] 
        except np.linalg.LinAlgError:
            print("Warning: SVD computation failed for plane normal. Using default normal [0,0,1].")
            return np.array([0,0,1])

        return plane_normal_vec

    if x_coords.size > 0: # Calculate normal vector only if there are points
        normal_vector = calculate_plane_normal(x_coords, y_coords, z_coords)
    else: # No points, use a default normal
        normal_vector = np.array([0,0,1])
        print("No data points to calculate plane normal, using default [0,0,1].")

    print(f"Calculated normal vector: {normal_vector}")

    # Optional: Ensure the normal vector points "upwards" generally
    if normal_vector[2] < 0:
        normal_vector = -normal_vector
        print(f"Flipped normal vector to point upwards: {normal_vector}")

    # --- 3. Convert Normal Vector to Matplotlib View Angles (Elevation and Azimuth) ---
    elev_angle = 30  # Default elevation
    azim_angle = -60 # Default azimuth
    quiver_length = 0.1 # Default quiver length

    norm_of_normal = np.linalg.norm(normal_vector)
    if norm_of_normal < 1e-9: # Avoid division by zero if normal is a zero vector
        print("Warning: Normal vector is close to zero. Using default view (elev=30, azim=-60).")
        # quiver_length remains its default 0.1, or 0 if no points (see below)
        if x_coords.size == 0:
            quiver_length = 0
    else:
        normalized_normal = normal_vector / norm_of_normal
        
        elev_angle = np.degrees(np.arcsin(np.clip(normalized_normal[2], -1.0, 1.0))) # Clip for safety
        azim_angle = np.degrees(np.arctan2(normalized_normal[1], normalized_normal[0]))
        
        # --- Quiver Length Calculation ---
        # Default quiver length in case of issues or single points
        # (already initialized to 0.1)

        # Check if there are any points to plot at all
        if x_coords.size > 0 and y_coords.size > 0 and z_coords.size > 0:
            # Calculate ptp for each coordinate if they have elements
            x_ptp = x_coords.ptp() if x_coords.size > 1 else 0.0
            y_ptp = y_coords.ptp() if y_coords.size > 1 else 0.0
            z_ptp = z_coords.ptp() if z_coords.size > 1 else 0.0
            
            mean_ptp = np.mean([x_ptp, y_ptp, z_ptp])
            calculated_length = mean_ptp * 0.2
            
            if calculated_length > 1e-6: # Check against a small epsilon
                quiver_length = calculated_length
            # else: quiver_length remains 0.1 (our default for single point or no spread)

        elif x_coords.size > 0 or y_coords.size > 0 or z_coords.size > 0:
            # Case: Some data exists, but perhaps not in all dimensions or very few points.
            print("Warning: Data is sparse for quiver length calculation; using default length for normal vector.")
            # quiver_length remains 0.1 (our default)
        else:
            # Case: No data points at all (x_coords, y_coords, z_coords are all empty)
            print("Warning: No data points found for quiver. Vector will not be meaningful.")
            quiver_length = 0 # Or skip quiver plotting

    print(f"Calculated view angles: Elevation = {elev_angle:.2f} deg, Azimuth = {azim_angle:.2f} deg")
    if x_coords.size > 0 : print(f"Quiver length for normal vector: {quiver_length:.3f}")


    # --- 4. Create the 3D Plot and Orient the View ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.view_init(elev=elev_angle, azim=azim_angle)

    if x_coords.size > 0: # Only attempt to scatter if there is data
        scatter = ax.scatter(x_coords, y_coords, z_coords, c=color_values, cmap='viridis', s=50)
        try:
            cbar = fig.colorbar(scatter, ax=ax, label='Total number of AP', shrink=0.5, aspect=10)
        except Exception as e:
            print(f"Could not create colorbar (possibly no data or single color): {e}")
    else:
        print("No data points to scatter plot.")


    ax.set_xlabel('X Coordinate (mm)')
    ax.set_ylabel('Y Coordinate (mm)')
    ax.set_zlabel('Z Coordinate (mm)')
    ax.set_title(f'{title_str}')

    # Optional: Plot the normal vector itself from the centroid
    if x_coords.size > 0 and quiver_length > 1e-7: # Check if there are points and quiver has a visible length
        plot_centroid = np.mean(np.vstack((x_coords, y_coords, z_coords)).T, axis=0)
        ax.quiver(plot_centroid[0], plot_centroid[1], plot_centroid[2],
                  normal_vector[0], normal_vector[1], normal_vector[2],
                  length=quiver_length,
                  color='red', label='Normal Vector', arrow_length_ratio=0.1)
        ax.legend()
    elif x_coords.size == 0:
        print("Skipping normal vector plot as no data points are available for centroid.")
    elif quiver_length <= 1e-7:
        print("Skipping normal vector plot as its calculated length is negligible or zero.")
    
    plt.tight_layout()        
    return fig, ax



def extract_ply_point_cloud(input_path: str) -> Optional[np.ndarray]:
    """
    Extracts a point cloud from a .ply file specified by the input path.
    If the path is a folder, it looks for the first .ply file found in it.

    Args:
        input_path (str): Path to a .ply file or a directory containing a .ply file.

    Returns:
        np.ndarray | None: A NumPy array of shape (N, 3) containing the point coordinates
                           if successful (can be an empty array if the PLY file has no points),
                           otherwise None if an error occurs (e.g., file not found,
                           not a PLY file, unreadable).
    """
    ply_file_path = None

    if not os.path.exists(input_path):
        print(f"Error: Path '{input_path}' does not exist.")
        return None

    if os.path.isdir(input_path):
        # print(f"Input path '{input_path}' is a directory. Searching for a .ply file...")
        found_ply = False
        try:
            for item in sorted(os.listdir(input_path)): # sorted() for deterministic behavior
                if item.lower().endswith(".ply") and os.path.isfile(os.path.join(input_path, item)):
                    ply_file_path = os.path.join(input_path, item)
                    # print(f"Found .ply file: '{ply_file_path}'")
                    found_ply = True
                    break
            if not found_ply:
                print(f"Error: No .ply file found in directory '{input_path}'.")
                return None
        except OSError as e:
            print(f"Error accessing directory '{input_path}': {e}")
            return None
            
    elif os.path.isfile(input_path):
        if input_path.lower().endswith(".ply"):
            ply_file_path = input_path
            # print(f"Input path is a .ply file: '{ply_file_path}'")
        else:
            print(f"Error: File '{input_path}' is not a .ply file.")
            return None
    else:
        print(f"Error: Path '{input_path}' is not a valid file or directory.")
        return None

    if ply_file_path:
        try:
            # print(f"Attempting to read point cloud from '{ply_file_path}'...")
            pcd = o3d.io.read_point_cloud(ply_file_path)
            
            if not pcd.has_points():
                # print(f"Warning: The file '{ply_file_path}' was loaded but contains no points.")
                return np.array([]) # Return empty array for valid PLY with no points
            
            points = np.asarray(pcd.points)
            # print(f"Successfully extracted {len(points)} points from '{ply_file_path}'.")
            return points
            
        except RuntimeError as e: # open3d often raises RuntimeError for file issues
            print(f"Error reading or processing PLY file '{ply_file_path}': {e}")
            return None
        except Exception as e: # Catch any other unexpected errors
            print(f"An unexpected error occurred with file '{ply_file_path}': {e}")
            return None
            
    return None # Should only be reached if ply_file_path was not set due to an error



def create_axis_rotation_video(
    fig, 
    ax, 
    output_filename="rotation_video.mp4", 
    fps=30, 
    dpi=300, 
    duration_sec_per_rotation=5, 
    rotation_axes=['z', 'x', 'y'],
    elev_for_z_rotation=30,
    roll_for_all_rotations=0,
    progress_bar=True
):
    """
    Generates a video of a 3D matplotlib plot rotating around specified axes.

    The rotations are performed sequentially for each axis in the 'rotation_axes' list.
    - Z-axis rotation: Azimuth changes while elevation and roll are fixed.
    - X-axis rotation: Elevation changes while azimuth is set to -90 (to make X horizontal) 
                       and roll is fixed. This simulates camera orbiting around data's X-axis.
    - Y-axis rotation: Elevation changes while azimuth is set to 0 (to make Y typically vertical)
                       and roll is fixed. This simulates camera orbiting around data's Y-axis.

    Args:
        fig (matplotlib.figure.Figure): The figure object containing the 3D plot.
        ax (matplotlib.axes.Axes3D): The 3D axes object.
        output_filename (str): Name of the output video file (e.g., "video.mp4", "video.gif").
        fps (int): Frames per second for the video.
        dpi (int): Dots per inch for the video resolution. "Full resolution" can be achieved
                   by using a high DPI like 300 or matching fig.dpi.
        duration_sec_per_rotation (float): Duration in seconds for one full 360-degree rotation 
                                           around a single axis.
        rotation_axes (list): A list of strings specifying the axes to rotate around and 
                              their order. Valid options: 'z', 'x', 'y'.
                              Example: ['z', 'x'] will first rotate around Z, then X.
        elev_for_z_rotation (float): The fixed elevation (degrees) to use during Z-axis rotation.
        roll_for_all_rotations (float): The fixed roll (degrees) to use for all rotations.
                                       Requires Matplotlib 3.6+ for 'roll' in view_init.
        progress_bar (bool): Whether to attempt to show a progress bar during saving (experimental,
                             depends on matplotlib writer and backend).

    Returns:
        None: The function saves the video to the specified 'output_filename'.
    
    Raises:
        ValueError: If an invalid axis is specified in rotation_axes.
        RuntimeError: If video saving fails (e.g., ffmpeg not found).
    """

    if not rotation_axes:
        print("No rotation axes specified. Video not generated.")
        return

    # Store initial view to restore later
    initial_elev = ax.elev
    initial_azim = ax.azim
    try:
        initial_roll = ax.roll if hasattr(ax, 'roll') else 0
    except Exception: # Some matplotlib versions might have issue accessing it even if present
        initial_roll = 0


    # Check Matplotlib version for roll support in view_init
    use_roll_param = parse_version(mpl_version) >= parse_version("3.6")
    if not use_roll_param and roll_for_all_rotations != 0:
        print(f"Warning: Matplotlib version {mpl_version} < 3.6. 'roll' parameter will be ignored.")

    num_frames_per_rot = int(duration_sec_per_rotation * fps)
    if num_frames_per_rot <= 0:
        print("Warning: duration_sec_per_rotation or fps is too low, results in <=0 frames per rotation. Video not generated.")
        return
        
    total_num_frames = len(rotation_axes) * num_frames_per_rot

    # Define standard views for each rotation type
    # Angles for 'elev' and 'azim' in view_init are in degrees.
    rotation_configs = {
        'z': {'param_to_vary': 'azim', 'fixed_elev': elev_for_z_rotation, 'fixed_azim': None, 'fixed_roll': roll_for_all_rotations},
        'x': {'param_to_vary': 'elev', 'fixed_elev': None, 'fixed_azim': -90, 'fixed_roll': roll_for_all_rotations}, # Azim=-90 makes X horizontal
        'y': {'param_to_vary': 'elev', 'fixed_elev': None, 'fixed_azim': 0,   'fixed_roll': roll_for_all_rotations}, # Azim=0 makes Y typically vertical
    }

    # Validate rotation_axes
    for axis_key in rotation_axes:
        if axis_key not in rotation_configs:
            raise ValueError(f"Invalid axis '{axis_key}' in rotation_axes. Valid are 'x', 'y', 'z'.")

    # Update function for FuncAnimation
    def update_view(frame_idx):
        current_rotation_idx = frame_idx // num_frames_per_rot
        
        # Ensure we don't go out of bounds if total_num_frames is miscalculated (should not happen)
        if current_rotation_idx >= len(rotation_axes):
            return [ax] 

        axis_key = rotation_axes[current_rotation_idx]
        config = rotation_configs[axis_key]

        frame_in_current_rot = frame_idx % num_frames_per_rot
        angle_degrees = (frame_in_current_rot / num_frames_per_rot) * 360.0

        view_params = {}
        if config['param_to_vary'] == 'azim':
            view_params['elev'] = config['fixed_elev']
            view_params['azim'] = angle_degrees
            if use_roll_param:
                view_params['roll'] = config['fixed_roll']
        elif config['param_to_vary'] == 'elev':
            view_params['elev'] = angle_degrees
            view_params['azim'] = config['fixed_azim']
            if use_roll_param:
                view_params['roll'] = config['fixed_roll']
        
        ax.view_init(**view_params)
        
        # Update a title to show progress (optional, can be slow)
        # current_title = f"Rotating around {axis_key.upper()}-axis: {angle_degrees:.1f}°"
        # ax.set_title(current_title) # This can make generation slower
        
        return [ax] # Return artists that have been modified

    print(f"Preparing to generate video with {total_num_frames} frames...")
    print(f"Each of {len(rotation_axes)} rotation(s) will have {num_frames_per_rot} frames.")

    # Create the animation object
    # blit=False is often necessary for 3D plots or complex updates.
    ani = FuncAnimation(fig, update_view, frames=total_num_frames, blit=False, interval=1000/fps)

    # Save the animation
    # Try to use imageio with ffmpeg if available, or matplotlib's ffmpeg writer.
    writer = None
    if output_filename.endswith(".gif"):
        try:
            writer = plt.matplotlib.animation.ImageMagickWriter(fps=fps)
            # Or 'imageio' if preferred for gifs
            # import imageio
            # writer = imageio.get_writer(output_filename, mode='I', fps=fps) # this is not a mpl writer
        except Exception:
            print("ImageMagickWriter not found for GIF, ensure ImageMagick is installed or try another format like .mp4.")
            # Fallback or error
    else: # For .mp4 or other ffmpeg formats
        try:
            # Prefer FFMpegWriter if explicitly available and configured
            if plt.rcParams['animation.ffmpeg_path'] != 'ffmpeg': # If user configured a path
                 writer = plt.matplotlib.animation.FFMpegWriter(fps=fps, codec='libx264')
            else: # Try to find ffmpeg automatically
                # This might pick up imageio's ffmpeg if imageio-ffmpeg is installed
                plt.rcParams['animation.writer'] = 'ffmpeg' 
                writer = plt.matplotlib.animation.writers['ffmpeg'](fps=fps, codec='libx264')
        except Exception as e_ffmpeg:
            print(f"Matplotlib FFMpegWriter setup failed: {e_ffmpeg}. Ensure ffmpeg is installed and in PATH, or imageio-ffmpeg is installed.")
            print("Attempting to use 'imageio' writer as a fallback for video (requires imageio and imageio-ffmpeg).")
    
    progress_callback_func = None
    if progress_bar:
        def _progress(current_frame, total_frames_progress):
            print(f'Saving frame {current_frame + 1} of {total_frames_progress}')
        progress_callback_func = _progress

    try:
        print(f"Saving animation to {output_filename} (this may take a while)...")
        if writer:
            ani.save(output_filename, writer=writer, dpi=dpi, progress_callback=progress_callback_func)
        else: # Let ani.save try to find a writer based on rcParams and file extension
            ani.save(output_filename, fps=fps, dpi=dpi, progress_callback=progress_callback_func)
        print(f"Video successfully saved to {output_filename}")
    except RuntimeError as e:
        print(f"RuntimeError during saving: {e}")
        print("This often means a suitable writer (like ffmpeg for .mp4) was not found or failed.")
        print("Please ensure ffmpeg is installed and in your system's PATH, or that imageio and imageio-ffmpeg are installed if your matplotlib version uses them as a backend.")
    except Exception as e:
        print(f"An unexpected error occurred during video saving: {e}")
    finally:
        # Restore initial view
        final_restore_params = {'elev': initial_elev, 'azim': initial_azim}
        if use_roll_param:
             final_restore_params['roll'] = initial_roll
        ax.view_init(**final_restore_params)
        # ax.set_title(initial_title) # If you were changing titles
        fig.canvas.draw_idle() # Redraw with the original view
        print("Original plot view restored.")




if __name__ == "__main__":
    # ----------------------
    # User control variables
    # ----------------------
    force_processing = False  # If user wants to force data processing even if results already exist
    save_figures = True
    save_results = True

    input_filename_pattern_r = r"semicontrolled_block-order(0[1-9]|1[0-8])_trial(0[1-9]|1[0-9]).csv"
    output_filename_end = f"_semicontrolled_RF-map.json"

    show = False  # If user wants to monitor what's happening
    show_single_touches = False  # If user wants to visualise single touches, one by one
    manual_check = False  # If user wants to take the time to check the trial and how it has been split
    show_final_summary = False

    # ----------------------
    # ----------------------
    # ----------------------

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")

    # get metadata paths
    md_stimuli_path = os.path.join(db_path, "1_primary", "logs", "2_stimuli_by_blocks")
    md_neuron_filename_abs = os.path.join(db_path, "1_primary", "nerve", "semicontrol_unit-name_to_unit-type.csv")
    # 2. check if neuron metadata file exists
    if not os.path.exists(md_neuron_filename_abs):
        s = f'The file {md_neuron_filename_abs} doesn''t  exist.'
        warnings.warn(s, Warning)
    
    # get input data directory
    db_path_input = os.path.join(db_path, "3_merged", "5.1.2_sorted_by_trial_lag_corrected")
    db_path_pcarm_base = os.path.join(db_path, "1_primary", "kinect")

    # get output directories
    common_output_folder_base = "5.2.0_RF_XYZ-PC_vectors"
    db_path_output = os.path.join(db_path, "3_merged", common_output_folder_base)
    output_figure_path = os.path.join(db_path, "3_merged", common_output_folder_base)
    if save_results and not os.path.exists(db_path_output):
        os.makedirs(db_path_output)
    if save_figures and not os.path.exists(output_figure_path):
        os.makedirs(output_figure_path)

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

    use_specific_blocks = False
    specific_blocks = ['block-order08', 'block-order09']

    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:
        curr_dir = os.path.join(db_path_input, session)
        files_abs, files = path_tools.find_files_in_directory(curr_dir, ending=input_filename_pattern_r)

        # Output
        filename_output = session + output_filename_end
        filename_output_abs = os.path.join(db_path_output, filename_output)
        if not force_processing and os.path.exists(filename_output_abs):
            s = f'The file {filename_output_abs} exists.'
            print(s, Warning)
            continue
        filename_figure_output = session + output_filename_end.replace(".json", ".mp4")
        filename_figure_output_abs = os.path.join(db_path_output, filename_figure_output)
        
        pc_during_spikes_neuron = []
        for data_filename_abs, data_filename in zip(files_abs, files):
            print(f"\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
            print(f"current file: {data_filename}")
            if use_specific_blocks:
                is_not_specific_block = True
                for block in specific_blocks:
                    if block in data_filename:
                        is_not_specific_block = False
                if is_not_specific_block:
                    continue
            
            # 1. extract metadata related to the current stimulus set and check if exists
            md_stimuli_filename = re.sub(r'_trial\d{2}\.csv', '_stimuli.csv', data_filename)
            md_stimuli_filename_abs = os.path.join(md_stimuli_path, session, md_stimuli_filename)
            if not os.path.exists(md_stimuli_filename_abs):
                s = f'The file {md_stimuli_filename_abs} doesn''t exist.'
                warnings.warn(s, Warning)
                continue

            scd = SemiControlledData(data_filename_abs, md_stimuli_filename_abs, md_neuron_filename_abs)  # resources
            scd.set_variables(dropna=False)
            if len(scd.contact.arm_pointcloud) == 0:
                continue

            # get time value when a spike occured
            index_with_spike = np.where(scd.neural.spike == 1)[0]

            # get contact_pointcloud with duplicate XYZ list between list ("flat interpolation") 
            # to make sure that there is a XYZ list (30Hz) when there is a spike (1kHz)
            pc_with_repeated_values = duplicate_lists_in_array(scd.contact.arm_pointcloud)
            
            # get contact pointcloud when spikes occur
            pc_contact = pc_with_repeated_values[index_with_spike]
            # /!\ if contact and neural data were well aligned, that should not occur, but:
            # remove nan and empty lists.
            df_filtered = pd.Series(pc_contact)
            #  1. remove the nan values from the filtered pc.
            df_filtered = df_filtered[~(df_filtered.isna())]
            #  2. remove empty lists.
            df_filtered = df_filtered[df_filtered.apply(lambda x: x != '[]')]
            # 3. store the extracted data
            pc_during_spikes_neuron.append(df_filtered.to_list())

        xyz_pc_per_spike = format_pc_str_to_xyz(pc_during_spikes_neuron)
        hist_points = {}
        for xyz_point in xyz_pc_per_spike:
            xyz_key = tuple(xyz_point)
            # Example: Adding the key for the first time or incrementing
            if xyz_key in hist_points:
                hist_points[xyz_key] += 1
                print(f"Key {xyz_key} existed. Incremented value to: {hist_points[xyz_key]}")
            else:
                # If you want to initialize it if it doesn't exist (e.g., start count at 1)
                hist_points[xyz_key] = 1
                print(f"Key {xyz_key} did not exist. Set value to: {hist_points[xyz_key]}")

        pcarm_points = extract_ply_point_cloud(os.path.join(db_path_pcarm_base, session))
        for pcarm_point in pcarm_points:
            pcarm_point_key = tuple(pcarm_point)
            if not (pcarm_point_key in hist_points):
                # If you want to initialize it if it doesn't exist (e.g., start count at 1)
                hist_points[pcarm_point_key] = 0
                print(f"Key {pcarm_point_key} did not exist. Set value to: {hist_points[pcarm_point_key]}")


        if show or save_figures:
            fig, ax = generate_oriented_3d_scatter(hist_points, filename_figure_output)
            if save_figures:
                create_axis_rotation_video(
                    fig, ax,
                    output_filename=filename_figure_output_abs,
                    fps=30,
                    dpi=300, # For "full resolution", adjust as needed
                    duration_sec_per_rotation=4, # Each axis rotation will take 4 seconds
                    rotation_axes=['z', 'x', 'y'], # Rotate around Z, then X, then Y
                    elev_for_z_rotation=25, # Elevation for Z-axis spin
                    roll_for_all_rotations=0, # Roll angle for all rotations
                    progress_bar=True 
                )
            if show:
                plt.show(block=True)
            plt.close(fig)
            del fig
            del ax

        if save_results:
            if not os.path.exists(output_dir_abs):
                os.mkdir(output_dir_abs)
            # --- Step 1: Prepare dictionary for JSON (convert tuple keys to strings) ---
            # We'll use str(tuple_key) which gives a string like "(-37.975609, 161.7317, 651.92682)"
            dict_for_json = {str(key): value for key, value in hist_points.items()}

            try:
                with open(filename_output_abs, 'w') as file: # 'w' means write in text mode
                    json.dump(dict_for_json, file, indent=4) # indent=4 makes the JSON file human-readable
                print(f"Dictionary successfully saved to {filename_output_abs}")
            except IOError as e:
                print(f"Error saving dictionary to JSON: {e}")

    print("done.")
