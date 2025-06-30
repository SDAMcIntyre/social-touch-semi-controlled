# Standard library imports
import os
import sys
import json
import csv
from tkinter import Toplevel, ttk
import tkinter as tk

# Third-party library imports
import cv2
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Local application/library specific imports
# (Assuming 'imported' and 'libraries' are local directories)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import imported.config as config
from imported.hand_mesh import HandMesh
from imported.kinematics import mpii_to_mano
from imported.utils import imresize
import libraries.misc.path_tools as path_tools
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



def adjust_view_parameters_interactively(reference_pcd: o3d.geometry.PointCloud):
    """
    Opens an Open3D window for the user to adjust the camera angle.
    The captured view parameters, along with window dimensions, will be returned
    for subsequent use and saving.
    """ 
    # Create a temporary root to get screen info without showing a window
    temp_root = tk.Tk()
    temp_root.withdraw() # Hide the window
    screen_width = temp_root.winfo_screenwidth()
    screen_height = temp_root.winfo_screenheight()
    temp_root.destroy()
    print(f"Detected screen resolution: {screen_width}x{screen_height}")

    # Store window dimensions to be returned later
    window_width = screen_width // 2
    window_height = screen_height

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Adjust 3D View (Right Half)", width=window_width, height=window_height)
    vis.add_geometry(reference_pcd)
    
    view_control = vis.get_view_control()
    view_params = view_control.convert_to_pinhole_camera_parameters()

    global tk_root
    tk_root = tk.Tk()
    tk_root.title("Controls")
    tk_root.geometry("300x100")
    main_frame = ttk.Frame(tk_root, padding="10")
    main_frame.pack(expand=True, fill='both')
    label = ttk.Label(main_frame, text="Adjust the 3D view, then click Proceed.")
    label.pack(pady=5)
    
    def _capture_view_parameters_and_proceed():
        """Closes the Tkinter window and signals the main process to proceed."""
        global tk_root
        if tk_root:
            tk_root.quit()
            tk_root.destroy()
            tk_root = None
    proceed_button = ttk.Button(main_frame, text="Proceed", command=_capture_view_parameters_and_proceed)
    proceed_button.pack(pady=10, fill='x', expand=True)

    print("Please adjust the view in the 'Adjust 3D View (Right Half)' window.")
    print("Click 'Proceed' in the 'Controls' window to continue.")
    
    # Main loop for interactive adjustment
    while tk_root is not None:
        vis.poll_events()
        vis.update_renderer()
        try:
            tk_root.update()
        except tk.TclError:
            # Tkinter window was likely closed by user or system
            break
    
    view_control = vis.get_view_control()
    view_params = view_control.convert_to_pinhole_camera_parameters()
    vis.destroy_window()
    
    if view_params:
        print("View parameters captured successfully.")
    else:
        print("View adjustment cancelled or failed. Default view will be used for video generation.")

    # Return captured parameters and the window dimensions
    return view_params, window_width, window_height


if __name__ == "__main__":
    force_processing = False  # If user wants to force data processing even if results already exist
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
            output_path_abs = curr_session_path.replace(db_path_input, db_path_output)
            output_filename_abs = os.path.join(output_path_abs, "3d_view_param.json")
            if not force_processing and os.path.exists(output_filename_abs):
                continue
            
            block_dir_abs = os.path.dirname(config_file_abs)
            files_path, _ = load_variables_from_json(config_file_abs)

            # --- 2/2 forearm point cloud Creation ---
            arm_ply_fname_abs = files_path["arm_ply_path"].replace("<session_dir>", curr_session_path) # if necessary, create the long-path-aware version for the Windows API: "\\\\?\\"+    
            arm_pcd = o3d.io.read_point_cloud(arm_ply_fname_abs)
            arm_pcd.estimate_normals()
            arm_pcd.orient_normals_to_align_with_direction(([0., 0., -1.]))
            
            # Get view parameters and window dimensions from the interactive function
            view_params, window_width, window_height = adjust_view_parameters_interactively(arm_pcd)

            if save_results:
                # Save the camera view parameters to its JSON file
                o3d.io.write_pinhole_camera_parameters(output_filename_abs, view_params)
                print(f"Camera view parameters saved to {output_filename_abs}")

                # --- NEW: Save window dimensions to a separate JSON file ---
                # Define the path for the dimensions file in the same directory
                output_dir = os.path.dirname(output_filename_abs)
                dimensions_filename_abs = os.path.join(output_dir, "3d_view_param_window_dimensions.json")
                
                # Prepare the data to be saved
                dimensions_data = {
                    "width": window_width,
                    "height": window_height
                }

                # Write the dictionary to the JSON file
                with open(dimensions_filename_abs, 'w') as f:
                    json.dump(dimensions_data, f, indent=4)
                print(f"Window dimensions saved to {dimensions_filename_abs}")
                # --- END NEW ---