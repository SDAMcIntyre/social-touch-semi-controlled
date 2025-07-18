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

import tkinter as tk
from tkinter import ttk
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import tkinter as tk
from tkinter import Toplevel


# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from imported.hand_mesh import HandMesh
from imported.utils import imresize  # OneEuroFilter, imresize
from imported.kinematics import mpii_to_mano
# from imported.wrappers import ModelPipeline
import imported.config as config

import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.primary.VideoFrameSelector import VideoFrameSelector
from libraries.primary.HandMeshTransformer import HandMeshTransformer


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



def select_frame(video_path):
    
    # Create a dummy root window to use the file dialog
    root = tk.Tk()
    root.withdraw() # Hide the dummy window

    # Create the main application window as a Toplevel window
    app_window = Toplevel(root)
    app = VideoFrameSelector(app_window, video_path)
    
    # Start the Tkinter event loop. This will block until the window is closed.
    app_window.mainloop()
    
    # After the loop ends, the main function continues
    root.destroy()

    # Print the selected frame number, if one was saved
    if app.selected_frame_id is not None:
        print("-" * 30)
        print(f"Process finished. Selected Frame ID: {app.selected_frame_id}")
        print("-" * 30)
    else:
        print("Process finished. No frame was saved.")

    return app.selected_frame_id, app.selected_frame_image



def print_model_info(handmesh, arm_pcd):

    # Convert to NumPy arrays for efficient computation
    hand_vertices = np.asarray(handmesh.vertices)
    forearm_points = np.asarray(arm_pcd.points)

    # --- Handmesh Information ---
    print("--- Handmesh Information ---")

    # Calculate min, max, and mean for the handmesh
    hand_min_bounds = np.min(hand_vertices, axis=0)
    hand_max_bounds = np.max(hand_vertices, axis=0)
    hand_mean = np.mean(hand_vertices, axis=0)

    # Print the span and mean for each axis of the handmesh
    print(f"Handmesh X-axis span: [{hand_min_bounds[0]:.4f}, {hand_max_bounds[0]:.4f}] | Mean: {hand_mean[0]:.4f}")
    print(f"Handmesh Y-axis span: [{hand_min_bounds[1]:.4f}, {hand_max_bounds[1]:.4f}] | Mean: {hand_mean[1]:.4f}")
    print(f"Handmesh Z-axis span: [{hand_min_bounds[2]:.4f}, {hand_max_bounds[2]:.4f}] | Mean: {hand_mean[2]:.4f}")
    print(f"Overall center of the handmesh: {hand_mean}\n")


    # --- Forearm Information ---
    print("--- Forearm Information ---")

    # Calculate min, max, and mean for the forearm point cloud
    forearm_min_bounds = np.min(forearm_points, axis=0)
    forearm_max_bounds = np.max(forearm_points, axis=0)
    forearm_mean = np.mean(forearm_points, axis=0)

    # Print the span and mean for each axis of the forearm
    print(f"Forearm X-axis span: [{forearm_min_bounds[0]:.4f}, {forearm_max_bounds[0]:.4f}] | Mean: {forearm_mean[0]:.4f}")
    print(f"Forearm Y-axis span: [{forearm_min_bounds[1]:.4f}, {forearm_max_bounds[1]:.4f}] | Mean: {forearm_mean[1]:.4f}")
    print(f"Forearm Z-axis span: [{forearm_min_bounds[2]:.4f}, {forearm_max_bounds[2]:.4f}] | Mean: {forearm_mean[2]:.4f}")
    print(f"Overall center of the forearm: {forearm_mean}")



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



def define_handmesh_transformation(frame_idx, rgb_frame, 
                                   files_path, parameters, 
                                   curr_session_path, block_dir_abs, db_path_handmesh, 
                                   save_file_path):
    
    # --- 1/2 hand Mesh Creation ---
    # -- load initial models
    handmesh_path = files_path["handmesh_model_path"].replace("<handmesh_models_dir>", db_path_handmesh)
    v_handMesh = np.loadtxt(handmesh_path)
    v_handMesh *= 1000  # 
    hand_mesh_model = HandMesh(config.HAND_MESH_MODEL_PATH)
    t_handMesh = np.asarray(hand_mesh_model.faces)
    # swap vertices and faces if right hand used for the experiment (as loaded model is a left hand)
    if parameters["hand_used"]['left'] == False:
        # 1. Flip the vertices
        v_handMesh[:, 0] *= -1
        # 2. Flip the triangle winding order to correct the normals
        triangles = np.asarray(t_handMesh)
        # Swaps the second and third vertex of each triangle (e.g., [0,1,2] -> [0,2,1])
        t_handMesh = triangles[:, [0, 2, 1]]
    
    # -- position the hand mesh based on the selected sticker and its XYZ position in the determined frame
    # 1/2 get the expected location of the sticker on the 3d model
    palm_vertex_idx = np.where((-0.045 < hand_mesh_model.verts[:,0]) & (hand_mesh_model.verts[:,0] < -0.035)   # x value of vertices
                                      &( 0    < hand_mesh_model.verts[:,1])                                     # y value of vertices
                                      &(-0.01 < hand_mesh_model.verts[:,2]) & (hand_mesh_model.verts[:,2] < 0.01))    # z value of vertices
    index_vertex_idx = np.where((0.065 < hand_mesh_model.verts[:,0]) & (0.02 < hand_mesh_model.verts[:,2]))  # index finger tip
    # store average initial position of the forefinger nail and the center of the palm
    fingernail_p = np.mean(v_handMesh[index_vertex_idx[0],:], axis=0)
    palm_p = np.mean(v_handMesh[palm_vertex_idx[0],:], axis=0)
    # 2/2 get the XYZ location of the sticker from the Kinect RGB+depth
    sticker_fname_abs = os.path.join(block_dir_abs, files_path["sticker_fname"])
    corrected_joints_color = np.genfromtxt(sticker_fname_abs, delimiter=',')
    nsample = corrected_joints_color.shape[0]
    stickers_pos = corrected_joints_color.reshape(nsample, 3, 3)    
    sticker_red_ps = interpolate_single_point(stickers_pos[:, 0, :], method='quadratic')
    sticker_green_ps = interpolate_single_point(stickers_pos[:, 1, :], method='quadratic')
    sticker_yellow_ps = interpolate_single_point(stickers_pos[:, 2, :], method='quadratic')

    sticker_red_p = sticker_red_ps[frame_idx, :]
    sticker_green_p = sticker_green_ps[frame_idx, :]
    sticker_yellow_p = sticker_yellow_ps[frame_idx, :]
    #show_stickers_path(sticker_red_p[0:500, :])

    if parameters["hand_stickers"]["color_idx"] == 0:  # red
      xyz_sticker = sticker_red_p - palm_p
    elif parameters["hand_stickers"]["color_idx"] == 1:  # green
      xyz_sticker = sticker_green_p - palm_p
    elif parameters["hand_stickers"]["color_idx"] == 2:  # yellow
      xyz_sticker = sticker_yellow_p - palm_p
    elif parameters["hand_stickers"]["color_idx"] == 3:  # red
      xyz_sticker = sticker_red_p - fingernail_p
    v_handMesh += xyz_sticker
    
    handmesh = o3d.geometry.TriangleMesh()
    handmesh.vertices = o3d.utility.Vector3dVector(v_handMesh)
    handmesh.triangles = o3d.utility.Vector3iVector(t_handMesh)
    handmesh.paint_uniform_color(config.HAND_COLOR)
    handmesh.compute_vertex_normals()

    # Create objects for stickers
    sticker_red_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    sticker_red_sphere.compute_vertex_normals()
    sticker_red_sphere.paint_uniform_color([1.0, 0.0, 0.0])
    sticker_red_sphere.translate(sticker_red_p)
    sticker_green_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    sticker_green_sphere.compute_vertex_normals()
    sticker_green_sphere.paint_uniform_color([0.0, 1.0, 0.0])
    sticker_green_sphere.translate(sticker_green_p)
    sticker_yellow_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
    sticker_yellow_sphere.compute_vertex_normals()
    sticker_yellow_sphere.paint_uniform_color([1.0, 1.0, 0.0])
    sticker_yellow_sphere.translate(sticker_yellow_p)

    list_of_meshes_to_transform = [handmesh, sticker_red_sphere, sticker_green_sphere, sticker_yellow_sphere]

    # define the origin of rotation 
    if parameters["hand_stickers"]["color_idx"] == 0 or \
       parameters["hand_stickers"]["color_idx"] == 1 or \
       parameters["hand_stickers"]["color_idx"] == 2:  # red, green, or yellow
        rotation_point = np.mean(v_handMesh[palm_vertex_idx[0],:], axis=0)
    else:
        rotation_point = np.mean(v_handMesh[index_vertex_idx[0],:], axis=0)
    rotation_point = np.mean(v_handMesh[index_vertex_idx[0],:], axis=0)
    
    # --- 2/2 forearm point cloud Creation ---
    arm_ply_fname_abs = files_path["arm_ply_path"].replace("<session_dir>", curr_session_path) # if necessary, create the long-path-aware version for the Windows API: "\\\\?\\"+    
    arm_pcd = o3d.io.read_point_cloud(arm_ply_fname_abs)
    arm_pcd.estimate_normals()
    arm_pcd.orient_normals_to_align_with_direction(([0., 0., -1.]))
    
    print(f"red point: {sticker_red_p}")
    print(f"green point: {sticker_green_p}")
    print(f"yellow point: {sticker_yellow_p}")
    print(f"rotation point: {rotation_point}")
    print_model_info(handmesh, arm_pcd)

    # --- Run the application ---
    # Create the main Tkinter window.
    root = tk.Tk()

    # Create the plot and its window FIRST.
    plot_window = tk.Toplevel(root)
    plot_window.title("Image Display")

    fig = Figure(figsize=(6, 4), dpi=100)
    # Adjust subplot margins to make the image fit better
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0) 
    ax = fig.add_subplot(111)
    
    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    # 2. Use ax.imshow() to display the 2D image data.
    ax.imshow(rgb_frame)
    # 3. Clean up the plot for image display.
    ax.set_title("2D RGB Image")
    ax.axis('off')  # Hide the x and y axis ticks and labels
    # 4. Redraw the canvas to show the image.
    canvas.draw()

    # Pass the mesh and the desired save path to the transformer
    app = HandMeshTransformer(meshes=list_of_meshes_to_transform,
                              reference_pcd=arm_pcd, 
                              tk_root=root,
                              rotation_point=rotation_point, 
                              save_path=save_file_path)
    app.run()



if __name__ == "__main__":
    use_multiprocessing = False

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
    

    use_specific_sessions = True
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
            output_filename_abs = os.path.join(output_path_abs, "handmesh_transformations.json")
            if not force_processing and os.path.exists(output_filename_abs):
                continue
            
            block_dir_abs = os.path.dirname(config_file_abs)
            files_path, parameters = load_variables_from_json(config_file_abs)

            video_fname_abs = os.path.join(block_dir_abs, files_path["video_fname"])
            frame_id, rgb_frame = select_frame(video_fname_abs)

            define_handmesh_transformation(frame_id, rgb_frame, 
                                           files_path, parameters, 
                                           curr_session_path, block_dir_abs, db_path_handmesh,
                                           output_filename_abs)

            if df_output is None:
                print("----------------- df_output was None (why?) -------------------")
                continue

            if save_results:
                if not os.path.exists(output_path_abs):
                    os.makedirs(output_path_abs)
                    print(f"Directory '{output_path_abs}' created.")
                df_output.to_csv(output_filename_abs, index=False)
                