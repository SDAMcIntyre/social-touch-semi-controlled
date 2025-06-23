import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import sys
import warnings
import multiprocessing

import tkinter as tk
from tkinter import filedialog, Scale, Button, Label, Toplevel
from PIL import Image, ImageTk
import cv2

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



def define_handmesh_transformation(frame_idx, rgb_frame, 
                                   files_path, parameters, 
                                   curr_session_path, block_dir_abs, db_path_handmesh, 
                                   save_file_path):
    
    # --- 1/2 hand Mesh Creation ---
    # -- load initial models
    handmesh_path = files_path["handmesh_model_path"].replace("<handmesh_models_dir>", db_path_handmesh)
    v_handMesh = np.loadtxt(handmesh_path)
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
    if parameters["hand_stickers"]["color_idx"] == 0:  # red
      xyz_sticker = corrected_joints_color[frame_idx, 0:3] - palm_p
    elif parameters["hand_stickers"]["color_idx"] == 1:  # green
      xyz_sticker = corrected_joints_color[frame_idx, 3:6] - palm_p
    elif parameters["hand_stickers"]["color_idx"] == 2:  # yellow
      xyz_sticker = corrected_joints_color[frame_idx, 6:9] - palm_p
    elif parameters["hand_stickers"]["color_idx"] == 3:  # red
      xyz_sticker = corrected_joints_color[frame_idx, 0:3] - fingernail_p
    v_handMesh += xyz_sticker
    
    handmesh = o3d.geometry.TriangleMesh()
    handmesh.vertices = o3d.utility.Vector3dVector(v_handMesh)
    handmesh.triangles = o3d.utility.Vector3iVector(t_handMesh)
    handmesh.paint_uniform_color(config.HAND_COLOR)
    handmesh.compute_vertex_normals()

    # define the origin of rotation 
    if parameters["hand_stickers"]["color_idx"] == 0 or \
       parameters["hand_stickers"]["color_idx"] == 1 or \
       parameters["hand_stickers"]["color_idx"] == 2:  # red, green, or yellow
        rotation_point = np.mean(v_handMesh[palm_vertex_idx[0],:], axis=0)
    else:
        rotation_point = np.mean(v_handMesh[index_vertex_idx[0],:], axis=0)

    # --- 2/2 forearm point cloud Creation ---
    arm_ply_fname_abs = files_path["arm_ply_path"].replace("<session_dir>", curr_session_path) # if necessary, create the long-path-aware version for the Windows API: "\\\\?\\"+    
    arm_pcd = o3d.io.read_point_cloud(arm_ply_fname_abs)
    arm_pcd.estimate_normals()
    arm_pcd.orient_normals_to_align_with_direction(([0., 0., -1.]))
    
    # --- Run the application ---
    # Pass the mesh and the desired save path to the transformer
    app = HandMeshTransformer(handmesh, arm_pcd, rotation_point, save_path=save_file_path)
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
                