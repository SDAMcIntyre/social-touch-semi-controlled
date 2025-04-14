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



# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'imported'))
import libraries.misc.path_tools as path_tools  # noqa: E402
from handtracking_constant_model_only_BD import Contact_quantities_ConstantHandGesture as extract_contact_quantities


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


def extract_somatosensory_data(config_file_abs, session_dir_abs, block_dir_abs, db_path_handmesh, show=False):
    files_path, parameters = load_variables_from_json(config_file_abs)

    video_fname_abs = os.path.join(block_dir_abs, files_path["video_fname"])
    sticker_fname_abs = os.path.join(block_dir_abs, files_path["sticker_fname"])
    arm_ply_fname_abs = files_path["arm_ply_path"].replace("<session_dir>", session_dir_abs)
    arm_normal_fname_abs = files_path["arm_normal_path"].replace("<session_dir>", session_dir_abs)
    handmesh_model_fname_abs = files_path["handmesh_model_path"].replace("<handmesh_models_dir>", db_path_handmesh)

    Root_Shift0 = parameters["hand_mesh"]["hardcoded_shift"]["general"]
    # convert time to frame by multiplying the time interval values of each section by 30 (Fs of Kinect is 30 Hz)
    Other_Shifts = parameters["hand_mesh"]["hardcoded_shift"]["specific_time_sections"]
    for section in Other_Shifts:
        section[0] *= 30
    Hand_Rotate0 = parameters["viewer"]["matrix_rotation_viewer"]
    Color_idx = parameters["hand_stickers"]["color_idx"]
    ArmNParas = parameters["arm"]['normal_vector_XY_index']
    left = parameters["hand_used"]['left']
    
    res_df = extract_contact_quantities(video_fname_abs, arm_ply_fname_abs, sticker_fname_abs, 
                                        Root_Shift0,Other_Shifts,Hand_Rotate0,
                                        Color_idx, 
                                        ArmNParas, arm_normal_fname_abs, 
                                        left, handmesh_model_fname_abs, 
                                        show=show, show_result=True)

    return res_df



def extract_somatosensory_data_onethread(args):
    (
        config_file_abs, 
        config_file, 
        curr_session_path, 
        db_path_input, 
        db_path_output, 
        db_path_handmesh, 
        force_processing, 
        save_results, 
        show
    ) = args
    
    block_dir_abs = os.path.dirname(config_file_abs)
    output_path_abs = block_dir_abs.replace(db_path_input, db_path_output)
    output_filename_abs = os.path.join(output_path_abs, "somatosensory_data.csv")
    
    if not force_processing and os.path.exists(output_filename_abs):
        return
    
    df_output = extract_somatosensory_data(config_file_abs, curr_session_path, block_dir_abs, db_path_handmesh, show=show)
    
    if df_output is None:
        print("----------------- df_output was None (why?) -------------------")
        return
    
    if save_results and not os.path.exists(output_path_abs):
        os.makedirs(output_path_abs)
        print(f"Directory '{output_path_abs}' created.")
    
    if save_results:
        df_output.to_csv(output_filename_abs, index=False)




if __name__ == "__main__":
    use_multiprocessing = False

    force_processing = False  # If user wants to force data processing even if results already exist
    save_results = True
    generate_report = True

    show = False  # If user wants to monitor what's happening

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path_linux = os.path.expanduser('~/Documents/datasets')

    # get input base directory
    db_path_input = os.path.join(db_path_linux, "semi-controlled", "1_primary", "kinect")
    db_path_handmesh = os.path.join(db_path_linux, "semi-controlled", "handmesh_models")

    # get output base directory
    db_path_output = os.path.join(db_path_linux, "semi-controlled", "2_processed", "kinect")
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
    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:
        curr_session_path = os.path.join(db_path_input, session)
        config_files_abs, config_files = path_tools.find_files_in_directory(curr_session_path, ending='_extraction-config.json')

        if use_multiprocessing:
            args_list = [
                (
                    config_file_abs, 
                    config_file, 
                    curr_session_path, 
                    db_path_input, 
                    db_path_output, 
                    db_path_handmesh, 
                    force_processing, 
                    save_results, 
                    show
                ) 
                for config_file_abs, config_file in zip(config_files_abs, config_files)
            ]
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                pool.map(extract_somatosensory_data_onethread, args_list)
        else:
            for config_file_abs, config_file in zip(config_files_abs, config_files):
                block_dir_abs = os.path.dirname(config_file_abs)
                
                output_path_abs = block_dir_abs.replace(db_path_input, db_path_output)
                output_filename_abs = os.path.join(output_path_abs, "somatosensory_data.csv")
                if not force_processing and os.path.exists(output_filename_abs):
                    continue
                
                df_output = extract_somatosensory_data(config_file_abs, curr_session_path, block_dir_abs, db_path_handmesh, show=show)

                if df_output is None:
                    print("----------------- df_output was None (why?) -------------------")
                    continue

                if save_results:
                    if not os.path.exists(output_path_abs):
                        os.makedirs(output_path_abs)
                        print(f"Directory '{output_path_abs}' created.")
                    df_output.to_csv(output_filename_abs, index=False)
                