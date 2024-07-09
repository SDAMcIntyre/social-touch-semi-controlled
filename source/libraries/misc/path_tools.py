import glob
import os
import socket
import tkinter as tk
from tkinter import filedialog


def find_files_in_sessions(input_path, sessions, ending='.csv'):
    if not isinstance(sessions, list):
        sessions = [sessions]

    files_abs = []
    files = []
    files_session = []

    for session in sessions:
        dir_path = os.path.join(input_path, session)
        [curr_files_abs, curr_files] = find_files_in_directory(dir_path, ending=ending)

        files_abs.append(curr_files_abs)
        files.append(curr_files)
        files_session.append([session * len(curr_files)])

    return files_abs, files, files_session


def find_files_in_directory(dir_path, ending='.csv'):
    files = []
    files_abs = []

    # Walk through the directory recursively
    for root, _, f in os.walk(dir_path):
        for file in f:
            if file.endswith(ending):
                files.append(file)
                files_abs.append(os.path.join(root, file))

    return files_abs, files


def get_database_path():
    # path to onedrive root folder
    match socket.gethostname():
        case "baz":
            data_dir_base = "E:\\"
        case _:
            data_dir_base = 'C:\\Users\\basdu83'
    # path to database root folder
    data_dir_base = os.path.join(data_dir_base,
                                 'OneDrive - Linköpings universitet',
                                 '_Teams',
                                 'touch comm MNG Kinect',
                                 'basil_tmp',
                                 'data')
    return data_dir_base


def get_metadata_path():
    # path to onedrive root folder
    match socket.gethostname():
        case "baz":
            data_dir_base = "E:\\"
        case _:
            data_dir_base = 'C:\\Users\\basdu83'
    # path to database root folder
    data_dir_base = os.path.join(data_dir_base,
                                 'OneDrive - Linköpings universitet',
                                 '_Teams',
                                 'touch comm MNG Kinect',
                                 'basil_tmp',
                                 'metadata')
    return data_dir_base


def get_result_path():
    # path to onedrive root folder
    match socket.gethostname():
        case "baz":
            data_dir_base = "E:\\"
        case _:
            data_dir_base = 'C:\\Users\\basdu83'
    # path to database root folder
    data_dir_base = os.path.join(data_dir_base,
                                 'OneDrive - Linköpings universitet',
                                 '_Teams',
                                 'touch comm MNG Kinect',
                                 'basil_tmp',
                                 'figures')
    return data_dir_base


def get_path_abs(input_dir, output_dir):
    input_dir_abs = get_path_abs_base(input_dir)
    output_dir_abs = get_path_abs_base(output_dir)
    return input_dir_abs, output_dir_abs


def get_path_abs_base(mid_dir):
    # path to database root folder
    data_dir_base = get_database_path()

    # destination
    dir_abs = os.path.join(data_dir_base, mid_dir)
    # target the current experiment, postprocessing and datatype
    dir_abs = os.path.join(dir_abs,
                           'semi-controlled',
                           'contact_and_neural',
                           'new_axes_3Dposition')  # 'new_axes_3Dposition' or 'new_axes'
    return dir_abs


def select_files_processed_data(input_dir, mode="manual"):
    match mode:
        case "automatic":
            selected_files = glob.glob(os.path.join(input_dir, '*ST*-unit*-semicontrol.csv'))
        case "manual":
            selected_files = select_files_manual(input_dir)
        case _:
            selected_files = ""
    return selected_files


def select_files_manual(initial_folder):
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
