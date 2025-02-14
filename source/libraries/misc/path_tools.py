import glob
import os
import re
import socket
import tkinter as tk
from tkinter import filedialog


def winapi_path(dos_path, encoding=None):
    if not isinstance(dos_path, str) and encoding is not None:
        dos_path = dos_path.decode(encoding)
    path = os.path.abspath(dos_path)
    if path.startswith(u"\\\\"):
        return u"\\\\?\\UNC\\" + path[2:]
    return u"\\\\?\\" + path


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


def find_files_in_directory(dir_path, ending=r'.csv$'):
    # ensure that the ending string written by user finished by a dollar for Regular Expression
    if not ending.endswith('$'):
        ending += r'$'

    files = []
    files_abs = []

    # Walk through the directory recursively
    for root, _, f in os.walk(dir_path):
        for file in f:
            if re.search(ending, file):
            #  if file.endswith(ending):
                files.append(file)
                files_abs.append(os.path.join(root, file))

    return files_abs, files


def get_onedrive_path_abs():
    # path to onedrive root folder
    if socket.gethostname == "basil":
        onedrive_path_abs = os.path.join('F:\\', 'OneDrive - Linköpings universitet', '_Teams')
    else:
        onedrive_path_abs = os.path.join('C:\\Users\\basdu83', 'OneDrive - Linköpings universitet', '_Teams')
    return onedrive_path_abs


def get_team_path_abs(cloud_location="Teams"):
    # path to onedrive root folder
    onedrive_path_abs = get_onedrive_path_abs()

    # path to database root folder
    if cloud_location == "Teams":
        team_path = "Social touch Kinect MNG"
    elif cloud_location == "Sarah repository":  # Sarah's shared folder
        team_path = os.path.join('touch comm MNG Kinect', 'basil_tmp')

    return os.path.join(onedrive_path_abs, team_path)


def get_database_path(cloud_location="Teams"):
    # path to database root folder
    team_path_abs = get_team_path_abs(cloud_location=cloud_location)
    return os.path.join(team_path_abs, 'data')


def get_metadata_path(cloud_location="Teams"):
    # path to database root folder
    team_path_abs = get_team_path_abs(cloud_location=cloud_location)
    return os.path.join(team_path_abs, 'metadata')


def get_result_path(cloud_location="Teams"):
    # path to database root folder
    team_path_abs = get_team_path_abs(cloud_location=cloud_location)
    return os.path.join(team_path_abs, 'figures')


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
    if mode == "automatic":
        selected_files = glob.glob(os.path.join(input_dir, '*ST*-unit*-semicontrol.csv'))
    elif mode == "manual":
        selected_files = select_files_manual(input_dir)
    else:
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
