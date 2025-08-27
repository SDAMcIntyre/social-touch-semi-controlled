import glob
import os
import re
import socket
import tkinter as tk
from tkinter import filedialog
from pathlib import Path


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


def get_project_data_root():
    """
    Determines the project data root directory.

    First, it tries to find the directory automatically. If that fails or the
    directory doesn't exist, it opens a GUI dialog for the user to select it.
    The dialog window is forced to appear on top of all other windows.

    Returns:
        Path: The path to the project_data_root directory.
        None: If the user cancels the directory selection.
    """
    try:
        # Attempt to find the path automatically as before
        base_path = Path(get_database_path())
        project_data_root = base_path / "semi-controlled"
        if project_data_root.is_dir():
            print(f"✅ Project DATA root automatically identified at: {project_data_root.resolve()}")
            return project_data_root
    except FileNotFoundError:
        # This case is hit if get_database_path() fails. We'll pass
        # and let the GUI handler take over.
        pass

    # If automatic detection fails or the directory doesn't exist, prompt the user
    print("⚠️ Project DATA root not found automatically.")
    print("Please select your 'semi-controlled' data folder using the dialog window.")

    # Set up the Tkinter root window
    root = tk.Tk()
    
    # --- MODIFICATION START ---
    # Force the window to the front
    root.attributes('-topmost', True)
    # --- MODIFICATION END ---
    
    # Hide the main Tkinter window
    root.withdraw()

    # Open the directory selection dialog
    selected_path = filedialog.askdirectory(
        title="Please Select the Project Data Folder"
    )

    # --- MODIFICATION START ---
    # Destroy the root window to free up resources
    root.destroy()
    # --- MODIFICATION END ---

    if not selected_path:  # Handles the case where the user closes the dialog
        print("❌ No folder selected. Exiting program.")
        return None

    project_data_root = Path(selected_path)
    print(f"👍 Project DATA root set by user to: {project_data_root.resolve()}")
    return project_data_root



def get_onedrive_path_abs():
    # path to onedrive root folder
    if socket.gethostname() == "basil":
        # onedrive_path_abs = os.path.join('F:\\', 'OneDrive - Linköpings universitet', '_Teams')
        # use of a symbolic link now to avoid dealing with special characters
        onedrive_path_abs = os.path.join('F:\\', 'liu-onedrive-nospecial-carac', '_Teams')
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
