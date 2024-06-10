import glob
import os
import socket
import tkinter as tk
from tkinter import filedialog


def get_path_root_abs():
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


def get_path_abs(input_dir, output_dir):
    input_dir_abs = get_path_abs_base(input_dir)
    output_dir_abs = get_path_abs_base(output_dir)
    return input_dir_abs, output_dir_abs


def get_path_abs_base(mid_dir):
    # path to database root folder
    data_dir_base = get_path_root_abs()

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
