import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os
import pandas as pd
from pathlib import Path
import random
import re
import shutil
import sys
import warnings

# homemade libraries
# current_dir = Path(__file__).resolve()
sys.path.append(str(Path(__file__).resolve().parent.parent))
import libraries.misc.path_tools as path_tools  # noqa: E402


def save_shift(event, session, final_shift, filename):
    # Read the existing content from the file
    lines = []
    session_found = False

    # Check if the file exists and read its content
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"{filename} not found, creating a new file.")

    # Process the lines to check for the session
    for i, line in enumerate(lines):
        if line.startswith(session + ","):  # Check if the line starts with the session string
            lines[i] = f"{session}, {final_shift}\n"  # Update the existing row
            session_found = True
            break

    # If the session was not found, add a new row
    if not session_found:
        lines.append(f"{session}, {final_shift}\n")

    # Write the updated content back to the file
    with open(filename, "w") as file:
        file.writelines(lines)
    
    print(f"Shift value saved to {filename}")


def get_shift(session, filename):
    # Initialize the shift variable
    shift_value = None

    # Check if the file exists and read its content
    try:
        with open(filename, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"{filename} not found.")
        return None

    # Process the lines to find the session
    for line in lines:
        if line.startswith(session + ","):  # Check if the line starts with the session string
            _, shift_value = line.split(", ")  # Extract the shift value
            shift_value = shift_value.strip()  # Remove any trailing newline characters
            break

    if shift_value is not None:
        print(f"Shift value for session '{session}' is: {shift_value}")
    else:
        print(f"No shift value found for session '{session}'.")

    return int(shift_value)


def define_shift(sessions, db_path_input, save_results=False, shiftv_filename=''):
    
    for session in sessions:
        # Search for the session file in the specified path
        file = [f.name for f in Path(db_path_input).iterdir() if session in f.name]
        if len(file) != 1:
            warnings.warn(f"Issue detected: not exactly 1 csv file found for {session}.")
            continue
        
        file = file[0]
        file_abs = os.path.join(db_path_input, file)
        print(f"Current file: {file}")
        
        # Load current data
        data_all = pd.read_csv(file_abs)

        # Get unique values
        block_ids = data_all['block_id'].unique()
        for block_id in block_ids:
            print(f"block_id: {block_id}")
            d = data_all[data_all["block_id"] == block_id]
            d = d.reset_index(drop=True)

            # normalise the signals
            vec2 = 0+ d["Nerve_freq"] / np.nanmax(d["Nerve_freq"])

            vec1 = 0.5+ d["Depth"] / np.nanmax(d["Depth"])
            vec3 = 0.5+ d["velAbsRaw"] / np.nanmax(d["velAbsRaw"])
            vec4 = 0+ d["velLongRaw"] / np.nanmax(d["velLongRaw"])
            vec5 = 0+ 0.25 * (d["Contact_Flag"] / np.nanmax(d["Contact_Flag"]))

            vec_target_init = d["trial_id"] / np.nanmax(d["trial_id"])

            vec_trial_start = np.append(np.diff(vec_target_init), 0)
            vec_trial_start = np.nan_to_num(vec_trial_start, nan=0)
            vec_target_vertloc = np.where(vec_trial_start != 0)[0]

            vec_target_update = vec_target_init
            vec_target_update_vertloc = vec_target_vertloc

            # Initial setup of the plot
            fig, (ax) = plt.subplots(figsize=(8, 6))
            #plt.subplots_adjust(bottom=0.25)

            # Plot vec1 and vec2 with initial target vector
            ax.plot(vec1, '-', color='b', label='Depth')
            ax.plot(vec3, ':', color='c', label='velAbsRaw')
            ax.plot(vec4, '-', color='y', label='velLongRaw')
            ax.plot(vec5, '-', color='k', label='Contact_Flag')
            ax.plot(vec2, 'g-', label='Nerve_freq')
            l_shifted, = ax.plot(vec_target_update, 'r--', label='Trial ID')
            # Add vertical red lines where the values change
            for v_xloc in vec_target_update_vertloc:
                ax.axvline(x=v_xloc, color='red')#, linestyle='--')
            ax.legend()
            ax.set_title(f"{session}:\nblock id = {block_id}/{len(block_ids)-1}\n$\it{'(not block order!)'}$")

            # Slider setup
            ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
            mini = round(-0.05 * len(vec_target_init))
            maxi = round(0.05 * len(vec_target_init))
            steps = round(abs(0.005 * mini))
            slider = Slider(ax_slider, 'Shift', mini, maxi, valinit=0, valstep=steps)

            # Update function for the slider
            final_shift = 0
            def update(val):
                nonlocal final_shift
                final_shift = int(slider.val)
                vec_target_update = np.roll(vec_target_init, final_shift)
                vec_target_update_vertloc = vec_target_vertloc + final_shift
                l_shifted.set_ydata(vec_target_update)
                # Clear previous vertical lines
                for line in ax.lines:
                    if line.get_color() == 'red':
                        line.remove()
                for v_xloc in vec_target_update_vertloc:
                    ax.axvline(x=v_xloc, color='red')#, linestyle='--')
                fig.canvas.draw_idle()

            slider.on_changed(update)

            # Connect the close event to the save_shift function with the filename
            if save_results:
                fig.canvas.mpl_connect('close_event', 
                                        lambda event: save_shift(event, session, final_shift, shiftv_filename))

            # Show the figure (blocking)
            manager = plt.get_current_fig_manager()
            manager.window.showMaximized()  # For Windows
            plt.show(block=True)



def apply_shift(sessions, db_path_input, db_path_output, shiftv_filename, save_results=False):
    for session in sessions:
        file = [f.name for f in Path(db_path_input).iterdir() if session in f.name]
        if len(file) != 1:
            warnings.warn(f"Issue detected: not exactly 1 csv file found for {session}.")
            continue
        file = file[0]
        file_abs = os.path.join(db_path_input, file)
        print(f"current file: {file}")
        
        # load current data
        data = pd.read_csv(file_abs)
        shift = get_shift(session, shiftv_filename)

        # Apply the shift to the columns related to the trial information (block id comprised)
        data['trial_id'] = data['trial_id'].shift(shift)
        data['block_id'] = data['block_id'].shift(shift)
        data['stimulus'] = data['stimulus'].shift(shift)
        data['vel'] = data['vel'].shift(shift)
        data['finger'] = data['finger'].shift(shift)
        data['force'] = data['force'].shift(shift)

        # save data on the hard drive ?
        if save_results:
            output_filename = file
            output_filename_abs = os.path.join(db_path_output, output_filename)
            # https://answers.microsoft.com/en-us/msoffice/forum/all/excel-file-open-the-file-name-is-too-long-rename/ef736fec-0bd4-42a9-806d-5b22dbfdda81#:~:text=To%20resolve%20this%20issue%2C%20you,structure%2C%20is%20still%20too%20long.
            #  Excel indicates that the total path length,
            #  including the filename and its directory structure,
            #  exceeds the Windows maximum limit of 260 characters.
            data.to_csv(output_filename_abs, index=False)
        
        print("done.")




if __name__ == "__main__":
    # parameters
    force_processing = True  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening
    
    shift_processing = "define"  # <define> or <apply> shift
    save_results = False

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled", "3_merged", "1_kinect_and_nerve_shandata")

    # get input base directory
    db_path_input = os.path.join(db_path, "0_by-units_renamed")
    # get output base directory
    db_path_output = os.path.join(db_path, "0_by-units_renamed_trial-corrected")
    # shift value output filename
    shiftv_filename =  os.path.join(db_path_output, "semi-controlled_trial-id_shift-values.txt")

    if not os.path.exists(db_path_output):
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
    
    if shift_processing == "define":
        define_shift(sessions, db_path_input, save_results=save_results, shiftv_filename=shiftv_filename)
    elif shift_processing == "apply":
        apply_shift(sessions, db_path_input, db_path_output, shiftv_filename=shiftv_filename, save_results=save_results)


















