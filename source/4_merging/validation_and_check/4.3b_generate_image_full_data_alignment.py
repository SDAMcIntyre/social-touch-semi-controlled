import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import re
import sys
import warnings
import time

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from libraries.materials.semicontrolled_data import SemiControlledData  # noqa: E402
from libraries.processing.semicontrolled_data_manager import SemiControlledDataManager  # noqa: E402
from libraries.processing.semicontrolled_data_splitter import SemiControlledDataSplitter  # noqa: E402
from libraries.plot.semicontrolled_data_visualizer import SemiControlledDataVisualizer  # noqa: E402
import libraries.misc.path_tools as path_tools  # noqa: E402
from libraries.misc.waitforbuttonpress_popup import WaitForButtonPressPopup
import libraries.plot.semicontrolled_data_visualizer_unitcontact as scdata_visualizer_neur  # noqa: E402


if __name__ == "__main__":
    force_processing = True  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening
    save_results = True

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = os.path.join(path_tools.get_database_path(), "semi-controlled")
    # get input data directory
    db_path_input = os.path.join(db_path, "3_merged", "2_kinect_and_nerve", "1_by-trials")
    # get metadata paths
    md_stimuli_path = os.path.join(db_path, "1_primary", "logs", "stimuli_by_blocks")
    md_neuron_filename_abs = os.path.join(db_path, "1_primary", "nerve", "semicontrol_unit-name_to_unit-type.csv")
    # get output directory
    result_path = os.path.join(path_tools.get_result_path(), "semi-controlled")
    db_path_output = os.path.join(result_path, "kinect_and_nerve", "0_by-trials")
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
    #sessions = ['2022-06-15_ST14-02']
    print(sessions)

    path_fig = os.path.join(db_path_output, "trials_overview")
    if not os.path.exists(path_fig):
        os.makedirs(path_fig)

    if show or save_results:
        scd_visualiser = SemiControlledDataVisualizer()
    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:
        curr_dir = os.path.join(db_path_input, session)
        files_abs, files = path_tools.find_files_in_directory(curr_dir, ending='.csv')

        output_session_abs = db_path_output

        for data_filename_abs, data_filename in zip(files_abs, files):
            print(f"current filename: {data_filename}")
            #if 2 != int(re.search("block-order\d{2}", data_filename).group().replace("block-order", "")): continue

            output_filename = os.path.join(path_fig, data_filename.replace(".csv", ".png"))
            if not force_processing:
                try:
                    with open(output_filename, 'r'):
                        print("Result file exists, jump to the next dataset.")
                        continue
                except FileNotFoundError:
                    pass

            # 1. extract metadata related to the current stimulus set and check if exists
            md_stimuli_filename = re.sub(r'_trial\d{2}\.csv', '_stimuli.csv', data_filename)
            md_stimuli_filename_abs = os.path.join(md_stimuli_path, session, md_stimuli_filename)
            if not os.path.exists(md_stimuli_filename_abs):
                warnings.warn(f'The file {md_stimuli_filename_abs} exists.', Warning)
                continue

            # 2. check if neuron metadata file exists
            if not os.path.exists(md_neuron_filename_abs):
                warnings.warn(f'The file {md_neuron_filename_abs} exists.', Warning)
                continue

            # 3. create a SemiControlledData's list of TOUCH EVENT:
            # 3.1 load the data
            scd = SemiControlledData(data_filename_abs, md_stimuli_filename_abs, md_neuron_filename_abs)  # resources
            scd.set_variables(dropna=False)

            scd_visualiser.update(scd)

            if show:
                WaitForButtonPressPopup()

            if save_results:
                # set correct dimensions
                screenratio = 1920 / 1080
                dpi = 100
                height = 1080 / dpi
                width = screenratio * height
                scd_visualiser.figpos.fig.set_size_inches(width / 2, height)
                scd_visualiser.fig2D_global.fig.set_size_inches(width / 2, height * 2 / 3)
                scd_visualiser.fig2D_TTL.fig.set_size_inches(width / 2, height * 1 / 3)

                # temporarily save the figures
                scd_visualiser.figpos.fig.savefig(os.path.join(db_path_output, "fispos_tmp.png"))
                scd_visualiser.fig2D_global.fig.savefig(os.path.join(db_path_output, "global_tmp.png"))
                scd_visualiser.fig2D_TTL.fig.savefig(os.path.join(db_path_output, "ttl_tmp.png"))

                # Load the temporary images
                figpos = Image.open(os.path.join(db_path_output, "fispos_tmp.png"))
                fig2D_global = Image.open(os.path.join(db_path_output, "global_tmp.png"))
                fig2D_TTL = Image.open(os.path.join(db_path_output, "ttl_tmp.png"))

                # Get the image dimensions
                width1, height1 = figpos.size
                width2, height2 = fig2D_global.size
                width3, height3 = fig2D_TTL.size

                # Create a new image with the combined height of the three images
                combined_image = Image.new('RGB', (width1 + width2, height1))

                # Get the dimensions of each image
                combined_image.paste(figpos, (0, 0))
                combined_image.paste(fig2D_global, (width1, 0))
                combined_image.paste(fig2D_TTL, (width1, height2))

                # Save the combined image
                combined_image.save(output_filename)

                # Optional: Show the combined image
                # combined_image.show()
    print("done.")

























