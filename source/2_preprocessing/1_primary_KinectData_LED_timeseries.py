import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from libraries.preprocessing.semicontrolled_data_KinectLED import KinectLED  # noqa: E402
import libraries.misc.path_tools as path_tools  # noqa: E402


def find_mkv_files(sessions):
    """
    Given a dictionary of directory names and their absolute paths,
    find all .mkv files nested inside these directories.

    Args:
    directories (dict): A dictionary where keys are directory names and
                        values are their absolute paths.

    Returns:
    list: A list of absolute paths to all .mkv files found.
    """
    data_dir_base = path_tools.get_path_root_abs()
    input_dir = "primary"
    datatype_str = "semi-controlled\kinect"

    # destination
    dir_abs = os.path.join(data_dir_base, input_dir)
    # target the current experiment, postprocessing and datatype
    dir_abs = os.path.join(dir_abs, datatype_str)

    mkv_files_session = []
    mkv_files = []
    mkv_files_abs = []

    for session in sessions:
        dir_path = os.path.join(dir_abs, session)
        # Walk through the directory recursively
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.mkv'):
                    mkv_files.append(file)
                    mkv_files_abs.append(os.path.join(root, file))
                    mkv_files_session.append(session)

    return mkv_files_abs, mkv_files, mkv_files_session


def get_output_dir():
    data_dir_base = path_tools.get_path_root_abs()
    output_dir = "processed"
    datatype_str = "semi-controlled\kinect"
    # destination
    dir_abs = os.path.join(data_dir_base, output_dir)
    # target the current experiment, postprocessing and datatype
    dir_abs = os.path.join(dir_abs, datatype_str)

    if not os.path.exists(dir_abs):
        os.makedirs(dir_abs)
        print(f"Directory '{dir_abs}' created.")
    else:
        print(f"Directory '{dir_abs}' already exists.")

    return dir_abs


if __name__ == "__main__":
    video_path = "E:\\OneDrive - Linköpings universitet\\_Teams\\touch comm MNG Kinect\\basil_tmp\\data\\primary\\semi-controlled\\kinect\\2021-12-10_ST12-01\\NoIR_2021-12-10_13-34-08_ST12_unit1_20bpm-stroke-2fingers.mkv"
    video_path3 = "E:\\OneDrive - Linköpings universitet\\_Teams\\touch comm MNG Kinect\\basil_tmp\\data\\primary\\semi-controlled\\kinect\\2022-06-22_ST18-04\\2022-06-22_17-36-14\\NoIR_2022-06-22_17-39-03_controlled-touch-MNG_ST18_4_block3.mkv"
    video_path4 = "E:\\OneDrive - Linköpings universitet\\_Teams\\touch comm MNG Kinect\\basil_tmp\\data\\primary\\semi-controlled\\kinect\\2022-06-22_ST18-04\\2022-06-22_17-36-14\\NoIR_2022-06-22_17-39-51_controlled-touch-MNG_ST18_4_block4.mkv"

    # Session names
    sessions = ['2022-06-15_ST13-01',
                '2022-06-15_ST13-02',
                '2022-06-15_ST13-03',
                '2022-06-15_ST14-01',
                '2022-06-15_ST14-02',
                '2022-06-15_ST14-03',
                '2022-06-15_ST14-04',
                '2022-06-15_ST15-01',
                '2022-06-15_ST15-02',
                '2022-06-17_ST16-02',
                '2022-06-17_ST16-03',
                '2022-06-17_ST16-04',
                '2022-06-17_ST16-05',
                '2022-06-22_ST18-01',
                '2022-06-22_ST18-02',
                '2022-06-22_ST18-04']
    sessions = ['2022-06-22_ST18-04']
    mkv_files_abs, mkv_files, mkv_files_session = find_mkv_files(sessions)
    output_dir_abs = get_output_dir()

    for mkv_filename_abs, mkv_filename, session in zip(mkv_files_abs, mkv_files, mkv_files_session):
        kinect_led = KinectLED(mkv_filename_abs)
        kinect_led.select_led()
        kinect_led.monitor_green_levels(threshold=.0025, show=True)

        plt.plot(kinect_led.time, kinect_led.led_on)

        # Save to CSV
        output_dirname = os.path.join(output_dir_abs, session)
        output_filename = mkv_filename.replace(".mkv", "") + "_LED_timeseries"
        kinect_led.save(output_dirname, output_filename)
