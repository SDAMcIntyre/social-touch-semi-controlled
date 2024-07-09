import warnings

import ffmpeg
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import subprocess
import sys

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import libraries.misc.path_tools as path_tools  # noqa: E402
import numpy as np
import json
import subprocess


def videoInfo(filename):
    proc = subprocess.run([
        *"ffprobe -v quiet -print_format json -show_format -show_streams".split(),
        filename
    ], capture_output=True)
    proc.check_returncode()
    return json.loads(proc.stdout)


def readVideo(filename):
    cmd = ["ffmpeg", "-i", filename]
    streams = 0
    for stream in videoInfo(filename)["streams"]:
        index = stream["index"]
        if stream["codec_type"] == "video":
            width = stream["width"]
            height = stream["height"]
            cmd += "-map", f"0:{index}"
            streams = streams + 1
    cmd += "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
    shape = np.array([height, width, 3])
    frames = []
    nframes_ffmpeg = 0
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        while True:
            data = proc.stdout.read(shape.prod())  # One byte per each element
            if not data:
                #return
                break
            nframes_ffmpeg += 1
            frame = np.frombuffer(data, dtype=np.uint8).reshape(shape)
            frames.append(frame)
            yield np.frombuffer(data, dtype=np.uint8).reshape(shape)


def get_video_duration(video_path):
    info = ffmpeg.probe(video_path)
    duration = 0.0
    if info['streams'][0]['tags']['title'] == "COLOR":
        time_str = info['streams'][0]['tags']['DURATION']
        # Split the string into hours, minutes, and the seconds.microseconds part
        hours, minutes, seconds_microseconds = time_str.split(':')
        # Further split the seconds.microseconds part into seconds and microseconds
        seconds, nanoseconds = seconds_microseconds.split('.')
        # Convert each part to a float and calculate the total number of seconds
        duration = (
                int(hours) * 3600 +  # hours to seconds
                int(minutes) * 60 +  # minutes to seconds
                int(seconds) +  # seconds
                int(nanoseconds) / 1_000_000_000  # nanoseconds to seconds
        )

    else:
        warnings.warn("the index [0] of the json info doesn't correspond to the COLOR data of the MKV video")
        warnings.warn("Apparently, COLOR (0) and DEPTH (index 1) don't share the same size / number of frames.")
        warnings.warn("Index [2] corresponds to IMU, which has been removed from the raw video.")

    nframes = info['streams'][0]['tags']['NUMBER_OF_FRAMES']
    frame_rate = info['streams'][0]['avg_frame_rate']
    num, denom = map(int, frame_rate.split('/'))
    return duration, nframes, num/denom


def with_opencv(filename):
    import cv2
    video = cv2.VideoCapture(filename)

    duration = video.get(cv2.CAP_PROP_POS_MSEC)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    return duration, frame_count


def with_ffprobe(filename):
    import subprocess, json

    result = subprocess.check_output(
            f'ffprobe -v quiet -show_streams -select_streams v:0 -of json "{filename}"',
            shell=True).decode()
    fields = json.loads(result)['streams'][0]

    duration = fields['tags']['DURATION']
    fps      = eval(fields['r_frame_rate'])
    return duration, fps


def get_frame_rate(video_path):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'json',
         video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    output = result.stdout.decode('utf-8')
    try:
        json_output = json.loads(output)
    except json.JSONDecodeError:
        # If the whole output is not valid JSON, extract the JSON part
        json_output = json.loads(output[output.index('{'):output.rindex('}') + 1])

    frame_rate = json_output['streams'][0]['r_frame_rate']
    num, denom = map(int, frame_rate.split('/'))
    return num / denom


def get_frame_count(video_path):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
         '-show_entries', 'stream=nb_read_frames', '-of', 'json', video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    output = result.stdout.decode('utf-8')
    try:
        json_output = json.loads(output)
    except json.JSONDecodeError:
        # If the whole output is not valid JSON, extract the JSON part
        json_output = json.loads(output[output.index('{'):output.rindex('}')+1])

    frames = json_output['streams'][0]['nb_read_frames']
    return int(frames)


#
#
# https://stackoverflow.com/questions/3844430/how-to-get-the-duration-of-a-video-in-python
# https://stackoverflow.com/questions/58824909/opencv-read-video-files-with-multiple-streams-tracks
#
if __name__ == "__main__":
    force_processing = False  # If user wants to force data processing even if results already exist
    show = False  # If user wants to monitor what's happening
    save_results = False

    print("Step 0: Extract the videos embedded in the selected sessions.")
    # get database directory
    db_path = path_tools.get_database_path()

    # get input base directory
    db_path_input = os.path.join(db_path, "semi-controlled")
    db_path_input_mkv = os.path.join(db_path_input, "primary", "kinect", "1_block-order")
    db_path_input_contact = os.path.join(db_path_input, "processed", "kinect", "contact", "1_block-order")
    db_path_input_led = os.path.join(db_path_input, "processed", "kinect", "led")

    # get output base directory
    db_path_output = os.path.join(db_path, "semi-controlled", "merged", "kinect", "0_validation_duration_files")
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
    print(sessions)

    contact_diff_ms_all = []
    led_diff_ms_all = []
    # it is important to split by MNG files / neuron recordings to create the correct subfolders.
    for session in sessions:

        curr_mkv_dir = os.path.join(db_path_input_mkv, session)
        curr_contact_dir = os.path.join(db_path_input_contact, session)
        curr_led_dir = os.path.join(db_path_input_led, session)

        files_mkv_abs, files_mkv = path_tools.find_files_in_directory(curr_mkv_dir, ending='kinect.mkv')

        for file_mkv_abs, file_mkv in zip(files_mkv_abs, files_mkv):
            valid = True
            # check if contact file exists for this contact file
            file_contact = file_mkv.replace("kinect.mkv", "contact.csv")
            file_contact_abs = os.path.join(curr_contact_dir, file_contact)
            try:
                with open(file_contact_abs, 'r'):
                    print("Matching contact file exists.")
            except FileNotFoundError:
                print("Matching contact file does not exist.")
                valid = False

            # check if led file exists for this contact file
            file_led = file_mkv.replace("kinect.mkv", "LED.csv")
            file_led_abs = os.path.join(curr_led_dir, file_led)
            try:
                with open(file_led_abs, 'r'):
                    print("Matching LED file exists.")
            except FileNotFoundError:
                print("Matching LED file does not exist.")
                valid = False

            # if either of the contact or LED file doesn't exist, go to next contact file
            if not valid:
                continue

            # 1. Duration of the mkv video
            mkv_duration,_,_ = get_video_duration(file_mkv_abs)
            #nframes = get_frame_count(file_mkv_abs)
            # 2. Duration of contact csv file
            contact = pd.read_csv(file_contact_abs)
            contact_duration = contact['t'].iloc[-1]

            # 3. Duration of LED csv file
            led = pd.read_csv(file_led_abs)
            led_duration = led['time (second)'].iloc[-1]

            contact_diff_ms = (mkv_duration-contact_duration) * 1000
            led_diff_ms = (mkv_duration-led_duration) * 1000

            print(f"current filename = {file_contact.replace('_contact.csv', '')}:")
            print(f"mkv = {mkv_duration:.3f} seconds.")
            print(f"contact = {contact_duration:.3f} seconds.")
            print(f"led = {led_duration:.3f} seconds.")
            print(f"contact diff = {contact_diff_ms:.1f} milliseconds.")
            print(f"led diff = {led_diff_ms:.1f} milliseconds.")
            print(f"------------------")
            print(f"contact nframe = {len(contact)}.")
            print(f"contact nframe = {len(led)}.")
            print(f"diff : {len(contact)-len(led)}.")
            print(f"------------------\n")
            # save data on the hard drive ?
            if save_results:
                pass

            contact_diff_ms_all.append(contact_diff_ms)
            led_diff_ms_all.append(led_diff_ms)
            print("done.")

    # Step 4: Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(contact_diff_ms_all, label='MKV - Contact duration')
    plt.plot(led_diff_ms_all, label='MKV - LED duration')

    # Set Y-axis to log scale
    #plt.yscale('log')

    # Add labels and title
    plt.xlabel('Index')
    plt.ylabel('Time (ms)')
    plt.title('Contact vs LED time difference for all valuable kinect videos')
    plt.legend()

    # Show the plot
    plt.show()

