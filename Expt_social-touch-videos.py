import os
import time

from psychopy import core, data, gui, event
from pynput import keyboard

from modules.arduino_comm import ArduinoComm
from modules.expert_interface import ExpertInterface
from modules.file_management import FileManager
from modules.kinect_comm import KinectComm

script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)
print("starting video cue script")

# -- GET INPUT FROM THE EXPERIMENTER --
expt_info = {
    '01. Experiment Name': 'controlled-touch-video',
    '02. Participant Code': 'video',
    '03. Unit Number': 0,
    '04. Folder for saving data': 'data',
    '05. Start from block no.': 1
    }
dlg = gui.DlgFromDict(expt_info, title='Experiment details')
if dlg.OK:
    pass  # continue
else:
    core.quit()  # the user hit cancel so exit

# add the time when the user pressed enter:
expt_info['Date and time'] = data.getDateStr(format='%Y-%m-%d_%H-%M-%S')
date_time = expt_info['Date and time']
experiment_name = expt_info['01. Experiment Name']
participant_id = expt_info['02. Participant Code']
data_folder = expt_info['04. Folder for saving data']
block_no = expt_info['05. Start from block no.']

# -- MAKE FOLDER/FILES TO SAVE DATA --
filename_core = experiment_name + '_' + participant_id + '_'
filename_prefix = date_time + '_' + filename_core
fm = FileManager(data_folder, filename_prefix)
fm.generate_infoFile(expt_info)

# -- SETUP STIMULUS CONTROL --
types = ['tap', 'stroke']
#types = ['stroke'] # demo

contact_areas = ['one finger tip', 'whole hand']
#contact_areas = ['whole hand']
# contact_areas = ['two finger pads'] # if stable recording
#contact_areas = ['whole hand'] # demo

#speeds = [3.0, 9.0, 18.0, 24.0] #cm/s
speeds = [3.0, 9.0, 18.0] # video
#speeds = [18.0] # video
#speeds = [3.0, 18.0] #cm/s # demo
#speeds = [1.0] #cm/s # if stable recording (before additional contact area)
# speeds = [6.0, 15.0, 21.0] #cm/s # lowest priority if stable recording

#forces = ['light', 'moderate', 'strong']
forces = ['light', 'strong'] # demo / video

stim_list = []
for type in types:
    for contact_area in contact_areas:
        for speed in speeds:
            for force in forces:
                stim_list.append({
                'type': type,
                'contact_area': contact_area,
                'speed': speed,
                'force': force
                })
                stim_list.append({
                'type': type,
                'contact_area': contact_area,
                'speed': speed,
                'force': force
                })

n_stim_per_block = len(speeds)*len(forces)*2
n_blocks = int(len(stim_list)/n_stim_per_block)

# -- SETUP AUDIO --
sounds_folder = "sounds"
#am = AudioManager(sounds_folder)

# -- SETUP EXPERIMENT CLOCKS --
expt_clock = core.Clock()
stim_clock = core.Clock()
short_clock = core.Clock()

# -- ABORT/EXIT ROUTINE --

def abort_experiment(key):
    if key == keyboard.Key.esc:
        try:
            ac.stop_signal()
        except:
            pass
        try:
            ac.close()
        except:
            pass
        try:
            kinect.stop_recording(0.5)
        except:
            pass
        fm.logEvent(expt_clock.getTime(), "experiment aborted")
        os._exit(0)

listener = keyboard.Listener(
    on_press=abort_experiment,
    on_release=abort_experiment)

listener.start()  # now the script will exit if you press escape

def space_to_continue(key):
    if key in ['space']:
        continue_cues = True

check_continue = keyboard.Listener(
    on_press=space_to_continue,
    on_release=space_to_continue)

check_continue.start()

# -- MAIN EXPERIMENT LOOP --
stim_no = (block_no-1)*n_stim_per_block # start with the first stimulus in the block
start_of_block = True
expt_clock.reset()
fm.logEvent(expt_clock.getTime(), "experiment started")
while stim_no < len(stim_list):

    continue_cues = False

    if start_of_block:

        start_of_block = False

    # pre-stimulus waiting period
    stim_clock.reset()
    while stim_clock.getTime() < 1.5:
        pass

    # print("press space to continue")
    # short_clock.reset()
    # while continue_cues == False:
    #     while short_clock.getTime() < 0.01:
    #         pass


    fm.logEvent(expt_clock.getTime(), "start")

    # metronome for timing during stimulus delivery
    if 1:
        ei = ExpertInterface(audioFolder="cues", imgFolder="img")
        ei.initialise(stim_list[stim_no]['type'],
                      stim_list[stim_no]['contact_area'],
                      stim_list[stim_no]['force'],
                      stim_list[stim_no]['speed'])
        ei.start_sequence()
        del ei

    fm.logEvent(
        expt_clock.getTime(),
        'stimulus presented: {}, {}cm/s, {}, {} force' .format(
            stim_list[stim_no]['type'],
            stim_list[stim_no]['speed'],
            stim_list[stim_no]['contact_area'],
            stim_list[stim_no]['force'],
        )
    )

    fm.logEvent(expt_clock.getTime(), "stop")

    # write to data file
    fm.dataWrite([
        stim_no+1,
        stim_list[stim_no]['type'],
        stim_list[stim_no]['speed'],
        stim_list[stim_no]['contact_area'],
        stim_list[stim_no]['force'],
        block_no,
        "no kinect file"
    ])

    fm.logEvent(
        expt_clock.getTime(),
        "stimulus {} of {} complete (in block {})" .format(stim_no+1, len(stim_list), block_no)
    )
    stim_no += 1

    # check if it's the end of the block
    if stim_no % n_stim_per_block == 0:

        fm.logEvent(
            expt_clock.getTime(),
            "block {} of {} complete" .format(block_no, n_blocks)
        )
        block_no += 1
        start_of_block = True

fm.logEvent(expt_clock.getTime(), "Experiment finished")
