import os
from psychopy import core, data, gui

from modules.arduino_comm import ArduinoComm
from modules.kinect_comm import KinectComm
from modules.file_management import FileManager
from modules.audio_management import AudioManager

script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

# -- GET INPUT FROM THE EXPERIMENTER --
expt_info = {
    '01. Experiment Name': 'controlled-touch-MNG',
    '02. Participant Code': 'ST12',
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
unit_name = expt_info['03. Unit Number']
data_folder = expt_info['04. Folder for saving data']
block_no = expt_info['05. Start from block no.']

# -- MAKE FOLDER/FILES TO SAVE DATA --
filename_core = experiment_name + '_' + participant_id + '_' + str(unit_name)
filename_prefix = date_time + '_' + filename_core
fm = FileManager(data_folder, filename_prefix)
fm.generate_infoFile(expt_info)

# -- SETUP STIMULUS CONTROL --
types = ['tap', 'stroke']
contact_areas = ['one finger tip', 'two finger pads', 'whole hand']
speeds = [1.0, 3.0, 6.0, 9.0, 15.0, 18.0, 21.0, 24.0]  # cm/s
forces = ['light', 'moderate', 'strong']

stim_list = []
for t in types:
    for contact_area in contact_areas:
        for speed in speeds:
            for force in forces:
                stim_list.append({
                    'type': t,
                    'contact_area': contact_area,
                    'speed': speed,
                    'force': force
                })

n_stim_per_block = len(speeds) * len(forces)
n_blocks = int(len(stim_list) / n_stim_per_block)

# -- SETUP AUDIO --
sounds_folder = "sounds"
am = AudioManager(sounds_folder)

# -- SETUP TRIGGER BOX CONNECTION --
ac = ArduinoComm()

# -- SETUP KINECT CONNECTION --
kinect_recorder_path = r'C:\Program Files\Azure Kinect SDK v1.2.0\tools'
kinect = KinectComm(kinect_recorder_path, fm.data_folder)

# -- MAIN EXPERIMENT LOOP --
stim_no = (block_no - 1) * n_stim_per_block
expt_clock = core.Clock()
fm.logEvent(expt_clock, "experiment started")
while stim_no < len(stim_list):

    # check if it's time to iterate the block no.
    if stim_no % n_stim_per_block == 0:
        fm.logEvent(expt_clock, "block {} of {} complete".format(block_no, n_blocks))
        block_no += 1
# trigger kinect recording
#
# visual and/or audio cue with the name of the upcoming stimulus
#
# trigger/sync signal -> TTL to nerve recording and LED to camera (make sure it is visible)
#
# metronome for timing during stimulus delivery
#
# trigger/sync signal off
#
# Kinect off
#
# log files
#
# can cancel at any time, saves data so far (data saved after every stimulus)
#
# filename: semi-controlled_date_time_participant_unit
#
# (consider including multiple stimuli per kinect recording - trade-off between faster experiment and easier tracking vs. failures affect fewer stimulus presentations)
#
