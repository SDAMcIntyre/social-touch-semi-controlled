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
filename_core = experiment_name + '_' + participant_id + '_' + '{}' .format(unit_name)
filename_prefix = date_time + '_' + filename_core
fm = FileManager(data_folder, filename_prefix)
fm.generate_infoFile(expt_info)

# -- SETUP STIMULUS CONTROL --
types = ['stroke','tap']
contact_areas = ['one finger tip', 'whole hand', 'two finger pads']
speeds = [1.0, 3.0, 6.0, 9.0, 15.0, 18.0, 21.0, 24.0] #cm/s
forces = ['light', 'moderate', 'strong']

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

n_stim_per_block = len(speeds)*len(forces)
n_blocks = int(len(stim_list)/n_stim_per_block)

# -- SETUP AUDIO --
sounds_folder = "sounds"
#am = AudioManager(sounds_folder)

# -- SETUP TRIGGER BOX CONNECTION --
ac = ArduinoComm()

# -- SETUP KINECT CONNECTION --
kinect_recorder_path = r'C:\Program Files\Azure Kinect SDK v1.2.0\tools'
kinect = KinectComm(kinect_recorder_path, fm.data_folder)

# -- MAIN EXPERIMENT LOOP --
stim_no = (block_no-1)*n_stim_per_block
start_of_block = True
stim_clock = core.Clock()
expt_clock = core.Clock()
fm.logEvent(expt_clock.getTime(),"experiment started")
while stim_no < len(stim_list):

    if start_of_block:

        # start kinect recording
        kinect.start_recording(filename_core + '_block{}' .format(block_no))
        kinect_start_time = expt_clock.getTime()
        fm.logEvent(kinect_start_time, "kinect started recording {}" .format(kinect.filename))

        kinect_start_delay = 2.0 # how long to wait to make sure the kinect has started
        while expt_clock.getTime() < kinect_start_time + kinect_start_delay:
            pass

        # trigger/sync signal -> TTL to nerve recording and LED to camera (make sure it is visible)
        fm.logEvent(
            expt_clock.getTime(),
            "sending {} pulses for block number" .format(block_no)
        )
        ac.send_pulses(block_no)

        start_of_block = False

    ac.send_on_signal()
    fm.logEvent(expt_clock.getTime(), "TTL/LED on")

    # pre-stimulus waiting period
    stim_clock.reset()
    while stim_clock.getTime() < 0.1:
        pass

    # metronome for timing during stimulus delivery
    #
    fm.logEvent(
        expt_clock.getTime(),
        'stimulus presented: {}, {}cm/s, {}, {} force' .format(
            stim_list[stim_no]['type'],
            stim_list[stim_no]['speed'],
            stim_list[stim_no]['contact_area'],
            stim_list[stim_no]['force'],
        )
    )

    # stand-in for stimulus duration
    stim_clock.reset()
    while stim_clock.getTime() < 0.1:
        pass

    # trigger/sync signal off

    ac.stop_signal()
    fm.logEvent(expt_clock.getTime(), "TTL/LED off")

    # write to data file
    fm.dataWrite(
        stim_no+1,
        stim_list[stim_no]['type'],
        stim_list[stim_no]['speed'],
        stim_list[stim_no]['contact_area'],
        stim_list[stim_no]['force'],
        block_no,
        kinect.filename
    )

    fm.logEvent(
        expt_clock.getTime(),
        "stimulus {} of {} complete (in block {})" .format(stim_no+1, len(stim_list), block_no)
    )
    stim_no += 1

    # check if it's the end of the block
    if stim_no % n_stim_per_block == 0:

        # Kinect off
        kinect.stop_recording(2)  # stop recording with a delay
        fm.logEvent(expt_clock.getTime(), "Kinect stopped")

        fm.logEvent(
            expt_clock.getTime(),
            "block {} of {} complete" .format(block_no, n_blocks)
        )
        block_no += 1
        start_of_block = True

fm.logEvent(expt_clock.getTime(), "Experiment finished")
ac.trigger.ser.close()