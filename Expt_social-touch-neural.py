import os
from psychopy import core, data, gui

os.chdir(os.path.dirname(os.path.realpath(__file__)))
from modules.ui_management import UserInterfaceExpt, ui_get_initialData

# -- GET INPUT FROM THE EXPERIMENTER --
expt_info = {
    'Participant Code': 'ST12',
    'Unit Number': '0',
    'Number of repeats': 30
    }
dlg = gui.DlgFromDict(expt_info, title='Experiment details')
# add the time when the user pressed enter:
expt_info['Date and time'] = data.getDateStr(format='%Y-%m-%d_%H-%M-%S')
if dlg.OK:
    pass  # continue
else:
    core.quit()  # the user hit cancel so exit

# -- SETUP STIMULUS CONTROL --
types = ['tap','stroke']
speeds = [1.0, 3.0, 6.0, 9.0, 15.0, 18.0, 21.0, 24.0] #cm/s
contact_areas = ['one finger tip', 'two finger pads', 'whole hand']
forces = ['light', 'moderate', 'strong']

stim_list = []
for type in types:
    for speed in speeds:
        for contact_area in contact_areas:
            for force in forces:
                stim_list.append({
                    'type': type,
                    'speed': speed,
                    'contact_area': contact_area,
                    'force': force
                })

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
