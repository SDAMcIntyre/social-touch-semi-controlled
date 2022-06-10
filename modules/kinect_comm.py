
import subprocess
import os
from subprocess import Popen, PIPE, CREATE_NEW_CONSOLE
import keyboard
# import time
from psychopy import data, core

# Interact with the Virginia's software


class KinectComm:

    def __init__(self, _locationScript, _outputDirectory):

        self.scriptPath = _locationScript + '\\k4arecorder.exe'
        self.outputDir = r'' + os.path.realpath(_outputDirectory)
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)

        self.filename = self.outputDir + '\\_.mkv'
        self.process = None

    def start_recording(self, filename_core):
        filename_prefix = (data.getDateStr(format='%Y-%m-%d_%H-%M-%S') + '_' + filename_core)

        self.filename = self.outputDir + '\\' + filename_prefix + '.mkv'

        self.process = subprocess.Popen(
            [self.scriptPath, self.filename],
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
        return self

    def stop_recording(self, delay):
        delay_timer = core.CountdownTimer()
        delay_timer.add(delay)
        # wait for delay
        while delay_timer.getTime() > 0:
            pass
        keyboard.press_and_release('ctrl+c')
        self.process.wait()

        return self

    def is_stopped(self):
        return self.process.poll() is None

    # import signal
    # self.p.send_signal(signal.CTRL_C_EVENT)

    def record_trial(self, filename_core, duration):
        trial_timer = core.CountdownTimer()
        trial_timer.add(duration)
        self.start_recording(filename_core)
        # wait for recording duration
        while trial_timer.getTime() > 0:
            pass
            # keyboard.record('esc')

        self.stop_recording()
        return self
