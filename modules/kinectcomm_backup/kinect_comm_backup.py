
import subprocess
import os
from subprocess import Popen, PIPE, CREATE_NEW_CONSOLE, CREATE_NEW_PROCESS_GROUP
from signal import CTRL_C_EVENT
import keyboard as kb
# import time
from psychopy import data, core
from modules.win_manager import WindowMgr

# Interact with the Virginia's software


class KinectComm:

    def __init__(self, _locationScript, _outputDirectory):

        self.scriptPath = _locationScript + '\\k4arecorder.exe'
        self.outputDir = r'' + os.path.realpath(_outputDirectory)
        if not os.path.exists(self.outputDir):
            os.makedirs(self.outputDir)

        self.filename = self.outputDir + '\\_.mkv'
        self.process = None

    def record_trial(self, filename_core, duration):
        trial_timer = core.CountdownTimer()
        trial_timer.add(duration)
        self.start_recording(filename_core)
        # wait for recording duration
        while trial_timer.getTime() > 0:
            pass

        self.stop_recording()
        return self

    def start_recording(self, filename_core):
        filename_prefix = (data.getDateStr(format='%Y-%m-%d_%H-%M-%S') + '_' + filename_core)

        self.filename = self.outputDir + '\\' + filename_prefix + '.mkv' + "-l 20"

        self.process = subprocess.Popen(
            [self.scriptPath, self.filename],
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            creationflags= CREATE_NEW_CONSOLE | CREATE_NEW_PROCESS_GROUP
        )
        return self

    def bring_to_front(self):
        w = WindowMgr()
        w.find_window_wildcard(".*k4arecorder.*")
        w.set_foreground()

        return self

    def stop_recording(self, delay):
        delay_timer = core.CountdownTimer()
        delay_timer.reset()
        delay_timer.add(delay)
        # wait for delay
        while delay_timer.getTime() > 0:
            pass
        self.bring_to_front()
        kb.press_and_release('ctrl+c')
        #self.process.send_signal(CTRL_C_EVENT)
        #self.process.terminate()
        self.process.wait()

        return self

    def is_stopped(self):
        return self.process.poll() is None
