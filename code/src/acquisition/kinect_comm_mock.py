
import subprocess
import os
from subprocess import Popen, PIPE, CREATE_NEW_CONSOLE
import keyboard
# import time
from psychopy import data, core

# Interact with the Virginia's software
from .kinect_comm import KinectComm


class KinectCommMock(KinectComm):

    def __init__(self, _locationScript, _outputDirectory):
        self.scriptPath = _locationScript + '\\k4arecorder.exe'
        self.outputDir = r'' + os.path.realpath(_outputDirectory)
        self.filename = self.outputDir + '\\_.mkv'
        self.process = None

    def start_recording(self, filename_core):
        filename_prefix = (data.getDateStr(format='%Y-%m-%d_%H-%M-%S') + '_' + filename_core)

        self.filename = self.outputDir + '\\' + filename_prefix + '.mkv'

        return self

    def stop_recording(self, delay):
        return self

    def is_stopped(self):
        return True

    # import signal
    # self.p.send_signal(signal.CTRL_C_EVENT)

    def record_trial(self, filename_core, duration):
        return self
