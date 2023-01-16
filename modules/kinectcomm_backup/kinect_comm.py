import subprocess
from subprocess import Popen, PIPE, CREATE_NEW_CONSOLE
import keyboard
import time
from psychopy import data
import os


# Interact with the Virginia's software


class KinectComm:

    def __init__(self, _locationScript, _outputDirectory):
        scriptName = 'k4arecorder.exe'
        self.scriptPath = _locationScript + '\\' + scriptName
        self.outputDir = _outputDirectory
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
            creationflags=subprocess.CREATE_NEW_CONSOLE)
        return self

    def stop_recording(self, delay):
        time.sleep(delay)
        done = False
        n_trials = 0
        while not done:
            try:
                keyboard.press_and_release('ctrl+c')
                done = self.process.wait(timeout=1) is not None
            except:
                pass
            n_trials += 1
            print("waiting...", n_trials)
        print("terminated!")
        #self.process.wait()

        return self

    def is_stopped(self):
        return self.process.poll() is None
