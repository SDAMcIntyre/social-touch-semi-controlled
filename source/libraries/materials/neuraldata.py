import chardet
import numpy as np
import os
import pandas as pd
import re
import warnings

class NeuralData:
    def __init__(self, data_filename, unit_name2type_filename):
        self._time: list[float] = []
        self.nsample: int = 0
        self.data_Fs = None  # Hz
        self.spike: list[float] = []
        self.iff: list[float] = []
        self.TTL: list[float] = []

        self.unit_id = None
        self.unit_type = None
        self.conduction_vel = None  # meter/second
        self.dist_electrode = None  # cm
        self.latency = None  # second

        try:
            # get current unit ID
            filename = os.path.basename(data_filename)
            # Use re.search to find the first match of the pattern in the filename
            match = re.search(r'ST\d{2}-\d{2}', filename)
            if match:
                # Extract the matched substring
                self.unit_id = match.group()
        except:
            pass

        try:
            # select the metadata information of the current neuron
            df = pd.read_csv(unit_name2type_filename)
            df = df[df.Unit_name == self.unit_id]
            # get current unit type
            self.unit_type = df["Unit_type"].values[0]
            # get current conduction velocity
            self.conduction_vel = df["conduction_velocity (m/s)"].values[0]
            self.dist_electrode = df["electrode_endorgan_distance (cm)"].values[0]
            self.latency = (self.dist_electrode/100) / self.conduction_vel
        except:
            pass

    def shift_tmp(self, lag):
        lag = int(lag)
        if lag > 0:
            self.spike = np.pad(self.spike, (lag, 0), 'constant')[:len(self.spike)]
            self.iff = np.pad(self.iff, (lag, 0), 'constant')[:len(self.iff)]
        elif lag < 0:
            self.spike = np.pad(self.spike, (0, -lag), 'constant')[-lag:]
            self.iff = np.pad(self.iff, (0, -lag), 'constant')[-lag:]

    def get_data_idx(self, idx):
        neural = NeuralData("", "")
        neural.time = self.time[idx]
        neural.spike = self.spike[idx]
        neural.iff = self.iff[idx]
        neural.TTL = self.TTL[idx]

        neural.unit_id = self.unit_id
        neural.unit_type = self.unit_type
        neural.conduction_vel = self.conduction_vel
        neural.dist_electrode = self.dist_electrode
        neural.latency = self.latency

        return neural

    def set_data_idx(self, idx):
        self.time = self.time[idx]
        self.spike = self.spike[idx]
        self.iff = self.iff[idx]
        self.TTL = self.TTL[idx]

    def append(self, neural_bis):
        self.time = np.concatenate((self.time, neural_bis.time))
        self.spike = np.concatenate((self.spike, neural_bis.spike))
        self.iff = np.concatenate((self.iff, neural_bis.iff))
        self.TTL = np.concatenate((self.TTL, neural_bis.TTL))

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        self._time = value
        self.nsample = len(self._time)
        dt = np.diff(self._time)
        if not len(dt) == 0:
            self.data_Fs = 1 / np.nanmean(dt)  # Hz
            #print(np.nanmean(dt))
            #print(self.data_Fs)
