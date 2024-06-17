import chardet
import numpy as np
import pandas as pd
import warnings

class NeuralData:
    def __init__(self, data_filename, unit_name2type_filename):
        self._time: list[float] = []
        self.nsample: int = 0
        self.data_Fs = None  # Hz
        self.spike: list[float] = []
        self.iff: list[float] = []

        self.unit_id = None
        self.unit_type = None
        self.conduction_vel = None  # meter/second
        self.dist_electrode = None  # cm
        self.latency = None  # second

        try:
            # get current unit ID
            filename_split = data_filename.split('/')
            csv_split = filename_split[-1].split('-')
            # Step 4: Find the index of the element that contains "ST"
            index = next((i for i, element in enumerate(csv_split) if "ST" in element), -1)

            participant_id = csv_split[index]
            unit_occ = csv_split[index + 1].replace("unit", "")
            if int(unit_occ) < 10:
                unit_occ = "0" + unit_occ
            self.unit_id = '-'.join([participant_id, unit_occ])
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

    def correct_conduction_velocity(self):
        if self.data_Fs is None:
            warnings.warn("Neural:correct_conduction_velocity> data hasn't been loaded yet. Abort.")
        # get the number of sample for the latency of the current unit
        nsample_latency = -1 * int(self.data_Fs * self.latency)
        # shift the signal by the latency
        self.shift(nsample_latency)

    def shift(self, lag):
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

    def append(self, neural_bis):
        self.time = np.concatenate((self.time, neural_bis.time))
        self.spike = np.concatenate((self.spike, neural_bis.spike))
        self.iff = np.concatenate((self.iff, neural_bis.iff))

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        self._time = value
        self.nsample = len(self._time)
        dt = np.diff(self._time)
        self.data_Fs = 1 / np.median(dt, skipna=True)  # Hz
