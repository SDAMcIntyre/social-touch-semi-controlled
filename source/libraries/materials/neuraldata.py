import numpy as np
import pandas as pd
import chardet

class NeuralData:
    def __init__(self, data_filename, unit_name2type_filename):
        self._time: list[float] = []
        self.nsample: int = 0
        self.data_Fs = None  # Hz
        self.spike: list[float] = []
        self.iff: list[float] = []

        self.unit_id = None
        self.unit_type = None

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

            # get current unit type
            df = pd.read_csv(unit_name2type_filename)
            self.unit_type = df.Unit_type[df.Unit_name == self.unit_id].values[0]
        except:
            pass

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
        return neural

    def set_data_idx(self, idx):
        self.time = self.time[idx]
        self.spike = self.spike[idx]
        self.iff = self.iff[idx]

        self.unit_id = self.unit_id
        self.unit_type = self.unit_type

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
        self.data_Fs = 1 / np.median(np.diff(self._time))  # Hz
