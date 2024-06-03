import numpy as np


class NeuralData:
    def __init__(self, csv_filename):
        self._time: list[float] = []
        self.nsample: int = 0
        self.data_Fs = None  # Hz

        try:
            filename_split = csv_filename.split('/')
            csv_split = filename_split[-1].split('-')
            # Step 4: Find the index of the element that contains "ST"
            index = next((i for i, element in enumerate(csv_split) if "ST" in element), -1)
            self.unit_id = '-'.join(csv_split[index:index+2])
        except:
            pass

        self.unit_type = None

        self.spike: list[float] = []
        self.iff: list[float] = []

    def get_data_idx(self, idx):
        neural = NeuralData("")
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
