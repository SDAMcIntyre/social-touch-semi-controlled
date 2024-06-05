import numpy as np
import warnings


class Metadata:
    def __init__(self, data_filename, unit_name2type_filename):
        self.data_filename = data_filename
        self.unit_name2type_filename = unit_name2type_filename

        self.block_id: float = 0
        self.trial_id: float = 0
        self._time: list[float] = []
        self.nsample: int = 0

        self.data_Fs = 0  # Hz

    def set_refreshRate(self):
        self.data_Fs = 1/np.median(np.diff(self._time))  # Hz

    def get_data_idx(self, idx):
        md = Metadata(self.data_filename, self.unit_name2type_filename)
        md.time = self.time[idx]
        md.block_id = self.block_id
        md.trial_id = self.trial_id
        return md

    def set_data_idx(self, idx):
        self.time = self.time[idx]
        self.block_id = self.block_id
        self.trial_id = self.trial_id

    def append(self, md_bis):
        self.time = np.concatenate((self.time, md_bis.time))
        if self.block_id != md_bis.block_id:
            warnings.warn("Warning: trial_ids are not equal.", UserWarning)
        if self.trial_id != md_bis.trial_id:
            warnings.warn("Warning: trial_ids are not equal.", UserWarning)

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        self._time = value
        self.nsample = len(self._time)
        self.set_refreshRate()
