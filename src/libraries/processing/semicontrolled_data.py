from itertools import groupby
import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
import seaborn as sns

from .metadata import Metadata  # noqa: E402
from .stimulusinfo import StimulusInfo  # noqa: E402
from .contactdata_raw import ContactDataRaw  # noqa: E402
from .neuraldata import NeuralData  # noqa: E402


class SemiControlledData:
    def __init__(self, csv_filename, split_by_trial=False, automatic_load=False):
        self.md = Metadata(csv_filename)
        self.stim = StimulusInfo()
        self.contact = ContactDataRaw()
        self.neural = NeuralData()

        if split_by_trial:
            self.split_by_trial()
        else:
            if automatic_load:
                self.set_variables()

    def split_by_trial(self):
        df = self.load_data_from_file()
        scd_trials = []

        # Separate the rows of each trial
        for key, group in groupby(enumerate(df.trial_id), key=itemgetter(1)):
            indices = [index for index, _ in group]
            scd = SemiControlledData(self.md.csv_filename, split_by_trial=False, automatic_load=False)
            scd.set_variables(df.iloc[indices])
            scd_trials.append(scd)

        return scd_trials

    def set_variables(self, df=None):
        if df is None:
            df = self.load_data_from_file()
        self.load_metadata(df)
        self.load_stimulus(df)
        self.load_contact(df)
        self.load_neural(df)

    def load_data_from_file(self):
        df = pd.read_csv(self.md.csv_filename)
        # remove lines that contains NaN values
        df.dropna(inplace=True)
        return df

    def load_metadata(self, df):
        # metadata
        self.md.time = df.t.values
        self.md.trial_id = df.trial_id.values

    def load_stimulus(self, df):
        # stimulus info
        self.stim.type = df.stimulus.values
        self.stim.vel = df.vel.values
        self.stim.size = df.finger.values
        self.stim.force = df.force.values

    def load_contact(self, df):
        # contact data
        self.contact.area = df.areaRaw.values
        self.contact.depth = df.depthRaw.values
        vx = df.velLongRaw.values
        vy = df.velLatRaw.values
        vz = df.velVertRaw.values
        self.contact.vel = [vx, vy, vz]
        #px = df.Position_x.values
        #py = df.Position_y.values
        #pz = df.Position_z.values
        #self.contact.pos = [px, py, pz]

    def load_neural(self, df):
        # neural data
        self.neural.spike = df.spike.values
        self.neural.iff = df.IFF.values


