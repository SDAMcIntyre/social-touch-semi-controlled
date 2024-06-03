import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import warnings

from libraries.materials.metadata import Metadata  # noqa: E402
from libraries.materials.stimulusinfo import StimulusInfo  # noqa: E402
from libraries.materials.contactdata import ContactData  # noqa: E402
from libraries.materials.neuraldata import NeuralData  # noqa: E402
import libraries.misc.time_cost_function as time_cost
#from libraries.misc.semicontrolled_data_visualizer import SemiControlledDataVisualizer  # noqa: E402


class SemiControlledData:
    def __init__(self, csv_filename, loading_process=None):
        self.md: Metadata = Metadata(csv_filename)
        self.stim: StimulusInfo = StimulusInfo()
        self.neural: NeuralData = NeuralData(csv_filename)

        self.contact: ContactData = ContactData()

        # allows to determine if the signal is considered good
        # based on expected stimulus information given to the experimenter
        self.trust_score = 0

        match loading_process:
            case "automatic_load":
                df = self.load_dataframe()
                self.set_variables(df)

    def get_stimulusInfoContact(self):
        type_ = self.stim.type
        velocity = self.stim.vel
        force = self.stim.force
        size = self.stim.size
        return type_, velocity, force, size

    def estimate_contact_averaging(self):
        # define more precisely the location of the contact
        touch_location = self.get_singular_touch_location()
        scd_touch = self.get_data_idx(touch_location)
        v = np.mean(scd_touch.get_instantaneous_velocity())
        d = np.mean(scd_touch.get_depth())
        a = np.mean(scd_touch.get_area())
        return v, d, a

    def get_singular_touch_location(self):
        return self.contact.get_contact_mask(self.stim.curr_max_vel_ratio())

    def get_instantaneous_velocity(self):
        return self.contact.get_instantaneous_velocity()

    def get_depth(self):
        return self.contact.get_depth()

    def get_area(self):
        return self.contact.get_area()

    def set_variables(self, df=None):
        if df is None:
            df = self.load_dataframe()
        self.load_metadata(df)
        self.load_stimulus(df)
        self.load_contact(df)
        self.load_neural(df)

    def load_dataframe(self):
        df = pd.read_csv(self.md.csv_filename)
        # remove lines that contains NaN values
        df.dropna(inplace=True)
        return df

    def load_metadata(self, df):
        # metadata
        self.md.time = df.t.values
        self.md.block_id = df.block_id.values[0]
        self.md.trial_id = df.trial_id.values[0]

    def load_stimulus(self, df):
        # stimulus info
        self.stim.type = df.stimulus.values[0]
        self.stim.vel = df.vel.values[0]
        self.stim.size = df.finger.values[0]
        self.stim.force = df.force.values[0]

    def load_contact(self, df):
        self.contact.time = df.t.values
        # contact data
        self.contact.area = df.areaRaw.values
        self.contact.depth = df.depthRaw.values
        vx = df.velLongRaw.values
        vy = df.velLatRaw.values
        vz = df.velVertRaw.values
        self.contact.vel = np.array([vx, vy, vz])
        # some dataset doesn't possess the position anymore
        try:
            px = df.Position_x.values
            py = df.Position_y.values
            pz = df.Position_z.values
            self.contact.pos = np.array([px, py, pz])
        except:
            pass

    def load_neural(self, df):
        self.neural.time = df.t.values
        # neural data
        self.neural.spike = df.spike.values
        self.neural.iff = df.IFF.values

    def get_data_idx(self, idx, hardcopy=False):
        if hardcopy:
            # copy.deepcopy is very heavy (cost 50 ms)
            scd = copy.deepcopy(self)
            scd.set_data_idx(idx)
        else:
            # cost 0.2 ms
            scd = SemiControlledData(self.md.csv_filename)
            scd.stim = self.stim.get_data()
            try:
                scd.md = self.md.get_data_idx(idx)
                scd.contact = self.contact.get_data_idx(idx)
                scd.neural = self.neural.get_data_idx(idx)
            except:
                print("hm.")

        return scd

    def set_data_idx(self, idx):
        self.md.set_data_idx(idx)
        self.contact.set_data_idx(idx)
        self.neural.set_data_idx(idx)

    def append(self, scd_bis):
        if not self.stim.is_similar(scd_bis.stim):
            warnings.warn("Warning: stimulus variables are not equal.", UserWarning)
        self.md.append(scd_bis.md)
        self.contact.append(scd_bis.contact)
        self.neural.append(scd_bis.neural)

    def get_contact_mask(self):
        m = None
        vel_ratio = self.stim.curr_max_vel_ratio()
        match self.stim.type:
            case "tap":
                m = self.contact.get_contact_mask(vel_ratio, mode="soft")
            case "stroke":
                m = self.contact.get_contact_mask(vel_ratio, mode="hard")
        return m

    def get_duration_ratio(self):
        duration_recorded = 1000 * (self.md.time[-1] - self.md.time[0])
        duration_expected = 1000 * self.stim.get_single_contact_duration_expected()
        return float(duration_recorded) / duration_expected

    def define_trust_score(self):
        score = .5 * self.trust_score_duration() + .5 * self.trust_score_position()
        self.trust_score = score
        return score

    def trust_score_duration(self,):
        duration_recorded = 1000 * (self.md.time[-1] - self.md.time[0])
        duration_expected = 1000 * self.stim.get_single_contact_duration_expected()

        # Calculate the percentage difference
        difference = abs(duration_recorded - duration_expected)
        average = (duration_recorded + duration_expected) / 2
        percentage_diff = (difference / average)

        # inverted sigmoid: the closer to 0 difference, the  higher the score is
        score = sigmoid_inv(x=percentage_diff, a=0, b=1, c=.4, d=0.2)

        return score

    def trust_score_position(self):
        return 1


def sigmoid_inv(x, a, b, c, d):
    y = a + b / (1 + np.exp(1*(x-c)/d))
    return y





