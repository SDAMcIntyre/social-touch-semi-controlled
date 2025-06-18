import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import warnings

from ..materials.metadata import Metadata  # noqa: E402
from ..materials.stimulusinfo import StimulusInfo  # noqa: E402
from ..materials.contactdata import ContactData  # noqa: E402
from ..materials.neuraldata import NeuralData  # noqa: E402
#import ..misc.time_cost_function as time_cost
#from libraries.misc.semicontrolled_data_visualizer import SemiControlledDataVisualizer  # noqa: E402


class SemiControlledData:
    def __init__(self, data_csv_filename, md_stim_filename="", md_neuron_filename="", load_instant=False, dropna=False):
        self.md: Metadata = Metadata(data_csv_filename, md_stim_filename, md_neuron_filename)
        self.stim: StimulusInfo = StimulusInfo(md_stim_filename)
        self.neural: NeuralData = NeuralData(data_csv_filename, md_neuron_filename)
        self.contact: ContactData = ContactData()

        # allows to determine if the signal is considered good
        # based on expected stimulus information given to the experimenter
        self.trust_score = 0

        if load_instant:
            df = self.load_dataframe(dropna=dropna)
            self.set_variables(df)

    # Create a SemiControlledData for each dataframe of the list
    # presumably each element of the input list is a trial
    def create_list_from_df(self, df_list):
        data_trials_out = []
        for df in df_list:
            scd = SemiControlledData(self.md.data_filename, self.md.unit_name2type_filename)
            scd.set_variables(df)
            data_trials_out.append(scd)
        return data_trials_out

    # create a dataframe of the current semicontrolled data
    def asDataFrame(self):
        pass

    def get_stimulusInfoContact(self):
        type_ = self.stim.type
        velocity = self.stim.vel
        force = self.stim.force
        size = self.stim.size
        return type_, velocity, force, size

    def estimate_contact_averaging(self):
        # define more precisely the location of the contact
        touch_location = self.contact.get_contact_mask(self.stim.curr_max_vel_ratio())
        scd_touch = self.get_data_idx(touch_location)
        v = np.mean(scd_touch.get_instantaneous_velocity())
        d = np.mean(scd_touch.get_depth())
        a = np.mean(scd_touch.get_area())
        return v, d, a

    def get_instantaneous_velocity(self):
        return self.contact.get_instantaneous_velocity()

    def get_depth(self):
        return self.contact.get_depth()

    def get_area(self):
        return self.contact.get_area()

    def set_variables(self, df=None, dropna=False, ignorestimulus=False):
        if df is None:
            df = self.load_dataframe(dropna=dropna)
        self.load_metadata(df)
        if not(ignorestimulus):
            self.load_stimulus()
        self.load_contact(df)
        self.load_neural(df)

    def load_dataframe(self, dropna=False):
        df = pd.read_csv(self.md.data_filename)
        # remove lines that contains NaN values
        if dropna:
            df.dropna(inplace=True)
        return df

    def load_metadata(self, df):
        # metadata
        self.md.time = df.time.values

    def load_stimulus(self):
        if self.md.md_stim_filename == "":
            warnings.warn(f"metadata stimulus filename doesn't exist: Ignore stimulus characteristics.")
            return

        df = pd.read_csv(self.md.md_stim_filename)
        current_row = df[df['trial_id'] == self.md.trial_id]
        # stimulus info
        self.stim.type = current_row.type.values[0]
        self.stim.vel = current_row.speed.values[0]
        self.stim.size = current_row.contact_area.values[0]
        self.stim.force = current_row.force.values[0]

    def load_contact(self, df):
        self.contact.time = df["time"].values

        # Kinect LED time series
        if 'LED on' in df.columns: 
            self.contact.TTL = df["LED on"].values
        # contact data
        self.contact.contact_flag = df["contact_detected"].values
        self.contact.area = df["contact_area"].values
        self.contact.depth = df["contact_depth"].values
        # some dataset doesn't possess the position anymore
        if "hand" in self.stim.size and 'palm_position_x' in df.columns:  # if the hand is used, take the hand tracker
            px = df["palm_position_x"].values
            py = df["palm_position_y"].values
            pz = df["palm_position_z"].values
            self.contact.pos = np.array([px, py, pz])
        elif 'index_position_x' in df.columns:
            px = df["index_position_x"].values
            py = df["index_position_y"].values
            pz = df["index_position_z"].values
            self.contact.pos = np.array([px, py, pz])
        else:
            pass
        # some dataset doesn't possess the velocity anymore
        if 'velLongRaw' in df.columns:
            vx = df.velLongRaw.values
            vy = df.velLatRaw.values
            vz = df.velVertRaw.values
            self.contact.vel = np.array([vx, vy, vz])
        else:
            pass
        # some dataset doesn't possess the forearm contact XYZ 
        if 'contact_arm_pointcloud' in df.columns:
            self.contact.arm_pointcloud = df['contact_arm_pointcloud'].values

    def load_neural(self, df):
        self.neural.time = df.time.values
        # neural data
        self.neural.spike = df.Nerve_spike.values
        self.neural.iff = df.Nerve_freq.values
        try:
            self.neural.TTL = df["Nerve_TTL"].values
        except KeyError:
            self.neural.TTL = np.nan

    def get_data_idx(self, idx, hardcopy=False):
        if hardcopy:
            # copy.deepcopy is very heavy (cost 50 ms)
            scd = copy.deepcopy(self)
            scd.set_data_idx(idx)
        else:
            # cost 0.2 ms
            scd = SemiControlledData(self.md.data_filename, self.md.md_stim_filename, self.md.unit_name2type_filename)
            scd.stim = self.stim.get_data()
            scd.md = self.md.get_data_idx(idx)
            scd.contact = self.contact.get_data_idx(idx)
            scd.neural = self.neural.get_data_idx(idx)

        return scd

    def set_data_idx(self, idx):
        if isinstance(idx, range):
            idx = list(idx)
        elif isinstance(idx, tuple):
            idx = list(range(idx[0], idx[1]))

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
        if self.stim.type == "tap":
            m = self.contact.get_contact_mask(vel_ratio, mode="soft")
        elif self.stim.type == "stroke":
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





