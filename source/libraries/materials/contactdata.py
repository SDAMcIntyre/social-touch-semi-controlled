import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from .metadata import Metadata  # noqa: E402
from libraries.materials.stimulusinfo import StimulusInfo  # noqa: E402


class ContactData:
    def __init__(self):
        self._time: list[float] = []
        self.nsample: int = 0

        self._led_on: list[float] = []
        self._green_levels: list[float] = []

        self._contact_flag: list[float] = []  # ON/OFF

        self._area: list[float] = []  # mm^2
        self._depth: list[float] = []  # mm

        self._pos: list[float] = []  # mm

        self._pos_1D: list[float] = []  # mm
        self._vel: list[float] = []  # mm/sec

        self.data_Fs = None  # Hz
        self.KINECT_FS = 30  # Hz

    def get_contact_mask(self, curr_max_vel_ratio, mode="hard"):
        depth_smooth = self.get_smooth_depth(curr_max_vel_ratio, nframe=1)
        return depth_smooth > 0

    def get_smooth_depth(self, curr_max_vel_ratio, nframe=1):
        '''get_smooth_signal
            smooth the signal based on the expected speed of the trial:
             - more smoothing for lower speed
             - return an array of identical size
        '''
        oversampling_to_trueFs = (self.data_Fs / self.KINECT_FS)
        window_size = nframe * oversampling_to_trueFs
        # between 0 and 1 of the max speed
        speed_multiplier = 1 / curr_max_vel_ratio
        # adjust the window based on the ratio
        window_size = int(window_size * speed_multiplier)

        # avoid np.convolve to output an array larger than self.depth
        if window_size > len(self.depth):
            window_size = len(self.depth)

        weights = np.repeat(1.0, window_size) / window_size
        depth_smooth = np.convolve(self.depth, weights, 'same')

        return depth_smooth

    def get_instantaneous_velocity(self, unit="cm/sec"):
        time_intervals = np.diff(self.time)
        velocity = np.array([0] * len(time_intervals))
        for idx, dt in enumerate(time_intervals):
            # calculate the euclidian distance between two consecutive points
            displacement = np.sqrt(np.sum((self.pos[:, idx] - self.pos[:, idx + 1]) ** 2))
            # get instantaneous velocity between two points
            velocity[idx] = displacement / dt

        match unit:
            case "cm/sec":
                velocity = velocity / 10
            case _:
                pass
        return velocity

    def get_depth(self):
        depth_values = []
        try:
            contacting = self.depth > 0
            depth_values = self.depth[contacting]
        except:
            warnings.warn("an error occured in <get_depth>")
        return depth_values

    def get_area(self):
        area_values = []
        try:
            contacting = self.area > 0
            area_values = self.area[contacting]
        except:
            warnings.warn("an error occured in <get_area>")

        return area_values

    def get_data_idx(self, idx):
        cd = ContactData()

        cd.time = self.time[idx]
        cd.led_on = self.led_on[idx]
        cd.green_levels = self.green_levels[idx]

        cd.contact_flag = self.contact_flag[idx]
        cd.area = self.area[idx]
        cd.depth = self.depth[idx]
        cd.vel = self.vel[:, idx]
        try:
            cd.pos = self.pos[:, idx]
        except:
            pass
        return cd

    def set_data_idx(self, idx):
        self.time = self.time[idx]

        self.led_on = self.led_on[idx]
        self.green_levels = self.green_levels[idx]

        self.contact_flag = self.contact_flag[idx]
        self.area = self.area[idx]
        self.depth = self.depth[idx]
        self.vel = self.vel[:, idx]
        try:
            self.pos = self.pos[:, idx]
        except:
            pass

    def append(self, contact_bis):
        self.time = np.concatenate((self.time, contact_bis.time))

        self.led_on = np.concatenate((self.led_on, contact_bis.led_on))
        self.green_levels = np.concatenate((self.green_levels, contact_bis.green_levels))

        self.contact_flag = np.concatenate((self.contact_flag, contact_bis.contact_flag))
        self.area = np.concatenate((self.area, contact_bis.area))
        self.depth = np.concatenate((self.depth, contact_bis.depth))
        self.vel = np.concatenate((self.vel, contact_bis.vel), axis=1)
        try:
            self.pos = np.concatenate((self.pos, contact_bis.pos), axis=1)
        except:
            pass

        self._led_on: list[float] = []
        self._green_levels: list[float] = []

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        self._time = value
        self.nsample = len(self._time)
        self.data_Fs = 1 / np.median(np.diff(self._time))  # Hz

    @property
    def led_on(self):
        return self._led_on

    @led_on.setter
    def led_on(self, value):
        self._led_on = value

    @property
    def green_levels(self):
        return self._green_levels

    @green_levels.setter
    def green_levels(self, value):
        self._green_levels = value

    @property
    def contact_flag(self):
        return self._contact_flag

    @contact_flag.setter
    def contact_flag(self, value):
        self._contact_flag = value

    @property
    def area(self):
        return self._area

    @area.setter
    def area(self, value):
        self._area = value

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, value):
        self._depth = value

    @property
    def vel(self):
        return self._vel

    @vel.setter
    def vel(self, value):
        self._vel = value

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value
        self.update_pos_1D()

    def update_pos_1D(self):
        # compress the signal into 1D (as the expected motion is supposed to be 1D anyway)
        pca = PCA(n_components=1)
        p = np.squeeze(pca.fit_transform(self.pos.transpose()))
        self.pos_1D = p

    @property
    def pos_1D(self):
        return self._pos_1D

    @pos_1D.setter
    def pos_1D(self, value):
        self._pos_1D = value
