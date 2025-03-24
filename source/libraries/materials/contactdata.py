import warnings
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA  # from pyppca import ppca  # pip install git+https://github.com/el-hult/pyppca

from .metadata import Metadata  # noqa: E402
from materials.stimulusinfo import StimulusInfo  # noqa: E402
from misc.interpolate_nan_values import interpolate_nan_values  # noqa: E402


class ContactData:
    def __init__(self):
        self._time: list[float] = []
        self.nsample: int = 0

        self._TTL: list = []  # float or nans
        self._green_levels: list = []  # float or nans

        self._contact_flag: list[float] = []  # ON/OFF

        self._area: list[float] = []  # mm^2
        self._depth: list[float] = []  # mm

        self._pos: list[float] = []  # mm

        self._pos_1D: list[float] = []  # mm
        self._vel: list[float] = []  # mm/sec

        self.data_Fs = None  # Hz
        self.KINECT_FS = 30  # Hz

    def interpolate_missing_values(self, method="linear"):
        if method not in ['linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic']:
            raise ValueError(f"Invalid method. Supported methods are: "
                             "'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'")

        def interpolate(v, method="linear"):
            # Indices where X is not NaN
            not_nan_indices = np.arange(len(v))[~np.isnan(v)]
            # Values of X that are not NaN
            not_nan_values = v[~np.isnan(v)]
            # Interpolation function
            interp_func = interp1d(not_nan_indices, not_nan_values, kind=method, fill_value="extrapolate")
            # Interpolated vector
            interpolated = v.copy()
            interpolated[np.isnan(v)] = interp_func(np.arange(len(v))[np.isnan(v)])
            return interpolated

        # Interpolation
        self.area = interpolate(self.area, method=method)
        self.depth = interpolate(self.depth, method=method)
        self.pos[0, :] = interpolate(self.pos[0, :], method=method)
        self.pos[1, :] = interpolate(self.pos[1, :], method=method)
        self.pos[2, :] = interpolate(self.pos[2, :], method=method)

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

        if unit == "cm/sec":
            velocity = velocity / 10
        
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
        # since 2024/07/09, dataset doesn't contain green levels anymore
        # Using shan's csv file means loosing the TTL information
        try:
            cd.TTL = self.TTL[idx]
            cd.green_levels = self.green_levels[idx]
        except:
            pass

        cd.contact_flag = self.contact_flag[idx]
        cd.area = self.area[idx]
        cd.depth = self.depth[idx]
        try:
            cd.pos = self.pos[:, idx]
        except:
            pass
        try:
            cd.vel = self.vel[:, idx]
        except:
            pass

        return cd

    def set_data_idx(self, idx):
        self.time = self.time[idx]

        try:
            self.TTL = self.TTL[idx]  
            self.green_levels = self.green_levels[idx]
        except:
            pass

        self.contact_flag = self.contact_flag[idx]
        self.area = self.area[idx]
        self.depth = self.depth[idx]
        try:
            self.pos = self.pos[:, idx]
        except:
            pass
        try:
            self.vel = self.vel[:, idx]
        except:
            pass

    def append(self, contact_bis):
        self.time = np.concatenate((self.time, contact_bis.time))

        self.TTL = np.concatenate((self.TTL, contact_bis.TTL))
        try:
            self.green_levels = np.concatenate((self.green_levels, contact_bis.green_levels))
        except:
            pass

        self.contact_flag = np.concatenate((self.contact_flag, contact_bis.contact_flag))
        self.area = np.concatenate((self.area, contact_bis.area))
        self.depth = np.concatenate((self.depth, contact_bis.depth))
        try:
            self.pos = np.concatenate((self.pos, contact_bis.pos), axis=1)
        except:
            pass
        try:
            self.vel = np.concatenate((self.vel, contact_bis.vel), axis=1)
        except:
            pass

        self._TTL: list[float] = []
        self._green_levels: list[float] = []

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        self._time = value
        self.nsample = len(self._time)
        self.data_Fs = 1 / np.nanmean(np.diff(self._time))  # Hz

    @property
    def TTL(self):
        return self._TTL

    @TTL.setter
    def TTL(self, value):
        self._TTL = value

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
        # is there is nan, use a github provided PCA to ignore nan

        if np.all(np.isnan(self.pos)):
            self.pos_1D = np.nan(len(self.pos[0]))
            return

        pos3D = self.pos.transpose()

        if np.isnan(pos3D).any():
            nsamples, _ = np.shape(pos3D)
            # detrend the axis
            M = np.nanmean(pos3D, axis=0)
            pos3D_centered = pos3D - np.matlib.repmat(M, nsamples, 1)
            # interpolate the nan values
            pos3D = interpolate_nan_values(pos3D_centered)

        # if there is some value
        if np.any(pos3D != 0):
            # get the first PCA
            pca = PCA(n_components=1)
            self.pos_1D = np.squeeze(pca.fit_transform(pos3D))
        else:
            # fill up the vector to avoid any warnings from PCA.fit_transform
            self.pos_1D = np.zeros(len(self.pos[0]))


    @property
    def pos_1D(self):
        return self._pos_1D

    @pos_1D.setter
    def pos_1D(self, value):
        self._pos_1D = value
