import numpy as np


class StimulusInfo:
    def __init__(self):
        self._type: str = ""
        self._vel: float = 0
        self._size: str = ""
        self._force: str = ""

        self.MAX_VEL = 24  # cm/sec
        self.MINIMUM_DURATION = 6  # second
        self.MOTIONPATH_LENGTH = 3  # cm
        self.PERIOD_MOTIONPATH_LENGTH = 2 * self.MOTIONPATH_LENGTH  # cm
        self.MIN_PERIOD = 0  # depends on the type of contact (see type.setter)

    def get_data(self, attr=""):
        data = []
        match attr:
            case "type":
                data = self._type
            case "vel":
                data = self._vel
            case "size":
                data = self._size
            case "force":
                data = self._force
        return data

    def get_data(self):
        stim = StimulusInfo()
        stim.type = self._type
        stim.vel = self._vel
        stim.size = self._size
        stim.force = self._force
        return stim

    def is_similar(self, stim_bis):
        if self.type != stim_bis.type:
            return False
        if self.vel != stim_bis.vel:
            return False
        if self.size != stim_bis.size:
            return False
        if self.force != stim_bis.force:
            return False
        return True

    def curr_max_vel_ratio(self):
        return self.vel/self.MAX_VEL

    def get_n_period_expected(self):
        nsec_per_period = self.PERIOD_MOTIONPATH_LENGTH / self.vel  # sec

        # if the minimal number of period requires more time than the minimal duration,
        # return the minimum number of period
        # else return the number of period within minimal duration
        if (nsec_per_period * self.MIN_PERIOD) > self.MINIMUM_DURATION:
            return self.MIN_PERIOD
        else:
            return self.MINIMUM_DURATION / nsec_per_period

    def get_single_contact_duration_expected(self):
        duration_sec = self.PERIOD_MOTIONPATH_LENGTH / self.vel  # sec
        # the definition of singular contact differs from Tap and Stroke:
        # For Stroke, there are two singular contacts during one period
        if self._type[0] == "stroke":
            duration_sec /= 2
        return duration_sec

    def print(self):
        s = "type: {}, vel: {}, size: {}, force: {}".format(self._type, self._vel, self._size, self._force)
        return s

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value
        match self.type:
            case "tap":
                self.MIN_PERIOD = 4
            case "stroke":
                self.MIN_PERIOD = 2

    @property
    def vel(self):
        return self._vel

    @vel.setter
    def vel(self, value):
        self._vel = value

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value

    @property
    def force(self):
        return self._force

    @force.setter
    def force(self, value):
        self._force = value
