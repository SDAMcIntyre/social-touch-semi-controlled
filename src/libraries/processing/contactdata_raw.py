import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture


class ContactDataRaw:
    def __init__(self):
        self._area: list[float] = []
        self._depth: list[float] = []
        self._vel: list[float] = []
        self._pos: list[float] = []

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



