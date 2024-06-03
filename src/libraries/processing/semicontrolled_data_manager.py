import matplotlib.pyplot as plt
import numpy as np

from libraries.misc.semicontrolled_data_visualizer import SemiControlledDataVisualizer  # noqa: E402
from libraries.materials.semicontrolled_data import SemiControlledData  # noqa: E402
from .semicontrolled_data_splitter import SemiControlledDataSplitter  # noqa: E402


class SemiControlledDataManager:

    def __init__(self):
        self.data: list[SemiControlledData] = []
        self.data_type: str = "all_together"

    def load_dataframe(self, filename):
        self.data = SemiControlledData(filename, "automatic_load")

    def load_by_trial(self, filename):
        scd_splitter = SemiControlledDataSplitter(filename)
        self.data = scd_splitter.split_by_trials()
        self.data_type = "trials"
        return self.data

    def load_by_single_touch_event(self, filenames, correction=True, show=False):
        self.data = []
        self.data_type = "single_touch_event"

        if isinstance(filenames, list):
            for filename in filenames:
                print("Processing file:", filename)
                scd_splitter = SemiControlledDataSplitter(filename)
                self.data.extend(scd_splitter.split_by_single_touch_event(correction=correction, show=show))
        else:
            print("Processing file:", filenames)
            scd_splitter = SemiControlledDataSplitter(filenames)
            self.data.extend(scd_splitter.split_by_single_touch_event(correction=correction, show=show))

        return self.data

    def get_ratio_durations(self):
        ratio = np.array([0] * len(self.data), dtype=float)
        for idx, scd in enumerate(self.data):
            ratio[idx] = scd.get_duration_ratio()
        return ratio

    def get_contact_expected(self):
        types = [f"" for i in range(0, len(self.data))]
        velocities = np.array([0] * len(self.data), dtype=float)
        forces = [f"" for i in range(0, len(self.data))]
        sizes = [f"" for i in range(0, len(self.data))]
        for idx, scd in enumerate(self.data):
            t, v, f, s = scd.get_stimulusInfoContact()
            types[idx] = t
            velocities[idx] = v
            forces[idx] = f
            sizes[idx] = s
        types = np.array(types)
        forces = np.array(forces)
        sizes = np.array(sizes)

        return types, velocities, forces, sizes

    def estimate_contact_averaging(self, verbose=False):
        velocity_cm = np.array([0] * len(self.data), dtype=float)
        depth_cm = np.array([0] * len(self.data), dtype=float)
        area_cm = np.array([0] * len(self.data), dtype=float)

        for idx, scd in enumerate(self.data):
            if verbose:
                print("------------")
                print(idx)
                print(scd.stim.type)
                print(scd.md.trial_id)

            v, d, a = scd.estimate_contact_averaging()
            velocity_cm[idx] = v
            depth_cm[idx] = d
            area_cm[idx] = a

        return velocity_cm, depth_cm, area_cm

    def get_instantaneous_velocities(self):
        vel = np.array(list([0]) * len(self.data), dtype=float)
        for idx, scd in enumerate(self.data):
            vel[idx] = scd.get_instantaneous_velocity()
        return vel

    def define_trust_scores(self):
        score = np.array([0] * len(self.data), dtype=float)
        for idx, scd in enumerate(self.data):
            score[idx] = scd.define_trust_score()
        return score







