from collections import defaultdict
import numpy as np
from typing import Dict, Tuple

from ..materials.semicontrolled_data import SemiControlledData  # noqa: E402
from ..materials.kinect_led import SemiControlledDataLED  # noqa: E402
from .semicontrolled_data_cleaning import SemiControlledCleaner  # noqa: E402
from .semicontrolled_data_correct_lag import SemiControlledCorrectLag  # noqa: E402
from .semicontrolled_data_splitter import SemiControlledDataSplitter  # noqa: E402


class SemiControlledDataManager:

    def __init__(self):
        self.data: list[SemiControlledData] = []
        self.data_type: str = "all_together"

    # set new data values (useful if data has been filtered i.e. with sort_per_unit_type)
    def set_data(self, data_list: list[SemiControlledData]):
        self.data = data_list

    def load_dataframe(self, filename):
        self.data = SemiControlledData(filename, "automatic_load")

    def load_by_trial(self, filename):
        scd_splitter = SemiControlledDataSplitter(filename)
        self.data = scd_splitter.split_by_trials()
        self.data_type = "trials"
        return self.data

    def preprocess_data_files(self, data_filenames, unit_name2type_filename, correction=True, show=False):
        self.data = []
        self.data_type = "single_touch_event"

        # if one file name is sent, transform the variable into a list
        if not isinstance(data_filenames, list):
            data_filenames = [data_filenames]

        for data_filename in data_filenames:
            print("Processing file:", data_filename)
            scd_list = self.preprocess_data_file(data_filename, unit_name2type_filename, correction=correction, show=show)
            self.data.extend(scd_list)

        return self.data

    def preprocess_data_file(self, data_filename, unit_name2type_filename, led_files_info_list, correction=True, show=False):
        scd = SemiControlledData(data_filename, unit_name2type_filename)
        led = SemiControlledDataLED()
        splitter = SemiControlledDataSplitter()
        cleaner = SemiControlledCleaner()
        correctlag = SemiControlledCorrectLag()

        # 1. load the csv file as a dataframe
        df = scd.load_dataframe(dropna=False)
        # 2. split full dataframe by blocks
        df_list = splitter.split_by_column_label(df, label="block_id")

        # 3. extract green LED
        data_led_list = led.load_class_list_from_infos(led_files_info_list)
        # 4. incorporate green LED into the dataframe + compensate for refresh rate difference
        df_list = splitter.merge_data(df_list, data_led_list)

        # 5. split blocks by trials
        df_list = splitter.split_by_column_label(df_list, label="trial_id")
        # 6. transform dataframe list into SemiControlledData list
        scd_list = scd.create_list_from_df(df_list)

        # 7. correct for contact data issues
        scd_list = cleaner.trials_narrow_time_series_to_essential(scd_list)
        # 8. correct for neural data latency (electrode vs end organ location)
        scd_list = cleaner.correct_electrode_location_latency(scd_list)
        # 10. correct for misalignment
        scd_list = correctlag.trials_align_contact_and_neural(scd_list)

        # 11. split by touch event
        scd_list = splitter.split_by_touch_event(scd_list, correction=correction, show=show)

        return scd_list

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

    def sort_per_unit_type(self, verbose=False) -> Tuple[Dict[str, list[SemiControlledData]], list[str]]:
        sorted_dict = defaultdict(list)

        for scd in self.data:
            key = scd.neural.unit_type
            if key is not None:
                sorted_dict[key].append(scd)

        sorted_dict = dict(sorted_dict)  # Convert defaultdict to regular dict
        keys = list(sorted_dict.keys())

        return sorted_dict, keys
