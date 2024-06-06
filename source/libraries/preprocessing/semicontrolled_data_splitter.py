from itertools import groupby
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import warnings

from .semicontrolled_data_correct_lag import SemiControlledCorrectLag  # noqa: E402
from .semicontrolled_data_cleaning import SemiControlledCleaner, smooth_scd_signal  # noqa: E402
from ..materials.semicontrolled_data import SemiControlledData  # noqa: E402
from ..plot.semicontrolled_data_visualizer import SemiControlledDataVisualizer  # noqa: E402
from ..misc.waitforbuttonpress_popup import WaitForButtonPressPopup
from ..misc.time_cost_function import time_it


class SemiControlledDataSplitter:

    def __init__(self):
        self.data_filename = ""
        self.unit_name2type_filename = ""

    def split_by_column_label(self, df_list, label=""):
        df_out = []

        # if not a list of dataframe, transform the variable into a list
        if not isinstance(df_list, list):
            df_list = [df_list]

        for df in df_list:
            # Separate the rows of each block
            for _, col_group in groupby(enumerate(df[label]), key=itemgetter(1)):
                indices_group = [index for index, _ in col_group]
                df_out.append(df.iloc[indices_group])
        return df_out

    @time_it
    def split_by_touch_event(self, scd_list, correction=True, show=False):
        """split_by_single
           split the current semicontrolled data into single touch event
           A period is determined differently based on the type (Tap or Stroke)
           """
        scd_list_out = []

        # second split the trial per stimulus/repeat/period/nb. time the POI is stimulated
        for scd in scd_list:
            match scd.stim.type:
                case "stroke":
                    scd_list = self.get_single_strokes(scd, correction=correction, show=show)
                case "tap":
                    scd_list = self.get_single_taps(scd, correction=correction, show=show)
                case _:
                    scd_list = []
            try:
                scd_list_out.extend(scd_list)
            except:
                # then it is not a list but just one SemiControlledData
                scd_list_out = [scd_list_out, scd_list]
        return scd_list_out

    def get_single_strokes(self, scd, correction=True, show=False):
        # smoothing of the 1D position signal
        pos_1D_smooth = smooth_scd_signal(scd.contact.pos_1D, scd, nframe=5, method="blind")

        # define the minimum distance between expected period/stimulation
        nb_period_expected = scd.stim.get_n_period_expected()
        # stroke: two stimulations are contained per period
        min_dist_peaks = .5 * scd.md.nsample/nb_period_expected

        # find peaks
        # when motion ends at the participant's hand
        pos_peaks_a, _ = signal.find_peaks(pos_1D_smooth.transpose(), distance=min_dist_peaks)
        # when motion ends at the participant's elbow
        pos_peaks_b, _ = signal.find_peaks(-1 * pos_1D_smooth.transpose(), distance=min_dist_peaks)
        # merge peaks
        pos_peaks = np.sort(np.concatenate((pos_peaks_a, pos_peaks_b)))

        if show:
            fig, ax = plt.subplots(1, 1)
            plt.plot(pos_1D_smooth, label='Signal')
            plt.plot(pos_peaks, pos_1D_smooth[pos_peaks], 'r.', label='Peaks')
            ax.set_title("Found peaks on smoothed signal")
            visualiser_all = SemiControlledDataVisualizer(scd)
            visualiser_indiv = SemiControlledDataVisualizer()

        scd_period_list = []
        for i in range(len(pos_peaks) - 1):
            interval = np.arange(1 + pos_peaks[i], pos_peaks[i + 1], dtype=int)
            scd_interval = scd.get_data_idx(interval)
            scd_period_list.append(scd_interval)
            if show:
                visualiser_indiv.update(scd_interval)
                WaitForButtonPressPopup()

        # correct potential irregularities of the splitting method
        if correction:
            duration_expected_ms = 1000 * scd.stim.get_single_contact_duration_expected()
            hp_duration_ms = .4 * duration_expected_ms
            scd_period_list = self.correct(scd_period_list, hp_duration_val=hp_duration_ms, show=show)

        return scd_period_list

    def get_single_taps(self, scd, correction=True, show=False):
        '''chunk_period_tap:
            Dividing the trial into period of contact for
            tapping gestures is done by taking advantage
            of the contact.depth features.
        '''
        # smooth the depth signal
        depth_smooth = self.smooth_scd_signal(scd.contact.depth, scd, nframe=2, method="adjust_with_speed")

        # define the minimum distance between expected period/stimulation
        nb_period_expected = scd.stim.get_n_period_expected()
        min_dist_peaks = .5 * scd.md.nsample/nb_period_expected

        # find locations of the periods
        peaks, _ = signal.find_peaks(depth_smooth, distance=min_dist_peaks)
        # create the interval extremities
        periods_idx = np.mean([peaks[1:], peaks[:-1]], axis=0)
        periods_idx = np.insert(periods_idx, 0, 0)
        periods_idx = np.append(periods_idx, scd.md.nsample-1)
        periods_idx = periods_idx.astype(int)

        if show:
            fig, ax = plt.subplots(1, 1)
            plt.plot(depth_smooth, label='Signal')
            plt.plot(periods_idx, depth_smooth[periods_idx], 'r.', label='Peaks')
            ax.set_title("Found peaks on smoothed signal")
            visualiser_all = SemiControlledDataVisualizer(scd)
            visualiser_indiv = SemiControlledDataVisualizer()

        # create the data list via defined intervals
        scd_period_list = []
        for i in range(len(periods_idx) - 1):
            interval = np.arange(1+periods_idx[i], periods_idx[i+1], dtype=int)
            scd_interval = scd.get_data_idx(interval)
            scd_period_list.append(scd_interval)
            if show:
                visualiser_indiv.update(scd_interval)
                WaitForButtonPressPopup()

        # correct potential irregularities of the splitting method
        if correction:
            duration_expected_ms = 1000 * scd.stim.get_single_contact_duration_expected()
            hp_duration_ms = .4 * duration_expected_ms
            scd_period_list = self.correct(scd_period_list, hp_duration_val=hp_duration_ms, show=show)

        return scd_period_list

    def correct(self, scd_single: list[SemiControlledData], hp_duration_val=50, force=False, show=False):
        '''correct
            merge very small chunk that have been created erroneously
            highpass_duration_val = milliseconds
        '''
        # minimal duration for erroneous signal is 50, except is user decided otherwise (using force=True)
        highpass_duration_val_threshold = 50
        if not force:
            if hp_duration_val < highpass_duration_val_threshold:
                hp_duration_val = highpass_duration_val_threshold

        idx = 0
        while len(scd_single) > 1 and idx < len(scd_single):
            scd = scd_single[idx]

            if scd.md.nsample < 2:
                del scd_single[idx]
                continue

            duration_recorded = 1000 * (scd.md.time[-1] - scd.md.time[0])
            if duration_recorded < hp_duration_val:
                if idx == 0:
                    scd_single[idx].append(scd_single[idx + 1])
                    del scd_single[idx + 1]
                elif idx == len(scd_single) - 1:
                    scd_single[idx - 1].append(scd_single[idx])
                    del scd_single[idx]
                # find the smallest neighbour
                else:
                    _prev = idx - 1
                    _next = idx + 1
                    if scd_single[_prev].md.nsample > scd_single[_next].md.nsample:
                        scd_single[_prev].append(scd_single[idx])
                        del scd_single[idx]
                    else:
                        scd_single[idx].append(scd_single[_next])
                        del scd_single[_next]
            else:
                idx += 1
        if show:
            visualiser = SemiControlledDataVisualizer()
            for scd in scd_single:
                visualiser.update(scd)
                WaitForButtonPressPopup()
            del visualiser

        # "time" can have gaps of several milliseconds, that is why
        # between singular touch, the number of elements can vary
        # The following if is supposed to never happen as it has been
        # checked in the while loop before.
        recorded_durs = [(1000 * (scd.md.time[-1] - scd.md.time[0])) for scd in scd_single]
        if any(recorded_durs < hp_duration_val):
            print("----------")
            print(hp_duration_val)
            print(recorded_durs)
            print("++++++++++")

        return scd_single










