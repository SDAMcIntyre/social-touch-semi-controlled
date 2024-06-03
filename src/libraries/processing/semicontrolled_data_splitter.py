from itertools import groupby
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA
import timeit
import warnings

from libraries.materials.semicontrolled_data import SemiControlledData  # noqa: E402
from libraries.misc.semicontrolled_data_visualizer import SemiControlledDataVisualizer  # noqa: E402
from libraries.misc.waitforbuttonpress_popup import WaitForButtonPressPopup
import libraries.misc.time_cost_function as time_cost


class SemiControlledDataSplitter:

    def __init__(self, filename):
        self.filename = filename

    def split_by_trials(self):
        data = []

        df = SemiControlledData(self.filename).load_dataframe()

        # Separate the rows of each block
        for _, block_group in groupby(enumerate(df.block_id), key=itemgetter(1)):
            indices_group = [index for index, _ in block_group]
            df_block = df.iloc[indices_group]
            # Separate the current block rows of each trial
            for _, group in groupby(enumerate(df_block.trial_id), key=itemgetter(1)):
                indices = [index for index, _ in group]
                scd = SemiControlledData(self.filename)
                scd.set_variables(df_block.iloc[indices])
                data.append(scd)
        return data

    @time_cost.time_it
    def split_by_single_touch_event(self, correction=True, show=False):
        """split_by_single
           split the current semicontrolled data into single touch event
           A period is determined differently based on the type (Tap or Stroke)
           """
        data = []
        # first separate the data by trial
        data_trials = self.split_by_trials()
        # second split the trial per stimulus/repeat/period/nb. time the POI is stimulated
        for scd in data_trials:
            match scd.stim.type:
                case "stroke":
                    # remove start and end data (motion/positioning artifact)
                    scd = self.remove_contact_artifacts(scd, method="simple")
                    # split
                    scd_list = self.get_single_strokes(scd, correction=correction, show=show)
                case "tap":
                    # remove start and end data (artifact)
                    scd = self.remove_contact_artifacts(scd, method="by-peak_soft")
                    # split
                    scd_list = self.get_single_taps(scd, correction=correction, show=show)
                case _:
                    scd_list = []
            try:
                data.extend(scd_list)
            except:
                # then it is not a list but just one SemiControlledData
                data = [data, scd_list]
        return data

    def get_single_strokes(self, scd, correction=True, show=False):
        # compress the signal into 1D (as the expected motion is supposed to be 1D anyway)
        pca = PCA(n_components=1)
        pos_1D = np.squeeze(pca.fit_transform(scd.contact.pos.transpose()))
        # smoothing
        pos_1D_smooth = self.get_smooth_signal(pos_1D, scd, nframe=5, method="blind")

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
        depth_smooth = self.get_smooth_signal(scd.contact.depth, scd, nframe=2, method="adjust_with_speed")

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
                    try:
                        scd_single[idx].append(scd_single[idx + 1])
                    except:
                        print("a")
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

    def remove_contact_artifacts(self, scd, method="by-peak"):
        '''narrow_data_to_essential
            remove the data based on the contact artifact that occurs
            at the beginning and at the end of the trial, where the
            experimenter move its hand toward and away from the
            stimulation area.

            parameters:
            scd (SemiControlledData): dataset of a trial
            method: "soft" will leave some space before and after the
                        first and last contact respectively (good for Tap)
                    "hard" will remove any space before and after the
                        first and last contact respectively (good for Stroke)
        '''
        # smooth the depth signal
        depth_smooth = self.get_smooth_signal(scd.contact.depth, scd, nframe=5)

        # find locations of the periods
        nb_period_expected = scd.stim.get_n_period_expected()
        n_sample_per_period = scd.md.nsample/nb_period_expected
        peaks, _ = signal.find_peaks(depth_smooth, distance=n_sample_per_period*0.8)

        match method:
            case "simple":
                # extract the first location of the initial contact
                starting_idx = next((idx for idx, v in enumerate(depth_smooth) if v), None)
                # extract the last location of the last contact
                ending_idx = len(depth_smooth) - next((idx for idx, v in enumerate(np.flip(depth_smooth)) if v), None)
            case "by-peak_hard":
                # extract the first location of the initial contact
                start_depth_values = depth_smooth[:peaks[0]]
                start_depth_values_r = np.flip(start_depth_values)
                non_zero_indices = np.nonzero(start_depth_values_r == 0)[0]
                starting_idx = non_zero_indices[0]
                # extract the last location of the last contact
                end_depth_values = depth_smooth[peaks[-1]:]
                non_zero_indices = np.nonzero(end_depth_values == 0)[0]
                ending_idx = non_zero_indices[0] if non_zero_indices.size > 0 else scd.md.nsample
            case "by-peak_soft":
                periods_idx = np.mean([peaks[1:], peaks[:-1]], axis=0)
                starting_idx = max(0,  peaks[0] - (periods_idx[0] - peaks[0]))
                ending_idx = min(scd.md.nsample,  peaks[-1] + (peaks[-1]-periods_idx[-1]))
            case _:
                warnings.warn("Warning: remove_contact_artifacts>method couldn't be found")
                starting_idx = 0
                ending_idx = scd.md.nsample
        interval = np.arange(starting_idx, ending_idx, dtype=int)
        scd.set_data_idx(interval)

        return scd

    def get_smooth_signal(self, sig, scd, nframe=5, method="blind"):
        '''get_smooth_signal
            smooth the signal based on the expected speed of the trial:
             - more smoothing for lower speed
             - return an array of identical size
        '''
        oversampling_to_trueFs = (scd.contact.data_Fs / scd.contact.KINECT_FS)
        window_size = nframe * oversampling_to_trueFs

        match method:
            case "adjust_with_speed":
                # between 0 and 1 of the max speed
                speed_multiplier = 1/scd.stim.curr_max_vel_ratio()
                # adjust the window based on the ratio
                window_size = int(window_size * speed_multiplier)
            case "blind", _:
                pass

        weights = np.repeat(1.0, window_size) / window_size
        sig_smooth = np.convolve(sig, weights, 'same')

        return sig_smooth










