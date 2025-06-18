import copy
from itertools import groupby
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from typing import Optional
from functools import singledispatchmethod, wraps
import warnings

from .semicontrolled_data_correct_lag import SemiControlledCorrectLag  # noqa: E402
from .semicontrolled_data_cleaning import SemiControlledCleaner, smooth_scd_signal, normalize_signal  # noqa: E402
from ..materials.semicontrolled_data import SemiControlledData  # noqa: E402
from ..plot.semicontrolled_data_visualizer import SemiControlledDataVisualizer  # noqa: E402
from ..misc.waitforbuttonpress_popup import WaitForButtonPressPopup
from ..misc.time_cost_function import time_it


class SemiControlledDataSplitter:

    def __init__(self, viz: Optional[SemiControlledDataVisualizer] = None,
                 show=False, show_single_touches=False, manual_check=False,
                 save_visualiser=False, save_visualiser_fname=None):
        self.viz = viz
        self.show = show
        self.show_single_touches = show_single_touches
        self.manual_check = manual_check
        self.save_visualiser = save_visualiser
        self.save_visualiser_fname = save_visualiser_fname

    # add the kinect data into the dataframe
    def merge_data(self, df_list, data_led_list, verbose=False):
        df_out = []

        # if not a list of dataframe, transform the variable into a list
        if not isinstance(df_list, list):
            df_list = [df_list]
            data_led_list = [data_led_list]

        # https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas
        with pd.option_context('mode.chained_assignment', None):
            for idx, df in enumerate(df_list):
                dataframe_time = df["t"].values
                if verbose:
                    t = f"SemiControlledDataSplitter> CSVFILE: Number of sample of the current block = {len(dataframe_time)}"
                    print(t)

                # resample the kinect data obj to the dataframe
                kinect_led_curr = data_led_list[idx]
                kinect_led_curr.resample(dataframe_time, show=False)

                # add the kinect data into the dataframe
                df['led_on'] = kinect_led_curr.led_on
                df['green_levels'] = kinect_led_curr.green_levels

                # populate the output dataframe
                df_out.append(df)

        return df_out

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

    def get_single_touches(self, scd_trial, endpoints_list):
        scd_list_out = []
        # second split the trial per stimulus/repeat/period/nb. time the POI is stimulated
        for (start, end) in endpoints_list:
            scd = scd_trial.get_data_idx(range(start, end))
            scd_list_out.append(scd)
        return scd_list_out


    @singledispatchmethod
    def split_by_touch_event(self, scd_arg, **kwargs):
        """
        Base dispatcher for splitting.
        Raises an error for unsupported types.
        """
        raise TypeError(f"Type {type(scd_arg).__name__} is not supported")

    @split_by_touch_event.register(SemiControlledData)
    @time_it
    def _(self, scd: SemiControlledData, method: str = "method_3", correction: bool = True):
        """
        Processes a SINGLE SemiControlledData object.
        
        This method contains the core logic for splitting one touch event,
        which was previously inside the for-loop.
        """
        print(f"\nProcessing single object: {scd}")
        # Choose the method to split tap trials
        tap_method_map = {
            "method_1": "iff",
            "method_2": "depth",
            "method_3": "iff, depth"
        }
        tap_method = tap_method_map.get(method, "iff, depth")

        scd.contact.update_pos_1D()

        # Call the appropriate splitting function based on stimulus type
        if scd.stim.type == "stroke":
            scd_list, endpoints_list = self.get_single_strokes(scd, correction=correction)
        elif scd.stim.type == "tap":
            scd_list, endpoints_list = self.get_single_taps(scd, correction=correction, method=tap_method)
        else:
            scd_list, endpoints_list = [], []

        if self.save_visualiser and self.viz is not None:
            self.viz.save(self.save_visualiser_fname)
            if not self.show:
                del self.viz
        
        return scd_list, endpoints_list

    @split_by_touch_event.register(list)
    @time_it
    def _(self, scd_list: list, method: str = "method_3", correction: bool = True):
        """
        Processes a LIST of SemiControlledData objects.
        
        This method iterates through the list and calls the single-item
        version of this method for each object, aggregating the results.
        """
        print(f"Processing a list of {len(scd_list)} objects...")
        scd_list_out = []
        endpoints_list_out = []

        for scd in scd_list:
            # Dispatch to the single-item version of the method
            single_scd_list, single_endpoints_list = self.split_by_touch_event(
                scd, method=method, correction=correction
            )

            if single_scd_list and single_endpoints_list:
                scd_list_out.extend(single_scd_list)
                endpoints_list_out.extend(single_endpoints_list)

        return scd_list_out, endpoints_list_out


    def get_single_strokes(self, scd, correction=True):
        # to display the entire trial on the figure
        scd_untouched = copy.deepcopy(scd)

        # ignore data before the TTL is ON to avoid processing artifact
        start_trial_idx = np.argmax(scd.contact.TTL != 0)
        scd.set_data_idx(range(start_trial_idx, scd.md.nsample))

        # ignore data before the initial and after releasing contact (transitions non-contact / contact)
        depth_smooth = smooth_scd_signal(scd.contact.depth, scd, nframe=2, method="adjust_with_speed")
        depth_smooth = normalize_signal(depth_smooth, dtype=np.ndarray)
        contact_on_off = (depth_smooth > 0).astype(int)
        contact_transition = np.diff(contact_on_off)
        start_contact_idx = np.argmax(contact_transition == 1)
        # check for contact start:
        # if over .5 second, it is likely that the initial contact was already made before the start of trial
        reaction_time = scd.md.time[start_contact_idx] - scd.md.time[0]
        if reaction_time > .5:
            start_contact_idx = 0
        end_contact_idx = len(contact_transition) - np.argmax(contact_on_off[::-1] == 1)
        scd.set_data_idx(range(start_contact_idx, end_contact_idx))

        # smoothing of the 1D position signal
        pos_1D_smooth = smooth_scd_signal(scd.contact.pos_1D, scd, nframe=5, method="blind")
        pos_1D_smooth = normalize_signal(pos_1D_smooth, dtype=np.ndarray)

        # define the minimum distance between expected period/stimulation
        nb_period_expected = scd.stim.get_n_period_expected()
        # stroke: two stimulations are contained per period
        min_dist_peaks = .5 * scd.md.nsample/nb_period_expected

        # find peaks
        #   when motion ends at the participant's hand
        pos_peaks_a, _ = signal.find_peaks(pos_1D_smooth.transpose(), distance=min_dist_peaks)
        #   when motion ends at the participant's elbow
        pos_peaks_b, _ = signal.find_peaks(-1 * pos_1D_smooth.transpose(), distance=min_dist_peaks)
        # merge peaks
        pos_peaks = np.sort(np.concatenate((pos_peaks_a, pos_peaks_b)))

        # realign the index with original dataset
        pos_peaks = pos_peaks + start_trial_idx + start_contact_idx

        # extract the chunks representing single touches
        scd_period_list = []
        endpoints_list = []
        for i in range(len(pos_peaks) - 1):
            endpoints = (pos_peaks[i], pos_peaks[i+1] - 1)
            interval = np.arange(endpoints[0], endpoints[1], dtype=int)
            scd_interval = scd_untouched.get_data_idx(interval)
            # save current data for further processing and saving
            scd_period_list.append(scd_interval)
            endpoints_list.append(endpoints)

        # correct potential irregularities of the splitting method
        if correction:
            duration_expected_ms = scd_untouched.contact.data_Fs * scd_untouched.stim.get_single_contact_duration_expected()
            hp_duration_ms = .4 * duration_expected_ms
            scd_period_list, endpoints_list = self.correct(scd_period_list, endpoints_list, hp_duration_ms=hp_duration_ms)
            print(f"{len(scd_period_list)} single touches found.")

        if len(scd_period_list) == 0:
            warnings.warn("File <{}> couldn't be split.".format(scd.md.data_filename_short))
            warnings.warn("Number of single touch detected = 0.")
            return [], []

        if self.save_visualiser or self.show:
            if self.viz is None:
                self.viz = SemiControlledDataVisualizer()
            self.viz.update(scd_untouched)
            self.viz.add_vertical_lines(scd_untouched.md.time[pos_peaks])

        if self.show:
            if self.manual_check:
                scd_untouched.contact.update_pos_1D()
                pos_1D_smooth = smooth_scd_signal(scd_untouched.contact.pos_1D, scd, nframe=5, method="blind")
                pos_1D_smooth = normalize_signal(pos_1D_smooth, dtype=np.ndarray)
                fig, ax = plt.subplots(1, 1)
                plt.plot(pos_1D_smooth, label='Signal')
                plt.plot(pos_peaks, pos_1D_smooth[pos_peaks], 'r.', label='Peaks')
                ax.set_title("Found peaks on smoothed signal")
                WaitForButtonPressPopup()
                plt.close(fig)

        return scd_period_list, endpoints_list

    def get_single_taps(self, scd, correction=True, method="iff, depth"):
        '''chunk_period_tap:
            Dividing the trial into period of contact for
            tapping gestures is done by taking advantage
            of the contact.depth features.
        '''
        # to display the entire trial on the figure
        scd_untouched = copy.deepcopy(scd)

        # finding start of the trial via TTL
        start_trial_idx = np.argmax(scd.neural.TTL != 0)
        # ignore data before the TTL is ON to avoid processing artifact
        scd.set_data_idx(range(start_trial_idx, scd.md.nsample))

        sig = None
        if "depth" in method:
            if np.any(scd.contact.depth is not np.nan and 0 < scd.contact.depth):
                # smooth the depth signal
                depth_smooth = smooth_scd_signal(scd.contact.depth, scd, nframe=2, method="adjust_with_speed")
                depth_smooth = normalize_signal(depth_smooth, dtype=np.ndarray)
                # make sure that there is data
                if sig is None:
                    sig = depth_smooth
                else:
                    sig += depth_smooth

        if "iff" in method:
            if np.any(scd.neural.iff is not np.nan and 0 < scd.neural.iff):
                # smooth the spike signal
                iff_smooth = smooth_scd_signal(scd.neural.iff, scd, nframe=2, method="adjust_with_speed")
                iff_smooth = normalize_signal(iff_smooth, dtype=np.ndarray)
                if sig is None:
                    sig = iff_smooth
                else:
                    sig += iff_smooth
                    
        if sig is None:
            w = f"\nFile <{scd.md.data_filename_short}> couldn't be split.\nNo input data different from nan or zero was found."
            warnings.warn(w)
            return [], []

        # Assemble them for peaks detection
            sig = normalize_signal(sig, dtype=np.ndarray)
        sig = smooth_scd_signal(sig, scd, nframe=2, method="adjust_with_speed")
        sig = normalize_signal(sig, dtype=np.ndarray)

        # define the minimum distance between expected period/stimulation
        nb_period_expected = scd.stim.get_n_period_expected()
        min_dist_peaks = .5 * scd.md.nsample/nb_period_expected

        # find locations of the periods
        peaks, _ = signal.find_peaks(sig, distance=min_dist_peaks, prominence=(0.3, None))
        # create the interval extremities
        periods_idx = np.mean([peaks[1:], peaks[:-1]], axis=0)

        # realign the index with original dataset
        periods_idx = periods_idx + start_trial_idx

        if len(periods_idx) < 2:
            warnings.warn(f"number of single touch detected lower than one, while it should be at least 3.")
        else:
            # first single touch:
            # define the left end point based on the same distance from the center of the touch on the right side
            # or first sample
            start_idx = max([0, int(periods_idx[0]-(periods_idx[1] - periods_idx[0]))])
            # last single touch:
            # define the right end point based on the same distance from the center of the touch on the left side
            # or max sample
            end_idx = min([scd_untouched.md.nsample-1, int(periods_idx[-1] + abs(int((periods_idx[-2] - periods_idx[-1]))))])
            periods_idx = np.insert(periods_idx, 0, start_idx)
            periods_idx = np.append(periods_idx, end_idx)
            periods_idx = periods_idx.astype(int)

        # create the data list via defined intervals
        scd_period_list = []
        endpoints_list = []
        for i in range(len(periods_idx) - 1):
            endpoints = (periods_idx[i], periods_idx[i+1]-1)
            interval = np.arange(endpoints[0], endpoints[1], dtype=int)
            scd_interval = scd_untouched.get_data_idx(interval)
            # save current data for further processing and saving
            scd_period_list.append(scd_interval)
            endpoints_list.append(endpoints)

        # correct potential irregularities of the splitting method
        if correction:
            duration_expected_ms = scd_untouched.contact.data_Fs * scd_untouched.stim.get_single_contact_duration_expected()
            hp_duration_ms = .4 * duration_expected_ms
            scd_period_list, endpoints_list = self.correct(scd_period_list, endpoints_list, hp_duration_ms=hp_duration_ms)
            print(f"{len(scd_period_list)} single touches found.")

        if self.save_visualiser or self.show:
            if self.viz is None:
                self.viz = SemiControlledDataVisualizer()
            self.viz.update(scd_untouched)

        if len(scd_period_list) == 0:
            w = f"\nFile <{scd.md.data_filename_short}> couldn't be split.\nNumber of single touch detected = 0."
            warnings.warn(w)
            return [], []
        
        if self.show:
            self.viz.add_vertical_lines(scd_untouched.md.time[periods_idx])
            if self.manual_check:
                fig, ax = plt.subplots(1, 1)
                ax.plot(sig, label='Signal')
                ax.plot(peaks, sig[peaks], 'g.', label='Peaks')
                ax.plot(periods_idx, sig[periods_idx], 'r.', label='inter-Peaks')
                ax.set_title("Found peaks on smoothed signal")
                WaitForButtonPressPopup()
                plt.close(fig)
                

        return scd_period_list, endpoints_list

    def correct(self, scd_list: list[SemiControlledData], endpoints_list: list[tuple], hp_duration_ms=50, force=False):
        '''correct
            merge very small chunk that have been created erroneously
            highpass_duration_val = milliseconds
        '''
        scd_list_backup = copy.deepcopy(scd_list)

        # find peaks used in get_single_taps/strokes can give peaks too close to each other.
        # minimal duration for erroneous signal has been estimated to 50 samples, except is user decided otherwise (using force=True)
        highpass_duration_val_threshold = 50
        if not force:
            if hp_duration_ms < highpass_duration_val_threshold:
                hp_duration_ms = highpass_duration_val_threshold

        idx = 0
        while len(scd_list) > 1 and idx < len(scd_list):
            scd = scd_list[idx]

            if scd.md.nsample < 2:
                del scd_list[idx], endpoints_list[idx]
                continue

            curr_duration_ms = 1000 * (scd.md.time[-1] - scd.md.time[0])
            # if the current data duration isn't long enough, it needs to be merged with a neighbour
            if curr_duration_ms < hp_duration_ms:
                # assign with the right
                if idx == 0:
                    scd_list[idx].append(scd_list[idx + 1])
                    # extend the right end coordinate to the next data
                    endpoints_list[idx] = (endpoints_list[idx][0], endpoints_list[idx+1][1])
                    del scd_list[idx + 1], endpoints_list[idx + 1]
                # assign with the left
                elif idx == len(scd_list) - 1:
                    scd_list[idx - 1].append(scd_list[idx])
                    # extend the left end coordinate to the next data
                    endpoints_list[idx-1] = (endpoints_list[idx-1][0], endpoints_list[idx][1])
                    del scd_list[idx], endpoints_list[idx]
                # find the smallest neighbour
                else:
                    _prev = idx - 1
                    _next = idx + 1
                    if scd_list[_prev].md.nsample > scd_list[_next].md.nsample:
                        scd_list[_prev].append(scd_list[idx])
                        # extend the left end coordinate to the next data
                        endpoints_list[_prev] = (endpoints_list[_prev][0], endpoints_list[idx][1])
                        del scd_list[idx], endpoints_list[idx]
                    else:
                        scd_list[idx].append(scd_list[_next])
                        # extend the left end coordinate to the next data
                        endpoints_list[idx] = (endpoints_list[idx][0], endpoints_list[_next][1])
                        del scd_list[_next], endpoints_list[_next]
            else:
                idx += 1
        if self.show_single_touches:
            visualiser = SemiControlledDataVisualizer()
            for scd in scd_list:
                visualiser.update(scd)
                WaitForButtonPressPopup()
            del visualiser

        # "time" can have gaps of several milliseconds, that is why
        # between singular touch, the number of elements can vary
        # The following if is supposed to never happen as it has been
        # checked in the while loop before.
        recorded_durs = [(1000 * (scd.md.time[-1] - scd.md.time[0])) for scd in scd_list]
        if (np.array(recorded_durs) < hp_duration_ms).any():
            print("----------")
            print(hp_duration_ms)
            print(recorded_durs)
            print("++++++++++")

        return scd_list, endpoints_list










