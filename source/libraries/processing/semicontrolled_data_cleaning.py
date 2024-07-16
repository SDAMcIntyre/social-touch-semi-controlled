import numpy as np
from scipy.signal import find_peaks, gaussian
import warnings

from ..materials.semicontrolled_data import SemiControlledData  # noqa: E402
from ..materials.neuraldata import NeuralData  # noqa: E402
from ..plot.semicontrolled_data_visualizer import SemiControlledDataVisualizer  # noqa: E402


class SemiControlledCleaner:
    def __init__(self):
        pass

    def trials_narrow_time_series_to_essential(self, scd_list):
        # remove start and end data (motion/positioning artifact)
        if not isinstance(scd_list, list):
            scd_list = [scd_list]
        scd_list_out = []
        for scd in scd_list:
            match scd.stim.type:
                case "stroke":
                    scd = self.remove_contact_artifacts(scd, method="simple")
                case "tap":
                    scd = self.remove_contact_artifacts(scd, method="by-peak_soft")
            scd_list_out.append(scd)
        return scd_list_out

    def correct_electrode_location_latency(self, scd_list):
        if not isinstance(scd_list, list):
            scd_list = [scd_list]
        for scd in scd_list:
            scd.neural.correct_conduction_velocity()
        return scd_list

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
        depth_smooth = smooth_scd_signal(scd.contact.depth, scd, nframe=5)

        # find locations of the periods
        nb_period_expected = scd.stim.get_n_period_expected()
        n_sample_per_period = scd.md.nsample/nb_period_expected
        peaks, _ = find_peaks(depth_smooth, distance=n_sample_per_period*0.8)

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


def smooth_scd_signal(sig, scd, nframe=5, method="blind", normalise=True):
    '''get_smooth_signal
        smooth the signal based on the expected speed of the trial:
         - more smoothing for lower speed
         - return an array of identical size
    '''
    oversampling_to_trueFs = (scd.contact.data_Fs / scd.contact.KINECT_FS)
    window_size = int(nframe * oversampling_to_trueFs)

    match method:
        case "adjust_with_speed":
            # between 0 and 1 of the max speed
            speed_multiplier = 1/scd.stim.curr_max_vel_ratio()
            # adjust the window based on the ratio
            window_size = int(window_size * speed_multiplier)
        case "blind", _:
            pass

    if window_size > len(sig):
        warnings.warn("window_size is longer than sig, shorten it to the signal's length...")
        window_size = len(sig)

    # Create a function to handle NaN values
    def nan_convolve(signal, weights):
        half_window = len(weights) // 2
        sig_padded = np.pad(signal, (half_window, half_window), mode='constant', constant_values=np.nan)
        smoothed = np.full_like(signal, np.nan)
        for i in range(len(signal)):
            window = sig_padded[i:i + len(weights)]
            smoothed[i] = np.nanmean(window * weights)
        return smoothed

    try:
        weights = np.repeat(1.0, window_size) / window_size
    except:
        pass
    if np.isnan(sig).any():
        sig_smooth = nan_convolve(sig, weights)
    else:
        sig_smooth = np.convolve(sig, weights, 'same')

    if len(sig_smooth) != len(sig):
        warnings.warn("sig_smooth is not the size of the original signal")

    # ensure the output is an array
    sig_smooth = np.array(sig_smooth)

    return sig_smooth


def normalize_signal(signal, dtype=list):
    min_val = np.nanmin(signal)
    max_val = np.nanmax(signal)
    normalized_signal = [(x - min_val) / (max_val - min_val) for x in signal]
    
    if dtype == np.ndarray:
        normalized_signal = np.array(normalized_signal)

    return normalized_signal


def gaussian_envelope(input_vector, envelope_width=0):
    """Modulate an input vector with a bell envelope"""
    length = len(input_vector)
    if not envelope_width:
        envelope_width = length/4
    envelope = gaussian(length, std=envelope_width)
    return input_vector * envelope