import numpy as np
import pandas as pd
from scipy.signal import correlate, detrend
from scipy.interpolate import interp2d
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings

from .semicontrolled_data_cleaning import smooth_scd_signal, normalize_signal, gaussian_envelope
from ..materials.neuraldata import NeuralData  # noqa: E402
from ..materials.semicontrolled_data import SemiControlledData  # noqa: E402
from ..plot.semicontrolled_data_visualizer import SemiControlledDataVisualizer  # noqa: E402
from ..misc.waitforbuttonpress_popup import WaitForButtonPressPopup  # noqa: E402


class SemiControlledCorrectLag:
    def __init__(self):
        self.scd = None

    def trials_align_contact_and_neural(self, data_trials_in, show=False):

        min_delay = -500  # in number of sample (approx 500 ms at 1 kHz)
        max_delay = 500
        best_lags = [[None for j in range(min_delay, max_delay+1)] for i in range(len(data_trials_in))]

        scd_lag = SemiControlledCorrectLag()
        for idx, scd in enumerate(data_trials_in):
            scd_lag.load(scd)
            best_lags[idx][:], _ = scd_lag.correct_alignment(min_delay, max_delay, show=show)
        best_lag = np.median(best_lags)

        #if show:
        if True:
            corr_avg = np.mean(best_lags, axis=0)
            # Set custom x-axis tick positions and labels
            x_labels = [str(int(num)) for num in np.linspace(min_delay, max_delay, 10)]
            x_labels_loc = np.linspace(0, len(best_lags[0]), len(x_labels))

            # Create subplots
            fig, axs = plt.subplots(2, 1, figsize=(8, 6))  # 2 rows, 1 column
            # Plot heatmap on the top subplot
            axs[0].imshow(oversample_matrix(best_lags, np.shape(best_lags,)[1]), cmap='hot', interpolation='nearest')
            axs[0].set_title('2D Array Heatmap')
            axs[0].set_xlabel('Columns')
            axs[0].set_ylabel('Rows')
            axs[0].set_xticks(x_labels_loc, x_labels)  # Set x ticks to match number of columns
            #fig.colorbar(axs[0].imshow(best_lags, cmap='hot', interpolation='nearest'), ax=axs[0])

            # Plot column averages on the bottom subplot
            axs[1].plot(corr_avg, marker='o', linestyle='-')
            axs[1].set_title('Average of Each Column')
            axs[1].set_xlabel('Columns')
            axs[1].set_ylabel('Average')
            axs[1].set_xticks(x_labels_loc, x_labels)  # Set x ticks to match number of columns

            plt.show()
            print(f"Best lag: {best_lag}")
            plt.close()
            #b = WaitForButtonPressPopup()

        show_individual = False
        if show_individual:
            s_origin = SemiControlledDataVisualizer()
            s_modified = SemiControlledDataVisualizer()

        data_trials_out = []
        for idx, scd in enumerate(data_trials_in):
            if show_individual:
                s_origin.update(scd)
            scd.neural.shift(best_lag)
            if show_individual:
                s_modified.update(scd)

            data_trials_out.append(scd)

        return data_trials_out

    def convert_position_to_velocity(self):
        pos = self.scd.contact.pos_1D
        time = self.scd.contact.time
        # Calculate velocity from position (first derivative)
        velocity = np.diff(pos) / np.diff(time)
        # add a zero for match the number of elements in position/neural signals
        velocity = np.append(velocity, 0)
        return velocity

    def convert_contact_flag(self):
        depth = self.scd.contact.depth
        contact_flag = (depth > 0).astype(int)
        return contact_flag

    def compute_cross_correlation(self, neural, contact):
        # Compute cross-correlation
        correlation = correlate(contact, neural, mode='full')
        lags = np.arange(-len(neural) + 1, len(neural))
        if len(correlation) != len(lags):
            warnings.warn("Correlation is not the same size as Lags")
        return correlation, lags

    def find_best_lag(self, correlation, lags, min_lag=float('-inf'), max_lag=float('inf')):
        valid_indices = np.where((lags >= min_lag) & (lags <= max_lag))[0]
        valid_correlation = correlation[valid_indices]
        valid_lags = lags[valid_indices]
        best_lag = valid_lags[np.argmax(valid_correlation)]
        return best_lag

    def adjust_signal(self, lag):
        # Adjust the neuronal signal according to the lag
        self.scd.neural.shift(lag)

    def load(self, scd: SemiControlledData):
        self.scd = scd

    def correct_alignment(self, min_lag, max_lag, show=False):
        # prepare neural signal
        neural_sig = self.scd.neural.iff
        neural_sig = smooth_scd_signal(neural_sig, self.scd)
        neural_sig = gaussian_envelope(neural_sig)
        neural_sig = normalize_signal(neural_sig)

        # contact neural signal
        contact_sig = []
        match self.scd.stim.type:
            case 'stroke':
                contact_sig = self.convert_position_to_velocity()
                contact_sig = smooth_scd_signal(contact_sig, self.scd, method="adjust_with_speed")
                contact_sig = detrend(contact_sig)
                contact_sig = gaussian_envelope(contact_sig)
                contact_sig = abs(contact_sig)
                contact_sig = normalize_signal(contact_sig)

            case "tap":
                contact_sig = self.convert_contact_flag()
                contact_sig = smooth_scd_signal(contact_sig, self.scd)
                contact_sig = normalize_signal(contact_sig)

        correlation, lags = self.compute_cross_correlation(neural_sig, contact_sig)
        best_lag = self.find_best_lag(correlation, lags, min_lag, max_lag)

        # narrow down the results based on the parameters lag's min and max
        window = range(np.where(lags == min_lag)[0][0], np.where(lags == max_lag)[0][0])
        correlation_window = correlation[window]
        lags_window = lags[window]

        # normalise the correlation_window
        correlation_window = normalize_signal(correlation_window)

        # Display results
        # if self.scd.stim.type == 'stroke':
        if show and self.scd.stim.type == 'stroke':
            neural = NeuralData("", "")
            neural.iff = neural_sig
            neural.shift(best_lag)

            fig, axs = plt.subplots(3, 1)
            # Get the current figure manager
            mgr = plt.get_current_fig_manager()
            # Set the position of the window on the right side
            mgr.window.geometry("+{}+{}".format(1920, 0))  # Adjust the coordinates as needed
            fig.suptitle(self.scd.stim.print(), fontsize=14)

            axs[0].set_title("original")
            axs[0].plot(normalize_signal(contact_sig), label='contact_sig')
            axs[0].plot(normalize_signal(neural_sig), label='neural_sig')
            axs[0].legend()

            axs[1].set_title("shifted")
            axs[1].plot(normalize_signal(contact_sig), label='contact_sig')
            axs[1].plot(normalize_signal(neural.iff), label='neural_sig')
            axs[1].legend()

            axs[2].set_title("correlation")
            axs[2].plot(lags_window, correlation_window, label='correlation')
            axs[2].legend()

            plt.show()

            print(f"Best lag: {best_lag}")
            plt.close()
            #b = WaitForButtonPressPopup()

        return correlation_window, best_lag


def oversample_matrix(matrix, target_size):
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)

    # Get the dimensions of the original matrix
    n, m = matrix.shape

    # Create the original grid
    x = np.linspace(0, 1, m)
    y = np.linspace(0, 1, n)
    original_grid_x, original_grid_y = np.meshgrid(x, y)

    # Create the interpolation function
    interp_func = interp2d(x, y, matrix, kind='linear')

    # Create the new grid for the target size (MxM)
    new_x = np.linspace(0, 1, target_size)
    new_y = np.linspace(0, 1, target_size)
    new_grid_x, new_grid_y = np.meshgrid(new_x, new_y)

    # Interpolate the values on the new grid
    new_matrix = interp_func(new_x, new_y)

    return new_matrix