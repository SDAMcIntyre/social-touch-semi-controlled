import warnings
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA  # from pyppca import ppca  # pip install git+https://github.com/el-hult/pyppca
import sys
import os

# homemade libraries
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from .metadata import Metadata  # noqa: E402
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

        self.pca_model = None           # Store the fitted PCA model if needed later
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
        dt = np.diff(self._time)
        if ~np.all(np.isnan(dt)):
            self.data_Fs = 1 / np.nanmean(dt)  # Hz
        else:
            print("contactdata::data_FS COULD NOT BE ASSESSED, attribute time contains only NaN values.")

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

    def update_pos_1D(self, automatic_range=True, pca_range=None):
        """
        Compresses the 3D position data (self.pos) into 1D using PCA.

        Handles NaN values by centering, interpolating, and then applying PCA.
        Allows specifying a sub-range of samples to fit the PCA model,
        which is then applied to the entire dataset.

        Args:
            pca_range (tuple, optional): A tuple (start_index, end_index)
                specifying the sample range (exclusive of end_index) to use
                for fitting the PCA model. If None, the entire dataset is used
                for fitting. Defaults to None.
        """
        if np.all(np.isnan(self.pos)):
            # Ensure output length matches number of samples
            self.pos_1D = np.full(self.pos.shape[1], np.nan)
            self.pca_model = None
            print("Warning: Input position data is all NaN.")
            return

        # Transpose to shape (nsamples, nfeatures) for PCA convention
        pos3D = self.pos.transpose()
        nsamples, nfeatures = pos3D.shape

        if nsamples == 0:
             self.pos_1D = np.array([])
             self.pca_model = None
             print("Warning: Input position data has zero samples.")
             return

        # --- Determine data range for mean calculation ---
        if automatic_range:
            # compute PCA of 3D position using meaningful data
            # only during stimulation -> reduce the range to when first and last contact occured
            valid_values_mask = ~np.isnan(self.depth) & (self.depth != 0)
            valid_indices = np.where(valid_values_mask)[0]
            if valid_indices.size > 0:
                start =valid_indices[0]
                end = valid_indices[-1]
                data_for_mean_calc = pos3D[start:end, :]
            else:
                print(f"Warning: Invalid pca_range format {pca_range}. Expected (start, end). Using full range for mean calculation.")
                data_for_mean_calc = pos3D

        elif pca_range:
            try:
                start, end = pca_range
                # Basic validation
                if not (0 <= start < end <= nsamples):
                    print(f"Warning: Invalid pca_range {pca_range} for nsamples={nsamples}. Using full range for mean calculation.")
                else:
                    data_for_mean_calc = pos3D[start:end, :]
                    if data_for_mean_calc.size == 0:
                         print(f"Warning: pca_range {pca_range} results in an empty slice. Using full range for mean calculation.")
                         data_for_mean_calc = pos3D # Fallback
            except (TypeError, ValueError) as e:
                 print(f"Warning: Invalid pca_range format {pca_range}. Expected (start, end). Using full range for mean calculation. Error: {e}")
                 pca_range = None # Reset pca_range if format is wrong
        else:
            data_for_mean_calc = pos3D
            
        # --- Centering Step ---
        # Calculate mean, ignoring NaNs, over the specified range (or full data)
        M = np.nanmean(data_for_mean_calc, axis=0)

        # Check if mean calculation failed (e.g., the slice was all NaNs)
        if np.isnan(M).any():
            print("Warning: Mean calculation over specified range resulted in NaN. Attempting mean over full data.")
            M = np.nanmean(pos3D, axis=0)
            if np.isnan(M).any():
                print("Error: Cannot compute a valid mean for centering (full data also results in NaN mean). Aborting PCA.")
                self.pos_1D = np.full(nsamples, np.nan)
                self.pca_model = None
                return

        # Center the *entire* dataset using the calculated mean
        # Using broadcasting which is generally preferred over np.matlib.repmat
        pos3D_centered = pos3D - M

        # --- Handle NaNs (Interpolation) ---
        pos3D_processed = pos3D_centered # Start with centered data
        if np.isnan(pos3D_centered).any():
            print("Info: NaN values detected, attempting interpolation...")
            try:
                # IMPORTANT: Ensure interpolate_nan_values handles NaNs appropriately
                # and returns an array of the same shape without NaNs.
                pos3D_processed = interpolate_nan_values(pos3D_centered)

                # Verify interpolation didn't fail catastrophically
                if np.isnan(pos3D_processed).any():
                     # If some NaNs remain (interpolation method might fail on edges/all-NaN columns),
                     # PCA might still fail. Consider alternative handling or erroring.
                     print("Warning: NaNs remain after interpolation. PCA might fail.")
                     # Option: Fill remaining NaNs with 0 or column mean before PCA?
                     # For now, we proceed, but PCA might raise an error.
                     # Example: Fill remaining NaNs with 0
                     # nan_mask = np.isnan(pos3D_processed)
                     # pos3D_processed[nan_mask] = 0
                     # print("Warning: Filling remaining NaNs with 0 before PCA.")

            except Exception as e:
                print(f"Error during NaN interpolation: {e}. Aborting PCA.")
                self.pos_1D = np.full(nsamples, np.nan)
                self.pca_model = None
                return

        # --- Check for Zero Variance Data after processing ---
        # PCA requires variance. Check if the data to be used for fitting has variance.
        fit_data_check = pos3D_processed
        if pca_range:
            # Use the validated range indices
            try:
                start, end = pca_range
                if 0 <= start < end <= nsamples:
                    fit_data_check = pos3D_processed[start:end, :]
                else:
                    # Range was invalid earlier, so fit_transform will use full data
                    pass # Use full pos3D_processed for check
            except: # Catch potential issues if pca_range became invalid
                 pass # Use full pos3D_processed for check


        # Check if data (or subset) is effectively constant/zero
        if fit_data_check.size == 0 or np.allclose(np.nanstd(fit_data_check, axis=0), 0):
            print("Warning: Data (or specified PCA range) has zero variance after processing. Setting pos_1D to zeros.")
            self.pos_1D = np.zeros(nsamples)
            self.pca_model = None # No meaningful PCA model
            return

        # --- Perform PCA ---
        pca = PCA(n_components=1)
        final_pos_1D = None

        try:
            if pca_range:
                 # Use validated range again
                start, end = pca_range # Assume validated above
                if 0 <= start < end <= nsamples:
                    fit_data = pos3D_processed[start:end, :]
                    print(f"Fitting PCA on samples [{start}:{end}]...")
                    pca.fit(fit_data)
                    print("Transforming entire dataset...")
                    final_pos_1D = np.squeeze(pca.transform(pos3D_processed))
                else:
                    # Should have been caught, but as a fallback:
                    print("Warning: Invalid pca_range detected at PCA stage. Fitting and transforming on full data.")
                    final_pos_1D = np.squeeze(pca.fit_transform(pos3D_processed))
            else:
                # No range specified, fit and transform on the whole processed data
                print("Fitting PCA and transforming full dataset...")
                final_pos_1D = np.squeeze(pca.fit_transform(pos3D_processed))

            self.pos_1D = final_pos_1D
            self.pca_model = pca # Store the fitted model

        except ValueError as e:
             # Catch potential errors from PCA (e.g., remaining NaNs if interpolation failed)
             print(f"Error during PCA execution: {e}. Aborting PCA.")
             self.pos_1D = np.full(nsamples, np.nan)
             self.pca_model = None


    @property
    def pos_1D(self):
        return self._pos_1D

    @pos_1D.setter
    def pos_1D(self, value):
        self._pos_1D = value
