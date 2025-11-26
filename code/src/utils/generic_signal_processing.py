import numpy as np
from sklearn.decomposition import PCA
from scipy.signal import savgol_filter
from typing import Optional, Tuple, Dict, Any, Literal

# --- Helper Function ---
def _normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Normalizes a 1D signal to the range [0, 1].
    Handles constant signals to avoid division by zero.
    """
    sig_min = np.min(signal)
    sig_max = np.max(signal)
    
    if sig_max - sig_min == 0:
        return np.zeros_like(signal)
    
    return (signal - sig_min) / (sig_max - sig_min)

# --- 1. Standardized Version (Unchanged) ---
def get_pca1_signal(chunk_xyz: np.ndarray) -> np.ndarray:
    """
    Computes the 1D PCA projection of 3D data with robust fitting and Z-alignment.

    Standardized version of the original _get_pca_signal logic.
    
    Pipeline:
    1. NaN Interpolation (Linear).
    2. Robust PCA Fit (Middle 60% of data).
    3. Z-Axis Orientation Correction.
    4. Savitzky-Golay Smoothing.
    5. MinMax Normalization.

    Args:
        chunk_xyz (np.ndarray): Input array of shape (N, P). Assumes columns are X, Y, Z.

    Returns:
        np.ndarray: Normalized 1D signal of shape (N,).
    """
    # Validation
    if chunk_xyz.ndim != 2 or chunk_xyz.shape[0] < 5:
        return np.zeros(chunk_xyz.shape[0])

    # Avoid side effects
    data = chunk_xyz.copy()
    n_samples, n_features = data.shape

    # 1. Interpolation
    for i in range(n_features):
        col = data[:, i]
        nans = np.isnan(col)
        if np.any(nans):
            if np.all(nans):
                return np.zeros(n_samples)
            # Vectorized boolean indexing is faster than np.where for simple masks
            valid_mask = ~nans
            data[nans, i] = np.interp(
                np.flatnonzero(nans), 
                np.flatnonzero(valid_mask), 
                col[valid_mask]
            )

    # 2. Robust Fitting (Middle 60%)
    start_idx = int(n_samples * 0.2)
    end_idx = int(n_samples * 0.8)

    if start_idx < end_idx:
        fit_subset = data[start_idx:end_idx, :]
    else:
        fit_subset = data

    pca = PCA(n_components=1)
    pca.fit(fit_subset)

    # 3. Transformation
    signal = pca.transform(data)[:, 0]

    # 4. Z-Awareness (Orientation Correction)
    # Default to last column if < 3 columns, else index 2 (Z)
    z_idx = 2 if n_features >= 3 else -1
    z_component = pca.components_[0, z_idx]

    if z_component < 0:
        signal = -signal

    # 5. Smoothing (Savitzky-Golay)
    window_len = max(5, int(n_samples * 0.05))
    if window_len % 2 == 0:
        window_len += 1
    
    # Check polyorder constraint
    poly_order = 2
    if window_len <= poly_order:
        window_len = poly_order + 1

    smoothed_signal = savgol_filter(signal, window_length=window_len, polyorder=poly_order)

    # 6. Normalization
    return _normalize_signal(smoothed_signal)


# --- 2. Configurable Version (Modified) ---
def get_pca1_signal_configurable(
    chunk_xyz: np.ndarray,
    *,
    enable_interpolation: bool = True,
    enable_robust_fit: bool = True,
    robust_fit_ratio: Tuple[float, float] = (0.2, 0.8),
    enable_z_correction: bool = True,
    z_correction_mode: Literal['positive', 'negative'] = 'positive',
    enable_smoothing: bool = True,
    smoothing_params: Optional[Dict[str, Any]] = None,
    enable_normalization: bool = True
) -> np.ndarray:
    """
    Computes 1D PCA projection with toggleable processing steps.

    Args:
        chunk_xyz (np.ndarray): Input data (N, P).
        enable_interpolation (bool): If True, fills NaNs via linear interpolation.
        enable_robust_fit (bool): If True, fits PCA on a subset of data.
        robust_fit_ratio (tuple): (start_percent, end_percent) for robust fitting.
        enable_z_correction (bool): If True, applies orientation correction based on Z-axis.
        z_correction_mode (str): Direction of alignment (only used if enable_z_correction is True).
                                 'positive': Align signal with +Z (Standard).
                                 'negative': Align signal with -Z (Opposite).
        enable_smoothing (bool): If True, applies Savitzky-Golay filter.
        smoothing_params (dict): Options for smoothing (e.g., {'polyorder': 2, 'window_ratio': 0.05}).
        enable_normalization (bool): If True, normalizes output to [0, 1].

    Returns:
        np.ndarray: The processed 1D signal.
    """
    # Validation
    if chunk_xyz.ndim != 2 or chunk_xyz.shape[0] < 5:
        return np.zeros(chunk_xyz.shape[0])

    data = chunk_xyz.copy()
    n_samples, n_features = data.shape

    # --- Step 1: Interpolation ---
    if enable_interpolation:
        for i in range(n_features):
            col = data[:, i]
            nans = np.isnan(col)
            if np.any(nans):
                if np.all(nans):
                    # Fallback: if interpolation needed but col is empty, PCA fails.
                    return np.zeros(n_samples)
                valid_mask = ~nans
                data[nans, i] = np.interp(
                    np.flatnonzero(nans),
                    np.flatnonzero(valid_mask),
                    col[valid_mask]
                )
    else:
        # If interpolation is off, Scikit-Learn PCA will raise error on NaNs.
        if np.isnan(data).any():
            return np.zeros(n_samples)

    # --- Step 2: PCA Fitting ---
    pca = PCA(n_components=1)

    if enable_robust_fit:
        start_idx = int(n_samples * robust_fit_ratio[0])
        end_idx = int(n_samples * robust_fit_ratio[1])
        
        if start_idx < end_idx:
            fit_data = data[start_idx:end_idx, :]
        else:
            fit_data = data
    else:
        fit_data = data

    pca.fit(fit_data)
    signal = pca.transform(data)[:, 0]

    # --- Step 3: Z-Orientation Correction ---
    if enable_z_correction:
        # Default to last column if < 3 columns, else index 2 (Z)
        z_idx = 2 if n_features >= 3 else -1
        z_component = pca.components_[0, z_idx]
        
        if z_correction_mode == 'positive':
            # Align with Positive Z (Signal goes up as Z goes up)
            if z_component < 0:
                signal = -signal
        elif z_correction_mode == 'negative':
            # Align with Negative Z (Signal goes up as Z goes down)
            if z_component > 0:
                signal = -signal

    # --- Step 4: Smoothing ---
    if enable_smoothing:
        # Default config
        params = {'window_ratio': 0.05, 'poly_order': 2, 'min_window': 5}
        if smoothing_params:
            params.update(smoothing_params)

        window_len = max(params['min_window'], int(n_samples * params['window_ratio']))
        if window_len % 2 == 0:
            window_len += 1
        
        polyorder = params['poly_order']
        if window_len <= polyorder:
            window_len = polyorder + 1
            
        signal = savgol_filter(signal, window_length=window_len, polyorder=polyorder)

    # --- Step 5: Normalization ---
    if enable_normalization:
        signal = _normalize_signal(signal)

    return signal