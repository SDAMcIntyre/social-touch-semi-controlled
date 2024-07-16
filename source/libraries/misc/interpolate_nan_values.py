import numpy as np
from scipy import interpolate

def interpolate_nan_values(xyz):
    """
    Interpolates NaN values in a vector of XYZ positions.

    Parameters:
        xyz (np.ndarray): An Nx3 array of XYZ positions with NaN values.

    Returns:
        np.ndarray: An Nx3 array with NaN values interpolated.
    """
    # Split the XYZ array into separate X, Y, and Z arrays
    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]

    # Create an index array
    indices = np.arange(len(xyz))

    # Function to interpolate a single dimension
    def interpolate_dim(values):
        # Identify NaNs and valid values
        nan_mask = np.isnan(values)
        valid_mask = ~nan_mask

        # Interpolate using the valid values
        interp_func = interpolate.interp1d(indices[valid_mask], values[valid_mask], bounds_error=False, fill_value=0)
        interpolated_values = values.copy()
        interpolated_values[nan_mask] = interp_func(indices[nan_mask])

        return interpolated_values

    # Interpolate each dimension
    X_interp = interpolate_dim(X)
    Y_interp = interpolate_dim(Y)
    Z_interp = interpolate_dim(Z)

    # Combine the interpolated dimensions back into a single array
    xyz_interp = np.vstack((X_interp, Y_interp, Z_interp)).T

    return xyz_interp
