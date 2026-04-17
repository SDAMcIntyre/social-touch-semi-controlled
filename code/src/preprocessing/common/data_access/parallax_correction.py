"""Parallax correction primitive for Azure Kinect RGB↔depth alignment.

The ``transformed_depth`` / ``transformed_depth_point_cloud`` outputs of
``pyk4a`` retain a residual sub-pixel offset after the rigid color/IR
baseline correction.  At the 500–800 mm operating range the median offset
is **(Δv, Δu) = (−1, +10) px** — i.e. the depth sample that belongs at
RGB pixel ``(v, u)`` lives at raw depth pixel ``(v − 1, u + 10)``.

Measurement details, IQR, and the consumer audit table are in:
  docs/development/knowledge-base/note-azure-kinect-rgb-depth-parallax.md

This module centralises the shift constants and the NaN-padded shift
function so that a future re-calibration touches a single place.
"""

import numpy as np

# (dv, du): to find the depth sample for RGB pixel (v, u),
# look in the raw depth map at (v + dv, u + du) = (v - 1, u + 10).
PARALLAX_SHIFT_PX_RGB_TO_DEPTH = (-1, +10)

PARALLAX_CORRECTION_VERSION = "median_shift_v1"


def apply_parallax_shift(arr: np.ndarray) -> np.ndarray:
    """Return a parallax-corrected copy of *arr*.

    ``out[v, u, ...] == arr[v + dv, u + du, ...]`` for in-bounds indices,
    NaN elsewhere.  With ``(dv, du) = (−1, +10)``::

        out[1:, :W-10] = arr[0:H-1, 10:W]   # in-bounds region
        out[0, :]      = NaN                 # top row
        out[:, W-10:]  = NaN                 # right 10 columns

    Parameters
    ----------
    arr:
        Shape ``(H, W)`` or ``(H, W, C)``.  Integer dtypes are promoted to
        float32 so NaN is representable; float dtypes are kept as-is.

    Returns
    -------
    np.ndarray
        Same shape as *arr*, float dtype, new allocation (no caching).
    """
    dv, du = PARALLAX_SHIFT_PX_RGB_TO_DEPTH  # (-1, +10)

    if arr.ndim == 2:
        H, W = arr.shape
    elif arr.ndim == 3:
        H, W, _ = arr.shape
    else:
        raise ValueError(f"arr must be 2-D or 3-D, got shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.integer):
        out = np.full(arr.shape, np.nan, dtype=np.float32)
    else:
        out = np.full(arr.shape, np.nan, dtype=arr.dtype)

    # out[v, u] = arr[v + dv, u + du]  with dv=-1, du=+10
    # In-bounds v range: v + dv >= 0  →  v >= 1       →  v in [1, H)
    # In-bounds u range: u + du < W   →  u < W - 10   →  u in [0, W-10)
    out[1:, :W - 10] = arr[0:H - 1, 10:W]

    return out
