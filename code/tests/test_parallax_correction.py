"""Unit tests for parallax_correction.apply_parallax_shift.

Covers all items from the Phase 1 Testing Plan in:
  docs/development/plans/active/kinect-frame-parallax-correction.md
"""

import numpy as np
import pytest

from preprocessing.common.data_access.parallax_correction import apply_parallax_shift

H, W = 600, 800


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_float(dtype=np.float32) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((H, W)).astype(dtype)


def _random_float_3c(dtype=np.float32) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.standard_normal((H, W, 3)).astype(dtype)


# ---------------------------------------------------------------------------
# Shape preservation
# ---------------------------------------------------------------------------

def test_shape_2d():
    src = _random_float()
    out = apply_parallax_shift(src)
    assert out.shape == (H, W)


def test_shape_3d():
    src = _random_float_3c()
    out = apply_parallax_shift(src)
    assert out.shape == (H, W, 3)


# ---------------------------------------------------------------------------
# NaN geometry
# ---------------------------------------------------------------------------

def test_nan_top_row():
    src = _random_float()
    out = apply_parallax_shift(src)
    assert np.all(np.isnan(out[0, :]))


def test_nan_right_10_cols():
    src = _random_float()
    out = apply_parallax_shift(src)
    assert np.all(np.isnan(out[:, W - 10:]))


def test_interior_pixel_finite():
    src = _random_float()
    out = apply_parallax_shift(src)
    assert np.isfinite(out[1, 0])


# ---------------------------------------------------------------------------
# In-bounds equivalence: out[v, u] == src[v - 1, u + 10]
# ---------------------------------------------------------------------------

def test_inbounds_equivalence_2d():
    src = _random_float()
    out = apply_parallax_shift(src)
    # Check a handful of in-bounds pixels
    for v, u in [(1, 0), (5, 50), (H - 1, W - 11)]:
        assert out[v, u] == pytest.approx(src[v - 1, u + 10])


def test_inbounds_equivalence_3d():
    src = _random_float_3c()
    out = apply_parallax_shift(src)
    for v, u in [(1, 0), (10, 100), (H - 1, W - 11)]:
        np.testing.assert_array_equal(out[v, u, :], src[v - 1, u + 10, :])


# ---------------------------------------------------------------------------
# dtype handling
# ---------------------------------------------------------------------------

def test_integer_promoted_to_float32():
    src = np.ones((H, W), dtype=np.int16)
    out = apply_parallax_shift(src)
    assert out.dtype == np.float32


def test_float64_unchanged():
    src = _random_float(dtype=np.float64)
    out = apply_parallax_shift(src)
    assert out.dtype == np.float64


def test_float32_unchanged():
    src = _random_float(dtype=np.float32)
    out = apply_parallax_shift(src)
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Sentinel pixel mapping:
# out[v, u] = src[v - 1, u + 10]  →  out[500, 500] = src[499, 510]
# ---------------------------------------------------------------------------

def test_sentinel_pixel():
    src = np.zeros((H, W), dtype=np.float32)
    sentinel = 999.0
    src[499, 510] = sentinel
    out = apply_parallax_shift(src)
    assert out[500, 500] == pytest.approx(sentinel)


# ---------------------------------------------------------------------------
# No wraparound (np.roll must NOT be used)
# ---------------------------------------------------------------------------

def test_no_wraparound_edges():
    """Values from the opposite edge must not appear in the NaN border."""
    src = np.ones((H, W), dtype=np.float32) * 42.0
    out = apply_parallax_shift(src)
    # Top row and right-10 cols must be NaN, not 42
    assert np.all(np.isnan(out[0, :]))
    assert np.all(np.isnan(out[:, W - 10:]))
