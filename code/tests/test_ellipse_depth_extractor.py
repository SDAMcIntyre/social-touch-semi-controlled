"""Unit tests for EllipseDepthExtractor.

Covers all items from the Testing Plan in:
  docs/development/plans/pending/sticker-depth-edge-gradient-bias-correction.md
"""

import math

import numpy as np
import pandas as pd
import pytest

from preprocessing.stickers_analysis.xyz.core.xyz_extractor_ellipse_depth import (
    EllipseDepthExtractor,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_uniform_point_cloud(h: int, w: int, z_val: float) -> np.ndarray:
    """Return (h, w, 3) float32 array with x=col, y=row, z=z_val."""
    col = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    row = np.tile(np.arange(h, dtype=np.float32), (w, 1)).T
    z = np.full((h, w), z_val, dtype=np.float32)
    return np.stack([col, row, z], axis=-1)


def _make_row(
    cx: float = 50.0,
    cy: float = 50.0,
    axes_major: float = 10.0,
    axes_minor: float = 8.0,
    angle: float = 0.0,
    status: str = "Tracking",
) -> pd.Series:
    return pd.Series({
        "center_x": cx,
        "center_y": cy,
        "axes_major": axes_major,
        "axes_minor": axes_minor,
        "angle": angle,
        "status": status,
    })


# ---------------------------------------------------------------------------
# _build_ellipse_mask
# ---------------------------------------------------------------------------

class TestBuildEllipseMask:

    def test_returns_correct_shape(self):
        mask = EllipseDepthExtractor._build_ellipse_mask(
            shape=(120, 160), center_x=80, center_y=60,
            axes_major=20, axes_minor=10, angle=0,
        )
        assert mask.shape == (120, 160)

    def test_dtype_is_bool(self):
        mask = EllipseDepthExtractor._build_ellipse_mask(
            shape=(100, 100), center_x=50, center_y=50,
            axes_major=10, axes_minor=10, angle=0,
        )
        assert mask.dtype == bool

    def test_center_pixel_is_true(self):
        mask = EllipseDepthExtractor._build_ellipse_mask(
            shape=(100, 100), center_x=50, center_y=50,
            axes_major=15, axes_minor=10, angle=0,
        )
        # mask[row, col] → center_y=50, center_x=50
        assert mask[50, 50]

    def test_pixel_count_approximates_ellipse_area(self):
        # Circle of radius 10 → area = π*10² ≈ 314.
        # cv2 rasterisation includes boundary pixels, so allow ±20%.
        mask = EllipseDepthExtractor._build_ellipse_mask(
            shape=(200, 200), center_x=100, center_y=100,
            axes_major=10, axes_minor=10, angle=0,
        )
        expected = math.pi * 10 * 10
        assert expected * 0.8 < mask.sum() < expected * 1.2

    def test_small_axes_do_not_raise(self):
        # Axes below 1 px should be clamped without exception.
        mask = EllipseDepthExtractor._build_ellipse_mask(
            shape=(50, 50), center_x=25, center_y=25,
            axes_major=0.4, axes_minor=0.3, angle=0,
        )
        assert mask.shape == (50, 50)


# ---------------------------------------------------------------------------
# _sample_depth_within_mask
# ---------------------------------------------------------------------------

class TestSampleDepthWithinMask:

    def _circle_mask(self, shape=(100, 100), cx=50, cy=50, r=10):
        return EllipseDepthExtractor._build_ellipse_mask(
            shape=shape, center_x=cx, center_y=cy,
            axes_major=r, axes_minor=r, angle=0,
        )

    def test_all_zero_cloud_returns_empty(self):
        pc = np.zeros((100, 100, 3), dtype=np.float32)
        mask = self._circle_mask()
        xs, ys, zs = EllipseDepthExtractor._sample_depth_within_mask(pc, mask)
        assert len(zs) == 0

    def test_all_nan_cloud_returns_empty(self):
        pc = np.full((100, 100, 3), np.nan, dtype=np.float32)
        mask = self._circle_mask()
        xs, ys, zs = EllipseDepthExtractor._sample_depth_within_mask(pc, mask)
        assert len(zs) == 0

    def test_returns_only_valid_points(self):
        pc = np.zeros((100, 100, 3), dtype=np.float32)
        pc[50, 50] = [10.0, 20.0, 700.0]
        pc[52, 50] = [10.0, 22.0, 701.0]
        mask = self._circle_mask()
        xs, ys, zs = EllipseDepthExtractor._sample_depth_within_mask(pc, mask)
        assert len(zs) == 2
        assert set(zs.tolist()) == {700.0, 701.0}

    def test_mask_clipped_to_cloud_bounds_does_not_raise(self):
        # Mask built for 100×100, sampled from a smaller 80×80 cloud.
        pc = np.zeros((80, 80, 3), dtype=np.float32)
        mask = self._circle_mask(shape=(100, 100))
        xs, ys, zs = EllipseDepthExtractor._sample_depth_within_mask(pc, mask)
        assert len(zs) == 0


# ---------------------------------------------------------------------------
# _aggregate_depth
# ---------------------------------------------------------------------------

class TestAggregateDepth:

    def test_homogeneous_returns_median_z(self):
        zs = np.array([698.0, 699.0, 700.0, 701.0, 702.0])
        xs = np.arange(5, dtype=float)
        ys = np.ones(5)
        _, _, z, _, _ = EllipseDepthExtractor._aggregate_depth(xs, ys, zs, 10.0)
        assert z == pytest.approx(np.median(zs))

    def test_homogeneous_z_std_is_correct(self):
        zs = np.array([700.0, 700.0, 700.0])
        xs = np.ones(3)
        ys = np.ones(3)
        _, _, _, z_std, _ = EllipseDepthExtractor._aggregate_depth(xs, ys, zs, 10.0)
        assert z_std == pytest.approx(0.0, abs=1e-6)

    def test_homogeneous_pixel_count_is_correct(self):
        zs = np.array([700.0, 701.0, 702.0])
        xs = np.ones(3)
        ys = np.ones(3)
        _, _, _, _, n = EllipseDepthExtractor._aggregate_depth(xs, ys, zs, 10.0)
        assert n == 3

    def test_spread_out_returns_shallow_cluster_z(self):
        # 30 hand pixels at 700 mm, 30 background pixels at 800 mm.
        # z_std ≈ 50 mm >> threshold 10 mm → shallow-cluster path selected.
        zs = np.concatenate([np.full(30, 700.0), np.full(30, 800.0)])
        xs = np.ones(60) * 10.0
        ys = np.ones(60) * 5.0
        _, _, z, z_std, n = EllipseDepthExtractor._aggregate_depth(xs, ys, zs, 10.0)
        assert z == pytest.approx(700.0, abs=2.0)
        assert n == 60

    def test_spread_out_z_is_less_than_all_sample_median(self):
        zs = np.concatenate([np.full(30, 700.0), np.full(30, 800.0)])
        xs = np.ones(60) * 10.0
        ys = np.ones(60) * 5.0
        # Force shallow-cluster path with threshold=10, force median path with threshold=1000.
        _, _, z_shallow, _, _ = EllipseDepthExtractor._aggregate_depth(xs, ys, zs, 10.0)
        _, _, z_median, _, _ = EllipseDepthExtractor._aggregate_depth(xs, ys, zs, 1000.0)
        assert z_shallow < z_median


# ---------------------------------------------------------------------------
# extract()
# ---------------------------------------------------------------------------

class TestExtract:

    def test_uses_ellipse_when_valid_data(self):
        extractor = EllipseDepthExtractor()
        pc = _make_uniform_point_cloud(100, 100, 700.0)
        row = _make_row()
        coords, monitor = extractor.extract(row, pc)
        assert not np.isnan(coords["z_mm"])
        assert coords["z_mm"] == pytest.approx(700.0, abs=1.0)

    def test_monitor_contains_z_std_and_n_depth_pixels(self):
        extractor = EllipseDepthExtractor()
        pc = _make_uniform_point_cloud(100, 100, 700.0)
        row = _make_row()
        _, monitor = extractor.extract(row, pc)
        assert "z_std" in monitor
        assert "n_depth_pixels" in monitor
        assert monitor["n_depth_pixels"] >= 3

    def test_fallback_when_axes_major_is_nan(self):
        extractor = EllipseDepthExtractor()
        pc = _make_uniform_point_cloud(100, 100, 700.0)
        row = _make_row(axes_major=np.nan)
        coords, monitor = extractor.extract(row, pc)
        assert monitor["n_depth_pixels"] == 1
        assert np.isnan(monitor["z_std"])
        assert coords["z_mm"] == pytest.approx(700.0, abs=1.0)

    def test_fallback_when_axes_minor_is_nan(self):
        extractor = EllipseDepthExtractor()
        pc = _make_uniform_point_cloud(100, 100, 700.0)
        row = _make_row(axes_minor=np.nan)
        coords, monitor = extractor.extract(row, pc)
        assert monitor["n_depth_pixels"] == 1
        assert np.isnan(monitor["z_std"])

    def test_fallback_when_mask_yields_fewer_than_three_valid_pixels(self):
        extractor = EllipseDepthExtractor()
        # All-zero PC except the centroid pixel: mask samples 1 valid pixel.
        pc = np.zeros((100, 100, 3), dtype=np.float32)
        pc[50, 50] = [50.0, 50.0, 700.0]
        row = _make_row()
        coords, monitor = extractor.extract(row, pc)
        assert monitor["n_depth_pixels"] == 1
        assert np.isnan(monitor["z_std"])
        assert coords["z_mm"] == pytest.approx(700.0)

    def test_fallback_when_point_cloud_is_none(self):
        extractor = EllipseDepthExtractor()
        row = _make_row()
        coords, monitor = extractor.extract(row, None)
        assert monitor["n_depth_pixels"] == 1
        assert np.isnan(monitor["z_std"])
        assert np.isnan(coords["z_mm"])

    def test_output_coords_keys(self):
        extractor = EllipseDepthExtractor()
        pc = _make_uniform_point_cloud(100, 100, 700.0)
        row = _make_row()
        coords, _ = extractor.extract(row, pc)
        assert set(coords.keys()) == {"x_mm", "y_mm", "z_mm"}

    def test_output_monitor_keys(self):
        extractor = EllipseDepthExtractor()
        pc = _make_uniform_point_cloud(100, 100, 700.0)
        row = _make_row()
        _, monitor = extractor.extract(row, pc)
        assert set(monitor.keys()) == {"px", "py", "z_std", "n_depth_pixels"}


# ---------------------------------------------------------------------------
# get_empty_result()
# ---------------------------------------------------------------------------

class TestGetEmptyResult:

    def test_all_fields_are_nan(self):
        extractor = EllipseDepthExtractor()
        coords, monitor = extractor.get_empty_result()
        for key, val in {**coords, **monitor}.items():
            assert np.isnan(val), f"Expected NaN for '{key}', got {val}"

    def test_coords_keys(self):
        extractor = EllipseDepthExtractor()
        coords, _ = extractor.get_empty_result()
        assert set(coords.keys()) == {"x_mm", "y_mm", "z_mm"}

    def test_monitor_keys(self):
        extractor = EllipseDepthExtractor()
        _, monitor = extractor.get_empty_result()
        assert set(monitor.keys()) == {"px", "py", "z_std", "n_depth_pixels"}


# ---------------------------------------------------------------------------
# should_process_row()
# ---------------------------------------------------------------------------

class TestShouldProcessRow:

    @pytest.mark.parametrize("status", ["Failed", "Black Frame", "Ignored"])
    def test_invalid_statuses_return_false(self, status):
        row = pd.Series({"status": status})
        assert not EllipseDepthExtractor.should_process_row(row)

    @pytest.mark.parametrize("status", ["Tracking", "Labeled", "Initial ROI"])
    def test_valid_statuses_return_true(self, status):
        row = pd.Series({"status": status})
        assert EllipseDepthExtractor.should_process_row(row)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_ellipse_entirely_outside_bounds_falls_back(self):
        """Ellipse centre outside the point cloud → empty mask → centroid fallback → NaN."""
        extractor = EllipseDepthExtractor()
        pc = np.zeros((100, 100, 3), dtype=np.float32)
        row = _make_row(cx=150.0, cy=150.0)
        coords, monitor = extractor.extract(row, pc)
        # Centroid OOB → NaN; fallback path (n_depth_pixels=1)
        assert np.isnan(coords["z_mm"])
        assert monitor["n_depth_pixels"] == 1

    def test_all_depth_zeros_in_ellipse_sensor_dropout(self):
        """All pixels within the ellipse report zero depth (sensor dropout) → NaN."""
        extractor = EllipseDepthExtractor()
        pc = np.zeros((100, 100, 3), dtype=np.float32)
        row = _make_row()
        coords, monitor = extractor.extract(row, pc)
        # Zero mask → fallback → zero centroid → NaN
        assert np.isnan(coords["z_mm"])

    def test_missing_ellipse_columns_uses_centroid_fallback(self):
        """Row without ellipse columns (low-score consolidation) → centroid fallback."""
        extractor = EllipseDepthExtractor()
        pc = _make_uniform_point_cloud(100, 100, 700.0)
        row = pd.Series({
            "center_x": 50.0,
            "center_y": 50.0,
            "status": "Tracking",
            # axes_major, axes_minor, angle columns are absent
        })
        coords, monitor = extractor.extract(row, pc)
        # .get("axes_major", np.nan) → NaN → fallback to single-pixel centroid
        assert monitor["n_depth_pixels"] == 1
        assert coords["z_mm"] == pytest.approx(700.0, abs=1.0)

    def test_very_small_ellipse_fewer_than_min_pixels_falls_back(self):
        """Ellipse with axes=1 px at center but surrounding pixels zero → fallback."""
        extractor = EllipseDepthExtractor()
        pc = np.zeros((100, 100, 3), dtype=np.float32)
        # Valid centroid for the fallback path
        pc[50, 50] = [50.0, 50.0, 700.0]
        row = _make_row(axes_major=0.5, axes_minor=0.5)
        coords, monitor = extractor.extract(row, pc)
        # Clamped to 1×1 ellipse; only 1 pixel sampled (if any valid) → fallback
        assert monitor["n_depth_pixels"] == 1
