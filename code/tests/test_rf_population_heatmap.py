"""Unit tests for rf_population_heatmap pure functions."""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup and stubs
# ---------------------------------------------------------------------------

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _stub(dotted: str, **attrs) -> None:
    if dotted in sys.modules:
        return
    mod = types.ModuleType(dotted)
    parts = dotted.split(".")
    pkg_dir = _SRC / Path(*parts)
    if pkg_dir.exists():
        mod.__path__ = [str(pkg_dir)]
    mod.__package__ = dotted
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[dotted] = mod
    if "." in dotted:
        parent, child = dotted.rsplit(".", 1)
        if parent in sys.modules:
            setattr(sys.modules[parent], child, sys.modules[dotted])


# Stub the heavy package roots so only the module under test is loaded.
_stub("utils")
_stub("utils.should_process_task",
      should_process_task=lambda **kw: True,
      clean_task_outputs=lambda **kw: None)
_stub("analysis")
_stub("analysis.touch_analytics")
_stub("analysis.touch_analytics.clustering_pipeline",
      GESTURE_TYPES=["tap", "stroke_proximal", "stroke_distal"])
_stub("analysis.receptive_field_mapping")

# ---------------------------------------------------------------------------
# Imports under test (after stubs are in place)
# ---------------------------------------------------------------------------

from analysis.receptive_field_mapping.data.rf_population_heatmap import (  # noqa: E402
    apply_vertex_threshold,
    build_gesture_touch_indices,
    compute_rf_heatmap,
    compute_threshold_from_ratio,
    compute_unique_touch_count,
)


# ===========================================================================
# test_compute_rf_heatmap
# ===========================================================================


class TestComputeRfHeatmap:
    """Tests for compute_rf_heatmap."""

    def _make_rf_data(self):
        """Three touches, 10 vertices.

        Touch 0: contacts vertices [0, 1] with values [10.0, 20.0]
        Touch 1: contacts vertices [1, 2] with values [30.0, 40.0]
        Touch 2: contacts vertices [3]    with values [50.0]
        Vertices 4–9 are uncontacted.
        """
        rf_vertex_indices = [
            np.array([0, 1]),
            np.array([1, 2]),
            np.array([3]),
        ]
        rf_values = [
            np.array([10.0, 20.0]),
            np.array([30.0, 40.0]),
            np.array([50.0]),
        ]
        return rf_vertex_indices, rf_values

    def test_mean_aggregation_contacted_vertices(self) -> None:
        rv, vals = self._make_rf_data()
        result = compute_rf_heatmap([0, 1, 2], rv, vals, n_verts=10)

        # vertex 0: only touch 0 → 10.0
        assert result[0] == pytest.approx(10.0)
        # vertex 1: touches 0 and 1 → mean(20.0, 30.0) = 25.0
        assert result[1] == pytest.approx(25.0)
        # vertex 2: only touch 1 → 40.0
        assert result[2] == pytest.approx(40.0)
        # vertex 3: only touch 2 → 50.0
        assert result[3] == pytest.approx(50.0)

    def test_uncontacted_vertices_are_nan(self) -> None:
        rv, vals = self._make_rf_data()
        result = compute_rf_heatmap([0, 1, 2], rv, vals, n_verts=10)

        for v in range(4, 10):
            assert math.isnan(result[v]), f"vertex {v} should be NaN"

    def test_empty_touch_list_returns_all_nan(self) -> None:
        rv, vals = self._make_rf_data()
        result = compute_rf_heatmap([], rv, vals, n_verts=10)

        assert result.shape == (10,)
        assert np.all(np.isnan(result))

    def test_subset_of_touches(self) -> None:
        rv, vals = self._make_rf_data()
        # Only touch 0: vertex 0 → 10.0, vertex 1 → 20.0, rest NaN.
        result = compute_rf_heatmap([0], rv, vals, n_verts=10)

        assert result[0] == pytest.approx(10.0)
        assert result[1] == pytest.approx(20.0)
        assert math.isnan(result[2])

    def test_touch_with_empty_vertex_array_skipped(self) -> None:
        rv = [np.array([]), np.array([5])]
        vals = [np.array([]), np.array([99.0])]
        result = compute_rf_heatmap([0, 1], rv, vals, n_verts=10)

        assert result[5] == pytest.approx(99.0)
        assert math.isnan(result[0])


# ===========================================================================
# test_compute_unique_touch_count
# ===========================================================================


class TestComputeUniqueTouchCount:
    """Tests for compute_unique_touch_count."""

    def test_per_vertex_deduplication(self) -> None:
        # Three contact points: touch 0 visits vertex 0 twice (should count once).
        cp_vertex_idx = np.array([0, 0, 1])
        cp_touch_idx = np.array([0, 0, 1])
        cp_mask = np.ones(3, dtype=bool)

        result = compute_unique_touch_count(cp_vertex_idx, cp_touch_idx, cp_mask, n_verts=5)

        assert result[0] == 1   # touch 0 visited vertex 0 twice → 1 unique touch
        assert result[1] == 1   # touch 1 visited vertex 1 once
        assert result[2] == 0
        assert result[3] == 0
        assert result[4] == 0

    def test_multiple_distinct_touches_per_vertex(self) -> None:
        # Vertex 0 is visited by touches 0, 1, and 2.
        cp_vertex_idx = np.array([0, 0, 0, 2])
        cp_touch_idx = np.array([0, 1, 2, 0])
        cp_mask = np.ones(4, dtype=bool)

        result = compute_unique_touch_count(cp_vertex_idx, cp_touch_idx, cp_mask, n_verts=5)

        assert result[0] == 3
        assert result[2] == 1

    def test_masked_out_point_excluded(self) -> None:
        # Two contact points for vertex 0; only the first is unmasked.
        cp_vertex_idx = np.array([0, 0])
        cp_touch_idx = np.array([0, 1])
        cp_mask = np.array([True, False])

        result = compute_unique_touch_count(cp_vertex_idx, cp_touch_idx, cp_mask, n_verts=3)

        # Only touch 0 is included → 1 unique touch for vertex 0.
        assert result[0] == 1

    def test_all_masked_returns_zeros(self) -> None:
        cp_vertex_idx = np.array([0, 1, 2])
        cp_touch_idx = np.array([0, 1, 2])
        cp_mask = np.zeros(3, dtype=bool)

        result = compute_unique_touch_count(cp_vertex_idx, cp_touch_idx, cp_mask, n_verts=5)

        assert np.all(result == 0)

    def test_output_shape_matches_n_verts(self) -> None:
        cp_vertex_idx = np.array([0])
        cp_touch_idx = np.array([0])
        cp_mask = np.ones(1, dtype=bool)

        result = compute_unique_touch_count(cp_vertex_idx, cp_touch_idx, cp_mask, n_verts=20)

        assert result.shape == (20,)
        assert result.dtype == np.int64


# ===========================================================================
# test_apply_vertex_threshold
# ===========================================================================


class TestApplyVertexThreshold:
    """Tests for apply_vertex_threshold."""

    def test_below_threshold_contacted_vertex_set_to_minus_one(self) -> None:
        heatmap_val = np.array([5.0, 3.0, np.nan, np.nan])
        unique_touch_count = np.array([1, 2, 0, 0])
        threshold = 2

        result = apply_vertex_threshold(heatmap_val, unique_touch_count, threshold)

        # vertex 0: count=1 < threshold=2 and contacted → set to -1.0
        assert result[0] == pytest.approx(-1.0)
        # vertex 1: count=2 == threshold → NOT below threshold, keep value
        assert result[1] == pytest.approx(3.0)

    def test_uncontacted_nan_vertices_unchanged(self) -> None:
        heatmap_val = np.array([np.nan, np.nan])
        unique_touch_count = np.array([0, 0])

        result = apply_vertex_threshold(heatmap_val, unique_touch_count, threshold=1)

        assert math.isnan(result[0])
        assert math.isnan(result[1])

    def test_above_threshold_vertices_keep_value(self) -> None:
        heatmap_val = np.array([7.5, 0.1])
        unique_touch_count = np.array([5, 10])

        result = apply_vertex_threshold(heatmap_val, unique_touch_count, threshold=3)

        assert result[0] == pytest.approx(7.5)
        assert result[1] == pytest.approx(0.1)

    def test_threshold_zero_nothing_masked(self) -> None:
        # threshold=0: no contacted vertex can be < 0, so nothing is set to -1.
        heatmap_val = np.array([1.0, 2.0, 3.0])
        unique_touch_count = np.array([1, 1, 1])

        result = apply_vertex_threshold(heatmap_val, unique_touch_count, threshold=0)

        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(2.0)
        assert result[2] == pytest.approx(3.0)

    def test_does_not_modify_input_array(self) -> None:
        heatmap_val = np.array([5.0, 3.0])
        unique_touch_count = np.array([1, 5])
        original = heatmap_val.copy()

        apply_vertex_threshold(heatmap_val, unique_touch_count, threshold=3)

        np.testing.assert_array_equal(heatmap_val, original)


# ===========================================================================
# test_compute_threshold_from_ratio
# ===========================================================================


class TestComputeThresholdFromRatio:
    """Tests for compute_threshold_from_ratio."""

    def test_25_percent_of_100(self) -> None:
        assert compute_threshold_from_ratio(25.0, 100) == 25

    def test_50_percent_of_40(self) -> None:
        assert compute_threshold_from_ratio(50.0, 40) == 20

    def test_100_percent_of_10(self) -> None:
        assert compute_threshold_from_ratio(100.0, 10) == 10

    def test_zero_percent_of_zero_returns_floor_of_1(self) -> None:
        # max(1, round(0)) → 1
        assert compute_threshold_from_ratio(0.0, 0) == 1

    def test_small_ratio_rounds_to_minimum_1(self) -> None:
        # 1% of 1 = 0.01 → rounds to 0 → clamped to 1
        assert compute_threshold_from_ratio(1.0, 1) == 1

    def test_result_is_int(self) -> None:
        result = compute_threshold_from_ratio(25.0, 100)
        assert isinstance(result, int)


# ===========================================================================
# test_build_gesture_touch_indices
# ===========================================================================


class TestBuildGestureTouchIndices:
    """Tests for build_gesture_touch_indices."""

    def test_returns_correct_indices_for_tap(self) -> None:
        gesture_types = np.array(["tap", "stroke_proximal", "tap", "stroke_distal"])
        result = build_gesture_touch_indices(gesture_types, "tap")

        np.testing.assert_array_equal(result, np.array([0, 2]))

    def test_returns_empty_when_no_match(self) -> None:
        gesture_types = np.array(["tap", "tap"])
        result = build_gesture_touch_indices(gesture_types, "stroke_distal")

        assert len(result) == 0

    def test_returns_all_when_all_match(self) -> None:
        gesture_types = np.array(["stroke_proximal"] * 5)
        result = build_gesture_touch_indices(gesture_types, "stroke_proximal")

        np.testing.assert_array_equal(result, np.arange(5))
