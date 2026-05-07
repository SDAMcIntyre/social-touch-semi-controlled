"""Unit and integration tests for rf_population_grid_pipeline."""
import sys
import types
from pathlib import Path
import numpy as np
import pytest

_SRC = Path(__file__).parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _stub(dotted: str, **attrs):
    if dotted not in sys.modules:
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


_stub("analysis")
_stub("analysis.receptive_field_mapping")

_stub("analysis.receptive_field_mapping.rf_data_loader",
      load_forearm_vertices=lambda path: None)

_stub("analysis.receptive_field_mapping.touch_population_data",
      load_population_data=None,
      load_population_rf_data=None)

_stub("utils")
_stub("utils.should_process_task", should_process_task=lambda **kw: True)

from analysis.receptive_field_mapping.rf_population_grid_pipeline import (  # noqa: E402
    PopulationRFGridConfig,
    build_feature_grid,
    compute_cell_rf,
    filter_touches_for_cell,
)


class TestBuildFeatureGrid:
    def _config(self, features):
        return PopulationRFGridConfig(
            features=features,
            neuron_mode="iff",
            vertex_threshold_ratio=0.25,
            per_gesture_type=False,
        )

    def test_single_feature_shape(self):
        config = self._config({"vel": {"min": 0.0, "max": 10.0, "step": 2.0, "span": 2.0}})
        grid = build_feature_grid(config)
        assert grid.shape == (5, 1)
        np.testing.assert_allclose(grid[:, 0], [0, 2, 4, 6, 8])

    def test_two_features_cartesian_product(self):
        config = self._config({
            "a": {"min": 0.0, "max": 3.0, "step": 1.0, "span": 1.0},
            "b": {"min": 0.0, "max": 10.0, "step": 5.0, "span": 5.0},
        })
        grid = build_feature_grid(config)
        assert grid.shape == (6, 2)

    def test_three_features_shape(self):
        config = self._config({
            "x": {"min": 0.0, "max": 2.0, "step": 1.0, "span": 1.0},
            "y": {"min": 0.0, "max": 3.0, "step": 1.0, "span": 1.0},
            "z": {"min": 0.0, "max": 4.0, "step": 1.0, "span": 1.0},
        })
        grid = build_feature_grid(config)
        assert grid.shape == (24, 3)

    def test_empty_features_raises(self):
        config = self._config({})
        with pytest.raises(ValueError, match="empty"):
            build_feature_grid(config)

    def test_zero_range_raises(self):
        config = self._config({"x": {"min": 5.0, "max": 5.0, "step": 1.0, "span": 1.0}})
        with pytest.raises(ValueError):
            build_feature_grid(config)


class TestFilterTouchesForCell:
    def test_all_inside(self):
        fm = np.array([[0.0, 0.0], [0.5, 0.5], [-0.5, -0.5]])
        mask = filter_touches_for_cell(fm, np.array([0.0, 0.0]), np.array([2.0, 2.0]))
        assert mask.all()

    def test_boundary_inclusive(self):
        fm = np.array([[1.0]])
        mask = filter_touches_for_cell(fm, np.array([2.0]), np.array([2.0]))
        assert mask[0]

    def test_outside_excluded(self):
        fm = np.array([[5.0, 5.0]])
        mask = filter_touches_for_cell(fm, np.array([0.0, 0.0]), np.array([2.0, 2.0]))
        assert not mask[0]

    def test_nan_excluded(self):
        fm = np.array([[np.nan, 0.0], [0.0, 0.0]])
        mask = filter_touches_for_cell(fm, np.array([0.0, 0.0]), np.array([2.0, 2.0]))
        assert not mask[0]
        assert mask[1]

    def test_partial_nan_excluded(self):
        fm = np.array([[0.0, np.nan]])
        mask = filter_touches_for_cell(fm, np.array([0.0, 0.0]), np.array([2.0, 2.0]))
        assert not mask[0]

    def test_multi_feature_one_outside(self):
        fm = np.array([[0.5, 99.0]])
        mask = filter_touches_for_cell(fm, np.array([0.0, 0.0]), np.array([2.0, 2.0]))
        assert not mask[0]


class TestComputeCellRF:
    def test_empty_mask_all_nan(self):
        rf_vi = [np.array([0, 1]), np.array([2])]
        rf_val = [np.array([1.0, 2.0]), np.array([3.0])]
        mask = np.array([False, False])
        result = compute_cell_rf(rf_vi, rf_val, mask, n_vertices=5, threshold_ratio=0.0)
        assert np.all(np.isnan(result))

    def test_single_touch_correct_values(self):
        rf_vi = [np.array([0, 1]), np.array([2])]
        rf_val = [np.array([2.0, 4.0]), np.array([3.0])]
        mask = np.array([True, False])
        result = compute_cell_rf(rf_vi, rf_val, mask, n_vertices=5, threshold_ratio=0.0)
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(4.0)
        assert np.isnan(result[2])
        assert np.isnan(result[3])
        assert np.isnan(result[4])

    def test_two_touches_mean(self):
        rf_vi = [np.array([0]), np.array([0])]
        rf_val = [np.array([1.0]), np.array([3.0])]
        mask = np.array([True, True])
        result = compute_cell_rf(rf_vi, rf_val, mask, n_vertices=3, threshold_ratio=0.0)
        assert result[0] == pytest.approx(2.0)
        assert np.isnan(result[1])
        assert np.isnan(result[2])

    def test_vertex_threshold_masking(self):
        # 4 touches, threshold_ratio=0.5 → min_touches = max(1, round(0.5*4)) = 2
        # Vertex 0: contacted by all 4 touches → kept
        # Vertex 1: contacted by 1 touch → masked to NaN
        rf_vi = [np.array([0, 1]), np.array([0]), np.array([0]), np.array([0])]
        rf_val = [np.array([1.0, 5.0]), np.array([1.0]), np.array([1.0]), np.array([1.0])]
        mask = np.array([True, True, True, True])
        result = compute_cell_rf(rf_vi, rf_val, mask, n_vertices=5, threshold_ratio=0.5)
        assert not np.isnan(result[0])
        assert np.isnan(result[1])

    def test_uncontacted_vertices_nan(self):
        rf_vi = [np.array([0])]
        rf_val = [np.array([1.0])]
        mask = np.array([True])
        result = compute_cell_rf(rf_vi, rf_val, mask, n_vertices=10, threshold_ratio=0.0)
        for i in range(1, 10):
            assert np.isnan(result[i])

    def test_empty_touch_rf_arrays(self):
        rf_vi = [np.array([], dtype=np.int64), np.array([2])]
        rf_val = [np.array([], dtype=np.float64), np.array([5.0])]
        mask = np.array([True, True])
        result = compute_cell_rf(rf_vi, rf_val, mask, n_vertices=5, threshold_ratio=0.0)
        assert result[2] == pytest.approx(5.0)
        assert np.isnan(result[0])


class TestGridSweepIntegration:
    """End-to-end test of build_feature_grid + filter_touches_for_cell + compute_cell_rf."""

    def test_full_sweep_synthetic(self):
        rng = np.random.default_rng(42)
        T = 10
        n_vertices = 50

        feat_vel = rng.uniform(0, 30, size=T)
        feat_pres = rng.uniform(0, 3, size=T)
        feature_matrix = np.column_stack([feat_vel, feat_pres])

        rf_vi = []
        rf_val = []
        for _ in range(T):
            verts = rng.integers(0, n_vertices, size=5)
            vals = rng.uniform(0, 10, size=5)
            rf_vi.append(verts)
            rf_val.append(vals)

        config = PopulationRFGridConfig(
            features={
                "vel": {"min": 0.0, "max": 30.0, "step": 10.0, "span": 10.0},
                "pres": {"min": 0.0, "max": 3.0, "step": 1.0, "span": 1.0},
            },
            neuron_mode="iff",
            vertex_threshold_ratio=0.0,
            per_gesture_type=False,
        )

        grid_centers = build_feature_grid(config)
        assert grid_centers.shape == (9, 2)

        spans = np.array([config.features["vel"]["span"], config.features["pres"]["span"]])

        rf_maps = []
        touch_counts = []
        for g, center in enumerate(grid_centers):
            mask = filter_touches_for_cell(feature_matrix, center, spans)
            touch_counts.append(int(mask.sum()))
            rf_map = compute_cell_rf(rf_vi, rf_val, mask, n_vertices, threshold_ratio=0.0)
            rf_maps.append(rf_map)

        rf_maps_arr = np.stack(rf_maps)
        touch_counts_arr = np.array(touch_counts, dtype=np.int64)

        assert rf_maps_arr.shape == (9, n_vertices)
        assert touch_counts_arr.shape == (9,)
        assert touch_counts_arr.sum() >= 0
        for g in range(9):
            if touch_counts_arr[g] == 0:
                assert np.all(np.isnan(rf_maps_arr[g]))

    def test_all_same_feature_values_single_nonempty_cell(self):
        T = 5
        n_vertices = 10
        feature_matrix = np.ones((T, 1)) * 5.0

        rf_vi = [np.array([0, 1]) for _ in range(T)]
        rf_val = [np.array([1.0, 2.0]) for _ in range(T)]

        config = PopulationRFGridConfig(
            features={"x": {"min": 0.0, "max": 10.0, "step": 2.0, "span": 2.0}},
            neuron_mode="iff",
            vertex_threshold_ratio=0.0,
            per_gesture_type=False,
        )
        grid_centers = build_feature_grid(config)
        spans = np.array([config.features["x"]["span"]])

        non_empty = 0
        for center in grid_centers:
            mask = filter_touches_for_cell(feature_matrix, center, spans)
            if mask.any():
                non_empty += 1

        assert non_empty >= 1

    def test_overlapping_windows_touch_in_multiple_cells(self):
        T = 1
        n_vertices = 5
        feature_matrix = np.array([[5.0]])

        config = PopulationRFGridConfig(
            features={"x": {"min": 0.0, "max": 10.0, "step": 2.0, "span": 6.0}},
            neuron_mode="iff",
            vertex_threshold_ratio=0.0,
            per_gesture_type=False,
        )
        grid_centers = build_feature_grid(config)
        spans = np.array([6.0])

        cells_containing_touch = 0
        for center in grid_centers:
            mask = filter_touches_for_cell(feature_matrix, center, spans)
            if mask[0]:
                cells_containing_touch += 1

        assert cells_containing_touch >= 2
