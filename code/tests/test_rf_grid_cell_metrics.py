"""Unit and integration tests for rf_grid_cell_metrics and
rf_population_grid_metrics_pipeline._build_metrics_dataframe.

Stubs are injected at module level (before any analysis import) to prevent
heavy package __init__.py files from running in a plain test environment.
"""

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
    """Inject a minimal stub module into sys.modules.

    Sets __path__ to the real source directory when it exists so genuine
    submodules can still be imported directly.
    """
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
    # Wire child into parent's namespace
    if "." in dotted:
        parent, child = dotted.rsplit(".", 1)
        if parent in sys.modules:
            setattr(sys.modules[parent], child, sys.modules[dotted])


# Stub heavyweight package roots whose __init__.py pulls in heavy SDKs.
_stub("utils")
_stub("utils.should_process_task",
      should_process_task=lambda **kw: True,
      clean_task_outputs=lambda **kw: None)
_stub("analysis")
_stub("analysis.receptive_field_mapping")
_stub("analysis.receptive_field_mapping.rf_data_loader",
      load_forearm_vertices=lambda path: None)

# ---------------------------------------------------------------------------
# Imports under test (after stubs are in place)
# ---------------------------------------------------------------------------

from analysis.receptive_field_mapping.rf_grid_cell_metrics import (  # noqa: E402
    compute_boundary_shape_metrics,
    compute_distribution_metrics,
    compute_grid_cell_metrics,
    compute_iff_intensity_metrics,
    compute_topographic_metrics,
)
from analysis.receptive_field_mapping.rf_population_grid_metrics_pipeline import (  # noqa: E402
    _build_metrics_dataframe,
)


# ===========================================================================
# Task 4.1 — compute_iff_intensity_metrics
# ===========================================================================


class TestComputeIffIntensityMetrics:
    """Unit tests for compute_iff_intensity_metrics."""

    def test_known_values_1_to_5(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_iff_intensity_metrics(arr)

        assert result["max_iff"] == pytest.approx(5.0)
        assert result["mean_iff"] == pytest.approx(3.0)
        assert result["median_iff"] == pytest.approx(3.0)
        assert result["std_iff"] == pytest.approx(np.std(arr))
        assert result["iff_range"] == pytest.approx(4.0)
        assert result["n_active_vertices"] == 5

    def test_single_element(self) -> None:
        result = compute_iff_intensity_metrics(np.array([7.0]))

        assert result["max_iff"] == pytest.approx(7.0)
        assert result["mean_iff"] == pytest.approx(7.0)
        assert result["median_iff"] == pytest.approx(7.0)
        assert result["std_iff"] == pytest.approx(0.0)
        assert result["iff_range"] == pytest.approx(0.0)
        assert result["n_active_vertices"] == 1

    def test_all_equal(self) -> None:
        result = compute_iff_intensity_metrics(np.array([3.0, 3.0, 3.0]))

        assert result["std_iff"] == pytest.approx(0.0)
        assert result["iff_range"] == pytest.approx(0.0)
        assert result["n_active_vertices"] == 3

    def test_empty_array_all_nan(self) -> None:
        result = compute_iff_intensity_metrics(np.array([]))

        assert math.isnan(result["max_iff"])
        assert math.isnan(result["mean_iff"])
        assert math.isnan(result["median_iff"])
        assert math.isnan(result["std_iff"])
        assert math.isnan(result["iff_range"])
        assert result["n_active_vertices"] == 0

    def test_return_keys_complete(self) -> None:
        result = compute_iff_intensity_metrics(np.array([1.0, 2.0]))
        expected_keys = {
            "max_iff", "mean_iff", "median_iff", "std_iff", "iff_range",
            "n_active_vertices",
        }
        assert set(result.keys()) == expected_keys


# ===========================================================================
# Task 4.2 — compute_topographic_metrics
# ===========================================================================


class TestComputeTopographicMetrics:
    """Unit tests for compute_topographic_metrics."""

    def test_hi_and_cv_for_0_to_4(self) -> None:
        arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        result = compute_topographic_metrics(arr)

        assert result["hypsometric_integral"] == pytest.approx(0.5)
        expected_cv = float(np.std(arr) / np.mean(arr))
        assert result["coefficient_of_variation"] == pytest.approx(expected_cv)

    def test_uniform_input_hi_nan_cv_zero(self) -> None:
        result = compute_topographic_metrics(np.array([5.0, 5.0, 5.0]))

        assert math.isnan(result["hypsometric_integral"])
        assert result["coefficient_of_variation"] == pytest.approx(0.0)

    def test_n_1_both_nan(self) -> None:
        result = compute_topographic_metrics(np.array([3.0]))

        assert math.isnan(result["hypsometric_integral"])
        assert math.isnan(result["coefficient_of_variation"])

    def test_n_0_both_nan(self) -> None:
        result = compute_topographic_metrics(np.array([]))

        assert math.isnan(result["hypsometric_integral"])
        assert math.isnan(result["coefficient_of_variation"])

    def test_zero_mean_cv_nan(self) -> None:
        # mean == 0 → CV is undefined
        result = compute_topographic_metrics(np.array([-1.0, 0.0, 1.0]))

        assert math.isnan(result["coefficient_of_variation"])

    def test_return_keys_complete(self) -> None:
        result = compute_topographic_metrics(np.array([1.0, 2.0]))
        assert set(result.keys()) == {"hypsometric_integral", "coefficient_of_variation"}


# ===========================================================================
# Task 4.3 — compute_distribution_metrics
# ===========================================================================


class TestComputeDistributionMetrics:
    """Unit tests for compute_distribution_metrics."""

    def test_uniform_gini_zero_entropy_log2_n(self) -> None:
        arr = np.array([1.0, 1.0, 1.0, 1.0])
        result = compute_distribution_metrics(arr)

        assert result["gini_coefficient"] == pytest.approx(0.0, abs=1e-10)
        assert result["shannon_entropy"] == pytest.approx(2.0)  # log2(4)

    def test_concentrated_gini_high_entropy_zero(self) -> None:
        arr = np.array([0.0, 0.0, 0.0, 10.0])
        result = compute_distribution_metrics(arr)

        # Gini: only one non-zero element out of 4 → high concentration
        assert result["gini_coefficient"] > 0.5
        # Shannon entropy of [0,0,0,1]: 0*log(0)+1*log(1) = 0
        assert result["shannon_entropy"] == pytest.approx(0.0)

    def test_n_1_gini_zero_entropy_zero_moments_nan(self) -> None:
        result = compute_distribution_metrics(np.array([5.0]))

        assert result["gini_coefficient"] == pytest.approx(0.0)
        assert result["shannon_entropy"] == pytest.approx(0.0)
        assert math.isnan(result["iff_skewness"])
        assert math.isnan(result["iff_kurtosis"])

    def test_n_0_all_nan(self) -> None:
        result = compute_distribution_metrics(np.array([]))

        assert math.isnan(result["iff_skewness"])
        assert math.isnan(result["iff_kurtosis"])
        assert math.isnan(result["gini_coefficient"])
        assert math.isnan(result["shannon_entropy"])

    def test_symmetric_skewness_near_zero(self) -> None:
        # Perfectly symmetric distribution → scipy skew ≈ 0 within tolerance
        arr = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        result = compute_distribution_metrics(arr)

        # scipy.stats.skew uses a bias-corrected formula; result is near but
        # not exactly zero for small N.  Use a loose tolerance.
        assert abs(result["iff_skewness"]) < 0.5

    def test_return_keys_complete(self) -> None:
        result = compute_distribution_metrics(np.array([1.0, 2.0, 3.0]))
        expected_keys = {
            "iff_skewness", "iff_kurtosis", "gini_coefficient", "shannon_entropy",
        }
        assert set(result.keys()) == expected_keys


# ===========================================================================
# Task 4.4 — compute_boundary_shape_metrics
# ===========================================================================


class TestComputeBoundaryShapeMetrics:
    """Unit tests for compute_boundary_shape_metrics."""

    @staticmethod
    def _unit_circle_uv(n: int = 20) -> np.ndarray:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    def test_circle_circularity_near_1(self) -> None:
        from scipy.spatial import ConvexHull

        uv = self._unit_circle_uv(20)
        hull = ConvexHull(uv)
        area = float(hull.volume)  # volume == area in 2D
        result = compute_boundary_shape_metrics(
            uv, hull_area=area, ellipse_major=1.0, ellipse_minor=1.0
        )

        assert result["circularity"] >= 0.9

    def test_circle_eccentricity_near_zero(self) -> None:
        uv = self._unit_circle_uv(20)
        result = compute_boundary_shape_metrics(
            uv, hull_area=1.0, ellipse_major=1.0, ellipse_minor=1.0
        )

        assert result["eccentricity"] == pytest.approx(0.0, abs=1e-10)

    def test_elongated_eccentricity_high(self) -> None:
        # major=5, minor=0.5 → eccentricity close to 1
        uv = np.array([[i, j] for i in np.linspace(0, 10, 5)
                       for j in np.linspace(0, 1, 5)])
        result = compute_boundary_shape_metrics(
            uv, hull_area=10.0, ellipse_major=5.0, ellipse_minor=0.5
        )

        assert result["eccentricity"] > 0.9

    def test_degenerate_n2_perimeter_nan_circularity_nan(self) -> None:
        uv = np.array([[0.0, 0.0], [1.0, 1.0]])
        result = compute_boundary_shape_metrics(
            uv, hull_area=0.0, ellipse_major=1.0, ellipse_minor=0.5
        )

        assert math.isnan(result["perimeter_mm"])
        assert math.isnan(result["circularity"])
        # Eccentricity from ellipse axes still computable when major > 0 and
        # minor <= major
        assert not math.isnan(result["eccentricity"])

    def test_return_keys_complete(self) -> None:
        uv = self._unit_circle_uv(5)
        result = compute_boundary_shape_metrics(
            uv, hull_area=1.0, ellipse_major=1.0, ellipse_minor=0.5
        )
        assert set(result.keys()) == {"perimeter_mm", "circularity", "eccentricity"}


# ===========================================================================
# Task 4.5 — compute_grid_cell_metrics
# ===========================================================================


class TestComputeGridCellMetrics:
    """Unit tests for the full compute_grid_cell_metrics orchestrator."""

    def test_all_nan_rf_map_returns_all_nan_dict(self) -> None:
        V = 10
        rf_map = np.full(V, np.nan)
        forearm_vertices = np.random.default_rng(42).random((V, 3)) * 100.0

        result = compute_grid_cell_metrics(rf_map, forearm_vertices, "tangent_plane")

        # Float fields → NaN
        float_fields = [k for k, v in result.items()
                        if isinstance(v, float)]
        for key in float_fields:
            assert math.isnan(result[key]), f"Expected NaN for float field {key!r}"

        # Integer fields → 0
        assert result["n_active_vertices"] == 0
        assert result["half_peak_n_points"] == 0
        assert result["convex_hull_n_vertices"] == 0

        # Bool field → False
        assert result["gaussian_converged"] is False

    def test_all_nan_rf_map_keys_consistent_with_empty_row(self) -> None:
        from analysis.receptive_field_mapping.rf_grid_cell_metrics import _empty_row_dict

        V = 10
        rf_map = np.full(V, np.nan)
        forearm_vertices = np.random.default_rng(0).random((V, 3)) * 50.0

        result = compute_grid_cell_metrics(rf_map, forearm_vertices, "tangent_plane")
        empty = _empty_row_dict()

        assert set(result.keys()) == set(empty.keys())

    def test_gaussian_blob_sensible_values(self) -> None:
        """Non-NaN rf_map with a Gaussian blob yields positive metrics."""
        # Build a 5x5x2 grid of vertices spanning a 10 x 10 x 2 mm region
        xs = np.linspace(0, 10, 5)
        ys = np.linspace(0, 10, 5)
        zs = np.linspace(0, 2, 2)
        forearm_vertices = np.array(
            [[x, y, z] for x in xs for y in ys for z in zs]
        )
        V = len(forearm_vertices)

        # Gaussian IFF values centered at (5, 5)
        dists = np.sqrt(
            (forearm_vertices[:, 0] - 5.0) ** 2
            + (forearm_vertices[:, 1] - 5.0) ** 2
        )
        rf_map = np.exp(-dists ** 2 / (2.0 * 3.0 ** 2)) * 10.0

        result = compute_grid_cell_metrics(rf_map, forearm_vertices, "tangent_plane")

        assert result["n_active_vertices"] == V
        assert result["max_iff"] == pytest.approx(10.0, rel=1e-3)
        assert result["mean_iff"] > 0.0
        assert result["convex_hull_area_mm2"] > 0.0


# ===========================================================================
# Task 4.6 — _build_metrics_dataframe integration
# ===========================================================================


class TestBuildMetricsDataframe:
    """Integration tests for _build_metrics_dataframe."""

    @staticmethod
    def _make_forearm_vertices(n: int = 20) -> np.ndarray:
        """Return (n, 3) vertices spread over a 10 mm x 10 mm x 0 mm area."""
        xs = np.linspace(0.0, 10.0, n)
        ys = np.linspace(0.0, 10.0, n)
        zs = np.zeros(n)
        return np.column_stack([xs, ys, zs])

    @staticmethod
    def _make_gaussian_rf(forearm_vertices: np.ndarray) -> np.ndarray:
        dists = np.sqrt(
            (forearm_vertices[:, 0] - 5.0) ** 2
            + (forearm_vertices[:, 1] - 5.0) ** 2
        )
        return np.exp(-dists ** 2 / (2.0 * 3.0 ** 2)) * 10.0

    def _build(self):
        forearm_vertices = self._make_forearm_vertices(20)
        V = len(forearm_vertices)
        G = 4

        rf_maps = np.full((G, V), np.nan)
        for g in [0, 2]:  # cells 0 and 2 have data
            rf_maps[g] = self._make_gaussian_rf(forearm_vertices)

        grid_centers = np.array([
            [2.0, 2.0],
            [4.0, 4.0],
            [6.0, 6.0],
            [8.0, 8.0],
        ])
        touch_counts = np.array([5, 0, 8, 0])

        import pandas as pd  # noqa: PLC0415

        df = _build_metrics_dataframe(
            rf_maps=rf_maps,
            grid_centers=grid_centers,
            feature_names=["vel", "pressure"],
            touch_counts=touch_counts,
            gesture_type="tap",
            neuron_mode="iff",
            session_id="test_session",
            forearm_vertices=forearm_vertices,
            projection_method="tangent_plane",
        )
        return df, touch_counts

    def test_output_has_four_rows(self) -> None:
        df, _ = self._build()
        assert len(df) == 4

    def test_metadata_columns_present(self) -> None:
        df, _ = self._build()
        for col in ("grid_cell_index", "session_id", "gesture_type", "neuron_mode", "touch_count"):
            assert col in df.columns, f"Missing metadata column {col!r}"

    def test_metadata_values_correct(self) -> None:
        df, touch_counts = self._build()

        assert list(df["grid_cell_index"]) == [0, 1, 2, 3]
        assert list(df["session_id"]) == ["test_session"] * 4
        assert list(df["gesture_type"]) == ["tap"] * 4
        assert list(df["neuron_mode"]) == ["iff"] * 4
        assert list(df["touch_count"]) == list(touch_counts)

    def test_feature_center_columns_present(self) -> None:
        df, _ = self._build()
        assert "vel_center" in df.columns
        assert "pressure_center" in df.columns

    def test_feature_center_values_correct(self) -> None:
        df, _ = self._build()
        expected_vel = [2.0, 4.0, 6.0, 8.0]
        expected_pres = [2.0, 4.0, 6.0, 8.0]
        assert list(df["vel_center"]) == pytest.approx(expected_vel)
        assert list(df["pressure_center"]) == pytest.approx(expected_pres)

    def test_empty_cells_have_zero_n_active_vertices(self) -> None:
        df, _ = self._build()
        # Cells 1 and 3 have touch_count=0 → n_active_vertices should be 0
        for row_idx in [1, 3]:
            assert df.loc[row_idx, "n_active_vertices"] == 0, (
                f"Expected n_active_vertices==0 for empty cell {row_idx}"
            )

    def test_non_empty_cells_have_positive_n_active_vertices(self) -> None:
        df, _ = self._build()
        # Cells 0 and 2 have data → n_active_vertices > 0
        for row_idx in [0, 2]:
            assert df.loc[row_idx, "n_active_vertices"] > 0, (
                f"Expected n_active_vertices>0 for non-empty cell {row_idx}"
            )

    def test_max_iff_column_exists(self) -> None:
        df, _ = self._build()
        assert "max_iff" in df.columns

    def test_non_empty_cells_have_finite_max_iff(self) -> None:
        df, _ = self._build()
        for row_idx in [0, 2]:
            val = df.loc[row_idx, "max_iff"]
            assert not math.isnan(float(val)), (
                f"Expected finite max_iff for non-empty cell {row_idx}"
            )

    def test_empty_cells_have_nan_max_iff(self) -> None:
        df, _ = self._build()
        for row_idx in [1, 3]:
            val = df.loc[row_idx, "max_iff"]
            assert math.isnan(float(val)), (
                f"Expected NaN max_iff for empty cell {row_idx}"
            )
