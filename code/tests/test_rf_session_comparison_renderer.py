from __future__ import annotations

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


_stub("utils")
_stub("utils.pipeline")
_stub("analysis")
_stub("analysis.receptive_field_mapping")
_stub(
    "analysis.receptive_field_mapping.rf_population_grid_pipeline",
    run_population_rf_grid=lambda *a, **kw: None,
    PopulationRFGridConfig=object,
)
_stub(
    "analysis.receptive_field_mapping.rf_population_grid_metrics_pipeline",
    run_population_rf_grid_metrics=lambda *a, **kw: None,
    PopulationRFGridMetricsConfig=object,
)

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from analysis.receptive_field_mapping.rf_session_comparison_renderer import (  # noqa: E402
    _cluster_session_rows,
    _nan_safe_correlation_distance,
)


# ===========================================================================
# Task 3.1 — basic known-correlation values
# ===========================================================================


class TestNanSafeCorrelationDistanceBasic:
    def test_perfectly_correlated_pair_distance_zero(self) -> None:
        row = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        matrix = np.array([row, row])
        dist = _nan_safe_correlation_distance(matrix)
        assert dist.shape == (1,)
        assert dist[0] == pytest.approx(0.0, abs=1e-10)

    def test_anticorrelated_pair_distance_two(self) -> None:
        row = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        matrix = np.array([row, -row])
        dist = _nan_safe_correlation_distance(matrix)
        assert dist.shape == (1,)
        assert dist[0] == pytest.approx(2.0, abs=1e-10)

    def test_uncorrelated_pair_distance_near_one(self) -> None:
        rng = np.random.default_rng(0)
        row0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Orthogonal-ish: alternating sign pattern, zero correlation with monotone
        row1 = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        matrix = np.array([row0, row1])
        dist = _nan_safe_correlation_distance(matrix)
        assert dist.shape == (1,)
        assert dist[0] == pytest.approx(1.0, abs=0.2)

    def test_three_rows_condensed_length(self) -> None:
        row = np.array([1.0, 2.0, 3.0])
        matrix = np.array([row, row, -row])
        dist = _nan_safe_correlation_distance(matrix)
        # 3 rows → C(3,2) = 3 pairs
        assert dist.shape == (3,)

    def test_distance_clamped_within_zero_two(self) -> None:
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((5, 10))
        dist = _nan_safe_correlation_distance(matrix)
        assert np.all(dist >= 0.0)
        assert np.all(dist <= 2.0)


# ===========================================================================
# Task 3.2 — partial NaN overlap between rows
# ===========================================================================


class TestNanSafeCorrelationDistanceWithNans:
    def test_shared_finite_bins_used_for_correlation(self) -> None:
        # row0 and row1 are identical on indices 2-4 (highly correlated in shared bins)
        # indices 0-1 are NaN in one or both rows
        row0 = np.array([np.nan, np.nan, 1.0, 2.0, 3.0])
        row1 = np.array([9.0, np.nan, 1.0, 2.0, 3.0])
        matrix = np.array([row0, row1])
        dist = _nan_safe_correlation_distance(matrix)
        # Shared bins: indices 2,3,4 → perfectly correlated → distance ~0
        assert dist[0] == pytest.approx(0.0, abs=1e-10)

    def test_no_shared_finite_bins_returns_one(self) -> None:
        row0 = np.array([1.0, 2.0, np.nan, np.nan])
        row1 = np.array([np.nan, np.nan, 3.0, 4.0])
        matrix = np.array([row0, row1])
        dist = _nan_safe_correlation_distance(matrix)
        assert dist[0] == pytest.approx(1.0)

    def test_single_shared_bin_returns_one(self) -> None:
        row0 = np.array([1.0, np.nan, np.nan])
        row1 = np.array([np.nan, np.nan, 5.0])
        # Only 0 shared finite bins → distance 1.0
        matrix = np.array([row0, row1])
        dist = _nan_safe_correlation_distance(matrix)
        assert dist[0] == pytest.approx(1.0)

    def test_nans_in_different_positions_uses_overlap(self) -> None:
        # row0 has NaN at index 0; row1 has NaN at index 4
        # shared finite: indices 1,2,3
        row0 = np.array([np.nan, 1.0, 2.0, 3.0, 4.0])
        row1 = np.array([5.0, 1.0, 2.0, 3.0, np.nan])
        matrix = np.array([row0, row1])
        dist = _nan_safe_correlation_distance(matrix)
        # Shared bins [1,2,3] are identical → distance ~0
        assert dist[0] == pytest.approx(0.0, abs=1e-10)


# ===========================================================================
# Task 3.3 — zero-variance (constant) row → distance 1.0
# ===========================================================================


class TestNanSafeCorrelationDistanceConstantRow:
    def test_constant_row_paired_with_varying_returns_one(self) -> None:
        row_const = np.array([3.0, 3.0, 3.0, 3.0])
        row_vary = np.array([1.0, 2.0, 3.0, 4.0])
        matrix = np.array([row_const, row_vary])
        dist = _nan_safe_correlation_distance(matrix)
        assert dist[0] == pytest.approx(1.0)

    def test_both_constant_returns_one(self) -> None:
        matrix = np.array([
            [5.0, 5.0, 5.0],
            [2.0, 2.0, 2.0],
        ])
        dist = _nan_safe_correlation_distance(matrix)
        assert dist[0] == pytest.approx(1.0)

    def test_three_rows_constant_row_in_middle(self) -> None:
        row_a = np.array([1.0, 2.0, 3.0])
        row_const = np.array([0.0, 0.0, 0.0])
        row_b = np.array([3.0, 2.0, 1.0])
        matrix = np.array([row_a, row_const, row_b])
        dist = _nan_safe_correlation_distance(matrix)
        # Pairs: (a, const)=1.0, (a, b)=2.0 (anticorrelated), (const, b)=1.0
        assert dist[0] == pytest.approx(1.0)   # (a, const)
        assert dist[1] == pytest.approx(2.0, abs=1e-10)   # (a, b)
        assert dist[2] == pytest.approx(1.0)   # (const, b)


# ===========================================================================
# Task 3.4 — cluster_session_rows reorders similar rows to be adjacent
# ===========================================================================


class TestClusterSessionRowsReorders:
    def test_reordering_places_similar_rows_adjacent(self) -> None:
        # row 0 and row 2 are highly correlated; row 1 is anticorrelated with both.
        # After clustering the two similar rows must be adjacent.
        # Use distinguishable rows so position lookup is unambiguous.
        row0 = np.array([1.0, 2.0, 3.0, 4.0])
        row1 = np.array([-1.0, -2.0, -3.0, -4.0])
        row2 = np.array([2.0, 4.0, 6.0, 8.0])   # same direction as row0
        matrix = np.array([row0, row1, row2])

        reordered, Z = _cluster_session_rows(matrix)

        assert Z is not None
        assert reordered.shape == matrix.shape

        # Identify where row0 and row2 landed in the reordered output
        pos0 = next(i for i in range(3) if np.allclose(reordered[i], row0))
        pos2 = next(i for i in range(3) if np.allclose(reordered[i], row2))

        assert abs(pos0 - pos2) == 1

    def test_linkage_matrix_returned(self) -> None:
        matrix = np.array([
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
            [1.0, 3.0, 2.0],
        ])
        _, Z = _cluster_session_rows(matrix)
        assert Z is not None
        # Linkage matrix for 3 rows has shape (2, 4)
        assert Z.shape == (2, 4)

    def test_output_shape_unchanged(self) -> None:
        rng = np.random.default_rng(7)
        matrix = rng.standard_normal((5, 8))
        reordered, Z = _cluster_session_rows(matrix)
        assert reordered.shape == matrix.shape
        assert Z is not None


# ===========================================================================
# Task 3.5 — fewer than 3 sessions returns (matrix, None)
# ===========================================================================


class TestClusterSessionRowsFewSessions:
    def test_single_row_returns_original_none_linkage(self) -> None:
        matrix = np.array([[1.0, 2.0, 3.0]])
        result, Z = _cluster_session_rows(matrix)
        assert Z is None
        assert np.array_equal(result, matrix)

    def test_two_rows_returns_original_none_linkage(self) -> None:
        matrix = np.array([
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
        ])
        result, Z = _cluster_session_rows(matrix)
        assert Z is None
        assert np.array_equal(result, matrix)

    def test_two_rows_one_all_nan_returns_none_linkage(self) -> None:
        matrix = np.array([
            [1.0, 2.0, 3.0],
            [np.nan, np.nan, np.nan],
        ])
        result, Z = _cluster_session_rows(matrix)
        assert Z is None

    def test_three_rows_all_nan_returns_none_linkage(self) -> None:
        matrix = np.full((3, 4), np.nan)
        result, Z = _cluster_session_rows(matrix)
        assert Z is None


# ===========================================================================
# Task 3.6 — all-NaN row is appended at the bottom
# ===========================================================================


class TestClusterSessionRowsAllNanRow:
    def test_all_nan_row_is_last(self) -> None:
        nan_row = np.array([np.nan, np.nan, np.nan, np.nan])
        matrix = np.array([
            [1.0, 2.0, 3.0, 4.0],
            nan_row,
            [4.0, 3.0, 2.0, 1.0],
            [1.5, 2.5, 3.5, 4.5],
        ])
        reordered, _ = _cluster_session_rows(matrix)
        assert np.all(~np.isfinite(reordered[-1]))

    def test_all_nan_row_content_preserved(self) -> None:
        nan_row = np.array([np.nan, np.nan, np.nan])
        matrix = np.array([
            [1.0, 2.0, 3.0],
            nan_row,
            [3.0, 2.0, 1.0],
            [2.0, 2.0, 2.0],
        ])
        reordered, _ = _cluster_session_rows(matrix)
        assert np.all(np.isnan(reordered[-1]))

    def test_multiple_all_nan_rows_all_at_bottom(self) -> None:
        nan_row = np.array([np.nan, np.nan, np.nan])
        matrix = np.array([
            [1.0, 2.0, 3.0],
            nan_row,
            [3.0, 2.0, 1.0],
            nan_row,
            [1.5, 2.5, 3.5],
        ])
        reordered, _ = _cluster_session_rows(matrix)
        # Last 2 rows should both be all-NaN
        assert np.all(np.isnan(reordered[-1]))
        assert np.all(np.isnan(reordered[-2]))

    def test_non_nan_rows_not_lost(self) -> None:
        nan_row = np.array([np.nan, np.nan, np.nan, np.nan])
        finite_rows = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [2.0, 2.0, 2.5, 3.0],
        ])
        matrix = np.vstack([finite_rows, nan_row])
        reordered, _ = _cluster_session_rows(matrix)
        # The 3 finite rows must all be present in the first 3 positions
        assert reordered.shape == matrix.shape
        for row in finite_rows:
            match = any(np.allclose(reordered[i], row) for i in range(3))
            assert match, f"Finite row {row} missing from reordered output"
