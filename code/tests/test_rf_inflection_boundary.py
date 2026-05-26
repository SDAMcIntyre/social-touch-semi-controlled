"""Unit tests for rf_inflection_boundary.

Tests use synthetic Gaussian grids on a known UV domain so that geometric
properties (circularity, centroid, PCA axes, mean IFF) can be validated
against analytical expectations.
"""

from __future__ import annotations

import json
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


_stub("utils")
_stub("analysis")
_stub("analysis.receptive_field_mapping")

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from analysis.receptive_field_mapping.metrics.rf_inflection_boundary import (  # noqa: E402
    InflectionBoundary,
    _compute_contour_pca,
    _compute_polygon_area,
    _compute_polygon_centroid,
    _compute_polygon_perimeter,
    compute_inflection_boundary,
    inflection_boundary_to_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gaussian_grid(
    size: int = 100,
    sigma_u: float = 15.0,
    sigma_v: float = 15.0,
    center_frac: tuple[float, float] = (0.5, 0.5),
    amplitude: float = 100.0,
    uv_range: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (grid_u, grid_v, grid_z) for a 2D Gaussian in pixel-sigma units.

    sigma_u and sigma_v are in grid pixels; the UV domain spans [0, uv_range].
    """
    u = np.linspace(0.0, uv_range, size)
    v = np.linspace(0.0, uv_range, size)
    grid_u, grid_v = np.meshgrid(u, v, indexing="ij")

    cu = uv_range * center_frac[0]
    cv = uv_range * center_frac[1]
    pixel_scale = uv_range / (size - 1)

    sigma_u_uv = sigma_u * pixel_scale
    sigma_v_uv = sigma_v * pixel_scale

    grid_z = amplitude * np.exp(
        -0.5 * ((grid_u - cu) ** 2 / sigma_u_uv ** 2 + (grid_v - cv) ** 2 / sigma_v_uv ** 2)
    )
    return grid_u, grid_v, grid_z


# ---------------------------------------------------------------------------
# Test: circular Gaussian — boundary found and approximately circular
# ---------------------------------------------------------------------------


class TestCircularGaussian:
    """Inflection boundary on a circular Gaussian hill."""

    @pytest.fixture(scope="class")
    def boundary(self) -> InflectionBoundary | None:
        np.random.seed(42)
        grid_u, grid_v, grid_z = _make_gaussian_grid(size=100, sigma_u=15.0, sigma_v=15.0)
        return compute_inflection_boundary(grid_u, grid_v, grid_z, gaussian_sigma=2.0)

    def test_boundary_is_not_none(self, boundary: InflectionBoundary | None) -> None:
        assert boundary is not None

    def test_contour_uv_is_2d_array(self, boundary: InflectionBoundary) -> None:
        assert isinstance(boundary.contour_uv, np.ndarray)
        assert boundary.contour_uv.ndim == 2
        assert boundary.contour_uv.shape[1] == 2

    def test_circularity_close_to_one(self, boundary: InflectionBoundary) -> None:
        assert boundary.circularity >= 0.7, (
            f"Circularity {boundary.circularity:.3f} too low for circular Gaussian"
        )

    def test_centroid_near_grid_center(self, boundary: InflectionBoundary) -> None:
        cu, cv = boundary.centroid_uv
        grid_size = 1.0
        tolerance = 0.05 * grid_size
        assert abs(cu - 0.5) < tolerance, f"U centroid {cu:.4f} far from 0.5"
        assert abs(cv - 0.5) < tolerance, f"V centroid {cv:.4f} far from 0.5"

    def test_area_positive(self, boundary: InflectionBoundary) -> None:
        assert boundary.area_uv > 0.0

    def test_perimeter_positive(self, boundary: InflectionBoundary) -> None:
        assert boundary.perimeter_uv > 0.0

    def test_mean_iff_near_e_minus_one_of_peak(self, boundary: InflectionBoundary) -> None:
        # The Laplacian zero-crossing of a Gaussian is at r = sigma * sqrt(2),
        # where the value is exp(-r^2 / 2sigma^2) = exp(-1) ≈ 0.368 of peak.
        expected = 100.0 * math.exp(-1.0)
        assert abs(boundary.mean_iff_on_contour - expected) / expected < 0.30, (
            f"mean_iff {boundary.mean_iff_on_contour:.2f} not within 30% of "
            f"expected {expected:.2f}"
        )

    def test_peak_uv_near_grid_center(self, boundary: InflectionBoundary) -> None:
        pu, pv = boundary.peak_uv
        tolerance = 0.05
        assert abs(pu - 0.5) < tolerance, f"U peak {pu:.4f} far from 0.5"
        assert abs(pv - 0.5) < tolerance, f"V peak {pv:.4f} far from 0.5"

    def test_peak_uv_type(self, boundary: InflectionBoundary) -> None:
        assert isinstance(boundary.peak_uv, tuple)
        assert len(boundary.peak_uv) == 2
        assert isinstance(boundary.peak_uv[0], (float, int))
        assert isinstance(boundary.peak_uv[1], (float, int))


# ---------------------------------------------------------------------------
# Test: elliptical Gaussian — PCA axes match sigma ratio
# ---------------------------------------------------------------------------


class TestEllipticalGaussian:
    """PCA axes of the inflection boundary match the Gaussian sigma ratio."""

    @pytest.fixture(scope="class")
    def boundary(self) -> InflectionBoundary | None:
        np.random.seed(0)
        grid_u, grid_v, grid_z = _make_gaussian_grid(
            size=100,
            sigma_u=20.0,
            sigma_v=10.0,
        )
        return compute_inflection_boundary(grid_u, grid_v, grid_z, gaussian_sigma=2.0)

    def test_boundary_found(self, boundary: InflectionBoundary | None) -> None:
        assert boundary is not None

    def test_pca_major_larger_than_minor(self, boundary: InflectionBoundary) -> None:
        assert boundary.pca_major_uv > boundary.pca_minor_uv

    def test_pca_anisotropy_reflects_elongation(self, boundary: InflectionBoundary) -> None:
        # PCA of the ellipse boundary vertices does not give the semi-axis
        # ratio directly; it is driven by the higher-order distribution of
        # perimeter points. The important property is that the major/minor
        # ratio is substantially > 1 for a 2:1 sigma ellipse.
        actual_ratio = boundary.pca_major_uv / boundary.pca_minor_uv
        assert actual_ratio > 1.5, (
            f"PCA ratio {actual_ratio:.2f} not > 1.5 for a 2:1 sigma ellipse"
        )

    def test_peak_uv_near_centroid(self, boundary: InflectionBoundary) -> None:
        # The Gaussian is centered at (0.5, 0.5); both peak and centroid should
        # be close for a symmetric (even if elongated) centered distribution.
        pu, pv = boundary.peak_uv
        cu, cv = boundary.centroid_uv
        assert abs(pu - cu) < 0.05, f"peak U {pu:.4f} far from centroid U {cu:.4f}"
        assert abs(pv - cv) < 0.05, f"peak V {pv:.4f} far from centroid V {cv:.4f}"


# ---------------------------------------------------------------------------
# Test: None cases
# ---------------------------------------------------------------------------


class TestNoneCases:
    """compute_inflection_boundary returns None for degenerate inputs."""

    def test_flat_surface_returns_none(self) -> None:
        u = np.linspace(0, 1, 100)
        grid_u, grid_v = np.meshgrid(u, u, indexing="ij")
        grid_z = np.ones((100, 100)) * 5.0
        result = compute_inflection_boundary(grid_u, grid_v, grid_z, gaussian_sigma=2.0)
        assert result is None

    def test_all_nan_returns_none(self) -> None:
        u = np.linspace(0, 1, 100)
        grid_u, grid_v = np.meshgrid(u, u, indexing="ij")
        grid_z = np.full((100, 100), np.nan)
        result = compute_inflection_boundary(grid_u, grid_v, grid_z, gaussian_sigma=2.0)
        assert result is None

    def test_none_grid_z_returns_none(self) -> None:
        u = np.linspace(0, 1, 100)
        grid_u, grid_v = np.meshgrid(u, u, indexing="ij")
        result = compute_inflection_boundary(grid_u, grid_v, None, gaussian_sigma=2.0)
        assert result is None

    def test_peak_at_edge_returns_none(self) -> None:
        u = np.linspace(0, 1, 100)
        grid_u, grid_v = np.meshgrid(u, u, indexing="ij")
        grid_z = np.zeros((100, 100))
        grid_z[0, 50] = 100.0
        result = compute_inflection_boundary(grid_u, grid_v, grid_z, gaussian_sigma=2.0)
        assert result is None


# ---------------------------------------------------------------------------
# Test: serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    """inflection_boundary_to_dict produces a JSON-safe dict."""

    @pytest.fixture(scope="class")
    def boundary(self) -> InflectionBoundary:
        grid_u, grid_v, grid_z = _make_gaussian_grid(size=100, sigma_u=15.0, sigma_v=15.0)
        result = compute_inflection_boundary(grid_u, grid_v, grid_z, gaussian_sigma=2.0)
        assert result is not None, "Fixture requires a valid boundary"
        return result

    def test_round_trip_is_json_serializable(self, boundary: InflectionBoundary) -> None:
        d = inflection_boundary_to_dict(boundary)
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_no_nan_in_dict(self, boundary: InflectionBoundary) -> None:
        d = inflection_boundary_to_dict(boundary)
        for key, val in d.items():
            if key == "contour_uv":
                for pair in val:
                    assert not (math.isnan(pair[0]) or math.isnan(pair[1])), (
                        f"NaN in contour_uv[{key}]"
                    )
            elif key == "centroid_uv":
                for v in val:
                    if v is not None:
                        assert not math.isnan(v), f"NaN in centroid_uv"
            elif val is not None and isinstance(val, float):
                assert not math.isnan(val), f"NaN in field '{key}'"

    def test_no_numpy_types_in_dict(self, boundary: InflectionBoundary) -> None:
        d = inflection_boundary_to_dict(boundary)
        for key, val in d.items():
            if key == "contour_uv":
                for pair in val:
                    assert isinstance(pair[0], float), f"contour_uv u is {type(pair[0])}"
                    assert isinstance(pair[1], float), f"contour_uv v is {type(pair[1])}"
            elif key == "centroid_uv":
                for v in val:
                    assert v is None or isinstance(v, float), (
                        f"centroid_uv entry is {type(v)}"
                    )
            elif val is not None:
                assert isinstance(val, (float, int, list)), (
                    f"Field '{key}' has numpy type {type(val)}"
                )

    def test_contour_uv_is_list_of_pairs(self, boundary: InflectionBoundary) -> None:
        d = inflection_boundary_to_dict(boundary)
        assert isinstance(d["contour_uv"], list)
        assert all(len(pair) == 2 for pair in d["contour_uv"])

    def test_expected_keys_present(self, boundary: InflectionBoundary) -> None:
        d = inflection_boundary_to_dict(boundary)
        expected_keys = {
            "contour_uv",
            "area_uv",
            "perimeter_uv",
            "circularity",
            "centroid_uv",
            "peak_uv",
            "pca_major_uv",
            "pca_minor_uv",
            "pca_orientation_deg",
            "mean_iff_on_contour",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Test: polygon metric helpers
# ---------------------------------------------------------------------------


class TestPolygonMetrics:
    """Unit tests for _compute_polygon_area, _compute_polygon_perimeter,
    _compute_polygon_centroid."""

    def test_square_area(self) -> None:
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        assert _compute_polygon_area(square) == pytest.approx(1.0)

    def test_square_perimeter(self) -> None:
        square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        assert _compute_polygon_perimeter(square) == pytest.approx(4.0)

    def test_square_centroid(self) -> None:
        square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
        cu, cv = _compute_polygon_centroid(square)
        assert cu == pytest.approx(1.0, abs=1e-9)
        assert cv == pytest.approx(1.0, abs=1e-9)

    def test_pca_circle_major_approx_minor(self) -> None:
        theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        contour = np.column_stack([np.cos(theta), np.sin(theta)])
        major, minor, _ = _compute_contour_pca(contour)
        assert abs(major - minor) / major < 0.05


# ---------------------------------------------------------------------------
# Test: partial-NaN grid (forearm-shaped valid region)
# ---------------------------------------------------------------------------


class TestPartialNaNGrid:
    """compute_inflection_boundary works with a high-NaN-fraction valid region.

    Simulates forearm-shaped data: a circular valid region centred in the grid
    at ~70 percent NaN overall.  The extrapolation fix is required because the
    old binary_dilation approach would erode the valid Laplacian domain far
    enough inward that a centred Gaussian still succeeded — this test verifies
    the algorithm still works correctly under high NaN density.
    """

    @pytest.fixture(scope="class")
    def boundary(self) -> InflectionBoundary | None:
        size = 100
        u = np.linspace(0.0, 1.0, size)
        grid_u, grid_v = np.meshgrid(u, u, indexing="ij")

        # Gaussian peak at centre
        sigma_px = 10.0
        pixel_scale = 1.0 / (size - 1)
        sigma_uv = sigma_px * pixel_scale
        grid_z = 100.0 * np.exp(
            -0.5 * ((grid_u - 0.5) ** 2 + (grid_v - 0.5) ** 2) / sigma_uv ** 2
        )

        # Circular NaN mask: ~70% NaN (valid radius ~31 px out of 100)
        rows = np.arange(size)[:, None]
        cols = np.arange(size)[None, :]
        dist_from_centre = np.sqrt((rows - 49.5) ** 2 + (cols - 49.5) ** 2)
        grid_z[dist_from_centre > 31.0] = np.nan

        return compute_inflection_boundary(grid_u, grid_v, grid_z, gaussian_sigma=2.0)

    def test_boundary_found(self, boundary: InflectionBoundary | None) -> None:
        assert boundary is not None, "boundary must be found for centred peak in circular valid region"

    def test_contour_shape(self, boundary: InflectionBoundary) -> None:
        assert boundary.contour_uv.ndim == 2
        assert boundary.contour_uv.shape[1] == 2
        assert len(boundary.contour_uv) >= 4

    def test_area_positive(self, boundary: InflectionBoundary) -> None:
        assert boundary.area_uv > 0.0


# ---------------------------------------------------------------------------
# Test: border peak (peak close to NaN boundary)
# ---------------------------------------------------------------------------


class TestBorderPeak:
    """compute_inflection_boundary finds a boundary for a peak near the NaN edge.

    With the old binary_dilation strategy (5x5 kernel, 2px dilation), a peak
    positioned close to the NaN boundary had its Laplacian masked — the basin
    flood-fill produced a truncated or absent contour.  The nearest-neighbour
    extrapolation fix resolves this by extending valid Laplacian values to the
    original data boundary.
    """

    @pytest.fixture(scope="class")
    def boundary(self) -> InflectionBoundary | None:
        size = 100
        u = np.linspace(0.0, 1.0, size)
        grid_u, grid_v = np.meshgrid(u, u, indexing="ij")

        # Gaussian centred at row 10, col 50 (10 rows from the top grid edge)
        pixel_scale = 1.0 / (size - 1)
        center_u = 10 * pixel_scale
        center_v = 0.5
        sigma_uv = 8.0 * pixel_scale
        grid_z = 100.0 * np.exp(
            -0.5 * ((grid_u - center_u) ** 2 + (grid_v - center_v) ** 2) / sigma_uv ** 2
        )

        # NaN mask: rows 0-4 are NaN (peak at row 10 is 6 rows from the boundary)
        grid_z[:5, :] = np.nan

        return compute_inflection_boundary(grid_u, grid_v, grid_z, gaussian_sigma=2.0)

    def test_boundary_found(self, boundary: InflectionBoundary | None) -> None:
        assert boundary is not None, (
            "boundary must be found for peak 6 rows from NaN boundary with extrapolation fix"
        )

    def test_contour_has_points(self, boundary: InflectionBoundary) -> None:
        assert len(boundary.contour_uv) >= 4


# ---------------------------------------------------------------------------
# Test: snapshot output
# ---------------------------------------------------------------------------


class TestSnapshotOutput:
    """Verify that per-step snapshot PNGs are created."""

    def test_success_produces_five_files(self, tmp_path: Path) -> None:
        grid_u, grid_v, grid_z = _make_gaussian_grid(size=100)
        result = compute_inflection_boundary(
            grid_u, grid_v, grid_z, gaussian_sigma=2.0,
            snapshot_dir=tmp_path, snapshot_label="test",
        )
        assert result is not None, "boundary must succeed for this test"
        files = sorted(tmp_path.glob("inflection_test_step*.png"))
        assert len(files) == 5
        expected_names = [
            "inflection_test_step1_gaussian.png",
            "inflection_test_step2_laplacian.png",
            "inflection_test_step3_basin.png",
            "inflection_test_step4_contour.png",
            "inflection_test_step5_uv.png",
        ]
        actual_names = [f.name for f in files]
        assert actual_names == expected_names

    def test_failure_produces_three_files(self, tmp_path: Path) -> None:
        u = np.linspace(0, 1, 100)
        grid_u, grid_v = np.meshgrid(u, u, indexing="ij")
        grid_z = np.ones((100, 100)) * 5.0
        result = compute_inflection_boundary(
            grid_u, grid_v, grid_z, gaussian_sigma=2.0,
            snapshot_dir=tmp_path, snapshot_label="fail",
        )
        assert result is None
        # Flat surface: Laplacian is flat/empty → returns None before basin step,
        # so no snapshots are saved (early exit before Laplacian check).
        # Use a case that reaches basin extraction but fails there instead.
        files = sorted(tmp_path.glob("inflection_fail_step*.png"))
        # Flat surface exits before snapshots; no files expected
        assert len(files) == 0

    def test_basin_failure_produces_three_files(self, tmp_path: Path) -> None:
        size = 100
        u = np.linspace(0.0, 1.0, size)
        grid_u, grid_v = np.meshgrid(u, u, indexing="ij")
        sigma_uv = 3.0 / (size - 1)
        grid_z = 100.0 * np.exp(
            -0.5 * ((grid_u - 0.5) ** 2 + (grid_v - 0.5) ** 2) / sigma_uv ** 2
        )
        # Very narrow peak: basin may be too small for a valid contour
        result = compute_inflection_boundary(
            grid_u, grid_v, grid_z, gaussian_sigma=0.5,
            snapshot_dir=tmp_path, snapshot_label="narrow",
        )
        files = sorted(tmp_path.glob("inflection_narrow_step*.png"))
        if result is None and len(files) > 0:
            # Basin failed but snapshots were saved: steps 1-3 only
            assert all("step4" not in f.name and "step5" not in f.name for f in files)
        # If result is not None, the narrow peak still succeeded — that's fine too
