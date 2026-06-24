"""Unit tests for rf_tap_stroke_comparison_pipeline.

Tests use synthetic NPZ data mimicking the ``spatial_extract_boundaries``
output format.  All rendering functions are mocked to avoid matplotlib
overhead — only the CSV summary output and metric computations are verified.
"""

from __future__ import annotations

import logging
import math
import sys
import types
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
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
_stub("utils.pipeline.monitoring")
_stub("analysis")
_stub("analysis.pipeline")
_stub("analysis.receptive_field_mapping")

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from analysis.pipeline.shared_constants import session_id_from_path  # noqa: E402
from analysis.pipeline.output_dirs import SPATIAL_EXTRACT_BOUNDARIES  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers: synthetic mesh + NPZ construction
# ---------------------------------------------------------------------------


def _make_simple_mesh(
    n_u: int = 5,
    n_v: int = 4,
    mm_width: float = 100.0,
    mm_height: float = 80.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (forearm_uv, forearm_V, forearm_faces) for a flat grid mesh.

    UV spans [0, 1] x [0, 1], 3D spans [0, mm_width] x [0, mm_height] x 0.
    """
    u_vals = np.linspace(0, 1, n_u)
    v_vals = np.linspace(0, 1, n_v)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")
    forearm_uv = np.column_stack([uu.ravel(), vv.ravel()])

    xx = uu * mm_width
    yy = vv * mm_height
    zz = np.zeros_like(uu)
    forearm_V = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    faces = []
    for i in range(n_u - 1):
        for j in range(n_v - 1):
            idx = i * n_v + j
            faces.append([idx, idx + 1, idx + n_v])
            faces.append([idx + 1, idx + n_v + 1, idx + n_v])
    forearm_faces = np.array(faces, dtype=np.int32)

    return forearm_uv, forearm_V, forearm_faces


def _make_circular_contour(
    center_u: float,
    center_v: float,
    radius: float,
    n_points: int = 64,
) -> np.ndarray:
    """Return an (n_points+1, 2) closed contour polygon (first == last)."""
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = np.column_stack([
        center_u + radius * np.cos(theta),
        center_v + radius * np.sin(theta),
    ])
    return np.vstack([pts, pts[:1]])


def _make_gaussian_grid(
    size: int = 150,
    center_u: float = 0.5,
    center_v: float = 0.5,
    sigma: float = 0.1,
    amplitude: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (grid_u, grid_v, grid_z) for a 2D Gaussian heatmap."""
    u = np.linspace(0, 1, size)
    v = np.linspace(0, 1, size)
    grid_u, grid_v = np.meshgrid(u, v, indexing="ij")
    grid_z = amplitude * np.exp(
        -0.5 * ((grid_u - center_u) ** 2 + (grid_v - center_v) ** 2) / sigma ** 2
    )
    return grid_u, grid_v, grid_z


def _build_synthetic_npz(
    tmp_dir: Path,
    session_id: str,
    iff_metric: str = "mean",
    forearm_uv: np.ndarray | None = None,
    forearm_V: np.ndarray | None = None,
    forearm_faces: np.ndarray | None = None,
    tap_center: tuple[float, float] = (0.4, 0.5),
    stroke_center: tuple[float, float] = (0.6, 0.5),
    all_center: tuple[float, float] = (0.5, 0.5),
    tap_amplitude: float = 60.0,
    stroke_amplitude: float = 40.0,
    all_amplitude: float = 50.0,
    tap_area_mm2: float = 200.0,
    stroke_area_mm2: float = 150.0,
    all_area_mm2: float = 250.0,
    tap_perimeter_mm: float = 50.0,
    stroke_perimeter_mm: float = 45.0,
    tap_circularity: float = 0.9,
    stroke_circularity: float = 0.85,
    tap_contour_radius: float = 0.1,
    stroke_contour_radius: float = 0.08,
    all_contour_radius: float = 0.12,
    n_touches_tap: int = 30,
    n_touches_stroke: int = 40,
    n_touches_all: int = 70,
    mean_iff_tap: float = 10.0,
    mean_iff_stroke: float = 8.0,
    mean_iff_all: float = 9.0,
    iff_at_centroid_tap: float = 25.0,
    iff_at_centroid_stroke: float = 20.0,
    iff_at_centroid_all: float = 22.0,
    pca_major_tap: float = 0.12,
    pca_minor_tap: float = 0.10,
    pca_major_stroke: float = 0.10,
    pca_minor_stroke: float = 0.08,
    pca_orientation_tap: float = 30.0,
    pca_orientation_stroke: float = 45.0,
    exclude_contour_gtype: str | None = None,
    exclude_centroid_gtype: str | None = None,
) -> Path:
    """Build and save a synthetic NPZ file mimicking spatial_extract_boundaries output."""
    if forearm_uv is None:
        forearm_uv, forearm_V, forearm_faces = _make_simple_mesh()

    gtypes = ['all', 'tap', 'stroke']
    centers = {'all': all_center, 'tap': tap_center, 'stroke': stroke_center}
    amplitudes = {'all': all_amplitude, 'tap': tap_amplitude, 'stroke': stroke_amplitude}
    contour_radii = {'all': all_contour_radius, 'tap': tap_contour_radius, 'stroke': stroke_contour_radius}
    areas = {'all': all_area_mm2, 'tap': tap_area_mm2, 'stroke': stroke_area_mm2}
    perimeters = {'all': 55.0, 'tap': tap_perimeter_mm, 'stroke': stroke_perimeter_mm}
    circularities = {'all': 0.92, 'tap': tap_circularity, 'stroke': stroke_circularity}
    n_touches = {'all': n_touches_all, 'tap': n_touches_tap, 'stroke': n_touches_stroke}
    mean_iffs = {'all': mean_iff_all, 'tap': mean_iff_tap, 'stroke': mean_iff_stroke}
    iff_at_centroids = {'all': iff_at_centroid_all, 'tap': iff_at_centroid_tap, 'stroke': iff_at_centroid_stroke}
    pca_majors = {'all': 0.14, 'tap': pca_major_tap, 'stroke': pca_major_stroke}
    pca_minors = {'all': 0.12, 'tap': pca_minor_tap, 'stroke': pca_minor_stroke}
    pca_orientations = {'all': 35.0, 'tap': pca_orientation_tap, 'stroke': pca_orientation_stroke}

    data: dict = {
        'forearm_uv': forearm_uv,
        'forearm_V': forearm_V,
        'forearm_faces': forearm_faces,
        'gesture_types': np.array(gtypes, dtype=object),
    }

    for gtype in gtypes:
        cu, cv = centers[gtype]

        grid_u, grid_v, grid_z = _make_gaussian_grid(
            center_u=cu, center_v=cv, amplitude=amplitudes[gtype],
        )
        data[f'grid_u_{gtype}'] = grid_u
        data[f'grid_v_{gtype}'] = grid_v
        data[f'grid_z_{gtype}'] = grid_z

        if exclude_centroid_gtype != gtype:
            data[f'boundary_centroid_uv_{gtype}'] = np.array([cu, cv])
            data[f'boundary_peak_uv_{gtype}'] = np.array([cu, cv])

        if exclude_contour_gtype != gtype:
            contour = _make_circular_contour(cu, cv, contour_radii[gtype])
            data[f'boundary_contour_uv_{gtype}'] = contour

        data[f'boundary_area_xyz_mm2_{gtype}'] = np.float64(areas[gtype])
        data[f'boundary_perimeter_xyz_mm_{gtype}'] = np.float64(perimeters[gtype])
        data[f'boundary_circularity_{gtype}'] = np.float64(circularities[gtype])
        data[f'boundary_pca_major_uv_{gtype}'] = np.float64(pca_majors[gtype])
        data[f'boundary_pca_minor_uv_{gtype}'] = np.float64(pca_minors[gtype])
        data[f'boundary_pca_orientation_deg_{gtype}'] = np.float64(pca_orientations[gtype])
        data[f'boundary_mean_iff_on_contour_{gtype}'] = np.float64(mean_iffs[gtype])
        data[f'boundary_iff_at_centroid_{gtype}'] = np.float64(iff_at_centroids[gtype])
        data[f'n_touches_{gtype}'] = np.int64(n_touches[gtype])
        data[f'threshold_{gtype}'] = np.float64(0.5)

    npz_dir = (
        tmp_dir / '4_analysed' / SPATIAL_EXTRACT_BOUNDARIES
        / f'iff_{iff_metric}' / session_id
    )
    npz_dir.mkdir(parents=True, exist_ok=True)
    npz_path = npz_dir / f'{session_id}_population_response_fields.npz'
    np.savez(npz_path, **data)
    return npz_path


def _make_session_config(
    tmp_path: Path,
    session_id: str = "ST14-01",
    iff_metric: str = "mean",
    **npz_kwargs,
) -> tuple[Path, Path, Path]:
    """Create a session config tuple and synthetic NPZ.

    Returns (csv_path, db_path, npz_path).
    """
    csv_path = tmp_path / f'{session_id}_semicontrolled_data.csv'
    csv_path.touch()
    db_path = tmp_path
    npz_path = _build_synthetic_npz(
        db_path, session_id, iff_metric=iff_metric, **npz_kwargs,
    )
    return csv_path, db_path, npz_path


# ---------------------------------------------------------------------------
# Mock targets — all rendering callables used by the pipeline
# ---------------------------------------------------------------------------

_PIPELINE_MOD = "analysis.receptive_field_mapping.pipelines.rf_tap_stroke_comparison_pipeline"
_RENDER_TARGETS = [
    f"{_PIPELINE_MOD}.render_tap_stroke_contour_overlay",
    f"{_PIPELINE_MOD}.render_tap_stroke_heatmap_triptych",
    f"{_PIPELINE_MOD}.render_tap_stroke_aggregate",
    f"{_PIPELINE_MOD}.render_tap_stroke_hotspot_aggregate",
    f"{_PIPELINE_MOD}.render_tap_stroke_metric_deltas",
    f"{_PIPELINE_MOD}.render_tap_stroke_population_strips",
    f"{_PIPELINE_MOD}.render_center_marked_heatmap",
    f"{_PIPELINE_MOD}.render_shift_decomposition",
    f"{_PIPELINE_MOD}.render_population_rf_circular_crop",
]


def _noop(*args, **kwargs):
    """No-op replacement for rendering functions."""
    pass


def _mock_renderers():
    """Return a context manager that patches all renderer callables with no-ops."""
    stack = ExitStack()
    for target in _RENDER_TARGETS:
        stack.enter_context(patch(target, side_effect=_noop))
    return stack


def _run_pipeline(session_configs, output_dir, iff_metric="mean", **kwargs):
    """Import and run the pipeline inside a rendering-mock context."""
    from analysis.receptive_field_mapping.pipelines.rf_tap_stroke_comparison_pipeline import (
        run_tap_stroke_comparison,
    )
    with _mock_renderers():
        run_tap_stroke_comparison(
            session_configs=session_configs,
            output_dir=output_dir,
            force_processing=True,
            iff_metric=iff_metric,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSyntheticNPZConstruction:
    """Verify that the synthetic NPZ builder produces valid data."""

    def test_npz_has_expected_keys(self, tmp_path: Path) -> None:
        csv_path, db_path, npz_path = _make_session_config(tmp_path)
        npz = np.load(npz_path, allow_pickle=True)
        for gtype in ('all', 'tap', 'stroke'):
            assert f'grid_u_{gtype}' in npz
            assert f'grid_v_{gtype}' in npz
            assert f'grid_z_{gtype}' in npz
            assert f'boundary_centroid_uv_{gtype}' in npz
            assert f'boundary_peak_uv_{gtype}' in npz
            assert f'boundary_contour_uv_{gtype}' in npz
            assert f'boundary_area_xyz_mm2_{gtype}' in npz
            assert f'n_touches_{gtype}' in npz
            assert f'threshold_{gtype}' in npz

    def test_session_id_extracted(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "ST14-01_semicontrolled_data.csv"
        csv_path.touch()
        assert session_id_from_path(csv_path) == "ST14-01"

    def test_mesh_shape(self) -> None:
        uv, V, faces = _make_simple_mesh(n_u=5, n_v=4)
        assert uv.shape == (20, 2)
        assert V.shape == (20, 3)
        assert faces.shape[1] == 3


class TestMetricComputation:
    """Task 6.2: verify peak_iff, equivalent_diameter_mm, rf_sharpness,
    containment are computed correctly from known inputs."""

    def test_peak_iff(self, tmp_path: Path) -> None:
        """peak_iff should be the maximum positive finite value in grid_z."""
        tap_amp = 60.0
        stroke_amp = 40.0
        csv_path, db_path, _ = _make_session_config(
            tmp_path, session_id="ST14-01",
            tap_amplitude=tap_amp, stroke_amplitude=stroke_amp,
        )
        output_dir = tmp_path / "output"
        _run_pipeline([(csv_path, db_path)], output_dir)

        df = pd.read_csv(output_dir / 'rf_tap_stroke_comparison_summary.csv')
        assert len(df) == 1
        assert df['peak_iff_tap'].iloc[0] == pytest.approx(tap_amp, rel=1e-3)
        assert df['peak_iff_stroke'].iloc[0] == pytest.approx(stroke_amp, rel=1e-3)

    def test_equivalent_diameter_mm(self, tmp_path: Path) -> None:
        """equivalent_diameter_mm = sqrt(4 * area_mm2 / pi)."""
        tap_area = 200.0
        stroke_area = 150.0
        csv_path, db_path, _ = _make_session_config(
            tmp_path, session_id="ST14-01",
            tap_area_mm2=tap_area, stroke_area_mm2=stroke_area,
        )
        output_dir = tmp_path / "output"
        _run_pipeline([(csv_path, db_path)], output_dir)

        df = pd.read_csv(output_dir / 'rf_tap_stroke_comparison_summary.csv')
        expected_tap = math.sqrt(4 * tap_area / math.pi)
        expected_stroke = math.sqrt(4 * stroke_area / math.pi)
        assert df['equivalent_diameter_mm_tap'].iloc[0] == pytest.approx(expected_tap, rel=1e-6)
        assert df['equivalent_diameter_mm_stroke'].iloc[0] == pytest.approx(expected_stroke, rel=1e-6)

    def test_rf_sharpness(self, tmp_path: Path) -> None:
        """rf_sharpness = peak_iff / mean_iff_on_contour."""
        tap_amp = 80.0
        mean_iff_tap = 20.0
        csv_path, db_path, _ = _make_session_config(
            tmp_path, session_id="ST14-01",
            tap_amplitude=tap_amp,
            mean_iff_tap=mean_iff_tap,
        )
        output_dir = tmp_path / "output"
        _run_pipeline([(csv_path, db_path)], output_dir)

        df = pd.read_csv(output_dir / 'rf_tap_stroke_comparison_summary.csv')
        expected_sharpness = tap_amp / mean_iff_tap
        assert df['rf_sharpness_tap'].iloc[0] == pytest.approx(expected_sharpness, rel=1e-3)

    def test_containment_overlapping_circles(self, tmp_path: Path) -> None:
        """For two overlapping circular contours, containment is between 0 and 1."""
        csv_path, db_path, _ = _make_session_config(
            tmp_path, session_id="ST14-01",
            tap_center=(0.5, 0.5),
            stroke_center=(0.55, 0.5),
            tap_contour_radius=0.15,
            stroke_contour_radius=0.15,
        )
        output_dir = tmp_path / "output"
        _run_pipeline([(csv_path, db_path)], output_dir)

        df = pd.read_csv(output_dir / 'rf_tap_stroke_comparison_summary.csv')
        c_tap_in_stroke = df['containment_tap_in_stroke'].iloc[0]
        c_stroke_in_tap = df['containment_stroke_in_tap'].iloc[0]
        assert 0.0 < c_tap_in_stroke <= 1.0
        assert 0.0 < c_stroke_in_tap <= 1.0

    def test_containment_nested_circles(self, tmp_path: Path) -> None:
        """If tap circle is fully inside stroke circle, containment_tap_in_stroke ~ 1.0."""
        csv_path, db_path, _ = _make_session_config(
            tmp_path, session_id="ST14-01",
            tap_center=(0.5, 0.5),
            stroke_center=(0.5, 0.5),
            tap_contour_radius=0.05,
            stroke_contour_radius=0.20,
        )
        output_dir = tmp_path / "output"
        _run_pipeline([(csv_path, db_path)], output_dir)

        df = pd.read_csv(output_dir / 'rf_tap_stroke_comparison_summary.csv')
        assert df['containment_tap_in_stroke'].iloc[0] == pytest.approx(1.0, abs=0.05)
        assert df['containment_stroke_in_tap'].iloc[0] < 0.5


class TestDeltaSignConvention:
    """Task 6.3: verify delta = tap - stroke sign convention."""

    def test_delta_area_positive_when_tap_larger(self, tmp_path: Path) -> None:
        """When tap area > stroke area, delta_area_mm2 > 0."""
        csv_path, db_path, _ = _make_session_config(
            tmp_path, session_id="ST14-01",
            tap_area_mm2=300.0,
            stroke_area_mm2=100.0,
        )
        output_dir = tmp_path / "output"
        _run_pipeline([(csv_path, db_path)], output_dir)

        df = pd.read_csv(output_dir / 'rf_tap_stroke_comparison_summary.csv')
        assert df['delta_area_mm2'].iloc[0] == pytest.approx(200.0)

    def test_delta_area_negative_when_tap_smaller(self, tmp_path: Path) -> None:
        """When tap area < stroke area, delta_area_mm2 < 0."""
        csv_path, db_path, _ = _make_session_config(
            tmp_path, session_id="ST14-01",
            tap_area_mm2=80.0,
            stroke_area_mm2=200.0,
        )
        output_dir = tmp_path / "output"
        _run_pipeline([(csv_path, db_path)], output_dir)

        df = pd.read_csv(output_dir / 'rf_tap_stroke_comparison_summary.csv')
        assert df['delta_area_mm2'].iloc[0] == pytest.approx(-120.0)

    def test_delta_perimeter_sign(self, tmp_path: Path) -> None:
        """delta_perimeter_mm = tap_perimeter - stroke_perimeter."""
        csv_path, db_path, _ = _make_session_config(
            tmp_path, session_id="ST14-01",
            tap_perimeter_mm=60.0,
            stroke_perimeter_mm=30.0,
        )
        output_dir = tmp_path / "output"
        _run_pipeline([(csv_path, db_path)], output_dir)

        df = pd.read_csv(output_dir / 'rf_tap_stroke_comparison_summary.csv')
        assert df['delta_perimeter_mm'].iloc[0] == pytest.approx(30.0)

    def test_delta_peak_iff_sign(self, tmp_path: Path) -> None:
        """delta_peak_iff = tap peak - stroke peak."""
        csv_path, db_path, _ = _make_session_config(
            tmp_path, session_id="ST14-01",
            tap_amplitude=100.0,
            stroke_amplitude=50.0,
        )
        output_dir = tmp_path / "output"
        _run_pipeline([(csv_path, db_path)], output_dir)

        df = pd.read_csv(output_dir / 'rf_tap_stroke_comparison_summary.csv')
        assert df['delta_peak_iff'].iloc[0] == pytest.approx(50.0, rel=1e-3)

    def test_all_deltas_are_tap_minus_stroke(self, tmp_path: Path) -> None:
        """All delta columns follow sign convention: tap - stroke."""
        csv_path, db_path, _ = _make_session_config(
            tmp_path, session_id="ST14-01",
            tap_area_mm2=200.0,
            stroke_area_mm2=150.0,
            tap_perimeter_mm=50.0,
            stroke_perimeter_mm=45.0,
            tap_circularity=0.9,
            stroke_circularity=0.85,
            mean_iff_tap=10.0,
            mean_iff_stroke=8.0,
            iff_at_centroid_tap=25.0,
            iff_at_centroid_stroke=20.0,
        )
        output_dir = tmp_path / "output"
        _run_pipeline([(csv_path, db_path)], output_dir)

        df = pd.read_csv(output_dir / 'rf_tap_stroke_comparison_summary.csv')
        row = df.iloc[0]

        assert row['delta_area_mm2'] == pytest.approx(200.0 - 150.0)
        assert row['delta_perimeter_mm'] == pytest.approx(50.0 - 45.0)
        assert row['delta_circularity'] == pytest.approx(0.9 - 0.85)
        assert row['delta_mean_iff_on_contour'] == pytest.approx(10.0 - 8.0)
        assert row['delta_iff_at_centroid'] == pytest.approx(25.0 - 20.0)


class TestEdgeCases:
    """Task 6.4: test sessions with missing tap or stroke boundary."""

    def test_missing_tap_contour_produces_nan_metrics(
        self, tmp_path: Path, caplog,
    ) -> None:
        """When boundary_contour_uv_tap is absent, metrics are NaN, no crash."""
        csv_path, db_path, _ = _make_session_config(
            tmp_path, session_id="ST14-01",
            exclude_contour_gtype="tap",
        )
        output_dir = tmp_path / "output"
        with caplog.at_level(logging.WARNING):
            _run_pipeline([(csv_path, db_path)], output_dir)

        df = pd.read_csv(output_dir / 'rf_tap_stroke_comparison_summary.csv')
        assert len(df) == 1
        assert pd.isna(df['area_mm2_tap'].iloc[0])
        assert pd.isna(df['delta_area_mm2'].iloc[0])
        assert pd.isna(df['contour_overlap_iou'].iloc[0])
        assert any("no boundary" in r.message.lower() or "missing" in r.message.lower()
                    for r in caplog.records)

    def test_missing_stroke_contour_produces_nan_metrics(
        self, tmp_path: Path, caplog,
    ) -> None:
        """When boundary_contour_uv_stroke is absent, metrics are NaN, no crash."""
        csv_path, db_path, _ = _make_session_config(
            tmp_path, session_id="ST14-01",
            exclude_contour_gtype="stroke",
        )
        output_dir = tmp_path / "output"
        with caplog.at_level(logging.WARNING):
            _run_pipeline([(csv_path, db_path)], output_dir)

        df = pd.read_csv(output_dir / 'rf_tap_stroke_comparison_summary.csv')
        assert len(df) == 1
        assert pd.isna(df['area_mm2_stroke'].iloc[0])
        assert pd.isna(df['contour_overlap_iou'].iloc[0])

    def test_missing_centroid_skips_session(
        self, tmp_path: Path, caplog,
    ) -> None:
        """When a required centroid is missing, the session is skipped entirely."""
        csv_path, db_path, _ = _make_session_config(
            tmp_path, session_id="ST14-01",
            exclude_centroid_gtype="tap",
        )
        output_dir = tmp_path / "output"
        with caplog.at_level(logging.WARNING):
            _run_pipeline([(csv_path, db_path)], output_dir)

        # Session was skipped — pipeline returns early; no CSV written
        csv_path_out = output_dir / 'rf_tap_stroke_comparison_summary.csv'
        assert not csv_path_out.exists()
        assert any("missing" in r.message.lower() for r in caplog.records)

    def test_empty_session_configs_raises(self, tmp_path: Path) -> None:
        """Passing empty session_configs should raise ValueError."""
        from analysis.receptive_field_mapping.pipelines.rf_tap_stroke_comparison_pipeline import (
            run_tap_stroke_comparison,
        )
        output_dir = tmp_path / "output"
        with pytest.raises(ValueError, match="session_configs is empty"):
            run_tap_stroke_comparison(
                session_configs=[],
                output_dir=output_dir,
                force_processing=True,
                iff_metric="mean",
            )

    def test_invalid_iff_metric_raises(self, tmp_path: Path) -> None:
        """Passing invalid iff_metric should raise ValueError."""
        from analysis.receptive_field_mapping.pipelines.rf_tap_stroke_comparison_pipeline import (
            run_tap_stroke_comparison,
        )
        csv_path = tmp_path / "ST14-01_semicontrolled_data.csv"
        csv_path.touch()
        output_dir = tmp_path / "output"
        with pytest.raises(ValueError, match="Invalid iff_metric"):
            run_tap_stroke_comparison(
                session_configs=[(csv_path, tmp_path)],
                output_dir=output_dir,
                force_processing=True,
                iff_metric="invalid",
            )


class TestPipelineCSVOutput:
    """Verify the summary CSV has the expected structure and column set."""

    def test_csv_has_all_expected_columns(self, tmp_path: Path) -> None:
        csv_path, db_path, _ = _make_session_config(tmp_path, session_id="ST14-01")
        output_dir = tmp_path / "output"
        _run_pipeline([(csv_path, db_path)], output_dir)

        df = pd.read_csv(output_dir / 'rf_tap_stroke_comparison_summary.csv')

        assert 'session_id' in df.columns
        assert 'uv_to_mm_scale' in df.columns

        for suffix in ('_tap', '_stroke'):
            assert f'area_mm2{suffix}' in df.columns
            assert f'perimeter_mm{suffix}' in df.columns
            assert f'circularity{suffix}' in df.columns
            assert f'peak_iff{suffix}' in df.columns
            assert f'equivalent_diameter_mm{suffix}' in df.columns
            assert f'rf_sharpness{suffix}' in df.columns
            assert f'iff_at_centroid{suffix}' in df.columns
            assert f'n_touches{suffix}' in df.columns

        assert 'delta_area_mm2' in df.columns
        assert 'delta_perimeter_mm' in df.columns
        assert 'delta_peak_iff' in df.columns
        assert 'delta_equivalent_diameter_mm' in df.columns
        assert 'delta_rf_sharpness' in df.columns
        assert 'delta_iff_at_centroid' in df.columns

        assert 'contour_overlap_iou' in df.columns
        assert 'contour_overlap_dice' in df.columns
        assert 'heatmap_pearson_r' in df.columns
        assert 'containment_tap_in_stroke' in df.columns
        assert 'containment_stroke_in_tap' in df.columns

        assert 'centroid_shift_along_arm_mm' in df.columns
        assert 'centroid_shift_across_arm_mm' in df.columns
        assert 'peak_shift_along_arm_mm' in df.columns
        assert 'peak_shift_across_arm_mm' in df.columns

    def test_sentinel_written(self, tmp_path: Path) -> None:
        csv_path, db_path, _ = _make_session_config(tmp_path, session_id="ST14-01")
        output_dir = tmp_path / "output"
        _run_pipeline([(csv_path, db_path)], output_dir)

        sentinel = output_dir / 'rf_tap_stroke_comparison_done.json'
        assert sentinel.exists()

    def test_skips_when_sentinel_exists_and_not_forced(self, tmp_path: Path) -> None:
        """Pipeline should skip when sentinel exists and force_processing=False."""
        import json
        from analysis.receptive_field_mapping.pipelines.rf_tap_stroke_comparison_pipeline import (
            run_tap_stroke_comparison,
        )

        csv_path, db_path, _ = _make_session_config(tmp_path, session_id="ST14-01")
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        sentinel = output_dir / 'rf_tap_stroke_comparison_done.json'
        with open(sentinel, 'w') as f:
            json.dump({'done': True, 'n_sessions': 1}, f)

        with _mock_renderers():
            run_tap_stroke_comparison(
                session_configs=[(csv_path, db_path)],
                output_dir=output_dir,
                force_processing=False,
                iff_metric="mean",
            )

        csv_out = output_dir / 'rf_tap_stroke_comparison_summary.csv'
        assert not csv_out.exists()

    def test_multi_session_csv(self, tmp_path: Path) -> None:
        """Pipeline should produce one row per session in the summary CSV."""
        configs = []
        for sid in ("ST14-01", "ST16-05"):
            sub = tmp_path / sid
            sub.mkdir()
            csv_p, db_p, _ = _make_session_config(sub, session_id=sid)
            configs.append((csv_p, db_p))

        output_dir = tmp_path / "output"
        _run_pipeline(configs, output_dir)

        df = pd.read_csv(output_dir / 'rf_tap_stroke_comparison_summary.csv')
        assert len(df) == 2
        assert set(df['session_id'].tolist()) == {"ST14-01", "ST16-05"}


class TestHeatmapCorrelation:
    """Test that heatmap Pearson correlation is computed correctly."""

    def test_identical_heatmaps_correlation_one(self, tmp_path: Path) -> None:
        """When tap and stroke heatmaps are identical, Pearson r ~ 1.0."""
        csv_path, db_path, _ = _make_session_config(
            tmp_path, session_id="ST14-01",
            tap_center=(0.5, 0.5),
            stroke_center=(0.5, 0.5),
            tap_amplitude=50.0,
            stroke_amplitude=50.0,
        )
        output_dir = tmp_path / "output"
        _run_pipeline([(csv_path, db_path)], output_dir)

        df = pd.read_csv(output_dir / 'rf_tap_stroke_comparison_summary.csv')
        assert df['heatmap_pearson_r'].iloc[0] == pytest.approx(1.0, abs=0.01)

    def test_different_heatmaps_correlation_less_than_one(self, tmp_path: Path) -> None:
        """When heatmaps differ, Pearson r < 1.0."""
        csv_path, db_path, _ = _make_session_config(
            tmp_path, session_id="ST14-01",
            tap_center=(0.3, 0.5),
            stroke_center=(0.7, 0.5),
        )
        output_dir = tmp_path / "output"
        _run_pipeline([(csv_path, db_path)], output_dir)

        df = pd.read_csv(output_dir / 'rf_tap_stroke_comparison_summary.csv')
        assert df['heatmap_pearson_r'].iloc[0] < 1.0


class TestOverlapMetrics:
    """Test IoU and Dice overlap between tap and stroke contours."""

    def test_identical_contours_iou_near_one(self, tmp_path: Path) -> None:
        """Identical contour locations and radii should give IoU ~ 1.0."""
        csv_path, db_path, _ = _make_session_config(
            tmp_path, session_id="ST14-01",
            tap_center=(0.5, 0.5),
            stroke_center=(0.5, 0.5),
            tap_contour_radius=0.10,
            stroke_contour_radius=0.10,
        )
        output_dir = tmp_path / "output"
        _run_pipeline([(csv_path, db_path)], output_dir)

        df = pd.read_csv(output_dir / 'rf_tap_stroke_comparison_summary.csv')
        assert df['contour_overlap_iou'].iloc[0] == pytest.approx(1.0, abs=0.02)
        assert df['contour_overlap_dice'].iloc[0] == pytest.approx(1.0, abs=0.02)

    def test_disjoint_contours_iou_zero(self, tmp_path: Path) -> None:
        """Non-overlapping contours should give IoU = 0."""
        csv_path, db_path, _ = _make_session_config(
            tmp_path, session_id="ST14-01",
            tap_center=(0.2, 0.2),
            stroke_center=(0.8, 0.8),
            tap_contour_radius=0.05,
            stroke_contour_radius=0.05,
        )
        output_dir = tmp_path / "output"
        _run_pipeline([(csv_path, db_path)], output_dir)

        df = pd.read_csv(output_dir / 'rf_tap_stroke_comparison_summary.csv')
        assert df['contour_overlap_iou'].iloc[0] == pytest.approx(0.0, abs=0.01)
        assert df['contour_overlap_dice'].iloc[0] == pytest.approx(0.0, abs=0.01)
