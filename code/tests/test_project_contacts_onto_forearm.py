"""Unit tests for the per-frame vertex mapping surfaced by the projection stage.

The projection stage snaps every contact point to the nearest forearm vertex by
XY distance.  The vertex index it computes is the identity a per-point sidecar
must be re-addressed with, so it has to leave the function — recomputing it
against the sidecar's own (float32) coordinates would pick a different vertex
wherever two candidates are near-equidistant, silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add scripts to path for imports
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts" / "_5_postprocessing"
sys.path.insert(0, str(_SCRIPTS_DIR))

pytest.importorskip("open3d", reason="project_contacts_onto_forearm imports Open3D")

from scipy.spatial import KDTree  # noqa: E402

from project_contacts_onto_forearm import (  # noqa: E402
    FRAME_INDEX_COLUMN,
    ProjectionResult,
    _project_single_csv,
)
from preprocessing.forearm_extraction.registration.csv_spatial_transformer import (  # noqa: E402
    parse_contact_points,
    serialize_contact_points,
)

#: A tiny forearm: a 4 x 3 lattice in XY, each column at its own depth.
VERTICES = np.array(
    [[x, y, 0.5 * x] for x in (0.0, 10.0, 20.0, 30.0) for y in (0.0, 5.0, 10.0)],
    dtype=np.float64,
)


def _cell(points) -> str:
    return serialize_contact_points([(float(x), float(y), float(z)) for x, y, z in points])


def _write_csv(path: Path, rows) -> None:
    """Write a minimal session CSV. *rows* is a list of (frame_index, cell)."""
    pd.DataFrame({
        "time": np.arange(len(rows), dtype=np.float64),
        FRAME_INDEX_COLUMN: [frame for frame, _ in rows],
        "contact_location_x": [0.0] * len(rows),
        "contact_location_y": [0.0] * len(rows),
        "contact_location_z": [0.0] * len(rows),
        "contact_points": [cell for _, cell in rows],
    }).to_csv(path, index=False)


def _project(tmp_path: Path, rows) -> tuple[ProjectionResult, pd.DataFrame]:
    src = tmp_path / "in.csv"
    out = tmp_path / "out.csv"
    _write_csv(src, rows)
    result = _project_single_csv(src, out, KDTree(VERTICES[:, :2]), VERTICES)
    return result, pd.read_csv(out)


#: Two contact frames, sitting at row positions 1 and 4 so a positional key
#: would be visibly different from the frame_index key.
_ROWS = [
    (168.0, "[]"),
    (168.0, _cell([(0.2, 0.1, 99.0), (19.8, 4.9, 99.0), (0.4, 9.6, 99.0)])),
    (169.0, "[]"),
    (170.0, "[]"),
    (205.0, _cell([(29.9, 10.2, 99.0)])),
    (206.0, "[]"),
]


class TestProjectionResult:
    """The widened return of _project_single_csv."""

    def test_returns_a_projection_result(self, tmp_path: Path) -> None:
        result, _ = _project(tmp_path, _ROWS)

        assert isinstance(result, ProjectionResult)

    def test_distances_are_still_one_entry_per_contact_point(self, tmp_path: Path) -> None:
        result, _ = _project(tmp_path, _ROWS)

        assert result.distances.shape == (4,)

    def test_no_contact_points_gives_empty_result(self, tmp_path: Path) -> None:
        result, _ = _project(tmp_path, [(168.0, "[]"), (169.0, "[]")])

        assert result.distances.shape == (0,)
        assert result.vertex_indices == {}

    def test_result_is_frozen(self, tmp_path: Path) -> None:
        result, _ = _project(tmp_path, _ROWS)

        with pytest.raises(AttributeError):
            result.distances = np.empty(0)  # type: ignore[misc]


class TestVertexIndexMapping:
    """The per-frame KD-tree indices — the future vertex_id."""

    def test_keyed_by_frame_index_not_row_position(self, tmp_path: Path) -> None:
        result, _ = _project(tmp_path, _ROWS)

        assert set(result.vertex_indices) == {168, 205}

    def test_keys_are_plain_ints(self, tmp_path: Path) -> None:
        result, _ = _project(tmp_path, _ROWS)

        assert all(type(key) is int for key in result.vertex_indices)

    def test_indices_are_intp(self, tmp_path: Path) -> None:
        result, _ = _project(tmp_path, _ROWS)

        assert all(idx.dtype == np.intp for idx in result.vertex_indices.values())

    def test_indices_name_the_expected_vertices(self, tmp_path: Path) -> None:
        """Hand-computed: nearest lattice vertex by XY for each query point."""
        result, _ = _project(tmp_path, _ROWS)

        # VERTICES is ordered x-major: index = 3 * x_step + y_step.
        np.testing.assert_array_equal(result.vertex_indices[168], [0, 7, 2])
        np.testing.assert_array_equal(result.vertex_indices[205], [11])

    def test_one_index_per_contact_point_in_cell_order(self, tmp_path: Path) -> None:
        result, written = _project(tmp_path, _ROWS)

        for frame, indices in result.vertex_indices.items():
            cell = written.loc[written[FRAME_INDEX_COLUMN] == frame, "contact_points"].iloc[-1]
            assert len(parse_contact_points(cell)) == len(indices)

    def test_indices_reproduce_the_written_coordinates(self, tmp_path: Path) -> None:
        """The mapping and the CSV must describe the same vertices."""
        result, written = _project(tmp_path, _ROWS)

        for frame, indices in result.vertex_indices.items():
            cell = written.loc[written[FRAME_INDEX_COLUMN] == frame, "contact_points"].iloc[-1]
            points = np.asarray(parse_contact_points(cell), dtype=np.float64)

            # The cell is written at %.1f, hence the tolerance.
            np.testing.assert_allclose(points, VERTICES[indices], atol=0.05)

    def test_rows_without_contact_points_get_no_mapping(self, tmp_path: Path) -> None:
        result, _ = _project(tmp_path, _ROWS)

        assert 169 not in result.vertex_indices
        assert 206 not in result.vertex_indices

    def test_repeated_vertices_are_kept(self, tmp_path: Path) -> None:
        """Two points may legitimately snap to the same vertex; neither is dropped."""
        result, _ = _project(tmp_path, [(168.0, _cell([(0.1, 0.1, 9.0), (0.2, 0.2, 8.0)]))])

        np.testing.assert_array_equal(result.vertex_indices[168], [0, 0])


class TestFrameIndexValidation:
    """A mis-keyed mapping is worse than no mapping — fail loudly."""

    def test_missing_frame_index_column_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "in.csv"
        pd.DataFrame({
            "contact_points": [_cell([(0.1, 0.1, 9.0)])],
            "contact_location_x": [0.0],
            "contact_location_y": [0.0],
            "contact_location_z": [0.0],
        }).to_csv(src, index=False)

        with pytest.raises(ValueError, match=FRAME_INDEX_COLUMN):
            _project_single_csv(
                src, tmp_path / "out.csv", KDTree(VERTICES[:, :2]), VERTICES
            )

    def test_duplicate_frame_index_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="ambiguous"):
            _project(tmp_path, [
                (168.0, _cell([(0.1, 0.1, 9.0)])),
                (168.0, _cell([(10.1, 0.1, 9.0)])),
            ])

    def test_nan_frame_index_on_a_contact_row_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="whole number"):
            _project(tmp_path, [(np.nan, _cell([(0.1, 0.1, 9.0)]))])

    def test_fractional_frame_index_on_a_contact_row_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="whole number"):
            _project(tmp_path, [(168.5, _cell([(0.1, 0.1, 9.0)]))])

    def test_nan_frame_index_on_an_empty_row_is_tolerated(self, tmp_path: Path) -> None:
        result, _ = _project(tmp_path, [
            (np.nan, "[]"),
            (168.0, _cell([(0.1, 0.1, 9.0)])),
        ])

        assert set(result.vertex_indices) == {168}
