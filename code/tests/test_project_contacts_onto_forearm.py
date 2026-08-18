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


# ---------------------------------------------------------------------------
# The contact depth field sidecar (Phase 6, tasks 6.4 - 6.7)
# ---------------------------------------------------------------------------

from project_contacts_onto_forearm import project_contacts_onto_forearm  # noqa: E402
from postprocessing.forearm_dedup_metadata import (  # noqa: E402
    EPSILON_SOURCE_DAG_CONFIG,
    forearm_dedup_metadata_path,
    write_forearm_dedup_metadata,
)
from preprocessing.motion_analysis.tactile_quantification.io.contact_depth_field_io import (  # noqa: E402
    read_contact_depth_field,
    write_contact_depth_field_table,
)

o3d = pytest.importorskip("open3d", reason="the projection stage needs Open3D")

FIELD_METADATA = {
    "schema_version": "1",
    "coordinate_space": "icp_registered",
    "units": "mm",
    "sign_convention": "negative_is_penetrating",
    "produced_by": "compute_somatosensory_characteristics",
    "source_recording": "unit-test",
    # What the merging filter stamps. Present here so the projection stage's
    # restamp to "postprocessing" is observable rather than a no-op.
    "pipeline_stage": "merging",
}

#: Contact points of two frames, deliberately off-lattice so projection moves them.
CONTACTS = {
    3: [(0.4, 0.3, 7.0), (10.6, 4.8, 8.0)],
    4: [(29.4, 9.7, 1.0)],
}
DEPTHS = {3: [-2.5, -1.25], 4: [-0.75]}


def _write_forearm_ply(tmp_path: Path) -> Path:
    """Materialise the lattice as a deduplicated forearm PLY plus its sidecar."""
    ply = tmp_path / "forearm_deduped" / "S_forearm.ply"
    ply.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(VERTICES)
    assert o3d.io.write_point_cloud(str(ply), pcd)
    write_forearm_dedup_metadata(
        ply,
        source_ply=tmp_path / "S_forearm_source.ply",
        epsilon=0.5,
        epsilon_source=EPSILON_SOURCE_DAG_CONFIG,
        stats={
            "n_original": len(VERTICES) + 3,
            "n_deduped": len(VERTICES),
            "n_removed": 3,
        },
    )
    return ply


def _write_stage_block(tmp_path: Path) -> tuple[Path, Path]:
    """Write one block's CSV and its matching sidecar into ``blocks_deduped/``."""
    block = tmp_path / "blocks_deduped"
    block.mkdir(parents=True, exist_ok=True)
    stem = "S_semicontrolled_block-order-01"
    csv = block / f"{stem}_merged_data.csv"
    pd.DataFrame(
        {
            "frame_index": [float(f) for f in sorted(CONTACTS)],
            "contact_points": [_cell(CONTACTS[f]) for f in sorted(CONTACTS)],
            "contact_depth": [
                max(abs(d) for d in DEPTHS[f]) for f in sorted(CONTACTS)
            ],
            "contact_location_x": [0.0] * len(CONTACTS),
            "contact_location_y": [0.0] * len(CONTACTS),
            "contact_location_z": [0.0] * len(CONTACTS),
        }
    ).to_csv(csv, index=False)

    frames, points, depths = [], [], []
    for frame in sorted(CONTACTS):
        for point, depth in zip(CONTACTS[frame], DEPTHS[frame]):
            frames.append(frame)
            points.append(point)
            depths.append(depth)
    points = np.asarray(points, dtype=np.float32)
    parquet = block / f"{stem}_contact_depth_field.parquet"
    write_contact_depth_field_table(
        pd.DataFrame(
            {
                "frame_index": np.asarray(frames, dtype=np.int32),
                "time_s": np.asarray(frames, dtype=np.float64) / 30.0,
                "x": points[:, 0],
                "y": points[:, 1],
                "z": points[:, 2],
                "signed_depth_mm": np.asarray(depths, dtype=np.float64),
            }
        ),
        parquet,
        metadata=dict(FIELD_METADATA),
    )
    return csv, parquet


def _project_stage(tmp_path: Path, **kwargs):
    ply = kwargs.pop("ply", None) or _write_forearm_ply(tmp_path)
    csv, parquet = _write_stage_block(tmp_path)
    out = tmp_path / "blocks_projected"
    return (
        project_contacts_onto_forearm(
            input_files=[csv],
            forearm_ply_path=ply,
            output_dir=out,
            projection_stats_path=out / "projection_stats.csv",
            **kwargs,
        ),
        ply,
        out,
    )


class TestProjectedDepthField:
    def test_it_returns_both_artifacts(self, tmp_path):
        (csvs, parquets), _, _ = _project_stage(tmp_path, force_processing=True)
        assert len(csvs) == len(parquets) == 1
        assert parquets[0].exists()

    def test_every_row_addresses_the_vertex_its_csv_point_snapped_to(self, tmp_path):
        (csvs, parquets), ply, _ = _project_stage(tmp_path, force_processing=True)
        table, _ = read_contact_depth_field(parquets[0])

        assert table["vertex_id"].dtype == np.int32
        resolved = VERTICES[table["vertex_id"].to_numpy()]
        np.testing.assert_allclose(
            resolved, table[["x", "y", "z"]].to_numpy(), atol=1e-4
        )

        # And the same vertices the CSV chose, in the same order.
        csv_points = []
        for _, row in pd.read_csv(csvs[0]).iterrows():
            csv_points.extend(parse_contact_points(row["contact_points"]))
        np.testing.assert_allclose(
            np.round(resolved, 1), np.asarray(csv_points), atol=1e-9
        )

    def test_depth_is_carried_through_bitwise(self, tmp_path):
        (_, parquets), _, _ = _project_stage(tmp_path, force_processing=True)
        table, _ = read_contact_depth_field(parquets[0])
        expected = [d for frame in sorted(DEPTHS) for d in DEPTHS[frame]]
        assert table["signed_depth_mm"].tolist() == expected

    def test_the_reference_ply_provenance_is_stamped(self, tmp_path):
        (_, parquets), ply, _ = _project_stage(tmp_path, force_processing=True)
        _, metadata = read_contact_depth_field(parquets[0])
        assert metadata["schema_version"] == "2"
        assert metadata["reference_ply"] == ply.name
        assert int(metadata["reference_ply_vertex_count"]) == len(VERTICES)
        assert float(metadata["dedup_epsilon"]) == 0.5

    def test_the_coordinate_space_is_not_restamped(self, tmp_path):
        (_, parquets), _, _ = _project_stage(tmp_path, force_processing=True)
        _, metadata = read_contact_depth_field(parquets[0])
        assert metadata["coordinate_space"] == "icp_registered"

    def test_the_pipeline_stage_is_restamped(self, tmp_path):
        """``blocks_projected/`` was written by postprocessing, not by merging.

        Projection does not move points between spaces, so
        ``coordinate_space`` is carried through -- but the artifact's producing
        stage did change, and an intermediate that still reads ``"merging"``
        misreports where the field has been.
        """
        (_, parquets), _, _ = _project_stage(tmp_path, force_processing=True)
        _, metadata = read_contact_depth_field(parquets[0])
        assert metadata["pipeline_stage"] == "postprocessing"
        assert FIELD_METADATA["pipeline_stage"] == "merging"

    def test_a_missing_provenance_sidecar_raises_naming_it(self, tmp_path):
        ply = _write_forearm_ply(tmp_path)
        forearm_dedup_metadata_path(ply).unlink()
        with pytest.raises(FileNotFoundError, match="_dedup_metadata.json"):
            _project_stage(tmp_path, ply=ply, force_processing=True)

    def test_a_provenance_count_that_disagrees_with_the_ply_raises(self, tmp_path):
        ply = _write_forearm_ply(tmp_path)
        write_forearm_dedup_metadata(
            ply,
            source_ply=tmp_path / "src.ply",
            epsilon=0.5,
            epsilon_source=EPSILON_SOURCE_DAG_CONFIG,
            stats={"n_original": 99, "n_deduped": 90, "n_removed": 9},
        )
        with pytest.raises(ValueError, match="out of step"):
            _project_stage(tmp_path, ply=ply, force_processing=True)

    def test_a_missing_input_sidecar_raises(self, tmp_path):
        ply = _write_forearm_ply(tmp_path)
        csv, parquet = _write_stage_block(tmp_path)
        parquet.unlink()
        out = tmp_path / "blocks_projected"
        with pytest.raises(FileNotFoundError, match="contact_depth_field.parquet"):
            project_contacts_onto_forearm(
                input_files=[csv], forearm_ply_path=ply, output_dir=out,
                projection_stats_path=out / "projection_stats.csv",
                force_processing=True,
            )


class TestProjectionIdempotency:
    """Task 6.7: the boundary is the block, matching this stage's existing check."""

    def test_an_unchanged_rerun_regenerates_nothing(self, tmp_path):
        (_, parquets), ply, out = _project_stage(tmp_path, force_processing=True)
        stamp = parquets[0].stat().st_mtime_ns

        csv = tmp_path / "blocks_deduped" / parquets[0].name.replace(
            "_contact_depth_field.parquet", "_merged_data.csv"
        )
        again, again_pq = project_contacts_onto_forearm(
            input_files=[csv], forearm_ply_path=ply, output_dir=out,
            projection_stats_path=out / "projection_stats.csv",
        )
        assert again_pq == parquets
        assert again_pq[0].stat().st_mtime_ns == stamp

    def test_deleting_only_the_parquet_re_runs_the_block(self, tmp_path):
        (_, parquets), ply, out = _project_stage(tmp_path, force_processing=True)
        parquets[0].unlink()

        csv = tmp_path / "blocks_deduped" / parquets[0].name.replace(
            "_contact_depth_field.parquet", "_merged_data.csv"
        )
        _, again_pq = project_contacts_onto_forearm(
            input_files=[csv], forearm_ply_path=ply, output_dir=out,
            projection_stats_path=out / "projection_stats.csv",
        )
        assert again_pq[0].exists()

    def test_a_missing_forearm_ply_returns_two_empty_lists(self, tmp_path):
        csv, _ = _write_stage_block(tmp_path)
        result = project_contacts_onto_forearm(
            input_files=[csv],
            forearm_ply_path=tmp_path / "absent.ply",
            output_dir=tmp_path / "out",
            projection_stats_path=tmp_path / "out" / "stats.csv",
        )
        assert result == ([], [])
