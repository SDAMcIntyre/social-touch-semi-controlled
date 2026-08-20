"""Unit tests for ``postprocessing.depth_field_stage_io``.

Everything here runs on synthetic tables: no Open3D, no PyQt5, no recording, no
parquet.  That is the point of the module under test — the five postprocessing
stages that own this logic import geometry engines and GUI toolkits at module
scope and cannot be imported by a test at all, so the logic was moved somewhere
it can be exercised directly.

Numbers are hand-computed wherever a transform is checked, so a passing test
means the arithmetic is right and not merely self-consistent.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from postprocessing.depth_field_stage_io import (
    CONTACT_DEPTH_COLUMN,
    CONTACT_POINTS_COLUMN,
    DEPTH_COLUMN,
    FRAME_INDEX_COLUMN,
    apply_dedup_mapping_to_field,
    apply_pca_calibration_to_field,
    apply_rigid_transform_to_field,
    apply_transform_schedule_to_field,
    apply_vertex_addressing_to_field,
    assert_max_depth_agrees_with_csv,
    assert_row_counts_agree_with_csv,
    csv_contact_depth_by_frame,
    csv_contact_point_counts_by_frame,
    depth_field_path_for_csv,
    field_max_depth_magnitude_by_frame,
    field_row_counts_by_frame,
)
from postprocessing.xyz_reference_from_gestures.calibration_pca_engine import (
    CalibrationResult,
    PCACalibrationEngine,
)
from preprocessing.forearm_extraction.registration.csv_spatial_transformer import (
    serialize_contact_points,
)
from preprocessing.motion_analysis.tactile_quantification.io.contact_depth_field_io import (
    ALL_COLUMN_DTYPES,
    COLUMN_DTYPES,
    VERTEX_ID_COLUMN,
)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_field(
    frames,
    xyz,
    depths,
    *,
    times=None,
    vertex_ids=None,
) -> pd.DataFrame:
    """Build a schema-conforming depth-field table with the exact dtypes."""
    frames = np.asarray(frames, dtype=np.int32)
    xyz = np.asarray(xyz, dtype=np.float32).reshape(len(frames), 3)
    depths = np.asarray(depths, dtype=np.float64)
    if times is None:
        times = frames.astype(np.float64) / 30.0

    data = {
        "frame_index": frames,
        "time_s": np.asarray(times, dtype=np.float64),
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "z": xyz[:, 2],
        DEPTH_COLUMN: depths,
    }
    if vertex_ids is not None:
        data[VERTEX_ID_COLUMN] = np.asarray(vertex_ids, dtype=np.int32)

    table = pd.DataFrame(data)
    for name, dtype in ALL_COLUMN_DTYPES.items():
        if name in table.columns:
            assert table[name].dtype == dtype, name
    return table


class FakeDedupMapping:
    """The structural shape ``deduplicate_xy_mapping`` returns, without importing it.

    Importing the real ``DedupMapping`` would mean a ``code/src`` test reaching
    into ``code/scripts``; the module under test is deliberately structural
    about this type for the same reason.
    """

    def __init__(self, kept_indices, labels) -> None:
        self.kept_indices = np.asarray(kept_indices, dtype=np.intp)
        self.labels = np.asarray(labels, dtype=np.intp)


def write_stage_csv(path: Path, points_by_frame, *, extra_rows=0) -> Path:
    """Write a minimal stage CSV with the two columns the check reads.

    ``extra_rows`` appends neural-only rows — NaN frame index, empty contact
    cell — which the merged CSV is full of and which the count must ignore.
    """
    records = []
    for frame in sorted(points_by_frame):
        points = [tuple(float(v) for v in p) for p in points_by_frame[frame]]
        records.append(
            {
                FRAME_INDEX_COLUMN: float(frame),
                CONTACT_POINTS_COLUMN: serialize_contact_points(points),
            }
        )
    for _ in range(extra_rows):
        records.append(
            {FRAME_INDEX_COLUMN: np.nan, CONTACT_POINTS_COLUMN: "[]"}
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(
        records, columns=[FRAME_INDEX_COLUMN, CONTACT_POINTS_COLUMN]
    ).to_csv(path, index=False)
    return path


def simple_field() -> pd.DataFrame:
    """Four rows over two frames, with distinct coordinates and depths."""
    return make_field(
        frames=[10, 10, 11, 11],
        xyz=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.5, 0.5, 2.5], [7.0, -8.0, 9.0]],
        depths=[-1.25, -2.5, -0.75, -3.125],
    )


def rotation_z(degrees: float) -> np.ndarray:
    """A 3x3 rotation about +Z."""
    theta = np.deg2rad(degrees)
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rigid(R: np.ndarray, t) -> np.ndarray:
    """Assemble a 4x4 from a rotation and a translation."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


# ---------------------------------------------------------------------------
# 1. Rigid transform
# ---------------------------------------------------------------------------


class TestRigidTransform:
    def test_identity_is_a_no_op(self):
        table = simple_field()
        out = apply_rigid_transform_to_field(table, np.eye(4))

        for column in ("x", "y", "z"):
            np.testing.assert_array_equal(
                out[column].to_numpy(), table[column].to_numpy()
            )

    def test_known_rotation_and_translation(self):
        """90 degrees about +Z then a translation, checked against hand values."""
        table = make_field(
            frames=[5, 5],
            xyz=[[1.0, 0.0, 2.0], [0.0, 3.0, -4.0]],
            depths=[-1.0, -2.0],
        )
        T = rigid(rotation_z(90.0), (10.0, 20.0, 30.0))

        out = apply_rigid_transform_to_field(table, T)

        # (1, 0, 2) -> (0, 1, 2) -> (10, 21, 32)
        # (0, 3, -4) -> (-3, 0, -4) -> (7, 20, 26)
        expected = np.array([[10.0, 21.0, 32.0], [7.0, 20.0, 26.0]])
        np.testing.assert_allclose(
            out[["x", "y", "z"]].to_numpy(dtype=np.float64), expected, atol=1e-5
        )

    def test_translation_only(self):
        table = simple_field()
        T = rigid(np.eye(3), (1.5, -2.5, 0.25))

        out = apply_rigid_transform_to_field(table, T)

        expected = table[["x", "y", "z"]].to_numpy(dtype=np.float64) + np.array(
            [1.5, -2.5, 0.25]
        )
        np.testing.assert_allclose(
            out[["x", "y", "z"]].to_numpy(dtype=np.float64), expected, atol=1e-5
        )

    def test_input_table_is_not_mutated(self):
        table = simple_field()
        before = table.copy(deep=True)

        apply_rigid_transform_to_field(table, rigid(rotation_z(37.0), (1.0, 2.0, 3.0)))

        pd.testing.assert_frame_equal(table, before)

    def test_vertex_id_is_carried_through(self):
        table = make_field(
            frames=[1, 1],
            xyz=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            depths=[-1.0, -2.0],
            vertex_ids=[7, 42],
        )
        out = apply_rigid_transform_to_field(table, rigid(rotation_z(15.0), (1, 2, 3)))

        np.testing.assert_array_equal(
            out[VERTEX_ID_COLUMN].to_numpy(), np.array([7, 42], dtype=np.int32)
        )
        assert out[VERTEX_ID_COLUMN].dtype == np.int32

    @pytest.mark.parametrize(
        "bad",
        [
            np.eye(3),
            np.zeros((4, 5)),
        ],
    )
    def test_wrong_shape_raises(self, bad):
        with pytest.raises(ValueError, match="must be 4x4"):
            apply_rigid_transform_to_field(simple_field(), bad)

    def test_non_finite_matrix_raises(self):
        T = np.eye(4)
        T[0, 3] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            apply_rigid_transform_to_field(simple_field(), T)

    def test_projective_bottom_row_raises(self):
        T = np.eye(4)
        T[3, 0] = 0.5
        with pytest.raises(ValueError, match=r"bottom row"):
            apply_rigid_transform_to_field(simple_field(), T)


# ---------------------------------------------------------------------------
# 2. Transform schedule
# ---------------------------------------------------------------------------


class TestTransformSchedule:
    def test_different_frames_get_different_matrices(self):
        table = make_field(
            frames=[0, 5, 10, 20],
            xyz=[[1.0, 0.0, 0.0]] * 4,
            depths=[-1.0, -2.0, -3.0, -4.0],
        )
        schedule = [
            (0, rigid(np.eye(3), (100.0, 0.0, 0.0))),
            (10, rigid(np.eye(3), (0.0, 200.0, 0.0))),
        ]

        out = apply_transform_schedule_to_field(table, schedule)

        expected = np.array(
            [
                [101.0, 0.0, 0.0],  # frame 0  -> segment [0, 10)
                [101.0, 0.0, 0.0],  # frame 5  -> segment [0, 10)
                [1.0, 200.0, 0.0],  # frame 10 -> segment [10, inf)
                [1.0, 200.0, 0.0],  # frame 20 -> segment [10, inf)
            ]
        )
        np.testing.assert_allclose(
            out[["x", "y", "z"]].to_numpy(dtype=np.float64), expected, atol=1e-4
        )

    def test_frames_before_the_first_segment_are_untouched(self):
        table = make_field(
            frames=[3, 9],
            xyz=[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            depths=[-1.0, -2.0],
        )
        schedule = [(5, rigid(np.eye(3), (10.0, 0.0, 0.0)))]

        out = apply_transform_schedule_to_field(table, schedule)

        np.testing.assert_allclose(
            out[["x", "y", "z"]].to_numpy(dtype=np.float64),
            np.array([[1.0, 1.0, 1.0], [12.0, 2.0, 2.0]]),
            atol=1e-5,
        )

    def test_empty_schedule_is_the_passthrough(self):
        table = simple_field()
        out = apply_transform_schedule_to_field(table, [])

        pd.testing.assert_frame_equal(out, table.reset_index(drop=True))

    def test_each_row_is_transformed_exactly_once(self):
        """Segments are disjoint: a doubled translation would prove otherwise."""
        table = make_field(
            frames=[0, 1, 2],
            xyz=[[0.0, 0.0, 0.0]] * 3,
            depths=[-1.0, -1.0, -1.0],
        )
        schedule = [
            (0, rigid(np.eye(3), (1.0, 0.0, 0.0))),
            (1, rigid(np.eye(3), (1.0, 0.0, 0.0))),
            (2, rigid(np.eye(3), (1.0, 0.0, 0.0))),
        ]

        out = apply_transform_schedule_to_field(table, schedule)

        np.testing.assert_allclose(out["x"].to_numpy(dtype=np.float64), 1.0)

    def test_duplicate_start_frame_is_accepted_and_the_later_wins(self):
        """``get_transform_schedule`` emits (0, T_prev) then (0, T_same_block)."""
        table = make_field(frames=[0], xyz=[[0.0, 0.0, 0.0]], depths=[-1.0])
        schedule = [
            (0, rigid(np.eye(3), (5.0, 0.0, 0.0))),
            (0, rigid(np.eye(3), (0.0, 7.0, 0.0))),
        ]

        out = apply_transform_schedule_to_field(table, schedule)

        np.testing.assert_allclose(
            out[["x", "y", "z"]].to_numpy(dtype=np.float64),
            np.array([[0.0, 7.0, 0.0]]),
            atol=1e-5,
        )

    def test_decreasing_start_frames_raise(self):
        schedule = [(10, np.eye(4)), (5, np.eye(4))]
        with pytest.raises(ValueError, match="non-decreasing"):
            apply_transform_schedule_to_field(simple_field(), schedule)

    def test_negative_start_frame_raises(self):
        with pytest.raises(ValueError, match="cannot begin before frame 0"):
            apply_transform_schedule_to_field(simple_field(), [(-1, np.eye(4))])

    def test_bad_matrix_in_a_later_entry_raises(self):
        schedule = [(0, np.eye(4)), (5, np.eye(3))]
        with pytest.raises(ValueError, match="schedule entry 1"):
            apply_transform_schedule_to_field(simple_field(), schedule)

    def test_input_table_is_not_mutated(self):
        table = simple_field()
        before = table.copy(deep=True)

        apply_transform_schedule_to_field(
            table, [(0, rigid(rotation_z(20.0), (1.0, 2.0, 3.0)))]
        )

        pd.testing.assert_frame_equal(table, before)


# ---------------------------------------------------------------------------
# 3. PCA calibration
# ---------------------------------------------------------------------------


def a_calibration() -> CalibrationResult:
    """A calibration with a non-trivial rotation in both phases."""
    return CalibrationResult(
        mean_1=np.array([1.0, -2.0, 0.5]),
        R1=rotation_z(30.0),
        mean_2=np.array([0.25, -0.75]),
        R2=rotation_z(45.0),
    )


class TestPcaCalibration:
    def test_matches_the_engine_applied_directly(self):
        table = simple_field()
        calib = a_calibration()

        out = apply_pca_calibration_to_field(table, calib)

        expected = PCACalibrationEngine.apply_full_transform(
            table[["x", "y", "z"]].to_numpy(dtype=np.float64), calib
        )
        np.testing.assert_allclose(
            out[["x", "y", "z"]].to_numpy(dtype=np.float64), expected, atol=1e-4
        )

    def test_hand_computed_for_an_axis_aligned_calibration(self):
        """Identity rotations reduce the transform to two centrings."""
        table = make_field(
            frames=[1], xyz=[[10.0, 20.0, 30.0]], depths=[-1.0]
        )
        calib = CalibrationResult(
            mean_1=np.array([1.0, 2.0, 3.0]),
            R1=np.eye(3),
            mean_2=np.array([4.0, 5.0]),
            R2=np.eye(3),
        )

        out = apply_pca_calibration_to_field(table, calib)

        # (10, 20, 30) - (1, 2, 3) = (9, 18, 27); then x -= 4, y -= 5.
        np.testing.assert_allclose(
            out[["x", "y", "z"]].to_numpy(dtype=np.float64),
            np.array([[5.0, 13.0, 27.0]]),
            atol=1e-5,
        )

    def test_input_table_is_not_mutated(self):
        table = simple_field()
        before = table.copy(deep=True)

        apply_pca_calibration_to_field(table, a_calibration())

        pd.testing.assert_frame_equal(table, before)

    def test_a_malformed_calibration_raises(self):
        calib = CalibrationResult(
            mean_1=np.array([1.0, 2.0]),  # wrong shape
            R1=np.eye(3),
            mean_2=np.array([0.0, 0.0]),
            R2=np.eye(3),
        )
        with pytest.raises(ValueError, match=r"calibration.mean_1 has shape"):
            apply_pca_calibration_to_field(simple_field(), calib)

    def test_a_non_finite_calibration_raises(self):
        calib = a_calibration()
        calib.R2 = calib.R2.copy()
        calib.R2[0, 0] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            apply_pca_calibration_to_field(simple_field(), calib)


# ---------------------------------------------------------------------------
# 4. Deduplication — the max-magnitude rule
# ---------------------------------------------------------------------------


class TestDedupMapping:
    def test_survivor_inherits_the_most_negative_depth_of_its_group(self):
        """Rows 0/1/2 collapse into row 0; row 3 stands alone."""
        table = make_field(
            frames=[7, 7, 7, 7],
            xyz=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [9.0, 9.0, 9.0]],
            depths=[-1.0, -5.5, -0.25, -2.0],
        )
        mappings = {7: FakeDedupMapping(kept_indices=[0, 3], labels=[0, 0, 0, 1])}

        out = apply_dedup_mapping_to_field(table, mappings)

        assert len(out) == 2
        # The survivor's own -1.0 is replaced by the group's deepest, -5.5.
        np.testing.assert_array_equal(
            out[DEPTH_COLUMN].to_numpy(), np.array([-5.5, -2.0])
        )
        # The dropped rows really are gone, and coordinates never move.
        np.testing.assert_allclose(
            out[["x", "y", "z"]].to_numpy(dtype=np.float64),
            np.array([[0.0, 0.0, 0.0], [9.0, 9.0, 9.0]]),
        )

    def test_the_inherited_value_is_bitwise_one_of_the_group_members(self):
        """Never an average, never a recomputation — a value taken verbatim."""
        deep = -np.float64(1) / 3  # not representable, so any arithmetic shows
        table = make_field(
            frames=[1, 1],
            xyz=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            depths=[-0.125, deep],
        )
        mappings = {1: FakeDedupMapping(kept_indices=[0], labels=[0, 0])}

        out = apply_dedup_mapping_to_field(table, mappings)

        assert out[DEPTH_COLUMN].to_numpy()[0].tobytes() == np.float64(deep).tobytes()

    def test_a_group_mixing_signs_keeps_the_penetrating_row(self):
        """Documented behaviour: negative is penetrating, so the minimum wins.

        A positive signed depth means the vertex is *outside* the hand; the
        field's positives are bounded by the contact-detection epsilon, so this
        case is synthetic.  It is asserted anyway because the rule must be
        unambiguous where the two readings of 'maximum magnitude' diverge.
        """
        table = make_field(
            frames=[2, 2],
            xyz=[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]],
            depths=[-1.0, 9.0],
        )
        mappings = {2: FakeDedupMapping(kept_indices=[0], labels=[0, 0])}

        out = apply_dedup_mapping_to_field(table, mappings)

        assert out[DEPTH_COLUMN].to_numpy()[0] == -1.0

    def test_per_frame_max_magnitude_is_preserved(self):
        """The invariant the rule was chosen for: the CSV's contact_depth holds."""
        rng = np.random.default_rng(20260818)
        for _ in range(20):
            n = int(rng.integers(3, 12))
            depths = -rng.random(n) * 30.0
            labels = rng.integers(0, max(2, n // 2), size=n)
            labels = np.unique(labels, return_inverse=True)[1]
            kept = np.array(
                [int(np.flatnonzero(labels == lab)[0]) for lab in np.unique(labels)]
            )
            table = make_field(
                frames=np.full(n, 4),
                xyz=rng.random((n, 3)) * 10.0,
                depths=depths,
            )

            out = apply_dedup_mapping_to_field(
                table, {4: FakeDedupMapping(kept_indices=np.sort(kept), labels=labels)}
            )

            assert np.max(np.abs(out[DEPTH_COLUMN].to_numpy())) == np.max(
                np.abs(depths)
            )

    def test_row_order_is_preserved_across_frames(self):
        table = make_field(
            frames=[1, 2, 1, 2],
            xyz=[[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]],
            depths=[-1.0, -2.0, -3.0, -4.0],
        )
        mappings = {
            1: FakeDedupMapping(kept_indices=[1], labels=[0, 0]),
            2: FakeDedupMapping(kept_indices=[0, 1], labels=[0, 1]),
        }

        out = apply_dedup_mapping_to_field(table, mappings)

        # Kept table rows: frame 1's second row (position 2), and both of
        # frame 2's (positions 1 and 3) — emitted in table order.
        np.testing.assert_array_equal(
            out[FRAME_INDEX_COLUMN].to_numpy(), np.array([2, 1, 2], dtype=np.int32)
        )
        np.testing.assert_allclose(out["x"].to_numpy(), np.array([1.0, 2.0, 3.0]))

    def test_nothing_collapsed_leaves_the_table_alone(self):
        table = simple_field()
        mappings = {
            10: FakeDedupMapping(kept_indices=[0, 1], labels=[0, 1]),
            11: FakeDedupMapping(kept_indices=[0, 1], labels=[0, 1]),
        }

        out = apply_dedup_mapping_to_field(table, mappings)

        pd.testing.assert_frame_equal(out, table.reset_index(drop=True))

    def test_input_table_is_not_mutated(self):
        table = simple_field()
        before = table.copy(deep=True)

        apply_dedup_mapping_to_field(
            table,
            {
                10: FakeDedupMapping(kept_indices=[0], labels=[0, 0]),
                11: FakeDedupMapping(kept_indices=[0], labels=[0, 0]),
            },
        )

        pd.testing.assert_frame_equal(table, before)

    def test_a_missing_frame_raises(self):
        with pytest.raises(ValueError, match="cover different frames"):
            apply_dedup_mapping_to_field(
                simple_field(),
                {10: FakeDedupMapping(kept_indices=[0], labels=[0, 0])},
            )

    def test_a_label_count_mismatch_raises(self):
        table = simple_field()
        mappings = {
            10: FakeDedupMapping(kept_indices=[0], labels=[0, 0, 0]),
            11: FakeDedupMapping(kept_indices=[0], labels=[0, 0]),
        }
        with pytest.raises(ValueError, match=r"Frame 10: .*3 input point"):
            apply_dedup_mapping_to_field(table, mappings)

    def test_two_survivors_in_one_cluster_raise(self):
        table = simple_field()
        mappings = {
            10: FakeDedupMapping(kept_indices=[0, 1], labels=[0, 0]),
            11: FakeDedupMapping(kept_indices=[0], labels=[0, 0]),
        }
        with pytest.raises(ValueError, match="share a cluster label"):
            apply_dedup_mapping_to_field(table, mappings)

    def test_a_dropped_cluster_raises(self):
        table = simple_field()
        mappings = {
            10: FakeDedupMapping(kept_indices=[0], labels=[0, 1]),
            11: FakeDedupMapping(kept_indices=[0], labels=[0, 0]),
        }
        with pytest.raises(ValueError, match="Every cluster must keep"):
            apply_dedup_mapping_to_field(table, mappings)

    def test_out_of_range_kept_index_raises(self):
        table = simple_field()
        mappings = {
            10: FakeDedupMapping(kept_indices=[5], labels=[0, 0]),
            11: FakeDedupMapping(kept_indices=[0], labels=[0, 0]),
        }
        with pytest.raises(ValueError, match="outside the 2 row"):
            apply_dedup_mapping_to_field(table, mappings)

    def test_unsorted_kept_indices_raise(self):
        table = simple_field()
        mappings = {
            10: FakeDedupMapping(kept_indices=[1, 0], labels=[0, 1]),
            11: FakeDedupMapping(kept_indices=[0], labels=[0, 0]),
        }
        with pytest.raises(ValueError, match="strictly ascending"):
            apply_dedup_mapping_to_field(table, mappings)

    def test_empty_survivor_set_raises(self):
        table = simple_field()
        mappings = {
            10: FakeDedupMapping(kept_indices=[], labels=[0, 0]),
            11: FakeDedupMapping(kept_indices=[0], labels=[0, 0]),
        }
        with pytest.raises(ValueError, match="keeps no rows at all"):
            apply_dedup_mapping_to_field(table, mappings)


# ---------------------------------------------------------------------------
# 5. Vertex re-addressing
# ---------------------------------------------------------------------------


REFERENCE_VERTICES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [2.5, -3.5, 4.5],
    ],
    dtype=np.float64,
)


class TestVertexAddressing:
    def test_coordinates_and_ids_come_from_the_named_vertices(self):
        table = make_field(
            frames=[3, 3, 4],
            xyz=[[9.0, 9.0, 9.0]] * 3,
            depths=[-1.0, -2.0, -3.0],
        )
        indices = {3: np.array([4, 0]), 4: np.array([2])}

        out = apply_vertex_addressing_to_field(table, indices, REFERENCE_VERTICES)

        np.testing.assert_array_equal(
            out[VERTEX_ID_COLUMN].to_numpy(), np.array([4, 0, 2], dtype=np.int32)
        )
        np.testing.assert_allclose(
            out[["x", "y", "z"]].to_numpy(dtype=np.float64),
            REFERENCE_VERTICES[[4, 0, 2]],
            atol=1e-5,
        )
        assert out[VERTEX_ID_COLUMN].dtype == np.int32

    def test_row_order_and_count_are_preserved(self):
        """Two points may snap to one vertex; neither is dropped or reordered."""
        table = make_field(
            frames=[8, 9, 8],
            xyz=[[0.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0]],
            depths=[-1.0, -2.0, -3.0],
        )
        indices = {8: np.array([1, 1]), 9: np.array([3])}

        out = apply_vertex_addressing_to_field(table, indices, REFERENCE_VERTICES)

        assert len(out) == 3
        np.testing.assert_array_equal(
            out[FRAME_INDEX_COLUMN].to_numpy(), np.array([8, 9, 8], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            out[VERTEX_ID_COLUMN].to_numpy(), np.array([1, 3, 1], dtype=np.int32)
        )
        np.testing.assert_array_equal(out[DEPTH_COLUMN].to_numpy(), [-1.0, -2.0, -3.0])

    def test_depth_is_untouched(self):
        table = simple_field()
        indices = {10: np.array([0, 1]), 11: np.array([2, 3])}

        out = apply_vertex_addressing_to_field(table, indices, REFERENCE_VERTICES)

        np.testing.assert_array_equal(
            out[DEPTH_COLUMN].to_numpy(), table[DEPTH_COLUMN].to_numpy()
        )

    def test_input_table_is_not_mutated(self):
        table = simple_field()
        before = table.copy(deep=True)

        apply_vertex_addressing_to_field(
            table, {10: np.array([0, 1]), 11: np.array([2, 3])}, REFERENCE_VERTICES
        )

        pd.testing.assert_frame_equal(table, before)

    def test_an_already_addressed_table_raises(self):
        table = make_field(
            frames=[1, 1],
            xyz=[[0.0, 0, 0], [1.0, 0, 0]],
            depths=[-1.0, -2.0],
            vertex_ids=[0, 1],
        )
        with pytest.raises(ValueError, match="already carries"):
            apply_vertex_addressing_to_field(
                table, {1: np.array([0, 1])}, REFERENCE_VERTICES
            )

    def test_out_of_range_index_raises(self):
        table = simple_field()
        indices = {10: np.array([0, 99]), 11: np.array([1, 2])}
        with pytest.raises(ValueError, match="outside the 5 vertices"):
            apply_vertex_addressing_to_field(table, indices, REFERENCE_VERTICES)

    def test_negative_index_raises(self):
        table = simple_field()
        indices = {10: np.array([0, -1]), 11: np.array([1, 2])}
        with pytest.raises(ValueError, match="outside the 5 vertices"):
            apply_vertex_addressing_to_field(table, indices, REFERENCE_VERTICES)

    def test_count_mismatch_raises(self):
        table = simple_field()
        indices = {10: np.array([0]), 11: np.array([1, 2])}
        with pytest.raises(ValueError, match=r"Frame 10: .*1 vertex"):
            apply_vertex_addressing_to_field(table, indices, REFERENCE_VERTICES)

    def test_frame_set_mismatch_raises(self):
        table = simple_field()
        with pytest.raises(ValueError, match="cover different frames"):
            apply_vertex_addressing_to_field(
                table, {10: np.array([0, 1])}, REFERENCE_VERTICES
            )

    def test_float_indices_raise(self):
        table = simple_field()
        indices = {10: np.array([0.0, 1.0]), 11: np.array([1, 2])}
        with pytest.raises(ValueError, match="never a float"):
            apply_vertex_addressing_to_field(table, indices, REFERENCE_VERTICES)

    def test_empty_reference_raises(self):
        with pytest.raises(ValueError, match="reference_vertices is empty"):
            apply_vertex_addressing_to_field(
                simple_field(),
                {10: np.array([0, 1]), 11: np.array([0, 1])},
                np.empty((0, 3)),
            )

    def test_wrong_reference_shape_raises(self):
        with pytest.raises(ValueError, match=r"must be \(V, 3\)"):
            apply_vertex_addressing_to_field(
                simple_field(),
                {10: np.array([0, 1]), 11: np.array([0, 1])},
                np.zeros((4, 2)),
            )


# ---------------------------------------------------------------------------
# 6. The row-count agreement check
# ---------------------------------------------------------------------------


class TestRowCountAgreement:
    def test_matched_input_passes(self, tmp_path):
        table = simple_field()
        csv = write_stage_csv(
            tmp_path / "stage.csv",
            {
                10: [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)],
                11: [(-1.5, 0.5, 2.5), (7.0, -8.0, 9.0)],
            },
            extra_rows=3,
        )

        assert assert_row_counts_agree_with_csv(table, csv) is None

    def test_it_names_the_first_offending_frame(self, tmp_path):
        table = make_field(
            frames=[10, 20, 20, 30, 30, 30],
            xyz=np.zeros((6, 3)),
            depths=[-1.0] * 6,
        )
        csv = write_stage_csv(
            tmp_path / "stage.csv",
            {
                10: [(0.0, 0.0, 0.0)],
                20: [(0.0, 0.0, 0.0)],  # one, not two   <- first offender
                30: [(0.0, 0.0, 0.0)],  # one, not three
            },
        )

        with pytest.raises(ValueError) as excinfo:
            assert_row_counts_agree_with_csv(table, csv)

        message = str(excinfo.value)
        assert message.startswith("Row-count disagreement at frame 20:")
        assert "2 row(s)" in message and "1 contact point(s)" in message
        assert "2 frame(s) disagree in total" in message

    def test_a_frame_missing_from_the_csv_raises(self, tmp_path):
        table = simple_field()
        csv = write_stage_csv(
            tmp_path / "stage.csv", {10: [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]}
        )

        with pytest.raises(ValueError, match="frame 11"):
            assert_row_counts_agree_with_csv(table, csv)

    def test_a_frame_missing_from_the_field_raises(self, tmp_path):
        table = make_field(frames=[10], xyz=[[1.0, 2.0, 3.0]], depths=[-1.0])
        csv = write_stage_csv(
            tmp_path / "stage.csv",
            {10: [(1.0, 2.0, 3.0)], 12: [(0.0, 0.0, 0.0)]},
        )

        with pytest.raises(ValueError, match="frame 12"):
            assert_row_counts_agree_with_csv(table, csv)

    def test_a_malformed_cell_is_caught(self, tmp_path):
        """The failure mode the check exists for: a silently dropped point."""
        csv = tmp_path / "stage.csv"
        pd.DataFrame(
            {
                FRAME_INDEX_COLUMN: [10.0],
                # The second point has two components; parse_contact_points
                # drops it without a word.
                CONTACT_POINTS_COLUMN: ["[[1.0 2.0 3.0] [4.0 5.0]]"],
            }
        ).to_csv(csv, index=False)
        table = make_field(
            frames=[10, 10],
            xyz=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            depths=[-1.0, -2.0],
        )

        with pytest.raises(ValueError, match="frame 10"):
            assert_row_counts_agree_with_csv(table, csv)

    def test_counts_ignore_empty_and_neural_only_rows(self, tmp_path):
        csv = write_stage_csv(
            tmp_path / "stage.csv", {4: [(0.0, 0.0, 0.0)]}, extra_rows=5
        )

        assert csv_contact_point_counts_by_frame(csv) == {4: 1}

    def test_a_missing_column_raises(self, tmp_path):
        csv = tmp_path / "stage.csv"
        pd.DataFrame({FRAME_INDEX_COLUMN: [1.0]}).to_csv(csv, index=False)

        with pytest.raises(ValueError, match="does not expose both"):
            csv_contact_point_counts_by_frame(csv)

    def test_a_fractional_frame_index_on_a_contact_row_raises(self, tmp_path):
        csv = tmp_path / "stage.csv"
        pd.DataFrame(
            {
                FRAME_INDEX_COLUMN: [1.5],
                CONTACT_POINTS_COLUMN: ["[[1.0 2.0 3.0]]"],
            }
        ).to_csv(csv, index=False)

        with pytest.raises(ValueError, match="not a whole number"):
            csv_contact_point_counts_by_frame(csv)

    def test_duplicate_contact_frames_raise(self, tmp_path):
        csv = tmp_path / "stage.csv"
        pd.DataFrame(
            {
                FRAME_INDEX_COLUMN: [7.0, 7.0],
                CONTACT_POINTS_COLUMN: ["[[1.0 2.0 3.0]]", "[[4.0 5.0 6.0]]"],
            }
        ).to_csv(csv, index=False)

        with pytest.raises(ValueError, match="two contact-bearing rows"):
            csv_contact_point_counts_by_frame(csv)

    def test_a_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            assert_row_counts_agree_with_csv(
                simple_field(), tmp_path / "nope.csv"
            )

    def test_field_counts(self):
        assert field_row_counts_by_frame(simple_field()) == {10: 2, 11: 2}


# ---------------------------------------------------------------------------
# 7. Dtypes and depth preservation
# ---------------------------------------------------------------------------


class TestDtypesAndDepthPreservation:
    @pytest.mark.parametrize(
        "apply",
        [
            pytest.param(
                lambda t: apply_rigid_transform_to_field(
                    t, rigid(rotation_z(31.7), (1.1, -2.2, 3.3))
                ),
                id="rigid",
            ),
            pytest.param(
                lambda t: apply_transform_schedule_to_field(
                    t, [(0, rigid(rotation_z(31.7), (1.1, -2.2, 3.3)))]
                ),
                id="schedule",
            ),
            pytest.param(
                lambda t: apply_pca_calibration_to_field(t, a_calibration()),
                id="pca",
            ),
            pytest.param(
                lambda t: apply_vertex_addressing_to_field(
                    t, {10: np.array([0, 1]), 11: np.array([2, 3])}, REFERENCE_VERTICES
                ),
                id="vertex-addressing",
            ),
        ],
    )
    def test_depth_is_bitwise_unchanged(self, apply):
        table = simple_field()

        out = apply(table)

        assert (
            out[DEPTH_COLUMN].to_numpy().tobytes()
            == table[DEPTH_COLUMN].to_numpy().tobytes()
        )

    @pytest.mark.parametrize(
        "apply",
        [
            pytest.param(
                lambda t: apply_rigid_transform_to_field(t, np.eye(4)), id="rigid"
            ),
            pytest.param(
                lambda t: apply_transform_schedule_to_field(t, [(0, np.eye(4))]),
                id="schedule",
            ),
            pytest.param(
                lambda t: apply_pca_calibration_to_field(t, a_calibration()),
                id="pca",
            ),
            pytest.param(
                lambda t: apply_dedup_mapping_to_field(
                    t,
                    {
                        10: FakeDedupMapping([0], [0, 0]),
                        11: FakeDedupMapping([0], [0, 0]),
                    },
                ),
                id="dedup",
            ),
        ],
    )
    def test_schema_dtypes_survive(self, apply):
        out = apply(simple_field())

        assert tuple(out.columns) == tuple(COLUMN_DTYPES)
        for name, dtype in COLUMN_DTYPES.items():
            assert out[name].dtype == dtype, name

    def test_vertex_addressing_yields_the_wide_layout(self):
        out = apply_vertex_addressing_to_field(
            simple_field(),
            {10: np.array([0, 1]), 11: np.array([2, 3])},
            REFERENCE_VERTICES,
        )

        assert tuple(out.columns) == tuple(ALL_COLUMN_DTYPES)
        for name, dtype in ALL_COLUMN_DTYPES.items():
            assert out[name].dtype == dtype, name

    def test_coordinates_are_the_float32_cast_of_the_float64_result(self):
        """One rounding, at the end — not float32 arithmetic all the way through."""
        table = make_field(
            frames=[1, 1],
            xyz=[[1.3, -2.7, 11.9], [104.7, 3.1, -55.55]],
            depths=[-1.0, -2.0],
        )
        T = rigid(rotation_z(37.3), (1234.567, -89.01, 0.4321))

        out = apply_rigid_transform_to_field(table, T)

        exact = (
            table[["x", "y", "z"]].to_numpy(dtype=np.float64) @ T[:3, :3].T + T[:3, 3]
        )
        assert (
            out[["x", "y", "z"]].to_numpy().tobytes()
            == exact.astype(np.float32).tobytes()
        )

    def test_a_wrong_dtype_table_is_rejected(self):
        table = simple_field()
        table["x"] = table["x"].astype(np.float64)

        with pytest.raises(ValueError, match="expected float32"):
            apply_rigid_transform_to_field(table, np.eye(4))

    def test_an_unknown_column_is_rejected(self):
        table = simple_field()
        table["extra"] = 1.0

        with pytest.raises(ValueError, match="neither"):
            apply_rigid_transform_to_field(table, np.eye(4))

    def test_an_empty_table_is_rejected(self):
        with pytest.raises(ValueError, match="zero rows"):
            apply_rigid_transform_to_field(simple_field().iloc[:0], np.eye(4))

    def test_a_non_dataframe_is_rejected(self):
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            apply_rigid_transform_to_field(np.zeros((3, 6)), np.eye(4))


# ---------------------------------------------------------------------------
# 8. Purity
# ---------------------------------------------------------------------------


def test_the_module_imports_no_geometry_engine_or_gui_toolkit():
    """Asserted, not trusted: the leaf exists to be importable without them."""
    import inspect

    from postprocessing import depth_field_stage_io

    source = inspect.getsource(depth_field_stage_io)
    for banned in ("open3d", "PyQt5", "pyvista", "prefect", "matplotlib"):
        assert (
            re.search(rf"^\s*(import|from)\s+{banned}", source, re.MULTILINE) is None
        ), banned


def test_the_module_imports_without_open3d_or_pyqt5():
    """The transitive property, proved in a fresh interpreter.

    The stub set mirrors ``conftest.py``; the point is that with the heavy
    package *roots* out of the way nothing this module reaches pulls a geometry
    engine or a GUI toolkit back in.
    """
    src = Path(__file__).resolve().parent.parent / "src"
    script = textwrap.dedent(
        f"""
        import sys, types
        from pathlib import Path
        SRC = Path(r"{src}")
        sys.path.insert(0, str(SRC))
        for dotted in (
            "preprocessing.motion_analysis",
            "preprocessing.forearm_extraction",
            "preprocessing.forearm_extraction.registration",
        ):
            module = types.ModuleType(dotted)
            module.__path__ = [str(SRC / Path(*dotted.split(".")))]
            module.__package__ = dotted
            sys.modules[dotted] = module

        import postprocessing.depth_field_stage_io  # noqa: F401

        leaked = [name for name in ("open3d", "PyQt5") if name in sys.modules]
        assert not leaked, leaked
        print("clean")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
    assert "clean" in completed.stdout


# ---------------------------------------------------------------------------
# The depth agreement check (task 6.2) and the artifact pairing rule
# ---------------------------------------------------------------------------


def write_stage_csv_with_depth(path: Path, points_by_frame, depths_by_frame,
                               *, extra_rows=0) -> Path:
    """Write a stage CSV carrying ``contact_depth`` as well as the points.

    ``extra_rows`` appends non-contact rows with ``contact_depth = 0.0`` — which
    is what the real merged CSV holds on every frame nothing touched, and which
    the check must ignore rather than treat as a frame with zero depth.
    """
    records = []
    for frame in sorted(points_by_frame):
        points = [tuple(float(v) for v in pt) for pt in points_by_frame[frame]]
        records.append(
            {
                FRAME_INDEX_COLUMN: float(frame),
                CONTACT_POINTS_COLUMN: serialize_contact_points(points),
                CONTACT_DEPTH_COLUMN: float(depths_by_frame[frame]),
            }
        )
    for index in range(extra_rows):
        records.append(
            {
                FRAME_INDEX_COLUMN: float(9000 + index),
                CONTACT_POINTS_COLUMN: "[]",
                CONTACT_DEPTH_COLUMN: 0.0,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(
        records,
        columns=[FRAME_INDEX_COLUMN, CONTACT_POINTS_COLUMN, CONTACT_DEPTH_COLUMN],
    ).to_csv(path, index=False)
    return path


class TestFieldMaxDepthMagnitude:
    def test_takes_the_largest_absolute_value_per_frame(self):
        table = make_field(
            frames=[3, 3, 3, 4],
            xyz=np.zeros((4, 3)),
            depths=[-1.5, -0.25, 0.75, -2.0],
        )
        assert field_max_depth_magnitude_by_frame(table) == {3: 1.5, 4: 2.0}

    def test_a_positive_value_counts_when_it_is_the_largest(self):
        table = make_field(
            frames=[7, 7], xyz=np.zeros((2, 3)), depths=[0.5, -0.25]
        )
        assert field_max_depth_magnitude_by_frame(table) == {7: 0.5}

    def test_rejects_a_table_that_is_not_the_schema(self):
        with pytest.raises(ValueError):
            field_max_depth_magnitude_by_frame(pd.DataFrame({"a": [1]}))


class TestCsvContactDepth:
    def test_reads_only_contact_bearing_rows(self, tmp_path):
        csv = write_stage_csv_with_depth(
            tmp_path / "s.csv",
            {10: [(0.0, 0.0, 0.0)], 11: [(1.0, 1.0, 1.0)]},
            {10: 2.5, 11: 4.0},
            extra_rows=3,
        )
        assert csv_contact_depth_by_frame(csv) == {10: 2.5, 11: 4.0}

    def test_a_missing_column_raises_naming_all_three(self, tmp_path):
        csv = write_stage_csv(tmp_path / "s.csv", {10: [(0.0, 0.0, 0.0)]})
        with pytest.raises(ValueError, match=CONTACT_DEPTH_COLUMN):
            csv_contact_depth_by_frame(csv)

    def test_a_duplicated_frame_is_ambiguous(self, tmp_path):
        csv = tmp_path / "dup.csv"
        cell = serialize_contact_points([(0.0, 0.0, 0.0)])
        pd.DataFrame(
            {
                FRAME_INDEX_COLUMN: [5.0, 5.0],
                CONTACT_POINTS_COLUMN: [cell, cell],
                CONTACT_DEPTH_COLUMN: [1.0, 2.0],
            }
        ).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="ambiguous"):
            csv_contact_depth_by_frame(csv)

    def test_a_fractional_frame_index_on_a_contact_row_raises(self, tmp_path):
        csv = tmp_path / "frac.csv"
        pd.DataFrame(
            {
                FRAME_INDEX_COLUMN: [5.5],
                CONTACT_POINTS_COLUMN: [serialize_contact_points([(0.0, 0.0, 0.0)])],
                CONTACT_DEPTH_COLUMN: [1.0],
            }
        ).to_csv(csv, index=False)
        with pytest.raises(ValueError, match="whole number"):
            csv_contact_depth_by_frame(csv)

    def test_a_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            csv_contact_depth_by_frame(tmp_path / "nope.csv")


class TestMaxDepthAgreement:
    def _pair(self, tmp_path, depths, csv_depths):
        table = make_field(
            frames=[10, 10, 11],
            xyz=np.zeros((3, 3)),
            depths=depths,
        )
        csv = write_stage_csv_with_depth(
            tmp_path / "s.csv",
            {10: [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], 11: [(2.0, 0.0, 0.0)]},
            csv_depths,
        )
        return table, csv

    def test_agreement_is_silent(self, tmp_path):
        table, csv = self._pair(tmp_path, [-1.0, -3.0, -2.0], {10: 3.0, 11: 2.0})
        assert assert_max_depth_agrees_with_csv(table, csv) is None

    def test_a_reduced_depth_raises_naming_the_frame(self, tmp_path):
        table, csv = self._pair(tmp_path, [-1.0, -1.5, -2.0], {10: 3.0, 11: 2.0})
        with pytest.raises(ValueError, match="frame 10"):
            assert_max_depth_agrees_with_csv(table, csv)

    def test_one_ulp_of_csv_round_trip_is_tolerated(self, tmp_path):
        table, csv = self._pair(
            tmp_path, [-1.0, -3.0, -2.0], {10: np.nextafter(3.0, 4.0), 11: 2.0}
        )
        assert assert_max_depth_agrees_with_csv(table, csv) is None

    def test_a_frame_only_the_field_has_raises(self, tmp_path):
        table = make_field(frames=[10, 12], xyz=np.zeros((2, 3)), depths=[-3.0, -1.0])
        csv = write_stage_csv_with_depth(
            tmp_path / "s.csv", {10: [(0.0, 0.0, 0.0)]}, {10: 3.0}
        )
        with pytest.raises(ValueError, match="frame 12"):
            assert_max_depth_agrees_with_csv(table, csv)

    def test_a_frame_only_the_csv_has_raises(self, tmp_path):
        table = make_field(frames=[10], xyz=np.zeros((1, 3)), depths=[-3.0])
        csv = write_stage_csv_with_depth(
            tmp_path / "s.csv",
            {10: [(0.0, 0.0, 0.0)], 12: [(1.0, 1.0, 1.0)]},
            {10: 3.0, 12: 1.0},
        )
        with pytest.raises(ValueError, match="frame 12"):
            assert_max_depth_agrees_with_csv(table, csv)

    def test_the_dedup_rule_keeps_the_invariant_true(self, tmp_path):
        """The reason the rule is a group *maximum magnitude* and not the survivor's own.

        The deepest row of frame 10 is dropped by the mapping; the survivor
        inherits its depth, so the CSV's pre-dedup ``contact_depth`` still holds.
        """
        table = make_field(
            frames=[10, 10, 11],
            xyz=[[0.0, 0.0, 0.0], [0.05, 0.0, 1.0], [5.0, 5.0, 0.0]],
            depths=[-1.0, -3.0, -2.0],
        )
        mappings = {
            10: FakeDedupMapping(kept_indices=[0], labels=[0, 0]),
            11: FakeDedupMapping(kept_indices=[0], labels=[0]),
        }
        reduced = apply_dedup_mapping_to_field(table, mappings)
        csv = write_stage_csv_with_depth(
            tmp_path / "s.csv",
            {10: [(0.0, 0.0, 0.0)], 11: [(5.0, 5.0, 0.0)]},
            {10: 3.0, 11: 2.0},
        )
        assert reduced[DEPTH_COLUMN].tolist() == [-3.0, -2.0]
        assert assert_max_depth_agrees_with_csv(reduced, csv) is None


class TestDepthFieldPathForCsv:
    def test_pairs_the_two_artifacts_by_stem(self):
        csv = Path("/data/blocks_deduped/S_semicontrolled_block-order-01_merged_data.csv")
        assert depth_field_path_for_csv(csv) == csv.with_name(
            "S_semicontrolled_block-order-01_contact_depth_field.parquet"
        )

    def test_refuses_to_guess_an_unrecognised_name(self):
        with pytest.raises(ValueError, match="_merged_data.csv"):
            depth_field_path_for_csv(Path("/data/whatever.csv"))

    def test_follows_the_pca_stage_filename_fork(self):
        # calibrate_pca_xyz is the one stage that renames its outputs. Both
        # artifacts fork together, so the pairing must survive the rename or
        # center_on_receptive_field cannot find its own inputs.
        csv = Path(
            "/data/blocks_pca_calibrated/"
            "S_semicontrolled_block-order-01_merged_data_pca-xyz.csv"
        )
        assert depth_field_path_for_csv(csv) == csv.with_name(
            "S_semicontrolled_block-order-01_contact_depth_field_pca-xyz.parquet"
        )

    def test_refuses_a_name_that_is_not_a_csv(self):
        with pytest.raises(ValueError, match="_merged_data.csv"):
            depth_field_path_for_csv(Path("/data/S_merged_data.parquet"))

    def test_refuses_an_ambiguous_name(self):
        # Two markers give two possible substitutions; picking one would pair
        # the CSV with a plausible-looking file that is not its own.
        with pytest.raises(ValueError, match="_merged_data.csv"):
            depth_field_path_for_csv(Path("/data/S_merged_data_merged_data.csv"))
