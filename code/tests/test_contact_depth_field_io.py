"""Tests for the long-form contact depth field sidecar.

Two families:

1. **Unit tests** on the writer/reader pair.  They need neither a recording nor
   Open3D: the writer duck-types four plain attributes off each frame, which is
   precisely what makes it a pure serialisation seam.  They always run.
2. **Integration tests** against a real recording, which check the invariants
   that only real data can establish — frame-set agreement with the
   somatosensory CSV in both directions, and per-frame bit-identity between
   ``max(|signed_depth_mm|)`` and the CSV's ``contact_depth``.  The reference
   bundle is not committable (size, participant data), so those skip loudly
   when it is absent rather than passing vacuously.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow", reason="pyarrow is required for the parquet sidecar")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from preprocessing.motion_analysis.tactile_quantification.io.contact_depth_field_io import (  # noqa: E402
    ALL_COLUMN_DTYPES,
    COLUMN_DTYPES,
    COORDINATE_SPACE,
    COORDINATE_SPACES,
    PRODUCED_BY,
    REFERENCE_PLY_METADATA_KEYS,
    SCHEMA_VERSION,
    SIGN_CONVENTION,
    SUPPORTED_SCHEMA_VERSIONS,
    UNITS,
    VERTEX_ID_COLUMN,
    read_contact_depth_field,
    validate_vertex_ids_against_reference,
    write_contact_depth_field,
    write_contact_depth_field_table,
)


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------
# The real ``ContactDepthFrame`` lives in a module that imports Open3D, which is
# absent from SDK-free unit-test environments.  The writer never touches the
# geometry engine — it reads ``frame_index``, ``time_s``, ``points`` and
# ``signed_depth_mm`` and nothing else — so a field-identical stand-in exercises
# exactly the same code path.  The real class is used whenever it is importable,
# which keeps the two definitions from drifting silently.

try:  # pragma: no cover - environment-dependent
    from preprocessing.motion_analysis.tactile_quantification.model.contact_depth_field import (
        ContactDepthFrame as _RealContactDepthFrame,
    )
except Exception:  # pragma: no cover - Open3D absent
    _RealContactDepthFrame = None


@dataclass(frozen=True)
class _StandInFrame:
    """Field-identical stand-in for ``ContactDepthFrame``."""

    frame_index: Optional[int]
    time_s: Optional[float]
    points: np.ndarray
    signed_depth_mm: np.ndarray
    normals: np.ndarray
    total_area_mm2: float
    mean_location: np.ndarray


_FRAME_CLS = _RealContactDepthFrame or _StandInFrame


def _make_frame(frame_index, time_s, points, depths):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    depths = np.asarray(depths, dtype=np.float64)
    mean_location = (
        points.mean(axis=0) if len(points) else np.zeros(3, dtype=np.float64)
    )
    return _FRAME_CLS(
        frame_index=frame_index,
        time_s=time_s,
        points=points,
        signed_depth_mm=depths,
        normals=np.zeros_like(points),
        total_area_mm2=float(len(points)),
        mean_location=mean_location,
    )


def _sample_series(seed: int = 20260812):
    """Three contacting frames with 4, 1 and 7 contact vertices."""
    rng = np.random.default_rng(seed)
    series = []
    for frame_index, (time_s, n_vertices) in enumerate(
        [(0.0, 4), (0.0333, 1), (0.4, 7)]
    ):
        points = rng.uniform(-80.0, 80.0, size=(n_vertices, 3))
        # Negative = penetrating; keep magnitudes inside the plausible band.
        depths = -rng.uniform(0.0, 12.0, size=n_vertices)
        series.append(_make_frame(frame_index, time_s, points, depths))
    return series


# ---------------------------------------------------------------------------
# 1. Round trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_values_and_dtypes(tmp_path):
    series = _sample_series()
    out = tmp_path / "rec_contact_depth_field.parquet"

    write_contact_depth_field(series, out, source_recording="rec")
    frame, metadata = read_contact_depth_field(out)

    assert list(frame.columns) == list(COLUMN_DTYPES)
    for column, dtype in COLUMN_DTYPES.items():
        assert frame[column].dtype == dtype, column

    expected_indices = np.concatenate(
        [np.full(len(f.points), f.frame_index) for f in series]
    )
    expected_times = np.concatenate(
        [np.full(len(f.points), f.time_s) for f in series]
    )
    expected_points = np.concatenate([f.points for f in series]).astype(np.float32)
    expected_depths = np.concatenate([f.signed_depth_mm for f in series])

    assert np.array_equal(frame["frame_index"].to_numpy(), expected_indices)
    assert np.array_equal(frame["time_s"].to_numpy(), expected_times.astype(np.float64))
    assert np.array_equal(frame[["x", "y", "z"]].to_numpy(), expected_points)

    # The headline invariant: depth survives at float64, to the bit.
    assert np.array_equal(frame["signed_depth_mm"].to_numpy(), expected_depths)

    assert metadata["source_recording"] == "rec"


def test_signed_depth_survives_round_trip_bit_identically(tmp_path):
    """Values chosen so a float32 store would provably lose them."""
    depths = np.array(
        [-0.1234567890123456, -12.345678901234567, -1e-9, -199.99999999999997],
        dtype=np.float64,
    )
    series = [_make_frame(0, 0.0, np.zeros((4, 3)), depths)]
    out = tmp_path / "rec_contact_depth_field.parquet"

    write_contact_depth_field(series, out, source_recording="rec")
    frame, _ = read_contact_depth_field(out)

    recovered = frame["signed_depth_mm"].to_numpy()
    assert np.array_equal(recovered, depths)
    # Guard against a vacuous pass: float32 really would have moved these.
    assert not np.array_equal(depths.astype(np.float32).astype(np.float64), depths)


def test_max_abs_depth_per_frame_survives_the_round_trip(tmp_path):
    """The invariant the CSV is checked against, stated on the sidecar alone."""
    series = _sample_series()
    out = tmp_path / "rec_contact_depth_field.parquet"

    write_contact_depth_field(series, out, source_recording="rec")
    frame, _ = read_contact_depth_field(out)

    recovered = (
        frame.groupby("frame_index")["signed_depth_mm"]
        .apply(lambda s: np.max(np.abs(s.to_numpy())))
        .to_numpy()
    )
    expected = np.array(
        [np.max(np.abs(f.signed_depth_mm)) for f in series], dtype=np.float64
    )
    assert np.array_equal(recovered, expected)


# ---------------------------------------------------------------------------
# 2. Metadata
# ---------------------------------------------------------------------------


def test_file_metadata_round_trips_all_six_keys(tmp_path):
    out = tmp_path / "ST14-01_block-order-01_contact_depth_field.parquet"
    write_contact_depth_field(
        _sample_series(), out, source_recording="ST14-01_block-order-01"
    )

    _, metadata = read_contact_depth_field(out)

    assert metadata["schema_version"] == SCHEMA_VERSION
    assert metadata["coordinate_space"] == COORDINATE_SPACE
    assert metadata["units"] == UNITS
    assert metadata["sign_convention"] == SIGN_CONVENTION
    assert metadata["source_recording"] == "ST14-01_block-order-01"
    assert metadata["produced_by"] == PRODUCED_BY


def test_file_is_self_describing_without_any_repo_import(tmp_path):
    """A bare pyarrow reader can recover units, space and sign convention."""
    out = tmp_path / "rec_contact_depth_field.parquet"
    write_contact_depth_field(_sample_series(), out, source_recording="rec")

    raw = pq.read_table(out).schema.metadata
    decoded = {k.decode(): v.decode() for k, v in raw.items()}

    assert decoded["units"] == "mm"
    assert decoded["coordinate_space"] == "kinect_space_1"
    assert decoded["sign_convention"] == "negative_is_penetrating"


def test_unknown_schema_version_raises(tmp_path):
    out = tmp_path / "rec_contact_depth_field.parquet"
    write_contact_depth_field(_sample_series(), out, source_recording="rec")

    # Rewrite the same payload under a future schema version.
    table = pq.read_table(out)
    bumped = table.replace_schema_metadata(
        {**{k.decode(): v.decode() for k, v in table.schema.metadata.items()},
         "schema_version": "99"}
    )
    future = tmp_path / "future.parquet"
    pq.write_table(bumped, future)

    with pytest.raises(ValueError, match="schema_version"):
        read_contact_depth_field(future)


def test_file_without_metadata_raises(tmp_path):
    bare = tmp_path / "bare.parquet"
    pq.write_table(
        pa.table({"frame_index": pa.array([0], type=pa.int32())}), bare
    )

    with pytest.raises(ValueError, match="no schema metadata"):
        read_contact_depth_field(bare)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_contact_depth_field(tmp_path / "absent.parquet")


# ---------------------------------------------------------------------------
# 3. Row accounting
# ---------------------------------------------------------------------------


def test_row_count_equals_the_summed_contact_vertex_count(tmp_path):
    series = _sample_series()
    out = tmp_path / "rec_contact_depth_field.parquet"

    write_contact_depth_field(series, out, source_recording="rec")
    frame, _ = read_contact_depth_field(out)

    assert len(frame) == sum(len(f.points) for f in series) == 12


def test_a_single_contact_vertex_writes_exactly_one_row(tmp_path):
    """The smallest legitimate frame is one row, not zero and not padded."""
    series = [_make_frame(41, 1.3667, [[1.0, 2.0, 3.0]], [-0.5])]
    out = tmp_path / "rec_contact_depth_field.parquet"

    write_contact_depth_field(series, out, source_recording="rec")
    frame, _ = read_contact_depth_field(out)

    assert len(frame) == 1
    assert frame.loc[0, "frame_index"] == 41
    assert frame.loc[0, "signed_depth_mm"] == -0.5


def test_non_contacting_frames_contribute_no_rows(tmp_path):
    """Long form is sparse: frame indices are not required to be contiguous."""
    series = [
        _make_frame(3, 0.1, [[0.0, 0.0, 0.0]], [-1.0]),
        _make_frame(97, 3.2333, [[1.0, 1.0, 1.0]], [-2.0]),
    ]
    out = tmp_path / "rec_contact_depth_field.parquet"

    write_contact_depth_field(series, out, source_recording="rec")
    frame, _ = read_contact_depth_field(out)

    assert frame["frame_index"].tolist() == [3, 97]


# ---------------------------------------------------------------------------
# 4. Fail-fast
# ---------------------------------------------------------------------------


def test_empty_series_raises_rather_than_writing_a_zero_row_file(tmp_path):
    out = tmp_path / "rec_contact_depth_field.parquet"

    with pytest.raises(ValueError, match="empty"):
        write_contact_depth_field([], out, source_recording="rec")

    assert not out.exists()


def test_none_series_raises(tmp_path):
    with pytest.raises(ValueError, match="None"):
        write_contact_depth_field(
            None, tmp_path / "rec.parquet", source_recording="rec"
        )


def test_unlabelled_frame_raises(tmp_path):
    """A frame with no identity cannot be joined to the CSV, so it is refused."""
    series = [_make_frame(None, None, [[0.0, 0.0, 0.0]], [-1.0])]

    with pytest.raises(ValueError, match="unlabelled"):
        write_contact_depth_field(
            series, tmp_path / "rec.parquet", source_recording="rec"
        )


def test_frame_missing_only_the_timestamp_raises(tmp_path):
    series = [_make_frame(7, None, [[0.0, 0.0, 0.0]], [-1.0])]

    with pytest.raises(ValueError, match="unlabelled"):
        write_contact_depth_field(
            series, tmp_path / "rec.parquet", source_recording="rec"
        )


def test_misaligned_frame_raises(tmp_path):
    series = [_make_frame(0, 0.0, np.zeros((3, 3)), np.zeros(2))]

    with pytest.raises(ValueError, match="index-aligned"):
        write_contact_depth_field(
            series, tmp_path / "rec.parquet", source_recording="rec"
        )


def test_frame_with_zero_contact_vertices_raises(tmp_path):
    """A non-contacting frame belongs absent from the series, not present-and-empty."""
    series = [_make_frame(0, 0.0, np.zeros((0, 3)), np.zeros(0))]

    with pytest.raises(ValueError, match="zero contact vertices"):
        write_contact_depth_field(
            series, tmp_path / "rec.parquet", source_recording="rec"
        )


def test_blank_source_recording_raises(tmp_path):
    with pytest.raises(ValueError, match="source_recording"):
        write_contact_depth_field(
            _sample_series(), tmp_path / "rec.parquet", source_recording=""
        )


def test_a_failed_write_leaves_no_half_file(tmp_path, monkeypatch):
    """A mid-write failure must not leave a file a later run would accept."""
    out = tmp_path / "rec_contact_depth_field.parquet"

    def _explode(*args, **kwargs):
        raise PermissionError("simulated mid-write failure")

    monkeypatch.setattr(pq, "write_table", _explode)

    with pytest.raises(PermissionError):
        write_contact_depth_field(_sample_series(), out, source_recording="rec")

    assert not out.exists()
    assert list(tmp_path.iterdir()) == []


def test_a_failed_write_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    out = tmp_path / "rec_contact_depth_field.parquet"
    write_contact_depth_field(_sample_series(), out, source_recording="rec")
    before = out.read_bytes()

    def _explode(*args, **kwargs):
        raise PermissionError("simulated mid-write failure")

    monkeypatch.setattr(pq, "write_table", _explode)
    with pytest.raises(PermissionError):
        write_contact_depth_field(_sample_series(), out, source_recording="rec")

    assert out.read_bytes() == before


def test_overwriting_replaces_wholesale_and_never_appends(tmp_path):
    out = tmp_path / "rec_contact_depth_field.parquet"
    write_contact_depth_field(_sample_series(), out, source_recording="rec")

    replacement = [_make_frame(0, 0.0, [[0.0, 0.0, 0.0]], [-1.0])]
    write_contact_depth_field(replacement, out, source_recording="rec")

    frame, _ = read_contact_depth_field(out)
    assert len(frame) == 1


# ---------------------------------------------------------------------------
# 5. Idempotency — the first-run trap
# ---------------------------------------------------------------------------
# The task lists both artifacts in ``output_paths``.  These assert the two
# behaviours that decision buys, directly on the utility the task calls, so a
# regression shows up without needing a recording.


def _touch(path: Path, mtime: float) -> Path:
    path.write_text("x")
    os.utime(path, (mtime, mtime))
    return path


def test_a_missing_sidecar_forces_reprocessing(tmp_path):
    """The first-run trap: a current CSV with no sidecar must not skip."""
    from utils.should_process_task import should_process_task

    source = _touch(tmp_path / "hand_motion.npz", 1000.0)
    csv = _touch(tmp_path / "rec_contact_and_kinematic_data.csv", 2000.0)
    parquet = tmp_path / "rec_contact_depth_field.parquet"

    assert should_process_task(
        output_paths=[csv, parquet], input_paths=[source], force=False
    )

    _touch(parquet, 2000.0)
    assert not should_process_task(
        output_paths=[csv, parquet], input_paths=[source], force=False
    )


def test_a_sidecar_newer_than_a_stale_csv_cannot_mask_the_staleness(tmp_path):
    """Staleness compares the newest input to the *oldest* output."""
    from utils.should_process_task import should_process_task

    csv = _touch(tmp_path / "rec_contact_and_kinematic_data.csv", 1000.0)
    source = _touch(tmp_path / "hand_motion.npz", 2000.0)
    parquet = _touch(tmp_path / "rec_contact_depth_field.parquet", 3000.0)

    assert should_process_task(
        output_paths=[csv, parquet], input_paths=[source], force=False
    )


def test_clean_task_outputs_removes_both_artifacts(tmp_path):
    from utils.should_process_task import clean_task_outputs

    csv = _touch(tmp_path / "rec_contact_and_kinematic_data.csv", 1000.0)
    parquet = _touch(tmp_path / "rec_contact_depth_field.parquet", 1000.0)

    clean_task_outputs([csv, parquet])

    assert not csv.exists()
    assert not parquet.exists()


# ---------------------------------------------------------------------------
# 6. The DataFrame writer
# ---------------------------------------------------------------------------
# ``write_contact_depth_field_table`` serialises rows a caller already holds,
# with metadata that caller states outright.  Its job is to change nothing and
# to refuse anything it would otherwise have to change, so these tests are
# about bit-identity and about what it rejects.


def _standard_metadata(**overrides) -> dict:
    """The six keys the frame writer emits, as a caller would carry them over."""
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "coordinate_space": COORDINATE_SPACE,
        "units": UNITS,
        "sign_convention": SIGN_CONVENTION,
        "source_recording": "rec",
        "produced_by": PRODUCED_BY,
    }
    metadata.update(overrides)
    return metadata


#: A reference forearm small enough to write ids against by hand.
_REFERENCE_VERTEX_COUNT = 5000


def _vertex_id_metadata(**overrides) -> dict:
    """Schema-v2 metadata for a table carrying ``vertex_id``."""
    metadata = _standard_metadata(
        coordinate_space="icp_registered",
        reference_ply="ST14-01_forearm_deduped.ply",
        reference_ply_vertex_count=str(_REFERENCE_VERTEX_COUNT),
        dedup_epsilon="0.5",
    )
    metadata.update(overrides)
    return metadata


def _vertex_id_table(vertex_ids=None) -> pd.DataFrame:
    """A four-row schema-v2 table: the six required columns then ``vertex_id``."""
    if vertex_ids is None:
        vertex_ids = [0, 17, 2499, _REFERENCE_VERTEX_COUNT - 1]
    n = len(vertex_ids)
    return pd.DataFrame(
        {
            "frame_index": np.arange(n, dtype=np.int32),
            "time_s": np.arange(n, dtype=np.float64) / 30.0,
            "x": np.linspace(-10.0, 10.0, n, dtype=np.float32),
            "y": np.linspace(0.0, 5.0, n, dtype=np.float32),
            "z": np.linspace(3.0, -3.0, n, dtype=np.float32),
            "signed_depth_mm": np.array(
                [-0.1234567890123456, -1.5, -12.345678901234567, -0.0009765625][:n],
                dtype=np.float64,
            ),
            "vertex_id": np.asarray(vertex_ids, dtype=np.int32),
        }
    )


def _written_sample_table(tmp_path: Path):
    """Read back a freshly written sidecar — the shape a downstream stage sees."""
    source = tmp_path / "rec_contact_depth_field.parquet"
    write_contact_depth_field(_sample_series(), source, source_recording="rec")
    frame, metadata = read_contact_depth_field(source)
    return frame, metadata


def test_table_writer_round_trips_a_reduced_table_bit_identically(tmp_path):
    """Drop a frame's rows, write the rest back, recover them unchanged."""
    frame, metadata = _written_sample_table(tmp_path)

    # A boolean mask leaves a gapped index; the writer must not care.
    reduced = frame[frame["frame_index"] != 1]
    assert 0 < len(reduced) < len(frame)

    out = tmp_path / "reduced.parquet"
    write_contact_depth_field_table(reduced, out, metadata=metadata)
    recovered, _ = read_contact_depth_field(out)

    assert list(recovered.columns) == list(COLUMN_DTYPES)
    for column, dtype in COLUMN_DTYPES.items():
        assert recovered[column].dtype == dtype, column
        # Not approximate: this task removes rows, it must not move a value.
        assert np.array_equal(
            recovered[column].to_numpy(), reduced[column].to_numpy()
        ), column

    assert recovered["frame_index"].tolist() == reduced["frame_index"].tolist()


def test_table_writer_preserves_float64_depth_and_float32_positions(tmp_path):
    """Values chosen so a float32 depth store would provably lose them."""
    depths = np.array(
        [-0.1234567890123456, -12.345678901234567, -1e-9, -199.99999999999997],
        dtype=np.float64,
    )
    positions = np.array(
        [[1.5, -2.25, 3.125]] * 4, dtype=np.float32
    )
    table = pd.DataFrame(
        {
            "frame_index": np.arange(4, dtype=np.int32),
            "time_s": np.arange(4, dtype=np.float64) / 30.0,
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "signed_depth_mm": depths,
        }
    )

    out = tmp_path / "precision.parquet"
    write_contact_depth_field_table(table, out, metadata=_standard_metadata())
    recovered, _ = read_contact_depth_field(out)

    assert recovered["signed_depth_mm"].dtype == np.dtype(np.float64)
    assert np.array_equal(recovered["signed_depth_mm"].to_numpy(), depths)
    # Guard against a vacuous pass: float32 really would have moved these.
    assert not np.array_equal(depths.astype(np.float32).astype(np.float64), depths)

    for axis in ("x", "y", "z"):
        assert recovered[axis].dtype == np.dtype(np.float32), axis
    assert np.array_equal(recovered[["x", "y", "z"]].to_numpy(), positions)


def test_table_writer_ignores_the_dataframe_index(tmp_path):
    """A filtered frame carries a gapped index; it must not reach the file."""
    frame, metadata = _written_sample_table(tmp_path)
    reduced = frame[frame["frame_index"] == 2]

    out = tmp_path / "reduced.parquet"
    write_contact_depth_field_table(reduced, out, metadata=metadata)

    assert pq.read_table(out).column_names == list(COLUMN_DTYPES)


def test_table_writer_metadata_round_trips_including_provenance_keys(tmp_path):
    """The standard six carry through, plus the keys merging adds."""
    frame, carried = _written_sample_table(tmp_path)
    metadata = {
        **carried,
        "pipeline_stage": "merging",
        "neural_quality_filtered": "true",
        "frames_dropped": "17",
    }

    out = tmp_path / "filtered.parquet"
    write_contact_depth_field_table(frame, out, metadata=metadata)
    _, recovered = read_contact_depth_field(out)

    assert recovered["schema_version"] == SCHEMA_VERSION
    assert recovered["coordinate_space"] == COORDINATE_SPACE
    assert recovered["units"] == UNITS
    assert recovered["sign_convention"] == SIGN_CONVENTION
    assert recovered["source_recording"] == "rec"
    assert recovered["produced_by"] == PRODUCED_BY
    assert recovered["pipeline_stage"] == "merging"
    assert recovered["neural_quality_filtered"] == "true"
    assert recovered["frames_dropped"] == "17"


def test_table_writer_output_is_self_describing_without_any_repo_import(tmp_path):
    frame, carried = _written_sample_table(tmp_path)
    out = tmp_path / "filtered.parquet"
    write_contact_depth_field_table(
        frame, out, metadata={**carried, "neural_quality_filtered": "true"}
    )

    decoded = {
        k.decode(): v.decode() for k, v in pq.read_table(out).schema.metadata.items()
    }
    assert decoded["units"] == "mm"
    assert decoded["coordinate_space"] == "kinect_space_1"
    assert decoded["sign_convention"] == "negative_is_penetrating"
    assert decoded["neural_quality_filtered"] == "true"


def test_table_writer_overwrites_wholesale_and_never_appends(tmp_path):
    frame, metadata = _written_sample_table(tmp_path)
    out = tmp_path / "filtered.parquet"

    write_contact_depth_field_table(frame, out, metadata=metadata)
    write_contact_depth_field_table(frame.iloc[:1], out, metadata=metadata)

    recovered, _ = read_contact_depth_field(out)
    assert len(recovered) == 1


def test_table_writer_failed_write_leaves_no_half_file(tmp_path, monkeypatch):
    """The atomic temp-file-plus-rename path is shared, not reimplemented."""
    frame, metadata = _written_sample_table(tmp_path)
    out = tmp_path / "filtered.parquet"

    def _explode(*args, **kwargs):
        raise PermissionError("simulated mid-write failure")

    monkeypatch.setattr(pq, "write_table", _explode)
    with pytest.raises(PermissionError):
        write_contact_depth_field_table(frame, out, metadata=metadata)

    assert not out.exists()
    assert not (tmp_path / "filtered.parquet.partial").exists()


# --- what the table writer refuses -----------------------------------------


def test_table_writer_rejects_wrong_column_order(tmp_path):
    frame, metadata = _written_sample_table(tmp_path)
    swapped = frame[["time_s", "frame_index", "x", "y", "z", "signed_depth_mm"]]

    out = tmp_path / "filtered.parquet"
    with pytest.raises(ValueError, match="wrong order"):
        write_contact_depth_field_table(swapped, out, metadata=metadata)

    assert not out.exists()


def test_table_writer_rejects_a_missing_column(tmp_path):
    frame, metadata = _written_sample_table(tmp_path)
    dropped = frame.drop(columns=["signed_depth_mm"])

    with pytest.raises(ValueError, match="missing=\\['signed_depth_mm'\\]"):
        write_contact_depth_field_table(
            dropped, tmp_path / "filtered.parquet", metadata=metadata
        )


def test_table_writer_rejects_an_arbitrary_extra_column(tmp_path):
    """Exactness is the feature: only the two declared layouts are accepted."""
    frame, metadata = _written_sample_table(tmp_path)
    widened = frame.assign(patch_area_mm2=np.ones(len(frame), dtype=np.float64))

    with pytest.raises(ValueError, match="unexpected=\\['patch_area_mm2'\\]"):
        write_contact_depth_field_table(
            widened, tmp_path / "filtered.parquet", metadata=metadata
        )


def test_table_writer_rejects_an_extra_column_alongside_vertex_id(tmp_path):
    """Widening for ``vertex_id`` did not open the schema to anything else."""
    table = _vertex_id_table().assign(
        patch_area_mm2=np.ones(4, dtype=np.float64)
    )

    with pytest.raises(ValueError, match="unexpected=\\['patch_area_mm2'\\]"):
        write_contact_depth_field_table(
            table, tmp_path / "widened.parquet", metadata=_vertex_id_metadata()
        )


def test_table_writer_rejects_a_downcast_depth_column(tmp_path):
    """The dtype that matters most: float32 depth breaks the CSV invariant."""
    frame, metadata = _written_sample_table(tmp_path)
    downcast = frame.assign(
        signed_depth_mm=frame["signed_depth_mm"].astype(np.float32)
    )

    out = tmp_path / "filtered.parquet"
    with pytest.raises(ValueError, match="signed_depth_mm"):
        write_contact_depth_field_table(downcast, out, metadata=metadata)

    assert not out.exists()


def test_table_writer_rejects_a_widened_frame_index_column(tmp_path):
    """int64 frame_index is the dtype a careless pandas round-trip produces."""
    frame, metadata = _written_sample_table(tmp_path)
    widened = frame.assign(frame_index=frame["frame_index"].astype(np.int64))

    with pytest.raises(ValueError, match="frame_index"):
        write_contact_depth_field_table(
            widened, tmp_path / "filtered.parquet", metadata=metadata
        )


def test_table_writer_rejects_an_empty_table(tmp_path):
    """Zero retained rows is an absent artifact, not a complete empty one."""
    frame, metadata = _written_sample_table(tmp_path)
    empty = frame.iloc[0:0]
    assert list(empty.columns) == list(COLUMN_DTYPES)

    out = tmp_path / "filtered.parquet"
    with pytest.raises(ValueError, match="empty"):
        write_contact_depth_field_table(empty, out, metadata=metadata)

    assert not out.exists()


def test_table_writer_rejects_none(tmp_path):
    with pytest.raises(ValueError, match="None"):
        write_contact_depth_field_table(
            None, tmp_path / "filtered.parquet", metadata=_standard_metadata()
        )


def test_table_writer_rejects_metadata_without_a_schema_version(tmp_path):
    frame, metadata = _written_sample_table(tmp_path)
    stripped = {k: v for k, v in metadata.items() if k != "schema_version"}

    out = tmp_path / "filtered.parquet"
    with pytest.raises(ValueError, match="schema_version"):
        write_contact_depth_field_table(frame, out, metadata=stripped)

    assert not out.exists()


def test_table_writer_rejects_an_unknown_schema_version(tmp_path):
    """Refused at write time, mirroring the reader's refusal at read time."""
    frame, metadata = _written_sample_table(tmp_path)

    out = tmp_path / "filtered.parquet"
    with pytest.raises(ValueError, match="schema_version"):
        write_contact_depth_field_table(
            frame, out, metadata={**metadata, "schema_version": "99"}
        )

    assert not out.exists()


def test_table_writer_rejects_non_string_metadata_values(tmp_path):
    """``frames_dropped=17`` must be stringified by the caller, deliberately."""
    frame, metadata = _written_sample_table(tmp_path)

    with pytest.raises(ValueError, match="frames_dropped"):
        write_contact_depth_field_table(
            frame,
            tmp_path / "filtered.parquet",
            metadata={**metadata, "frames_dropped": 17},
        )


# ---------------------------------------------------------------------------
# 7. Schema v2 — vertex_id and reference-PLY provenance
# ---------------------------------------------------------------------------
# Two things are being defended here.  Backward compatibility: ~99 v1 files
# exist in production and this reader is the only way anything opens them.  And
# exactness: widening the validator to a second layout must not have widened it
# to "anything with the right six columns somewhere in it".


def _write_v1_file(path: Path, *, metadata_overrides=None) -> Path:
    """Write a genuine v1 file — six columns, ``schema_version="1"`` — via bare pyarrow.

    Constructed here rather than by the current writer on purpose: the point is
    to prove the reader opens a file written before v2 existed, and a file the
    current writer produced would prove only that it can read itself.
    """
    metadata = {
        "schema_version": "1",
        "coordinate_space": "kinect_space_1",
        "units": "mm",
        "sign_convention": "negative_is_penetrating",
        "source_recording": "legacy_rec",
        "produced_by": "compute_somatosensory_characteristics",
    }
    metadata.update(metadata_overrides or {})

    schema = pa.schema(
        [
            pa.field("frame_index", pa.int32()),
            pa.field("time_s", pa.float64()),
            pa.field("x", pa.float32()),
            pa.field("y", pa.float32()),
            pa.field("z", pa.float32()),
            pa.field("signed_depth_mm", pa.float64()),
        ]
    ).with_metadata(metadata)

    table = pa.Table.from_arrays(
        [
            pa.array(np.array([3, 3, 97], dtype=np.int32), type=pa.int32()),
            pa.array(np.array([0.1, 0.1, 3.2333], dtype=np.float64), type=pa.float64()),
            pa.array(np.array([1.5, 2.5, 3.5], dtype=np.float32), type=pa.float32()),
            pa.array(np.array([-1.0, -2.0, -3.0], dtype=np.float32), type=pa.float32()),
            pa.array(np.array([0.25, 0.5, 0.75], dtype=np.float32), type=pa.float32()),
            pa.array(
                np.array([-0.1234567890123456, -2.0, -12.5], dtype=np.float64),
                type=pa.float64(),
            ),
        ],
        schema=schema,
    )
    pq.write_table(table, path)
    return path


# --- backward compatibility -------------------------------------------------


def test_schema_version_is_two_and_one_is_still_supported():
    assert SCHEMA_VERSION == "2"
    assert SUPPORTED_SCHEMA_VERSIONS == frozenset({"1", "2"})


def test_a_v1_file_still_reads(tmp_path):
    """The ~99 Space-1 artifacts on disk are v1 files; they must keep opening."""
    path = _write_v1_file(tmp_path / "legacy_contact_depth_field.parquet")

    frame, metadata = read_contact_depth_field(path)

    assert metadata["schema_version"] == "1"
    assert metadata["coordinate_space"] == "kinect_space_1"
    assert list(frame.columns) == list(COLUMN_DTYPES)
    for column, dtype in COLUMN_DTYPES.items():
        assert frame[column].dtype == dtype, column
    assert frame["frame_index"].tolist() == [3, 3, 97]
    assert frame["signed_depth_mm"].iloc[0] == -0.1234567890123456


def test_a_v1_table_can_still_be_rewritten_as_v1(tmp_path):
    """Merging carries ``schema_version`` through verbatim; that must still work."""
    path = _write_v1_file(tmp_path / "legacy.parquet")
    frame, metadata = read_contact_depth_field(path)

    out = tmp_path / "filtered.parquet"
    write_contact_depth_field_table(
        frame[frame["frame_index"] == 3],
        out,
        metadata={**metadata, "pipeline_stage": "merging"},
    )

    recovered, recovered_metadata = read_contact_depth_field(out)
    assert recovered_metadata["schema_version"] == "1"
    assert list(recovered.columns) == list(COLUMN_DTYPES)
    assert len(recovered) == 2


def test_an_unknown_schema_version_still_raises(tmp_path):
    """Widening the supported set to two versions did not widen it to all."""
    path = _write_v1_file(
        tmp_path / "future.parquet", metadata_overrides={"schema_version": "3"}
    )

    with pytest.raises(ValueError, match="schema_version"):
        read_contact_depth_field(path)


# --- the two legal layouts --------------------------------------------------


def test_v2_round_trips_without_vertex_id(tmp_path):
    """v2 is a widening, not a replacement: the narrow layout stays legal."""
    frame, metadata = _written_sample_table(tmp_path)
    assert metadata["schema_version"] == "2"
    assert VERTEX_ID_COLUMN not in frame.columns

    out = tmp_path / "narrow.parquet"
    write_contact_depth_field_table(frame, out, metadata=metadata)
    recovered, recovered_metadata = read_contact_depth_field(out)

    assert list(recovered.columns) == list(COLUMN_DTYPES)
    assert recovered_metadata["schema_version"] == "2"
    for column in COLUMN_DTYPES:
        assert np.array_equal(
            recovered[column].to_numpy(), frame[column].to_numpy()
        ), column


def test_v2_round_trips_with_vertex_id(tmp_path):
    table = _vertex_id_table()
    out = tmp_path / "projected.parquet"

    write_contact_depth_field_table(table, out, metadata=_vertex_id_metadata())
    recovered, metadata = read_contact_depth_field(out)

    assert list(recovered.columns) == list(ALL_COLUMN_DTYPES)
    for column, dtype in ALL_COLUMN_DTYPES.items():
        assert recovered[column].dtype == dtype, column
        assert np.array_equal(
            recovered[column].to_numpy(), table[column].to_numpy()
        ), column

    assert metadata["reference_ply"] == "ST14-01_forearm_deduped.ply"
    assert metadata["reference_ply_vertex_count"] == str(_REFERENCE_VERTEX_COUNT)
    assert metadata["dedup_epsilon"] == "0.5"
    assert metadata["coordinate_space"] == "icp_registered"


def test_a_v2_file_with_vertex_id_is_self_describing(tmp_path):
    """A bare pyarrow reader recovers the reference-PLY identity."""
    out = tmp_path / "projected.parquet"
    write_contact_depth_field_table(
        _vertex_id_table(), out, metadata=_vertex_id_metadata()
    )

    raw = pq.read_table(out)
    decoded = {k.decode(): v.decode() for k, v in raw.schema.metadata.items()}

    assert raw.column_names == list(ALL_COLUMN_DTYPES)
    assert raw.schema.field(VERTEX_ID_COLUMN).type == pa.int32()
    for key in REFERENCE_PLY_METADATA_KEYS:
        assert decoded[key]


def test_vertex_id_in_the_wrong_position_raises(tmp_path):
    """The optional column is a suffix, not a member of an unordered set."""
    table = _vertex_id_table()
    misordered = table[
        [
            "frame_index",
            "vertex_id",
            "time_s",
            "x",
            "y",
            "z",
            "signed_depth_mm",
        ]
    ]

    out = tmp_path / "misordered.parquet"
    with pytest.raises(ValueError, match="wrong order"):
        write_contact_depth_field_table(
            misordered, out, metadata=_vertex_id_metadata()
        )

    assert not out.exists()


def test_vertex_id_must_be_int32_not_int64(tmp_path):
    """int64 is what a careless pandas round-trip produces; it is refused."""
    table = _vertex_id_table()
    widened = table.assign(vertex_id=table["vertex_id"].astype(np.int64))

    out = tmp_path / "widened.parquet"
    with pytest.raises(ValueError, match="vertex_id"):
        write_contact_depth_field_table(widened, out, metadata=_vertex_id_metadata())

    assert not out.exists()


# --- vertex_id demands its provenance ---------------------------------------


@pytest.mark.parametrize("omitted", REFERENCE_PLY_METADATA_KEYS)
def test_vertex_id_without_its_full_provenance_raises(tmp_path, omitted):
    """All three keys or none: a partly-identified index is not partly useful."""
    metadata = {
        k: v for k, v in _vertex_id_metadata().items() if k != omitted
    }

    out = tmp_path / "unprovenanced.parquet"
    with pytest.raises(ValueError, match=omitted):
        write_contact_depth_field_table(_vertex_id_table(), out, metadata=metadata)

    assert not out.exists()


def test_vertex_id_under_schema_version_1_raises(tmp_path):
    """A stage that adds the column must restamp the version, not carry v1 through."""
    metadata = _vertex_id_metadata(schema_version="1")

    out = tmp_path / "mislabelled.parquet"
    with pytest.raises(ValueError, match="schema_version"):
        write_contact_depth_field_table(_vertex_id_table(), out, metadata=metadata)

    assert not out.exists()


def test_reading_a_v1_file_that_carries_vertex_id_raises(tmp_path):
    """The reader mirrors the writer: a file contradicting itself is refused."""
    schema = pa.schema(
        [
            pa.field("frame_index", pa.int32()),
            pa.field("time_s", pa.float64()),
            pa.field("x", pa.float32()),
            pa.field("y", pa.float32()),
            pa.field("z", pa.float32()),
            pa.field("signed_depth_mm", pa.float64()),
            pa.field("vertex_id", pa.int32()),
        ]
    ).with_metadata({"schema_version": "1", "units": "mm"})
    table = _vertex_id_table()
    contradictory = pa.Table.from_arrays(
        [pa.array(table[f.name].to_numpy(), type=f.type) for f in schema],
        schema=schema,
    )
    out = tmp_path / "contradictory.parquet"
    pq.write_table(contradictory, out)

    with pytest.raises(ValueError, match="contradicts its own declared schema"):
        read_contact_depth_field(out)


def test_a_file_missing_a_required_column_raises(tmp_path):
    schema = pa.schema([pa.field("frame_index", pa.int32())]).with_metadata(
        {"schema_version": "2"}
    )
    out = tmp_path / "truncated.parquet"
    pq.write_table(
        pa.Table.from_arrays(
            [pa.array(np.array([0], dtype=np.int32), type=pa.int32())], schema=schema
        ),
        out,
    )

    with pytest.raises(ValueError, match="required column"):
        read_contact_depth_field(out)


# --- the coordinate-space vocabulary ----------------------------------------


def test_every_declared_coordinate_space_is_writable(tmp_path):
    """The four names in the vocabulary are the four postprocessing frames."""
    assert COORDINATE_SPACES == frozenset(
        {"kinect_space_1", "icp_registered", "pca_calibrated", "rf_centered"}
    )

    frame, metadata = _written_sample_table(tmp_path)
    for space in sorted(COORDINATE_SPACES):
        out = tmp_path / f"{space}.parquet"
        write_contact_depth_field_table(
            frame, out, metadata={**metadata, "coordinate_space": space}
        )
        _, recovered = read_contact_depth_field(out)
        assert recovered["coordinate_space"] == space


def test_a_misspelled_coordinate_space_raises(tmp_path):
    """A typo'd space is worse than a rejected write, so the write is rejected."""
    frame, metadata = _written_sample_table(tmp_path)

    out = tmp_path / "typo.parquet"
    with pytest.raises(ValueError, match="coordinate_space"):
        write_contact_depth_field_table(
            frame, out, metadata={**metadata, "coordinate_space": "rf_centred"}
        )

    assert not out.exists()


def test_a_malformed_reference_ply_vertex_count_raises(tmp_path):
    out = tmp_path / "bad_count.parquet"
    with pytest.raises(ValueError, match="reference_ply_vertex_count"):
        write_contact_depth_field_table(
            _vertex_id_table(),
            out,
            metadata=_vertex_id_metadata(reference_ply_vertex_count="lots"),
        )


def test_a_malformed_dedup_epsilon_raises(tmp_path):
    out = tmp_path / "bad_epsilon.parquet"
    with pytest.raises(ValueError, match="dedup_epsilon"):
        write_contact_depth_field_table(
            _vertex_id_table(),
            out,
            metadata=_vertex_id_metadata(dedup_epsilon="0"),
        )


# --- validate_vertex_ids_against_reference ----------------------------------


def test_validation_passes_when_the_reference_matches():
    table = _vertex_id_table()
    validate_vertex_ids_against_reference(
        table,
        _vertex_id_metadata(),
        reference_vertex_count=_REFERENCE_VERTEX_COUNT,
    )


def test_validation_raises_on_a_vertex_count_mismatch():
    """The epsilon-drift guard: a renumbered forearm is a different mesh."""
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_vertex_ids_against_reference(
            _vertex_id_table(),
            _vertex_id_metadata(),
            reference_vertex_count=_REFERENCE_VERTEX_COUNT - 1,
            reference_description="forearm_rf_centered/ST14-01.ply",
        )


def test_the_mismatch_message_names_the_epsilon_and_the_ply():
    """The message must point at the cause, not merely at the symptom."""
    with pytest.raises(ValueError) as excinfo:
        validate_vertex_ids_against_reference(
            _vertex_id_table(),
            _vertex_id_metadata(),
            reference_vertex_count=4321,
            reference_description="forearm_rf_centered/ST14-01.ply",
        )

    message = str(excinfo.value)
    assert "0.5" in message
    assert "ST14-01_forearm_deduped.ply" in message
    assert "4321" in message
    assert str(_REFERENCE_VERTEX_COUNT) in message


def test_validation_raises_on_an_out_of_range_vertex_id():
    table = _vertex_id_table([0, 5, _REFERENCE_VERTEX_COUNT])

    with pytest.raises(ValueError, match="outside"):
        validate_vertex_ids_against_reference(
            table,
            _vertex_id_metadata(),
            reference_vertex_count=_REFERENCE_VERTEX_COUNT,
        )


def test_validation_raises_on_a_negative_vertex_id():
    table = _vertex_id_table([0, -1, 5])

    with pytest.raises(ValueError, match="outside"):
        validate_vertex_ids_against_reference(
            table,
            _vertex_id_metadata(),
            reference_vertex_count=_REFERENCE_VERTEX_COUNT,
        )


def test_the_last_valid_index_is_accepted_and_the_next_is_not():
    """``n_vertices - 1`` is in range; ``n_vertices`` is the classic off-by-one."""
    validate_vertex_ids_against_reference(
        _vertex_id_table([_REFERENCE_VERTEX_COUNT - 1]),
        _vertex_id_metadata(),
        reference_vertex_count=_REFERENCE_VERTEX_COUNT,
    )

    with pytest.raises(ValueError, match="outside"):
        validate_vertex_ids_against_reference(
            _vertex_id_table([_REFERENCE_VERTEX_COUNT]),
            _vertex_id_metadata(),
            reference_vertex_count=_REFERENCE_VERTEX_COUNT,
        )


def test_the_count_check_runs_before_the_range_check():
    """In-range ids against the wrong mesh are the dangerous case, so count first."""
    table = _vertex_id_table([0, 1, 2])

    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_vertex_ids_against_reference(
            table,
            _vertex_id_metadata(),
            reference_vertex_count=10,
        )


def test_validation_raises_when_the_table_has_no_vertex_id(tmp_path):
    frame, _ = _written_sample_table(tmp_path)

    with pytest.raises(ValueError, match="no 'vertex_id' column"):
        validate_vertex_ids_against_reference(
            frame,
            _vertex_id_metadata(),
            reference_vertex_count=_REFERENCE_VERTEX_COUNT,
        )


def test_validation_raises_when_the_metadata_records_no_vertex_count():
    metadata = {
        k: v
        for k, v in _vertex_id_metadata().items()
        if k != "reference_ply_vertex_count"
    }

    with pytest.raises(ValueError, match="reference_ply_vertex_count"):
        validate_vertex_ids_against_reference(
            _vertex_id_table(),
            metadata,
            reference_vertex_count=_REFERENCE_VERTEX_COUNT,
        )


def test_validation_raises_on_an_int64_vertex_id_column():
    table = _vertex_id_table()
    widened = table.assign(vertex_id=table["vertex_id"].astype(np.int64))

    with pytest.raises(ValueError, match="int32"):
        validate_vertex_ids_against_reference(
            widened,
            _vertex_id_metadata(),
            reference_vertex_count=_REFERENCE_VERTEX_COUNT,
        )


@pytest.mark.parametrize("bad_count", [0, -1])
def test_validation_raises_on_a_non_positive_reference_count(bad_count):
    with pytest.raises(ValueError, match="reference_vertex_count"):
        validate_vertex_ids_against_reference(
            _vertex_id_table(),
            _vertex_id_metadata(),
            reference_vertex_count=bad_count,
        )


def test_validation_raises_when_handed_something_other_than_a_count():
    """Purity: this helper takes a number, never a mesh or a path."""
    with pytest.raises(ValueError, match="must be an integer"):
        validate_vertex_ids_against_reference(
            _vertex_id_table(),
            _vertex_id_metadata(),
            reference_vertex_count="forearm.ply",
        )


def test_the_io_module_imports_no_geometry_engine():
    """The purity contract, asserted rather than trusted.

    2.4 takes a vertex *count* precisely so that validating a ``vertex_id``
    against a PLY does not drag a mesh loader behind the serialisation seam.
    """
    import inspect
    import re

    from preprocessing.motion_analysis.tactile_quantification.io import (
        contact_depth_field_io,
    )

    source = inspect.getsource(contact_depth_field_io)
    for banned in ("open3d", "prefect", "ruamel", "sklearn"):
        assert (
            re.search(rf"^\s*(import|from)\s+{banned}", source, re.MULTILINE) is None
        ), banned


# ---------------------------------------------------------------------------
# 8. Integration against a real recording
# ---------------------------------------------------------------------------
# Shares the reference bundle documented in ``test_contact_depth_field.py``:
# point SOCIAL_TOUCH_CONTACT_REFERENCE_DIR at a directory holding
# reference_contact.csv, hand_meshes.npz and forearm_mesh.obj.

_REFERENCE_DIR_ENV = "SOCIAL_TOUCH_CONTACT_REFERENCE_DIR"


def _reference_bundle():
    root = os.environ.get(_REFERENCE_DIR_ENV)
    if root is None:
        return None
    path = Path(root)
    required = {
        "csv": path / "reference_contact.csv",
        "hand": path / "hand_meshes.npz",
        "forearm": path / "forearm_mesh.obj",
    }
    missing = [str(p) for p in required.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{_REFERENCE_DIR_ENV} is set to {path} but these are missing: {missing}"
        )
    return required


def _run_reference_recording(bundle, tmp_path):
    """Recompute a real recording through processor → writer → reader."""
    o3d = pytest.importorskip("open3d")
    from preprocessing.motion_analysis.tactile_quantification.model.objects_interaction_processor import (
        ObjectsInteractionProcessor,
    )

    forearm = o3d.io.read_triangle_mesh(str(bundle["forearm"]))
    if not forearm.has_vertex_normals():
        forearm.compute_vertex_normals()

    payload = np.load(bundle["hand"])
    vertices_per_frame = payload["vertices"]
    triangles = o3d.utility.Vector3iVector(payload["triangles"])

    processor = ObjectsInteractionProcessor(reference_geometry=forearm)

    series = []
    per_frame_point_counts = {}
    for index, vertices in enumerate(vertices_per_frame):
        hand = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
            triangles,
        )
        _, _, field = processor.process_single_frame(
            current_mesh=hand, frame_index=index, time_s=index / 30.0
        )
        if field is not None:
            series.append(field)
            per_frame_point_counts[index] = len(field.points)

    out = tmp_path / "reference_contact_depth_field.parquet"
    write_contact_depth_field(series, out, source_recording="reference")
    frame, metadata = read_contact_depth_field(out)
    return frame, metadata, per_frame_point_counts


@pytest.fixture(scope="module")
def _reference_sidecar(tmp_path_factory):
    bundle = _reference_bundle()
    if bundle is None:
        pytest.skip(
            f"No reference recording bundle: set {_REFERENCE_DIR_ENV} to a directory "
            "containing reference_contact.csv, hand_meshes.npz and forearm_mesh.obj. "
            "The bundle is not committed (size and participant data)."
        )
    tmp_path = tmp_path_factory.mktemp("contact_depth_field")
    frame, metadata, counts = _run_reference_recording(bundle, tmp_path)
    # float_precision="round_trip" is mandatory, not decoration: the default
    # parser perturbs ~9% of the values in a real recording by one ULP, which
    # would fail the bit-identity assertions below on correctly written data.
    reference = pd.read_csv(bundle["csv"], float_precision="round_trip")
    return frame, metadata, counts, reference


def test_reference_frame_sets_agree_in_both_directions(_reference_sidecar):
    """No frame appears in one artifact and not the other."""
    frame, _, _, reference = _reference_sidecar

    sidecar_frames = set(frame["frame_index"].unique().tolist())
    csv_frames = set(
        reference.loc[reference["contact_detected"] == 1, "frame_index"].tolist()
    )

    assert sidecar_frames == csv_frames
    assert len(sidecar_frames) > 0


def test_reference_max_abs_depth_matches_the_csv_bit_identically(_reference_sidecar):
    frame, _, _, reference = _reference_sidecar

    recovered = frame.groupby("frame_index")["signed_depth_mm"].apply(
        lambda s: np.max(np.abs(s.to_numpy()))
    )
    expected = reference.set_index("frame_index").loc[
        recovered.index, "contact_depth"
    ]

    assert np.array_equal(
        recovered.to_numpy(dtype=np.float64),
        expected.to_numpy(dtype=np.float64),
    )


def test_reference_row_counts_match_the_parsed_contact_points(_reference_sidecar):
    """Aligned by count and index — never by coordinate value.

    ``serialize_contact_points`` quantises the CSV blob to ``%.1f`` while the
    sidecar carries full precision, so a value-wise comparison is meaningless.
    """
    frame, _, counts, reference = _reference_sidecar

    if "contact_points" not in reference.columns:
        pytest.skip("reference CSV carries no contact_points column")

    from preprocessing.forearm_extraction.registration.csv_spatial_transformer import (
        parse_contact_points,
    )

    indexed = reference.set_index("frame_index")
    sidecar_counts = frame.groupby("frame_index").size()

    for frame_index, count in sidecar_counts.items():
        parsed = parse_contact_points(indexed.loc[frame_index, "contact_points"])
        assert count == len(parsed) == counts[frame_index], frame_index


def test_reference_metadata_declares_the_coordinate_space(_reference_sidecar):
    _, metadata, _, _ = _reference_sidecar

    assert metadata["coordinate_space"] == "kinect_space_1"
    assert metadata["units"] == "mm"
    assert metadata["sign_convention"] == "negative_is_penetrating"
    assert metadata["schema_version"] == SCHEMA_VERSION
