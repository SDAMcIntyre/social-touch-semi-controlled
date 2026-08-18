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
    COLUMN_DTYPES,
    COORDINATE_SPACE,
    PRODUCED_BY,
    SCHEMA_VERSION,
    SIGN_CONVENTION,
    UNITS,
    read_contact_depth_field,
    write_contact_depth_field,
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
# 6. Integration against a real recording
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
