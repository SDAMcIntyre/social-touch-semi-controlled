"""Tests for the contact-depth-field adapter that feeds the Neural+Kinect viewer.

Everything here runs against synthetic parquet sidecars written to ``tmp_path``
with the production writer, so the tests exercise the real read path.  No Qt, no
VTK, no Kinect SDK: the adapter is deliberately a leaf that knows about a path,
a table and numpy arrays, which is exactly what makes it testable this way.

The assertions that matter are the ones about *not* inventing anything:

* the colour range is computed over the **whole recording**, never per frame;
* ``penetration_depth_mm`` is exactly ``-signed_depth_mm``, not approximately;
* a frame with no contact is absent, not present-and-empty;
* a **missing** sidecar produces an explicit, announceable absent state rather
  than a crash or a silently flat rendering;
* and — section 7 — building a block's *loader* reads nothing at all.  A batch
  is around a hundred blocks; the eager version of this code read every one of
  them before the viewer window opened, so the laziness is a guarded property,
  not an implementation detail.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow", reason="pyarrow is required for the parquet sidecar")

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from preprocessing.motion_analysis.tactile_quantification.io.contact_depth_field_io import (  # noqa: E402
    COORDINATE_SPACE,
    PRODUCED_BY,
    SCHEMA_VERSION,
    SIGN_CONVENTION,
    UNITS,
    write_contact_depth_field_table,
)
from merging.contact_depth_field_series import (  # noqa: E402
    BoundedContactDepthFieldCache,
    ContactDepthFieldResolution,
    ContactDepthFieldSeries,
    load_contact_depth_field_series,
    make_contact_depth_field_loader,
    resolve_contact_depth_field,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _metadata(**overrides: str) -> Dict[str, str]:
    """Standard sidecar metadata, matching what the merging filter writes."""
    base = {
        "schema_version": SCHEMA_VERSION,
        "coordinate_space": COORDINATE_SPACE,
        "units": UNITS,
        "sign_convention": SIGN_CONVENTION,
        "source_recording": "2022-06-15_ST14-01_semicontrolled_block-order01_kinect",
        "produced_by": PRODUCED_BY,
        "pipeline_stage": "merging",
        "neural_quality_filtered": "true",
        "frames_dropped": "0",
    }
    base.update(overrides)
    return base


def _table(rows: Sequence[tuple]) -> pd.DataFrame:
    """Build a schema-conforming long-form table from ``(frame, t, x, y, z, d)``."""
    frame = pd.DataFrame(
        {
            "frame_index": np.array([r[0] for r in rows], dtype=np.int32),
            "time_s": np.array([r[1] for r in rows], dtype=np.float64),
            "x": np.array([r[2] for r in rows], dtype=np.float32),
            "y": np.array([r[3] for r in rows], dtype=np.float32),
            "z": np.array([r[4] for r in rows], dtype=np.float32),
            "signed_depth_mm": np.array([r[5] for r in rows], dtype=np.float64),
        }
    )
    return frame


def _write(path: Path, rows: Sequence[tuple], **meta: str) -> Path:
    write_contact_depth_field_table(_table(rows), path, metadata=_metadata(**meta))
    return path


#: Three frames: 2 vertices, 3 vertices, 1 vertex.  Frame 1 carries no contact
#: and therefore contributes no rows at all.  The deepest penetration
#: (signed -9.5) is on frame 3, the shallowest (signed -0.25) on frame 0 — so a
#: per-frame colour scale and a global one are distinguishable.
_ROWS = [
    # frame, time_s,  x,     y,     z,     signed_depth_mm
    (0, 0.000, 1.0, 2.0, 3.0, -0.25),
    (0, 0.000, 1.5, 2.5, 3.5, -0.50),
    (2, 0.067, 10.0, 20.0, 30.0, -4.00),
    (2, 0.067, 11.0, 21.0, 31.0, -4.50),
    (2, 0.067, 12.0, 22.0, 32.0, 1.25),  # positive = above the surface
    (3, 0.100, 40.0, 50.0, 60.0, -9.50),
]


@pytest.fixture()
def sidecar(tmp_path: Path) -> Path:
    return _write(tmp_path / "block_contact_depth_field.parquet", _ROWS)


# ---------------------------------------------------------------------------
# 1. Frame indexing
# ---------------------------------------------------------------------------


def test_frame_returns_the_right_points_and_depths(sidecar: Path) -> None:
    """Each frame yields its own (N, 3) / (N,) pair, in file order."""
    series = load_contact_depth_field_series(sidecar)

    points, depths = series.frame(0)
    assert points.shape == (2, 3)
    assert depths.shape == (2,)
    assert np.array_equal(points, np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], np.float32))
    assert np.array_equal(depths, np.array([0.25, 0.50], np.float64))

    points, depths = series.frame(2)
    assert points.shape == (3, 3)
    assert depths.shape == (3,)
    assert np.array_equal(
        points,
        np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0], [12.0, 22.0, 32.0]], np.float32),
    )

    points, depths = series.frame(3)
    assert points.shape == (1, 3)
    assert depths.shape == (1,)


def test_frame_without_contact_is_absent_not_empty(sidecar: Path) -> None:
    """Frame 1 has no rows: it must be absent, never a zero-length field.

    Collapsing "no contact" into "a contact patch of zero vertices" is exactly
    the confusion the sidecar's zero-versus-absent rule exists to prevent.
    """
    series = load_contact_depth_field_series(sidecar)
    assert series.frame(1) is None
    assert 1 not in series.points_by_frame
    assert 1 not in series.penetration_depth_by_frame


def test_frame_beyond_the_recording_returns_none(sidecar: Path) -> None:
    series = load_contact_depth_field_series(sidecar)
    assert series.frame(9999) is None


def test_frame_accepts_a_numpy_integer_index(sidecar: Path) -> None:
    """The viewer's frame index arrives as a plain int, but numpy ints must work."""
    series = load_contact_depth_field_series(sidecar)
    assert series.frame(np.int64(0)) is not None


def test_counts_reflect_the_whole_recording(sidecar: Path) -> None:
    series = load_contact_depth_field_series(sidecar)
    assert series.frame_count == 3          # frames 0, 2, 3 — not frame 1
    assert series.vertex_count == len(_ROWS)


def test_points_and_depths_keep_the_schema_dtypes(sidecar: Path) -> None:
    """float32 positions, float64 depths — the sidecar's precision decision."""
    series = load_contact_depth_field_series(sidecar)
    points, depths = series.frame(2)
    assert points.dtype == np.float32
    assert depths.dtype == np.float64


def test_rows_out_of_frame_order_are_grouped_correctly(tmp_path: Path) -> None:
    """The schema promises a column set, not a row order."""
    shuffled = [_ROWS[i] for i in (5, 0, 3, 2, 1, 4)]
    series = load_contact_depth_field_series(
        _write(tmp_path / "shuffled.parquet", shuffled)
    )
    assert series.frame_count == 3
    assert series.frame(0)[0].shape == (2, 3)
    assert series.frame(2)[0].shape == (3, 3)
    assert series.frame(3)[0].shape == (1, 3)
    assert series.clim_penetration_mm == (-1.25, 9.5)


# ---------------------------------------------------------------------------
# 2. Penetration is exactly the negated signed depth
# ---------------------------------------------------------------------------


def test_penetration_is_exactly_the_negated_signed_depth(sidecar: Path) -> None:
    """Bit-exact, not approximate: negation of a float64 is lossless."""
    expected = -np.array([r[5] for r in _ROWS], dtype=np.float64)
    by_frame = {frame: [] for frame in (0, 2, 3)}
    for row, value in zip(_ROWS, expected):
        by_frame[row[0]].append(value)

    series = load_contact_depth_field_series(sidecar)
    for frame, values in by_frame.items():
        _, depths = series.frame(frame)
        wanted = np.array(values, dtype=np.float64)
        assert depths.tobytes() == wanted.tobytes()


def test_positive_signed_depth_becomes_negative_penetration(sidecar: Path) -> None:
    """A vertex above the surface must not be silently clamped to zero."""
    series = load_contact_depth_field_series(sidecar)
    _, depths = series.frame(2)
    assert depths[2] == -1.25


# ---------------------------------------------------------------------------
# 3. The colour range is global, not per frame
# ---------------------------------------------------------------------------


def test_clim_is_computed_over_the_whole_recording(sidecar: Path) -> None:
    """Deepest and shallowest come from different frames, so a per-frame scale
    could never produce this pair."""
    series = load_contact_depth_field_series(sidecar)
    assert series.clim_penetration_mm == (-1.25, 9.5)
    assert series.signed_depth_range_mm == (-9.5, 1.25)


def test_clim_is_not_any_single_frames_range(sidecar: Path) -> None:
    """Guard against a regression to per-frame autoscaling."""
    series = load_contact_depth_field_series(sidecar)
    for frame in (0, 2, 3):
        _, depths = series.frame(frame)
        per_frame = (float(np.min(depths)), float(np.max(depths)))
        assert series.clim_penetration_mm != per_frame


def test_clim_bounds_contain_every_depth_in_the_recording(sidecar: Path) -> None:
    series = load_contact_depth_field_series(sidecar)
    low, high = series.clim_penetration_mm
    for frame in series.points_by_frame:
        _, depths = series.frame(frame)
        assert float(np.min(depths)) >= low
        assert float(np.max(depths)) <= high


def test_clim_negation_swaps_the_signed_bounds(tmp_path: Path) -> None:
    """Negation reverses order, so (-max, -min), never (-min, -max)."""
    rows = [
        (0, 0.0, 0.0, 0.0, 0.0, -8.0),
        (1, 0.1, 0.0, 0.0, 0.0, -2.0),
    ]
    series = load_contact_depth_field_series(_write(tmp_path / "s.parquet", rows))
    assert series.signed_depth_range_mm == (-8.0, -2.0)
    assert series.clim_penetration_mm == (2.0, 8.0)


def test_a_single_depth_value_yields_a_degenerate_but_valid_clim(tmp_path: Path) -> None:
    rows = [(0, 0.0, 1.0, 2.0, 3.0, -3.0)]
    series = load_contact_depth_field_series(_write(tmp_path / "one.parquet", rows))
    assert series.clim_penetration_mm == (3.0, 3.0)


# ---------------------------------------------------------------------------
# 4. Absent sidecar — an explicit state, never a silent default
# ---------------------------------------------------------------------------


def test_missing_parquet_yields_an_explicit_absent_state(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist_contact_depth_field.parquet"
    resolution = resolve_contact_depth_field(missing)

    assert isinstance(resolution, ContactDepthFieldResolution)
    assert resolution.is_present is False
    assert resolution.series is None
    assert resolution.path == missing
    # The message is the whole point: absence must be announceable.
    assert str(missing) in resolution.message
    assert "DISABLED" in resolution.message


def test_missing_parquet_does_not_raise(tmp_path: Path) -> None:
    """The viewer must still run — just without depth colouring."""
    resolve_contact_depth_field(tmp_path / "nope.parquet")


def test_present_sidecar_reports_what_it_loaded(sidecar: Path) -> None:
    resolution = resolve_contact_depth_field(sidecar)
    assert resolution.is_present is True
    assert isinstance(resolution.series, ContactDepthFieldSeries)
    assert str(sidecar) in resolution.message
    assert "3 contact frames" in resolution.message


def test_direct_load_of_a_missing_file_raises(tmp_path: Path) -> None:
    """Only ``resolve_...`` tolerates absence; the loader itself is fail-fast."""
    with pytest.raises(FileNotFoundError):
        load_contact_depth_field_series(tmp_path / "nope.parquet")


def test_an_undecodable_sidecar_raises_rather_than_reading_as_absent(
    tmp_path: Path,
) -> None:
    """A file that exists but is not a depth field is broken, not absent.

    Treating it as absent would hide a corrupt artifact behind a
    plausible-looking picture.
    """
    broken = tmp_path / "broken_contact_depth_field.parquet"
    broken.write_bytes(b"not a parquet file")
    with pytest.raises(Exception):
        resolve_contact_depth_field(broken)


def test_a_sidecar_without_a_coordinate_space_raises(tmp_path: Path) -> None:
    """Points whose space is unknown must not be drawn on a guess."""
    path = tmp_path / "nospace.parquet"
    metadata = _metadata()
    del metadata["coordinate_space"]
    write_contact_depth_field_table(_table(_ROWS), path, metadata=metadata)
    with pytest.raises(ValueError, match="coordinate_space"):
        load_contact_depth_field_series(path)


def test_coordinate_space_is_carried_through(sidecar: Path) -> None:
    series = load_contact_depth_field_series(sidecar)
    assert series.coordinate_space == COORDINATE_SPACE


# ---------------------------------------------------------------------------
# 5. The DTO refuses inputs it could only render misleadingly
# ---------------------------------------------------------------------------


def test_series_rejects_an_empty_field() -> None:
    with pytest.raises(ValueError, match="empty"):
        ContactDepthFieldSeries(
            points_by_frame={},
            penetration_depth_by_frame={},
            clim_penetration_mm=(0.0, 1.0),
            signed_depth_range_mm=(-1.0, 0.0),
            coordinate_space=COORDINATE_SPACE,
        )


def test_series_rejects_misaligned_points_and_depths() -> None:
    with pytest.raises(ValueError, match="index-aligned"):
        ContactDepthFieldSeries(
            points_by_frame={0: np.zeros((3, 3), np.float32)},
            penetration_depth_by_frame={0: np.zeros((2,), np.float64)},
            clim_penetration_mm=(0.0, 1.0),
            signed_depth_range_mm=(-1.0, 0.0),
            coordinate_space=COORDINATE_SPACE,
        )


def test_series_rejects_a_frame_present_in_only_one_mapping() -> None:
    with pytest.raises(ValueError, match="different"):
        ContactDepthFieldSeries(
            points_by_frame={0: np.zeros((1, 3), np.float32)},
            penetration_depth_by_frame={1: np.zeros((1,), np.float64)},
            clim_penetration_mm=(0.0, 1.0),
            signed_depth_range_mm=(-1.0, 0.0),
            coordinate_space=COORDINATE_SPACE,
        )


def test_series_rejects_an_inverted_clim() -> None:
    with pytest.raises(ValueError, match="inverted"):
        ContactDepthFieldSeries(
            points_by_frame={0: np.zeros((1, 3), np.float32)},
            penetration_depth_by_frame={0: np.zeros((1,), np.float64)},
            clim_penetration_mm=(5.0, 1.0),
            signed_depth_range_mm=(-5.0, -1.0),
            coordinate_space=COORDINATE_SPACE,
        )


def test_resolution_rejects_an_empty_message(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="message"):
        ContactDepthFieldResolution(path=tmp_path / "x", series=None, message="")


# ---------------------------------------------------------------------------
# 6. Depth values are invariant under a rigid transform
# ---------------------------------------------------------------------------


def test_depths_are_untouched_by_the_registration_transform(sidecar: Path) -> None:
    """A rigid transform moves vertices; it cannot change a distance.

    The viewer transforms ``points`` and passes ``penetration_depth_mm``
    through.  This documents the invariant the viewer relies on: the adapter
    hands out the same array object for a frame every time it is asked, so a
    caller that transforms the points does not disturb the depths.
    """
    series = load_contact_depth_field_series(sidecar)
    points, depths = series.frame(2)
    before = depths.tobytes()

    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    translation = np.array([100.0, -50.0, 7.5], dtype=np.float64)
    moved = points.astype(np.float64) @ rotation.T + translation

    assert not np.array_equal(moved.astype(np.float32), points)
    assert series.frame(2)[1].tobytes() == before


# ---------------------------------------------------------------------------
# 7. Laziness — building a loader must read nothing
#
# This section is the regression guard for the bug it was written after: the
# pipeline resolved every block's depth field while assembling the batch, so 99
# blocks of 10^5-10^6 vertices were read before the viewer window appeared.  The
# fix moved the read behind a zero-argument callable.  Without these tests that
# regression is invisible — the eager version produced identical pictures.
# ---------------------------------------------------------------------------


class _CountingReporter:
    """Collects resolution messages and counts how often it was called."""

    def __init__(self) -> None:
        self.messages: List[str] = []

    def __call__(self, message: str) -> None:
        self.messages.append(message)

    @property
    def calls(self) -> int:
        return len(self.messages)


def test_building_many_loaders_reads_nothing(tmp_path: Path) -> None:
    """N loaders, zero reads. The headline invariant of the lazy contract."""
    cache = BoundedContactDepthFieldCache(maxsize=2)

    paths = [_write(tmp_path / f"block_{i:02d}.parquet", _ROWS) for i in range(20)]
    reporter = _CountingReporter()
    loaders = [make_contact_depth_field_loader(p, reporter, cache) for p in paths]

    assert len(loaders) == 20
    assert reporter.calls == 0, "building a loader must not resolve anything"
    assert len(cache) == 0, "building a loader must not populate the cache"


def test_loading_one_block_invokes_exactly_one_loader(tmp_path: Path) -> None:
    """Opening one block reads that block and no other."""
    cache = BoundedContactDepthFieldCache(maxsize=2)
    reporter = _CountingReporter()
    paths = [_write(tmp_path / f"b{i}.parquet", _ROWS) for i in range(5)]
    loaders = [make_contact_depth_field_loader(p, reporter, cache) for p in paths]

    series = loaders[2]()

    assert isinstance(series, ContactDepthFieldSeries)
    assert reporter.calls == 1
    assert str(paths[2]) in reporter.messages[0]
    assert len(cache) == 1


def test_the_same_loader_serves_two_specs_with_one_read(tmp_path: Path) -> None:
    """Plain and transformed specs share a loader object; the parquet is read once.

    The two viewers differ only in where the points are drawn.  Reading the
    field twice for one block would be pure waste.
    """
    cache = BoundedContactDepthFieldCache(maxsize=2)
    reporter = _CountingReporter()
    loader = make_contact_depth_field_loader(
        _write(tmp_path / "shared.parquet", _ROWS), reporter, cache
    )

    plain = loader()
    transformed = loader()

    assert plain is transformed, "a cache hit must return the same object"
    assert reporter.calls == 1, "the second call must not re-resolve"


def test_the_cache_is_bounded_so_a_batch_cannot_accumulate(tmp_path: Path) -> None:
    """Visiting many blocks must not leave them all resident.

    An unbounded memo would reach exactly the all-blocks-in-RAM state the lazy
    loader exists to prevent, just one block at a time.
    """
    cache = BoundedContactDepthFieldCache(maxsize=2)
    reporter = _CountingReporter()
    loaders = [
        make_contact_depth_field_loader(
            _write(tmp_path / f"c{i}.parquet", _ROWS), reporter, cache
        )
        for i in range(6)
    ]

    for loader in loaders:
        loader()

    assert len(cache) == 2
    assert cache.maxsize == 2
    assert reporter.calls == 6


def test_an_evicted_block_is_re_read_and_re_reported(tmp_path: Path) -> None:
    """Eviction costs a read; it must never quietly hand back a stale absence."""
    cache = BoundedContactDepthFieldCache(maxsize=2)
    reporter = _CountingReporter()
    first = make_contact_depth_field_loader(
        _write(tmp_path / "first.parquet", _ROWS), reporter, cache
    )
    second = make_contact_depth_field_loader(
        _write(tmp_path / "second.parquet", _ROWS), reporter, cache
    )
    third = make_contact_depth_field_loader(
        _write(tmp_path / "third.parquet", _ROWS), reporter, cache
    )

    first()
    second()
    third()          # evicts `first`
    assert reporter.calls == 3

    again = first()  # must be a genuine re-read, announced again
    assert isinstance(again, ContactDepthFieldSeries)
    assert reporter.calls == 4


def test_a_recently_used_block_is_not_evicted(tmp_path: Path) -> None:
    """The cache is LRU, not FIFO: re-opening a block keeps it resident."""
    cache = BoundedContactDepthFieldCache(maxsize=2)
    reporter = _CountingReporter()
    a = make_contact_depth_field_loader(
        _write(tmp_path / "a.parquet", _ROWS), reporter, cache
    )
    b = make_contact_depth_field_loader(
        _write(tmp_path / "b.parquet", _ROWS), reporter, cache
    )
    c = make_contact_depth_field_loader(
        _write(tmp_path / "c.parquet", _ROWS), reporter, cache
    )

    a()
    b()
    a()               # refreshes `a`, so `b` is now the least recent
    assert reporter.calls == 2

    c()               # evicts `b`, not `a`
    assert reporter.calls == 3
    a()               # still cached
    assert reporter.calls == 3


def test_a_lazily_loaded_absent_sidecar_is_reported_when_the_block_opens(
    tmp_path: Path,
) -> None:
    """Absence is announced at open time — later than before, but never dropped."""
    missing = tmp_path / "gone_contact_depth_field.parquet"
    cache = BoundedContactDepthFieldCache(maxsize=2)
    reporter = _CountingReporter()
    loader = make_contact_depth_field_loader(missing, reporter, cache)

    assert reporter.calls == 0          # nothing said at build time...
    assert loader() is None
    assert reporter.calls == 1          # ...everything said at open time
    assert "DISABLED" in reporter.messages[0]
    assert str(missing) in reporter.messages[0]


def test_an_absent_sidecar_is_cached_as_absent_not_re_resolved(tmp_path: Path) -> None:
    """``None`` is a resolved outcome, so it caches like any other."""
    cache = BoundedContactDepthFieldCache(maxsize=2)
    reporter = _CountingReporter()
    loader = make_contact_depth_field_loader(
        tmp_path / "absent.parquet", reporter, cache
    )

    assert loader() is None
    assert loader() is None
    assert reporter.calls == 1
    assert len(cache) == 1


def test_a_broken_sidecar_still_raises_through_the_loader(tmp_path: Path) -> None:
    """Laziness delays the read; it must not soften the failure."""
    broken = tmp_path / "broken.parquet"
    broken.write_bytes(b"not a parquet file")
    cache = BoundedContactDepthFieldCache(maxsize=2)
    loader = make_contact_depth_field_loader(broken, _CountingReporter(), cache)

    with pytest.raises(Exception):
        loader()


def test_a_cache_that_can_hold_nothing_is_rejected() -> None:
    with pytest.raises(ValueError, match="maxsize"):
        BoundedContactDepthFieldCache(maxsize=0)


def test_the_loaded_series_is_identical_to_a_direct_load(tmp_path: Path) -> None:
    """Going through the loader must not change a single number."""
    path = _write(tmp_path / "same.parquet", _ROWS)
    direct = load_contact_depth_field_series(path)
    lazy = make_contact_depth_field_loader(
        path, _CountingReporter(), BoundedContactDepthFieldCache(maxsize=2)
    )()

    assert lazy.clim_penetration_mm == direct.clim_penetration_mm
    assert lazy.signed_depth_range_mm == direct.signed_depth_range_mm
    assert lazy.coordinate_space == direct.coordinate_space
    assert sorted(lazy.points_by_frame) == sorted(direct.points_by_frame)
    for frame in direct.points_by_frame:
        assert np.array_equal(lazy.frame(frame)[0], direct.frame(frame)[0])
        assert lazy.frame(frame)[1].tobytes() == direct.frame(frame)[1].tobytes()


# ---------------------------------------------------------------------------
# The Phase-5 DTO widening, seen from the merging side
# ---------------------------------------------------------------------------
#
# ``ContactDepthFieldSeries`` gained two optional fields so that the
# postprocessing stage viewer can colour a forearm PLY by ``vertex_id``.  The
# acceptance test for that widening is that every test above passes unmodified;
# these two state the property those tests only imply, so that a future change
# that starts populating the fields here — or that makes them required — fails
# loudly rather than by a distant assertion in another file.
#
# The merging sidecar is written before projection, so it never carries a
# ``vertex_id``.  Nothing in this viewer's path may begin to depend on one.


def test_a_merging_sidecar_carries_no_vertex_identity(sidecar: Path) -> None:
    """No ``vertex_id`` column, so both halves of the widening stay ``None``.

    Not a gap: vertex identity is assigned at the postprocessing projection
    stage, against a reference forearm this space knows nothing about.
    """
    series = load_contact_depth_field_series(sidecar)

    assert series.has_vertex_ids is False
    assert series.vertex_id_by_frame is None
    assert series.reference_ply_provenance is None


def test_the_widened_fields_are_optional_at_construction() -> None:
    """A series built without them is valid — that is what keeps this viewer working."""
    series = ContactDepthFieldSeries(
        points_by_frame={0: np.zeros((1, 3), np.float32)},
        penetration_depth_by_frame={0: np.zeros((1,), np.float64)},
        clim_penetration_mm=(0.0, 1.0),
        signed_depth_range_mm=(-1.0, 0.0),
        coordinate_space=COORDINATE_SPACE,
    )

    assert series.has_vertex_ids is False
