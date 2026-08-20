"""Tests for the contact-depth-field wiring behind the postprocessing stage viewer.

Three phases of ``render-contact-depth-field-in-postprocessing-viewer`` are
covered here.  Phase 1 makes the six postprocessing stages *carry* a depth-field
loader: nothing renders and, crucially, nothing reads.  Phase 2 adds the
stage-aware policy leaf that turns ``(stage, loader, path)`` into either a
space-validated series or an announced absence.  Phase 3 draws the field, and
contributes the one piece of its render path that is not rendering: the join
from a slider position to a Kinect frame.  These tests guard the properties that
make all three worth doing at all.

**Path pairing.**  A stage's sidecar is found by name, from the stage CSV the
resolver has already computed.  Four stages write their CSV under the merging
pipeline's own ``*_merged_data.csv`` name and two write it under the
``*_merged_data_pca-xyz.csv`` name that ``calibrate_pca_xyz`` forks to, so a
derivation that stripped a trailing suffix instead of substituting the marker
would pair stages 4 and 5 with the wrong file — or with nothing.

**Laziness.**  Six loaders are built before the viewer window exists, one per
dropdown entry, and only the stage the user actually opens may be read.  A
stage's sidecar is ~12 MB on disk and larger once parsed into per-frame arrays;
an eager version would read all six before the first pixel and would draw
byte-identical pictures, which is precisely why the laziness has to be asserted
rather than assumed.

**Space validation.**  Six stages span four coordinate spaces, and a field drawn
in the wrong one lands as a plausible-looking patch in the wrong place rather
than as an error.  ``ContactDepthFieldSeries.coordinate_space`` has always
carried the declared space "so a caller can refuse to draw a wrong-space field";
``resolve_stage_depth_field`` is the first caller that actually refuses, so every
mismatched (stage, space) pair is exercised here rather than trusted.

**The one legitimate second space.**  ``center_on_receptive_field`` copies its
input through unchanged when it can estimate no receptive-field centre, and
deliberately leaves the sidecar declaring ``pca_calibrated`` -- the space the
points are genuinely still in.  Stage 5 therefore accepts two spaces and only
two, and the obvious way to implement that is a widening that quietly reaches
every other stage as well, so section 4b tests the *asymmetry*: stage 4 must
still refuse ``rf_centered``, stages 1-3 must still refuse everything but
``icp_registered``, and the passthrough must be reported as one while the
translating case must not.

**The frame join.**  The sidecar is keyed by Kinect ``frame_index`` and the
viewer's slider walks row positions; the merged CSV is upsampled ~33x to the
nerve rate and a block does not start at frame 0, so the two coincide almost
nowhere.  A positional join is the one shortcut here that fails *silently* — it
animates smoothly, in the right place, showing the wrong frames — so the join is
a pure function in the leaf rather than an expression inside the widget, and it
is tested against frame indices that are deliberately non-contiguous and unequal
to row position.

Everything here runs against synthetic parquet sidecars written to ``tmp_path``
with the production writer.  No Qt, no VTK, no Open3D — the read path, the path
resolver and the policy leaf are all leaves, and that is what keeps this file
headless.  ``conftest.py`` stubs the ``postprocessing.gui`` package root, whose
``__init__`` imports the four viewers, so that the Qt-free leaf inside it can be
imported on its own.
"""

from __future__ import annotations

import ast
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
    COORDINATE_SPACE_ICP_REGISTERED,
    COORDINATE_SPACE_KINECT_1,
    COORDINATE_SPACE_PCA_CALIBRATED,
    COORDINATE_SPACE_RF_CENTERED,
    PRODUCED_BY,
    REFERENCE_PLY_METADATA_KEYS,
    SCHEMA_VERSION,
    SIGN_CONVENTION,
    UNITS,
    VERTEX_ID_COLUMN,
    write_contact_depth_field_table,
)
from postprocessing.depth_field_stage_io import depth_field_path_for_csv  # noqa: E402
from merging.contact_depth_field_series import (  # noqa: E402
    BoundedContactDepthFieldCache,
    ContactDepthFieldSeries,
    load_contact_depth_field_series,
    make_contact_depth_field_loader,
)
from postprocessing.gui.stage_depth_field import (  # noqa: E402
    ACCEPTED_SPACES_BY_STAGE,
    CANONICAL_SPACE_BY_STAGE,
    FOREARM_DEPTH_SCALAR_NAME,
    FRAME_INDEX_COLUMN,
    KINECT_ANCHOR_COLUMN,
    PASSTHROUGH_SPACE_BY_STAGE,
    PRODUCING_TASK_BY_STAGE,
    STAGE_LABELS,
    StageDepthField,
    depth_frame_at_position,
    forearm_depth_scalars,
    kinect_anchor_rows,
    kinect_frame_at_position,
    kinect_frame_indices,
    resolve_stage_depth_field,
)


# ---------------------------------------------------------------------------
# The six stages, as ``resolve_stage_paths`` lays them out on disk
# ---------------------------------------------------------------------------

_SESSION_ID = "2022-06-15_ST14-02"
_BLOCK_ID = "block-order01"

_RAW_CSV_NAME = f"{_SESSION_ID}_semicontrolled_{_BLOCK_ID}_merged_data.csv"
_PCA_CSV_NAME = f"{_SESSION_ID}_semicontrolled_{_BLOCK_ID}_merged_data_pca-xyz.csv"

#: ``(stage_index, output_directory, csv_name, declared_coordinate_space)``.
#:
#: Stage 0 reads ``blocks_merged/``, which is written *before*
#: ``filter_contact_depth_field_by_neural_quality`` — that task's sidecar lands
#: in ``blocks_filtered/``, a directory the dropdown does not show.  Stage 0
#: therefore has no sidecar of its own, and its space entry is ``None``: it is
#: an expected absence, not a gap to be papered over by pointing the stage at
#: another directory's file.
_STAGES: Sequence[tuple] = (
    (0, "blocks_merged", _RAW_CSV_NAME, None),
    (1, "blocks_registered", _RAW_CSV_NAME, COORDINATE_SPACE_ICP_REGISTERED),
    (2, "blocks_deduped", _RAW_CSV_NAME, COORDINATE_SPACE_ICP_REGISTERED),
    (3, "blocks_projected", _RAW_CSV_NAME, COORDINATE_SPACE_ICP_REGISTERED),
    (4, "blocks_pca_calibrated", _PCA_CSV_NAME, COORDINATE_SPACE_PCA_CALIBRATED),
    (5, "blocks_rf_centered", _PCA_CSV_NAME, COORDINATE_SPACE_RF_CENTERED),
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _metadata(coordinate_space: str) -> Dict[str, str]:
    """Sidecar metadata for one postprocessing stage."""
    return {
        "schema_version": SCHEMA_VERSION,
        "coordinate_space": coordinate_space,
        "units": UNITS,
        "sign_convention": SIGN_CONVENTION,
        "source_recording": f"{_SESSION_ID}_semicontrolled_{_BLOCK_ID}_kinect",
        "produced_by": PRODUCED_BY,
        "pipeline_stage": "postprocessing",
        "neural_quality_filtered": "true",
        "frames_dropped": "0",
    }


def _table(rows: Sequence[tuple]) -> pd.DataFrame:
    """Build a schema-conforming long-form table from ``(frame, t, x, y, z, d)``."""
    return pd.DataFrame(
        {
            "frame_index": np.array([r[0] for r in rows], dtype=np.int32),
            "time_s": np.array([r[1] for r in rows], dtype=np.float64),
            "x": np.array([r[2] for r in rows], dtype=np.float32),
            "y": np.array([r[3] for r in rows], dtype=np.float32),
            "z": np.array([r[4] for r in rows], dtype=np.float32),
            "signed_depth_mm": np.array([r[5] for r in rows], dtype=np.float64),
        }
    )


#: Frame indices are deliberately non-contiguous and unequal to row position —
#: the CSV the viewer joins against is upsampled to the nerve rate, so a
#: positional join would look plausible and be wrong.
_ROWS = [
    # frame, time_s,  x,     y,     z,     signed_depth_mm
    (7, 0.233, 1.0, 2.0, 3.0, -0.25),
    (7, 0.233, 1.5, 2.5, 3.5, -0.50),
    (19, 0.633, 10.0, 20.0, 30.0, -4.00),
    (23, 0.766, 40.0, 50.0, 60.0, -9.50),
]


def _stage_csv_paths(base: Path) -> List[Path]:
    """The six stage CSV paths, laid out as ``resolve_stage_paths`` lays them out."""
    return [base / directory / name for _, directory, name, _ in _STAGES]


def _write_stage_sidecars(base: Path) -> List[Path]:
    """Write a real sidecar next to every *sidecar-bearing* stage CSV.

    Stage 0 is skipped on purpose; see ``_STAGES``.

    Returns:
        The six sidecar paths, whether or not a file was written at each.
    """
    sidecars = [depth_field_path_for_csv(csv) for csv in _stage_csv_paths(base)]
    for (_, _, _, space), sidecar in zip(_STAGES, sidecars):
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        if space is None:
            continue
        write_contact_depth_field_table(
            _table(_ROWS), sidecar, metadata=_metadata(space)
        )
    return sidecars


class _CountingReporter:
    """Collects resolution messages and counts how often it was called."""

    def __init__(self) -> None:
        self.messages: List[str] = []

    def __call__(self, message: str) -> None:
        self.messages.append(message)

    @property
    def calls(self) -> int:
        return len(self.messages)


# ---------------------------------------------------------------------------
# 1. Path pairing — each stage CSV finds its own sidecar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage_idx, directory, csv_name, _space", _STAGES)
def test_each_stage_csv_pairs_with_a_sibling_sidecar(
    tmp_path: Path, stage_idx: int, directory: str, csv_name: str, _space
) -> None:
    """The sidecar sits beside its CSV, in that stage's own output directory."""
    csv_path = tmp_path / directory / csv_name

    sidecar = depth_field_path_for_csv(csv_path)

    assert sidecar.parent == csv_path.parent, (
        f"stage {stage_idx} must look for its sidecar in {directory}/, not elsewhere"
    )
    assert sidecar.suffix == ".parquet"
    assert "_contact_depth_field" in sidecar.name
    assert "_merged_data" not in sidecar.name


def test_the_pca_fork_keeps_its_own_suffix(tmp_path: Path) -> None:
    """Stages 4-5 carry ``_pca-xyz``; the marker is substituted, not stripped.

    Substituting the ``_merged_data`` marker is what keeps the derivation valid
    on the far side of ``calibrate_pca_xyz``.  A derivation that replaced a
    trailing suffix would hand stages 4 and 5 the pre-PCA sidecar name.
    """
    pca_sidecar = depth_field_path_for_csv(tmp_path / "blocks_rf_centered" / _PCA_CSV_NAME)
    raw_sidecar = depth_field_path_for_csv(tmp_path / "blocks_registered" / _RAW_CSV_NAME)

    assert pca_sidecar.name.endswith("_contact_depth_field_pca-xyz.parquet")
    assert raw_sidecar.name.endswith("_contact_depth_field.parquet")
    assert pca_sidecar.name != raw_sidecar.name


def test_all_six_sidecar_paths_are_distinct(tmp_path: Path) -> None:
    """Six stages, six sidecars. A collision would silently show one stage twice."""
    sidecars = [depth_field_path_for_csv(csv) for csv in _stage_csv_paths(tmp_path)]

    assert len(sidecars) == len(_STAGES)
    assert len(set(sidecars)) == len(_STAGES)


def test_a_non_conforming_csv_name_refuses_to_be_paired(tmp_path: Path) -> None:
    """The name is the join between the two artifacts; guessing is not on offer."""
    with pytest.raises(ValueError, match="_merged_data"):
        depth_field_path_for_csv(tmp_path / "blocks_registered" / "something_else.csv")


# ---------------------------------------------------------------------------
# 2. Laziness — building the six loaders reads nothing
# ---------------------------------------------------------------------------
#
# Per *block*.  The same guarantee at the scale the session-level viewer
# operates at — 99 blocks x 6 stages behind one bounded cache, plus the block's
# stage-path resolution itself, which loads a forearm point cloud — is in
# ``test_stage_selection.py`` section 5.  Both halves are asserted; neither
# follows from the other.


def test_building_the_six_stage_loaders_reads_nothing(tmp_path: Path) -> None:
    """The headline invariant: six stages wired, zero bytes read.

    Six loaders are built while the viewer window does not yet exist.  An eager
    version would read every stage of the block up front and draw exactly the
    same pictures, so this assertion is the only thing standing between the lazy
    contract and a silent regression back to eager loading.
    """
    sidecars = _write_stage_sidecars(tmp_path)
    cache = BoundedContactDepthFieldCache(maxsize=len(_STAGES))
    reporter = _CountingReporter()

    loaders = [make_contact_depth_field_loader(p, reporter, cache) for p in sidecars]

    assert len(loaders) == len(_STAGES)
    assert reporter.calls == 0, "building a stage loader must not resolve anything"
    assert len(cache) == 0, "building a stage loader must not populate the cache"


def test_opening_one_stage_reads_only_that_stage(tmp_path: Path) -> None:
    """The user opens the ICP-registered stage; the other five stay unread."""
    sidecars = _write_stage_sidecars(tmp_path)
    cache = BoundedContactDepthFieldCache(maxsize=len(_STAGES))
    reporter = _CountingReporter()
    loaders = [make_contact_depth_field_loader(p, reporter, cache) for p in sidecars]

    series = loaders[1]()

    assert isinstance(series, ContactDepthFieldSeries)
    assert series.coordinate_space == COORDINATE_SPACE_ICP_REGISTERED
    assert reporter.calls == 1
    assert str(sidecars[1]) in reporter.messages[0]
    assert len(cache) == 1


def test_returning_to_a_stage_costs_no_second_read(tmp_path: Path) -> None:
    """The six loaders share one cache, so a dropdown round-trip is one read each.

    ``STAGE_DEPTH_FIELD_CACHE_SIZE`` is sized to the number of stages precisely
    so that walking the dropdown and coming back does not re-read.
    """
    sidecars = _write_stage_sidecars(tmp_path)
    cache = BoundedContactDepthFieldCache(maxsize=len(_STAGES))
    reporter = _CountingReporter()
    loaders = [make_contact_depth_field_loader(p, reporter, cache) for p in sidecars]

    for loader in loaders[1:]:
        loader()
    revisited = [loader() for loader in loaders[1:]]

    assert reporter.calls == len(_STAGES) - 1, "a cache hit must not report a read"
    assert all(isinstance(s, ContactDepthFieldSeries) for s in revisited)
    assert len(cache) == len(_STAGES) - 1


def test_walking_all_six_stages_holds_at_most_the_cache_bound(tmp_path: Path) -> None:
    """Residency is a property of the cache size, not of how long the window is open."""
    sidecars = _write_stage_sidecars(tmp_path)
    bound = 2
    cache = BoundedContactDepthFieldCache(maxsize=bound)
    reporter = _CountingReporter()
    loaders = [make_contact_depth_field_loader(p, reporter, cache) for p in sidecars]

    for loader in loaders:
        loader()
        assert len(cache) <= bound

    # The evicted stage is re-read rather than silently returning a stale hit.
    loaders[1]()
    assert reporter.calls == len(_STAGES) + 1


# ---------------------------------------------------------------------------
# 3. Stage 0 — an expected absence, not a bug
# ---------------------------------------------------------------------------


def test_stage_zero_resolves_to_a_missing_sidecar(tmp_path: Path) -> None:
    """``blocks_merged/`` precedes the depth field; its sidecar must not exist.

    The Space-1 sidecar lives in ``blocks_filtered/``.  Pointing stage 0 there
    would show the user a field in a different coordinate space than the
    geometry beside it, so the pairing stays strictly within the stage's own
    directory and the absence is the correct outcome.
    """
    sidecars = _write_stage_sidecars(tmp_path)

    assert not sidecars[0].exists()
    assert sidecars[0].parent.name == "blocks_merged"
    assert all(s.exists() for s in sidecars[1:]), "stages 1-5 must have sidecars"


def test_stage_zero_loader_reports_an_announced_absence(tmp_path: Path) -> None:
    """Absent is ``None`` plus a message — never a silent flat rendering."""
    sidecars = _write_stage_sidecars(tmp_path)
    cache = BoundedContactDepthFieldCache(maxsize=len(_STAGES))
    reporter = _CountingReporter()
    loader = make_contact_depth_field_loader(sidecars[0], reporter, cache)

    assert loader() is None
    assert reporter.calls == 1
    assert reporter.messages[0].strip(), "an absent sidecar must announce itself"
    assert str(sidecars[0]) in reporter.messages[0]


# ---------------------------------------------------------------------------
# 4. The stage-aware leaf — space policy, absence, refusal
# ---------------------------------------------------------------------------


_SIDECAR_BEARING = [row for row in _STAGES if row[3] is not None]

#: Every (stage, declared space) pair that must be refused: each sidecar-bearing
#: stage crossed with every space it accepts *neither* as its own output nor as
#: a passthrough.  Derived from the production mapping rather than restated, so
#: that a stage gaining a second accepted space is subtracted from this matrix
#: automatically instead of turning into a spurious failure -- and so that a
#: stage gaining one it should not have keeps failing here.
_ALL_SPACES = (
    COORDINATE_SPACE_KINECT_1,
    COORDINATE_SPACE_ICP_REGISTERED,
    COORDINATE_SPACE_PCA_CALIBRATED,
    COORDINATE_SPACE_RF_CENTERED,
)

_MISMATCHES = [
    (stage_idx, wrong)
    for stage_idx, _, _, _expected in _SIDECAR_BEARING
    for wrong in _ALL_SPACES
    if wrong not in ACCEPTED_SPACES_BY_STAGE[stage_idx]
]


class _ExplodingLoader:
    """A loader that fails the test if it is ever called."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self):
        self.calls += 1
        raise AssertionError("this loader must never be consulted")


def _loader_and_path(tmp_path: Path, stage_idx: int):
    """``(loader, sidecar_path)`` for one stage, with all sidecars on disk."""
    sidecars = _write_stage_sidecars(tmp_path)
    cache = BoundedContactDepthFieldCache(maxsize=len(_STAGES))
    reporter = _CountingReporter()
    return (
        make_contact_depth_field_loader(sidecars[stage_idx], reporter, cache),
        sidecars[stage_idx],
    )


def test_the_expected_space_map_omits_stage_zero_and_covers_the_rest() -> None:
    """Membership in the map *is* the definition of a sidecar-bearing stage.

    Stage 0 is absent by construction, not by oversight; repairing the absence
    would point the stage at another directory's file, in another space.
    """
    assert 0 not in CANONICAL_SPACE_BY_STAGE
    assert 0 not in ACCEPTED_SPACES_BY_STAGE
    assert sorted(CANONICAL_SPACE_BY_STAGE) == [s[0] for s in _SIDECAR_BEARING]
    assert sorted(PRODUCING_TASK_BY_STAGE) == sorted(CANONICAL_SPACE_BY_STAGE)
    assert sorted(ACCEPTED_SPACES_BY_STAGE) == sorted(CANONICAL_SPACE_BY_STAGE)
    for stage_idx, _, _, expected in _SIDECAR_BEARING:
        assert CANONICAL_SPACE_BY_STAGE[stage_idx] == expected
        assert expected in ACCEPTED_SPACES_BY_STAGE[stage_idx]
    assert len(STAGE_LABELS) == len(_STAGES)


@pytest.mark.parametrize("stage_idx, _dir, _csv, expected_space", _SIDECAR_BEARING)
def test_each_stage_accepts_its_own_coordinate_space(
    tmp_path: Path, stage_idx: int, _dir: str, _csv: str, expected_space: str
) -> None:
    """The happy path: three stages share ``icp_registered``, two do not."""
    loader, sidecar = _loader_and_path(tmp_path, stage_idx)

    resolved = resolve_stage_depth_field(stage_idx, loader, sidecar)

    assert isinstance(resolved, StageDepthField)
    assert resolved.is_present
    assert resolved.series.coordinate_space == expected_space
    assert resolved.message.strip()
    assert STAGE_LABELS[stage_idx] in resolved.message


@pytest.mark.parametrize("stage_idx, declared_space", _MISMATCHES)
def test_a_wrong_space_field_is_refused(
    tmp_path: Path, stage_idx: int, declared_space: str
) -> None:
    """Both spaces and the file are named; "wrong space" alone is not actionable.

    A field in the wrong space carries correct depths at incorrect positions, so
    it renders as a plausible patch somewhere it does not belong — the one class
    of defect a picture cannot be relied on to reveal.
    """
    expected_space = CANONICAL_SPACE_BY_STAGE[stage_idx]
    _, directory, csv_name, _ = _STAGES[stage_idx]
    sidecar = depth_field_path_for_csv(tmp_path / directory / csv_name)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    write_contact_depth_field_table(
        _table(_ROWS), sidecar, metadata=_metadata(declared_space)
    )
    cache = BoundedContactDepthFieldCache(maxsize=len(_STAGES))
    loader = make_contact_depth_field_loader(sidecar, _CountingReporter(), cache)

    with pytest.raises(ValueError) as excinfo:
        resolve_stage_depth_field(stage_idx, loader, sidecar)

    message = str(excinfo.value)
    assert expected_space in message
    assert declared_space in message
    assert str(sidecar) in message
    assert STAGE_LABELS[stage_idx] in message


def test_stage_zero_is_absent_without_consulting_any_loader(tmp_path: Path) -> None:
    """``blocks_merged/`` precedes the depth field; reading to then refuse is worse.

    A loader is passed deliberately, and must not be called: there is no space
    stage 0's field could legitimately declare, so opening a file to reject it
    would only cost a read.
    """
    loader = _ExplodingLoader()

    resolved = resolve_stage_depth_field(0, loader, tmp_path / "unused.parquet")

    assert loader.calls == 0
    assert not resolved.is_present
    assert resolved.series is None
    assert "blocks_merged" in resolved.message
    assert STAGE_LABELS[1] in resolved.message, "point the user at the first real stage"


def test_stage_zero_needs_no_loader_at_all() -> None:
    """The stage the resolver wired nothing for behaves identically."""
    resolved = resolve_stage_depth_field(0, None, None)

    assert not resolved.is_present
    assert resolved.message.strip()


@pytest.mark.parametrize("stage_idx, _dir, _csv, _space", _SIDECAR_BEARING)
def test_a_missing_sidecar_is_an_announced_absence(
    tmp_path: Path, stage_idx: int, _dir: str, _csv: str, _space: str
) -> None:
    """Absent is ``None`` plus a message naming the task that would produce it."""
    sidecar = depth_field_path_for_csv(tmp_path / _dir / _csv)
    cache = BoundedContactDepthFieldCache(maxsize=len(_STAGES))
    loader = make_contact_depth_field_loader(sidecar, _CountingReporter(), cache)

    resolved = resolve_stage_depth_field(stage_idx, loader, sidecar)

    assert not resolved.is_present
    assert str(sidecar) in resolved.message
    assert PRODUCING_TASK_BY_STAGE[stage_idx] in resolved.message


def test_a_stage_wired_without_a_loader_reports_its_absence(tmp_path: Path) -> None:
    """``depth_field_loader is None`` means no field was wired — still announced."""
    sidecar = depth_field_path_for_csv(tmp_path / "blocks_registered" / _RAW_CSV_NAME)

    resolved = resolve_stage_depth_field(1, None, sidecar)

    assert not resolved.is_present
    assert PRODUCING_TASK_BY_STAGE[1] in resolved.message


def test_a_corrupt_sidecar_raises_rather_than_reading_as_absent(tmp_path: Path) -> None:
    """A present-but-undecodable artifact must not be laundered into absence.

    Falling back to flat colour here would hide a broken artifact behind a
    plausible picture — which is why this module has no ``except`` at all.
    """
    sidecar = depth_field_path_for_csv(tmp_path / "blocks_registered" / _RAW_CSV_NAME)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    # A readable parquet with the right columns but none of the sidecar's schema
    # metadata: the reader's documented corrupt case, and the one most likely to
    # occur in practice, since it is what any plain ``to_parquet`` produces.
    _table(_ROWS).to_parquet(sidecar, index=False)
    cache = BoundedContactDepthFieldCache(maxsize=len(_STAGES))
    loader = make_contact_depth_field_loader(sidecar, _CountingReporter(), cache)

    with pytest.raises(ValueError, match="schema_version"):
        resolve_stage_depth_field(1, loader, sidecar)


def test_a_file_that_is_not_parquet_at_all_also_raises(tmp_path: Path) -> None:
    """The other corrupt case: bytes that are not a parquet file."""
    sidecar = depth_field_path_for_csv(tmp_path / "blocks_registered" / _RAW_CSV_NAME)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_bytes(b"this is not a parquet file")
    cache = BoundedContactDepthFieldCache(maxsize=len(_STAGES))
    loader = make_contact_depth_field_loader(sidecar, _CountingReporter(), cache)

    with pytest.raises(ValueError):
        resolve_stage_depth_field(1, loader, sidecar)


@pytest.mark.parametrize("returned", [object(), "a string", 42, []])
def test_a_loader_returning_a_non_series_raises(tmp_path: Path, returned) -> None:
    """Drawing whatever this is would put unvalidated geometry in the scene."""
    sidecar = depth_field_path_for_csv(tmp_path / "blocks_registered" / _RAW_CSV_NAME)

    with pytest.raises(TypeError, match="ContactDepthFieldSeries"):
        resolve_stage_depth_field(1, lambda: returned, sidecar)


def test_a_loader_without_its_path_is_a_wiring_error() -> None:
    """The loader hides its path, so the path travels beside it — or not at all.

    Without it a mismatch or absence message could not name the file it is
    about, which is the difference between an actionable error and a puzzle.
    """
    with pytest.raises(ValueError, match="sidecar_path"):
        resolve_stage_depth_field(1, lambda: None, None)


@pytest.mark.parametrize("stage_idx", [-1, len(_STAGES), 99])
def test_an_unknown_stage_index_raises(tmp_path: Path, stage_idx: int) -> None:
    """A stage the dropdown cannot show is a wiring error, not a fieldless stage."""
    with pytest.raises(ValueError, match="names no postprocessing stage"):
        resolve_stage_depth_field(stage_idx, None, tmp_path / "unused.parquet")


def test_every_outcome_carries_a_non_empty_message(tmp_path: Path) -> None:
    """The message is the whole reason absence cannot be silent."""
    sidecars = _write_stage_sidecars(tmp_path)
    cache = BoundedContactDepthFieldCache(maxsize=len(_STAGES))
    reporter = _CountingReporter()
    loaders = [make_contact_depth_field_loader(p, reporter, cache) for p in sidecars]

    for stage_idx in range(len(_STAGES)):
        resolved = resolve_stage_depth_field(
            stage_idx, loaders[stage_idx], sidecars[stage_idx]
        )
        assert resolved.message.strip(), f"stage {stage_idx} must announce its outcome"


def test_a_stage_depth_field_refuses_an_empty_message() -> None:
    """The DTO enforces it, so no construction site can opt out."""
    with pytest.raises(ValueError, match="message is empty"):
        StageDepthField(series=None, message="   ")


def test_resolving_a_stage_reads_exactly_once(tmp_path: Path) -> None:
    """The leaf adds no read of its own: the loader is called once, or not at all."""
    sidecars = _write_stage_sidecars(tmp_path)
    cache = BoundedContactDepthFieldCache(maxsize=len(_STAGES))
    reporter = _CountingReporter()
    loader = make_contact_depth_field_loader(sidecars[5], reporter, cache)

    resolved = resolve_stage_depth_field(5, loader, sidecars[5])

    assert resolved.is_present
    assert reporter.calls == 1
    assert len(cache) == 1


# ---------------------------------------------------------------------------
# 4b. The legitimate second spaces -- and the asymmetry that keeps them few
# ---------------------------------------------------------------------------
#
# Two of the pipeline's transforms are conditional, and when the condition does
# not hold the producing task copies the CSV *and* the sidecar through
# byte-for-byte, leaving ``coordinate_space`` at whatever the input declared.
# Restamping would assert a transform that never happened, so in both cases the
# producer is right and the viewer is what has to accommodate it.
#
# * ``center_on_receptive_field`` cannot always estimate a receptive-field
#   centre.  When it cannot, stage 5's sidecar keeps ``pca_calibrated`` --
#   ``2022-06-14_ST13-01`` lands there on all four of its blocks.
# * ``apply_icp_registration`` applies a *schedule*, and the schedule can be
#   empty -- ``apply_transform_schedule_to_field`` documents that as "the
#   explicit passthrough the ICP stage takes when a block has no registration
#   snapshot", and the same branch is taken for a whole session with no
#   ``registration_transforms.json``.  Stage 1's sidecar then keeps
#   ``kinect_space_1``, and because dedup and projection move no points, stages
#   2 and 3 inherit it.  ``2022-06-22_ST18-01`` lands there on all sixteen of
#   its blocks: its ``blocks_registered/`` sidecar is bitwise identical to its
#   ``blocks_filtered/`` input -- 747209 rows in, 747209 out, max |delta|
#   0.000000 mm -- while ``2022-06-15_ST14-02`` moved by up to 2.050193 mm and
#   declares ``icp_registered``.
#
# The fix is a *per-stage* widening, and the obvious way to get it wrong is a
# map that quietly widens the rest.  Everything below is written to catch that:
# the accepted sets are asserted exactly, stage 4 is shown still refusing
# ``rf_centered`` (a field that moved past the stage being displayed as though
# it had not) and still refusing ``kinect_space_1``, stage 5 still refusing
# ``kinect_space_1``, and stages 1-3 still refusing both spaces ahead of them.


def test_the_accepted_sets_are_exactly_these_and_no_wider() -> None:
    """The accepted sets, stated exactly. Anything wider is the bug this guards."""
    icp_or_kinect = frozenset(
        {COORDINATE_SPACE_ICP_REGISTERED, COORDINATE_SPACE_KINECT_1}
    )
    assert ACCEPTED_SPACES_BY_STAGE[1] == icp_or_kinect
    assert ACCEPTED_SPACES_BY_STAGE[2] == icp_or_kinect
    assert ACCEPTED_SPACES_BY_STAGE[3] == icp_or_kinect
    assert ACCEPTED_SPACES_BY_STAGE[4] == frozenset({COORDINATE_SPACE_PCA_CALIBRATED})
    assert ACCEPTED_SPACES_BY_STAGE[5] == frozenset(
        {COORDINATE_SPACE_RF_CENTERED, COORDINATE_SPACE_PCA_CALIBRATED}
    )
    assert sorted(PASSTHROUGH_SPACE_BY_STAGE) == [1, 2, 3, 5]
    assert PASSTHROUGH_SPACE_BY_STAGE[1] == COORDINATE_SPACE_KINECT_1
    assert PASSTHROUGH_SPACE_BY_STAGE[2] == COORDINATE_SPACE_KINECT_1
    assert PASSTHROUGH_SPACE_BY_STAGE[3] == COORDINATE_SPACE_KINECT_1
    assert PASSTHROUGH_SPACE_BY_STAGE[5] == COORDINATE_SPACE_PCA_CALIBRATED
    # Stage 4 is the one stage with no passthrough branch in its producer, and
    # it accepts exactly one space.  This is the assertion a blanket widening
    # fails: a map that widened "every stage" would give stage 4 a second space
    # too.
    assert 4 not in PASSTHROUGH_SPACE_BY_STAGE
    for stage_idx, spaces in ACCEPTED_SPACES_BY_STAGE.items():
        assert len(spaces) == (1 if stage_idx == 4 else 2)


def test_no_stage_accepts_a_space_produced_after_it() -> None:
    """Every passthrough points backwards, never forwards.

    A passthrough names a space the field never *left*.  A space some later
    stage produces is the opposite -- a field that already moved past the stage
    being displayed -- and accepting one would draw the patch a transform away
    from the geometry beside it while calling it a passthrough.
    """
    for stage_idx, passthrough in PASSTHROUGH_SPACE_BY_STAGE.items():
        # Stages 1-3 share one canonical space, so "produced by a later stage"
        # has to exclude this stage's own -- otherwise stage 1's ``icp_registered``
        # would read as stage 2's output rather than as its own.
        own = CANONICAL_SPACE_BY_STAGE[stage_idx]
        ahead = {
            space
            for later_idx, space in CANONICAL_SPACE_BY_STAGE.items()
            if later_idx > stage_idx and space != own
        }
        assert passthrough not in ahead, (
            f"stage {stage_idx}'s passthrough '{passthrough}' is produced by a "
            "later stage"
        )


def test_kinect_space_reaches_exactly_the_registration_carrying_stages() -> None:
    """``kinect_space_1`` must not leak past the stages that can still be in it.

    The ICP passthrough is inherited by dedup and projection because neither
    moves a point.  PCA calibration always transforms, so nothing downstream of
    stage 3 can legitimately still be in Kinect space -- and a map that let it
    would show pre-registration geometry under a calibrated label.
    """
    carrying = {
        stage_idx
        for stage_idx, spaces in ACCEPTED_SPACES_BY_STAGE.items()
        if COORDINATE_SPACE_KINECT_1 in spaces
    }
    assert carrying == {1, 2, 3}


def _resolve_with_space(tmp_path: Path, stage_idx: int, declared_space: str):
    """Resolve one stage against a sidecar written in *declared_space*."""
    _, directory, csv_name, _ = _STAGES[stage_idx]
    sidecar = depth_field_path_for_csv(tmp_path / directory / csv_name)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    write_contact_depth_field_table(
        _table(_ROWS), sidecar, metadata=_metadata(declared_space)
    )
    cache = BoundedContactDepthFieldCache(maxsize=len(_STAGES))
    loader = make_contact_depth_field_loader(sidecar, _CountingReporter(), cache)
    return resolve_stage_depth_field(stage_idx, loader, sidecar), sidecar


def test_stage_five_accepts_the_translating_case(tmp_path: Path) -> None:
    """``rf_centered``: the transform ran, and nothing extra is announced."""
    resolved, _ = _resolve_with_space(tmp_path, 5, COORDINATE_SPACE_RF_CENTERED)

    assert resolved.is_present
    assert resolved.series.coordinate_space == COORDINATE_SPACE_RF_CENTERED
    assert resolved.is_passthrough is False
    assert resolved.passthrough_note is None
    assert "PASSTHROUGH" not in resolved.message


def test_stage_five_accepts_the_passthrough_case(tmp_path: Path) -> None:
    """``pca_calibrated`` in ``blocks_rf_centered/``: accepted, and drawable.

    The producer is right and this is why: no translation was applied, so the
    points really are still PCA calibrated, and the previous expectation refused
    an entire class of session (all four blocks of ``2022-06-14_ST13-01``) over
    a file that was telling the truth.
    """
    resolved, sidecar = _resolve_with_space(
        tmp_path, 5, COORDINATE_SPACE_PCA_CALIBRATED
    )

    assert resolved.is_present
    assert resolved.series.coordinate_space == COORDINATE_SPACE_PCA_CALIBRATED
    assert str(sidecar) in resolved.message


def test_the_passthrough_is_reported_as_one(tmp_path: Path) -> None:
    """Accepted is not the same as unremarked.

    Drawing ``pca_calibrated`` points under a label that says "RF Centered"
    without saying so is the same misstatement as restamping the file, made
    quieter.  The note names both spaces and the task that skipped its
    transform, and it is folded into ``message`` too, so a caller that surfaces
    only the message still surfaces the passthrough.
    """
    resolved, _ = _resolve_with_space(tmp_path, 5, COORDINATE_SPACE_PCA_CALIBRATED)

    assert resolved.is_passthrough is True
    note = resolved.passthrough_note
    assert note is not None and note.strip()
    assert COORDINATE_SPACE_PCA_CALIBRATED in note
    assert COORDINATE_SPACE_RF_CENTERED in note
    assert PRODUCING_TASK_BY_STAGE[5] in note
    assert STAGE_LABELS[5] in note
    assert note in resolved.message


@pytest.mark.parametrize(
    "declared_space", [COORDINATE_SPACE_ICP_REGISTERED, COORDINATE_SPACE_KINECT_1]
)
def test_stage_five_still_refuses_every_other_space(
    tmp_path: Path, declared_space: str
) -> None:
    """Two spaces, not "anything". The refusal keeps its quality of message."""
    with pytest.raises(ValueError) as excinfo:
        _resolve_with_space(tmp_path, 5, declared_space)

    message = str(excinfo.value)
    assert declared_space in message
    assert COORDINATE_SPACE_RF_CENTERED in message
    assert COORDINATE_SPACE_PCA_CALIBRATED in message
    assert STAGE_LABELS[5] in message


def test_stage_four_still_refuses_the_rf_centered_space(tmp_path: Path) -> None:
    """The asymmetry, in the direction that matters most.

    Stage 5's second space is ``pca_calibrated`` -- the space *before* it.  The
    reverse is not symmetric and must never become so: a ``rf_centered`` field
    in ``blocks_pca_calibrated/`` is a field that moved past the stage being
    displayed, and drawing it would put the patch a translation away from the
    geometry beside it.
    """
    with pytest.raises(ValueError) as excinfo:
        _resolve_with_space(tmp_path, 4, COORDINATE_SPACE_RF_CENTERED)

    message = str(excinfo.value)
    assert COORDINATE_SPACE_RF_CENTERED in message
    assert COORDINATE_SPACE_PCA_CALIBRATED in message
    assert STAGE_LABELS[4] in message


def test_stage_four_refuses_the_kinect_space(tmp_path: Path) -> None:
    """The ICP passthrough must stop at stage 3.

    PCA calibration always transforms -- ``calibrate_pca_xyz`` has no
    passthrough branch -- so a ``kinect_space_1`` field in
    ``blocks_pca_calibrated/`` is unregistered, uncalibrated geometry under a
    calibrated label, not a passthrough.
    """
    with pytest.raises(ValueError) as excinfo:
        _resolve_with_space(tmp_path, 4, COORDINATE_SPACE_KINECT_1)

    message = str(excinfo.value)
    assert COORDINATE_SPACE_KINECT_1 in message
    assert COORDINATE_SPACE_PCA_CALIBRATED in message
    assert STAGE_LABELS[4] in message


@pytest.mark.parametrize(
    "declared_space",
    [COORDINATE_SPACE_PCA_CALIBRATED, COORDINATE_SPACE_RF_CENTERED],
)
@pytest.mark.parametrize("stage_idx", [1, 2, 3])
def test_stages_one_to_three_still_refuse_every_space_ahead_of_them(
    tmp_path: Path, stage_idx: int, declared_space: str
) -> None:
    """Two spaces, not "anything".

    Accepting ``kinect_space_1`` on these three stages is a claim about a
    transform that did *not* run.  A field in a space produced *later* is the
    opposite claim, and must still be refused by all three.
    """
    with pytest.raises(ValueError) as excinfo:
        _resolve_with_space(tmp_path, stage_idx, declared_space)

    message = str(excinfo.value)
    assert COORDINATE_SPACE_ICP_REGISTERED in message
    assert COORDINATE_SPACE_KINECT_1 in message
    assert declared_space in message
    assert STAGE_LABELS[stage_idx] in message


@pytest.mark.parametrize("stage_idx", [1, 2, 3])
def test_stages_one_to_three_accept_the_registered_case(
    tmp_path: Path, stage_idx: int
) -> None:
    """``icp_registered``: the schedule ran, and nothing extra is announced.

    This is ``2022-06-15_ST14-02``, whose coordinates moved by up to 2.050193 mm
    through registration.
    """
    resolved, _ = _resolve_with_space(
        tmp_path, stage_idx, COORDINATE_SPACE_ICP_REGISTERED
    )

    assert resolved.is_present
    assert resolved.series.coordinate_space == COORDINATE_SPACE_ICP_REGISTERED
    assert resolved.is_passthrough is False
    assert resolved.passthrough_note is None
    assert "PASSTHROUGH" not in resolved.message


@pytest.mark.parametrize("stage_idx", [1, 2, 3])
def test_stages_one_to_three_accept_the_icp_passthrough_case(
    tmp_path: Path, stage_idx: int
) -> None:
    """``kinect_space_1`` in the three registration-carrying directories.

    The producer is right and this is why: the ICP schedule was empty, so
    nothing moved and the points really are still in Kinect space.  The previous
    expectation refused an entire session over three files that were telling the
    truth -- ``2022-06-22_ST18-01``, on all sixteen of its blocks, whose stage-1
    sidecar is bitwise identical to its ``blocks_filtered/`` input.
    """
    resolved, sidecar = _resolve_with_space(
        tmp_path, stage_idx, COORDINATE_SPACE_KINECT_1
    )

    assert resolved.is_present
    assert resolved.series.coordinate_space == COORDINATE_SPACE_KINECT_1
    assert str(sidecar) in resolved.message


@pytest.mark.parametrize("stage_idx", [1, 2, 3])
def test_the_icp_passthrough_is_reported_as_one(
    tmp_path: Path, stage_idx: int
) -> None:
    """Accepted is not the same as unremarked.

    Drawing unregistered points under a label that says "ICP Registered" without
    saying so is the same misstatement as restamping the file, made quieter.
    """
    resolved, _ = _resolve_with_space(tmp_path, stage_idx, COORDINATE_SPACE_KINECT_1)

    assert resolved.is_passthrough is True
    note = resolved.passthrough_note
    assert note is not None and note.strip()
    assert COORDINATE_SPACE_KINECT_1 in note
    assert COORDINATE_SPACE_ICP_REGISTERED in note
    assert PRODUCING_TASK_BY_STAGE[stage_idx] in note
    assert STAGE_LABELS[stage_idx] in note
    assert note in resolved.message


@pytest.mark.parametrize("stage_idx, _dir, _csv, expected", _SIDECAR_BEARING)
def test_a_stage_in_its_own_space_never_reports_a_passthrough(
    tmp_path: Path, stage_idx: int, _dir: str, _csv: str, expected: str
) -> None:
    """A stage resolving in its own space is never announced as a passthrough.

    ``_SIDECAR_BEARING`` carries each stage's *canonical* space, so this is the
    normally-transformed session on every stage at once -- including stages 1-3,
    where ``2022-06-15_ST14-02`` must stay unannounced.
    """
    resolved, _ = _resolve_with_space(tmp_path, stage_idx, expected)

    assert resolved.is_present
    assert resolved.is_passthrough is False
    assert resolved.passthrough_note is None


def test_a_passthrough_note_without_a_series_is_refused() -> None:
    """A passthrough is a claim about a loaded field's space, not about absence."""
    with pytest.raises(ValueError, match="without a series"):
        StageDepthField(series=None, message="absent", passthrough_note="note")


def test_a_blank_passthrough_note_is_refused(tmp_path: Path) -> None:
    """It exists only to be shown; a blank one announces the passthrough to nobody."""
    series = _rf_centered_series(tmp_path)

    with pytest.raises(ValueError, match="passthrough_note is blank"):
        StageDepthField(series=series, message="loaded", passthrough_note="  ")


# ---------------------------------------------------------------------------
# 5. The frame join — a slider position is not a frame index
# ---------------------------------------------------------------------------
#
# This is the section Phase 3.2 exists for.  The sidecar is keyed by Kinect
# ``frame_index``; the slider walks *row positions* of the viewer's frame-anchor
# table.  The merged CSV is upsampled ~33x to the nerve rate and a block does
# not begin at frame 0, so the two coincide almost nowhere — yet a positional
# join fails silently, producing a smooth animation of the wrong frames.  Every
# test below is written so that a positional join could only pass it by
# coincidence, and ``test_a_positional_join_would_give_a_different_answer``
# asserts that the coincidence has not crept in.

#: The Kinect frames the viewer shows, in slider order.  Non-contiguous (frames
#: were dropped) and offset from zero (the block starts mid-recording), so row
#: position *p* never equals ``frame_index[p]``.
_KINECT_FRAME_INDICES = [5, 7, 11, 19, 23, 24]

#: What each slider position must resolve to, given ``_ROWS`` above: frames 7,
#: 19 and 23 carry contact, the rest carry none.  Depths are penetration, i.e.
#: ``-signed_depth_mm``, negated exactly once by the adapter.
_EXPECTED_DEPTHS_BY_POSITION = {
    0: None,                # frame 5  — no contact
    1: [0.25, 0.50],        # frame 7  — two vertices
    2: None,                # frame 11 — no contact
    3: [4.00],              # frame 19
    4: [9.50],              # frame 23
    5: None,                # frame 24 — no contact
}


def _kinect_df(frame_indices: Sequence, **extra) -> pd.DataFrame:
    """A minimal stand-in for the viewer's ``_kinect_df``: one row per frame."""
    data = {
        "time_kinect": np.arange(len(frame_indices), dtype=np.float64) / 30.0,
        FRAME_INDEX_COLUMN: list(frame_indices),
    }
    data.update(extra)
    return pd.DataFrame(data)


def _rf_centered_series(tmp_path: Path) -> ContactDepthFieldSeries:
    """A real sidecar, written by the production writer, read by the adapter."""
    sidecar = depth_field_path_for_csv(_stage_csv_paths(tmp_path)[5])
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    write_contact_depth_field_table(
        _table(_ROWS), sidecar, metadata=_metadata(COORDINATE_SPACE_RF_CENTERED)
    )
    return load_contact_depth_field_series(sidecar)


def test_kinect_frame_indices_preserves_row_order() -> None:
    """Element *p* is the frame shown at slider position *p*; nothing is sorted."""
    indices = kinect_frame_indices(_kinect_df(_KINECT_FRAME_INDICES))

    assert indices.dtype == np.int64
    assert indices.tolist() == _KINECT_FRAME_INDICES


@pytest.mark.parametrize("position", sorted(_EXPECTED_DEPTHS_BY_POSITION))
def test_the_depths_at_a_position_are_those_of_that_rows_frame(
    tmp_path: Path, position: int
) -> None:
    """The whole point of Phase 3.2: join by ``frame_index``, never by position."""
    series = _rf_centered_series(tmp_path)
    indices = kinect_frame_indices(_kinect_df(_KINECT_FRAME_INDICES))

    resolved = depth_frame_at_position(series, indices, position)
    expected = _EXPECTED_DEPTHS_BY_POSITION[position]

    if expected is None:
        assert resolved is None, (
            f"slider position {position} shows Kinect frame "
            f"{_KINECT_FRAME_INDICES[position]}, which carries no contact"
        )
        return

    points, depths = resolved
    assert depths.tolist() == expected
    assert len(points) == len(expected)
    # Identical to going through the series with the row's own frame index.
    direct = series.frame(_KINECT_FRAME_INDICES[position])
    assert direct is not None
    np.testing.assert_array_equal(depths, direct[1])
    np.testing.assert_array_equal(points, direct[0])


def test_a_positional_join_would_give_a_different_answer(tmp_path: Path) -> None:
    """Guards the guard: the fixture must be able to tell the two joins apart.

    If the synthetic frame indices ever drifted into agreeing with row position,
    every test above would keep passing while testing nothing.
    """
    series = _rf_centered_series(tmp_path)
    indices = kinect_frame_indices(_kinect_df(_KINECT_FRAME_INDICES))

    assert not any(int(indices[p]) == p for p in range(len(indices))), (
        "no slider position may coincide with its frame index, or the join is "
        "untested"
    )

    by_position = [
        depth_frame_at_position(series, indices, p) for p in range(len(indices))
    ]
    positional = [series.frame(p) for p in range(len(indices))]

    assert any(entry is not None for entry in by_position)
    assert all(entry is None for entry in positional), (
        "a positional join happens to find contact here, so this fixture can no "
        "longer distinguish the correct join from the dangerous one"
    )


def test_a_kinect_table_without_a_frame_index_column_raises() -> None:
    """No positional fallback: an unjoinable table is fatal, not flat-coloured."""
    df = pd.DataFrame({"time_kinect": [0.0, 0.033], "contact_points": ["[]", "[]"]})

    with pytest.raises(ValueError, match=f"no '{FRAME_INDEX_COLUMN}' column"):
        kinect_frame_indices(df, "blocks_rf_centered/whatever.csv")


def test_the_unjoinable_error_names_the_artifact_and_its_columns() -> None:
    """An unjoinable CSV must be identifiable from the message alone."""
    df = pd.DataFrame({"time_kinect": [0.0], "contact_points": ["[]"]})

    with pytest.raises(ValueError) as excinfo:
        kinect_frame_indices(df, "blocks_rf_centered/block-order01.csv")

    message = str(excinfo.value)
    assert "blocks_rf_centered/block-order01.csv" in message
    assert "contact_points" in message


def test_a_missing_frame_index_value_raises() -> None:
    """A displayed row whose frame is unknown cannot be looked up."""
    df = _kinect_df([5.0, np.nan, 11.0])

    with pytest.raises(ValueError, match="missing or non-finite"):
        kinect_frame_indices(df)


def test_a_fractional_frame_index_raises() -> None:
    """Half a frame addresses nothing."""
    df = _kinect_df([5.0, 7.5, 11.0])

    with pytest.raises(ValueError, match="non-integral"):
        kinect_frame_indices(df)


@pytest.mark.parametrize("position", [-1, 6, 99])
def test_a_position_outside_the_stage_raises(tmp_path: Path, position: int) -> None:
    """Clamping would silently repeat an end frame."""
    series = _rf_centered_series(tmp_path)
    indices = kinect_frame_indices(_kinect_df(_KINECT_FRAME_INDICES))

    with pytest.raises(IndexError, match="outside the 6 frames"):
        depth_frame_at_position(series, indices, position)


def test_joining_without_a_series_raises() -> None:
    """Reaching the join with no field means an is-present check was skipped."""
    indices = kinect_frame_indices(_kinect_df(_KINECT_FRAME_INDICES))

    with pytest.raises(TypeError, match="without a depth field"):
        depth_frame_at_position(None, indices, 0)


def test_the_join_is_reachable_from_a_resolved_stage(tmp_path: Path) -> None:
    """End to end, headless: loader -> policy leaf -> frame join.

    This is the call sequence the viewer performs per frame, minus the actor
    update — which is the only part that needs a window.
    """
    sidecars = _write_stage_sidecars(tmp_path)
    cache = BoundedContactDepthFieldCache(maxsize=len(_STAGES))
    reporter = _CountingReporter()
    loader = make_contact_depth_field_loader(sidecars[5], reporter, cache)

    resolved = resolve_stage_depth_field(5, loader, sidecars[5])
    assert resolved.is_present

    indices = kinect_frame_indices(_kinect_df(_KINECT_FRAME_INDICES))
    drawn = [
        depth_frame_at_position(resolved.series, indices, p)
        for p in range(len(indices))
    ]

    assert [None if d is None else d[1].tolist() for d in drawn] == [
        _EXPECTED_DEPTHS_BY_POSITION[p] for p in range(len(indices))
    ]


# ---------------------------------------------------------------------------
# 6. The forearm join — vertex_id onto a PLY, or nothing at all
# ---------------------------------------------------------------------------
#
# Phase 5 paints the depth on the surface rather than only on the patch, and the
# join that does it is the one place in this feature where a wrong answer is
# invisible.  ``vertex_id`` indexes a *specific* forearm: a re-dedup at another
# epsilon renumbers every vertex, and nothing upstream notices because task
# idempotency is decided from file timestamps.  So the provenance check runs
# before the scatter, unconditionally, and a count disagreement raises rather
# than colouring a plausible-looking wrong surface.
#
# The other half is the missing/zero distinction: a vertex nobody touched is
# ``NaN``, never ``0.0``.  A grazing contact at the rim of the patch is a
# genuine 0.00 mm and must stay distinguishable from the untouched skin next to
# it.

#: The forearm the synthetic sidecars below are written against.
_REFERENCE_VERTEX_COUNT = 12
_REFERENCE_PLY_NAME = "forearm_deduped.ply"
_DEDUP_EPSILON = "0.001"

#: ``(frame, time_s, x, y, z, signed_depth_mm, vertex_id)``.
#:
#: Frame 7 addresses vertex 4 twice, which the projection stage explicitly
#: permits: it is a per-point nearest-neighbour lookup with no uniqueness
#: constraint, so two contact points may land on one vertex.  The two rows carry
#: different depths so that the tie-break is observable rather than assumed.
_VERTEX_ROWS = [
    (7, 0.233, 1.0, 2.0, 3.0, -0.25, 4),
    (7, 0.233, 1.5, 2.5, 3.5, -0.50, 4),
    (7, 0.233, 2.0, 3.0, 4.0, 0.00, 9),
    (19, 0.633, 10.0, 20.0, 30.0, -4.00, 0),
    (23, 0.766, 40.0, 50.0, 60.0, -9.50, 11),
]


def _vertex_metadata(coordinate_space: str, vertex_count: int) -> Dict[str, str]:
    """Stage metadata plus the provenance triple a ``vertex_id`` requires."""
    metadata = _metadata(coordinate_space)
    metadata.update(
        {
            "reference_ply": _REFERENCE_PLY_NAME,
            "reference_ply_vertex_count": str(vertex_count),
            "dedup_epsilon": _DEDUP_EPSILON,
        }
    )
    return metadata


def _vertex_table(rows: Sequence[tuple]) -> pd.DataFrame:
    """A schema-v2 table: the six required columns plus ``vertex_id``."""
    table = _table([r[:6] for r in rows])
    table[VERTEX_ID_COLUMN] = np.array([r[6] for r in rows], dtype=np.int32)
    return table


def _provenance() -> Dict[str, str]:
    """The triple as the series carries it, for direct DTO construction."""
    return {
        "reference_ply": _REFERENCE_PLY_NAME,
        "reference_ply_vertex_count": str(_REFERENCE_VERTEX_COUNT),
        "dedup_epsilon": _DEDUP_EPSILON,
    }


@pytest.fixture()
def projected_series(tmp_path: Path):
    """A stage-3 series loaded from a real sidecar that carries ``vertex_id``."""
    sidecar = tmp_path / "blocks_projected" / "field.parquet"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    write_contact_depth_field_table(
        _vertex_table(_VERTEX_ROWS),
        sidecar,
        metadata=_vertex_metadata(
            COORDINATE_SPACE_ICP_REGISTERED, _REFERENCE_VERTEX_COUNT
        ),
    )
    return load_contact_depth_field_series(sidecar)


@pytest.fixture()
def series_without_vertex_ids(tmp_path: Path):
    """A stage-1 series: a valid sidecar with no ``vertex_id`` column."""
    sidecar = tmp_path / "blocks_registered" / "field.parquet"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    write_contact_depth_field_table(
        _table(_ROWS), sidecar, metadata=_metadata(COORDINATE_SPACE_ICP_REGISTERED)
    )
    return load_contact_depth_field_series(sidecar)


def test_a_v2_sidecar_carries_its_vertex_ids_and_provenance(projected_series) -> None:
    """The widening is what makes the surface join possible at all."""
    series = projected_series

    assert series.has_vertex_ids
    assert set(series.vertex_id_by_frame) == set(series.points_by_frame)
    assert series.vertex_id_by_frame[7].tolist() == [4, 4, 9]
    # int32 by schema, asserted rather than coerced: a widened index is an index
    # that came from somewhere other than the projection stage.
    assert series.vertex_id_by_frame[7].dtype == np.int32
    assert set(series.reference_ply_provenance) == set(REFERENCE_PLY_METADATA_KEYS)
    assert series.reference_ply_provenance["reference_ply"] == _REFERENCE_PLY_NAME
    assert series.reference_ply_provenance["reference_ply_vertex_count"] == str(
        _REFERENCE_VERTEX_COUNT
    )


def test_a_series_without_vertex_ids_carries_neither_half(
    series_without_vertex_ids,
) -> None:
    """The widening is optional in both halves, and they travel together."""
    assert series_without_vertex_ids.has_vertex_ids is False
    assert series_without_vertex_ids.vertex_id_by_frame is None
    assert series_without_vertex_ids.reference_ply_provenance is None


def test_the_scatter_lands_on_the_vertices_the_sidecar_names(projected_series) -> None:
    """Each touched vertex gets its own depth; the array is not shifted or sorted."""
    scalars = forearm_depth_scalars(projected_series, 19, _REFERENCE_VERTEX_COUNT)

    assert scalars is not None
    assert scalars.shape == (_REFERENCE_VERTEX_COUNT,)
    # signed -4.00 stored, +4.00 displayed: the negation happens once, upstream.
    assert scalars[0] == pytest.approx(4.0)
    assert np.count_nonzero(np.isfinite(scalars)) == 1


def test_untouched_vertices_are_nan_and_a_grazing_contact_is_zero(
    projected_series,
) -> None:
    """Missing is not zero.

    Frame 7 touches vertex 9 at exactly 0.00 mm — a real, grazing contact at the
    rim of the patch.  Every other vertex was not touched at all.  If untouched
    vertices were filled with 0.0 the two would render identically, and the
    patch would appear to extend across the whole forearm at its shallowest
    colour.
    """
    scalars = forearm_depth_scalars(projected_series, 7, _REFERENCE_VERTEX_COUNT)

    assert scalars[9] == 0.0, "a genuine 0.00 mm contact must survive as 0.0"
    assert not np.isnan(scalars[9])
    untouched = [v for v in range(_REFERENCE_VERTEX_COUNT) if v not in (4, 9)]
    assert np.all(np.isnan(scalars[untouched]))


def test_two_contact_points_on_one_vertex_resolve_to_the_deeper(
    projected_series,
) -> None:
    """Frame 7 addresses vertex 4 twice, at 0.25 mm and 0.50 mm.

    Row order must not decide the colour of a vertex: the sidecar promises a
    column set, not a row order, and the loader is free to re-sort by frame.
    The deeper of the two wins, which is order-independent.
    """
    scalars = forearm_depth_scalars(projected_series, 7, _REFERENCE_VERTEX_COUNT)

    assert scalars[4] == pytest.approx(0.50)


def test_a_frame_with_no_contact_paints_nothing_rather_than_zero(
    projected_series,
) -> None:
    """Frame 12 carries no rows at all; the whole forearm is untouched."""
    scalars = forearm_depth_scalars(projected_series, 12, _REFERENCE_VERTEX_COUNT)

    assert scalars is not None, "the stage can answer; the answer is 'nothing'"
    assert np.all(np.isnan(scalars))


def test_a_stage_without_vertex_ids_returns_none_not_an_empty_array(
    series_without_vertex_ids,
) -> None:
    """Stages 1-2 have no index, and ``None`` is the only honest answer.

    An empty array would read as "a forearm of zero vertices"; a zero-filled one
    as "touched everywhere at zero depth".  Both answer a question this stage
    cannot answer, and the alternative — snapping the points to their nearest
    vertex here — picks different vertices than the projection stage did.
    """
    assert (
        forearm_depth_scalars(
            series_without_vertex_ids, 7, _REFERENCE_VERTEX_COUNT
        )
        is None
    )


def test_a_forearm_of_the_wrong_size_refuses_the_join(projected_series) -> None:
    """The silent-renumbering hazard, and why validation precedes the scatter.

    A forearm re-deduplicated at a different epsilon has a different vertex
    count and a completely different numbering.  Every id here would still be
    *in range* against the larger mesh, so the range check alone would pass and
    the surface would be painted with the right depths on the wrong vertices.
    """
    with pytest.raises(ValueError, match="provenance mismatch"):
        forearm_depth_scalars(projected_series, 7, _REFERENCE_VERTEX_COUNT + 3)


def test_the_provenance_check_runs_before_the_scatter(projected_series) -> None:
    """Even a frame with no contact refuses a mismatched forearm.

    Otherwise a mismatch would surface only on the first frame that happens to
    touch — which, on a block that starts with the hand off the arm, is not the
    first frame drawn.
    """
    with pytest.raises(ValueError, match="provenance mismatch"):
        forearm_depth_scalars(projected_series, 12, _REFERENCE_VERTEX_COUNT + 3)


def test_a_forearm_with_no_vertices_refuses_the_join(projected_series) -> None:
    with pytest.raises(ValueError, match="no vertices"):
        forearm_depth_scalars(projected_series, 7, 0)


def test_the_forearm_join_refuses_to_run_without_a_field() -> None:
    """Reaching the join with no series means an ``is_present`` check was skipped."""
    with pytest.raises(TypeError, match="without a depth field"):
        forearm_depth_scalars(None, 7, _REFERENCE_VERTEX_COUNT)


def test_the_dto_refuses_vertex_ids_without_their_provenance() -> None:
    """An index whose mesh identity is unknown cannot be validated against anything."""
    with pytest.raises(ValueError, match="present together"):
        ContactDepthFieldSeries(
            points_by_frame={0: np.zeros((1, 3), np.float32)},
            penetration_depth_by_frame={0: np.zeros((1,), np.float64)},
            clim_penetration_mm=(0.0, 1.0),
            signed_depth_range_mm=(-1.0, 0.0),
            coordinate_space=COORDINATE_SPACE_ICP_REGISTERED,
            vertex_id_by_frame={0: np.zeros((1,), np.int32)},
        )


def test_the_dto_refuses_misaligned_vertex_ids() -> None:
    with pytest.raises(ValueError, match="aligned with the points"):
        ContactDepthFieldSeries(
            points_by_frame={0: np.zeros((3, 3), np.float32)},
            penetration_depth_by_frame={0: np.zeros((3,), np.float64)},
            clim_penetration_mm=(0.0, 1.0),
            signed_depth_range_mm=(-1.0, 0.0),
            coordinate_space=COORDINATE_SPACE_ICP_REGISTERED,
            vertex_id_by_frame={0: np.zeros((2,), np.int32)},
            reference_ply_provenance=_provenance(),
        )


def test_the_dto_refuses_a_widened_vertex_index() -> None:
    """int32 by schema. A widened index came from somewhere other than the sidecar."""
    with pytest.raises(ValueError, match="int32"):
        ContactDepthFieldSeries(
            points_by_frame={0: np.zeros((1, 3), np.float32)},
            penetration_depth_by_frame={0: np.zeros((1,), np.float64)},
            clim_penetration_mm=(0.0, 1.0),
            signed_depth_range_mm=(-1.0, 0.0),
            coordinate_space=COORDINATE_SPACE_ICP_REGISTERED,
            vertex_id_by_frame={0: np.zeros((1,), np.int64)},
            reference_ply_provenance=_provenance(),
        )


def test_the_forearm_join_uses_the_same_position_to_frame_map_as_the_patch(
    projected_series,
) -> None:
    """One expression turns a slider position into a frame key; both joins use it.

    The forearm layer takes a *frame index*, not a position.  If the widget
    passed it a position the surface would be painted from a different frame
    than the patch drawn on top of it — two layers of one field, disagreeing,
    with nothing on screen to say so.
    """
    kinect_df = pd.DataFrame({FRAME_INDEX_COLUMN: [7, 12, 19, 23]})
    indices = kinect_frame_indices(kinect_df, "<synthetic>")

    position = 2
    frame = kinect_frame_at_position(indices, position)
    patch = depth_frame_at_position(projected_series, indices, position)
    surface = forearm_depth_scalars(projected_series, frame, _REFERENCE_VERTEX_COUNT)

    assert frame == 19
    assert patch is not None
    # The patch's single vertex is vertex 0, and that is the one the surface lit.
    assert np.flatnonzero(np.isfinite(surface)).tolist() == [0]
    assert surface[0] == pytest.approx(patch[1][0])


def test_the_forearm_scalar_array_is_not_the_contact_one() -> None:
    """Two datasets, two meanings; a shared name would make a mix-up plausible."""
    assert FOREARM_DEPTH_SCALAR_NAME != "penetration_depth_mm"


# ---------------------------------------------------------------------------
# 6b. The timeseries join — slider position -> row of the *full* CSV
# ---------------------------------------------------------------------------
#
# The same silent-failure shape as the frame join above, in the other direction.
# The neural panel plots the whole merged CSV against ``np.arange(len(full_df))``
# while the slider walks the anchor rows, and the stage viewer used to bridge the
# two with ``int(position * len(full_df) / n_frames)``.  That multiplier is exact
# only if every anchor is equally spaced *and* the trailing nerve-rate rows after
# the last anchor happen to measure exactly one spacing.  Neither holds in
# general: real blocks measure 33 or 34 rows between anchors, and any stage that
# drops rows, or a nerve recording that outlives the Kinect one, moves the ratio
# away from the spacing entirely.
#
# These tests are written so the multiplier cannot pass them: each fixture is
# non-uniform in a way a single ratio cannot express, and
# ``test_a_uniform_scale_lands_on_another_frames_row`` asserts the old expression
# actively disagrees, so restoring it turns the suite red rather than merely
# leaving it silent.


def _upsampled_csv(
    gaps: Sequence[int],
    tail: int,
    first_frame: int = 41,
) -> pd.DataFrame:
    """A merged CSV in the shape the stage viewer reads it.

    One *anchor* row per Kinect frame — ``time_kinect`` and ``frame_index``
    present — followed by ``gap - 1`` nerve-rate rows in which every Kinect
    column is NaN, and ``tail`` such rows after the last anchor.  ``Nerve_freq``
    is finite on every row, exactly as the real artifact is: the nerve channel is
    what the extra rows exist to carry.
    """
    time_kinect: List[float] = []
    frame_index: List[float] = []
    frame = first_frame
    for gap in gaps:
        time_kinect.append(frame / 30.0)
        frame_index.append(float(frame))
        time_kinect.extend([np.nan] * (gap - 1))
        frame_index.extend([np.nan] * (gap - 1))
        frame += 1
    time_kinect.append(frame / 30.0)
    frame_index.append(float(frame))
    time_kinect.extend([np.nan] * tail)
    frame_index.extend([np.nan] * tail)

    n = len(time_kinect)
    return pd.DataFrame(
        {
            KINECT_ANCHOR_COLUMN: time_kinect,
            FRAME_INDEX_COLUMN: frame_index,
            "Nerve_freq": np.linspace(0.0, 60.0, n),
        }
    )


#: A realistic block: ~33.33 rows per frame, jittering between 33 and 34, with a
#: trailing nerve-rate run after the last frame.  Measured on
#: ``2022-06-15_ST14-02``, whose six blocks all sit at 33/34.
_REALISTIC_GAPS = [33, 33, 34] * 40

#: The pathological shape a single ratio cannot even approximate: the nerve
#: channel keeps recording long after the last Kinect frame, so
#: ``len(full_df) / n_frames`` is nowhere near the anchor spacing.
_LONG_TAIL_GAPS = [33] * 99


def _viewer_kinect_df(full_df: pd.DataFrame) -> pd.DataFrame:
    """Exactly what ``_load_stage_data`` derives, and what the slider walks."""
    return full_df.dropna(subset=[KINECT_ANCHOR_COLUMN]).reset_index(drop=True)


def _old_linear_rows(full_df: pd.DataFrame, n_frames: int) -> np.ndarray:
    """The removed expression: ``int(frame_idx * len(full_df) / n_frames)``."""
    scale = len(full_df) / n_frames
    return (np.arange(n_frames) * scale).astype(np.int64)


@pytest.mark.parametrize(
    "gaps, tail",
    [(_REALISTIC_GAPS, 33), (_LONG_TAIL_GAPS, 3300), ([40, 12, 91, 33], 7)],
)
def test_the_anchor_row_carries_the_very_record_the_scene_draws(
    gaps: Sequence[int], tail: int
) -> None:
    """The identity the cursor's correctness reduces to.

    For every slider position *p*, ``full_df.iloc[anchors[p]]`` must be the same
    record as ``kinect_df.iloc[p]`` — the row the 3D scene is drawing.  Asserted
    on both keys the two halves of the pipeline use: ``time_kinect``, which
    defines the anchor set, and ``frame_index``, which keys the depth field.
    """
    full_df = _upsampled_csv(gaps, tail)
    kinect_df = _viewer_kinect_df(full_df)

    rows = kinect_anchor_rows(full_df)

    assert np.array_equal(
        full_df[KINECT_ANCHOR_COLUMN].to_numpy()[rows],
        kinect_df[KINECT_ANCHOR_COLUMN].to_numpy(),
    )
    assert np.array_equal(
        full_df[FRAME_INDEX_COLUMN].to_numpy()[rows],
        kinect_df[FRAME_INDEX_COLUMN].to_numpy(),
    )


@pytest.mark.parametrize(
    "gaps, tail",
    [(_REALISTIC_GAPS, 33), (_LONG_TAIL_GAPS, 3300), ([40, 12, 91, 33], 7)],
)
def test_the_anchors_meet_the_panels_contract(gaps: Sequence[int], tail: int) -> None:
    """Shape, dtype and monotonicity — what ``NeuralDataPanel`` validates.

    Passing this is necessary and nowhere near sufficient: the panel checks the
    shape of the mapping, never its content, so a correctly-shaped wrong answer
    is accepted in silence.  That is why the identity above is the real test and
    this one only guards the constructor's preconditions.
    """
    full_df = _upsampled_csv(gaps, tail)
    n_frames = len(_viewer_kinect_df(full_df))

    rows = kinect_anchor_rows(full_df)

    assert rows.dtype == np.int64
    assert rows.shape == (n_frames,)
    assert np.all(np.diff(rows) > 0)
    assert rows[-1] < len(full_df)


def test_a_uniform_scale_lands_on_another_frames_row() -> None:
    """The regression guard: the removed expression must fail this fixture.

    With the nerve channel outliving the Kinect one the ratio is 66 rows per
    frame against a true spacing of 33, so the old cursor drifts a further frame
    away every frame.  Restoring the multiplication makes this test red.
    """
    full_df = _upsampled_csv(_LONG_TAIL_GAPS, tail=3300)
    n_frames = len(_viewer_kinect_df(full_df))

    rows = kinect_anchor_rows(full_df)
    old = _old_linear_rows(full_df, n_frames)

    disagreements = int(np.count_nonzero(old != rows))
    assert disagreements > 0.9 * n_frames, (
        f"only {disagreements}/{n_frames} rows differ; this fixture exists to "
        "make the linear scale unambiguously wrong"
    )
    # And wrong by whole frames, not by a rounding: the last position's cursor
    # sits beyond the end of the anchors entirely.
    assert old[-1] > rows[-1]


def test_even_an_almost_uniform_block_is_not_uniform_enough() -> None:
    """33-or-34 jitter alone already puts the cursor off the anchor row.

    This is the ordinary case — every block of ``2022-06-15_ST14-02`` looks like
    this — and it is why "close enough" is not a defence: the row the old scale
    picks is frequently an interpolated nerve-rate row that belongs to no frame
    at all, so what the cursor reads is not any frame's measurement.
    """
    full_df = _upsampled_csv(_REALISTIC_GAPS, tail=33)
    n_frames = len(_viewer_kinect_df(full_df))

    rows = kinect_anchor_rows(full_df)
    old = _old_linear_rows(full_df, n_frames)

    off_anchor = int(np.count_nonzero(~np.isin(old, rows)))
    assert off_anchor > 0, (
        "the fixture must reproduce the ordinary defect, not just the extreme one"
    )
    assert np.array_equal(rows, np.flatnonzero(full_df[KINECT_ANCHOR_COLUMN].notna()))


def test_the_click_inverse_returns_the_frame_it_started_from() -> None:
    """Round trip: ``update_cursor``'s row -> a click there -> the same position.

    This reproduces ``NeuralDataPanel._on_canvas_click``'s anchor branch rather
    than importing it, because the panel needs PyQt5 and a display.  What is
    being asserted is a property of the *anchors*: a click anywhere strictly
    inside the half-gap around an anchor resolves to that anchor's position, and
    that property is what makes the panel's nearest-anchor search an inverse
    rather than an approximation.  The live widget is exercised by the offscreen
    smoke run recorded in the plan.
    """
    full_df = _upsampled_csv(_REALISTIC_GAPS, tail=33)
    rows = kinect_anchor_rows(full_df)

    def click_to_position(xdata: float) -> int:
        pos = int(np.searchsorted(rows, xdata))
        if pos >= rows.size:
            pos = rows.size - 1
        elif pos > 0 and abs(xdata - rows[pos - 1]) <= abs(rows[pos] - xdata):
            pos -= 1
        return pos

    for p in range(rows.size):
        lo_gap = rows[p] - rows[p - 1] if p > 0 else 33
        hi_gap = rows[p + 1] - rows[p] if p + 1 < rows.size else 33
        for jitter in (-0.49 * lo_gap, 0.0, 0.49 * hi_gap):
            assert click_to_position(float(rows[p]) + jitter) == p


def test_a_csv_without_the_anchor_column_refuses_to_guess() -> None:
    """No column, no anchors — and a uniform scale is not an acceptable answer."""
    full_df = _upsampled_csv(_REALISTIC_GAPS, tail=33).drop(
        columns=[KINECT_ANCHOR_COLUMN]
    )

    with pytest.raises(ValueError, match=KINECT_ANCHOR_COLUMN):
        kinect_anchor_rows(full_df, "block-order-01.csv")


def test_rows_without_a_single_anchor_are_fatal() -> None:
    """Nerve-rate rows the viewer cannot place against any frame."""
    full_df = pd.DataFrame(
        {
            KINECT_ANCHOR_COLUMN: [np.nan] * 40,
            FRAME_INDEX_COLUMN: [np.nan] * 40,
            "Nerve_freq": np.arange(40.0),
        }
    )

    with pytest.raises(ValueError, match="not one of them"):
        kinect_anchor_rows(full_df, "block-order-01.csv")


def test_an_empty_csv_yields_no_anchors_rather_than_raising() -> None:
    """No rows is a different fact from rows that anchor nothing.

    An empty frame has no frames to display and the viewer never builds a panel
    for it; returning an empty array states that, and the shape check downstream
    still holds because ``_total_frames`` is zero too.
    """
    empty = pd.DataFrame(
        {KINECT_ANCHOR_COLUMN: [], FRAME_INDEX_COLUMN: [], "Nerve_freq": []}
    )

    rows = kinect_anchor_rows(empty)

    assert rows.shape == (0,)
    assert rows.dtype == np.int64


def test_the_stage_viewer_no_longer_holds_a_cursor_scale() -> None:
    """The wrong mechanism is gone from the call site, not merely unused.

    Checked in the source because the viewer imports PyQt5, PyVista and Open3D
    and cannot be imported here.  A leftover ``_neural_scale`` attribute is an
    invitation to reuse it, and reuse would be invisible: a cursor placed by a
    scale still moves smoothly.
    """
    viewer = _SRC / "postprocessing" / "gui" / "postprocessing_stage_viewer.py"
    source = viewer.read_text(encoding="utf-8")

    assert "_neural_scale" not in source, (
        "postprocessing_stage_viewer.py still defines or uses _neural_scale; the "
        "cursor must be placed by an anchor lookup, never by a multiplier"
    )
    assert "kinect_anchor_rows" in source


# ---------------------------------------------------------------------------
# 7. Purity — the leaf stays headless
# ---------------------------------------------------------------------------

#: The toolkits that would make this module untestable without a display.  The
#: leaf lives under ``gui/`` for locality only; every line of it is decidable
#: from a stage index, a loader and a path.
_FORBIDDEN_IMPORT_ROOTS = frozenset({"PyQt5", "pyvista", "pyvistaqt", "vtk", "open3d"})


def _imported_roots(module_path: Path) -> set:
    """Top-level package name of every import statement in *module_path*."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.parametrize(
    "module",
    [
        _SRC / "postprocessing" / "gui" / "stage_depth_field.py",
        _SRC / "merging" / "contact_depth_field_series.py",
    ],
)
def test_the_policy_leaf_and_its_dependency_import_no_gui_toolkit(module: Path) -> None:
    """Checked statically, so the result does not depend on what else ran first.

    ``sys.modules`` is useless for this in a full-suite run — a sibling test
    module importing PyQt5 would poison it — whereas the import statements in
    the file are the actual contract.  The dependency is checked too, since an
    import one level down would drag the toolkit in just as effectively.
    """
    offenders = _imported_roots(module) & _FORBIDDEN_IMPORT_ROOTS

    assert not offenders, (
        f"{module.name} imports {sorted(offenders)}; the policy leaf and the "
        "adapter beneath it must stay importable without a display, which is "
        "the only reason they are separate modules from the viewers."
    )
