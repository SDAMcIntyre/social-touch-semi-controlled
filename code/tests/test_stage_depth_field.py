"""Tests for the contact-depth-field wiring behind the postprocessing stage viewer.

Two phases of ``render-contact-depth-field-in-postprocessing-viewer`` are
covered here.  Phase 1 makes the six postprocessing stages *carry* a depth-field
loader: nothing renders and, crucially, nothing reads.  Phase 2 adds the
stage-aware policy leaf that turns ``(stage, loader, path)`` into either a
space-validated series or an announced absence.  These tests guard the
properties that make both worth doing at all.

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
    COORDINATE_SPACE_PCA_CALIBRATED,
    COORDINATE_SPACE_RF_CENTERED,
    PRODUCED_BY,
    SCHEMA_VERSION,
    SIGN_CONVENTION,
    UNITS,
    write_contact_depth_field_table,
)
from postprocessing.depth_field_stage_io import depth_field_path_for_csv  # noqa: E402
from merging.contact_depth_field_series import (  # noqa: E402
    BoundedContactDepthFieldCache,
    ContactDepthFieldSeries,
    make_contact_depth_field_loader,
)
from postprocessing.gui.stage_depth_field import (  # noqa: E402
    EXPECTED_SPACE_BY_STAGE,
    PRODUCING_TASK_BY_STAGE,
    STAGE_LABELS,
    StageDepthField,
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
#: stage crossed with the two spaces it is *not*.
_MISMATCHES = [
    (stage_idx, wrong)
    for stage_idx, _, _, expected in _SIDECAR_BEARING
    for wrong in (
        COORDINATE_SPACE_ICP_REGISTERED,
        COORDINATE_SPACE_PCA_CALIBRATED,
        COORDINATE_SPACE_RF_CENTERED,
    )
    if wrong != expected
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
    assert 0 not in EXPECTED_SPACE_BY_STAGE
    assert sorted(EXPECTED_SPACE_BY_STAGE) == [s[0] for s in _SIDECAR_BEARING]
    assert sorted(PRODUCING_TASK_BY_STAGE) == sorted(EXPECTED_SPACE_BY_STAGE)
    for stage_idx, _, _, expected in _SIDECAR_BEARING:
        assert EXPECTED_SPACE_BY_STAGE[stage_idx] == expected
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
    expected_space = EXPECTED_SPACE_BY_STAGE[stage_idx]
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
# 5. Purity — the leaf stays headless
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
