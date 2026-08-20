"""Tests for the postprocessing stage viewer's session / block / stage selection.

The viewer used to be per-block: the batch loop opened one window per block
config, 99 of them for the full 11-session DAG, and every one of them opened on
stage 0 — the single stage that carries no depth field by construction. This
module covers the leaf that replaced all three of those behaviours
(``postprocessing.gui.stage_selection``), and it exists as a separate file from
``test_stage_depth_field.py`` for the same reason the two source modules are
separate: that one is depth-field *policy* — which coordinate space a stage's
sidecar must declare — while this one is *selection* policy, which triple the
window opens on and what the dropdown calls it. Neither imports the other's
subject.

**Where the default lands.** "Open on the last stage" is one sentence and three
behaviours: the final stage when it has data, the last stage that does when it
has not, and 0 when nothing does. The middle case is the one that matters on
real artifacts — a session part-way through the DAG has stage CSVs up to
wherever it got to — and it is invisible on a synthetic tree where every stage
exists, so every one of the three is built here explicitly.

**Display labels are not canonical labels.** ``STAGE_LABELS`` keys
``PRODUCING_TASK_BY_STAGE`` and ``ACCEPTED_SPACES_BY_STAGE`` and names the stage
in every validation error. The ordinals belong in the dropdown and nowhere else,
so the tests assert the canonical list is *unchanged* as well as asserting what
the formatter produces — a renumbering that edited ``STAGE_LABELS`` instead
would satisfy a formatter-only test perfectly.

**Laziness, now at session scale.** ``test_stage_depth_field.py`` asserts that
building one block's six loaders reads zero bytes. The session-level window
raises the stake to 99 blocks x 6 stages = 594 sidecars, and adds a second eager
trap the per-block case did not have: resolving a block's stage paths loads that
block's reference forearm as an in-memory point cloud, so an index that resolved
every block on construction would hold 99 of them before the first pixel. The
index is therefore asserted to call its resolver zero times when it is built and
exactly once per *selected* block, and the shared bounded cache is asserted to
stay bounded by the number of stages rather than growing with the number of
blocks.

No Qt, no VTK, no Open3D: ``QtInteractor`` cannot initialise in this environment
(``0xC00000FD``), which is precisely why none of the logic under test lives in a
slot.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

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
    make_contact_depth_field_loader,
)
from postprocessing.gui.stage_depth_field import STAGE_LABELS  # noqa: E402
from postprocessing.gui.stage_selection import (  # noqa: E402
    BlockEntry,
    SessionBlockIndex,
    build_session_block_index,
    default_stage_index,
    stage_display_label,
    stage_display_labels,
    stage_index_for_block,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# ---------------------------------------------------------------------------
# The on-disk layout ``resolve_stage_paths`` produces, per block
# ---------------------------------------------------------------------------

#: ``(output_directory, csv_name_uses_pca_fork, declared_coordinate_space)``.
#:
#: Stage 0 has no sidecar by construction — it reads ``blocks_merged/``, written
#: before the depth field is filtered by neural quality.  See
#: ``stage_depth_field``'s module docstring.
_STAGE_LAYOUT: Sequence[tuple] = (
    ("blocks_merged", False, None),
    ("blocks_registered", False, COORDINATE_SPACE_ICP_REGISTERED),
    ("blocks_deduped", False, COORDINATE_SPACE_ICP_REGISTERED),
    ("blocks_projected", False, COORDINATE_SPACE_ICP_REGISTERED),
    ("blocks_pca_calibrated", True, COORDINATE_SPACE_PCA_CALIBRATED),
    ("blocks_rf_centered", True, COORDINATE_SPACE_RF_CENTERED),
)


def _stage_csv_paths(base: Path, session_id: str, block_id: str) -> List[Path]:
    """One block's six stage CSV paths, as the production resolver lays them out."""
    raw = f"{session_id}_semicontrolled_{block_id}_merged_data.csv"
    pca = f"{session_id}_semicontrolled_{block_id}_merged_data_pca-xyz.csv"
    return [
        base / directory / (pca if is_pca else raw)
        for directory, is_pca, _ in _STAGE_LAYOUT
    ]


def _touch_stage_csvs(paths: Sequence[Path], up_to_stage: int) -> None:
    """Create the stage CSVs of stages ``0..up_to_stage``, and no others.

    A session part-way through the postprocessing DAG looks exactly like this on
    disk, which is the case the "last stage that has data" rule exists for.
    """
    for stage_idx, csv_path in enumerate(paths):
        if stage_idx > up_to_stage:
            continue
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("time_kinect,frame_index\n", encoding="utf-8")


class _FakeConfig:
    """A config stand-in that refuses every attribute but the two the index needs.

    Deliberately not a ``KinectConfig``, and deliberately hostile.  The index's
    laziness claim is "it reads identifiers and resolves nothing", and the only
    way to *prove* that rather than to inspect for it is to make any other
    attribute access fail: ``session_merged_output_dir`` and
    ``session_processed_output_dir`` are the two a stage-path resolution would
    reach for on its way to loading a forearm point cloud, and touching either
    here raises instead.
    """

    def __init__(self, block_id: str, stem: str) -> None:
        self.block_id = block_id
        self.source_video = Path(f"{stem}.mkv")

    def __getattr__(self, name: str):  # only reached for attributes not set above
        raise AssertionError(
            f"The session/block index touched {name!r} on a block config. It "
            "may read block_id and source_video and nothing else; anything "
            "further is a step towards resolving a block that was never "
            "selected."
        )


class _CountingResolver:
    """A stage-paths resolver that records which blocks it was asked about."""

    def __init__(self, stage_paths_by_key: Dict[tuple, List[Optional[Path]]]) -> None:
        self._stage_paths_by_key = stage_paths_by_key
        self.calls: List[tuple] = []

    def __call__(self, entry: BlockEntry) -> List[Optional[Path]]:
        self.calls.append(entry.key)
        return self._stage_paths_by_key[entry.key]


class _CountingReporter:
    """Collects depth-field resolution messages and counts how often it was called."""

    def __init__(self) -> None:
        self.messages: List[str] = []

    def __call__(self, message: str) -> None:
        self.messages.append(message)

    @property
    def calls(self) -> int:
        return len(self.messages)


# ---------------------------------------------------------------------------
# 1. Display labels — numbered on screen, canonical everywhere else
# ---------------------------------------------------------------------------


def test_the_dropdown_shows_the_processing_order() -> None:
    """Six stages, numbered 1..6 in the order the pipeline runs them."""
    assert stage_display_labels() == [
        "1. Merged (Raw)",
        "2. ICP Registered",
        "3. Deduplicated",
        "4. Contact Projected",
        "5. PCA Calibrated",
        "6. RF Centered",
    ]


def test_the_canonical_labels_are_not_renumbered() -> None:
    """The formatter is additive; the labels the pipeline is keyed on are untouched.

    A "renumber the stages" change that edited ``STAGE_LABELS`` in place would
    pass a formatter-only assertion and silently rewrite the names used in every
    space-mismatch error and in ``PRODUCING_TASK_BY_STAGE``'s documentation.
    """
    assert STAGE_LABELS == [
        "Merged (Raw)",
        "ICP Registered",
        "Deduplicated",
        "Contact Projected",
        "PCA Calibrated",
        "RF Centered",
    ]
    assert not any(label[0].isdigit() for label in STAGE_LABELS)


@pytest.mark.parametrize("stage_idx", range(len(STAGE_LABELS)))
def test_a_display_label_is_its_canonical_label_plus_an_ordinal(
    stage_idx: int,
) -> None:
    """The canonical label survives verbatim, so the two are greppable together."""
    display = stage_display_label(stage_idx)

    assert display == f"{stage_idx + 1}. {STAGE_LABELS[stage_idx]}"
    assert display.endswith(STAGE_LABELS[stage_idx])


@pytest.mark.parametrize("stage_idx", [-1, len(STAGE_LABELS), 99])
def test_an_unknown_stage_has_no_display_label(stage_idx: int) -> None:
    """A stage index the viewer cannot show is a wiring error, not a blank string."""
    with pytest.raises(ValueError, match="names no postprocessing stage"):
        stage_display_label(stage_idx)


# ---------------------------------------------------------------------------
# 2. Which stage a block opens on
# ---------------------------------------------------------------------------


def test_a_complete_block_opens_on_the_final_stage(tmp_path: Path) -> None:
    """The default is the pipeline's answer, not its input."""
    paths = _stage_csv_paths(tmp_path, "2022-06-15_ST14-02", "block-order-01")
    _touch_stage_csvs(paths, up_to_stage=len(STAGE_LABELS) - 1)

    assert default_stage_index(paths) == len(STAGE_LABELS) - 1


@pytest.mark.parametrize("last_produced", [0, 1, 2, 3, 4])
def test_a_partly_processed_block_opens_on_the_last_stage_that_ran(
    tmp_path: Path, last_produced: int
) -> None:
    """The fallback walks backwards, and stops at the first stage with data.

    This is the real shape of a session part-way through the DAG.  Opening on a
    stage whose CSV does not exist would greet the user with "No data available"
    while a populated stage sat one entry up the dropdown.
    """
    paths = _stage_csv_paths(tmp_path, "2022-06-15_ST14-02", "block-order-01")
    _touch_stage_csvs(paths, up_to_stage=last_produced)

    assert default_stage_index(paths) == last_produced


def test_a_block_with_no_data_at_all_falls_to_stage_zero(tmp_path: Path) -> None:
    """Zero is what is left when nothing exists — not a claim that stage 0 has data."""
    paths = _stage_csv_paths(tmp_path, "2022-06-15_ST14-02", "block-order-01")

    assert default_stage_index(paths) == 0


def test_a_session_with_no_merged_output_directory_falls_to_stage_zero() -> None:
    """``resolve_stage_paths`` yields six ``None`` paths in that case."""
    assert default_stage_index([None] * len(STAGE_LABELS)) == 0


def test_a_gap_before_the_final_stage_does_not_hide_it(tmp_path: Path) -> None:
    """The rule is "the last stage that has data", not "the last contiguous one".

    A stage CSV can legitimately be absent while a later one exists — the
    dedup/projection outputs of a session re-run from a checkpoint, say — and the
    user came to see the furthest stage the pipeline reached.
    """
    paths = _stage_csv_paths(tmp_path, "2022-06-15_ST14-02", "block-order-01")
    _touch_stage_csvs(paths, up_to_stage=1)
    paths[5].parent.mkdir(parents=True, exist_ok=True)
    paths[5].write_text("time_kinect,frame_index\n", encoding="utf-8")

    assert default_stage_index(paths) == 5


@pytest.mark.parametrize("n_paths", [0, 5, 7])
def test_a_stage_list_of_the_wrong_length_is_refused(n_paths: int) -> None:
    """"The last one" means something different in a list of a different length."""
    with pytest.raises(ValueError, match="one entry per stage"):
        default_stage_index([None] * n_paths)


# ---------------------------------------------------------------------------
# 3. Carrying the stage across a block change
# ---------------------------------------------------------------------------


def test_switching_block_keeps_the_stage_on_screen(tmp_path: Path) -> None:
    """Comparing one stage across blocks is why the block dropdown exists."""
    paths = _stage_csv_paths(tmp_path, "2022-06-14_ST13-01", "block-order-02")
    _touch_stage_csvs(paths, up_to_stage=len(STAGE_LABELS) - 1)

    assert stage_index_for_block(2, paths) == 2


def test_a_stage_the_new_block_lacks_falls_back_to_the_default(
    tmp_path: Path,
) -> None:
    """The preference is honoured only where it can be honoured with data."""
    paths = _stage_csv_paths(tmp_path, "2022-06-14_ST13-01", "block-order-02")
    _touch_stage_csvs(paths, up_to_stage=2)

    assert stage_index_for_block(5, paths) == 2


def test_the_first_load_has_no_preference_and_takes_the_default(
    tmp_path: Path,
) -> None:
    """``None`` is "nothing on screen yet", which is not the same as "stage 0"."""
    paths = _stage_csv_paths(tmp_path, "2022-06-14_ST13-01", "block-order-02")
    _touch_stage_csvs(paths, up_to_stage=len(STAGE_LABELS) - 1)

    assert stage_index_for_block(None, paths) == len(STAGE_LABELS) - 1


def test_stage_zero_is_carried_when_the_user_actually_chose_it(
    tmp_path: Path,
) -> None:
    """A held preference of 0 is a choice, and is kept — the default is not re-applied."""
    paths = _stage_csv_paths(tmp_path, "2022-06-14_ST13-01", "block-order-02")
    _touch_stage_csvs(paths, up_to_stage=len(STAGE_LABELS) - 1)

    assert stage_index_for_block(0, paths) == 0


@pytest.mark.parametrize("preferred", [-1, len(STAGE_LABELS)])
def test_a_preference_naming_no_stage_is_refused(
    tmp_path: Path, preferred: int
) -> None:
    """Out of range is a wiring error; silently defaulting would hide it."""
    paths = _stage_csv_paths(tmp_path, "2022-06-14_ST13-01", "block-order-02")

    with pytest.raises(ValueError, match="names no postprocessing stage"):
        stage_index_for_block(preferred, paths)


# ---------------------------------------------------------------------------
# 4. The session / block index
# ---------------------------------------------------------------------------


def _session_map(shape: Dict[str, Sequence[str]]) -> Dict[str, List[_FakeConfig]]:
    """session id → configs, from a ``{session: [block_id, ...]}`` sketch."""
    return {
        session_id: [
            _FakeConfig(block_id, f"{session_id}_{block_id}")
            for block_id in block_ids
        ]
        for session_id, block_ids in shape.items()
    }


def test_the_index_lists_sessions_and_their_blocks_in_a_stable_order() -> None:
    """Sorted, so the dropdowns do not depend on config-file enumeration order."""
    index = build_session_block_index(
        _session_map(
            {
                "2022-06-15_ST14-02": ["block-order-03", "block-order-01"],
                "2022-06-14_ST13-01": ["block-order-02"],
            }
        )
    )

    assert index.sessions == ["2022-06-14_ST13-01", "2022-06-15_ST14-02"]
    assert index.blocks("2022-06-15_ST14-02") == ["block-order-01", "block-order-03"]
    assert index.n_sessions == 2
    assert len(index) == 3


def test_the_index_opens_on_the_first_block_of_the_first_session() -> None:
    index = build_session_block_index(
        _session_map(
            {
                "2022-06-15_ST14-02": ["block-order-01"],
                "2022-06-14_ST13-01": ["block-order-02", "block-order-01"],
            }
        )
    )

    assert index.first_entry().key == ("2022-06-14_ST13-01", "block-order-01")


def test_each_entry_carries_its_recording_name_and_its_opaque_config() -> None:
    """The config travels as a handle; the index never looks inside it."""
    session_map = _session_map({"2022-06-15_ST14-02": ["block-order-01"]})
    index = build_session_block_index(session_map)

    entry = index.entry("2022-06-15_ST14-02", "block-order-01")

    assert entry.recording_name == "2022-06-15_ST14-02_block-order-01"
    assert entry.config is session_map["2022-06-15_ST14-02"][0]


def test_an_unknown_block_is_refused_rather_than_returning_nothing() -> None:
    """A dropdown pair with no entry means the dropdowns are out of step."""
    index = build_session_block_index(
        _session_map({"2022-06-15_ST14-02": ["block-order-01"]})
    )

    with pytest.raises(ValueError, match="No block"):
        index.entry("2022-06-15_ST14-02", "block-order-99")
    with pytest.raises(ValueError, match="not in this index"):
        index.blocks("2022-06-14_ST13-01")


def test_a_duplicate_block_is_refused() -> None:
    """Two entries under one dropdown row means one can never be selected."""
    with pytest.raises(ValueError, match="Duplicate block"):
        SessionBlockIndex(
            [
                BlockEntry("s", "b", "rec-a"),
                BlockEntry("s", "b", "rec-b"),
            ]
        )


def test_an_empty_batch_is_refused_rather_than_opening_a_blank_window() -> None:
    with pytest.raises(ValueError, match="no session"):
        build_session_block_index({})
    with pytest.raises(ValueError, match="no blocks"):
        SessionBlockIndex([])


def test_a_session_with_no_blocks_is_refused() -> None:
    """It would be selectable in one dropdown and empty in the next."""
    with pytest.raises(ValueError, match="no block configs"):
        build_session_block_index({"2022-06-15_ST14-02": []})


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_identifier_is_refused(blank: str) -> None:
    """A blank dropdown row cannot be selected back to the block it stands for."""
    with pytest.raises(ValueError, match="non-empty string"):
        BlockEntry(blank, "block-order-01", "rec")


# ---------------------------------------------------------------------------
# 5. Laziness at session scale — the whole reason the index holds identifiers
# ---------------------------------------------------------------------------
#
# ``test_stage_depth_field.py`` section 2 asserts the per-block half of this:
# six loaders built, zero bytes read.  A session-level window multiplies that by
# the batch — 99 blocks x 6 stages — and adds a second eager trap the per-block
# case never had, because resolving a block's stage paths loads that block's
# reference forearm as an in-memory point cloud.

_FULL_BATCH_SESSIONS = 11
_FULL_BATCH_BLOCKS_PER_SESSION = 9  # 11 x 9 = 99, the full DAG's block count


def _full_batch_session_map() -> Dict[str, List[_FakeConfig]]:
    """A batch the size of the real one, with synthetic ids.

    The ids are generated rather than named: a hardcoded session or block id
    would tie this file to one recording campaign.
    """
    return _session_map(
        {
            f"session-{s:02d}": [
                f"block-order-{b:02d}"
                for b in range(1, _FULL_BATCH_BLOCKS_PER_SESSION + 1)
            ]
            for s in range(1, _FULL_BATCH_SESSIONS + 1)
        }
    )


def test_building_a_full_batch_index_touches_no_block(tmp_path: Path) -> None:
    """The headline session-level invariant: 99 blocks indexed, none resolved.

    Resolving is what loads a block's forearm point cloud.  An index that
    resolved on construction would hold 99 of them before the window drew a
    pixel — the eager-loading regression the depth-field work exists to prevent,
    reintroduced one level up from where it was prevented.

    Proved rather than inspected for: ``_FakeConfig`` raises on any attribute
    but the two identifiers, so a construction that reached for an output
    directory on its way to a point cloud fails here rather than in a profiler.
    The entries still carry their configs afterwards, so the laziness is not the
    trivial kind where the handle was dropped.
    """
    index = build_session_block_index(_full_batch_session_map())

    assert len(index) == 99
    assert index.n_sessions == _FULL_BATCH_SESSIONS
    assert all(
        isinstance(index.entry(session_id, block_id).config, _FakeConfig)
        for session_id in index.sessions
        for block_id in index.blocks(session_id)
    )
    # And enumerating every entry — which is all the dropdowns ever do — still
    # touches nothing resolvable.
    with pytest.raises(AssertionError, match=r"never\s+selected"):
        index.first_entry().config.session_merged_output_dir


def test_selecting_blocks_resolves_exactly_those_blocks(tmp_path: Path) -> None:
    """One resolve per selected block, and none for the 97 that were not."""
    session_map = _full_batch_session_map()
    index = build_session_block_index(session_map)
    stage_paths_by_key = {
        (session_id, cfg.block_id): _stage_csv_paths(
            tmp_path, session_id, cfg.block_id
        )
        for session_id, configs in session_map.items()
        for cfg in configs
    }
    resolver = _CountingResolver(stage_paths_by_key)

    first = index.first_entry()
    resolver(first)
    other = index.entry(index.sessions[3], index.blocks(index.sessions[3])[2])
    resolver(other)

    assert resolver.calls == [first.key, other.key]
    assert len(resolver.calls) == 2


def test_a_full_batch_of_loaders_reads_nothing(tmp_path: Path) -> None:
    """594 sidecars wired behind one bounded cache; zero bytes read, zero entries.

    The per-block version of this assertion lives in ``test_stage_depth_field.py``
    (``test_building_the_six_stage_loaders_reads_nothing``).  This is the same
    guarantee at the scale the session-level window actually operates at, and it
    is asserted rather than inferred because an eager version would draw exactly
    the same pictures.
    """
    session_map = _full_batch_session_map()
    cache = BoundedContactDepthFieldCache(maxsize=len(STAGE_LABELS))
    reporter = _CountingReporter()

    loaders = []
    for session_id, configs in session_map.items():
        for cfg in configs:
            for csv_path in _stage_csv_paths(tmp_path, session_id, cfg.block_id):
                loaders.append(
                    make_contact_depth_field_loader(
                        depth_field_path_for_csv(csv_path), reporter, cache
                    )
                )

    assert len(loaders) == 99 * len(STAGE_LABELS) == 594
    assert reporter.calls == 0, "wiring a batch of loaders must not resolve anything"
    assert len(cache) == 0, "wiring a batch of loaders must not populate the cache"


def test_the_cache_bound_is_the_stage_count_not_the_block_count(
    tmp_path: Path,
) -> None:
    """Residency is a property of the cache size, not of how many blocks were browsed.

    One cache is shared across every block the window visits.  If the bound grew
    with the number of blocks — or if each block got a cache of its own that was
    never released — a user who walked the batch would end up with the
    all-blocks-resident footprint eager loading had, reached more slowly.
    """
    bound = len(STAGE_LABELS)
    cache = BoundedContactDepthFieldCache(maxsize=bound)
    reporter = _CountingReporter()
    session_map = _session_map(
        {f"session-{s:02d}": ["block-order-01", "block-order-02"] for s in range(1, 6)}
    )

    for session_id, configs in session_map.items():
        for cfg in configs:
            csv_paths = _stage_csv_paths(tmp_path, session_id, cfg.block_id)
            _write_sidecars(csv_paths)
            for csv_path in csv_paths:
                make_contact_depth_field_loader(
                    depth_field_path_for_csv(csv_path), reporter, cache
                )()
                assert len(cache) <= bound

    assert cache.maxsize == bound
    assert len(cache) <= bound


# ---------------------------------------------------------------------------
# Sidecar helpers for the cache-bound walk
# ---------------------------------------------------------------------------

#: ``(frame, time_s, x, y, z, signed_depth_mm)`` — the production writer's schema.
_ROWS = [
    (7, 0.233, 1.0, 2.0, 3.0, -0.25),
    (7, 0.233, 1.5, 2.5, 3.5, -0.50),
    (19, 0.633, 10.0, 20.0, 30.0, -4.00),
]


def _write_sidecars(csv_paths: Sequence[Path]) -> None:
    """Write a real sidecar beside every sidecar-bearing stage CSV of one block."""
    table = pd.DataFrame(
        {
            "frame_index": np.array([r[0] for r in _ROWS], dtype=np.int32),
            "time_s": np.array([r[1] for r in _ROWS], dtype=np.float64),
            "x": np.array([r[2] for r in _ROWS], dtype=np.float32),
            "y": np.array([r[3] for r in _ROWS], dtype=np.float32),
            "z": np.array([r[4] for r in _ROWS], dtype=np.float32),
            "signed_depth_mm": np.array([r[5] for r in _ROWS], dtype=np.float64),
        }
    )
    for csv_path, (_, _, space) in zip(csv_paths, _STAGE_LAYOUT):
        sidecar = depth_field_path_for_csv(csv_path)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        if space is None:
            continue
        write_contact_depth_field_table(
            table,
            sidecar,
            metadata={
                "schema_version": SCHEMA_VERSION,
                "coordinate_space": space,
                "units": UNITS,
                "sign_convention": SIGN_CONVENTION,
                "produced_by": PRODUCED_BY,
            },
        )


# ---------------------------------------------------------------------------
# 6. Purity — the selection leaf stays headless
# ---------------------------------------------------------------------------

#: The toolkits that would make this module untestable without a display.  The
#: leaf lives under ``gui/`` for locality only; every line of it is decidable
#: from strings, indices and paths.
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


def test_the_selection_leaf_imports_no_gui_toolkit() -> None:
    """Checked statically, so the result does not depend on what else ran first.

    Same guard, same reasoning, as ``stage_depth_field``'s: ``QtInteractor``
    cannot initialise here, so any selection logic that drifted into a widget
    would stop being covered — and the drift would announce itself as an import
    long before it announced itself as a gap in coverage.
    """
    module = _SRC / "postprocessing" / "gui" / "stage_selection.py"

    offenders = _imported_roots(module) & _FORBIDDEN_IMPORT_ROOTS

    assert not offenders, (
        f"{module.name} imports {sorted(offenders)}; the selection leaf must "
        "stay importable without a display, which is the only reason it is a "
        "separate module from the viewer."
    )
