"""Selection policy for the postprocessing stage viewer's three dropdowns.

The viewer beside this module shows one *(session, block, stage)* triple at a
time.  Which triple it opens on, which one a dropdown change lands on, and what
each stage is *called* in the dropdown are three decisions that have nothing to
do with rendering — and every one of them is wrong in a way that only shows up
as a picture, so none of them may live inside a Qt slot where they cannot be
tested.  This module is where they live instead.

Why this lives under ``gui/`` yet imports no GUI
------------------------------------------------
The same arrangement, and the same reason, as ``stage_depth_field.py`` next
door: locality of the only consumer, but every line here is decidable from
strings, indices and paths, so nothing forces a Qt, VTK or Open3D import.
``QtInteractor`` cannot even initialise in the test environment, so logic that
sits in a slot is logic that is never checked.  **Do not import Qt, VTK,
PyVista or Open3D here.**

Display labels are not the canonical labels
-------------------------------------------
:data:`~postprocessing.gui.stage_depth_field.STAGE_LABELS` is canonical.  It
keys :data:`~postprocessing.gui.stage_depth_field.PRODUCING_TASK_BY_STAGE` and
:data:`~postprocessing.gui.stage_depth_field.ACCEPTED_SPACES_BY_STAGE`, and it
is the name every validation error uses when it tells the user which stage
refused to open.  :func:`stage_display_label` prefixes an ordinal for the
dropdown *only*, so that the processing order is visible on screen without the
numbering leaking into a message that has to be greppable against the pipeline's
own vocabulary.  Renumbering ``STAGE_LABELS`` itself would do the opposite.

Laziness is the whole point of the index
----------------------------------------
A full batch is 11 sessions and 99 blocks.  Resolving one block's stage paths
loads that block's reference forearm as an in-memory point cloud, so an index
that resolved every block up front would hold 99 of them before the first pixel
— the eager-loading regression the depth-field work was done specifically to
avoid, reintroduced one level up.  :class:`SessionBlockIndex` therefore holds
*identifiers* and nothing else: it never calls a resolver, and the viewer calls
one for a block only when that block is selected.  The index is a pure data
structure, which is what makes that claim testable without a window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .stage_depth_field import STAGE_LABELS

__all__ = [
    "BlockEntry",
    "SessionBlockIndex",
    "build_session_block_index",
    "default_stage_index",
    "stage_display_label",
    "stage_display_labels",
    "stage_index_for_block",
]


# ---------------------------------------------------------------------------
# Display labels
# ---------------------------------------------------------------------------


def stage_display_label(stage_idx: int) -> str:
    """Return the dropdown text for *stage_idx* — its ordinal plus its label.

    The stages are a pipeline, and a bare list of six names does not say which
    one runs first.  The ordinal is 1-based because it is read by a human beside
    five others, not indexed by anything.

    **Display only.**  Nothing keyed on a stage name may use this: the canonical
    label is :data:`~postprocessing.gui.stage_depth_field.STAGE_LABELS`, and an
    error message that said "4. Contact Projected" would no longer match the
    vocabulary the rest of the pipeline — and this module's own docstring —
    uses.

    Args:
        stage_idx: Index into :data:`STAGE_LABELS`.

    Returns:
        ``"<ordinal>. <canonical label>"``.

    Raises:
        ValueError: If *stage_idx* names no stage.
    """
    if not 0 <= stage_idx < len(STAGE_LABELS):
        raise ValueError(
            f"stage_idx={stage_idx} names no postprocessing stage; the viewer "
            f"offers {len(STAGE_LABELS)} (0..{len(STAGE_LABELS) - 1}): "
            f"{STAGE_LABELS}."
        )
    return f"{stage_idx + 1}. {STAGE_LABELS[stage_idx]}"


def stage_display_labels() -> List[str]:
    """Every stage's dropdown text, in processing order."""
    return [stage_display_label(idx) for idx in range(len(STAGE_LABELS))]


# ---------------------------------------------------------------------------
# Which stage a block opens on
# ---------------------------------------------------------------------------


def default_stage_index(csv_paths: Sequence[Optional[Path]]) -> int:
    """Return the stage a block should open on: the last one that has data.

    The last stage is the pipeline's answer — the thing the user came to look
    at — and it is also the only stage whose depth-field sidecar has been
    through every transform.  Stage 0 is the *worst* default available: it reads
    ``blocks_merged/``, which by construction carries no depth field at all (see
    ``stage_depth_field``'s module docstring), so opening there greets the user
    with a disabled "Colour by depth" control every single time.

    The walk backwards is the guard: a block whose final stage has not been
    produced yet opens on the last stage that *was*, rather than on an empty
    scene.  Only a block with no stage CSV at all falls to 0, and that 0 is not
    a pretence that stage 0 has data — it is the one index left when none do,
    and the viewer renders it as "No data available".

    Args:
        csv_paths: One entry per stage, in stage order, each the stage's merged
            CSV or ``None`` when the session has no merged output directory.

    Returns:
        The index to open on.

    Raises:
        ValueError: If *csv_paths* does not have one entry per stage.  A short
            list would silently make some other stage "the last one".
    """
    if len(csv_paths) != len(STAGE_LABELS):
        raise ValueError(
            f"csv_paths must have one entry per stage ({len(STAGE_LABELS)}), "
            f"got {len(csv_paths)}. Choosing 'the last stage with data' from a "
            "list of a different length would pick a different stage than the "
            "dropdown shows."
        )
    for stage_idx in range(len(csv_paths) - 1, -1, -1):
        csv_path = csv_paths[stage_idx]
        if csv_path is not None and csv_path.exists():
            return stage_idx
    return 0


def stage_index_for_block(
    preferred_stage_idx: Optional[int],
    csv_paths: Sequence[Optional[Path]],
) -> int:
    """Return the stage to show for a block the user just switched to.

    Keeping the stage across a block change is what makes the two dropdowns
    usable together: comparing the same stage across blocks is the reason the
    block dropdown exists at all, and silently jumping back to a default would
    make every comparison a two-click operation.  The preference is honoured
    only where that stage actually has data in the new block; otherwise the
    block opens where :func:`default_stage_index` says it should.

    Args:
        preferred_stage_idx: The stage currently on screen, or ``None`` on the
            first load, when there is no preference to carry.
        csv_paths: One entry per stage for the block being switched to.

    Returns:
        The index to show.

    Raises:
        ValueError: If *preferred_stage_idx* names no stage, or if *csv_paths*
            does not have one entry per stage.
    """
    if preferred_stage_idx is not None:
        if not 0 <= preferred_stage_idx < len(STAGE_LABELS):
            raise ValueError(
                f"preferred_stage_idx={preferred_stage_idx} names no "
                f"postprocessing stage; the viewer offers {len(STAGE_LABELS)} "
                f"(0..{len(STAGE_LABELS) - 1})."
            )
        if len(csv_paths) != len(STAGE_LABELS):
            raise ValueError(
                f"csv_paths must have one entry per stage "
                f"({len(STAGE_LABELS)}), got {len(csv_paths)}."
            )
        preferred_csv = csv_paths[preferred_stage_idx]
        if preferred_csv is not None and preferred_csv.exists():
            return preferred_stage_idx
    return default_stage_index(csv_paths)


# ---------------------------------------------------------------------------
# The session / block index
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockEntry:
    """One selectable block: what to call it, and what to resolve it from.

    Attributes:
        session_id: The session dropdown entry this block belongs to.
        block_id: The block dropdown entry, unique within *session_id*.
        recording_name: What the window title and the 3D scene label call this
            block.  Carried rather than derived, because deriving it would mean
            this Qt-free leaf knowing the shape of a ``KinectConfig``.
        config: The opaque object the viewer's stage-paths resolver was built to
            accept — in production a ``KinectConfig``.  **Deliberately typed
            ``Any`` and never touched here.** Nothing in this module reads it, so
            nothing in this module can be tempted to resolve it; that is the
            whole of how the index stays free of I/O.
    """

    session_id: str
    block_id: str
    recording_name: str
    config: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("session_id", "block_id", "recording_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"BlockEntry.{name} must be a non-empty string, got "
                    f"{value!r}. A blank dropdown entry cannot be selected back "
                    "to the block it stands for."
                )

    @property
    def key(self) -> Tuple[str, str]:
        """The identity of this block — what "same block" means to the viewer."""
        return (self.session_id, self.block_id)


class SessionBlockIndex:
    """Ordered sessions, and the ordered blocks of each — identifiers only.

    This is the whole of what the viewer needs to populate two of its three
    dropdowns, and deliberately not one byte more.  It performs **no I/O**, holds
    no stage paths, opens no point cloud and calls no resolver; see the module
    docstring for why that is the load-bearing property rather than an
    incidental one.

    Args:
        entries: The selectable blocks, in any order.  Sessions and blocks are
            each presented sorted, so the dropdowns do not depend on the order
            the batch happened to enumerate config files in.

    Raises:
        ValueError: If *entries* is empty, or if two entries share a
            ``(session_id, block_id)`` key.  A duplicate key would make one of
            the two blocks unreachable from the dropdown while both occupied a
            row in it.
    """

    def __init__(self, entries: Sequence[BlockEntry]) -> None:
        if not entries:
            raise ValueError(
                "SessionBlockIndex was built with no blocks. There is no "
                "selection a viewer could open on; the caller must decide what "
                "an empty batch means rather than opening an empty window."
            )

        by_key: Dict[Tuple[str, str], BlockEntry] = {}
        by_session: Dict[str, List[BlockEntry]] = {}
        for entry in entries:
            if not isinstance(entry, BlockEntry):
                raise TypeError(
                    f"SessionBlockIndex entries must be BlockEntry, got "
                    f"{type(entry).__name__}."
                )
            if entry.key in by_key:
                raise ValueError(
                    f"Duplicate block {entry.key}. Two entries under one "
                    "dropdown row means one of them can never be selected."
                )
            by_key[entry.key] = entry
            by_session.setdefault(entry.session_id, []).append(entry)

        self._by_key = by_key
        self._sessions: List[str] = sorted(by_session)
        self._blocks: Dict[str, List[str]] = {
            session_id: sorted(e.block_id for e in session_entries)
            for session_id, session_entries in by_session.items()
        }

    def __len__(self) -> int:
        """The number of selectable blocks, across every session."""
        return len(self._by_key)

    @property
    def sessions(self) -> List[str]:
        """Every session id, sorted — the session dropdown, in order."""
        return list(self._sessions)

    @property
    def n_sessions(self) -> int:
        """How many sessions the index covers."""
        return len(self._sessions)

    def blocks(self, session_id: str) -> List[str]:
        """The block ids of *session_id*, sorted — the block dropdown, in order.

        Raises:
            ValueError: If *session_id* is not in the index.  Repopulating the
                block dropdown for a session that does not exist would leave it
                empty, which reads as "this session has no blocks".
        """
        if session_id not in self._blocks:
            raise ValueError(
                f"Session {session_id!r} is not in this index; it holds "
                f"{self._sessions}."
            )
        return list(self._blocks[session_id])

    def entry(self, session_id: str, block_id: str) -> BlockEntry:
        """Return the block *block_id* of session *session_id*.

        Raises:
            ValueError: If no such block exists.  Every dropdown pair the user
                can produce comes from this index, so a miss means the two
                dropdowns are out of step — which is exactly the desync the
                viewer's revert guard exists to prevent, and it must be loud.
        """
        entry = self._by_key.get((session_id, block_id))
        if entry is None:
            known = self._blocks.get(session_id)
            where = (
                f"session {session_id!r} holds {known}"
                if known is not None
                else f"session {session_id!r} is not in this index"
            )
            raise ValueError(
                f"No block {block_id!r} in session {session_id!r}: {where}."
            )
        return entry

    def first_entry(self) -> BlockEntry:
        """The block the viewer opens on — the first block of the first session."""
        session_id = self._sessions[0]
        return self.entry(session_id, self._blocks[session_id][0])


def build_session_block_index(
    session_map: Mapping[str, Sequence[Any]],
) -> SessionBlockIndex:
    """Build a :class:`SessionBlockIndex` from the batch's session map.

    The same input the session-level forearm inspector takes
    (``Dict[str, List[KinectConfig]]``), so the two session-level viewers are
    driven from one structure rather than two.  Configs are duck-typed —
    ``block_id`` and ``source_video.stem`` are read and nothing else — which is
    what keeps this leaf importable without the primary-processing package.

    **Nothing is resolved.**  The configs are stored as opaque handles; the
    viewer calls its resolver on the one block the user selects.

    Args:
        session_map: session id → that session's block configs.

    Returns:
        The index.

    Raises:
        ValueError: If *session_map* is empty, or if any session maps to no
            blocks.  An empty session would put a selectable session in the
            dropdown whose block dropdown is blank.
    """
    if not session_map:
        raise ValueError(
            "session_map is empty; there is no session to show. The caller must "
            "decide what an empty batch means rather than opening a blank window."
        )

    entries: List[BlockEntry] = []
    for session_id, configs in session_map.items():
        if not configs:
            raise ValueError(
                f"Session {session_id!r} maps to no block configs. A session "
                "with no blocks is selectable but not viewable."
            )
        for config in configs:
            entries.append(
                BlockEntry(
                    session_id=str(session_id),
                    block_id=str(config.block_id),
                    recording_name=str(config.source_video.stem),
                    config=config,
                )
            )
    return SessionBlockIndex(entries)
