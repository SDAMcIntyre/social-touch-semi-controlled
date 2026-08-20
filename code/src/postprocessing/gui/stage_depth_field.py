"""Stage-aware policy for the postprocessing stage viewer's contact depth field.

This module is the whole of the *policy* between a stage the user selected in
the dropdown and the depth-field sidecar that stage may or may not carry: which
coordinate space that stage's field must declare, what to say when it is absent,
and what to refuse outright.  The viewer above it renders; the adapter below it
(:mod:`merging.contact_depth_field_series`) reads parquet.  Neither knows about
postprocessing stages, and that is deliberate.

Why this lives under ``gui/`` yet imports no GUI
------------------------------------------------
Locality: the only consumer is the stage viewer next door.  But every line here
is decidable from a stage index, a loader and a path, so nothing forces a Qt,
VTK or Open3D import — and consequently this file, unlike the 685-line widget
beside it, is testable headlessly.  The same arrangement makes
``merging/contact_depth_field_series.py`` testable while sitting beside a
viewer.  **Do not import Qt, VTK, PyVista or Open3D here.**

Six stages, four spaces, seven directories
------------------------------------------
These three counts are not the same, and conflating them is the mistake this
module exists to make impossible:

* Six **stages** — the entries of :data:`STAGE_LABELS`, which is what the
  dropdown shows.
* Four **coordinate spaces** — the closed vocabulary of
  ``contact_depth_field_io``.  Three consecutive stages share
  ``icp_registered``, because deduplication and projection re-address and drop
  rows but move no point.
* Seven pipeline **output directories** — ``blocks_filtered/`` exists and is not
  in the dropdown.  This is why stage 0 has no sidecar: it reads
  ``blocks_merged/``, written *before*
  ``filter_contact_depth_field_by_neural_quality``, whose sidecar lands in
  ``blocks_filtered/``.  Stage 0's absence is a fact about the pipeline, not a
  gap to be papered over by pointing it at another directory's file — hence it
  is absent from :data:`CANONICAL_SPACE_BY_STAGE` by construction, and resolving
  it never consults a loader at all.

Refusing a wrong-space field
----------------------------
``ContactDepthFieldSeries.coordinate_space`` has always carried the declared
space verbatim "so a caller can refuse to draw a wrong-space field".  Until now
no caller did: the Neural+Kinect viewer sees exactly one space.  This viewer
sees four across six stages, so this module is the first caller that actually
refuses.  A field drawn in the wrong space would land the patch somewhere
plausible-looking and wrong — a defect that first surfaces as a wrong scientific
result, downstream, in another repository.  Fail fast instead.

One stage accepts two spaces, and it is not a widening
------------------------------------------------------
``center_on_receptive_field`` cannot always estimate a receptive-field centre.
When it cannot, it copies the CSV *and* the sidecar through byte-for-byte and
deliberately leaves ``coordinate_space`` at ``pca_calibrated``, because that is
what the points are still in; restamping them ``rf_centered`` would assert a
translation that never happened.  The producer is right, and whole sessions land
there — ``2022-06-14_ST13-01`` does, on all four of its blocks.  So stage 5
accepts ``rf_centered`` **or** ``pca_calibrated`` and nothing else, expressed as
:data:`CANONICAL_SPACE_BY_STAGE` plus :data:`PASSTHROUGH_SPACE_BY_STAGE` rather
than as a hand-written set, so that widening one stage cannot widen another.
The asymmetry is load-bearing: stage 4 must still refuse ``rf_centered`` (a
field that moved *past* the stage being shown as though it had not), and stages
1-3 must still refuse everything but ``icp_registered``.

And the passthrough is announced, never absorbed.  Showing ``pca_calibrated``
data on a stage labelled "RF Centered" without saying so is a quieter version of
the lie the restamp would have told, so :attr:`StageDepthField.passthrough_note`
carries the fact up to a surface the user is already looking at.

Absent, corrupt, and no-contact are three different facts
---------------------------------------------------------
* **Absent** — the sidecar file does not exist.  :attr:`StageDepthField.series`
  is ``None`` with a non-empty :attr:`StageDepthField.message`.
* **Corrupt** — the file exists and cannot be decoded.  The ``ValueError`` from
  the reader propagates untouched; there is deliberately no ``except`` anywhere
  in this module.  Falling back to flat colour here would hide a broken artifact
  behind a plausible picture.
* **No contact this frame** — ``series.frame(i) is None``; the caller draws
  nothing.  Not this module's concern.

Two joins, not one
------------------
Drawing the field needs two lookups, and both live here so that neither is an
expression buried in a widget.  :func:`depth_frame_at_position` turns a slider
position into a Kinect frame and reads the patch.  :func:`forearm_depth_scalars`
then turns that frame's ``vertex_id`` column into a scalar per forearm vertex,
so the depth can be painted on the surface rather than only on the patch.  The
second join exists only from the projection stage onward and returns ``None``
before it; see the block comment above that function for why the missing case
must stay missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from merging.contact_depth_field_series import (
    ContactDepthFieldLoader,
    ContactDepthFieldSeries,
)
from preprocessing.motion_analysis.tactile_quantification.io.contact_depth_field_io import (
    COORDINATE_SPACE_ICP_REGISTERED,
    COORDINATE_SPACE_PCA_CALIBRATED,
    COORDINATE_SPACE_RF_CENTERED,
    VERTEX_ID_COLUMN,
    validate_vertex_ids_against_reference,
)

__all__ = [
    "ACCEPTED_SPACES_BY_STAGE",
    "CANONICAL_SPACE_BY_STAGE",
    "FOREARM_DEPTH_SCALAR_NAME",
    "FRAME_INDEX_COLUMN",
    "PASSTHROUGH_REASON_BY_STAGE",
    "PASSTHROUGH_SPACE_BY_STAGE",
    "PRODUCING_TASK_BY_STAGE",
    "STAGE_LABELS",
    "StageDepthField",
    "depth_frame_at_position",
    "forearm_depth_scalars",
    "kinect_frame_at_position",
    "kinect_frame_indices",
    "resolve_stage_depth_field",
]


#: The merged-CSV column carrying the Kinect frame each row belongs to.  The
#: depth-field sidecar is keyed by the same number, which is what makes the two
#: joinable at all.
FRAME_INDEX_COLUMN: str = "frame_index"

#: Name of the point-data array the forearm PLY carries when it is coloured by
#: depth.  Deliberately *not* ``CONTACT_SCALAR_NAME``: the two arrays live on
#: different datasets and mean different things — one is "the depth of this
#: contact vertex", the other "the depth of whatever touched this forearm vertex
#: on this frame, or nothing".  Sharing a name would make a mistaken
#: ``set_active_scalars`` on the wrong dataset silently plausible.
FOREARM_DEPTH_SCALAR_NAME: str = "forearm_penetration_depth_mm"


#: The postprocessing stages the viewer's dropdown offers, in order.
#:
#: Defined in this Qt-free leaf rather than in the widget beside it because the
#: policy below names stages in its error messages, and a second copy of these
#: six strings is a second copy that will drift.
#: ``postprocessing_stage_viewer`` re-exports this name, so every existing
#: importer is unaffected.
STAGE_LABELS: List[str] = [
    "Merged (Raw)",
    "ICP Registered",
    "Deduplicated",
    "Contact Projected",
    "PCA Calibrated",
    "RF Centered",
]

#: The coordinate space each stage's **own transform** produces, and therefore
#: the space its sidecar declares whenever that transform actually ran.
#:
#: Stages 1-3 share :data:`COORDINATE_SPACE_ICP_REGISTERED` because
#: deduplication and projection consume an index mapping rather than applying a
#: transform: rows are dropped and re-addressed, but no point moves, so the
#: space name written by ``apply_icp_registration`` stays correct through both.
#:
#: **Stage 0 is absent from this mapping by construction** — see the module
#: docstring.  Membership in this mapping is therefore the definition of
#: "sidecar-bearing stage", and the absence must not be repaired.
CANONICAL_SPACE_BY_STAGE: Mapping[int, str] = MappingProxyType(
    {
        1: COORDINATE_SPACE_ICP_REGISTERED,
        2: COORDINATE_SPACE_ICP_REGISTERED,
        3: COORDINATE_SPACE_ICP_REGISTERED,
        4: COORDINATE_SPACE_PCA_CALIBRATED,
        5: COORDINATE_SPACE_RF_CENTERED,
    }
)

#: The space a stage's sidecar declares when its producing task ran but applied
#: **no transform at all** — that task's documented passthrough branch.
#:
#: Exactly one stage has one, and it is a property of the *producer*, recorded
#: here rather than negotiated: ``center_on_receptive_field`` writes the
#: passthrough with ``_copy_field_unchanged``, a byte copy whose docstring
#: states the reason — "the points did not move, so the file's declared
#: ``coordinate_space`` — ``pca_calibrated`` — is still the truth".  A stage
#: with no entry here accepts exactly one space; adding an entry is a claim
#: about one named task's branch, not a relaxation of the check.
PASSTHROUGH_SPACE_BY_STAGE: Mapping[int, str] = MappingProxyType(
    {
        5: COORDINATE_SPACE_PCA_CALIBRATED,
    }
)

#: Why a passthrough stage's producing task may have applied no transform.
#:
#: Kept to what the artifacts support.  The branch that copies the field through
#: is entered on exactly one condition — ``rf_center is None`` — which covers
#: two recorded outcomes (``no_cluster_found`` and ``no_contact_points``); the
#: sidecar distinguishes neither, so the note names the file that does rather
#: than guessing between them.
PASSTHROUGH_REASON_BY_STAGE: Mapping[int, str] = MappingProxyType(
    {
        5: (
            "it could not estimate a receptive-field centre, so it copied its "
            "input through unchanged instead of translating it (the session's "
            "'rf_center_origin.json' records which no-centre outcome occurred)"
        ),
    }
)

if set(PASSTHROUGH_SPACE_BY_STAGE) - set(CANONICAL_SPACE_BY_STAGE):
    raise ValueError(
        "PASSTHROUGH_SPACE_BY_STAGE names a stage with no canonical space: "
        f"{sorted(set(PASSTHROUGH_SPACE_BY_STAGE) - set(CANONICAL_SPACE_BY_STAGE))}."
    )
if set(PASSTHROUGH_SPACE_BY_STAGE) != set(PASSTHROUGH_REASON_BY_STAGE):
    raise ValueError(
        "Every passthrough stage must carry the reason its transform was "
        "skipped. A passthrough the viewer cannot explain is one it must not "
        "silently accept."
    )
for _stage_idx, _passthrough in PASSTHROUGH_SPACE_BY_STAGE.items():
    if _passthrough == CANONICAL_SPACE_BY_STAGE[_stage_idx]:
        raise ValueError(
            f"Stage {_stage_idx}'s passthrough space equals its canonical space "
            f"('{_passthrough}'), so the passthrough could never be detected."
        )

#: The closed set of coordinate spaces each stage's sidecar may declare.
#:
#: **Derived, never hand-written.**  A hand-written set is how one stage's
#: legitimate second space becomes every stage's, which is precisely the
#: failure this arrangement exists to prevent: the union is taken per stage,
#: from that stage's own two mappings, so stage 5 gaining ``pca_calibrated``
#: leaves stage 4 accepting ``pca_calibrated`` alone and stages 1-3 accepting
#: ``icp_registered`` alone.
ACCEPTED_SPACES_BY_STAGE: Mapping[int, frozenset] = MappingProxyType(
    {
        stage_idx: frozenset(
            (
                canonical,
                *(
                    (PASSTHROUGH_SPACE_BY_STAGE[stage_idx],)
                    if stage_idx in PASSTHROUGH_SPACE_BY_STAGE
                    else ()
                ),
            )
        )
        for stage_idx, canonical in CANONICAL_SPACE_BY_STAGE.items()
    }
)

#: The postprocessing DAG task that writes each stage's sidecar, named in the
#: absent message so that "it is missing" comes with "here is how to produce
#: it".  Task names are the ones in ``configs/postprocess_workflow_kinect_auto_dag.yaml``.
PRODUCING_TASK_BY_STAGE: Mapping[int, str] = MappingProxyType(
    {
        1: "apply_icp_registration",
        2: "deduplicate_xy",
        3: "project_contacts_onto_forearm",
        4: "calibrate_pca_xyz",
        5: "center_on_receptive_field",
    }
)

#: The first stage that carries a depth field, pointed at from stage 0's message
#: so the user is told where to look rather than only what is missing.
_FIRST_SIDECAR_BEARING_STAGE: int = min(CANONICAL_SPACE_BY_STAGE)


@dataclass(frozen=True)
class StageDepthField:
    """One stage's depth field, or its announced absence.

    Absence is a *state* carried with its own explanation, never a silent drop
    into flat colouring: the message exists precisely so that surfacing it (as a
    disabled control's tooltip, or a console line) is easier than not surfacing
    it.

    Attributes:
        series: The stage's validated depth field, or ``None`` when the stage
            has no sidecar.  When present, its ``coordinate_space`` has already
            been checked against the stage.
        message: Non-empty human-readable text.  States either what was loaded
            or that depth colouring is unavailable for this stage and why.  When
            *passthrough_note* is set it is part of this text too, so a caller
            that surfaces only the message still surfaces the passthrough.
        passthrough_note: ``None`` on the ordinary path.  Non-empty text when
            this stage's producing task took its documented no-transform branch,
            so the field sits in :data:`PASSTHROUGH_SPACE_BY_STAGE` rather than
            in :data:`CANONICAL_SPACE_BY_STAGE` — accepted, but not silently.
            Showing ``pca_calibrated`` points on a stage labelled "RF Centered"
            without saying so is the same misstatement as restamping the file,
            made quieter; this is the field that stops it.

    Raises:
        ValueError: If *message* is empty, if *passthrough_note* is present but
            blank, or if a passthrough is claimed without a series to claim it
            about.
    """

    series: Optional[ContactDepthFieldSeries]
    message: str
    passthrough_note: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError(
                "message is empty. Both outcomes must be announceable; a silent "
                "absent state is exactly what this type exists to prevent."
            )
        if self.passthrough_note is not None:
            if not self.passthrough_note.strip():
                raise ValueError(
                    "passthrough_note is blank. It exists only to be shown; an "
                    "empty one announces the passthrough to nobody, which is "
                    "the state it was added to make impossible."
                )
            if self.series is None:
                raise ValueError(
                    "passthrough_note was set without a series. A passthrough "
                    "is a statement about the space a *loaded* field declares; "
                    "there is no such statement to make about an absent one."
                )

    @property
    def is_present(self) -> bool:
        """Whether this stage has a depth field to draw."""
        return self.series is not None

    @property
    def is_passthrough(self) -> bool:
        """Whether this stage resolved to its passthrough space, not its own.

        ``True`` means the field is real, validated and drawable, and that the
        producing task applied no transform — so the coordinates belong to the
        previous stage's space.  Callers must say so rather than quietly drawing
        it under this stage's label.
        """
        return self.passthrough_note is not None


def _stage_label(stage_idx: int) -> str:
    """Return the dropdown label for *stage_idx*.

    Args:
        stage_idx: Index into :data:`STAGE_LABELS`.

    Returns:
        The label shown in the dropdown.

    Raises:
        ValueError: If *stage_idx* names no stage.  A stage index the viewer
            cannot show is a wiring error, not a stage without a depth field.
    """
    if not 0 <= stage_idx < len(STAGE_LABELS):
        raise ValueError(
            f"stage_idx={stage_idx} names no postprocessing stage; the viewer "
            f"offers {len(STAGE_LABELS)} (0..{len(STAGE_LABELS) - 1}): "
            f"{STAGE_LABELS}."
        )
    return STAGE_LABELS[stage_idx]


def _stage_zero_message() -> str:
    """The message for the one stage that legitimately has no sidecar."""
    return (
        f"Stage 0 ('{STAGE_LABELS[0]}') reads 'blocks_merged/', which is written "
        "before the contact depth field is filtered by neural quality — that "
        "sidecar lands in 'blocks_filtered/', a directory this dropdown does not "
        "show. There is therefore no depth field in this stage's own coordinate "
        "space, and depth colouring is unavailable here. This is expected, not a "
        f"missing artifact; select stage {_FIRST_SIDECAR_BEARING_STAGE} "
        f"('{STAGE_LABELS[_FIRST_SIDECAR_BEARING_STAGE]}') for the first stage "
        "that carries one."
    )


def _absent_message(stage_idx: int, label: str, sidecar_path: Optional[Path]) -> str:
    """The message for a sidecar-bearing stage whose sidecar is not there."""
    task = PRODUCING_TASK_BY_STAGE[stage_idx]
    where = f"at '{sidecar_path}'" if sidecar_path is not None else "for this stage"
    return (
        f"No contact depth field {where} for stage {stage_idx} ('{label}'). "
        "Depth colouring is DISABLED for this stage; contact points render in "
        "flat colour. This is an absent artifact, not a display preference — run "
        f"the postprocessing DAG task '{task}' to produce it."
    )


def _accepted_spaces_phrase(stage_idx: int) -> str:
    """Name a stage's accepted spaces, canonical first, for an error message.

    Canonical first, and never sorted alphabetically: the first space named is
    the one the stage's own transform produces, and any second is a passthrough.
    Reordering them would put a stage's exceptional case ahead of its normal
    one in the one sentence a user reads when the check fires.
    """
    canonical = CANONICAL_SPACE_BY_STAGE[stage_idx]
    passthrough = PASSTHROUGH_SPACE_BY_STAGE.get(stage_idx)
    if passthrough is None:
        return f"'{canonical}'"
    return f"'{canonical}' or '{passthrough}'"


def _passthrough_note(stage_idx: int, label: str) -> str:
    """State that a stage is showing its passthrough space, and why.

    Deliberately says nothing the artifacts do not support: the space it is in,
    the space it is not, the task that skipped its transform, and where the
    reason is recorded.  The geometry is correct — the CSV beside the sidecar
    was copied through the same branch, so the two still agree; it is the stage
    *label* that overstates what happened, which is the whole of what this says.
    """
    canonical = CANONICAL_SPACE_BY_STAGE[stage_idx]
    passthrough = PASSTHROUGH_SPACE_BY_STAGE[stage_idx]
    reason = PASSTHROUGH_REASON_BY_STAGE[stage_idx]
    task = PRODUCING_TASK_BY_STAGE[stage_idx]
    return (
        f"PASSTHROUGH: these coordinates are in '{passthrough}', not "
        f"'{canonical}'. Stage {stage_idx} ('{label}') is showing the output of "
        f"'{task}' on a session where {reason}. The points and the CSV beside "
        "them went through the same branch, so the picture is consistent; the "
        "stage label is the only thing here that claims a transform which did "
        "not happen."
    )


def resolve_stage_depth_field(
    stage_idx: int,
    loader: Optional[ContactDepthFieldLoader],
    sidecar_path: Optional[Path],
) -> StageDepthField:
    """Resolve the depth field belonging to one postprocessing stage.

    This is the single place where a stage index meets a depth field.  It calls
    the loader (which is where the parquet read actually happens), type-checks
    what came back, and validates the declared coordinate space against the
    stage the user selected.

    Stage 0 short-circuits: it is absent from :data:`ACCEPTED_SPACES_BY_STAGE`,
    so no loader is consulted for it even if one was wired.  There is no
    coordinate space it could legitimately declare, so there is nothing to
    validate a series against, and reading a file to then refuse it would be
    worse than not reading it.

    A stage that has a :data:`PASSTHROUGH_SPACE_BY_STAGE` entry accepts that
    space as well as its own, and the returned
    :attr:`StageDepthField.passthrough_note` says which of the two it got.  The
    acceptance is per stage and derived, so it cannot leak sideways: stage 4
    still refuses ``rf_centered`` and stages 1-3 still refuse everything but
    ``icp_registered``.

    Args:
        stage_idx: Index into :data:`STAGE_LABELS`.
        loader: The stage's zero-argument depth-field loader, or ``None`` when
            no depth field was wired for it at all.
        sidecar_path: Where this stage's sidecar was looked for.  Used **only**
            for messages and errors; nothing here opens it.  It travels
            alongside the loader because a
            :data:`~merging.contact_depth_field_series.ContactDepthFieldLoader`
            deliberately hides its path and ``ContactDepthFieldSeries`` carries
            none either, yet an error that says "wrong space" without saying
            *which file* is not actionable. The caller forwards the path its
            resolver already derived; it must not derive one itself.

    Returns:
        A :class:`StageDepthField` — present with a space-validated series, or
        absent with a message stating why.

    Raises:
        ValueError: If *stage_idx* names no stage; if a loader was supplied
            without the path it was built from; or if the loaded field declares
            a coordinate space this stage accepts neither as its own output nor
            as its passthrough.
        TypeError: If the loader returns something that is neither a
            :class:`~merging.contact_depth_field_series.ContactDepthFieldSeries`
            nor ``None``.
        Exception: Whatever the loader raises for a **corrupt** sidecar
            propagates untouched — most usually ``ValueError`` from the reader.
            There is no ``except`` in this function on purpose: a present but
            undecodable artifact must not be laundered into the absent state.
    """
    label = _stage_label(stage_idx)

    accepted_spaces = ACCEPTED_SPACES_BY_STAGE.get(stage_idx)
    if accepted_spaces is None:
        return StageDepthField(series=None, message=_stage_zero_message())

    if loader is None:
        return StageDepthField(
            series=None, message=_absent_message(stage_idx, label, sidecar_path)
        )

    if sidecar_path is None:
        raise ValueError(
            f"Stage {stage_idx} ('{label}') was given a depth-field loader but "
            "no sidecar_path. The two travel together: the loader hides the "
            "path it reads, so without it a space-mismatch or absence message "
            "cannot name the file it is about."
        )

    series = loader()

    if series is None:
        return StageDepthField(
            series=None, message=_absent_message(stage_idx, label, sidecar_path)
        )

    if not isinstance(series, ContactDepthFieldSeries):
        raise TypeError(
            f"Stage {stage_idx} ('{label}'): the depth-field loader returned "
            f"{type(series).__name__}; it must return a ContactDepthFieldSeries "
            "or None. Drawing whatever this is would put unvalidated geometry "
            "in the scene."
        )

    if series.coordinate_space not in accepted_spaces:
        raise ValueError(
            f"Stage {stage_idx} ('{label}') expects a contact depth field in "
            f"coordinate space {_accepted_spaces_phrase(stage_idx)}, but "
            f"'{sidecar_path}' declares '{series.coordinate_space}'. Refusing to "
            "draw it: the depth values would be correct and the positions would "
            "not, which renders as a plausible patch in the wrong place rather "
            "than as an error."
        )

    note = (
        _passthrough_note(stage_idx, label)
        if series.coordinate_space == PASSTHROUGH_SPACE_BY_STAGE.get(stage_idx)
        else None
    )
    message = (
        f"Contact depth field for stage {stage_idx} ('{label}') loaded from "
        f"'{sidecar_path}': {series.summary()}."
    )
    if note is not None:
        # Folded into the message as well as carried separately, so a caller
        # that only ever surfaces ``message`` still surfaces the passthrough.
        message = f"{message} {note}"

    return StageDepthField(series=series, message=message, passthrough_note=note)


# ---------------------------------------------------------------------------
# The frame join
# ---------------------------------------------------------------------------
#
# The single most dangerous shortcut available to a consumer of this data is to
# index the depth field by the slider's position.  It is dangerous precisely
# because it *works*: it produces a smooth, plausible animation of the right
# shape, in the right place, of the wrong frames.
#
# The two sides are keyed differently.  A depth-field sidecar is keyed by Kinect
# ``frame_index``, and only frames that carry contact appear in it at all.  The
# merged CSV is upsampled to the nerve sampling rate (~33 rows per Kinect
# frame); the viewer's ``_kinect_df`` keeps only the anchor rows, so its row
# *positions* are a dense 0..N-1 enumeration of the frames that survived, which
# equals ``frame_index`` only when the block starts at frame 0 and no frame was
# ever dropped.  Neither is guaranteed, and neither is checkable from the shape
# of the data — a positional join is silent when it is wrong.
#
# So the join goes through the column, always, and its absence raises.


def kinect_frame_indices(
    kinect_df: Any,
    source: Any = "<stage CSV>",
) -> np.ndarray:
    """Return the Kinect frame index of every displayable row, in slider order.

    The returned array is positional: element *p* is the Kinect frame the viewer
    shows at slider position *p*.  It is the only bridge between a slider
    position and a depth-field lookup key, and building it is the reason a
    missing column has to be fatal rather than papered over.

    Args:
        kinect_df: The stage's frame-anchor rows (``time_kinect`` non-NaN), in
            display order.  Duck-typed: anything with ``columns`` and
            ``__getitem__`` returning a ``to_numpy``-able column.
        source: Names the offending artifact in error messages only.

    Returns:
        ``(len(kinect_df),)`` int64 Kinect frame indices.

    Raises:
        ValueError: If the column is absent, holds a missing value, or holds a
            non-integral value.  Every one of these means the row's frame is
            unknown, and the only alternative to raising is a positional join
            that draws another frame's data without saying so.
    """
    if FRAME_INDEX_COLUMN not in kinect_df.columns:
        raise ValueError(
            f"'{source}' has no '{FRAME_INDEX_COLUMN}' column, so no row can be "
            "matched to a Kinect frame. The contact depth field is keyed by "
            f"'{FRAME_INDEX_COLUMN}'; joining it by row position instead would "
            "draw a different frame's depths at every position and would look "
            f"entirely plausible while doing it. Columns found: "
            f"{list(kinect_df.columns)}"
        )

    values = np.asarray(kinect_df[FRAME_INDEX_COLUMN].to_numpy(dtype=np.float64))
    if values.size and not np.all(np.isfinite(values)):
        missing = int(np.count_nonzero(~np.isfinite(values)))
        raise ValueError(
            f"'{source}' has {missing} row(s) whose '{FRAME_INDEX_COLUMN}' is "
            "missing or non-finite, among the rows the viewer treats as Kinect "
            "frames. A frame with no index cannot be looked up in the depth "
            "field."
        )
    if values.size and not np.all(values == np.floor(values)):
        bad = values[values != np.floor(values)][:5]
        raise ValueError(
            f"'{source}' has non-integral '{FRAME_INDEX_COLUMN}' values (e.g. "
            f"{bad.tolist()}). A fractional frame addresses nothing."
        )
    return values.astype(np.int64)


def depth_frame_at_position(
    series: ContactDepthFieldSeries,
    frame_indices: np.ndarray,
    position: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return the depth field of the frame shown at slider *position*.

    This is the join, in one place, so that it is testable without a window and
    so there is exactly one expression in the codebase that turns a slider
    position into a depth-field key.

    Args:
        series: The stage's depth field.
        frame_indices: The map from :func:`kinect_frame_indices`.
        position: Slider position — a *row position*, not a frame index.

    Returns:
        ``(points, penetration_depth_mm)`` for the frame at *position*, or
        ``None`` when that frame carries no contact.  ``None`` is a fact about
        the recording, not a lookup failure.

    Raises:
        TypeError: If *series* is ``None``.  A caller that has no series must
            not reach the join at all; arriving here with one missing means the
            "is depth colouring active" test was skipped somewhere.
        IndexError: If *position* is outside *frame_indices*.  Clamping it would
            silently repeat an end frame.
    """
    if series is None:
        raise TypeError(
            "depth_frame_at_position was called without a depth field. Check "
            "StageDepthField.is_present before joining; there is nothing to "
            "look up and nothing sensible to return."
        )
    return series.frame(kinect_frame_at_position(frame_indices, position))


def kinect_frame_at_position(frame_indices: np.ndarray, position: int) -> int:
    """Return the Kinect frame index the viewer shows at slider *position*.

    The one expression in the codebase that turns a slider position into a
    depth-field key.  Both joins — the contact patch and the forearm surface —
    go through it, so neither can drift into indexing the field positionally
    while the other does not.

    Args:
        frame_indices: The map from :func:`kinect_frame_indices`.
        position: Slider position — a *row position*, not a frame index.

    Returns:
        The Kinect ``frame_index`` of that row.

    Raises:
        IndexError: If *position* is outside *frame_indices*.  Clamping it would
            silently repeat an end frame.
    """
    if not 0 <= position < len(frame_indices):
        raise IndexError(
            f"slider position {position} is outside the {len(frame_indices)} "
            "frames this stage has. It cannot be clamped: the neighbouring "
            "frame's depths are not this frame's."
        )
    return int(frame_indices[position])


# ---------------------------------------------------------------------------
# The forearm join
# ---------------------------------------------------------------------------
#
# Colouring the contact patch answers "how deep was each contact vertex".
# Colouring the forearm answers "how deep was this piece of skin pressed", which
# is the question the surface itself asks, and it needs a second join: from the
# sidecar's ``vertex_id`` to a row of the forearm PLY's vertex array.
#
# Three things make that join dangerous enough to be worth isolating here.
#
# 1. ``vertex_id`` exists only from the projection stage onward.  Stages before
#    it have none, and the honest answer for them is *nothing to show* — never a
#    nearest-vertex snap computed here, which picks different vertices than the
#    projection stage did (``depth_field_stage_io`` documents why: the CSV's
#    points came back through a ``%.1f`` round trip and the parquet's did not)
#    and which has already been measured overshooting a surface by 29 mm in the
#    analogous case.
# 2. The index is only meaningful against the exact mesh it was assigned to.  A
#    forearm re-deduplicated at a different epsilon renumbers every vertex, and
#    because task idempotency is decided from file timestamps nothing upstream
#    notices.  ``validate_vertex_ids_against_reference`` is therefore called
#    *before* any scatter, never after and never conditionally.
# 3. A vertex nobody touched is **missing**, not zero-depth.  It gets ``NaN``,
#    which the renderer paints in its own flat colour, so it is distinguishable
#    from a genuine 0.00 mm grazing contact at the edge of the patch.


def forearm_depth_scalars(
    series: ContactDepthFieldSeries,
    frame_index: int,
    vertex_count: int,
) -> Optional[np.ndarray]:
    """Scatter one frame's penetration depths onto a forearm's vertex array.

    Args:
        series: The stage's depth field.
        frame_index: The **Kinect frame index** to draw — not a slider position.
            Convert one to the other with :func:`kinect_frame_indices`.
        vertex_count: ``len(ply.points)`` of the forearm this stage displays.
            A count, never a mesh: the validator behind this function must stay
            importable where no geometry engine is installed.

    Returns:
        A ``(vertex_count,)`` float64 array of penetration depths in
        millimetres, ``NaN`` at every vertex this frame did not touch — or
        ``None`` when *series* carries no ``vertex_id`` at all, which is the
        case for every stage before projection.  ``None`` means "this stage
        cannot answer the question"; an all-``NaN`` array means "it can, and the
        answer for this frame is that nothing was touched".

    Raises:
        TypeError: If *series* is ``None``.  A caller with no field must not
            reach the join; arriving here with one missing means an
            ``is_present`` check was skipped.
        ValueError: If *vertex_count* is not a positive integer, or if the
            sidecar's recorded reference-PLY vertex count disagrees with it, or
            if any ``vertex_id`` falls outside the mesh.  A disagreement means
            the ids were assigned against a different mesh, so every one of them
            points at a different vertex than it did when it was written.
    """
    if series is None:
        raise TypeError(
            "forearm_depth_scalars was called without a depth field. Check "
            "StageDepthField.is_present before joining; there is nothing to "
            "look up and nothing sensible to return."
        )
    if series.vertex_id_by_frame is None:
        return None

    ids = series.vertex_id_by_frame.get(int(frame_index))
    if ids is None:
        # No contact on this frame.  The provenance is still checked below,
        # against an empty index, so a mismatched forearm fails on the first
        # frame drawn rather than on the first frame that happens to touch.
        ids = np.empty((0,), dtype=np.int32)
        depths = np.empty((0,), dtype=np.float64)
    else:
        depths = series.penetration_depth_by_frame[int(frame_index)]

    # First, always, and with the count the caller actually holds.  The count
    # check inside runs before the range check on purpose: ids that happen to
    # remain in range after a renumbering are the dangerous case.
    validate_vertex_ids_against_reference(
        pd.DataFrame({VERTEX_ID_COLUMN: ids}),
        dict(series.reference_ply_provenance),
        reference_vertex_count=vertex_count,
        reference_description="the forearm this stage displays",
    )

    scalars = np.full(int(vertex_count), np.nan, dtype=np.float64)
    if ids.size:
        # Projection is a per-point nearest-neighbour lookup with no uniqueness
        # constraint, so two contact points of one frame may legitimately land
        # on the same forearm vertex.  Plain assignment would resolve that by
        # row order — an arbitrary, silent choice that changes with a re-sort.
        # The deepest of the two wins instead: order-independent, and the
        # conservative reading of "how hard was this piece of skin pressed".
        # Seeding the touched entries with -inf keeps NaN meaning *untouched*,
        # since ``maximum`` propagates NaN and would otherwise erase them.
        scalars[ids] = -np.inf
        np.maximum.at(scalars, ids, depths)
    return scalars
