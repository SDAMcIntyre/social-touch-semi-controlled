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
  is absent from :data:`EXPECTED_SPACE_BY_STAGE` by construction, and resolving
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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import List, Mapping, Optional

from merging.contact_depth_field_series import (
    ContactDepthFieldLoader,
    ContactDepthFieldSeries,
)
from preprocessing.motion_analysis.tactile_quantification.io.contact_depth_field_io import (
    COORDINATE_SPACE_ICP_REGISTERED,
    COORDINATE_SPACE_PCA_CALIBRATED,
    COORDINATE_SPACE_RF_CENTERED,
)

__all__ = [
    "EXPECTED_SPACE_BY_STAGE",
    "PRODUCING_TASK_BY_STAGE",
    "STAGE_LABELS",
    "StageDepthField",
    "resolve_stage_depth_field",
]


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

#: The coordinate space each stage's sidecar **must** declare.
#:
#: Stages 1-3 share :data:`COORDINATE_SPACE_ICP_REGISTERED` because
#: deduplication and projection consume an index mapping rather than applying a
#: transform: rows are dropped and re-addressed, but no point moves, so the
#: space name written by ``apply_icp_registration`` stays correct through both.
#:
#: **Stage 0 is absent from this mapping by construction** — see the module
#: docstring.  Membership in this mapping is therefore the definition of
#: "sidecar-bearing stage", and the absence must not be repaired.
EXPECTED_SPACE_BY_STAGE: Mapping[int, str] = MappingProxyType(
    {
        1: COORDINATE_SPACE_ICP_REGISTERED,
        2: COORDINATE_SPACE_ICP_REGISTERED,
        3: COORDINATE_SPACE_ICP_REGISTERED,
        4: COORDINATE_SPACE_PCA_CALIBRATED,
        5: COORDINATE_SPACE_RF_CENTERED,
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
_FIRST_SIDECAR_BEARING_STAGE: int = min(EXPECTED_SPACE_BY_STAGE)


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
            or that depth colouring is unavailable for this stage and why.

    Raises:
        ValueError: If *message* is empty.
    """

    series: Optional[ContactDepthFieldSeries]
    message: str

    def __post_init__(self) -> None:
        if not self.message.strip():
            raise ValueError(
                "message is empty. Both outcomes must be announceable; a silent "
                "absent state is exactly what this type exists to prevent."
            )

    @property
    def is_present(self) -> bool:
        """Whether this stage has a depth field to draw."""
        return self.series is not None


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

    Stage 0 short-circuits: it is absent from :data:`EXPECTED_SPACE_BY_STAGE`,
    so no loader is consulted for it even if one was wired.  There is no
    coordinate space it could legitimately declare, so there is nothing to
    validate a series against, and reading a file to then refuse it would be
    worse than not reading it.

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
            a coordinate space other than the one this stage produces.
        TypeError: If the loader returns something that is neither a
            :class:`~merging.contact_depth_field_series.ContactDepthFieldSeries`
            nor ``None``.
        Exception: Whatever the loader raises for a **corrupt** sidecar
            propagates untouched — most usually ``ValueError`` from the reader.
            There is no ``except`` in this function on purpose: a present but
            undecodable artifact must not be laundered into the absent state.
    """
    label = _stage_label(stage_idx)

    expected_space = EXPECTED_SPACE_BY_STAGE.get(stage_idx)
    if expected_space is None:
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

    if series.coordinate_space != expected_space:
        raise ValueError(
            f"Stage {stage_idx} ('{label}') expects a contact depth field in "
            f"coordinate space '{expected_space}', but '{sidecar_path}' declares "
            f"'{series.coordinate_space}'. Refusing to draw it: the depth values "
            "would be correct and the positions would not, which renders as a "
            "plausible patch in the wrong place rather than as an error."
        )

    return StageDepthField(
        series=series,
        message=(
            f"Contact depth field for stage {stage_idx} ('{label}') loaded from "
            f"'{sidecar_path}': {series.summary()}."
        ),
    )
