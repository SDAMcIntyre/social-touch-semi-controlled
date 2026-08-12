"""Standalone proof-of-concept driver for the per-vertex contact depth field.

Runs :func:`signed_contact_depth_mm` over **one recording** (one session, one
block) and holds the resulting per-frame field series in memory, ready for the
interactive viewer.

Pipeline shape
--------------
This module is filter ``[1]`` of three; it loads geometry and drives the
compute, it does not implement either::

    [1] load_recording_geometry()   -> hand meshes, forearm meshes, timestamps   <- HERE
    [2] signed_contact_depth_mm()   -> ContactDepthFrame                          (pure)
    [3] ContactDepthFieldViewer     -> interactive 3D render                      (pure sink)

Accordingly this file must never contain rendering code and must never import
PyVista, VTK or Qt.  The viewer is reached through the single seam
:func:`launch_viewer`.

Deliberate proof-of-concept debt
--------------------------------
Listed explicitly so it stays visible rather than becoming silent permanent
infrastructure:

* **No persistence.**  Nothing is written to disk.  The long-form record is
  fixed in the plan; the Parquet sidecar writer is not built.
* **Whole recording in memory.**  Every frame's field is retained; the footprint
  is reported at the end so it can size the future sidecar.
* **Hardcoded paths.**  The ``__main__`` configuration block names one
  recording, following the convention of
  ``code/scripts/_3_preprocessing/_2_hand_tracking/stabilise_hand_motion.py``.
* **No DAG integration.**  No Prefect flow, no task in any ``configs/*_dag.yaml``.
* **One recording at a time.**  No session-level or multi-block batching.

Decision record
---------------
Sign convention, units and query direction are defined once, in
``preprocessing/motion_analysis/tactile_quantification/model/contact_depth_field.py``.
In one line each: the raycasting scene is built from the **hand** and queried at
**forearm** vertices; ``signed_depth_mm < 0`` means penetrating; everything is
in **millimetres**, Kinect Space 1, with no unit conversion anywhere.

Space 1 only.  The forearm geometry must come from the per-block
``forearm_pointclouds/*_mesh.obj``.  Pairing a Space-1 hand with a downstream
``forearm_rf_centered/`` or ``forearm_pca_calibrated/`` mesh produces silently
garbage depths.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

# CuPy must be initialised while NumPy's dtype registry is still pristine, i.e.
# before Open3D / pyk4a C-extensions are loaded by `open3d` or `preprocessing.*`.
# A lazy import inside a function does not work.  See
# docs/development/knowledge-base/note-cupy-import-order.md.
try:  # noqa: SIM105 - the guard is the point; CPU-only machines must still run.
    import cupy  # noqa: F401  - imported for its side effect on import order.
except Exception as _cupy_exc:  # pragma: no cover - environment dependent.
    print(f"CuPy unavailable ({type(_cupy_exc).__name__}: {_cupy_exc}); continuing on CPU.")
    del _cupy_exc

import numpy as np
import open3d as o3d
import trimesh

from preprocessing.forearm_extraction import (
    ForearmCatalog,
    ForearmFrameParametersFileHandler,
    ForearmParameters,
    get_forearms_with_fallback,
)
from preprocessing.forearm_extraction.models.forearm_catalog import VideoIdentifier
from preprocessing.motion_analysis import (
    HandMetadataFileHandler,
    HandMetadataManager,
    HandMotionManager,
)
from preprocessing.motion_analysis.tactile_quantification.model.contact_depth_field import (
    EPSILON,
    ContactDepthFrame,
    signed_contact_depth_mm,
    validate_pose_transform,
)

__all__ = [
    "ContactDepthFieldSeries",
    "FrameOutcome",
    "FrameStatus",
    "RecordingGeometry",
    "compute_depth_field_series",
    "format_recording_report",
    "launch_viewer",
    "load_recording_geometry",
    "run_contact_depth_field_poc",
]


# =============================================================================
# Contracts
# =============================================================================


class FrameStatus(str, Enum):
    """Why a frame does or does not carry a depth field.

    ``NO_CONTACT`` and ``POSE_ABSENT`` are *different facts* and must never
    collapse into one another, nor into a zero-depth frame: once this field
    weights neural firing rate, an absent-read-as-zero silently down-weights
    real spikes.
    """

    CONTACT = "contact"
    NO_CONTACT = "no_contact"
    POSE_ABSENT = "pose_absent"


@dataclass(frozen=True)
class FrameOutcome:
    """The result of processing one frame of the recording.

    Attributes:
        frame_index: Kinect frame index; the position in the motion sequence.
        time_s: Frame timestamp in seconds, taken from the motion NPZ.
        status: Which of the three outcomes occurred.
        field: The per-vertex field, present if and only if ``status`` is
            :attr:`FrameStatus.CONTACT`.
        exclusion_reason: Why the frame carries no field, present if and only if
            ``status`` is :attr:`FrameStatus.POSE_ABSENT`.
    """

    frame_index: int
    time_s: float
    status: FrameStatus
    field: Optional[ContactDepthFrame] = None
    exclusion_reason: Optional[str] = None

    def __post_init__(self) -> None:
        has_field = self.field is not None
        if has_field != (self.status is FrameStatus.CONTACT):
            raise AssertionError(
                f"Frame {self.frame_index}: status {self.status.value} is "
                f"inconsistent with field presence ({has_field}). A field exists "
                "if and only if the frame is a contact frame."
            )
        has_reason = self.exclusion_reason is not None
        if has_reason != (self.status is FrameStatus.POSE_ABSENT):
            raise AssertionError(
                f"Frame {self.frame_index}: status {self.status.value} is "
                f"inconsistent with an exclusion reason ({has_reason}). Only an "
                "absent pose carries a reason, and it must always carry one."
            )


@dataclass(frozen=True)
class ContactDepthFieldSeries:
    """The whole recording's field series, plus the statistics the viewer needs.

    Every statistic the viewer would otherwise have to derive is computed here,
    once, over the whole recording.  In particular ``clim_penetration_mm`` is
    global: per-frame autoscaling would make the animation lie about relative
    depth.

    Attributes:
        outcomes: One entry per frame, in frame order.  Index-aligned with the
            recording, so ``outcomes[i].frame_index == i``.
        clim_penetration_mm: ``(low, high)`` colour limits in *penetration*
            millimetres (``-signed_depth_mm``, so positive = deeper), or ``None``
            when the recording contains no contact at all.  ``None`` is an
            explicit "undefined" marker, never a fabricated ``(0, 1)``.
        signed_depth_range_mm: ``(min, max)`` of the raw signed field across the
            recording, or ``None`` when there is no contact.
        field_array_bytes: Total ``nbytes`` of the retained field arrays.
        repeated_pose_frame_indices: Frames whose pose is bit-identical to the
            preceding frame — see :meth:`RecordingGeometry.hand_mesh`.
    """

    outcomes: Tuple[FrameOutcome, ...]
    clim_penetration_mm: Optional[Tuple[float, float]]
    signed_depth_range_mm: Optional[Tuple[float, float]]
    field_array_bytes: int
    repeated_pose_frame_indices: Tuple[int, ...]

    @classmethod
    def from_outcomes(
        cls,
        outcomes: Sequence[FrameOutcome],
        repeated_pose_frame_indices: Sequence[int],
    ) -> "ContactDepthFieldSeries":
        """Derive the recording-wide statistics from the per-frame outcomes."""
        fields = [o.field for o in outcomes if o.field is not None]

        if fields:
            minimum = float(min(float(np.min(f.signed_depth_mm)) for f in fields))
            maximum = float(max(float(np.max(f.signed_depth_mm)) for f in fields))
            signed_range: Optional[Tuple[float, float]] = (minimum, maximum)
            # Penetration is the negated signed depth, so the bounds swap.
            clim: Optional[Tuple[float, float]] = (-maximum, -minimum)
        else:
            signed_range = None
            clim = None

        field_bytes = sum(
            f.points.nbytes + f.signed_depth_mm.nbytes + f.normals.nbytes for f in fields
        )

        return cls(
            outcomes=tuple(outcomes),
            clim_penetration_mm=clim,
            signed_depth_range_mm=signed_range,
            field_array_bytes=field_bytes,
            repeated_pose_frame_indices=tuple(repeated_pose_frame_indices),
        )

    def count(self, status: FrameStatus) -> int:
        """Number of frames with the given status."""
        return sum(1 for outcome in self.outcomes if outcome.status is status)


@dataclass(frozen=True)
class RecordingGeometry:
    """Everything one recording needs to produce a depth field series.

    The hand meshes are *not* materialised as a list.  A 3000-frame recording
    would hold roughly 110 MB of Open3D meshes for no benefit: each mesh is
    consumed once by the compute stage, and the viewer can regenerate any single
    frame on demand for far less than the cost of an SDF query.
    :meth:`hand_mesh` is the one place that turns a frame index into geometry,
    so the compute stage and the viewer cannot disagree about what "the hand at
    frame *i*" means.

    Attributes:
        hand_motion_path: The NPZ actually consumed — raw or ``_stabilised``.
            Depth magnitudes are sensitive to the pose-smoothing configuration,
            so this is provenance, not decoration.
        current_video_filename: The ``.mp4`` name; must contain ``_block-orderNN``.
        motion_manager: Loaded hand poses.  Read-only from here on.
        forearm_meshes_by_frame: Static forearm terrain keyed by the frame index
            at which it takes effect.  Key ``0`` is required.
        forearm_stems_by_frame: Output stem of the snapshot behind each key that
            came from *this* block; keys absent here were supplied by the
            fallback and are explained by ``forearm_fallback_notices``.
        forearm_fallback_notices: Log records emitted by
            ``get_forearms_with_fallback`` during the load.  The function returns
            geometry only, so this capture is the sole provenance channel.
        excluded_vertex_ids: Hand vertices removed before contact processing.
        selected_point_labels: Anatomical landmark labels from the hand
            metadata.  Unused by the depth field; reported for provenance.
        repeated_pose_frame_indices: Frames whose ``(translation, rotation,
            scale)`` triple is bit-identical to the previous frame's.
    """

    hand_motion_path: Path
    current_video_filename: str
    motion_manager: HandMotionManager
    forearm_meshes_by_frame: Dict[int, o3d.geometry.TriangleMesh]
    forearm_stems_by_frame: Dict[int, str]
    forearm_fallback_notices: Tuple[str, ...]
    excluded_vertex_ids: Tuple[int, ...]
    selected_point_labels: Tuple[str, ...]
    repeated_pose_frame_indices: Tuple[int, ...]

    def __len__(self) -> int:
        return len(self.motion_manager)

    def time_s(self, frame_index: int) -> float:
        """Timestamp of ``frame_index``, in seconds."""
        return float(self.motion_manager.timestamps[frame_index])

    def pose_matrix(self, frame_index: int) -> np.ndarray:
        """The ``(4, 4)`` world transform ``HandMotionManager`` applies at ``frame_index``.

        Mirrors ``HandMotionManager.__getitem__``, which builds ``T * R * S``
        internally and exposes no accessor for it.  The matrix is needed here
        because :func:`validate_pose_transform` must see it *before* the mesh is
        built — the pure field function only ever receives an already
        transformed mesh and so cannot detect a winding-inverting transform.
        """
        rotation_xyzw = np.asarray(self.motion_manager.rotations[frame_index])
        matrix = trimesh.transformations.quaternion_matrix(np.roll(rotation_xyzw, 1))
        matrix[:3, :3] *= self.motion_manager.scales[frame_index]
        matrix[:3, 3] = np.asarray(self.motion_manager.translations[frame_index])
        return matrix

    def pose_is_finite(self, frame_index: int) -> bool:
        """Whether the stored pose and vertices for ``frame_index`` are all finite."""
        manager = self.motion_manager
        return bool(
            np.all(np.isfinite(manager.translations[frame_index]))
            and np.all(np.isfinite(manager.rotations[frame_index]))
            and np.isfinite(manager.scales[frame_index])
            and np.all(np.isfinite(manager.vertices_sequence[frame_index]))
        )

    def hand_mesh(self, frame_index: int) -> o3d.geometry.TriangleMesh:
        """Return a fresh, exclusion-filtered world-space hand mesh.

        ``remove_vertices_by_index`` mutates in place, which is safe only
        because ``HandMotionManager.__getitem__`` builds a new mesh on every
        call.  Replicates the controller's ordering: the exclusion happens
        before contact computation.
        """
        mesh = self.motion_manager[frame_index]
        if self.excluded_vertex_ids:
            mesh.remove_vertices_by_index(list(self.excluded_vertex_ids))
        return mesh


# =============================================================================
# Filter [1] — load one recording's geometry
# =============================================================================


class _LogRecordCollector(logging.Handler):
    """Collects formatted log messages emitted while it is attached."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: List[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(f"{record.levelname}: {record.getMessage()}")


@contextmanager
def _capture_root_logging() -> Iterator[_LogRecordCollector]:
    """Temporarily capture root-logger records without suppressing them."""
    collector = _LogRecordCollector()
    root = logging.getLogger()
    previous_level = root.level
    if previous_level > logging.INFO:
        root.setLevel(logging.INFO)
    root.addHandler(collector)
    try:
        yield collector
    finally:
        root.removeHandler(collector)
        root.setLevel(previous_level)


def _find_repeated_pose_frames(motion_manager: HandMotionManager) -> Tuple[int, ...]:
    """Frames whose pose repeats the previous frame's bit-for-bit.

    ``HandMotionManager`` has no representation for "pose absent": when sticker
    tracking drops out it copies the previous frame's transform and stores the
    result indistinguishably from a tracked frame.  A bit-identical repeat of a
    Procrustes/basis fit over noisy sticker coordinates is the signature of that
    carry-forward branch.

    This is a *diagnostic*, not a classification: it does not change any frame's
    :class:`FrameStatus`.  It exists so a recording that is largely carried
    forward cannot be mistaken for a fully tracked one.
    """
    repeated: List[int] = []
    for index in range(1, len(motion_manager)):
        if (
            np.array_equal(motion_manager.translations[index], motion_manager.translations[index - 1])
            and np.array_equal(motion_manager.rotations[index], motion_manager.rotations[index - 1])
            and motion_manager.scales[index] == motion_manager.scales[index - 1]
        ):
            repeated.append(index)
    return tuple(repeated)


def _block_snapshot_stems(
    forearm_params: Sequence[ForearmParameters],
    current_video_filename: str,
) -> Dict[int, str]:
    """Output stems of the forearm snapshots belonging to this recording's block.

    Mirrors the block-matching ``get_forearms_with_fallback`` performs, so the
    keys reported here line up with the keys it returns for this block.  Keys it
    returns that are *missing* here came from the fallback path.
    """
    identifier = VideoIdentifier.from_filename(current_video_filename)
    if identifier is None:
        raise ValueError(
            f"Could not parse a block number from '{current_video_filename}'. The "
            "current video filename must be the .mp4 name and must contain "
            "'_block-orderNN'."
        )

    stems: Dict[int, str] = {}
    for params in forearm_params:
        params_identifier = VideoIdentifier.from_filename(params.video_filename)
        if params_identifier is None:
            continue
        if (
            params_identifier.prefix == identifier.prefix
            and params_identifier.block_number == identifier.block_number
        ):
            stems[params.representative_frame_id] = params.build_output_stem(
                Path(params.video_filename).stem
            )
    return stems


def load_recording_geometry(
    *,
    hand_motion_path: Path,
    hand_metadata_path: Path,
    forearm_metadata_path: Path,
    forearm_pointcloud_dir: Path,
    current_video_filename: str,
    fps: float = 30.0,
) -> RecordingGeometry:
    """Load one recording's hand poses and forearm terrain.

    Every loader used here returns ``None`` on failure rather than raising, so
    each result is checked and re-raised with the offending path named.

    Args:
        hand_motion_path: ``<video_stem>_handmodel_motion.npz`` (or the
            ``_stabilised`` variant).
        hand_metadata_path: ``<video_stem>_handmodel_metadata.json``.
        forearm_metadata_path: ``<session_id>_arm_roi_metadata.json``.
        forearm_pointcloud_dir: ``<session_processed>/forearm_pointclouds/``.
        current_video_filename: The ``.mp4`` file name of this block.
        fps: Frame rate used to seed the motion manager; the NPZ overrides it
            when it carries an ``fps`` key.

    Returns:
        The loaded :class:`RecordingGeometry`.

    Raises:
        FileNotFoundError: If a required input file or directory is absent.
        ValueError: If a loader yields nothing usable, if the block name cannot
            be parsed, or if no forearm reference exists for frame 0.
    """
    for label, path in (
        ("hand motion NPZ", hand_motion_path),
        ("hand metadata JSON", hand_metadata_path),
        ("forearm metadata JSON", forearm_metadata_path),
        ("forearm pointcloud directory", forearm_pointcloud_dir),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    motion_manager = HandMotionManager(fps=float(fps))
    motion_manager.load(str(hand_motion_path))
    if len(motion_manager) == 0:
        raise ValueError(f"Hand motion NPZ contains zero frames: {hand_motion_path}")
    if motion_manager.faces is None:
        raise ValueError(
            f"Hand motion NPZ has no 'faces' array: {hand_motion_path}. Without "
            "topology the hand is a point set and no raycasting scene can be built."
        )

    hand_metadata: Optional[HandMetadataManager] = HandMetadataFileHandler.load(hand_metadata_path)
    if hand_metadata is None:
        raise ValueError(
            f"HandMetadataFileHandler returned None for {hand_metadata_path}. The "
            "excluded vertex ids are required: computing contact against the "
            "unfiltered hand mesh would silently change the contact patch."
        )

    forearm_params: Optional[List[ForearmParameters]] = ForearmFrameParametersFileHandler.load(
        forearm_metadata_path
    )
    if not forearm_params:
        raise ValueError(
            f"ForearmFrameParametersFileHandler returned no parameters for "
            f"{forearm_metadata_path}."
        )

    catalog = ForearmCatalog(forearm_params, forearm_pointcloud_dir)
    with _capture_root_logging() as collector:
        forearm_meshes = get_forearms_with_fallback(
            catalog, current_video_filename, use_mesh=True
        )
    fallback_notices = tuple(collector.messages)

    if not forearm_meshes:
        raise ValueError(
            f"No forearm mesh could be loaded for '{current_video_filename}' from "
            f"{forearm_pointcloud_dir}. Notices: {fallback_notices or '(none)'}"
        )
    if 0 not in forearm_meshes:
        raise ValueError(
            f"No forearm reference for frame 0 of '{current_video_filename}'; keys "
            f"present: {sorted(forearm_meshes)}. get_forearms_with_fallback "
            "guarantees key 0 whenever anything loads, so this indicates the "
            "catalog changed shape."
        )

    return RecordingGeometry(
        hand_motion_path=hand_motion_path,
        current_video_filename=current_video_filename,
        motion_manager=motion_manager,
        forearm_meshes_by_frame=forearm_meshes,
        forearm_stems_by_frame=_block_snapshot_stems(forearm_params, current_video_filename),
        forearm_fallback_notices=fallback_notices,
        excluded_vertex_ids=tuple(hand_metadata.excluded_vertex_ids),
        selected_point_labels=tuple(hand_metadata.selected_points),
        repeated_pose_frame_indices=_find_repeated_pose_frames(motion_manager),
    )


# =============================================================================
# Drive filter [2] over the recording
# =============================================================================


def compute_depth_field_series(
    geometry: RecordingGeometry,
    *,
    epsilon: float = EPSILON,
    progress_every: int = 250,
) -> ContactDepthFieldSeries:
    """Run the per-vertex depth field over every frame of the recording.

    Replicates the frame semantics of ``ObjectsInteractionController.run()``:
    the forearm reference is swapped whenever the frame index is a key of
    ``forearm_meshes_by_frame``, and the excluded hand vertices are removed
    before contact is computed.

    Every frame is computed here, up front.  No signed-distance work may leak
    into a viewer callback: precomputation is what keeps scrubbing responsive
    and keeps the colour scale honest.

    Args:
        geometry: The loaded recording.
        epsilon: Grazing-contact tolerance, forwarded to the field function.
        progress_every: Print a progress line every N frames.

    Returns:
        The full :class:`ContactDepthFieldSeries`.

    Raises:
        ValueError: Propagated from the field function, annotated with the frame
            index that produced it. A geometry failure is a hard stop, never a
            skipped frame.
    """
    num_frames = len(geometry)
    print(f"Computing the contact depth field over {num_frames} frames...")

    outcomes: List[FrameOutcome] = []
    current_forearm: Optional[o3d.geometry.TriangleMesh] = None

    for frame_index in range(num_frames):
        if frame_index in geometry.forearm_meshes_by_frame:
            current_forearm = geometry.forearm_meshes_by_frame[frame_index]
        if current_forearm is None:
            raise ValueError(
                f"Frame {frame_index}: no forearm reference is in effect. Frame 0 "
                "must seed one."
            )

        time_s = geometry.time_s(frame_index)

        if not geometry.pose_is_finite(frame_index):
            outcomes.append(
                FrameOutcome(
                    frame_index=frame_index,
                    time_s=time_s,
                    status=FrameStatus.POSE_ABSENT,
                    exclusion_reason="non-finite pose or vertices in the motion NPZ",
                )
            )
            continue

        validate_pose_transform(
            geometry.pose_matrix(frame_index), label=f"hand pose at frame {frame_index}"
        )

        try:
            field = signed_contact_depth_mm(
                geometry.hand_mesh(frame_index),
                current_forearm,
                epsilon=epsilon,
                frame_index=frame_index,
                time_s=time_s,
            )
        except ValueError as exc:
            raise ValueError(
                f"Contact depth field failed at frame {frame_index} "
                f"(t = {time_s:.3f} s) of '{geometry.current_video_filename}': {exc}"
            ) from exc

        outcomes.append(
            FrameOutcome(
                frame_index=frame_index,
                time_s=time_s,
                status=FrameStatus.CONTACT if field is not None else FrameStatus.NO_CONTACT,
                field=field,
            )
        )

        if (frame_index + 1) % progress_every == 0 or (frame_index + 1) == num_frames:
            print(f"  Computed frame {frame_index + 1}/{num_frames}")

    return ContactDepthFieldSeries.from_outcomes(
        outcomes, geometry.repeated_pose_frame_indices
    )


# =============================================================================
# Reporting
# =============================================================================


def format_recording_report(
    geometry: RecordingGeometry,
    series: ContactDepthFieldSeries,
) -> str:
    """Render the run summary: provenance, frame accounting, and the depth range."""
    lines: List[str] = ["", "=" * 78, "CONTACT DEPTH FIELD — RECORDING SUMMARY", "=" * 78]

    lines.append(f"Recording          : {geometry.current_video_filename}")
    lines.append(f"Hand motion NPZ    : {geometry.hand_motion_path.name}")
    lines.append(f"Excluded vertices  : {len(geometry.excluded_vertex_ids)}")
    lines.append(
        "Landmark labels    : "
        + (", ".join(geometry.selected_point_labels) or "(none)")
    )

    lines.append("")
    lines.append("Forearm reference (Kinect Space 1, forearm_pointclouds/*_mesh.obj)")
    for frame_id in sorted(geometry.forearm_meshes_by_frame):
        stem = geometry.forearm_stems_by_frame.get(frame_id)
        origin = f"{stem}_mesh.obj" if stem else "FALLBACK — not a snapshot of this block"
        lines.append(f"  from frame {frame_id:>5} : {origin}")
    if geometry.forearm_fallback_notices:
        lines.append("  loader notices:")
        lines.extend(f"    {notice}" for notice in geometry.forearm_fallback_notices)
    else:
        lines.append("  loader notices: (none — no fallback occurred)")

    n_frames = len(series.outcomes)
    n_contact = series.count(FrameStatus.CONTACT)
    n_no_contact = series.count(FrameStatus.NO_CONTACT)
    n_absent = series.count(FrameStatus.POSE_ABSENT)

    lines.append("")
    lines.append("Frame accounting")
    lines.append(f"  total frames            : {n_frames}")
    lines.append(f"  contact frames          : {n_contact}")
    lines.append(f"  no-contact frames       : {n_no_contact}")
    lines.append(f"  absent-pose frames      : {n_absent}")
    if n_absent:
        reasons = sorted(
            {o.exclusion_reason for o in series.outcomes if o.exclusion_reason is not None}
        )
        lines.extend(f"    excluded because: {reason}" for reason in reasons)
    lines.append(
        f"  carried-forward poses   : {len(series.repeated_pose_frame_indices)} "
        "(bit-identical repeat of the previous pose — the NPZ's tracking-dropout "
        "signature; computed as if tracked)"
    )

    lines.append("")
    lines.append("Depth field")
    if series.signed_depth_range_mm is None:
        lines.append("  no contact anywhere in this recording; colour limits are undefined")
    else:
        low, high = series.signed_depth_range_mm
        clim_low, clim_high = series.clim_penetration_mm
        contact_vertices = sum(
            len(o.field.signed_depth_mm) for o in series.outcomes if o.field is not None
        )
        lines.append(f"  signed depth range      : [{low:.6f}, {high:.6f}] mm (negative = penetrating)")
        lines.append(f"  global clim (penetration): [{clim_low:.6f}, {clim_high:.6f}] mm")
        lines.append(f"  contact vertices total  : {contact_vertices}")
        lines.append(
            f"  mean vertices per patch : {contact_vertices / n_contact:.1f}"
        )

    megabytes = series.field_array_bytes / (1024.0 * 1024.0)
    lines.append("")
    lines.append(
        f"In-memory field footprint : {megabytes:.2f} MB "
        f"({series.field_array_bytes} bytes of points + depths + normals; "
        "excludes Python object overhead)"
    )
    lines.append("=" * 78)
    return "\n".join(lines)


# =============================================================================
# Filter [3] seam — the viewer
# =============================================================================


def launch_viewer(
    geometry: RecordingGeometry,
    series: ContactDepthFieldSeries,
) -> None:
    """Open the interactive viewer on a precomputed field series.

    The viewer is Phase 3 of the plan and does not exist yet.  This is the only
    seam between the compute half and the render half: when
    ``ContactDepthFieldViewer`` lands, its import and construction replace the
    body of this function and nothing else in this module changes.  The import
    must stay inside the function so that the compute path never pulls in Qt or
    PyVista.
    """
    raise NotImplementedError(
        "ContactDepthFieldViewer is not implemented yet (plan phase 3). The field "
        f"series is complete: {len(series.outcomes)} frames, "
        f"{series.count(FrameStatus.CONTACT)} with contact, clim="
        f"{series.clim_penetration_mm}, context geometry available from "
        f"{type(geometry).__name__}."
    )


# =============================================================================
# Orchestration
# =============================================================================


def run_contact_depth_field_poc(
    *,
    hand_motion_path: Path,
    hand_metadata_path: Path,
    forearm_metadata_path: Path,
    forearm_pointcloud_dir: Path,
    current_video_filename: str,
    fps: float = 30.0,
    epsilon: float = EPSILON,
    show_viewer: bool = False,
) -> Tuple[RecordingGeometry, ContactDepthFieldSeries]:
    """Load one recording, compute its depth field series, and report on it.

    Writes nothing to disk — the series exists only in memory.

    Returns:
        The loaded geometry and the computed series, so a caller can inspect
        them without re-running the computation.
    """
    geometry = load_recording_geometry(
        hand_motion_path=hand_motion_path,
        hand_metadata_path=hand_metadata_path,
        forearm_metadata_path=forearm_metadata_path,
        forearm_pointcloud_dir=forearm_pointcloud_dir,
        current_video_filename=current_video_filename,
        fps=fps,
    )
    series = compute_depth_field_series(geometry, epsilon=epsilon)
    print(format_recording_report(geometry, series))

    if show_viewer:
        launch_viewer(geometry, series)

    return geometry, series


if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # Configuration — the one recording this proof of concept runs on.
    # ---------------------------------------------------------------------
    dataset_path = r"F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/02_data/semi-controlled/"
    session_id = "2022-06-15_ST14-01"
    block = "block-order-01"
    video_stem = "2022-06-15_ST14-01_semicontrolled_block-order01_kinect"

    # "_handmodel_motion.npz" (raw) or "_handmodel_motion_stabilised.npz".
    # Depth magnitudes are sensitive to the pose-smoothing configuration, so the
    # choice is recorded in the run report.
    motion_npz_suffix = "_handmodel_motion.npz"

    SHOW_VIEWER = False  # Phase 3; raises NotImplementedError until it lands.
    # ---------------------------------------------------------------------

    session_processed_dir = Path(dataset_path) / "2_processed" / "kinect" / session_id
    kinematics_dir = session_processed_dir / block / "kinematics_analysis"
    forearm_dir = session_processed_dir / "forearm_pointclouds"

    run_contact_depth_field_poc(
        hand_motion_path=kinematics_dir / (video_stem + motion_npz_suffix),
        hand_metadata_path=kinematics_dir / (video_stem + "_handmodel_metadata.json"),
        forearm_metadata_path=forearm_dir / (session_id + "_arm_roi_metadata.json"),
        forearm_pointcloud_dir=forearm_dir,
        current_video_filename=video_stem + ".mp4",
        show_viewer=SHOW_VIEWER,
    )
