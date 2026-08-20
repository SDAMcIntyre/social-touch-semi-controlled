"""Adapter: a contact-depth-field sidecar as the Neural+Kinect viewer consumes it.

This module sits between the parquet sidecar produced by the merging DAG
(``filter_contact_depth_field_by_neural_quality``) and
:class:`merging.gui.neural_kinect_scene_viewer.NeuralKinectViewer`.  It exists so
that the viewer can stay a **pure sink**: every number the viewer would otherwise
have to derive — the per-frame point/depth pair, and above all the colour range —
is computed here, once, before a single frame is drawn.

Why the colour range is computed here and not there
---------------------------------------------------
``clim_penetration_mm`` is **global over the whole recording**.  Per-frame
autoscaling would make the animation lie about relative depth: a shallow frame
and a deep frame would render with identical colours, and the same frame would
look different depending on whether it was reached by scrubbing forward or back.
The producing side owns that decision, which is why the range travels with the
data instead of being recomputed downstream (guide 02 §9, and the design
invariants of ``tactile_quantification/gui/contact_depth_field_viewer.py``).

Sign convention
---------------
Storage is **signed**, negative meaning penetrating.  Display is
**penetration**: ``penetration_depth_mm = -signed_depth_mm``, so positive means
deeper into the forearm and a larger number reads as "more".  The negation
happens exactly once, here.  Because negation reverses order, the penetration
colour range is ``(-max(signed), -min(signed))``, not ``(-min, -max)``.

Zero versus absent
------------------
A frame that carries no rows in the sidecar had **no contact**; :meth:`
ContactDepthFieldSeries.frame` returns ``None`` for it, and the viewer draws
nothing.  A *missing sidecar file* is a different fact entirely — the recording's
depth field was never produced — and is reported as an explicit
:class:`ContactDepthFieldResolution` with ``series=None`` and a message the
caller must print.  The two must never collapse into "draw flat colour and say
nothing".

Purity contract
---------------
This module knows about a path, a table and numpy arrays.  It must not learn
about Qt, VTK/PyVista, actors, sessions, DAGs or the workflow entry points.  Equally, nothing it
returns carries a file path or a session identity into the viewer: the DTO is
geometry, scalars and a colour range, and nothing else.

Laziness contract
-----------------
A batch is around a hundred blocks and a block's field is 10^5-10^6 vertices, so
"resolve every block up front" is not an option: it is minutes of parquet before
the first pixel and gigabytes resident afterwards.  A block therefore travels as
a :data:`ContactDepthFieldLoader` — a zero-argument callable built by
:func:`make_contact_depth_field_loader` — and the read happens when the consumer
opens that block, not when the batch is assembled.  Building loaders must stay
free; :class:`BoundedContactDepthFieldCache` puts a hard ceiling on what stays
resident once the reads do happen.

Vertex identity, when the sidecar carries it
--------------------------------------------
From the projection stage onward a sidecar also names, per row, the forearm
vertex the contact point was snapped to (``vertex_id``, schema v2).  When the
column is present it is carried through here alongside the reference-PLY
provenance triple that makes it interpretable; when it is not — a v1 sidecar, or
a v2 one written before projection — both are ``None``.  Absent means absent:
this module never derives a vertex identity by re-querying a KD-tree, because
the coordinates it holds came back through a different rounding path than the
ones the index was assigned from and would pick different vertices.

Coordinate space
----------------
Whatever the sidecar declares — in practice Kinect Space 1, millimetres.  This
module applies **no** transform.  The viewer's registered-frame mode rotates and
translates the *points* with the block's ICP matrix; the depth *values* are
invariant under a rigid transform and are never recomputed.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from preprocessing.motion_analysis.tactile_quantification.io.contact_depth_field_io import (
    REFERENCE_PLY_METADATA_KEYS,
    VERTEX_ID_COLUMN,
    read_contact_depth_field,
)

__all__ = [
    "BoundedContactDepthFieldCache",
    "ContactDepthFieldLoader",
    "ContactDepthFieldResolution",
    "ContactDepthFieldSeries",
    "load_contact_depth_field_series",
    "make_contact_depth_field_loader",
    "resolve_contact_depth_field",
]

#: A zero-argument handle on one block's depth field.  Calling it performs the
#: read; ``None`` is the explicit *absent* outcome, already reported by whoever
#: built the loader.  The consumer (the viewer) is handed one of these instead of
#: a path so that it never learns about parquet, directories or session ids — it
#: calls what it was given, exactly when it needs the data and not before.
ContactDepthFieldLoader = Callable[[], Optional["ContactDepthFieldSeries"]]


@dataclass(frozen=True)
class ContactDepthFieldSeries:
    """One recording's depth field, indexed by Kinect frame, plus its colour range.

    Attributes:
        points_by_frame: ``frame_index -> (N, 3) float32`` contact-vertex
            positions in millimetres, in the sidecar's declared coordinate space.
            A frame with no contact is **absent from the mapping**, never present
            with zero rows.
        penetration_depth_by_frame: ``frame_index -> (N,) float64`` penetration
            depths, positive meaning deeper, index-aligned with the matching
            entry of :attr:`points_by_frame`.
        clim_penetration_mm: ``(low, high)`` colour limits in penetration
            millimetres, computed once over the **whole** recording.
        signed_depth_range_mm: ``(min, max)`` of the raw signed field, kept so a
            caller can report the storage-side numbers without re-reading.
        coordinate_space: The space declared by the sidecar's metadata, carried
            through verbatim so a caller can refuse to draw a wrong-space field.
        vertex_id_by_frame: ``frame_index -> (N,) int32`` forearm vertex indices,
            index-aligned with the matching entry of :attr:`points_by_frame`, or
            ``None`` when the sidecar carries no ``vertex_id`` column at all.
            ``None`` is the honest answer for every stage before projection; it
            must never be replaced by a re-derived nearest vertex.  The dtype is
            the schema's ``int32`` and is asserted rather than coerced — an index
            that silently widened is an index that came from somewhere else.
        reference_ply_provenance: The three metadata keys that make a
            ``vertex_id`` interpretable (``reference_ply``,
            ``reference_ply_vertex_count``, ``dedup_epsilon``), carried verbatim
            so a caller can hand them to
            ``validate_vertex_ids_against_reference`` before joining the index to
            a mesh.  ``None`` exactly when :attr:`vertex_id_by_frame` is: the two
            are meaningless apart.

    Raises:
        ValueError: If the two mappings disagree, a frame's arrays are not
            index-aligned, or a vertex-identity half is present without the
            other.
    """

    points_by_frame: Dict[int, np.ndarray]
    penetration_depth_by_frame: Dict[int, np.ndarray]
    clim_penetration_mm: Tuple[float, float]
    signed_depth_range_mm: Tuple[float, float]
    coordinate_space: str
    # Appended, optional and defaulted: a v1 sidecar and a pre-projection v2 one
    # have no vertex identity, and every construction site that predates schema
    # v2 stays valid without being touched.
    vertex_id_by_frame: Optional[Dict[int, np.ndarray]] = None
    reference_ply_provenance: Optional[Dict[str, str]] = None

    def __post_init__(self) -> None:
        if not self.points_by_frame:
            raise ValueError(
                "points_by_frame is empty. A depth field with no contact frame at "
                "all is not a field; the sidecar should never have been written."
            )
        if set(self.points_by_frame) != set(self.penetration_depth_by_frame):
            missing = set(self.points_by_frame) ^ set(self.penetration_depth_by_frame)
            raise ValueError(
                "points_by_frame and penetration_depth_by_frame cover different "
                f"frames; symmetric difference includes {sorted(missing)[:5]}."
            )
        for frame_index, points in self.points_by_frame.items():
            depths = self.penetration_depth_by_frame[frame_index]
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(
                    f"Frame {frame_index}: points must be (N, 3), got {points.shape}."
                )
            if depths.ndim != 1 or len(depths) != len(points):
                raise ValueError(
                    f"Frame {frame_index}: {len(points)} points vs {depths.shape} "
                    "depths; the field must be index-aligned with the points."
                )
            if len(points) == 0:
                raise ValueError(
                    f"Frame {frame_index}: an empty field is not a field. A frame "
                    "with no contact must be absent from the mapping entirely, so "
                    "that 'no contact' cannot be read as 'contact of zero depth'."
                )

        low, high = self.clim_penetration_mm
        if not (np.isfinite(low) and np.isfinite(high)):
            raise ValueError(
                f"clim_penetration_mm is not finite: {self.clim_penetration_mm}."
            )
        if high < low:
            raise ValueError(
                f"clim_penetration_mm is inverted: {self.clim_penetration_mm}."
            )

        self._validate_vertex_identity()

    def _validate_vertex_identity(self) -> None:
        """Check the optional vertex-identity half, if it is there at all.

        Raises:
            ValueError: If only one of the two vertex-identity fields is
                present, if the index does not cover exactly the frames the
                points do, if a frame's indices are not index-aligned with its
                points, if the dtype is not the schema's ``int32``, or if the
                provenance triple is incomplete.
        """
        has_ids = self.vertex_id_by_frame is not None
        has_provenance = self.reference_ply_provenance is not None
        if has_ids != has_provenance:
            raise ValueError(
                "vertex_id_by_frame and reference_ply_provenance must be "
                f"present together (ids={has_ids}, provenance={has_provenance}). "
                "A vertex index whose mesh identity is unknown cannot be "
                "validated against anything, and provenance for an index that "
                "is not there describes nothing."
            )
        if not has_ids:
            return

        missing_keys = [
            key
            for key in REFERENCE_PLY_METADATA_KEYS
            if not str(self.reference_ply_provenance.get(key, "")).strip()
        ]
        if missing_keys:
            raise ValueError(
                f"reference_ply_provenance omits or blanks {missing_keys}; all "
                f"of {list(REFERENCE_PLY_METADATA_KEYS)} are required together. "
                "The file names the mesh, the count proves it is the same mesh, "
                "and the epsilon proves it was deduplicated the same way."
            )

        if set(self.vertex_id_by_frame) != set(self.points_by_frame):
            difference = set(self.vertex_id_by_frame) ^ set(self.points_by_frame)
            raise ValueError(
                "vertex_id_by_frame and points_by_frame cover different frames; "
                f"symmetric difference includes {sorted(difference)[:5]}."
            )

        for frame_index, ids in self.vertex_id_by_frame.items():
            points = self.points_by_frame[frame_index]
            if ids.ndim != 1 or len(ids) != len(points):
                raise ValueError(
                    f"Frame {frame_index}: {len(points)} points vs {ids.shape} "
                    "vertex ids; the index must be aligned with the points it "
                    "addresses."
                )
            if ids.dtype != np.int32:
                raise ValueError(
                    f"Frame {frame_index}: vertex_id has dtype {ids.dtype}, "
                    "expected int32. A widened index is an index that came from "
                    "somewhere other than the sidecar schema."
                )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def frame(self, frame_index: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return ``(points, penetration_depth_mm)`` for *frame_index*.

        Args:
            frame_index: Kinect frame index.

        Returns:
            The ``(N, 3)`` / ``(N,)`` pair for that frame, or ``None`` when the
            frame carries no contact.  ``None`` means *no contact*, which is a
            fact about the recording — not a lookup failure.
        """
        points = self.points_by_frame.get(int(frame_index))
        if points is None:
            return None
        return points, self.penetration_depth_by_frame[int(frame_index)]

    @property
    def has_vertex_ids(self) -> bool:
        """Whether this field names the forearm vertex each row addresses.

        ``False`` for every stage before projection.  A caller that wants to
        colour a mesh by this field must branch on it and show nothing when it
        is ``False`` — the alternative, snapping the points to their nearest
        vertex here, picks different vertices than the projection stage did.
        """
        return self.vertex_id_by_frame is not None

    @property
    def frame_count(self) -> int:
        """Number of frames that carry contact."""
        return len(self.points_by_frame)

    @property
    def vertex_count(self) -> int:
        """Total number of (frame, vertex) rows across the recording."""
        return sum(len(points) for points in self.points_by_frame.values())

    def summary(self) -> str:
        """One-line human-readable description, for a startup log."""
        low, high = self.clim_penetration_mm
        return (
            f"{self.frame_count} contact frames, {self.vertex_count} vertices, "
            f"penetration {low:.3f} to {high:.3f} mm (fixed global colour scale), "
            f"space={self.coordinate_space}"
        )


@dataclass(frozen=True)
class ContactDepthFieldResolution:
    """The outcome of looking for a recording's depth field: present, or absent.

    Absence is a *state*, not a fallback.  It is carried explicitly, together
    with a ready-made :attr:`message`, so a caller cannot quietly drop into flat
    colouring without saying so — the message exists precisely so that reporting
    it is easier than not reporting it.

    Attributes:
        path: Where the sidecar was looked for.
        series: The loaded series, or ``None`` when the file does not exist.
        message: Non-empty text the caller prints verbatim at startup.  States
            either what was loaded or that depth colouring is disabled and why.
    """

    path: Path
    series: Optional[ContactDepthFieldSeries]
    message: str

    def __post_init__(self) -> None:
        if not self.message:
            raise ValueError(
                "message is empty. Both outcomes must be announceable; a silent "
                "absent state is exactly what this type exists to prevent."
            )

    @property
    def is_present(self) -> bool:
        """Whether a depth field was found and loaded."""
        return self.series is not None


def resolve_contact_depth_field(path: Path) -> ContactDepthFieldResolution:
    """Load the depth field at *path*, or report its absence explicitly.

    Only **non-existence** is an absent state.  A file that exists but cannot be
    decoded — unknown ``schema_version``, missing metadata, wrong columns — is a
    broken artifact and raises, because rendering the scene without it would
    hide the breakage behind a plausible-looking picture.

    Args:
        path: Expected location of ``*_contact_depth_field.parquet``.

    Returns:
        A :class:`ContactDepthFieldResolution` whose :attr:`~
        ContactDepthFieldResolution.message` the caller must print.

    Raises:
        ValueError: If the file exists but is not a readable depth field.
    """
    path = Path(path)
    if not path.exists():
        return ContactDepthFieldResolution(
            path=path,
            series=None,
            message=(
                f"No contact depth field at '{path}'. Depth colouring is DISABLED "
                "for this block; contact points render in flat colour. This is an "
                "absent artifact, not a display preference — run the merging DAG "
                "task 'filter_contact_depth_field_by_neural_quality' to produce it."
            ),
        )

    series = load_contact_depth_field_series(path)
    return ContactDepthFieldResolution(
        path=path,
        series=series,
        message=f"Contact depth field loaded from '{path}': {series.summary()}.",
    )


def load_contact_depth_field_series(path: Path) -> ContactDepthFieldSeries:
    """Read a sidecar and index it by frame, computing the global colour range.

    The whole table is read once and held as two contiguous arrays; the per-frame
    entries are **views** into them, so indexing costs no extra memory beyond the
    dictionaries themselves.

    Args:
        path: The ``.parquet`` sidecar.

    Returns:
        The indexed series, with ``clim_penetration_mm`` computed over every row
        in the file.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file is not a readable depth field, or carries no rows.
    """
    path = Path(path)
    table, metadata = read_contact_depth_field(path)

    if len(table) == 0:
        raise ValueError(
            f"{path} contains zero rows. An empty depth field is indistinguishable "
            "from a complete one on the next run; the producer must raise rather "
            "than write it."
        )

    coordinate_space = metadata.get("coordinate_space")
    if not coordinate_space:
        raise ValueError(
            f"{path} declares no coordinate_space. Drawing a field whose space is "
            "unknown would put points in the scene on a guess."
        )

    frame_index = table["frame_index"].to_numpy(dtype=np.int64, copy=True)
    points = np.ascontiguousarray(
        table[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True)
    )
    signed_depth = table["signed_depth_mm"].to_numpy(dtype=np.float64, copy=True)

    # Optional, and only from the projection stage onward.  The dtype is
    # asserted rather than requested: passing ``dtype=np.int32`` here would
    # silently narrow an int64 column that had no business being int64, and the
    # whole value of the index is that it is the one the projection stage wrote.
    vertex_id: Optional[np.ndarray] = None
    provenance: Optional[Dict[str, str]] = None
    if VERTEX_ID_COLUMN in table.columns:
        vertex_id = table[VERTEX_ID_COLUMN].to_numpy(copy=True)
        if vertex_id.dtype != np.int32:
            raise ValueError(
                f"{path} column '{VERTEX_ID_COLUMN}' has dtype "
                f"{vertex_id.dtype}, expected int32 per the sidecar schema."
            )
        missing_keys = [
            key for key in REFERENCE_PLY_METADATA_KEYS if key not in metadata
        ]
        if missing_keys:
            raise ValueError(
                f"{path} carries a '{VERTEX_ID_COLUMN}' column but its metadata "
                f"omits {missing_keys}. A vertex index is meaningless without "
                "the identity of the mesh it indexes, and joining it to a "
                "forearm on trust is exactly the silent-renumbering hazard the "
                "provenance exists to catch."
            )
        provenance = {key: str(metadata[key]) for key in REFERENCE_PLY_METADATA_KEYS}

    # The schema promises a column set, not a row order.  Sorting here is
    # canonicalisation so that each frame's rows are contiguous and can be handed
    # out as views; it is not error recovery, and it changes no value.
    if not np.all(np.diff(frame_index) >= 0):
        order = np.argsort(frame_index, kind="stable")
        frame_index = frame_index[order]
        points = np.ascontiguousarray(points[order])
        signed_depth = signed_depth[order]
        if vertex_id is not None:
            vertex_id = vertex_id[order]

    # Negation is applied to the whole column at once, before slicing, so every
    # frame's depths come from the same single arithmetic operation.
    penetration = -signed_depth

    signed_min = float(np.min(signed_depth))
    signed_max = float(np.max(signed_depth))
    if not (np.isfinite(signed_min) and np.isfinite(signed_max)):
        raise ValueError(
            f"{path} contains a non-finite signed_depth_mm "
            f"(min={signed_min}, max={signed_max}); a colour scale over it would "
            "be meaningless."
        )
    # Negation reverses order, so the penetration bounds are the swapped,
    # negated signed bounds.
    clim = (-signed_max, -signed_min)

    unique_frames, starts, counts = np.unique(
        frame_index, return_index=True, return_counts=True
    )

    points_by_frame: Dict[int, np.ndarray] = {}
    penetration_by_frame: Dict[int, np.ndarray] = {}
    vertex_id_by_frame: Optional[Dict[int, np.ndarray]] = (
        None if vertex_id is None else {}
    )
    for value, start, count in zip(unique_frames, starts, counts):
        stop = start + count
        key = int(value)
        points_by_frame[key] = points[start:stop]
        penetration_by_frame[key] = penetration[start:stop]
        if vertex_id is not None:
            vertex_id_by_frame[key] = vertex_id[start:stop]

    return ContactDepthFieldSeries(
        points_by_frame=points_by_frame,
        penetration_depth_by_frame=penetration_by_frame,
        clim_penetration_mm=clim,
        signed_depth_range_mm=(signed_min, signed_max),
        coordinate_space=coordinate_space,
        vertex_id_by_frame=vertex_id_by_frame,
        reference_ply_provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Lazy access: a callable per block, plus a bounded cache behind it
# ---------------------------------------------------------------------------


class BoundedContactDepthFieldCache:
    """At most *maxsize* resolved depth fields, keyed by sidecar path (LRU).

    Why a *bounded* cache and not a plain memo
    ------------------------------------------
    A batch is on the order of 100 blocks and a block's field is 10^5-10^6
    vertices.  A per-loader memo — one slot each, never evicted — would hold
    every block a session visits, so a user who scrolls through the whole batch
    ends up with the same all-blocks-resident footprint that eager loading had,
    just reached more slowly.  A hard ceiling makes the worst case a property of
    this object rather than of the user's browsing.

    Two entries is the smallest size that keeps the behaviour the eager version
    had *within* a block: the plain and transformed specs for the same block
    share one loader, and stepping to the next block and back is one read, not
    two.  Anything larger buys nothing that matters and costs memory.

    A cache hit is silent.  A miss resolves and reports, so a message is
    printed exactly when a read actually happens — never as a claim about a
    read that was served from memory.

    Args:
        maxsize: Maximum number of resolved outcomes retained.

    Raises:
        ValueError: If *maxsize* is below 1.
    """

    def __init__(self, maxsize: int = 2) -> None:
        if maxsize < 1:
            raise ValueError(
                f"maxsize must be at least 1, got {maxsize}. A cache that can "
                "hold nothing is not a cache; drop the cache instead."
            )
        self._maxsize = int(maxsize)
        self._entries: "OrderedDict[Path, Optional[ContactDepthFieldSeries]]" = OrderedDict()

    @property
    def maxsize(self) -> int:
        """Maximum number of retained entries."""
        return self._maxsize

    def __len__(self) -> int:
        return len(self._entries)

    def get_or_resolve(
        self,
        path: Path,
        report: Callable[[str], None],
    ) -> Optional[ContactDepthFieldSeries]:
        """Return the series for *path*, resolving and reporting it on a miss.

        Args:
            path: The sidecar location.
            report: Called with :attr:`ContactDepthFieldResolution.message` when
                a resolution actually happens.  Not called on a cache hit.

        Returns:
            The loaded series, or ``None`` when the sidecar does not exist —
            the same explicit absent state :func:`resolve_contact_depth_field`
            produces, already announced through *report*.

        Raises:
            ValueError: If the file exists but is not a readable depth field.
        """
        key = Path(path)
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]

        resolution = resolve_contact_depth_field(key)
        report(resolution.message)

        self._entries[key] = resolution.series
        while len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)
        return resolution.series


def make_contact_depth_field_loader(
    path: Path,
    report: Callable[[str], None],
    cache: BoundedContactDepthFieldCache,
) -> ContactDepthFieldLoader:
    """Build the zero-argument loader a block spec carries.

    Nothing is read here.  Building a loader for every block of a batch must
    stay free — that is the whole point of handing the consumer a callable
    rather than a loaded series.

    Args:
        path: The block's sidecar location.
        report: Where the resolution message goes when a read happens.
        cache: Shared bounded cache; see
            :class:`BoundedContactDepthFieldCache` for why it is bounded.

    Returns:
        A callable returning the series, or ``None`` for an absent sidecar.
    """
    resolved_path = Path(path)

    def _load() -> Optional[ContactDepthFieldSeries]:
        return cache.get_or_resolve(resolved_path, report)

    return _load
