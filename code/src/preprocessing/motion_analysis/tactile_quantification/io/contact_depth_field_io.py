"""Read/write the long-form per-vertex contact depth field sidecar.

The sidecar is a parquet file holding **exactly one row per (frame, contact
vertex)**.  It is the companion of the per-frame
``*_contact_and_kinematic_data.csv``; the two join on ``frame_index``.

Schema
------
====================  =========  ===============================================
column                dtype      meaning
====================  =========  ===============================================
``frame_index``       int32      Kinect frame index; joins to the CSV
``time_s``            float64    frame timestamp, seconds
``x``, ``y``, ``z``   float32    contact vertex position, mm, declared space
``signed_depth_mm``   float64    signed distance to the hand surface, mm
``vertex_id``         int32      *optional*; row index into the reference PLY
====================  =========  ===============================================

Exactly **two** column layouts are legal: the six required columns in the order
above, and those six followed by ``vertex_id``.  Nothing else — not an extra
column, not a permutation.  The exactness is the feature: a table that grew a
column nobody declared is a caller bug, and accepting it here is how a shadow
column reaches disk and is later mistaken for schema.

``signed_depth_mm`` is stored at **float64, not float32**, on purpose.  The
pipeline's strongest correctness check is that ``max(|signed_depth_mm|)``
recovered per frame equals the CSV's ``contact_depth`` *bit-identically*.
float32 would degrade that to "approximately", which is not a checkable
invariant.  ``x/y/z`` stay float32 (0.1 µm at millimetre scale, against a CSV
counterpart already quantised to 0.1 mm by ``serialize_contact_points``).

Sign convention
---------------
**Negative means penetrating.**  ``penetration_depth_mm = -signed_depth_mm`` is
a display-only derivation; storage stays signed.  See
``model/contact_depth_field.py`` for the single definition of the field.

Units
-----
Millimetres throughout, Kinect-native.  There is no unit conversion anywhere in
this pipeline.

Coordinate space
----------------
``x/y/z`` start in **Kinect Space 1** — the raw Kinect frame, *before* ICP
registration, PCA calibration and receptive-field centring — and postprocessing
moves them through three further frames.  A file therefore declares the space
it is *actually* in, in its metadata, rather than leaving it to convention:

=====================  ======================================================
``coordinate_space``   frame
=====================  ======================================================
``kinect_space_1``     raw Kinect, as produced by preprocessing and merging
``icp_registered``     after ``apply_icp_registration`` (dedup and projection
                       do not move points, so they inherit this)
``pca_calibrated``     after ``calibrate_pca_xyz``
``rf_centered``        after ``center_on_receptive_field`` — terminal
=====================  ======================================================

The vocabulary is **closed and validated at write time** (:data:`COORDINATE_SPACES`).
A free-text space would let ``"icp-registered"`` or ``"rf_centred"`` land on
disk and read back as an unrecognised-but-accepted string, at which point a
consumer either guesses or silently skips the file.  A rejected write is a
strictly better outcome than a typo'd provenance record, and this module knows
the full vocabulary because the vocabulary is a property of the *format*, not
of any stage.  Validation is deliberately **write-side only**: the space name
does not affect decoding, and a reader that refused an old name would break
backward compatibility over a string it does not use.

``vertex_id`` and reference-PLY identity
-----------------------------------------
``vertex_id`` is the row index, *in file order*, of the reference-forearm PLY
vertex a contact point was snapped to at the projection stage.  It is absent
before projection and present after.  It is more durable than ``x/y/z``:
the three post-projection forearm PLYs are written in file order with no
reorder, filter or count change, so vertex *i* is the same physical vertex in
all of them, whereas the coordinates are re-rounded to 0.1 mm at three separate
points.

An index is meaningless without the thing it indexes, so a table carrying
``vertex_id`` must also carry all three of ``reference_ply``,
``reference_ply_vertex_count`` and ``dedup_epsilon``, and must declare a schema
version that knows about the column.  Both are enforced, not documented: an
index whose PLY identity was lost is not partially useful, it is silently
wrong.  ``dedup_epsilon`` is in that triple because re-deduplicating the
forearm at a different epsilon renumbers every vertex.

:func:`validate_vertex_ids_against_reference` is the read-side counterpart —
the check that turns "these ids point somewhere else now" into a raise.  It
takes a vertex *count*, never a PLY: this module does not open meshes.

This field is NOT forward-fillable and NOT interpolable
-------------------------------------------------------
A scalar depth track can be resampled; a per-vertex field cannot.  The contact
patch changes membership frame to frame, so row *i* of frame *n* and row *i* of
frame *n+1* are not the same vertex.  Cubic interpolation across a touch
boundary has been measured overshooting 29 mm off-surface
(``bug-rf-explorer-nearest-vertex-distance.md``).  Join on ``frame_index``;
never ffill, never interpolate.

Zero versus absent
------------------
A frame with no contact contributes **zero rows**, and carries
``contact_detected == False`` in the CSV.  That is *zero contact*.  A frame
whose hand pose could not be measured is *absent* and raises upstream — it is
never written as zero.  Collapsing the two would silently down-weight real
spikes once this field weights instantaneous firing rate.

Two writers, one schema
-----------------------
:func:`write_contact_depth_field` serialises a sequence of
:class:`ContactDepthFrame`.  It is the producing path and therefore owns the
standard metadata block, including ``produced_by``.

:func:`write_contact_depth_field_table` serialises an already-schema-conforming
long-form DataFrame together with caller-supplied metadata.  It exists for
downstream stages that read a sidecar, drop rows, and write the reduced table
back: reconstructing :class:`ContactDepthFrame` objects to re-serialise rows
that were just deserialised would be pure ceremony, and would round-trip the
values through a second conversion for no benefit.

Neither writer knows *why* rows are present or absent — row selection is the
caller's business.  The writers' only obligation is that what they are handed
reaches disk unaltered, in schema order, with its metadata attached.  The table
writer consequently validates and refuses rather than coercing: a column in the
wrong position or the wrong dtype is a caller bug, and quietly repairing it is
how a downcast ``signed_depth_mm`` would reach disk unnoticed.

Purity contract
---------------
This module knows about sequences of :class:`ContactDepthFrame`,
schema-conforming DataFrames, and a path.  It must never learn about sessions,
configs, DAGs, the workflow entry points, or the CSV.  That is what keeps the storage format
swappable: only the function bodies here change if parquet is ever replaced.

For the same reason :class:`ContactDepthFrame` is imported for typing only:
serialisation reads four plain attributes and needs no geometry engine, so this
module stays importable (and testable) without Open3D.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:  # pragma: no cover - typing only; avoids an Open3D import
    from ..model.contact_depth_field import ContactDepthFrame

__all__ = [
    "ALL_COLUMN_DTYPES",
    "COLUMN_DTYPES",
    "COORDINATE_SPACE",
    "COORDINATE_SPACES",
    "COORDINATE_SPACE_ICP_REGISTERED",
    "COORDINATE_SPACE_KINECT_1",
    "COORDINATE_SPACE_PCA_CALIBRATED",
    "COORDINATE_SPACE_RF_CENTERED",
    "OPTIONAL_COLUMN_DTYPES",
    "PRODUCED_BY",
    "REFERENCE_PLY_METADATA_KEYS",
    "SCHEMA_VERSION",
    "SIGN_CONVENTION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "UNITS",
    "VERTEX_ID_COLUMN",
    "VERTEX_ID_SCHEMA_VERSIONS",
    "read_contact_depth_field",
    "validate_vertex_ids_against_reference",
    "write_contact_depth_field",
    "write_contact_depth_field_table",
]


#: Version of the on-disk schema.  Present from day one: the closest prior art
#: (``population-rf-vertex-data-export.md``) shipped without one and had its
#: file renamed and its schema widened five days later.
#:
#: ``"2"`` adds the optional ``vertex_id`` column and the reference-PLY
#: provenance triple.  It is a pure widening: a v1-shaped table is a valid v2
#: payload, which is why the frame writer needed no change to start stamping it.
SCHEMA_VERSION: str = "2"

#: Schema versions this reader understands.  Anything else raises rather than
#: being decoded on a guess.  ``"1"`` stays in the set indefinitely: the
#: Space-1 artifacts already on disk are v1 files, and this reader is the only
#: way anything in the pipeline opens them.
SUPPORTED_SCHEMA_VERSIONS: frozenset = frozenset({"1", "2"})

#: Versions whose column set includes :data:`VERTEX_ID_COLUMN`.  Writing the
#: column under any other version would stamp a file whose declared schema
#: disagrees with its own columns — the exact drift the version string exists
#: to prevent.
VERTEX_ID_SCHEMA_VERSIONS: frozenset = frozenset({"2"})

#: Raw Kinect frame, millimetres, before ICP / PCA / RF-centring.
COORDINATE_SPACE_KINECT_1: str = "kinect_space_1"

#: After ``apply_icp_registration``.  Deduplication and projection re-address
#: and drop rows but move no point, so they leave this name in place.
COORDINATE_SPACE_ICP_REGISTERED: str = "icp_registered"

#: After ``calibrate_pca_xyz``.
COORDINATE_SPACE_PCA_CALIBRATED: str = "pca_calibrated"

#: After ``center_on_receptive_field`` — the terminal space.
COORDINATE_SPACE_RF_CENTERED: str = "rf_centered"

#: The closed vocabulary.  Checked on write; see the module docstring for why
#: it is closed and why it is not checked on read.
COORDINATE_SPACES: frozenset = frozenset(
    {
        COORDINATE_SPACE_KINECT_1,
        COORDINATE_SPACE_ICP_REGISTERED,
        COORDINATE_SPACE_PCA_CALIBRATED,
        COORDINATE_SPACE_RF_CENTERED,
    }
)

#: The space this module's producing writer emits.  Retained under its original
#: name because callers import it; :data:`COORDINATE_SPACE_KINECT_1` is the
#: same string spelled so the four spaces read as a set.
COORDINATE_SPACE: str = COORDINATE_SPACE_KINECT_1

#: The provenance triple that makes a ``vertex_id`` interpretable.  All three
#: are required whenever the column is present, and meaningless without each
#: other: the file names the PLY, the count proves it is the same PLY, and the
#: epsilon proves it was deduplicated the same way.
REFERENCE_PLY_METADATA_KEYS: Tuple[str, ...] = (
    "reference_ply",
    "reference_ply_vertex_count",
    "dedup_epsilon",
)

UNITS: str = "mm"

SIGN_CONVENTION: str = "negative_is_penetrating"

#: The only producer of this artifact today.  A constant rather than a
#: parameter: the writer is not given, and must not acquire, pipeline
#: knowledge, and a provenance string that every caller must supply correctly
#: is a provenance string that will eventually be supplied incorrectly.
PRODUCED_BY: str = "compute_somatosensory_characteristics"

#: The one optional column.  A module-level constant rather than a literal so
#: that callers testing for its presence and this module agree on the spelling.
VERTEX_ID_COLUMN: str = "vertex_id"

_SCHEMA_FIELDS = [
    pa.field("frame_index", pa.int32()),
    pa.field("time_s", pa.float64()),
    pa.field("x", pa.float32()),
    pa.field("y", pa.float32()),
    pa.field("z", pa.float32()),
    pa.field("signed_depth_mm", pa.float64()),
]

#: ``int32``, not ``int64``: a forearm PLY has ~10^4-10^5 vertices, and the
#: column sits on a table that already reaches a gigabyte per session.  The
#: dtype is asserted rather than coerced for the same reason ``signed_depth_mm``
#: is — an index that silently widened is an index that came from somewhere
#: other than the projection stage.
_VERTEX_ID_FIELD = pa.field(VERTEX_ID_COLUMN, pa.int32())

#: Column name -> numpy dtype for the **required** columns, used to assert the
#: read-back contract.  Deliberately excludes ``vertex_id``: callers iterate
#: this mapping to mean "every column that is always there".
COLUMN_DTYPES: Dict[str, np.dtype] = {
    "frame_index": np.dtype(np.int32),
    "time_s": np.dtype(np.float64),
    "x": np.dtype(np.float32),
    "y": np.dtype(np.float32),
    "z": np.dtype(np.float32),
    "signed_depth_mm": np.dtype(np.float64),
}

#: Column name -> numpy dtype for the columns that may or may not be present.
OPTIONAL_COLUMN_DTYPES: Dict[str, np.dtype] = {
    VERTEX_ID_COLUMN: np.dtype(np.int32),
}

#: Every column this schema knows, in on-disk order.  Optional columns come
#: after the required ones, so the wide layout is the narrow layout plus a
#: suffix and neither reader nor writer needs to reorder anything.
ALL_COLUMN_DTYPES: Dict[str, np.dtype] = {**COLUMN_DTYPES, **OPTIONAL_COLUMN_DTYPES}

#: The only two legal column layouts.  Not "at least these columns" — exactly
#: one of these two lists, in this order.
_VALID_COLUMN_LAYOUTS: Tuple[Tuple[str, ...], ...] = (
    tuple(COLUMN_DTYPES),
    tuple(ALL_COLUMN_DTYPES),
)


def _schema_fields_for(*, has_vertex_id: bool):
    """Return the arrow fields for the layout ``has_vertex_id`` selects."""
    return list(_SCHEMA_FIELDS) + ([_VERTEX_ID_FIELD] if has_vertex_id else [])


def _write_arrow_table_atomically(table: pa.Table, output_path: Path) -> None:
    """Write ``table`` to ``output_path`` via a sibling temp file and a rename.

    A failure mid-write then leaves either the previous file or nothing — never
    a truncated file that a later run's existence check would accept as
    complete.  Shared by both writers so the two cannot drift apart on the one
    property that makes the artifact safe to resume against.
    """
    output_path = Path(output_path)
    temp_path = output_path.with_name(output_path.name + ".partial")
    try:
        pq.write_table(table, temp_path)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _validate_table_against_schema(table: pd.DataFrame) -> bool:
    """Raise unless ``table`` matches one of the two legal layouts exactly.

    Names, order and dtypes are all checked against
    :data:`_VALID_COLUMN_LAYOUTS` — the required six, or the required six plus
    ``vertex_id``.  Nothing is coerced: the caller handing over a mis-ordered or
    mis-typed frame has a bug, and repairing it here would hide a downcast of
    ``signed_depth_mm`` behind a successful write.

    Returns:
        Whether the table carries ``vertex_id``, so the caller can select the
        matching arrow schema and hold the metadata to the matching contract.
    """
    if table is None:
        raise ValueError(
            "table is None. A depth-field table with no rows at all is a fact "
            "worth surfacing, not a zero-row file to write silently."
        )
    if not isinstance(table, pd.DataFrame):
        raise ValueError(
            f"table must be a pandas DataFrame, got {type(table).__name__}. "
            "This writer serialises an already-schema-conforming long-form "
            "table; it does not construct one."
        )

    actual_columns = list(table.columns)
    has_vertex_id = VERTEX_ID_COLUMN in actual_columns

    # Which of the two layouts the caller *meant* is decided by whether
    # ``vertex_id`` appears at all, so the diagnostic below names the layout
    # being aimed at rather than always the narrow one.
    expected_columns = list(
        ALL_COLUMN_DTYPES if has_vertex_id else COLUMN_DTYPES
    )

    if tuple(actual_columns) not in _VALID_COLUMN_LAYOUTS:
        missing = [c for c in expected_columns if c not in actual_columns]
        unexpected = [c for c in actual_columns if c not in expected_columns]
        if missing or unexpected:
            raise ValueError(
                f"table columns do not match the contact depth field schema: "
                f"missing={missing}, unexpected={unexpected}. Expected exactly "
                f"{expected_columns}, got {actual_columns}. The schema admits "
                f"only {[list(layout) for layout in _VALID_COLUMN_LAYOUTS]}; a "
                "column outside that set is a caller bug, not an extension "
                "point."
            )
        raise ValueError(
            f"table columns are in the wrong order: expected {expected_columns}, "
            f"got {actual_columns}. Column order is part of the on-disk schema; "
            "refusing to reorder silently."
        )

    for column in actual_columns:
        expected_dtype = ALL_COLUMN_DTYPES[column]
        actual_dtype = table[column].dtype
        if actual_dtype != expected_dtype:
            raise ValueError(
                f"table column {column!r} has dtype {actual_dtype}, expected "
                f"{expected_dtype}. Precision is a schema decision here; "
                "refusing to cast it silently."
            )

    if len(table) == 0:
        raise ValueError(
            "table is empty: no row survived selection. Refusing to write a "
            "zero-row sidecar, which a later run would treat as a valid, "
            "complete artifact. Zero rows and an absent artifact must not "
            "collapse into the same thing on disk."
        )

    return has_vertex_id


def _parse_metadata_int(metadata: Dict[str, str], key: str) -> int:
    """Return ``metadata[key]`` as a positive int, raising if it is not one."""
    raw = metadata[key]
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"metadata {key!r} is {raw!r}, which is not an integer. It is "
            "recorded as a string because parquet metadata is bytes, but it "
            "must still round-trip to the number it claims to be."
        ) from exc
    if value <= 0:
        raise ValueError(
            f"metadata {key!r} is {value}, which is not a positive count. A "
            "reference mesh with no vertices cannot be indexed into."
        )
    return value


def _parse_metadata_float(metadata: Dict[str, str], key: str) -> float:
    """Return ``metadata[key]`` as a finite positive float, raising otherwise."""
    raw = metadata[key]
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"metadata {key!r} is {raw!r}, which is not a number. It is "
            "recorded as a string because parquet metadata is bytes, but it "
            "must still round-trip to the number it claims to be."
        ) from exc
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"metadata {key!r} is {raw!r}, which is not a finite positive "
            "value. A non-positive or non-finite deduplication epsilon does "
            "not describe a run that happened."
        )
    return value


def _validate_supplied_metadata(
    metadata: Dict[str, str], *, has_vertex_id: bool = False
) -> None:
    """Raise unless ``metadata`` is a str->str mapping declaring a known version.

    The schema version is checked at *write* time, mirroring the reader: a file
    stamped with a version this module does not understand is unreadable the
    moment it lands, and finding that out on the next read is finding out too
    late.

    The same argument extends to the rest of the provenance block, so it is all
    checked here: ``coordinate_space`` against the closed vocabulary,
    ``reference_ply_vertex_count`` and ``dedup_epsilon`` against the numbers
    they claim to be, and — when the table carries ``vertex_id`` — the presence
    of the whole reference-PLY triple.  None of it is defaulted or repaired;
    provenance invented by the writer is worse than provenance absent.

    Args:
        metadata: The complete file metadata the caller means to write.
        has_vertex_id: Whether the table being written carries the optional
            ``vertex_id`` column, which raises the metadata requirements.
    """
    if not isinstance(metadata, dict):
        raise ValueError(
            f"metadata must be a dict of str->str, got {type(metadata).__name__}."
        )

    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError(
                f"metadata entry {key!r}: {value!r} is not str->str (got "
                f"{type(key).__name__} -> {type(value).__name__}). Parquet file "
                "metadata is bytes; every value must be stringified by the "
                "caller, deliberately, rather than by this writer's guess at a "
                "format."
            )

    if "schema_version" not in metadata:
        raise ValueError(
            "metadata carries no 'schema_version'. Every contact depth field "
            "file must declare the schema it was written against; a file "
            "without one cannot be interpreted safely by the reader."
        )

    version = metadata["schema_version"]
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"metadata declares schema_version={version!r}, which this module "
            f"does not support (known: {sorted(SUPPORTED_SCHEMA_VERSIONS)}). "
            "Refusing to write a file its own reader would reject."
        )

    if "coordinate_space" in metadata:
        space = metadata["coordinate_space"]
        if space not in COORDINATE_SPACES:
            raise ValueError(
                f"metadata declares coordinate_space={space!r}, which is not a "
                f"space this format knows (known: {sorted(COORDINATE_SPACES)}). "
                "The vocabulary is closed on purpose: a misspelled space name "
                "reads back as an unrecognised string, and a consumer then "
                "either guesses the frame or skips the file. Refusing the "
                "write is the cheaper failure."
            )

    if "reference_ply_vertex_count" in metadata:
        _parse_metadata_int(metadata, "reference_ply_vertex_count")

    if "dedup_epsilon" in metadata:
        _parse_metadata_float(metadata, "dedup_epsilon")

    if "reference_ply" in metadata and not metadata["reference_ply"].strip():
        raise ValueError(
            "metadata 'reference_ply' is blank. The key exists to name the "
            "mesh a vertex_id indexes; an empty name identifies nothing."
        )

    if has_vertex_id:
        if version not in VERTEX_ID_SCHEMA_VERSIONS:
            raise ValueError(
                f"table carries {VERTEX_ID_COLUMN!r} but metadata declares "
                f"schema_version={version!r}, which has no such column (versions "
                f"that do: {sorted(VERTEX_ID_SCHEMA_VERSIONS)}). A stage that "
                "adds the column must restamp the version rather than carry the "
                "input's through, or the file's declared schema contradicts its "
                "own columns."
            )
        missing_provenance = [
            key for key in REFERENCE_PLY_METADATA_KEYS if key not in metadata
        ]
        if missing_provenance:
            raise ValueError(
                f"table carries {VERTEX_ID_COLUMN!r} but metadata omits "
                f"{missing_provenance}. A vertex index is meaningless without "
                f"the identity of the mesh it indexes: all of "
                f"{list(REFERENCE_PLY_METADATA_KEYS)} are required together. "
                "Refusing to write an index nothing can be validated against."
            )


def write_contact_depth_field(
    frames: Sequence[ContactDepthFrame],
    output_path: Path,
    *,
    source_recording: str,
) -> None:
    """Write the long-form sidecar.  Overwrites wholesale; never appends.

    Args:
        frames: The contacting frames of one recording, in frame order.  Frames
            with no contact are simply not present — the producer drops them.
            Every frame must carry a populated ``frame_index`` and ``time_s``;
            an unlabelled frame cannot be joined to the CSV and is refused.
        output_path: Destination ``.parquet`` path.  Its parent must exist.
        source_recording: The recording stem this field belongs to, recorded in
            file metadata.

    Raises:
        ValueError: If ``frames`` is empty or ``None``, if any frame lacks
            frame identity, or if any frame violates the index-alignment
            contract between ``points`` and ``signed_depth_mm``.
    """
    if frames is None:
        raise ValueError(
            "frames is None. A recording with no contact at all is a fact worth "
            "surfacing, not a zero-row file to write silently."
        )
    if len(frames) == 0:
        raise ValueError(
            "frames is empty: no frame in this recording reported contact. "
            "Refusing to write a zero-row sidecar, which a later run would "
            "treat as a valid, complete artifact."
        )
    if not source_recording:
        raise ValueError("source_recording must be a non-empty string.")

    frame_index_chunks = []
    time_chunks = []
    point_chunks = []
    depth_chunks = []

    for frame in frames:
        if frame.frame_index is None or frame.time_s is None:
            raise ValueError(
                f"ContactDepthFrame is unlabelled (frame_index="
                f"{frame.frame_index!r}, time_s={frame.time_s!r}). The sidecar "
                "joins to the CSV on frame_index; an unlabelled frame cannot be "
                "written."
            )

        n_vertices = len(frame.points)
        if n_vertices != len(frame.signed_depth_mm):
            raise ValueError(
                f"Frame {frame.frame_index}: {n_vertices} points vs "
                f"{len(frame.signed_depth_mm)} depth values. The field must be "
                "index-aligned with the contact points."
            )
        if n_vertices == 0:
            raise ValueError(
                f"Frame {frame.frame_index} carries zero contact vertices. A "
                "non-contacting frame must be absent from the series, not "
                "present with an empty patch."
            )

        frame_index_chunks.append(np.full(n_vertices, frame.frame_index, dtype=np.int32))
        time_chunks.append(np.full(n_vertices, frame.time_s, dtype=np.float64))
        point_chunks.append(np.asarray(frame.points, dtype=np.float32))
        depth_chunks.append(np.asarray(frame.signed_depth_mm, dtype=np.float64))

    frame_indices = np.concatenate(frame_index_chunks)
    times = np.concatenate(time_chunks)
    points = np.concatenate(point_chunks, axis=0)
    depths = np.concatenate(depth_chunks)

    # The producing path runs before projection, so it never has a vertex_id to
    # write: the narrow layout is not a simplification here, it is the truth.
    schema = pa.schema(_schema_fields_for(has_vertex_id=False)).with_metadata(
        {
            "schema_version": SCHEMA_VERSION,
            "coordinate_space": COORDINATE_SPACE,
            "units": UNITS,
            "sign_convention": SIGN_CONVENTION,
            "source_recording": source_recording,
            "produced_by": PRODUCED_BY,
        }
    )

    table = pa.Table.from_arrays(
        [
            pa.array(frame_indices, type=pa.int32()),
            pa.array(times, type=pa.float64()),
            pa.array(points[:, 0], type=pa.float32()),
            pa.array(points[:, 1], type=pa.float32()),
            pa.array(points[:, 2], type=pa.float32()),
            pa.array(depths, type=pa.float64()),
        ],
        schema=schema,
    )

    _write_arrow_table_atomically(table, output_path)


def write_contact_depth_field_table(
    table: pd.DataFrame,
    output_path: Path,
    *,
    metadata: Dict[str, str],
) -> None:
    """Write a schema-conforming long-form table with explicit file metadata.

    The counterpart of :func:`write_contact_depth_field` for callers that
    already hold rows rather than :class:`ContactDepthFrame` objects — a stage
    that read a sidecar, selected a subset of its rows, and is writing the
    result back.  Overwrites wholesale; never appends.

    Nothing is recomputed, rescaled, rounded or reordered: the rows handed in
    are the rows written out, in schema order, at their existing precision.
    Why those rows and not others is the caller's business and is not knowledge
    this module holds.

    Args:
        table: Long-form rows matching :data:`COLUMN_DTYPES` — optionally
            followed by ``vertex_id`` — in name, order and dtype.  Its index is
            ignored and not written, so a table filtered with a boolean mask
            needs no ``reset_index``.
        output_path: Destination ``.parquet`` path.  Its parent must exist.
        metadata: Complete file metadata, written verbatim.  Unlike the frame
            writer, this function supplies no defaults — the caller states the
            provenance it means, including any keys beyond the standard six.
            Must contain ``schema_version``; all keys and values must be ``str``.
            When ``table`` carries ``vertex_id`` it must additionally declare a
            version in :data:`VERTEX_ID_SCHEMA_VERSIONS` and every key in
            :data:`REFERENCE_PLY_METADATA_KEYS`.

    Raises:
        ValueError: If ``table`` is ``None``, is not a DataFrame, is empty, or
            deviates from the two legal column layouts in names, order or
            dtypes; or if ``metadata`` is not a ``str``-to-``str`` mapping, omits
            ``schema_version``, declares a version outside
            :data:`SUPPORTED_SCHEMA_VERSIONS`, declares a ``coordinate_space``
            outside :data:`COORDINATE_SPACES`, carries a malformed
            ``reference_ply_vertex_count`` or ``dedup_epsilon``, or omits the
            reference-PLY provenance a ``vertex_id`` column requires.
    """
    has_vertex_id = _validate_table_against_schema(table)
    _validate_supplied_metadata(metadata, has_vertex_id=has_vertex_id)

    schema_fields = _schema_fields_for(has_vertex_id=has_vertex_id)
    schema = pa.schema(schema_fields).with_metadata(dict(metadata))

    # Column by column against the declared arrow types rather than
    # ``Table.from_pandas``: dtypes were just proved to match exactly, so this
    # is a reinterpretation and not a cast, and it cannot smuggle in pandas'
    # own index/schema metadata alongside the caller's.
    arrow_table = pa.Table.from_arrays(
        [
            pa.array(table[field.name].to_numpy(), type=field.type)
            for field in schema_fields
        ],
        schema=schema,
    )

    _write_arrow_table_atomically(arrow_table, output_path)


def read_contact_depth_field(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Return ``(long-form rows, file metadata)``.

    Args:
        path: The ``.parquet`` sidecar to read.

    Returns:
        A ``(DataFrame, metadata)`` pair.  The DataFrame carries the required
        columns and dtypes documented in the module docstring, in schema order,
        followed by ``vertex_id`` when the file has one — presence is read off
        the file, never assumed from the version, because v2 is legal both with
        and without it.  The metadata dict is decoded from UTF-8 and contains at
        least ``schema_version``, ``coordinate_space``, ``units``,
        ``sign_convention``, ``source_recording`` and ``produced_by``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file carries no schema metadata, declares a
            ``schema_version`` this reader does not understand, omits a required
            column, or carries ``vertex_id`` under a version that has no such
            column.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Contact depth field sidecar not found: {path}")

    table = pq.read_table(path)

    raw_metadata = table.schema.metadata
    if not raw_metadata:
        raise ValueError(
            f"{path} carries no schema metadata. A contact depth field sidecar "
            "must declare its schema version, coordinate space, units and sign "
            "convention; a file without them cannot be interpreted safely."
        )

    metadata = {
        key.decode("utf-8"): value.decode("utf-8") for key, value in raw_metadata.items()
    }

    version = metadata.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"{path} declares schema_version={version!r}, which this reader does "
            f"not support (known: {sorted(SUPPORTED_SCHEMA_VERSIONS)}). Refusing "
            "to decode it on a guess."
        )

    file_columns = table.column_names
    missing = [c for c in COLUMN_DTYPES if c not in file_columns]
    if missing:
        raise ValueError(
            f"{path} omits the required column(s) {missing}. Every contact "
            f"depth field file carries {list(COLUMN_DTYPES)}; columns found: "
            f"{file_columns}."
        )

    has_vertex_id = VERTEX_ID_COLUMN in file_columns
    if has_vertex_id and version not in VERTEX_ID_SCHEMA_VERSIONS:
        raise ValueError(
            f"{path} carries a {VERTEX_ID_COLUMN!r} column but declares "
            f"schema_version={version!r}, which has no such column (versions "
            f"that do: {sorted(VERTEX_ID_SCHEMA_VERSIONS)}). The file "
            "contradicts its own declared schema; refusing to decide which "
            "half to believe."
        )

    columns = list(COLUMN_DTYPES) + ([VERTEX_ID_COLUMN] if has_vertex_id else [])
    frame = table.to_pandas()
    return frame[columns], metadata


def validate_vertex_ids_against_reference(
    table: pd.DataFrame,
    metadata: Dict[str, str],
    *,
    reference_vertex_count: int,
    reference_description: str = "the reference mesh",
) -> None:
    """Raise unless ``table``'s ``vertex_id`` column indexes the mesh it claims to.

    Two independent failures are caught, and they fail differently on purpose.

    The **count check** compares the ``reference_ply_vertex_count`` recorded
    when the ids were assigned against the vertex count of the mesh they are now
    being joined to.  This is the guard against a re-deduplicated forearm: a
    changed ``dedup_epsilon`` renumbers every vertex, and because task
    idempotency is decided from file timestamps rather than from the DAG config,
    nothing upstream notices.  A count disagreement is the cheapest observable
    consequence of that renumbering, and it is deliberately checked *before* the
    range check — ids that happen to remain in range are the dangerous case.

    The **range check** then proves every id addresses a vertex that exists.

    Purity: this takes a *count*, not a mesh.  Opening a PLY would put a
    geometry engine behind the serialisation seam and make this module
    unimportable wherever Open3D is absent, which is most of the test suite.
    Reading the mesh, and deciding which mesh, is the caller's business.

    Args:
        table: A depth-field table, as returned by
            :func:`read_contact_depth_field`.  Must carry ``vertex_id``.
        metadata: That table's file metadata.
        reference_vertex_count: The vertex count of the mesh the caller is about
            to join ``vertex_id`` against.
        reference_description: How to name that mesh in error messages — a path,
            a stem, anything that lets a reader find it.

    Raises:
        ValueError: If ``reference_vertex_count`` is not a positive integer; if
            ``table`` carries no ``vertex_id``; if the column is not ``int32``;
            if ``metadata`` omits ``reference_ply_vertex_count`` or records a
            count other than ``reference_vertex_count``; or if any ``vertex_id``
            falls outside ``[0, reference_vertex_count)``.
    """
    if not isinstance(reference_vertex_count, (int, np.integer)) or isinstance(
        reference_vertex_count, bool
    ):
        raise ValueError(
            f"reference_vertex_count must be an integer, got "
            f"{type(reference_vertex_count).__name__}. This helper validates "
            "against a vertex count, not against a mesh; the caller reads the "
            "mesh and passes its count."
        )
    reference_vertex_count = int(reference_vertex_count)
    if reference_vertex_count <= 0:
        raise ValueError(
            f"reference_vertex_count is {reference_vertex_count}; "
            f"{reference_description} has no vertices to index into."
        )

    if VERTEX_ID_COLUMN not in table.columns:
        raise ValueError(
            f"table carries no {VERTEX_ID_COLUMN!r} column, so there is nothing "
            f"to validate against {reference_description}. The column is "
            "assigned at the projection stage; a table from an earlier stage "
            "cannot be joined to a reference mesh by index."
        )

    actual_dtype = table[VERTEX_ID_COLUMN].dtype
    expected_dtype = OPTIONAL_COLUMN_DTYPES[VERTEX_ID_COLUMN]
    if actual_dtype != expected_dtype:
        raise ValueError(
            f"table column {VERTEX_ID_COLUMN!r} has dtype {actual_dtype}, "
            f"expected {expected_dtype}. A widened index is an index that came "
            "from somewhere other than this schema."
        )

    if "reference_ply_vertex_count" not in metadata:
        raise ValueError(
            f"metadata carries no 'reference_ply_vertex_count', so a "
            f"{VERTEX_ID_COLUMN!r} in it cannot be proved to index "
            f"{reference_description}. Provenance found: {sorted(metadata)}."
        )

    recorded = _parse_metadata_int(metadata, "reference_ply_vertex_count")
    if recorded != reference_vertex_count:
        raise ValueError(
            f"vertex_id provenance mismatch: the ids were assigned against a "
            f"mesh of {recorded} vertices (reference_ply="
            f"{metadata.get('reference_ply', '<unrecorded>')!r}, dedup_epsilon="
            f"{metadata.get('dedup_epsilon', '<unrecorded>')!r}), but "
            f"{reference_description} has {reference_vertex_count}. The two are "
            "not the same mesh, so every vertex_id points at a different vertex "
            "than it did when it was written — most likely the forearm was "
            "re-deduplicated at a different epsilon. Refusing to join them."
        )

    ids = table[VERTEX_ID_COLUMN].to_numpy()
    if ids.size:
        minimum = int(ids.min())
        maximum = int(ids.max())
        if minimum < 0 or maximum >= reference_vertex_count:
            out_of_range = ids[(ids < 0) | (ids >= reference_vertex_count)]
            raise ValueError(
                f"{out_of_range.size} vertex_id value(s) fall outside "
                f"[0, {reference_vertex_count}) for {reference_description} "
                f"(observed range [{minimum}, {maximum}]; e.g. "
                f"{np.unique(out_of_range)[:10].tolist()}). A vertex index that "
                "addresses no vertex cannot be resolved to a position."
            )
