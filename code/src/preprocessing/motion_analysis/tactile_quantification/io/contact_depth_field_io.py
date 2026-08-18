"""Read/write the long-form per-vertex contact depth field sidecar.

The sidecar is a parquet file holding **exactly one row per (frame, contact
vertex)**.  It is the companion of the per-frame
``*_contact_and_kinematic_data.csv``; the two join on ``frame_index``.

Schema
------
=================  =========  ==================================================
column             dtype      meaning
=================  =========  ==================================================
``frame_index``    int32      Kinect frame index; joins to the CSV
``time_s``         float64    frame timestamp, seconds
``x``, ``y``, ``z``  float32  contact vertex position, mm, Kinect Space 1
``signed_depth_mm``  float64  signed distance to the hand surface, mm
=================  =========  ==================================================

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
``x/y/z`` are in **Kinect Space 1** — the raw Kinect frame, *before* ICP
registration, PCA calibration and receptive-field centring.  They become
wrong-space the moment postprocessing runs, which is why every file declares
``coordinate_space`` in its metadata rather than leaving it to convention.

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
    "COORDINATE_SPACE",
    "PRODUCED_BY",
    "SCHEMA_VERSION",
    "SIGN_CONVENTION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "UNITS",
    "read_contact_depth_field",
    "write_contact_depth_field",
    "write_contact_depth_field_table",
]


#: Version of the on-disk schema.  Present from day one: the closest prior art
#: (``population-rf-vertex-data-export.md``) shipped without one and had its
#: file renamed and its schema widened five days later.
SCHEMA_VERSION: str = "1"

#: Schema versions this reader understands.  Anything else raises rather than
#: being decoded on a guess.
SUPPORTED_SCHEMA_VERSIONS: frozenset = frozenset({"1"})

#: Raw Kinect frame, millimetres, before ICP / PCA / RF-centring.
COORDINATE_SPACE: str = "kinect_space_1"

UNITS: str = "mm"

SIGN_CONVENTION: str = "negative_is_penetrating"

#: The only producer of this artifact today.  A constant rather than a
#: parameter: the writer is not given, and must not acquire, pipeline
#: knowledge, and a provenance string that every caller must supply correctly
#: is a provenance string that will eventually be supplied incorrectly.
PRODUCED_BY: str = "compute_somatosensory_characteristics"

_SCHEMA_FIELDS = [
    pa.field("frame_index", pa.int32()),
    pa.field("time_s", pa.float64()),
    pa.field("x", pa.float32()),
    pa.field("y", pa.float32()),
    pa.field("z", pa.float32()),
    pa.field("signed_depth_mm", pa.float64()),
]

#: Column name -> numpy dtype, used to assert the read-back contract.
COLUMN_DTYPES: Dict[str, np.dtype] = {
    "frame_index": np.dtype(np.int32),
    "time_s": np.dtype(np.float64),
    "x": np.dtype(np.float32),
    "y": np.dtype(np.float32),
    "z": np.dtype(np.float32),
    "signed_depth_mm": np.dtype(np.float64),
}


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


def _validate_table_against_schema(table: pd.DataFrame) -> None:
    """Raise unless ``table`` matches :data:`COLUMN_DTYPES` exactly.

    Names, order and dtypes are all checked.  Nothing is coerced: the caller
    handing over a mis-ordered or mis-typed frame has a bug, and repairing it
    here would hide a downcast of ``signed_depth_mm`` behind a successful write.
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

    expected_columns = list(COLUMN_DTYPES)
    actual_columns = list(table.columns)
    if actual_columns != expected_columns:
        missing = [c for c in expected_columns if c not in actual_columns]
        unexpected = [c for c in actual_columns if c not in expected_columns]
        if missing or unexpected:
            raise ValueError(
                f"table columns do not match the contact depth field schema: "
                f"missing={missing}, unexpected={unexpected}. Expected exactly "
                f"{expected_columns}, got {actual_columns}."
            )
        raise ValueError(
            f"table columns are in the wrong order: expected {expected_columns}, "
            f"got {actual_columns}. Column order is part of the on-disk schema; "
            "refusing to reorder silently."
        )

    for column, expected_dtype in COLUMN_DTYPES.items():
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


def _validate_supplied_metadata(metadata: Dict[str, str]) -> None:
    """Raise unless ``metadata`` is a str->str mapping declaring a known version.

    The schema version is checked at *write* time, mirroring the reader: a file
    stamped with a version this module does not understand is unreadable the
    moment it lands, and finding that out on the next read is finding out too
    late.
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

    schema = pa.schema(_SCHEMA_FIELDS).with_metadata(
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
        table: Long-form rows matching :data:`COLUMN_DTYPES` in name, order and
            dtype.  Its index is ignored and not written, so a table filtered
            with a boolean mask needs no ``reset_index``.
        output_path: Destination ``.parquet`` path.  Its parent must exist.
        metadata: Complete file metadata, written verbatim.  Unlike the frame
            writer, this function supplies no defaults — the caller states the
            provenance it means, including any keys beyond the standard six.
            Must contain ``schema_version``; all keys and values must be ``str``.

    Raises:
        ValueError: If ``table`` is ``None``, is not a DataFrame, is empty, or
            deviates from :data:`COLUMN_DTYPES` in column names, column order or
            dtypes; or if ``metadata`` is not a ``str``-to-``str`` mapping, omits
            ``schema_version``, or declares a version outside
            :data:`SUPPORTED_SCHEMA_VERSIONS`.
    """
    _validate_table_against_schema(table)
    _validate_supplied_metadata(metadata)

    schema = pa.schema(_SCHEMA_FIELDS).with_metadata(dict(metadata))

    # Column by column against the declared arrow types rather than
    # ``Table.from_pandas``: dtypes were just proved to match exactly, so this
    # is a reinterpretation and not a cast, and it cannot smuggle in pandas'
    # own index/schema metadata alongside the caller's.
    arrow_table = pa.Table.from_arrays(
        [
            pa.array(table[field.name].to_numpy(), type=field.type)
            for field in _SCHEMA_FIELDS
        ],
        schema=schema,
    )

    _write_arrow_table_atomically(arrow_table, output_path)


def read_contact_depth_field(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Return ``(long-form rows, file metadata)``.

    Args:
        path: The ``.parquet`` sidecar to read.

    Returns:
        A ``(DataFrame, metadata)`` pair.  The DataFrame carries the columns and
        dtypes documented in the module docstring, in schema order.  The
        metadata dict is decoded from UTF-8 and contains at least
        ``schema_version``, ``coordinate_space``, ``units``,
        ``sign_convention``, ``source_recording`` and ``produced_by``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file carries no schema metadata, or declares a
            ``schema_version`` this reader does not understand.
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

    frame = table.to_pandas()
    return frame[list(COLUMN_DTYPES)], metadata
