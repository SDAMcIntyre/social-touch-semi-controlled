"""Per-stage transformation of the contact-depth-field sidecar, and the
row-count agreement check that keeps it synchronised with its CSV.

Why this is a module and not code inside the stage scripts
----------------------------------------------------------
Three of the five postprocessing stages that move contact coordinates
(``apply_icp_registration``, ``project_contacts_onto_forearm``,
``set_xyz_reference_from_gestures``, ``center_on_receptive_field``,
``deduplicate_xy_points``) import ``open3d`` or ``matplotlib`` at module scope
and are therefore **not importable** in the unit-test environment.  Logic placed
in them cannot be tested at all.  Everything the five stages need to do to the
sidecar lives here instead, on synthetic tables, so the stage scripts stay thin
drivers.  This mirrors ``merging/contact_depth_field_series.py`` and
``merging/frame_navigation.py``, which exist for exactly the same reason.

Two mechanisms, not five
------------------------
Every stage is one of two kinds, and the distinction is the whole design:

**Replayable** — ICP, PCA calibration, RF-centring.  The transform is a matrix
or a :class:`CalibrationResult`, applicable to an ``(N, 3)`` array.  The parquet
is moved by *the same object* the CSV was moved by, never a re-derived one.
Coordinates change; ``signed_depth_mm`` is bitwise untouched, because a depth is
a measurement of penetration and a rigid transform does not change how far the
hand went in.

**Index-consuming** — deduplication and projection.  Neither is a coordinate
transform; each computes a per-row index mapping that the parquet must be
filtered or re-addressed by.  Re-running DBSCAN or the KD-tree against the
parquet's own coordinates would pick different clusters and different nearest
vertices wherever two candidates are near-equidistant — the CSV holds float64
parsed back from a ``%.1f`` string, the parquet holds full float32 — and would
do so **silently**, leaving two artifacts that describe different geometry with
no error anywhere.  Hence: never recompute an index against the parquet.

The row-count agreement check
-----------------------------
:func:`assert_row_counts_agree_with_csv` is the safety net the whole propagation
rests on.  ``parse_contact_points`` drops a malformed point silently — its
``if len(parts) == 3`` has no ``else`` — so a single corrupt ``contact_points``
cell shortens the CSV's point list without shortening the parquet's rows, and
every subsequent index mapping is then applied to the wrong rows.  Comparing
per-frame counts after every stage turns that into a loud failure at the stage
that caused it.  The check parses with ``parse_contact_points`` itself, so it
sees exactly what the stage saw rather than a stricter second opinion.

Precision contract
------------------
All coordinate arithmetic runs in **float64** and is cast back to **float32** on
return, because the schema stores ``x/y/z`` as float32 and the writer's dtype
validation rejects anything else.  ``signed_depth_mm`` is float64 and passes
through the rigid, scheduled and PCA paths **bitwise unchanged** — only
:func:`apply_dedup_mapping_to_field` may alter it, and only by replacing a
survivor's value with another value taken verbatim from its own collapsed group.

Purity contract
---------------
Every ``apply_*`` function is pure: it takes a table and returns a new table,
and never mutates its arguments.  The returned table is re-indexed ``0..N-1``.
This module knows the depth-field schema, numpy, pandas, and the two CSV columns
:data:`FRAME_INDEX_COLUMN` and :data:`CONTACT_POINTS_COLUMN`.  It must not learn
which stage called it, nor about Prefect, ``KinectConfig``, session identity,
DAG configs, or any path beyond the one CSV it is handed.  It imports no
geometry engine and no GUI toolkit: a vertex re-addressing takes a vertex
*array*, and the count check takes a *path to a CSV*, never a mesh.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd

from postprocessing.xyz_reference_from_gestures.calibration_pca_engine import (
    CalibrationResult,
    PCACalibrationEngine,
)
from preprocessing.forearm_extraction.registration.csv_spatial_transformer import (
    apply_rigid_transform as apply_rigid_transform_to_points,
    parse_contact_points,
)
from preprocessing.motion_analysis.tactile_quantification.io.contact_depth_field_io import (
    ALL_COLUMN_DTYPES,
    COLUMN_DTYPES,
    VERTEX_ID_COLUMN,
)

__all__ = [
    "CONTACT_POINTS_COLUMN",
    "COORDINATE_COLUMNS",
    "DEPTH_COLUMN",
    "FRAME_INDEX_COLUMN",
    "DedupMappingLike",
    "TransformSchedule",
    "apply_dedup_mapping_to_field",
    "apply_pca_calibration_to_field",
    "apply_rigid_transform_to_field",
    "apply_transform_schedule_to_field",
    "apply_vertex_addressing_to_field",
    "assert_row_counts_agree_with_csv",
    "csv_contact_point_counts_by_frame",
    "field_row_counts_by_frame",
]

#: The frame key both artifacts join on.  Row *position* is not a key: the
#: merged CSV is upsampled to the nerve rate, so a positional offset means
#: nothing outside one particular file.
FRAME_INDEX_COLUMN: str = "frame_index"

#: The one CSV column this module parses, besides the frame key.
CONTACT_POINTS_COLUMN: str = "contact_points"

#: The coordinate triplet, in schema order.
COORDINATE_COLUMNS: Tuple[str, str, str] = ("x", "y", "z")

#: The measurement that survives every replayable stage untouched.
DEPTH_COLUMN: str = "signed_depth_mm"

#: An ordered ``[(start_frame, T_4x4), ...]`` list as
#: ``csv_spatial_transformer.get_transform_schedule`` returns it.  Entry *i*
#: covers the half-open interval ``[start_i, start_{i+1})``; the last entry
#: covers ``[start, +inf)``.  Frames below the first ``start_frame`` are left
#: alone — the CSV path leaves them alone too.
TransformSchedule = Sequence[Tuple[int, np.ndarray]]

_LEGAL_COLUMN_LAYOUTS: Tuple[Tuple[str, ...], ...] = (
    tuple(COLUMN_DTYPES),
    tuple(ALL_COLUMN_DTYPES),
)


class DedupMappingLike(Protocol):
    """Structural type of what ``deduplicate_xy_mapping`` returns per frame.

    Declared structurally rather than imported so that this module does not
    depend on a stage script — the stage scripts depend on this module, and the
    import graph must run one way.  ``_5_postprocessing.deduplicate_xy_points.
    DedupMapping`` satisfies it as written, so a stage passes its
    ``frame_mappings`` straight through with no conversion.

    Attributes:
        kept_indices: Ascending 1-D integer array of the input rows that
            survived deduplication.
        labels: One entry per *input* row, giving the cluster that row was
            assigned to.  Rows sharing a label collapsed into a single survivor.
    """

    kept_indices: np.ndarray
    labels: np.ndarray


# ---------------------------------------------------------------------------
# Table validation and coordinate plumbing
# ---------------------------------------------------------------------------


def _validate_field_table(table: pd.DataFrame, *, what: str = "table") -> bool:
    """Raise unless *table* is a schema-conforming depth field; report its shape.

    Args:
        table: The candidate table.
        what: How to name it in error messages.

    Returns:
        ``True`` when the table carries ``vertex_id``.

    Raises:
        ValueError: If *table* is not a DataFrame, is empty, or deviates from
            the two legal column layouts in names, order or dtypes.
    """
    if not isinstance(table, pd.DataFrame):
        raise ValueError(
            f"{what} must be a pandas DataFrame, got {type(table).__name__}."
        )
    if len(table) == 0:
        raise ValueError(
            f"{what} has zero rows. An empty depth field is indistinguishable "
            "from a complete one on the next run; a stage that produced no rows "
            "must say so rather than pass an empty table on."
        )

    columns = tuple(table.columns)
    if columns not in _LEGAL_COLUMN_LAYOUTS:
        raise ValueError(
            f"{what} has columns {list(columns)}, which is neither "
            f"{list(_LEGAL_COLUMN_LAYOUTS[0])} nor "
            f"{list(_LEGAL_COLUMN_LAYOUTS[1])}. The layout is exact — names and "
            "order — so that a column nobody declared cannot reach disk."
        )

    for name in columns:
        expected = ALL_COLUMN_DTYPES[name]
        actual = table[name].dtype
        if actual != expected:
            raise ValueError(
                f"{what} column {name!r} has dtype {actual}, expected {expected}. "
                "Nothing is coerced here: a widened dtype means the column came "
                "from somewhere other than the schema's reader."
            )

    return VERTEX_ID_COLUMN in columns


def _coordinates_float64(table: pd.DataFrame) -> np.ndarray:
    """Return the ``(N, 3)`` coordinate block as a fresh float64 array."""
    return np.ascontiguousarray(
        table[list(COORDINATE_COLUMNS)].to_numpy(dtype=np.float64, copy=True)
    )


def _with_coordinates(table: pd.DataFrame, xyz: np.ndarray) -> pd.DataFrame:
    """Return a copy of *table* whose ``x/y/z`` are *xyz*, cast to float32.

    Every other column — ``signed_depth_mm`` above all — is copied verbatim, so
    a caller that only moved coordinates cannot accidentally perturb a
    measurement.

    Args:
        table: The source table; not modified.
        xyz: ``(N, 3)`` float64 coordinates in the destination space.

    Returns:
        A new table, indexed ``0..N-1``.

    Raises:
        ValueError: If *xyz* has the wrong shape, is not finite, or overflows
            float32 on the cast back.
    """
    if xyz.shape != (len(table), 3):
        raise ValueError(
            f"Transformed coordinates have shape {xyz.shape}, expected "
            f"{(len(table), 3)}. A transform must not add or drop rows."
        )
    if not np.all(np.isfinite(xyz)):
        raise ValueError(
            "Transformed coordinates contain non-finite values. A rigid or PCA "
            "transform of finite input cannot produce one; the transform itself "
            "is malformed."
        )

    narrowed = xyz.astype(np.float32)
    if not np.all(np.isfinite(narrowed)):
        raise ValueError(
            "Transformed coordinates overflow float32, which is the schema's "
            "storage dtype for x/y/z. Coordinates are millimetres; a value this "
            "large is a unit or transform error, not a precision one."
        )

    out = table.copy().reset_index(drop=True)
    for axis, name in enumerate(COORDINATE_COLUMNS):
        out[name] = narrowed[:, axis]
    return out


def _frame_index_array(table: pd.DataFrame) -> np.ndarray:
    """Return the frame key column as ``int64``.

    The schema stores it as ``int32`` with no missing value, so this is a
    widening and never a coercion of NaN.
    """
    return table[FRAME_INDEX_COLUMN].to_numpy(dtype=np.int64, copy=True)


def _row_positions_by_frame(table: pd.DataFrame) -> Dict[int, np.ndarray]:
    """Map each frame to the *positional* rows it occupies, in table order.

    Positions, not index labels, because an index-consuming stage addresses the
    k-th contact point of a frame and the table's index may be anything.
    """
    frames = _frame_index_array(table)
    positions: Dict[int, List[int]] = {}
    for position, frame in enumerate(frames):
        positions.setdefault(int(frame), []).append(position)
    return {
        frame: np.asarray(rows, dtype=np.intp) for frame, rows in positions.items()
    }


def _require_same_frames(
    table_frames: Iterable[int],
    mapping_frames: Iterable[int],
    *,
    mapping_name: str,
) -> None:
    """Raise unless the depth field and the per-frame mapping cover one frame set.

    A frame in one and not the other *is* the desynchronisation this whole
    design exists to prevent: the CSV believes a frame has contact points and
    the parquet has no rows for it, or the reverse.  Never tolerated, never
    skipped.
    """
    in_table = set(int(f) for f in table_frames)
    in_mapping = set(int(f) for f in mapping_frames)
    if in_table == in_mapping:
        return

    only_table = sorted(in_table - in_mapping)
    only_mapping = sorted(in_mapping - in_table)
    raise ValueError(
        f"The depth field and {mapping_name} cover different frames. "
        f"{len(only_table)} frame(s) only in the field "
        f"(first: {only_table[:5]}), {len(only_mapping)} only in the mapping "
        f"(first: {only_mapping[:5]}). The two artifacts must describe the same "
        "contact frames at every stage."
    )


# ---------------------------------------------------------------------------
# (a) Replayable: a 4x4 rigid transform, whole-table or per-frame-segment
# ---------------------------------------------------------------------------


def _validate_transform_4x4(transform: np.ndarray, *, what: str) -> np.ndarray:
    """Return *transform* as a validated ``(4, 4)`` float64 array.

    Raises:
        ValueError: If it is not 4x4, not finite, or its bottom row is not
            ``[0, 0, 0, 1]``.  The bottom row is checked because
            ``apply_rigid_transform`` ignores it: a projective matrix would be
            silently applied as its affine part.
    """
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{what} must be 4x4, got shape {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{what} contains non-finite entries:\n{matrix}")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0)):
        raise ValueError(
            f"{what} has bottom row {matrix[3].tolist()}, expected [0, 0, 0, 1]. "
            "Only the rotation and translation are applied, so a projective "
            "matrix would be silently truncated to its affine part."
        )
    return matrix


def apply_rigid_transform_to_field(
    table: pd.DataFrame, transform_4x4: np.ndarray
) -> pd.DataFrame:
    """Move every row's ``x/y/z`` by one 4x4 rigid transform.

    Uses the pipeline's single implementation,
    ``csv_spatial_transformer.apply_rigid_transform``, so the parquet and the
    CSV are moved by the same arithmetic and not merely by the same matrix.

    ``signed_depth_mm`` is not touched.  Penetration depth is invariant under a
    rigid transform — the hand and the forearm move together — so re-deriving it
    here would replace a measurement with a coordinate artefact.

    Args:
        table: A schema-conforming depth field.  Not modified.
        transform_4x4: The rigid transform, in the same convention the CSV path
            uses.

    Returns:
        A new table in the destination space, indexed ``0..N-1``.

    Raises:
        ValueError: If *table* is not schema-conforming, or *transform_4x4* is
            not a finite 4x4 with bottom row ``[0, 0, 0, 1]``.
    """
    _validate_field_table(table, what="depth field table")
    matrix = _validate_transform_4x4(transform_4x4, what="transform_4x4")

    moved = apply_rigid_transform_to_points(_coordinates_float64(table), matrix)
    return _with_coordinates(table, moved)


def apply_transform_schedule_to_field(
    table: pd.DataFrame, schedule: TransformSchedule
) -> pd.DataFrame:
    """Move each row by the transform its frame falls under.

    The ICP stage applies a **schedule**, not a matrix: a forearm snapshot is
    valid from its capture frame forward, so a block can change transform
    mid-recording.  The segmentation here is the same half-open one
    ``transform_spatial_columns_scheduled`` applies to the CSV — entry *i*
    covers ``[start_i, start_{i+1})``, the last covers ``[start, +inf)`` — and
    the caller must pass **the same schedule object**, built from the CSV's
    frame range.  Rebuilding it from the parquet would use a different
    ``max_frame``, because the parquet holds only contacting frames.

    Segments are disjoint, so each row is transformed exactly once, from its
    original coordinates.  A row whose frame precedes the first segment is left
    untouched, as it is in the CSV.

    Args:
        table: A schema-conforming depth field.  Not modified.
        schedule: Ordered ``[(start_frame, T_4x4), ...]``.  Empty means no
            transform applies and the table passes through unchanged — the
            explicit passthrough the ICP stage takes when a block has no
            registration snapshot.

    Returns:
        A new table, indexed ``0..N-1``.

    Raises:
        ValueError: If *table* is not schema-conforming, if a start frame is
            negative, if the start frames are not non-decreasing, or if any
            matrix fails :func:`_validate_transform_4x4`.
    """
    _validate_field_table(table, what="depth field table")

    entries = list(schedule)
    coordinates = _coordinates_float64(table)
    if not entries:
        return _with_coordinates(table, coordinates)

    starts: List[int] = []
    matrices: List[np.ndarray] = []
    for position, entry in enumerate(entries):
        start, matrix = entry
        start = int(start)
        if start < 0:
            raise ValueError(
                f"Schedule entry {position} starts at frame {start}; a segment "
                "cannot begin before frame 0."
            )
        if starts and start < starts[-1]:
            raise ValueError(
                f"Schedule entry {position} starts at frame {start}, before its "
                f"predecessor at {starts[-1]}. The segments are half-open "
                "intervals and must be given in non-decreasing frame order."
            )
        starts.append(start)
        matrices.append(
            _validate_transform_4x4(matrix, what=f"schedule entry {position}")
        )

    frames = _frame_index_array(table)
    moved = coordinates.copy()
    for position, (start, matrix) in enumerate(zip(starts, matrices)):
        end = starts[position + 1] if position + 1 < len(starts) else None
        mask = frames >= start
        if end is not None:
            mask &= frames < end
        if not mask.any():
            continue
        moved[mask] = apply_rigid_transform_to_points(coordinates[mask], matrix)

    return _with_coordinates(table, moved)


# ---------------------------------------------------------------------------
# (b) Replayable: the PCA calibration
# ---------------------------------------------------------------------------


def apply_pca_calibration_to_field(
    table: pd.DataFrame, calibration: CalibrationResult
) -> pd.DataFrame:
    """Move every row's ``x/y/z`` by the session's PCA calibration.

    The caller must pass **the same** :class:`CalibrationResult` the CSV was
    calibrated with.  Re-fitting a PCA on the parquet's own points would define
    a different coordinate system from the same session's data, which is the
    silent-divergence failure this whole propagation is built to avoid — and the
    fit must not see the depth field at all, because that would change the space
    itself and invalidate every existing ``rf_center_origin.json``.

    ``PCACalibrationEngine.apply_full_transform`` writes into the array it is
    given part-way through, so it is handed a private copy; the caller's table
    and the array it derived are both left intact.

    Args:
        table: A schema-conforming depth field.  Not modified.
        calibration: The session's calibration result.

    Returns:
        A new table in the PCA-calibrated space, indexed ``0..N-1``.

    Raises:
        ValueError: If *table* is not schema-conforming, or *calibration* does
            not carry finite ``mean_1`` ``(3,)``, ``R1`` ``(3, 3)``,
            ``mean_2`` ``(2,)`` and ``R2`` ``(3, 3)``.
    """
    _validate_field_table(table, what="depth field table")
    _validate_calibration(calibration)

    coordinates = _coordinates_float64(table)
    moved = PCACalibrationEngine.apply_full_transform(coordinates.copy(), calibration)
    return _with_coordinates(table, np.asarray(moved, dtype=np.float64))


def _validate_calibration(calibration: CalibrationResult) -> None:
    """Raise unless *calibration* carries the four finite arrays it must."""
    expected_shapes = {"mean_1": (3,), "R1": (3, 3), "mean_2": (2,), "R2": (3, 3)}
    for name, shape in expected_shapes.items():
        if not hasattr(calibration, name):
            raise ValueError(
                f"calibration has no attribute {name!r}; expected a "
                "CalibrationResult carrying mean_1, R1, mean_2 and R2."
            )
        value = np.asarray(getattr(calibration, name), dtype=np.float64)
        if value.shape != shape:
            raise ValueError(
                f"calibration.{name} has shape {value.shape}, expected {shape}."
            )
        if not np.all(np.isfinite(value)):
            raise ValueError(f"calibration.{name} contains non-finite entries.")


# ---------------------------------------------------------------------------
# (c) Index-consuming: the deduplication mapping
# ---------------------------------------------------------------------------


def apply_dedup_mapping_to_field(
    table: pd.DataFrame,
    frame_mappings: Mapping[int, DedupMappingLike],
) -> pd.DataFrame:
    """Collapse each frame's rows by the mapping the CSV was collapsed by.

    Two things happen, and only these two:

    1. **Rows are dropped.**  Exactly the rows named by each frame's
       ``kept_indices`` survive, in the table's original order.  No coordinate
       moves; deduplication is not a transform, so the coordinate space is
       unchanged and the caller restamps nothing.
    2. **Each survivor inherits its group's depth.**  A survivor's
       ``signed_depth_mm`` becomes the **most negative** value among the rows
       that collapsed into it.

    Why the most negative value.  Storage is signed with *negative meaning
    penetrating*, so the deepest penetration in a collapsed group is its
    minimum, and that is the magnitude the group as a whole reached.  Keeping
    the survivor's own value instead would let deduplication quietly reduce the
    reported depth of a frame — and the pipeline's strongest cross-artifact
    check is that per-frame ``max(|signed_depth_mm|)`` still equals the CSV's
    ``contact_depth``, which was computed *before* deduplication.  The chosen
    value is taken verbatim from the group, never averaged or recomputed: it is
    still a measurement that was actually made, merely re-attributed to the
    vertex that now stands for the group.

    (The field's positive values, where they exist at all, are bounded by the
    contact-detection epsilon — a vertex further outside the hand than that is
    not in the contact patch — so "most negative" and "largest absolute value"
    select the same row on real data.  Where a group mixes signs, the most
    negative row is the one that penetrated, and it is the one that is kept.)

    Args:
        table: A schema-conforming depth field, before deduplication.  Not
            modified.
        frame_mappings: ``frame_index -> mapping``, as
            ``deduplicate_contact_points_csv`` returns in its ``frame_mappings``
            key.  Must cover exactly the frames present in *table*.

    Returns:
        A new table holding only the survivors, indexed ``0..N-1``.

    Raises:
        ValueError: If *table* is not schema-conforming; if the frame sets
            disagree; if a frame's ``labels`` length differs from its row count;
            if ``kept_indices`` is not an ascending set of valid, uniquely
            labelled rows; or if every row of a frame is dropped.
    """
    _validate_field_table(table, what="depth field table")

    rows_by_frame = _row_positions_by_frame(table)
    _require_same_frames(
        rows_by_frame, frame_mappings, mapping_name="the deduplication mapping"
    )

    depths = table[DEPTH_COLUMN].to_numpy(dtype=np.float64, copy=True)
    survivors: List[np.ndarray] = []
    inherited: List[np.ndarray] = []

    for frame in sorted(rows_by_frame):
        rows = rows_by_frame[frame]
        mapping = frame_mappings[frame]
        labels = _validate_labels(mapping, frame=frame, n_rows=len(rows))
        kept = _validate_kept_indices(mapping, frame=frame, labels=labels)

        # One minimum per cluster, then read it back for each survivor.  The
        # scatter-reduce is what makes this O(n) per frame instead of O(n x k).
        group_minimum = np.full(int(labels.max()) + 1, np.inf, dtype=np.float64)
        np.minimum.at(group_minimum, labels, depths[rows])

        survivors.append(rows[kept])
        inherited.append(group_minimum[labels[kept]])

    survivor_positions = np.concatenate(survivors)
    survivor_depths = np.concatenate(inherited)

    # Sorting by position restores the table's own row order, which the
    # per-frame loop above broke apart.
    order = np.argsort(survivor_positions, kind="stable")
    survivor_positions = survivor_positions[order]
    survivor_depths = survivor_depths[order]

    out = table.iloc[survivor_positions].copy().reset_index(drop=True)
    out[DEPTH_COLUMN] = survivor_depths
    return out


def _validate_labels(
    mapping: DedupMappingLike, *, frame: int, n_rows: int
) -> np.ndarray:
    """Return *mapping*'s labels as ``intp``, validated against *n_rows*."""
    labels = np.asarray(getattr(mapping, "labels"))
    if labels.ndim != 1:
        raise ValueError(
            f"Frame {frame}: labels must be one-dimensional, got shape "
            f"{labels.shape}."
        )
    if len(labels) != n_rows:
        raise ValueError(
            f"Frame {frame}: the deduplication mapping describes {len(labels)} "
            f"input point(s) but the depth field holds {n_rows} row(s) for that "
            "frame. The mapping was computed from the CSV, so a disagreement "
            "means the two artifacts have already desynchronised."
        )
    labels = labels.astype(np.intp, copy=False)
    if labels.size and labels.min() < 0:
        raise ValueError(
            f"Frame {frame}: labels contain {int(labels.min())}. The clustering "
            "runs with min_samples=1 and therefore emits no noise label; a "
            "negative label means these are not the labels it produced."
        )
    return labels


def _validate_kept_indices(
    mapping: DedupMappingLike, *, frame: int, labels: np.ndarray
) -> np.ndarray:
    """Return *mapping*'s kept indices as ``intp``, validated against *labels*."""
    kept = np.asarray(getattr(mapping, "kept_indices"))
    if kept.ndim != 1:
        raise ValueError(
            f"Frame {frame}: kept_indices must be one-dimensional, got shape "
            f"{kept.shape}."
        )
    kept = kept.astype(np.intp, copy=False)
    if kept.size == 0:
        raise ValueError(
            f"Frame {frame}: the deduplication mapping keeps no rows at all. A "
            "frame that had contact points before deduplication still has at "
            "least one after it; an empty survivor set is a mapping bug."
        )
    if kept.min() < 0 or kept.max() >= len(labels):
        raise ValueError(
            f"Frame {frame}: kept_indices span [{int(kept.min())}, "
            f"{int(kept.max())}], outside the {len(labels)} row(s) that frame "
            "has."
        )
    if np.any(np.diff(kept) <= 0):
        raise ValueError(
            f"Frame {frame}: kept_indices must be strictly ascending and unique; "
            "the surviving rows are a subset of the input rows in their original "
            "order."
        )

    kept_labels = labels[kept]
    if len(np.unique(kept_labels)) != len(kept_labels):
        raise ValueError(
            f"Frame {frame}: two kept rows share a cluster label. Each collapsed "
            "group has exactly one survivor, so a shared label means the group "
            "would inherit its depth twice."
        )
    if len(kept_labels) != len(np.unique(labels)):
        raise ValueError(
            f"Frame {frame}: {len(kept_labels)} survivor(s) for "
            f"{len(np.unique(labels))} cluster(s). Every cluster must keep "
            "exactly one row, or a group's depth is lost entirely."
        )
    return kept


# ---------------------------------------------------------------------------
# (d) Index-consuming: the projection's vertex re-addressing
# ---------------------------------------------------------------------------


def apply_vertex_addressing_to_field(
    table: pd.DataFrame,
    vertex_indices: Mapping[int, np.ndarray],
    reference_vertices: np.ndarray,
) -> pd.DataFrame:
    """Snap every row onto the forearm vertex the CSV's KD-tree chose for it.

    Each row's ``x/y/z`` is replaced by the coordinates of its matched vertex,
    and the matching index is written to ``vertex_id``.  Row order and row count
    are preserved exactly: projection is a per-point nearest-neighbour lookup
    with no uniqueness constraint, so two rows may legitimately address one
    vertex, and none is ever dropped.

    The indices must come from the CSV's own query.  Re-querying the tree with
    the parquet's coordinates would pick a different vertex wherever two
    candidates are near-equidistant, because the CSV's points came back through
    a ``%.1f`` round trip and the parquet's did not.

    ``vertex_id`` is what makes the field space-independent from here on: the
    three forearm PLYs downstream are transformed in place, in file order, with
    no reordering or count change, so vertex *i* is the same physical vertex in
    all of them.  The provenance that makes the index interpretable —
    ``reference_ply``, ``reference_ply_vertex_count``, ``dedup_epsilon`` — is
    metadata, and stamping it is the calling stage's job.

    Args:
        table: A schema-conforming depth field **without** ``vertex_id``.  Not
            modified.
        vertex_indices: ``frame_index -> (M,) integer array`` of the forearm
            vertex each of that frame's contact points snapped to, in the order
            the points appear.  Must cover exactly the frames in *table*.
        reference_vertices: The ``(V, 3)`` vertex array the indices address, in
            PLY file order — the same array the KD-tree was built from.

    Returns:
        A new table with re-addressed coordinates and an ``int32`` ``vertex_id``
        column, indexed ``0..N-1``.

    Raises:
        ValueError: If *table* already carries ``vertex_id``; if the frame sets
            disagree; if a frame's index count differs from its row count; if
            *reference_vertices* is not a finite ``(V, 3)`` array with ``V > 0``;
            or if any index falls outside ``[0, V)``.
    """
    has_vertex_id = _validate_field_table(table, what="depth field table")
    if has_vertex_id:
        raise ValueError(
            f"The depth field already carries a {VERTEX_ID_COLUMN!r} column. "
            "Vertex identity is assigned once, at the projection stage; "
            "re-addressing an already-addressed field would overwrite an index "
            "into one PLY with an index into another."
        )

    vertices = np.asarray(reference_vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(
            f"reference_vertices must be (V, 3), got shape {vertices.shape}."
        )
    if len(vertices) == 0:
        raise ValueError(
            "reference_vertices is empty; there is no vertex for a contact "
            "point to address."
        )
    if not np.all(np.isfinite(vertices)):
        raise ValueError("reference_vertices contains non-finite coordinates.")

    rows_by_frame = _row_positions_by_frame(table)
    _require_same_frames(
        rows_by_frame, vertex_indices, mapping_name="the projection mapping"
    )

    assigned = np.empty(len(table), dtype=np.int64)
    for frame, rows in rows_by_frame.items():
        indices = np.asarray(vertex_indices[frame])
        if indices.ndim != 1:
            raise ValueError(
                f"Frame {frame}: vertex indices must be one-dimensional, got "
                f"shape {indices.shape}."
            )
        if len(indices) != len(rows):
            raise ValueError(
                f"Frame {frame}: the projection mapping names {len(indices)} "
                f"vertex/vertices but the depth field holds {len(rows)} row(s) "
                "for that frame. The mapping was computed from the CSV, so a "
                "disagreement means the two artifacts have already "
                "desynchronised."
            )
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(
                f"Frame {frame}: vertex indices have dtype {indices.dtype}; a "
                "vertex identity is an integer row number, never a float."
            )
        assigned[rows] = indices

    out_of_range = (assigned < 0) | (assigned >= len(vertices))
    if out_of_range.any():
        first = int(np.flatnonzero(out_of_range)[0])
        raise ValueError(
            f"Row {first} addresses vertex {int(assigned[first])}, outside the "
            f"{len(vertices)} vertices of the reference mesh. Every vertex_id "
            "must index the PLY it was assigned against."
        )

    out = _with_coordinates(table, vertices[assigned])
    out[VERTEX_ID_COLUMN] = assigned.astype(np.int32)
    return out


# ---------------------------------------------------------------------------
# The row-count agreement check
# ---------------------------------------------------------------------------


def field_row_counts_by_frame(table: pd.DataFrame) -> Dict[int, int]:
    """Return ``frame_index -> number of rows`` for a depth field.

    Args:
        table: A schema-conforming depth field.

    Returns:
        One entry per frame that carries rows.  A frame with no contact is
        absent, never present with a count of zero.

    Raises:
        ValueError: If *table* is not schema-conforming.
    """
    _validate_field_table(table, what="depth field table")
    frames, counts = np.unique(_frame_index_array(table), return_counts=True)
    return {int(frame): int(count) for frame, count in zip(frames, counts)}


def csv_contact_point_counts_by_frame(csv_path: Path) -> Dict[int, int]:
    """Return ``frame_index -> number of parsed contact points`` for a stage CSV.

    Parsing goes through ``parse_contact_points``, the same function the stages
    use, so this reports what the stage actually saw — including the points it
    silently discarded.  A stricter parser here would report a count no stage
    ever worked with and would make the agreement check meaningless.

    Only ``frame_index`` and ``contact_points`` are read.  Everything else in
    the CSV is the stages' business, not this module's.

    Args:
        csv_path: A CSV written by a postprocessing stage.

    Returns:
        One entry per frame whose cell parsed to at least one point.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
        ValueError: If either column is absent, if a contact-bearing row's
            ``frame_index`` is not a whole number, or if two contact-bearing
            rows share one ``frame_index``.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Stage CSV not found: {csv_path}")

    try:
        frame = pd.read_csv(
            csv_path, usecols=[FRAME_INDEX_COLUMN, CONTACT_POINTS_COLUMN]
        )
    except ValueError as error:
        raise ValueError(
            f"{csv_path} does not expose both {FRAME_INDEX_COLUMN!r} and "
            f"{CONTACT_POINTS_COLUMN!r}; the per-frame contact point count "
            f"cannot be taken from it ({error})."
        ) from error

    counts: Dict[int, int] = {}
    for row_position, (raw_frame, cell) in enumerate(
        zip(frame[FRAME_INDEX_COLUMN], frame[CONTACT_POINTS_COLUMN])
    ):
        points = parse_contact_points(cell)
        if not points:
            continue

        as_float = float(raw_frame)
        if not np.isfinite(as_float) or as_float != int(as_float):
            raise ValueError(
                f"Row {row_position} of {csv_path} carries contact points but "
                f"its {FRAME_INDEX_COLUMN} is {raw_frame!r}, which is not a "
                "whole number. Contact rows must be anchored to a Kinect frame."
            )
        key = int(as_float)
        if key in counts:
            raise ValueError(
                f"{csv_path} has two contact-bearing rows with "
                f"{FRAME_INDEX_COLUMN}={key} (second at row {row_position}); the "
                "per-frame count would be ambiguous."
            )
        counts[key] = len(points)

    return counts


def assert_row_counts_agree_with_csv(table: pd.DataFrame, csv_path: Path) -> None:
    """Raise unless the depth field and the CSV agree, frame by frame.

    This is the propagation's central safety net.  ``parse_contact_points``
    discards a malformed point without a word, so one corrupt
    ``contact_points`` cell shortens the CSV's point list while the parquet
    keeps all its rows; every index mapping computed downstream would then be
    applied to the wrong rows, and nothing would complain.  Running this after
    every stage localises the break to the stage that caused it.

    Both artifacts are written by the same stage, so the check is between a
    table already in memory and the CSV that stage just wrote — never between
    two stages.

    Args:
        table: The depth field the stage produced.
        csv_path: The CSV **the same stage** produced.

    Returns:
        ``None``.  Agreement is the silent outcome.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
        ValueError: If *table* is not schema-conforming, if the CSV cannot be
            read for counts, or if any frame's counts differ — naming the
            lowest-numbered offending frame and both counts.
    """
    field_counts = field_row_counts_by_frame(table)
    csv_counts = csv_contact_point_counts_by_frame(csv_path)

    offenders = [
        frame
        for frame in sorted(set(field_counts) | set(csv_counts))
        if field_counts.get(frame, 0) != csv_counts.get(frame, 0)
    ]
    if not offenders:
        return

    first = offenders[0]
    raise ValueError(
        f"Row-count disagreement at frame {first}: the depth field holds "
        f"{field_counts.get(first, 0)} row(s) but {csv_path} parses to "
        f"{csv_counts.get(first, 0)} contact point(s) for that frame. "
        f"{len(offenders)} frame(s) disagree in total; this is the lowest. The "
        "two artifacts describe the same contact vertices, so a difference "
        "means one of them lost points — most likely a malformed "
        "contact_points cell, which parse_contact_points drops silently."
    )
