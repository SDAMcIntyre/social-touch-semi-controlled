"""Postprocessing step 4: Project contact points onto PCA-calibrated forearm surface.

For each session CSV produced by stage 2 (PCA-calibrated), snaps every contact
point to the nearest vertex (by XY distance only) on the session's
PCA-calibrated forearm PLY using a KD-tree built from (x, y) coordinates.
This guarantees that all output contact points lie exactly on the reference
surface at the correct lateral position, regardless of depth offset.

``contact_location_x/y/z`` is recomputed as the mean of the projected points.
Rows with empty contact_points pass through unchanged.

The per-vertex contact depth field
----------------------------------
Projection is the stage that gives the depth field its ``vertex_id``.  Each
sidecar row is moved onto the same forearm vertex the CSV's point was moved
onto, and the index of that vertex is written to the file.  From here on the
field is **space-independent**: the three forearm PLYs downstream are
transformed in place, in file order, with no reordering or count change, so
vertex *i* is the same physical vertex in all of them and a consumer can resolve
coordinates in whichever space it wants.

Two rules govern how it is done:

1. **The CSV's own KD-tree answer, never a second query.**  The CSV's points
   came back from a ``%.1f`` round trip; the parquet's did not.  Re-querying the
   tree with the parquet's coordinates would pick a different vertex wherever
   two candidates are near-equidistant, and would do so silently.
   ``_project_single_csv`` therefore surfaces its per-frame indices and the
   sidecar consumes them.
2. **Depth is not re-measured.**  Projection *moves* a point; it does not
   change how far the hand penetrated at it.  ``signed_depth_mm`` is carried
   through bitwise and asserted to be.

A ``vertex_id`` is meaningless without the identity of the mesh it indexes, so
the reference-PLY provenance triple — ``reference_ply``,
``reference_ply_vertex_count`` and ``dedup_epsilon`` — is stamped at the same
time, read from the sidecar the dedup stage wrote beside the PLY.  The epsilon
is taken from that file rather than re-derived, because under ``monitor: true``
it was chosen interactively and exists nowhere else.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import KDTree

from preprocessing.forearm_extraction.registration.csv_spatial_transformer import (
    parse_contact_points,
    serialize_contact_points,
)
from preprocessing.motion_analysis.tactile_quantification.io.contact_depth_field_io import (
    SCHEMA_VERSION,
    read_contact_depth_field,
    write_contact_depth_field_table,
)
from postprocessing.depth_field_stage_io import (
    DEPTH_COLUMN,
    PIPELINE_STAGE_POSTPROCESSING,
    apply_vertex_addressing_to_field,
    assert_row_counts_agree_with_csv,
    depth_field_path_for_csv,
)
from postprocessing.forearm_dedup_metadata import (
    ForearmDedupMetadata,
    read_forearm_dedup_metadata,
)
from utils.should_process_task import should_process_task, clean_task_outputs

logger = logging.getLogger(__name__)

#: CSV column that identifies the Kinect frame a row belongs to. The projection
#: mapping is keyed by this value, never by row position: the merged CSV is
#: upsampled to the nerve rate, so a row index means nothing outside one
#: particular file, while ``frame_index`` is what every other stage aligns on.
#: (Deliberately duplicated from ``deduplicate_xy_points``: stage scripts must
#: not import from one another.)
FRAME_INDEX_COLUMN = "frame_index"


@dataclass(frozen=True)
class ProjectionResult:
    """What one CSV's projection produced beyond the CSV itself.

    Attributes:
        distances: Flat array of per-point 3D projection distances, one entry
            per contact point across all rows, in row order. Empty if the CSV
            had no contact points.
        vertex_indices: ``frame_index`` → 1-D ``intp`` array of the forearm
            vertex each of that frame's contact points snapped to, in the order
            the points appear in the cell. These integers index the forearm PLY
            vertex array in file order, so they are the vertex identity a
            per-point sidecar must be re-addressed with — re-querying the
            KD-tree against a sidecar's own coordinates would pick different
            vertices wherever two candidates are near-equidistant, and would do
            so silently.
    """

    distances: np.ndarray
    vertex_indices: Dict[int, np.ndarray]


def _frame_index_key(value, *, csv_path: Path, row_position: int) -> int:
    """Coerce a ``frame_index`` cell to the integer key the mapping is stored under.

    The column arrives as float64 because rows without a Kinect frame hold NaN,
    so the value must be checked, not merely cast: a NaN or a fractional index
    would otherwise become a silently wrong dict key.
    """
    as_float = float(value)
    if not np.isfinite(as_float) or as_float != int(as_float):
        raise ValueError(
            f"Row {row_position} of {csv_path} carries contact points but its "
            f"{FRAME_INDEX_COLUMN} is {value!r}, which is not a whole number. "
            f"Contact rows must be anchored to a Kinect frame."
        )
    return int(as_float)


def _load_forearm_vertices(ply_path: Path) -> np.ndarray:
    """Load a PLY point cloud and return its vertices as an ``(N, 3)`` float64 array.

    Args:
        ply_path: Path to the PCA-calibrated forearm PLY file.

    Returns:
        Array of shape ``(N, 3)``.  May be empty (shape ``(0, 3)``) if the PLY
        contains no points.
    """
    pcd = o3d.io.read_point_cloud(str(ply_path))
    return np.asarray(pcd.points, dtype=np.float64)


def _project_single_csv(
    input_csv: Path,
    output_csv: Path,
    kdtree: KDTree,
    vertices: np.ndarray,
) -> ProjectionResult:
    """Project the contact points in one CSV onto the forearm surface.

    For every non-empty ``contact_points`` cell, each (x, y, z) point is
    independently snapped to its nearest forearm vertex by XY distance only
    via a KD-tree query on (x, y) coordinates.  The projected point receives
    the full (x, y, z) of the matched vertex.
    ``contact_location_x/y/z`` is updated to the mean of the projected points.
    Rows with no contact points are written through unchanged.

    All M contact points in each row are guaranteed to appear in the output —
    no uniqueness constraint is enforced, so two nearby points may snap to the
    same vertex.  The ``assert`` below documents that guarantee and will raise
    immediately if it is ever violated (fail-fast).

    Args:
        input_csv: Source CSV (session data to be projected).
        output_csv: Destination CSV.
        kdtree: KD-tree built from the forearm PLY vertices (XY only).
        vertices: The ``(N, 3)`` vertex array used to build *kdtree*.

    Returns:
        A :class:`ProjectionResult` carrying the per-point projection distances
        and, per ``frame_index``, the vertex indices the points snapped to.

    Raises:
        ValueError: If the ``frame_index`` column is absent, if a contact-bearing
            row has a non-integral ``frame_index``, or if two contact-bearing
            rows share one ``frame_index`` (which would make the mapping
            ambiguous).
    """
    df = pd.read_csv(input_csv)

    if FRAME_INDEX_COLUMN not in df.columns:
        raise ValueError(
            f"{input_csv} has no {FRAME_INDEX_COLUMN!r} column; the per-frame "
            f"projection mapping cannot be keyed."
        )

    projected_contact_points = []
    location_x, location_y, location_z = [], [], []
    all_distances: List[np.ndarray] = []
    vertex_indices: Dict[int, np.ndarray] = {}

    for row_position, (_, row) in enumerate(df.iterrows()):
        raw = row.get("contact_points", "[]")
        points = parse_contact_points(raw)

        if not points:
            projected_contact_points.append(raw)
            location_x.append(row.get("contact_location_x"))
            location_y.append(row.get("contact_location_y"))
            location_z.append(row.get("contact_location_z"))
            continue

        query = np.array(points, dtype=np.float64)  # (M, 3)
        M = len(points)
        _, indices = kdtree.query(query[:, :2])
        projected = vertices[indices]
        assert len(projected) == M, (
            f"Projection dropped {M - len(projected)} of {M} contact points — "
            "impossible with per-point NN."
        )

        frame_index = _frame_index_key(
            row[FRAME_INDEX_COLUMN], csv_path=input_csv, row_position=row_position
        )
        if frame_index in vertex_indices:
            raise ValueError(
                f"{input_csv} has two contact-bearing rows with "
                f"{FRAME_INDEX_COLUMN}={frame_index} (second at row {row_position}); "
                f"the per-frame projection mapping would be ambiguous."
            )
        vertex_indices[frame_index] = np.asarray(indices, dtype=np.intp)

        displacements_3d = np.linalg.norm(query - projected, axis=1)
        all_distances.append(displacements_3d)

        projected_tuples = [(float(p[0]), float(p[1]), float(p[2])) for p in projected]
        projected_contact_points.append(serialize_contact_points(projected_tuples))

        mean_pt = projected.mean(axis=0)
        location_x.append(mean_pt[0])
        location_y.append(mean_pt[1])
        location_z.append(mean_pt[2])

    df["contact_points"] = projected_contact_points
    df["contact_location_x"] = location_x
    df["contact_location_y"] = location_y
    df["contact_location_z"] = location_z

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    return ProjectionResult(
        distances=(
            np.concatenate(all_distances) if all_distances else np.empty(0, dtype=np.float64)
        ),
        vertex_indices=vertex_indices,
    )



def _resolve_input_parquets(
    input_files: Sequence[Path], input_parquets: Optional[Sequence[Path]]
) -> List[Path]:
    """Return the sidecar path for every input CSV, proven to exist.

    Args:
        input_files: The block CSVs this stage will process.
        input_parquets: Explicit sidecar paths, index-aligned with
            *input_files*.  When ``None`` they are derived with
            ``depth_field_path_for_csv`` — the sidecar sits beside its CSV, and
            keeping the rule in one place stops the stage and its caller
            disagreeing about which file belongs to which block.

    Returns:
        One existing sidecar path per input CSV, in the same order.

    Raises:
        ValueError: If *input_parquets* is supplied with a different length.
        FileNotFoundError: If any sidecar is absent.  The depth field is a hard
            input of postprocessing; skipping a block without one would leave a
            gap nothing downstream would notice.
    """
    if input_parquets is None:
        resolved = [depth_field_path_for_csv(path) for path in input_files]
    else:
        resolved = [Path(path) for path in input_parquets]
        if len(resolved) != len(input_files):
            raise ValueError(
                f"input_parquets has {len(resolved)} entr(ies) but input_files "
                f"has {len(input_files)}. The two lists are matched by position, "
                "one sidecar per block CSV, so a length mismatch would pair a "
                "block with another block's depth field."
            )

    missing = [path for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Contact depth field sidecar(s) missing: "
            f"{[str(path) for path in missing]}. The per-vertex depth field is a "
            "required input of this stage — it is written beside the block CSV "
            "by the deduplicate_xy stage. Re-run that stage for the affected "
            "block(s) rather than projecting without it."
        )
    return resolved


def _resolve_reference_provenance(
    forearm_ply_path: Path, vertices: np.ndarray
) -> ForearmDedupMetadata:
    """Read the deduped-forearm sidecar and prove it describes *this* PLY.

    A ``vertex_id`` is an index into one specific mesh.  The sidecar names the
    epsilon that produced that mesh and the vertex count it ended up with; if
    the count does not match the PLY actually loaded, the PLY on disk was
    re-deduplicated at some other epsilon and every index this stage is about to
    write would point at the wrong vertex.

    Args:
        forearm_ply_path: The deduplicated forearm PLY the KD-tree was built over.
        vertices: The vertex array loaded from it.

    Returns:
        The parsed sidecar.

    Raises:
        FileNotFoundError: If the sidecar is absent, naming the file.
        ValueError: If the sidecar is malformed, or if its recorded vertex count
            disagrees with the PLY.
    """
    provenance = read_forearm_dedup_metadata(forearm_ply_path)
    if provenance.n_vertices_deduped != len(vertices):
        raise ValueError(
            f"{provenance.path} records {provenance.n_vertices_deduped} "
            f"deduplicated vertices but {forearm_ply_path.name} holds "
            f"{len(vertices)}. The PLY and its provenance sidecar are out of "
            "step, so the epsilon that numbered these vertices is unknown and "
            "no vertex_id written against them could be validated. Re-run the "
            "deduplicate_xy task for this session."
        )
    return provenance


def _assert_depth_preserved(
    original: pd.DataFrame, projected: pd.DataFrame, *, parquet_name: str
) -> None:
    """Raise unless ``signed_depth_mm`` survived projection bit for bit.

    Projection moves a contact point onto the reference surface; it does not
    re-measure how far the hand went in.  The leaf module preserves the column
    by construction; this asserts it at the stage boundary anyway, because a
    silently re-derived depth would be indistinguishable from a measured one in
    the output file.
    """
    before = original[DEPTH_COLUMN].to_numpy()
    after = projected[DEPTH_COLUMN].to_numpy()
    if after.dtype != before.dtype or not np.array_equal(after, before):
        raise ValueError(
            f"{parquet_name}: projection altered {DEPTH_COLUMN!r} (dtype "
            f"{before.dtype} -> {after.dtype}). Projection re-addresses a "
            "penetration measurement to a canonical vertex; it never re-measures "
            "it."
        )


def _write_projected_field(
    input_parquet: Path,
    output_parquet: Path,
    output_csv: Path,
    vertex_indices: Dict[int, np.ndarray],
    vertices: np.ndarray,
    provenance: ForearmDedupMetadata,
    reference_ply_name: str,
) -> None:
    """Re-address a sidecar onto the reference forearm and stamp its provenance.

    *vertex_indices* must be the mapping the CSV's own KD-tree query produced
    for *output_csv*; the tree is never queried again here.

    ``coordinate_space`` is carried through unchanged: projection snaps points
    onto the reference surface **within** the frame they were already in.

    Args:
        input_parquet: The block's depth field as the dedup stage left it.
        output_parquet: Destination in ``blocks_projected/``.
        output_csv: The projected CSV **this stage just wrote** for the same block.
        vertex_indices: ``frame_index -> (M,) intp`` from the CSV's projection.
        vertices: The ``(V, 3)`` array the KD-tree was built from.
        provenance: The deduplicated-forearm sidecar, already checked against
            *vertices*.
        reference_ply_name: Filename of the PLY the indices address.

    Raises:
        ValueError: If the mapping does not cover exactly the frames the field
            holds, if a frame's index count differs from its row count, if
            projection disturbed ``signed_depth_mm``, or if the written field and
            CSV disagree about a frame's contact-point count.
    """
    table, source_metadata = read_contact_depth_field(input_parquet)
    addressed = apply_vertex_addressing_to_field(table, vertex_indices, vertices)
    _assert_depth_preserved(table, addressed, parquet_name=input_parquet.name)

    # The schema version is restamped, not carried: the file now has a column
    # version 1 does not know, and the writer refuses the contradiction.
    metadata: Dict[str, str] = dict(source_metadata)
    metadata["schema_version"] = SCHEMA_VERSION
    metadata["pipeline_stage"] = PIPELINE_STAGE_POSTPROCESSING
    metadata["reference_ply"] = reference_ply_name
    metadata["reference_ply_vertex_count"] = str(provenance.n_vertices_deduped)
    metadata["dedup_epsilon"] = repr(provenance.dedup_epsilon)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    write_contact_depth_field_table(addressed, output_parquet, metadata=metadata)
    assert_row_counts_agree_with_csv(addressed, output_csv)


def project_contacts_onto_forearm(
    input_files: List[Path],
    forearm_ply_path: Path,
    output_dir: Path,
    projection_stats_path: Path,
    *,
    input_parquets: Optional[Sequence[Path]] = None,
    force_processing: bool = False,
) -> Tuple[List[Path], List[Path]]:
    """Project contact points in session CSVs onto the PCA-calibrated forearm surface.

    Builds a KD-tree from *forearm_ply_path* once, then processes each CSV in
    *input_files*, snapping every contact point to the nearest forearm vertex.

    Each block's per-vertex contact depth field is re-addressed onto the same
    vertices by the same indices, and gains a ``vertex_id`` column plus the
    reference-PLY provenance that makes the index interpretable.

    Args:
        input_files: PCA-calibrated session CSVs (stage 2 outputs).
        forearm_ply_path: Path to the PCA-calibrated forearm PLY (stage 3 output).
            If the path does not exist, the stage is skipped with a warning.
        output_dir: Destination directory (``blocks_contact_projected/``).
        input_parquets: Per-block contact depth field sidecars, index-aligned
            with *input_files*.  When omitted they are derived from the CSV
            paths — the sidecar sits beside its CSV.
        force_processing: Re-run even if outputs are up-to-date.
        projection_stats_path: Path for a combined CSV summarising per-session
            projection distances (mm).  Treated as an additional output: if it
            is missing every session is reprocessed so the stats can be fully
            populated.  Columns: ``session``, ``n_points_projected``,
            ``mean_displacement_mm``, ``median_displacement_mm``,
            ``p95_displacement_mm``, ``max_displacement_mm``.

    Returns:
        ``(csv_paths, parquet_paths)``: the output CSVs written (or already
        up-to-date) and their depth field sidecars, index-aligned with each
        other.  Both are empty when the forearm PLY is missing or empty.

    Raises:
        FileNotFoundError: If a block's depth field sidecar, or the deduplicated
            forearm's provenance sidecar, is missing.
        ValueError: If *input_parquets* is misaligned with *input_files*, if the
            provenance disagrees with the PLY, if projection disturbs
            ``signed_depth_mm``, or if any written pair of artifacts disagrees
            about a frame's contact-point count.
    """
    if not forearm_ply_path.exists():
        logger.warning(
            "Forearm PLY not found (%s). Skipping contact-point projection.",
            forearm_ply_path,
        )
        return [], []

    vertices = _load_forearm_vertices(forearm_ply_path)
    if len(vertices) == 0:
        logger.warning(
            "Forearm PLY has no points: %s. Skipping contact-point projection.",
            forearm_ply_path.name,
        )
        return [], []

    resolved_parquets = _resolve_input_parquets(input_files, input_parquets)
    provenance = _resolve_reference_provenance(forearm_ply_path, vertices)

    kdtree = KDTree(vertices[:, :2])
    output_paths: List[Path] = []
    output_parquet_paths: List[Path] = []
    stats_rows: List[dict] = []

    # Idempotency stays at this stage's own boundary — the block, not the
    # session — with both artifacts on both sides of the check, so deleting
    # either one regenerates the pair and they can never drift apart.
    for input_csv, input_parquet in zip(input_files, resolved_parquets):
        output_csv = output_dir / input_csv.name
        output_parquet = output_dir / input_parquet.name

        if not should_process_task(
            input_paths=[
                input_csv,
                input_parquet,
                forearm_ply_path,
                provenance.path,
            ],
            output_paths=[output_csv, output_parquet, projection_stats_path],
            force=force_processing,
        ):
            logger.info("Contact projection up-to-date. Skipping: %s", output_csv.name)
            output_paths.append(output_csv)
            output_parquet_paths.append(output_parquet)
            continue
        clean_task_outputs([output_csv, output_parquet])
        projection = _project_single_csv(input_csv, output_csv, kdtree, vertices)
        distances = projection.distances
        logger.info("Projected contact points: %s", output_csv.name)
        output_paths.append(output_csv)

        # The CSV's own KD-tree answer, re-used verbatim. Never a second query.
        _write_projected_field(
            input_parquet,
            output_parquet,
            output_csv,
            projection.vertex_indices,
            vertices,
            provenance,
            forearm_ply_path.name,
        )
        output_parquet_paths.append(output_parquet)
        logger.info("Wrote projected depth field: %s", output_parquet.name)

        n = len(distances)
        stats_rows.append(
                {
                    "session": input_csv.name,
                    "n_points_projected": n,
                    "mean_displacement_mm": float(distances.mean()) if n else float("nan"),
                    "median_displacement_mm": float(np.median(distances)) if n else float("nan"),
                    "p95_displacement_mm": float(np.percentile(distances, 95)) if n else float("nan"),
                    "max_displacement_mm": float(distances.max()) if n else float("nan"),
                }
        )

    if stats_rows:
        projection_stats_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(stats_rows).to_csv(projection_stats_path, index=False)
        logger.info("Wrote projection stats: %s", projection_stats_path.name)

    return output_paths, output_parquet_paths
