"""Postprocessing step 4: Project contact points onto PCA-calibrated forearm surface.

For each session CSV produced by stage 2 (PCA-calibrated), snaps every contact
point to the nearest vertex (by XY distance only) on the session's
PCA-calibrated forearm PLY using a KD-tree built from (x, y) coordinates.
This guarantees that all output contact points lie exactly on the reference
surface at the correct lateral position, regardless of depth offset.

``contact_location_x/y/z`` is recomputed as the mean of the projected points.
Rows with empty contact_points pass through unchanged.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import open3d as o3d
import pandas as pd
from scipy.spatial import KDTree

from preprocessing.forearm_extraction.registration.csv_spatial_transformer import (
    parse_contact_points,
    serialize_contact_points,
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


def project_contacts_onto_forearm(
    input_files: List[Path],
    forearm_ply_path: Path,
    output_dir: Path,
    projection_stats_path: Path,
    *,
    force_processing: bool = False,
) -> List[Path]:
    """Project contact points in session CSVs onto the PCA-calibrated forearm surface.

    Builds a KD-tree from *forearm_ply_path* once, then processes each CSV in
    *input_files*, snapping every contact point to the nearest forearm vertex.

    Args:
        input_files: PCA-calibrated session CSVs (stage 2 outputs).
        forearm_ply_path: Path to the PCA-calibrated forearm PLY (stage 3 output).
            If the path does not exist, the stage is skipped with a warning.
        output_dir: Destination directory (``blocks_contact_projected/``).
        force_processing: Re-run even if outputs are up-to-date.
        projection_stats_path: Path for a combined CSV summarising per-session
            projection distances (mm).  Treated as an additional output: if it
            is missing every session is reprocessed so the stats can be fully
            populated.  Columns: ``session``, ``n_points_projected``,
            ``mean_displacement_mm``, ``median_displacement_mm``,
            ``p95_displacement_mm``, ``max_displacement_mm``.

    Returns:
        List of output CSV paths that were written (or already up-to-date).
    """
    if not forearm_ply_path.exists():
        logger.warning(
            "Forearm PLY not found (%s). Skipping contact-point projection.",
            forearm_ply_path,
        )
        return []

    vertices = _load_forearm_vertices(forearm_ply_path)
    if len(vertices) == 0:
        logger.warning(
            "Forearm PLY has no points: %s. Skipping contact-point projection.",
            forearm_ply_path.name,
        )
        return []

    kdtree = KDTree(vertices[:, :2])
    output_paths: List[Path] = []
    stats_rows: List[dict] = []

    for input_csv in input_files:
        output_csv = output_dir / input_csv.name

        if not should_process_task(
            input_paths=[input_csv, forearm_ply_path],
            output_paths=[output_csv, projection_stats_path],
            force=force_processing,
        ):
            logger.info("Contact projection up-to-date. Skipping: %s", output_csv.name)
            output_paths.append(output_csv)
            continue
        clean_task_outputs(output_csv)
        projection = _project_single_csv(input_csv, output_csv, kdtree, vertices)
        distances = projection.distances
        logger.info("Projected contact points: %s", output_csv.name)
        output_paths.append(output_csv)

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

    return output_paths
