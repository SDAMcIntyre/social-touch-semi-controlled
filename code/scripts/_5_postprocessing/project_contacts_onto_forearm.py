"""Postprocessing step 4: Project contact points onto PCA-calibrated forearm surface.

For each session CSV produced by stage 2 (PCA-calibrated), snaps every contact
point to the nearest vertex on the session's PCA-calibrated forearm PLY using a
KD-tree.  This guarantees that all output contact points lie exactly on the
reference surface, eliminating small spatial discrepancies caused by mesh
resolution, penetration-depth variation, and registration artifacts.

``contact_location_x/y/z`` is recomputed as the mean of the projected points.
Rows with empty contact_points pass through unchanged.
"""
import logging
from pathlib import Path
from typing import List

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
) -> np.ndarray:
    """Project the contact points in one CSV onto the forearm surface.

    For every non-empty ``contact_points`` cell, each (x, y, z) point is
    independently snapped to its nearest forearm vertex via a KD-tree query.
    ``contact_location_x/y/z`` is updated to the mean of the projected points.
    Rows with no contact points are written through unchanged.

    All M contact points in each row are guaranteed to appear in the output —
    no uniqueness constraint is enforced, so two nearby points may snap to the
    same vertex.  The ``assert`` below documents that guarantee and will raise
    immediately if it is ever violated (fail-fast).

    Args:
        input_csv: Source CSV (session data to be projected).
        output_csv: Destination CSV.
        kdtree: KD-tree built from the forearm PLY vertices.
        vertices: The ``(N, 3)`` vertex array used to build *kdtree*.

    Returns:
        Flat array of per-point projection distances (one entry per contact
        point across all rows).  Empty if no contact points were present.
    """
    df = pd.read_csv(input_csv)

    projected_contact_points = []
    location_x, location_y, location_z = [], [], []
    all_distances: List[np.ndarray] = []

    for _, row in df.iterrows():
        raw = row.get("contact_points", "[]")
        points = parse_contact_points(raw)

        if not points:
            projected_contact_points.append(raw)
            location_x.append(row.get("contact_location_x"))
            location_y.append(row.get("contact_location_y"))
            location_z.append(row.get("contact_location_z"))
            continue

        # Query the single nearest forearm vertex for each contact point
        # independently.  All M points are guaranteed in the output.
        query = np.array(points, dtype=np.float64)  # (M, 3)
        M = len(points)
        distances, indices = kdtree.query(query)
        projected = vertices[indices]
        assert len(projected) == M, (
            f"Projection dropped {M - len(projected)} of {M} contact points — "
            "impossible with per-point NN."
        )

        all_distances.append(np.asarray(distances, dtype=np.float64).ravel())

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

    return np.concatenate(all_distances) if all_distances else np.empty(0, dtype=np.float64)


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

    kdtree = KDTree(vertices)
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
        distances = _project_single_csv(input_csv, output_csv, kdtree, vertices)
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
