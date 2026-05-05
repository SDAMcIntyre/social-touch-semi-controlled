from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd

from preprocessing.forearm_extraction.registration.csv_spatial_transformer import (
    parse_contact_points,
    serialize_contact_points,
)


def deduplicate_xy(points: np.ndarray, epsilon: float = 0.1) -> tuple[np.ndarray, int]:
    """Deduplicate points by (x, y) position, keeping the one with lowest z per bin.

    Args:
        points: Array of shape (N, 3) in mm, float64.
        epsilon: Bin size in mm. Two points within this distance in (x, y) are
            considered duplicates.

    Returns:
        Tuple of (deduped_points, n_removed). deduped_points contains rows from
        the input array in their original relative order.
    """
    if len(points) == 0:
        return points, 0

    bin_xy = np.round(points[:, :2] / epsilon).astype(np.int64)

    bin_to_best_index: dict[tuple, int] = {}
    for i in range(len(points)):
        key = (int(bin_xy[i, 0]), int(bin_xy[i, 1]))
        if key not in bin_to_best_index:
            bin_to_best_index[key] = i
        else:
            existing = bin_to_best_index[key]
            if points[i, 2] < points[existing, 2]:
                bin_to_best_index[key] = i

    kept_indices = sorted(bin_to_best_index.values())
    deduped = points[kept_indices]
    n_removed = len(points) - len(deduped)

    assert len(deduped) + n_removed == len(points), (
        f"Deduplication invariant violated: {len(deduped)} + {n_removed} != {len(points)}"
    )

    return deduped, n_removed


def deduplicate_forearm_ply(input_ply: Path, output_ply: Path, epsilon: float = 0.1) -> dict:
    """Deduplicate a forearm PLY point cloud by (x, y) position.

    Loads the PLY, removes (x, y) duplicates keeping the lowest z (outermost
    surface), and writes the result to output_ply.

    Args:
        input_ply: Path to the input PLY file.
        output_ply: Path to write the deduplicated PLY.
        epsilon: Bin size in mm for (x, y) deduplication.

    Returns:
        Dict with keys n_original, n_deduped, n_removed.
    """
    pcd = o3d.io.read_point_cloud(str(input_ply))
    vertices = np.asarray(pcd.points, dtype=np.float64)

    if len(vertices) == 0:
        output_ply.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(output_ply), pcd)
        return {"n_original": 0, "n_deduped": 0, "n_removed": 0}

    deduped, n_removed = deduplicate_xy(vertices, epsilon)

    deduped_pcd = o3d.geometry.PointCloud()
    deduped_pcd.points = o3d.utility.Vector3dVector(deduped)

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_ply), deduped_pcd)

    return {
        "n_original": len(vertices),
        "n_deduped": len(deduped),
        "n_removed": n_removed,
    }


def deduplicate_contact_points_csv(
    input_csv: Path, output_csv: Path, epsilon: float = 0.1
) -> dict:
    """Deduplicate contact_points in each row of a session CSV by (x, y) position.

    For each row with non-empty contact_points, removes (x, y) duplicates keeping
    the lowest z per bin, then recomputes contact_location_x/y/z as the mean of
    the remaining points. Rows with empty or null contact_points pass through
    unchanged.

    Args:
        input_csv: Path to the input session CSV.
        output_csv: Path to write the deduplicated CSV.
        epsilon: Bin size in mm for (x, y) deduplication.

    Returns:
        Dict with keys n_rows_processed, total_points_before, total_points_after.
    """
    df = pd.read_csv(input_csv)

    n_rows_processed = 0
    total_points_before = 0
    total_points_after = 0

    new_contact_points = []
    new_location_x = []
    new_location_y = []
    new_location_z = []

    for _, row in df.iterrows():
        raw = row.get("contact_points", None)
        points = parse_contact_points(raw) if raw is not None and str(raw).strip() else []

        if not points:
            new_contact_points.append(raw)
            new_location_x.append(row.get("contact_location_x"))
            new_location_y.append(row.get("contact_location_y"))
            new_location_z.append(row.get("contact_location_z"))
            continue

        pts_array = np.array(points, dtype=np.float64)
        total_points_before += len(pts_array)

        deduped_array, _ = deduplicate_xy(pts_array, epsilon)
        total_points_after += len(deduped_array)
        n_rows_processed += 1

        deduped_tuples = [
            (float(p[0]), float(p[1]), float(p[2])) for p in deduped_array
        ]
        new_contact_points.append(serialize_contact_points(deduped_tuples))

        mean_pt = deduped_array.mean(axis=0)
        new_location_x.append(mean_pt[0])
        new_location_y.append(mean_pt[1])
        new_location_z.append(mean_pt[2])

    df["contact_points"] = new_contact_points
    df["contact_location_x"] = new_location_x
    df["contact_location_y"] = new_location_y
    df["contact_location_z"] = new_location_z

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    return {
        "n_rows_processed": n_rows_processed,
        "total_points_before": total_points_before,
        "total_points_after": total_points_after,
    }
