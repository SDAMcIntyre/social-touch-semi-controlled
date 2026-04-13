"""Postprocessing step 5: Center coordinate origin on the receptive field.

For each session, estimates the receptive field (RF) center from the
projected block CSVs using selectivity-weighted DBSCAN clustering, then
translates all spatial columns and the forearm PLY by the RF center offset
so that the coordinate origin sits at the RF center.

If RF estimation fails (no valid DBSCAN cluster), a warning is logged and
input data is passed through unchanged.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import pandas as pd

from analysis.receptive_field_mapping.rf_mapping_config import (
    GroupedSpatialData,
    SelectivityDBSCANConfig,
)
from analysis.receptive_field_mapping.rf_mapping_engine import RFMappingEngine
from preprocessing.forearm_extraction.registration.csv_spatial_transformer import (
    parse_contact_points,
    transform_spatial_columns_in_place,
)
from utils.should_process_task import clean_task_outputs, should_process_task

logger = logging.getLogger(__name__)


def _compute_rf_center(
    block_csvs: List[Path],
    config: SelectivityDBSCANConfig,
) -> Tuple[Optional[np.ndarray], dict]:
    """Estimate the RF center from block CSVs via selectivity-weighted DBSCAN.

    Reads all block CSVs, accumulates spike and total contact-point counts
    per unique 3D position, computes per-point selectivity (spike/total),
    and clusters the high-selectivity points with DBSCAN.  The RF center is
    the selectivity-weighted centroid of the dominant (largest) cluster.

    Args:
        block_csvs: Block-level CSVs from ``blocks_contact_projected/``.
        config: Selectivity threshold and DBSCAN parameters.

    Returns:
        Tuple of:
        - RF center as a ``(3,)`` float64 array, or ``None`` on failure.
        - Metadata dict suitable for JSON serialisation.
    """
    gsd = GroupedSpatialData(group_label="all")
    required_cols = {"single_touch_id", "Nerve_spike", "contact_points"}

    for csv_path in block_csvs:
        if not csv_path.exists():
            logger.warning("Block CSV not found, skipping: %s", csv_path)
            continue

        try:
            available = set(pd.read_csv(csv_path, nrows=0).columns)
            missing = required_cols - available
            if missing:
                logger.warning(
                    "Skipping %s: missing columns %s", csv_path.name, missing
                )
                continue
            df = pd.read_csv(csv_path, usecols=list(required_cols))
        except Exception:
            logger.exception("Error reading %s", csv_path.name)
            continue

        touch_ids = df["single_touch_id"].to_numpy()
        spikes = df["Nerve_spike"].to_numpy()
        raw_points = df["contact_points"]

        for idx in range(len(df)):
            if touch_ids[idx] == 0:
                continue
            pts = parse_contact_points(raw_points.iat[idx])
            if not pts:
                continue
            gsd.total_counts.update(pts)
            if spikes[idx] == 1:
                gsd.spike_counts.update(pts)

    if not gsd.total_counts:
        logger.warning(
            "No contact points found in block CSVs — cannot estimate RF center."
        )
        return None, {"status": "no_contact_points", "rf_center": None}

    selectivity = RFMappingEngine.compute_selectivity(gsd.spike_counts, gsd.total_counts)
    rf_result = RFMappingEngine.cluster_receptive_field(
        selectivity_scores=selectivity,
        config=config,
        group_label="all",
    )

    if not rf_result.clusters:
        logger.warning(
            "DBSCAN found no valid clusters (evaluated %d points, %d above threshold).",
            rf_result.total_points_evaluated,
            rf_result.points_above_threshold,
        )
        return None, {
            "status": "no_cluster_found",
            "rf_center": None,
            "total_points_evaluated": rf_result.total_points_evaluated,
            "points_above_threshold": rf_result.points_above_threshold,
        }

    dominant = max(rf_result.clusters, key=lambda c: c.point_count)
    rf_center = np.average(
        dominant.points, weights=dominant.selectivity_scores, axis=0
    )

    return rf_center.astype(np.float64), {
        "status": "ok",
        "rf_center": rf_center.tolist(),
        "dominant_cluster_id": int(dominant.cluster_id),
        "dominant_cluster_point_count": int(dominant.point_count),
        "dominant_cluster_mean_selectivity": float(dominant.mean_selectivity),
        "n_clusters_found": len(rf_result.clusters),
        "total_points_evaluated": rf_result.total_points_evaluated,
        "points_above_threshold": rf_result.points_above_threshold,
    }


def _build_translation_matrix(offset: np.ndarray) -> np.ndarray:
    """Build a 4x4 matrix that translates points by ``-offset``.

    Applying this matrix moves the RF center (at *offset*) to the origin.
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = -offset
    return T


def _translate_single_csv(
    input_csv: Path,
    output_csv: Path,
    translation_matrix: np.ndarray,
) -> None:
    """Apply a 4x4 translation to all spatial columns in one block CSV.

    Args:
        input_csv: Source block CSV.
        output_csv: Destination CSV (parent created if needed).
        translation_matrix: 4x4 translation; identity is a no-op.
    """
    df = pd.read_csv(input_csv)
    transform_spatial_columns_in_place(df, translation_matrix)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def _translate_forearm_ply(
    input_ply: Path,
    output_ply: Path,
    offset: np.ndarray,
) -> None:
    """Translate a forearm PLY by ``-offset`` (RF center → origin).

    Args:
        input_ply: Source PCA-calibrated forearm PLY.
        output_ply: Destination PLY (parent created if needed).
        offset: RF center in mm; vertices are shifted by ``-offset``.
    """
    pcd = o3d.io.read_point_cloud(str(input_ply))
    pcd.translate(-offset, relative=True)
    pcd.points = o3d.utility.Vector3dVector(np.round(np.asarray(pcd.points), 1))
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_ply), pcd)


def center_on_receptive_field(
    input_files: List[Path],
    forearm_ply_path: Path,
    output_dir: Path,
    forearm_output_dir: Path,
    rf_origin_path: Path,
    *,
    force_processing: bool = False,
) -> List[Path]:
    """Center all spatial data on the receptive field coordinate origin.

    Estimates the RF center from *input_files* (block CSVs from
    ``blocks_contact_projected/``) using selectivity-weighted DBSCAN, then
    translates all spatial columns in each CSV and the forearm PLY so that
    the RF center becomes ``(0, 0, 0)``.

    When RF estimation fails (no cluster above threshold), a warning is logged,
    ``rf_origin_path`` is written with ``status: "no_cluster_found"``, and
    the input files are copied to *output_dir* unchanged so downstream tasks
    can still run.

    Args:
        input_files: Block CSVs from ``blocks_contact_projected/``.
        forearm_ply_path: PCA-calibrated forearm PLY (stage 3 output).
        output_dir: Output directory for RF-centered block CSVs
            (``blocks_rf_centered/``).
        forearm_output_dir: Output directory for the translated forearm PLY
            (``forearm_rf_centered/``).
        rf_origin_path: Destination for the per-session
            ``rf_center_origin.json`` metadata file.
        force_processing: Re-run even if outputs are up-to-date.

    Returns:
        List of output CSV paths (written or already up-to-date).
    """
    if not input_files:
        logger.warning("center_on_receptive_field: no input files provided.")
        return []

    output_csvs = [output_dir / f.name for f in input_files]
    forearm_output_ply = (
        forearm_output_dir / forearm_ply_path.name
        if forearm_ply_path.exists()
        else None
    )

    all_outputs: List[Path] = list(output_csvs) + [rf_origin_path]
    if forearm_output_ply:
        all_outputs.append(forearm_output_ply)

    input_paths: List[Path] = list(input_files)
    if forearm_ply_path.exists():
        input_paths.append(forearm_ply_path)

    if not should_process_task(
        input_paths=input_paths,
        output_paths=all_outputs,
        force=force_processing,
    ):
        logger.info("RF centering up-to-date. Skipping session.")
        return output_csvs

    clean_task_outputs(all_outputs)

    # --- Compute RF center ---
    dbscan_config = SelectivityDBSCANConfig()
    rf_center, metadata = _compute_rf_center(input_files, dbscan_config)

    if rf_center is None:
        logger.warning(
            "RF center estimation failed (status: %s). Passing data through unchanged.",
            metadata.get("status"),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for src, dst in zip(input_files, output_csvs):
            shutil.copy2(src, dst)
        if forearm_output_ply and forearm_ply_path.exists():
            forearm_output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(forearm_ply_path, forearm_output_ply)
        rf_origin_path.parent.mkdir(parents=True, exist_ok=True)
        rf_origin_path.write_text(json.dumps(metadata, indent=2))
        return output_csvs

    logger.info(
        "RF center estimated at [%.2f, %.2f, %.2f] mm "
        "(cluster %d, %d points, mean selectivity %.3f).",
        rf_center[0], rf_center[1], rf_center[2],
        metadata["dominant_cluster_id"],
        metadata["dominant_cluster_point_count"],
        metadata["dominant_cluster_mean_selectivity"],
    )

    T = _build_translation_matrix(rf_center)

    # --- Translate block CSVs ---
    output_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in zip(input_files, output_csvs):
        _translate_single_csv(src, dst, T)
        logger.info("RF-centered block CSV: %s", dst.name)

    # --- Translate forearm PLY ---
    if forearm_ply_path.exists() and forearm_output_ply:
        _translate_forearm_ply(forearm_ply_path, forearm_output_ply, rf_center)
        logger.info("RF-centered forearm PLY: %s", forearm_output_ply.name)
    else:
        logger.warning(
            "Forearm PLY not found (%s); skipping PLY translation.", forearm_ply_path
        )

    # --- Write RF origin metadata ---
    rf_origin_path.parent.mkdir(parents=True, exist_ok=True)
    rf_origin_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Wrote RF origin metadata: %s", rf_origin_path.name)

    return output_csvs
