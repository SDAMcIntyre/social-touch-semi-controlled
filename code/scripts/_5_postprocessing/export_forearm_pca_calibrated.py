"""Postprocessing step 3: Export forearm-of-reference PLY in PCA-calibrated space.

Loads the unified registered forearm PLY for the session, applies the same full
PCA transform used in stage 2, and writes the result to
``forearm_pca_calibrated/{session_id}_forearm.ply``.

Single-forearm sessions (no ``_unified_registered.ply``) fall back to the single
forearm PLY found in ``forearm_pointclouds/``.
"""
import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d

from utils.should_process_task import should_process_task, clean_task_outputs
from primary_processing import KinectConfig
from postprocessing.xyz_reference_from_gestures import PCACalibrationEngine, CalibrationResult

logger = logging.getLogger(__name__)

_CALIB_JSON_NAME = "pca-xyz_transformation-matrices.json"


def _find_forearm_ply(session_configs: List[KinectConfig]) -> Optional[Path]:
    """Return the unified registered PLY, or fall back to single forearm PLY."""
    first = session_configs[0]
    forearm_dir = first.session_processed_output_dir / "forearm_pointclouds"
    unified = forearm_dir / f"{first.session_id}_unified_registered.ply"
    if unified.exists():
        return unified
    # Single-forearm fallback: any .ply in the directory
    plies = sorted(forearm_dir.glob("*.ply"))
    if plies:
        logger.info(
            "[%s] No unified registered PLY found; using fallback: %s",
            first.session_id,
            plies[0].name,
        )
        return plies[0]
    return None


def export_forearm_pca_calibrated(
    session_configs: List[KinectConfig],
    pca_output_dir: Path,
    output_dir: Path,
    *,
    force_processing: bool = False,
) -> Optional[Path]:
    """Transform the forearm-of-reference PLY into PCA-calibrated space and save it.

    Args:
        session_configs: KinectConfig objects for the session (one per block).
        pca_output_dir: Directory produced by stage 2 (contains the calibration JSON).
        output_dir: Destination directory (``forearm_pca_calibrated/``).
        force_processing: Re-run even if output is up-to-date.

    Returns:
        Path to the output PLY, or None if the step was skipped due to missing inputs.
    """
    session_id = session_configs[0].session_id

    calib_json_path = pca_output_dir / _CALIB_JSON_NAME
    if not calib_json_path.exists():
        logger.warning(
            "[%s] PCA calibration JSON not found (%s). Skipping forearm PLY export.",
            session_id,
            calib_json_path,
        )
        return None

    forearm_ply_path = _find_forearm_ply(session_configs)
    if forearm_ply_path is None:
        logger.warning(
            "[%s] No forearm PLY found in forearm_pointclouds/. Skipping.",
            session_id,
        )
        return None

    output_path = output_dir / f"{session_id}_forearm.ply"

    if not should_process_task(
        input_paths=[forearm_ply_path, calib_json_path],
        output_paths=[output_path],
        force=force_processing,
    ):
        logger.info("[%s] Forearm PLY export up-to-date. Skipping.", session_id)
        return output_path
    clean_task_outputs(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load calibration
    with open(calib_json_path) as f:
        calib = CalibrationResult.from_dict(json.load(f))

    # Load forearm PLY
    pcd = o3d.io.read_point_cloud(str(forearm_ply_path))
    vertices = np.asarray(pcd.points)
    if len(vertices) == 0:
        logger.warning("[%s] Forearm PLY has no points: %s", session_id, forearm_ply_path.name)
        return None

    # Apply full PCA transform
    transformed = PCACalibrationEngine.apply_full_transform(vertices.copy(), calib)

    # Write output PLY
    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(np.round(transformed, 1))
    if pcd.has_colors():
        out_pcd.colors = pcd.colors
    o3d.io.write_point_cloud(str(output_path), out_pcd)

    logger.info("[%s] Wrote PCA-calibrated forearm PLY: %s", session_id, output_path.name)
    return output_path
