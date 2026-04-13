"""RF camera angle picker task for the analysis workflow.

Collects scene data for all sessions with successful RF centering, then
launches a single multi-session viewer where the researcher can switch
sessions, adjust rendering settings, and save camera angles.

Called at the end of ``map_receptive_fields_clustered_flow`` after RF
cluster mapping completes.
"""

try:
    import cupy  # noqa: F401  — must precede preprocessing imports
except Exception:
    pass

import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import pandas as pd

from analysis.receptive_field_mapping.rf_mapping_engine import RFMappingEngine
from analysis.receptive_field_mapping.rf_cluster_visualizer import (
    _compute_surface_normal,
)
from preprocessing.forearm_extraction.registration.csv_spatial_transformer import (
    parse_contact_points,
)

logger = logging.getLogger(__name__)


@dataclass
class SessionSceneData:
    """Pre-loaded scene data for one session."""

    session_id: str
    forearm_points: np.ndarray
    contact_points: Optional[np.ndarray] = None
    selectivity_scores: Optional[np.ndarray] = None
    initial_normal: Optional[np.ndarray] = None
    saved_camera_params: Optional[dict] = None
    camera_params_path: Path = field(default_factory=lambda: Path())


def _compute_selectivity_overlay(
    rf_centered_files: List[Path],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute per-point selectivity from RF-centered block CSVs.

    Returns (points, scores) arrays or (None, None) when no data is found.
    """
    required_cols = {"single_touch_id", "Nerve_spike", "contact_points"}
    spike_counts: Counter = Counter()
    total_counts: Counter = Counter()

    for csv_path in rf_centered_files:
        if not csv_path.exists():
            continue
        try:
            available = set(pd.read_csv(csv_path, nrows=0).columns)
            if required_cols - available:
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
            total_counts.update(pts)
            if spikes[idx] == 1:
                spike_counts.update(pts)

    if not total_counts:
        return None, None

    selectivity: Dict[Tuple[float, float, float], float] = (
        RFMappingEngine.compute_selectivity(spike_counts, total_counts)
    )
    points = np.array(list(selectivity.keys()), dtype=np.float64)
    scores = np.array(list(selectivity.values()), dtype=np.float64)
    return points, scores


def collect_session_scene_data(
    session_output_dirs: Dict[str, Path],
) -> Dict[str, SessionSceneData]:
    """Load scene data for all sessions with valid RF centering.

    Args:
        session_output_dirs: Mapping of session_id → session output directory.

    Returns:
        Dict of session_id → SessionSceneData for sessions that are ready
        to display (valid RF centering + non-empty forearm PLY).
    """
    sessions: Dict[str, SessionSceneData] = {}

    for session_id, output_dir in session_output_dirs.items():
        # Check RF centering status
        rf_origin_path = output_dir / "rf_center_origin.json"
        if not rf_origin_path.exists():
            logger.info("[%s] No rf_center_origin.json — skipping.", session_id)
            continue
        try:
            rf_meta = json.loads(rf_origin_path.read_text())
        except Exception:
            logger.warning("[%s] Could not read RF origin file — skipping.", session_id)
            continue
        if rf_meta.get("status") != "ok":
            logger.info(
                "[%s] RF centering status is '%s' — skipping.",
                session_id, rf_meta.get("status"),
            )
            continue

        # Find forearm PLY
        forearm_dir = output_dir / "forearm_rf_centered"
        forearm_plys = sorted(forearm_dir.glob("*.ply")) if forearm_dir.exists() else []
        if not forearm_plys:
            logger.warning("[%s] No forearm PLY in %s — skipping.", session_id, forearm_dir)
            continue
        pcd = o3d.io.read_point_cloud(str(forearm_plys[-1]))
        forearm_points = np.asarray(pcd.points)
        if len(forearm_points) == 0:
            logger.warning("[%s] Forearm PLY is empty — skipping.", session_id)
            continue

        # Compute selectivity overlay from RF-centered CSVs
        rf_centered_dir = output_dir / "blocks_rf_centered"
        rf_csvs = sorted(rf_centered_dir.glob("*.csv")) if rf_centered_dir.exists() else []
        contact_points, selectivity_scores = _compute_selectivity_overlay(rf_csvs)

        # Compute initial camera normal
        initial_normal = None
        if contact_points is not None and len(contact_points) > 0:
            centroid = contact_points.mean(axis=0)
            initial_normal = _compute_surface_normal(forearm_points, centroid)

        # Load existing camera params if available
        camera_params_path = output_dir / "camera_params.json"
        saved_camera = None
        if camera_params_path.exists():
            try:
                saved_camera = json.loads(camera_params_path.read_text())
            except Exception:
                pass

        sessions[session_id] = SessionSceneData(
            session_id=session_id,
            forearm_points=forearm_points,
            contact_points=contact_points,
            selectivity_scores=selectivity_scores,
            initial_normal=initial_normal,
            saved_camera_params=saved_camera,
            camera_params_path=camera_params_path,
        )

    return sessions


def pick_rf_camera_angle_batch(
    session_output_dirs: Dict[str, Path],
    *,
    force_processing: bool = False,
) -> None:
    """Launch the multi-session camera angle picker.

    Collects scene data for all valid sessions, then opens a single viewer
    where the researcher can switch sessions, adjust rendering, and save
    camera angles.

    Args:
        session_output_dirs: Mapping of session_id → session output directory.
        force_processing: If True, show viewer even for sessions with
            existing camera_params.json.
    """
    sessions = collect_session_scene_data(session_output_dirs)

    if not force_processing:
        # Filter out sessions that already have up-to-date camera params
        sessions = {
            sid: data for sid, data in sessions.items()
            if data.saved_camera_params is None
        }

    if not sessions:
        logger.info("No sessions need camera angle picking. Skipping viewer.")
        return

    logger.info(
        "Launching camera angle picker for %d session(s): %s",
        len(sessions), ", ".join(sorted(sessions)),
    )

    from PyQt5.QtWidgets import QApplication
    from analysis.receptive_field_mapping.gui.rf_camera_angle_picker import RFCameraAnglePicker

    app = QApplication.instance()
    own_app = app is None
    if own_app:
        app = QApplication(sys.argv)

    viewer = RFCameraAnglePicker(sessions)
    viewer.show()
    app.exec_()

    # Write saved cameras to disk
    for session_id, cam_params in viewer.saved_cameras.items():
        path = sessions[session_id].camera_params_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cam_params, indent=2))
        logger.info("Saved camera parameters for %s", session_id)
