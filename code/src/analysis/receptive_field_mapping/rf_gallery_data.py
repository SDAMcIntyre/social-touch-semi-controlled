"""Gallery data loading for the RF Cluster Gallery Viewer.

Scans extraction output directories produced by run_cluster_rf_extraction()
and loads all artifacts into GalleryData / GalleryCell dataclasses.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import trimesh

from .rf_extraction_io import (
    load_cluster_session_data,
    load_neuron_cluster_touches,
    load_neuron_contacts,
    load_neuron_touches,
    load_rf_camera_rotation,
    load_sessions_metadata,
)
from .rf_data_loader import load_forearm_vertex_colors, load_forearm_vertices

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GalleryCell:
    """One renderable cell: a specific (session, cluster) pair."""

    session_id: str
    cluster_label: str
    cluster_folder: str
    is_noise: bool

    # Geometry
    forearm_mesh: Optional[trimesh.Trimesh]
    forearm_vertices: Optional[np.ndarray]
    forearm_vertex_colors: Optional[np.ndarray]
    tangent_rotation: Optional[np.ndarray]

    # Spike data
    spike_counts_df: pd.DataFrame

    # Hull contacts
    neuron_contacts_xyz: np.ndarray
    cluster_contacts_xyz: np.ndarray

    # Metadata
    neuron_touches: int
    neuron_cluster_touches: int
    cluster_description: dict
    rf_metrics: Optional[dict]



@dataclass
class GalleryData:
    """All data for one combo/clusterer pair."""

    combo_name: str
    clusterer_name: str
    output_base_dir: Path = field(default_factory=lambda: Path("."))
    cells: Dict[Tuple[str, str], GalleryCell] = field(default_factory=dict)
    session_ids: List[str] = field(default_factory=list)
    cluster_labels: List[str] = field(default_factory=list)
    gesture_type: Optional[str] = None


# ---------------------------------------------------------------------------
# Cluster-label helpers
# ---------------------------------------------------------------------------

def _is_noise_label(cluster_label: str) -> bool:
    """Return True if the label represents a noise cluster (-1 or 'noise')."""
    if cluster_label == "noise":
        return True
    try:
        return int(cluster_label) < 0
    except (ValueError, TypeError):
        return False


def _cluster_folder_to_label(folder_name: str) -> str:
    """Derive the cluster label string from a cluster folder name.

    Examples:
      'cluster_00' → '0'
      'cluster_05' → '5'
      'cluster_noise' → 'noise'
      'cluster_-1' → '-1'
    """
    suffix = folder_name[len("cluster_"):]
    if suffix == "noise":
        return "noise"
    try:
        return str(int(suffix))
    except ValueError:
        return suffix


def _sort_key_for_cluster_label(label: str) -> Tuple[int, int]:
    """Sort key: numeric labels first (ascending), noise cluster last."""
    if _is_noise_label(label):
        return (1, 0)
    try:
        return (0, int(label))
    except (ValueError, TypeError):
        return (0, 0)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_gallery_data(
    output_dir: Path,
    combo_name: str,
    clusterer_name: str,
    gesture_type: str | None = None,
) -> GalleryData:
    """Scan extraction output for one combo/clusterer pair and load all artifacts.

    Parameters
    ----------
    output_dir:
        Root of the RF cluster artifact directory (same as the output_dir
        passed to run_cluster_rf_extraction()).
    combo_name:
        Feature combination name (e.g. 'pressure_velocity_mean').
    clusterer_name:
        Clustering profile name (e.g. 'kmeans_k5').
    gesture_type:
        When the cluster group used ``per_type_clustering``, pass the gesture type
        (``"tap"``, ``"stroke_proximal"``, ``"stroke_distal"``) to load the correct
        per-type artifact tree.  ``None`` loads the flat artifact tree.

    Returns
    -------
    GalleryData with all loadable cells populated.

    Raises
    ------
    FileNotFoundError
        If the base output directory or extraction_summary.json is missing.
    ValueError
        On any unexpected artifact corruption not covered by the three
        explicit edge-case policies (missing PLY, empty cluster, missing
        session spike data).
    """
    base_output = output_dir / combo_name / clusterer_name
    if gesture_type is not None:
        base_output = base_output / gesture_type

    if not base_output.exists():
        raise FileNotFoundError(
            f"load_gallery_data: base output directory does not exist: {base_output}"
        )

    extraction_json = base_output / "extraction_summary.json"
    if not extraction_json.exists():
        raise FileNotFoundError(
            f"load_gallery_data: extraction_summary.json missing — "
            f"run extraction first: {extraction_json}"
        )

    neuron_touches_map = load_neuron_touches(base_output)
    sessions_metadata = load_sessions_metadata(base_output)

    gallery = GalleryData(
        combo_name=combo_name,
        clusterer_name=clusterer_name,
        output_base_dir=base_output,
        gesture_type=gesture_type,
    )

    cluster_dirs = sorted(base_output.glob("cluster_*"))
    if not cluster_dirs:
        raise ValueError(
            f"load_gallery_data: no cluster_* directories found in {base_output}"
        )

    all_session_ids: set = set()
    cluster_labels_found: List[str] = []

    session_tangent_rotations: Dict[str, Optional[np.ndarray]] = {}

    for cluster_dir in cluster_dirs:
        cluster_folder = cluster_dir.name
        cluster_label = _cluster_folder_to_label(cluster_folder)
        is_noise = _is_noise_label(cluster_label)

        sessions_dir = cluster_dir / "sessions"
        if not sessions_dir.exists() or not any(sessions_dir.iterdir()):
            logger.warning(
                "load_gallery_data: cluster '%s' has no session subdirs — skipping.",
                cluster_folder,
            )
            continue

        cluster_description = _load_cluster_description(cluster_dir)
        neuron_cluster_touches_map = _load_neuron_cluster_touches_safe(cluster_dir, cluster_folder)
        rf_metrics = _load_rf_metrics(cluster_dir)

        session_subdirs = sorted(sessions_dir.iterdir())
        cluster_has_any_session = False

        for session_subdir in session_subdirs:
            if not session_subdir.is_dir():
                continue
            session_id = session_subdir.name

            try:
                spike_counts_df, cluster_contacts_xyz = load_cluster_session_data(
                    cluster_dir, session_id
                )
            except ValueError as exc:
                logger.warning(
                    "load_gallery_data: skipping session '%s' in cluster '%s': %s",
                    session_id, cluster_folder, exc,
                )
                continue

            try:
                neuron_contacts_xyz = load_neuron_contacts(base_output, session_id)
            except ValueError as exc:
                raise ValueError(
                    f"load_gallery_data: cannot load neuron_contacts for session "
                    f"'{session_id}': {exc}"
                ) from exc

            forearm_vertices, forearm_colors, forearm_mesh = _load_forearm_geometry(
                base_output, session_id, sessions_metadata, cluster_folder,
            )

            if session_id in session_tangent_rotations:
                tangent_rotation = session_tangent_rotations[session_id]
            else:
                tangent_rotation = load_rf_camera_rotation(
                    output_dir.parent / 'rf_camera_settings',
                    session_id,
                )
                session_tangent_rotations[session_id] = tangent_rotation

            cell = GalleryCell(
                session_id=session_id,
                cluster_label=cluster_label,
                cluster_folder=cluster_folder,
                is_noise=is_noise,
                forearm_mesh=forearm_mesh,
                forearm_vertices=forearm_vertices,
                forearm_vertex_colors=forearm_colors,
                tangent_rotation=tangent_rotation,
                spike_counts_df=spike_counts_df,
                neuron_contacts_xyz=neuron_contacts_xyz,
                cluster_contacts_xyz=cluster_contacts_xyz,
                neuron_touches=neuron_touches_map.get(session_id, 0),
                neuron_cluster_touches=neuron_cluster_touches_map.get(session_id, 0),
                cluster_description=cluster_description,
                rf_metrics=rf_metrics,
            )

            gallery.cells[(session_id, cluster_label)] = cell
            all_session_ids.add(session_id)
            cluster_has_any_session = True

        if cluster_has_any_session:
            cluster_labels_found.append(cluster_label)

    gallery.session_ids = sorted(all_session_ids)
    gallery.cluster_labels = sorted(cluster_labels_found, key=_sort_key_for_cluster_label)

    return gallery


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_cluster_description(cluster_dir: Path) -> dict:
    path = cluster_dir / "cluster_description.json"
    if not path.exists():
        raise FileNotFoundError(
            f"_load_cluster_description: missing {path}"
        )
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        raise ValueError(
            f"_load_cluster_description: corrupt JSON at {path}"
        ) from exc


def _load_neuron_cluster_touches_safe(cluster_dir: Path, cluster_folder: str) -> Dict[str, int]:
    """Load neuron_cluster_touches.json, raising on corruption but not on absence."""
    try:
        return load_neuron_cluster_touches(cluster_dir)
    except ValueError as exc:
        raise ValueError(
            f"_load_neuron_cluster_touches_safe: failed for {cluster_folder}: {exc}"
        ) from exc


def _load_rf_metrics(cluster_dir: Path) -> Optional[dict]:
    """Load rf_metrics.json if present; return None if absent."""
    path = cluster_dir / "rf_metrics.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        raise ValueError(
            f"_load_rf_metrics: corrupt JSON at {path}"
        ) from exc


def _load_forearm_geometry(
    base_output: Path,
    session_id: str,
    sessions_metadata: Dict[str, dict],
    cluster_folder: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[trimesh.Trimesh]]:
    """Load forearm vertices and vertex colors for a session.

    Mesh construction (Delaunay) is deferred to the gallery viewer so it
    can apply per-session thresholds interactively.

    Returns (vertices, vertex_colors, mesh).  *mesh* is always None
    (Delaunay is built on demand in the viewer).
    """
    ply_str = sessions_metadata.get(session_id, {}).get("forearm_ply")
    if not ply_str:
        logger.warning(
            "load_gallery_data: no forearm PLY for session '%s' (cluster '%s') — "
            "cell will be rendered without mesh.",
            session_id, cluster_folder,
        )
        return None, None, None

    forearm_ply = Path(ply_str)
    if not forearm_ply.exists():
        logger.warning(
            "load_gallery_data: forearm PLY not found for session '%s' "
            "(cluster '%s'): %s — cell will be rendered without mesh.",
            session_id, cluster_folder, forearm_ply,
        )
        return None, None, None

    forearm_vertices = load_forearm_vertices(forearm_ply)
    if forearm_vertices is None:
        logger.warning(
            "load_gallery_data: forearm vertices empty for session '%s' "
            "(cluster '%s') — cell will be rendered without mesh.",
            session_id, cluster_folder,
        )
        return None, None, None

    forearm_colors = load_forearm_vertex_colors(forearm_ply)

    return forearm_vertices, forearm_colors, None
