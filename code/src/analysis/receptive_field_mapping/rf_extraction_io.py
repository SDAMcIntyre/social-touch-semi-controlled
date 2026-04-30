"""Save/load helpers for RF cluster extraction intermediate artifacts.

Each load function validates file existence and data shape, raising ValueError
on corrupt or missing artifacts (fail-fast convention).
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _session_dir(output_dir: Path, session_id: str) -> Path:
    return output_dir / 'sessions' / session_id


def _cluster_session_dir(cluster_dir: Path, session_id: str) -> Path:
    return cluster_dir / 'sessions' / session_id


# ---------------------------------------------------------------------------
# Neuron contacts (all-cluster, per session)
# ---------------------------------------------------------------------------

def save_neuron_contacts(output_dir: Path, session_id: str, xyz: np.ndarray) -> None:
    d = _session_dir(output_dir, session_id)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / 'neuron_contacts_xyz.npy', xyz.astype(np.float64))


def load_neuron_contacts(output_dir: Path, session_id: str) -> np.ndarray:
    path = _session_dir(output_dir, session_id) / 'neuron_contacts_xyz.npy'
    if not path.exists():
        raise ValueError(
            f"load_neuron_contacts: artifact missing for session '{session_id}': {path}"
        )
    try:
        arr = np.load(path)
    except Exception as exc:
        raise ValueError(
            f"load_neuron_contacts: corrupt .npy for session '{session_id}': {path}"
        ) from exc
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"load_neuron_contacts: expected (N, 3) array, got {arr.shape} "
            f"for session '{session_id}': {path}"
        )
    return arr


# ---------------------------------------------------------------------------
# Forearm vertices (artifact path, per session)
# ---------------------------------------------------------------------------

def save_forearm_vertices(output_dir: Path, session_id: str, vertices: np.ndarray) -> None:
    d = _session_dir(output_dir, session_id)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / 'forearm_vertices.npy', vertices.astype(np.float64))


def load_forearm_vertices_artifact(output_dir: Path, session_id: str) -> np.ndarray:
    path = _session_dir(output_dir, session_id) / 'forearm_vertices.npy'
    if not path.exists():
        raise ValueError(
            f"load_forearm_vertices_artifact: missing for session '{session_id}': {path}"
        )
    try:
        arr = np.load(path)
    except Exception as exc:
        raise ValueError(
            f"load_forearm_vertices_artifact: corrupt .npy for session '{session_id}': {path}"
        ) from exc
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"load_forearm_vertices_artifact: expected (N, 3) array, got {arr.shape} "
            f"for session '{session_id}': {path}"
        )
    return arr


# ---------------------------------------------------------------------------
# Cluster × session data (spike DataFrame + cluster contacts)
# ---------------------------------------------------------------------------

def save_cluster_session_data(
    cluster_dir: Path,
    session_id: str,
    spike_df: pd.DataFrame,
    contacts_xyz: np.ndarray,
) -> None:
    d = _cluster_session_dir(cluster_dir, session_id)
    d.mkdir(parents=True, exist_ok=True)
    spike_df.to_csv(d / 'session_spike_counts.csv', index=False)
    np.save(d / 'cluster_contacts_xyz.npy', contacts_xyz.astype(np.float64))


def load_cluster_session_data(
    cluster_dir: Path,
    session_id: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    d = _cluster_session_dir(cluster_dir, session_id)
    spike_path = d / 'session_spike_counts.csv'
    contacts_path = d / 'cluster_contacts_xyz.npy'

    if not spike_path.exists():
        raise ValueError(
            f"load_cluster_session_data: missing spike CSV for session '{session_id}': {spike_path}"
        )
    if not contacts_path.exists():
        raise ValueError(
            f"load_cluster_session_data: missing contacts for session '{session_id}': {contacts_path}"
        )

    try:
        spike_df = pd.read_csv(spike_path)
    except Exception as exc:
        raise ValueError(
            f"load_cluster_session_data: corrupt spike CSV for session '{session_id}': {spike_path}"
        ) from exc

    try:
        contacts = np.load(contacts_path)
    except Exception as exc:
        raise ValueError(
            f"load_cluster_session_data: corrupt contacts for session '{session_id}': {contacts_path}"
        ) from exc

    if contacts.ndim != 2 or contacts.shape[1] != 3:
        raise ValueError(
            f"load_cluster_session_data: expected (N, 3) contacts, got {contacts.shape} "
            f"for session '{session_id}': {contacts_path}"
        )
    return spike_df, contacts


# ---------------------------------------------------------------------------
# Neuron touches (total per session across all clusters)
# ---------------------------------------------------------------------------

def save_neuron_touches(output_dir: Path, neuron_touches: Dict[str, int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'neuron_touches.json', 'w') as f:
        json.dump(neuron_touches, f, indent=2)


def load_neuron_touches(output_dir: Path) -> Dict[str, int]:
    path = output_dir / 'neuron_touches.json'
    if not path.exists():
        raise ValueError(f"load_neuron_touches: artifact missing: {path}")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        raise ValueError(f"load_neuron_touches: corrupt JSON: {path}") from exc


# ---------------------------------------------------------------------------
# Neuron cluster touches (per session within one cluster)
# ---------------------------------------------------------------------------

def save_neuron_cluster_touches(cluster_dir: Path, neuron_cluster_touches: Dict[str, int]) -> None:
    cluster_dir.mkdir(parents=True, exist_ok=True)
    with open(cluster_dir / 'neuron_cluster_touches.json', 'w') as f:
        json.dump(neuron_cluster_touches, f, indent=2)


def load_neuron_cluster_touches(cluster_dir: Path) -> Dict[str, int]:
    path = cluster_dir / 'neuron_cluster_touches.json'
    if not path.exists():
        raise ValueError(f"load_neuron_cluster_touches: artifact missing: {path}")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        raise ValueError(f"load_neuron_cluster_touches: corrupt JSON: {path}") from exc


# ---------------------------------------------------------------------------
# Extraction summary sentinel
# ---------------------------------------------------------------------------

def save_extraction_summary(
    output_dir: Path,
    feature_combination: str,
    clusterer: str,
    summary_data: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'extraction_summary.json', 'w') as f:
        json.dump(
            {
                'feature_combination': feature_combination,
                'clusterer': clusterer,
                'clusters': summary_data,
            },
            f,
            indent=2,
        )


def load_extraction_summary(output_dir: Path) -> dict:
    path = output_dir / 'extraction_summary.json'
    if not path.exists():
        raise ValueError(
            f"load_extraction_summary: sentinel missing (extraction has not run): {path}"
        )
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        raise ValueError(
            f"load_extraction_summary: corrupt extraction_summary.json: {path}"
        ) from exc


# ---------------------------------------------------------------------------
# Sessions metadata (PLY paths and other per-session lookups)
# ---------------------------------------------------------------------------

def save_sessions_metadata(output_dir: Path, metadata: Dict[str, dict]) -> None:
    """Save per-session metadata dict, e.g. {session_id: {"forearm_ply": str|None}}."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'sessions_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def load_sessions_metadata(output_dir: Path) -> Dict[str, dict]:
    path = output_dir / 'sessions_metadata.json'
    if not path.exists():
        raise ValueError(f"load_sessions_metadata: artifact missing: {path}")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        raise ValueError(f"load_sessions_metadata: corrupt JSON: {path}") from exc


# ---------------------------------------------------------------------------
# Visualization summary sentinel
# ---------------------------------------------------------------------------

def save_visualization_summary(
    output_dir: Path,
    projection_method: Optional[str],
    disjoint_mask_distance_mm: float,
    extraction_summary_mtime: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'rf_visualization_summary.json', 'w') as f:
        json.dump(
            {
                'projection_method': projection_method,
                'disjoint_mask_distance_mm': disjoint_mask_distance_mm,
                'extraction_summary_mtime': extraction_summary_mtime,
            },
            f,
            indent=2,
        )


def load_visualization_summary(output_dir: Path) -> dict:
    path = output_dir / 'rf_visualization_summary.json'
    if not path.exists():
        raise ValueError(f"load_visualization_summary: sentinel missing: {path}")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        raise ValueError(
            f"load_visualization_summary: corrupt rf_visualization_summary.json: {path}"
        ) from exc


def visualization_is_up_to_date(
    output_dir: Path,
    projection_method: Optional[str],
    disjoint_mask_distance_mm: float,
    force: bool = False,
) -> bool:
    """Return True if visualization artifacts match the given params and are not stale."""
    if force:
        return False
    extraction_path = output_dir / 'extraction_summary.json'
    if not extraction_path.exists():
        return False
    try:
        vis = load_visualization_summary(output_dir)
    except ValueError:
        return False
    return (
        vis.get('projection_method') == projection_method
        and vis.get('disjoint_mask_distance_mm') == disjoint_mask_distance_mm
        and vis.get('extraction_summary_mtime') == extraction_path.stat().st_mtime
    )


# ---------------------------------------------------------------------------
# Delaunay threshold persistence (user-preference file, not a pipeline artifact)
# ---------------------------------------------------------------------------

def load_delaunay_thresholds(output_dir: Path) -> Dict[str, float]:
    """Load per-session Delaunay max-edge thresholds from JSON sidecar.

    Returns an empty dict when the file does not exist (first launch).
    """
    path = output_dir / 'delaunay_thresholds.json'
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def save_delaunay_thresholds(output_dir: Path, thresholds: Dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'delaunay_thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=2)


# ---------------------------------------------------------------------------
# Session camera persistence (user-preference file, not a pipeline artifact)
# ---------------------------------------------------------------------------

def load_session_cameras(output_dir: Path) -> Dict[str, dict]:
    """Load per-session camera parameters from JSON sidecar.

    Returns an empty dict when the file does not exist (first launch).
    Raises ValueError on malformed JSON.
    """
    path = output_dir / 'session_cameras.json'
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        raise ValueError(
            f"load_session_cameras: corrupt session_cameras.json: {path}"
        ) from exc


def save_session_cameras(output_dir: Path, cameras: Dict[str, dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'session_cameras.json', 'w') as f:
        json.dump(cameras, f, indent=2)
