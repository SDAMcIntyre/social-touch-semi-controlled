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
    camera_settings_mtime: Optional[float] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'rf_visualization_summary.json', 'w') as f:
        json.dump(
            {
                'projection_method': projection_method,
                'disjoint_mask_distance_mm': disjoint_mask_distance_mm,
                'extraction_summary_mtime': extraction_summary_mtime,
                'camera_settings_mtime': camera_settings_mtime,
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
    camera_settings_path: Optional[Path] = None,
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
        and vis.get('camera_settings_mtime') == (
            camera_settings_path.stat().st_mtime if (camera_settings_path is not None and camera_settings_path.exists()) else None
        )
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


# ---------------------------------------------------------------------------
# Cluster description formatting
# ---------------------------------------------------------------------------

def description_summary_line(desc: dict, separator: str = ' — ') -> str:
    gen = desc.get('generation_params')
    if gen:
        return _format_generation_params(gen, desc, separator)
    return _format_legacy_ranges(desc, separator)


def _format_generation_params(gen: dict, desc: dict, separator: str) -> str:
    parts: list[str] = []
    algo = gen.get('algorithm', 'unknown')

    if algo == 'type_stratified':
        base = gen.get('base_algorithm', '?')
        header = f"type_stratified ({base})"
        type_key = gen.get('type')
        if type_key:
            parts.append(f"type: {type_key}")
        if base == 'binning':
            if gen.get('n_bins') is not None:
                parts.append(f"n_bins: {gen['n_bins']}")
            if gen.get('primary_feature'):
                parts.append(f"feature: {gen['primary_feature']}")
        elif base == 'kmeans' and gen.get('k') is not None:
            parts.append(f"k: {gen['k']}")
    elif algo == 'binning':
        header = 'binning'
        if gen.get('n_bins') is not None:
            parts.append(f"n_bins: {gen['n_bins']}")
        if gen.get('bin_method'):
            parts.append(f"method: {gen['bin_method']}")
        if gen.get('primary_feature'):
            parts.append(f"feature: {gen['primary_feature']}")
    elif algo == 'kmeans':
        header = 'kmeans'
        if gen.get('k') is not None:
            parts.append(f"k: {gen['k']}")
    elif algo == 'dbscan':
        header = 'dbscan'
        if gen.get('eps') is not None:
            parts.append(f"eps: {gen['eps']:.3f}")
        if gen.get('min_samples') is not None:
            parts.append(f"min_samples: {gen['min_samples']}")
    elif algo == 'hierarchical':
        header = 'hierarchical'
        if gen.get('k') is not None:
            parts.append(f"k: {gen['k']}")
    elif algo == 'gmm':
        header = 'gmm'
        if gen.get('k') is not None:
            parts.append(f"k: {gen['k']}")
        if gen.get('covariance_type'):
            parts.append(f"cov: {gen['covariance_type']}")
        if gen.get('features'):
            parts.append(f"features: {', '.join(gen['features'])}")
    else:
        header = algo

    br = desc.get('bin_range')
    if br:
        parts.append(f"{br['feature']}: [{br['low']}, {br['high']}]")

    dr = desc.get('display_ranges') or {}
    if dr:
        for label, r in dr.items():
            parts.append(f"{label}: [{r['min']}, {r['max']}]")
    else:
        fr = desc.get('feature_ranges') or {}
        for col, r in fr.items():
            parts.append(f"{col}: [{r['min']}, {r['max']}]")

    if parts:
        return header + separator + separator.join(parts)
    return header


# ---------------------------------------------------------------------------
# RF camera settings (pipeline authoritative rotation source, per session)
# ---------------------------------------------------------------------------

RF_CAMERA_SETTINGS_FILENAME = "rf_camera_settings.json"


def load_rf_camera_settings(output_dir: Path) -> Dict[str, dict]:
    """Load per-session RF camera settings from the pipeline settings file.

    Returns an empty dict when the file does not exist (first launch).
    Raises ValueError on malformed JSON.
    """
    path = output_dir / RF_CAMERA_SETTINGS_FILENAME
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:
        raise ValueError(
            f"load_rf_camera_settings: corrupt {RF_CAMERA_SETTINGS_FILENAME}: {path}"
        ) from exc


def save_rf_camera_settings(output_dir: Path, cameras: Dict[str, dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / RF_CAMERA_SETTINGS_FILENAME, "w") as f:
        json.dump(cameras, f, indent=2)


def load_rf_camera_rotation(output_dir: Path, session_id: str) -> np.ndarray:
    """Load the rotation matrix for a single session from saved camera settings.

    Raises ValueError when the settings file is absent or the session key is missing.
    The caller must run set_rf_camera_settings first.
    """
    from analysis.receptive_field_mapping.tangent_plane_alignment import (
        camera_settings_to_rotation,
    )

    cameras = load_rf_camera_settings(output_dir)
    if not cameras:
        raise ValueError(
            f"load_rf_camera_rotation: no camera settings found at {output_dir / RF_CAMERA_SETTINGS_FILENAME}. "
            "Run 'set_rf_camera_settings' first."
        )
    if session_id not in cameras:
        raise ValueError(
            f"load_rf_camera_rotation: session '{session_id}' not found in camera settings. "
            f"Available sessions: {sorted(cameras)}. Run 'set_rf_camera_settings' first."
        )
    return camera_settings_to_rotation(cameras[session_id])


def _format_legacy_ranges(desc: dict, separator: str) -> str:
    parts: list[str] = []
    if 'feature_ranges' in desc:
        for col_name, r in desc['feature_ranges'].items():
            parts.append(f"{col_name}: [{r['min']}, {r['max']}]")
    elif 'display_ranges' in desc:
        for label, r in desc['display_ranges'].items():
            parts.append(f"{label}: [{r['min']}, {r['max']}]")
    elif 'bin_range' in desc:
        br = desc['bin_range']
        parts.append(f"{br['feature']}: [{br['low']}, {br['high']}]")
    elif desc.get('primary_feature') and 'feature_ranges' in desc:
        pf = desc['primary_feature']
        if pf in desc['feature_ranges']:
            r = desc['feature_ranges'][pf]
            parts.append(f"{pf}: [{r['min']}, {r['max']}]")
    return separator.join(parts)
