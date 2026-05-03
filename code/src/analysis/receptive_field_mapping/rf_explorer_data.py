"""Per-frame data model and loader for the RF Feature-Space Explorer GUI."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .rf_data_loader import load_forearm_vertices
from .tangent_plane_alignment import compute_tangent_plane_rotation

logger = logging.getLogger(__name__)


@dataclass
class ExplorerSessionData:
    forearm_vertices: np.ndarray
    tangent_rotation: np.ndarray


@dataclass
class ExplorerData:
    pressure: np.ndarray
    velocity_signed: np.ndarray
    gesture_types: np.ndarray
    spikes: np.ndarray
    frame_vertex_idx: np.ndarray
    session_data: ExplorerSessionData

    @property
    def n_frames(self) -> int:
        return len(self.pressure)


# ------------------------------------------------------------------
# Sidecar cache helpers
# ------------------------------------------------------------------

def _explorer_cache_path(series_csv_path: Path) -> Path:
    """Return the .npz sidecar cache path for *series_csv_path*."""
    return series_csv_path.parent / f"{series_csv_path.stem}_explorer_cache.npz"


def _save_explorer_cache(series_csv_path: Path, data: ExplorerData) -> None:
    """Persist *data* to a compressed .npz sidecar next to *series_csv_path*.

    gesture_types (object array of strings) are encoded as integer codes plus a
    labels array so they survive the round-trip through numpy's .npz format.
    Write failures are logged as warnings rather than raised, to keep the caller
    on the happy path.
    """
    cache_path = _explorer_cache_path(series_csv_path)

    unique_labels, codes = np.unique(data.gesture_types, return_inverse=True)

    try:
        np.savez_compressed(
            cache_path,
            pressure=data.pressure,
            velocity_signed=data.velocity_signed,
            gesture_type_codes=codes.astype(np.int32),
            gesture_type_labels=unique_labels,
            spikes=data.spikes,
            frame_vertex_idx=data.frame_vertex_idx,
            forearm_vertices=data.session_data.forearm_vertices,
            tangent_rotation=data.session_data.tangent_rotation,
        )
    except Exception as exc:
        logger.warning(
            "_save_explorer_cache: could not write cache %s — %s", cache_path, exc
        )


def _load_explorer_cache(
    series_csv_path: Path,
    forearm_ply_path: Path,
) -> Optional[ExplorerData]:
    """Load the .npz sidecar cache for *series_csv_path* when it is fresh.

    Returns an ``ExplorerData`` on a valid cache hit, or ``None`` when:
    - the cache file does not exist,
    - the cache is older than either source file (mtime check),
    - the loaded arrays have unexpected shapes (raises ``ValueError``).

    A ``ValueError`` on shape mismatch propagates to the caller so that corrupt
    cache files fail loudly rather than silently returning wrong data.
    """
    cache_path = _explorer_cache_path(series_csv_path)

    if not cache_path.exists():
        return None

    cache_mtime = cache_path.stat().st_mtime
    csv_mtime = series_csv_path.stat().st_mtime
    ply_mtime = forearm_ply_path.stat().st_mtime

    if cache_mtime < max(csv_mtime, ply_mtime):
        logger.debug(
            "_load_explorer_cache: stale cache for %s — recomputing", series_csv_path.name
        )
        return None

    npz = np.load(cache_path, allow_pickle=False)

    pressure = npz["pressure"]
    velocity_signed = npz["velocity_signed"]
    gesture_type_codes = npz["gesture_type_codes"]
    gesture_type_labels = npz["gesture_type_labels"]
    spikes = npz["spikes"]
    frame_vertex_idx = npz["frame_vertex_idx"]
    forearm_vertices = npz["forearm_vertices"]
    tangent_rotation = npz["tangent_rotation"]

    # Validate shapes before trusting the cache.
    n = len(pressure)
    for name, arr, expected_ndim in [
        ("velocity_signed", velocity_signed, 1),
        ("gesture_type_codes", gesture_type_codes, 1),
        ("spikes", spikes, 1),
        ("frame_vertex_idx", frame_vertex_idx, 1),
    ]:
        if arr.ndim != expected_ndim or len(arr) != n:
            raise ValueError(
                f"_load_explorer_cache: cached array '{name}' has shape {arr.shape}, "
                f"expected 1-D array of length {n}: {cache_path}"
            )
    if forearm_vertices.ndim != 2 or forearm_vertices.shape[1] != 3:
        raise ValueError(
            f"_load_explorer_cache: cached 'forearm_vertices' has shape "
            f"{forearm_vertices.shape} (expected (N, 3)): {cache_path}"
        )
    if tangent_rotation.ndim != 2 or tangent_rotation.shape != (3, 3):
        raise ValueError(
            f"_load_explorer_cache: cached 'tangent_rotation' has shape "
            f"{tangent_rotation.shape} (expected (3, 3)): {cache_path}"
        )

    gesture_types = gesture_type_labels[gesture_type_codes]

    session_data = ExplorerSessionData(
        forearm_vertices=forearm_vertices,
        tangent_rotation=tangent_rotation,
    )
    return ExplorerData(
        pressure=pressure,
        velocity_signed=velocity_signed,
        gesture_types=gesture_types,
        spikes=spikes,
        frame_vertex_idx=frame_vertex_idx,
        session_data=session_data,
    )


def load_explorer_data(
    series_csv_path: Path,
    forearm_ply_path: Path,
) -> ExplorerData:
    cached = _load_explorer_cache(series_csv_path, forearm_ply_path)
    if cached is not None:
        logger.debug(
            "load_explorer_data: cache hit for %s", series_csv_path.name
        )
        return cached

    df = pd.read_csv(series_csv_path)

    mask = df["contact_location_x"].notna()
    df = df[mask].reset_index(drop=True)

    pressure = df["pressure"].to_numpy(dtype=np.float64)
    velocity_signed = df["hand_velocity_signed"].to_numpy(dtype=np.float64)
    gesture_types = df["gesture_type"].to_numpy(dtype=object)
    spikes = df["Nerve_spike"].to_numpy(dtype=bool)
    contact_pts = df[
        ["contact_location_x", "contact_location_y", "contact_location_z"]
    ].to_numpy(dtype=np.float64)

    vertices = load_forearm_vertices(forearm_ply_path)
    if vertices is None:
        raise ValueError(
            f"load_explorer_data: could not load forearm vertices from {forearm_ply_path}"
        )

    contact_centroid = contact_pts.mean(axis=0)
    rotation = compute_tangent_plane_rotation(vertices, contact_centroid)
    if rotation is None:
        raise ValueError(
            f"load_explorer_data: compute_tangent_plane_rotation returned None "
            f"for {forearm_ply_path}"
        )

    rotated_vertices = (rotation @ vertices.T).T
    rotated_contacts = (rotation @ contact_pts.T).T

    tree = cKDTree(rotated_vertices)
    distances, frame_vertex_idx = tree.query(rotated_contacts)

    bad = distances > 15.0
    if np.any(bad):
        logger.warning(
            "load_explorer_data: dropping %d frame(s) with nearest-vertex distance "
            "> 15mm (max=%.2f mm) — likely mesh sparsity at contact boundary",
            bad.sum(), distances.max(),
        )
        good = ~bad
        pressure = pressure[good]
        velocity_signed = velocity_signed[good]
        gesture_types = gesture_types[good]
        spikes = spikes[good]
        frame_vertex_idx = frame_vertex_idx[good]

    session_data = ExplorerSessionData(
        forearm_vertices=rotated_vertices,
        tangent_rotation=rotation,
    )

    result = ExplorerData(
        pressure=pressure,
        velocity_signed=velocity_signed,
        gesture_types=gesture_types,
        spikes=spikes,
        frame_vertex_idx=frame_vertex_idx,
        session_data=session_data,
    )

    _save_explorer_cache(series_csv_path, result)

    return result
