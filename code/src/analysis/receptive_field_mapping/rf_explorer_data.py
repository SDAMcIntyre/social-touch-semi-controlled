"""Per-frame data model and loader for the RF Feature-Space Explorer GUI."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import re
import time
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .rf_data_loader import load_forearm_vertices
from .rf_extraction_io import RF_CAMERA_SETTINGS_FILENAME, load_rf_camera_rotation

logger = logging.getLogger(__name__)


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


@dataclass
class ExplorerSessionData:
    forearm_vertices: np.ndarray
    tangent_rotation: np.ndarray


@dataclass
class ExplorerData:
    """Per-frame scatter data and per-contact-point vertex mapping.

    Source-frame arrays (``pressure``, ``velocity_signed``, ``gesture_types``,
    ``spikes``, ``iff``) are indexed 0 … n_frames-1.

    Contact-point arrays (``cp_frame_idx``, ``cp_vertex_idx``) have length
    n_contact_pts ≥ n_frames.  ``cp_frame_idx[i]`` is the index into the
    source-frame arrays for contact point ``i``; ``cp_vertex_idx[i]`` is the
    nearest forearm vertex for that contact point.
    """

    pressure: np.ndarray
    velocity_signed: np.ndarray
    gesture_types: np.ndarray
    spikes: np.ndarray
    iff: np.ndarray
    cp_frame_idx: np.ndarray
    cp_vertex_idx: np.ndarray
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
            gesture_type_labels=np.array(unique_labels, dtype=str),
            spikes=data.spikes,
            iff=data.iff,
            cp_frame_idx=data.cp_frame_idx,
            cp_vertex_idx=data.cp_vertex_idx,
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
    camera_settings_json: Path,
) -> Optional[ExplorerData]:
    """Load the .npz sidecar cache for *series_csv_path* when it is fresh.

    Returns an ``ExplorerData`` on a valid cache hit, or ``None`` when:
    - the cache file does not exist,
    - the cache is older than any source file (mtime check),
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
    cam_mtime = camera_settings_json.stat().st_mtime if camera_settings_json.exists() else 0.0

    if cache_mtime < max(csv_mtime, ply_mtime, cam_mtime):
        logger.debug(
            "_load_explorer_cache: stale cache for %s — recomputing", series_csv_path.name
        )
        return None

    npz = np.load(cache_path, allow_pickle=True)

    if "cp_frame_idx" not in npz:
        logger.debug(
            "_load_explorer_cache: old-format cache (no cp_frame_idx) for %s — recomputing",
            series_csv_path.name,
        )
        return None

    if "iff" not in npz:
        logger.debug(
            "_load_explorer_cache: old-format cache (no iff) for %s — recomputing",
            series_csv_path.name,
        )
        return None

    pressure = npz["pressure"]
    velocity_signed = npz["velocity_signed"]
    gesture_type_codes = npz["gesture_type_codes"]
    gesture_type_labels = npz["gesture_type_labels"]
    spikes = npz["spikes"]
    iff = npz["iff"]
    cp_frame_idx = npz["cp_frame_idx"]
    cp_vertex_idx = npz["cp_vertex_idx"]
    forearm_vertices = npz["forearm_vertices"]
    tangent_rotation = npz["tangent_rotation"]

    # Validate frame-level arrays before trusting the cache.
    n = len(pressure)
    for name, arr in [
        ("velocity_signed", velocity_signed),
        ("gesture_type_codes", gesture_type_codes),
        ("spikes", spikes),
        ("iff", iff),
    ]:
        if arr.ndim != 1 or len(arr) != n:
            raise ValueError(
                f"_load_explorer_cache: cached array '{name}' has shape {arr.shape}, "
                f"expected 1-D array of length {n}: {cache_path}"
            )

    # Validate contact-point arrays (lengths must agree with each other but
    # are ≥ n_frames, so we cannot check against n directly).
    if cp_frame_idx.ndim != 1:
        raise ValueError(
            f"_load_explorer_cache: cached 'cp_frame_idx' has shape {cp_frame_idx.shape}, "
            f"expected 1-D array: {cache_path}"
        )
    if cp_vertex_idx.ndim != 1 or len(cp_vertex_idx) != len(cp_frame_idx):
        raise ValueError(
            f"_load_explorer_cache: cached 'cp_vertex_idx' has shape {cp_vertex_idx.shape}, "
            f"expected 1-D array of length {len(cp_frame_idx)}: {cache_path}"
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
        iff=iff.astype(np.float64),
        cp_frame_idx=cp_frame_idx.astype(np.int64),
        cp_vertex_idx=cp_vertex_idx.astype(np.int64),
        session_data=session_data,
    )


def load_explorer_data(
    series_csv_path: Path,
    forearm_ply_path: Path,
) -> ExplorerData:
    session_id = series_csv_path.stem.removesuffix("_series_augmented")
    camera_settings_dir = series_csv_path.parent.parent / "rf_camera_settings"
    camera_settings_json = camera_settings_dir / RF_CAMERA_SETTINGS_FILENAME

    t_session = time.perf_counter()
    cached = _load_explorer_cache(series_csv_path, forearm_ply_path, camera_settings_json)
    if cached is not None:
        print(f"{_ts()} | [RF Explorer] [{series_csv_path.stem}]: cache hit ({time.perf_counter() - t_session:.2f}s)", flush=True)
        return cached

    tag = series_csv_path.stem
    print(f"{_ts()} | [RF Explorer] [{tag}]: computing (cache miss)...", flush=True)
    t = time.perf_counter()
    df = pd.read_csv(series_csv_path)
    print(f"{_ts()} | [RF Explorer] [{tag}]   CSV read: {time.perf_counter() - t:.1f}s  rows={len(df)}", flush=True)

    for required_col in ("contact_points", "Nerve_freq"):
        if required_col not in df.columns:
            raise ValueError(
                f"load_explorer_data: '{required_col}' column missing from {series_csv_path}"
            )

    cp_raw = df["contact_points"].fillna("[]").values

    # "[[x y z] [x y z] ...]" → count('[') = n_pts + 1 (outer bracket).
    t = time.perf_counter()
    n_rows = len(cp_raw)
    progress_step = max(n_rows // 5, 1)
    pts_per_row = np.empty(n_rows, dtype=np.int64)
    for i, s in enumerate(cp_raw):
        pts_per_row[i] = s.count('[') - 1
        if (i + 1) % progress_step == 0:
            print(f"{_ts()} | [RF Explorer] [{tag}]   bracket count: {i + 1}/{n_rows} rows", flush=True)
    print(f"{_ts()} | [RF Explorer] [{tag}]   bracket count: {time.perf_counter() - t:.1f}s", flush=True)

    valid_mask = pts_per_row > 0
    if not valid_mask.any():
        raise ValueError(
            f"load_explorer_data: no valid contact points found in {series_csv_path}"
        )
    df = df[valid_mask].reset_index(drop=True)
    pts_per_row = pts_per_row[valid_mask]
    cp_raw = cp_raw[valid_mask]

    pressure = df["pressure"].to_numpy(dtype=np.float64)
    velocity_signed = df["hand_velocity_signed"].to_numpy(dtype=np.float64)
    gesture_types = df["gesture_type"].to_numpy(dtype=object)
    spikes = df["Nerve_spike"].to_numpy(dtype=bool)
    iff = df["Nerve_freq"].to_numpy(dtype=np.float64)

    # Deduplicate forward-filled contact_points to unique 30Hz frames.
    n_valid = len(cp_raw)
    change_mask = np.empty(n_valid, dtype=bool)
    change_mask[0] = True
    change_mask[1:] = cp_raw[1:] != cp_raw[:-1]
    unique_indices = np.where(change_mask)[0]
    n_unique = len(unique_indices)
    run_lengths = np.empty(n_unique, dtype=np.int64)
    run_lengths[:-1] = np.diff(unique_indices)
    run_lengths[-1] = n_valid - unique_indices[-1]
    unique_pts_per_row = pts_per_row[unique_indices]
    unique_cp_raw = cp_raw[unique_indices]
    print(
        f"{_ts()} | [RF Explorer] [{tag}]   dedup: {n_valid} -> {n_unique} unique frames "
        f"(avg run {run_lengths.mean():.1f})",
        flush=True,
    )

    # Parse only unique 30Hz frames into pre-allocated array (no intermediate list).
    t = time.perf_counter()
    _bracket_re = re.compile(r'\[([^\[\]]+)\]')
    total_unique_pts = int(unique_pts_per_row.sum())
    if total_unique_pts == 0:
        raise ValueError(
            f"load_explorer_data: no contact point coordinates parsed from {series_csv_path}"
        )
    all_pts = np.empty((total_unique_pts, 3), dtype=np.float64)
    offset = 0
    progress_step = max(n_unique // 5, 1)
    for i, s in enumerate(unique_cp_raw):
        for m in _bracket_re.findall(s):
            coords = m.split()
            all_pts[offset, 0] = float(coords[0])
            all_pts[offset, 1] = float(coords[1])
            all_pts[offset, 2] = float(coords[2])
            offset += 1
        if (i + 1) % progress_step == 0:
            print(f"{_ts()} | [RF Explorer] [{tag}]   contact parse: {i + 1}/{n_unique} unique frames", flush=True)
    all_pts = all_pts[:offset]
    print(f"{_ts()} | [RF Explorer] [{tag}]   contact points: {time.perf_counter() - t:.1f}s  pts={len(all_pts)}", flush=True)

    nan_mask = np.any(np.isnan(all_pts), axis=1)
    if nan_mask.any():
        logger.warning(
            "load_explorer_data: dropping %d contact point(s) with NaN coordinates in %s",
            nan_mask.sum(), series_csv_path.name,
        )
        # Recount per unique group after NaN removal.
        unique_group_per_pt = np.repeat(np.arange(n_unique, dtype=np.int64), unique_pts_per_row)
        keep = ~nan_mask
        all_pts = all_pts[keep]
        unique_pts_per_row = np.bincount(
            unique_group_per_pt[keep], minlength=n_unique
        ).astype(np.int64)
        if len(all_pts) == 0:
            raise ValueError(
                f"load_explorer_data: all contact points have NaN coordinates in {series_csv_path}"
            )

    t = time.perf_counter()
    print(f"{_ts()} | [RF Explorer] [{tag}]   loading forearm mesh...", flush=True)
    vertices = load_forearm_vertices(forearm_ply_path)
    if vertices is None:
        raise ValueError(
            f"load_explorer_data: could not load forearm vertices from {forearm_ply_path}"
        )
    print(f"{_ts()} | [RF Explorer] [{tag}]   forearm mesh: {time.perf_counter() - t:.1f}s  verts={len(vertices)}", flush=True)

    t = time.perf_counter()
    rotation = load_rf_camera_rotation(camera_settings_dir, session_id)
    rotated_vertices = (rotation @ vertices.T).T
    rotated_contacts = (rotation @ all_pts.T).T
    print(f"{_ts()} | [RF Explorer] [{tag}]   rotation: {time.perf_counter() - t:.1f}s", flush=True)

    t_kdtree = time.perf_counter()
    tree = cKDTree(rotated_vertices)
    distances, unique_vertex_idx = tree.query(rotated_contacts)
    print(f"{_ts()} | [RF Explorer] [{tag}]   KDTree: {time.perf_counter() - t_kdtree:.1f}s  queries={len(rotated_contacts)}", flush=True)

    # Filter contact points whose nearest vertex is > 15mm away.
    bad_pts = distances > 15.0
    if np.any(bad_pts):
        logger.warning(
            "load_explorer_data: dropping %d contact point(s) with nearest-vertex "
            "distance > 15mm (max=%.2f mm) — likely mesh sparsity at contact boundary",
            bad_pts.sum(), distances[bad_pts].max(),
        )
        good_pts = ~bad_pts
        unique_vertex_idx = unique_vertex_idx[good_pts]
        unique_group_per_pt = np.repeat(np.arange(n_unique, dtype=np.int64), unique_pts_per_row)
        unique_pts_per_row = np.bincount(
            unique_group_per_pt[good_pts], minlength=n_unique
        ).astype(np.int64)

    # Expand unique-level vertex indices to full 1kHz resolution.
    # Each unique group's vertex block is tiled by its run_length.
    full_pts_per_row = np.repeat(unique_pts_per_row, run_lengths)
    surviving_row_mask = full_pts_per_row > 0

    if not surviving_row_mask.any():
        raise ValueError(
            f"load_explorer_data: all frames were dropped after distance filtering "
            f"for {series_csv_path}"
        )

    # Drop frames that lost all contact points.
    if not surviving_row_mask.all():
        n_dropped = int(n_valid - surviving_row_mask.sum())
        logger.warning(
            "load_explorer_data: dropping %d source frame(s) whose contact points "
            "all exceeded the 15mm distance threshold",
            n_dropped,
        )
        pressure = pressure[surviving_row_mask]
        velocity_signed = velocity_signed[surviving_row_mask]
        gesture_types = gesture_types[surviving_row_mask]
        spikes = spikes[surviving_row_mask]
        iff = iff[surviving_row_mask]
        full_pts_per_row = full_pts_per_row[surviving_row_mask]

    # Build cp_frame_idx: maps each contact point to its 1kHz frame index.
    n_surviving = int(surviving_row_mask.sum())
    cp_frame_idx = np.repeat(np.arange(n_surviving, dtype=np.int64), full_pts_per_row)

    # Build cp_vertex_idx by tiling each unique group's vertex block.
    offsets = np.zeros(n_unique + 1, dtype=np.int64)
    np.cumsum(unique_pts_per_row, out=offsets[1:])
    parts = []
    for u in range(n_unique):
        k = unique_pts_per_row[u]
        if k == 0:
            continue
        block = unique_vertex_idx[offsets[u]:offsets[u] + k]
        parts.append(np.tile(block, run_lengths[u]))
    cp_vertex_idx = np.concatenate(parts).astype(np.int64)

    session_data = ExplorerSessionData(
        forearm_vertices=rotated_vertices,
        tangent_rotation=rotation,
    )

    result = ExplorerData(
        pressure=pressure,
        velocity_signed=velocity_signed,
        gesture_types=gesture_types,
        spikes=spikes,
        iff=iff,
        cp_frame_idx=cp_frame_idx,
        cp_vertex_idx=cp_vertex_idx,
        session_data=session_data,
    )

    t = time.perf_counter()
    _save_explorer_cache(series_csv_path, result)
    print(f"{_ts()} | [RF Explorer] [{tag}]   cache saved: {time.perf_counter() - t:.1f}s", flush=True)

    print(
        f"{_ts()} | [RF Explorer] [{tag}]: done in "
        f"{time.perf_counter() - t_session:.1f}s  frames={len(result.pressure)}  "
        f"contact_pts={len(result.cp_frame_idx)}",
        flush=True,
    )
    return result
