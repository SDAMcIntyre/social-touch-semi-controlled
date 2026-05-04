"""Per-touch data model and loader for the Touch Playback Explorer GUI."""

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .rf_data_loader import load_forearm_vertices
from .tangent_plane_alignment import compute_tangent_plane_rotation

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = (
    "contact_points",
    "trial_id",
    "single_touch_id",
    "Nerve_spike",
    "gesture_type",
)

_DISTANCE_THRESHOLD_MM = 15.0

_bracket_re = re.compile(r'\[([^\[\]]+)\]')


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------

@dataclass
class PlaybackSessionData:
    forearm_vertices: np.ndarray    # (N, 3) rotated to tangent plane
    tangent_rotation: np.ndarray    # (3, 3) rotation matrix


@dataclass
class TouchEvent:
    trial_id: int
    single_touch_id: int
    gesture_type: str                       # 'tap', 'stroke_proximal', etc.
    # Per deduplicated 30Hz frame:
    frame_contact_pts: list                 # list of (K_i, 3) rotated contact coords
    frame_vertex_indices: list              # list of (K_i,) nearest vertex per contact pt
    frame_spikes: np.ndarray               # (n_frames,) bool — any spike in the 1kHz run


@dataclass
class PlaybackData:
    session_data: PlaybackSessionData
    trial_ids: list                         # sorted unique ints, excluding 0
    touches_by_trial: dict                  # trial_id -> sorted list[TouchEvent]


# ------------------------------------------------------------------
# Sidecar cache helpers
# ------------------------------------------------------------------

def _playback_cache_path(series_csv_path: Path) -> Path:
    """Return the .npz sidecar cache path for *series_csv_path*."""
    return series_csv_path.parent / f"{series_csv_path.stem}_playback_cache.npz"


def _save_playback_cache(
    series_csv_path: Path,
    data: PlaybackData,
) -> None:
    """Persist *data* to a compressed .npz sidecar next to *series_csv_path*.

    The nested per-trial / per-touch / per-frame structure is flattened using
    offset/count arrays so the whole dataset fits in a single .npz.

    Write failures are logged as warnings and NOT raised, so callers stay on
    the happy path.
    """
    cache_path = _playback_cache_path(series_csv_path)

    # Collect all touches in a stable order (sorted by trial_id, then single_touch_id).
    all_touches: list[TouchEvent] = []
    for tid in sorted(data.touches_by_trial.keys()):
        all_touches.extend(data.touches_by_trial[tid])

    n_touches = len(all_touches)

    touch_keys = np.array(
        [[t.trial_id, t.single_touch_id] for t in all_touches],
        dtype=np.int64,
    )  # (n_touches, 2)

    gesture_types = np.array([t.gesture_type for t in all_touches], dtype=str)

    # frame_spikes: concatenate all (n_frames_i,) bool arrays.
    frame_spikes_counts = np.array([len(t.frame_spikes) for t in all_touches], dtype=np.int64)
    frame_spikes_data = np.concatenate(
        [t.frame_spikes for t in all_touches]
    ).astype(bool) if n_touches > 0 else np.array([], dtype=bool)

    # Contact points: flatten per-touch, per-frame arrays.
    # cp_pts_data      : (total_pts, 3) float64 — all rotated contact coords
    # cp_vertex_data   : (total_pts,)   int64   — vertex index per contact pt
    # cp_pts_frame_touch: (total_pts,)  int64   — touch index (into all_touches)
    # cp_pts_frame_idx : (total_pts,)   int64   — frame index within that touch
    pts_parts: list[np.ndarray] = []
    vtx_parts: list[np.ndarray] = []
    touch_tag_parts: list[np.ndarray] = []
    frame_tag_parts: list[np.ndarray] = []

    for ti, touch in enumerate(all_touches):
        for fi, (pts, vtx) in enumerate(zip(touch.frame_contact_pts, touch.frame_vertex_indices)):
            pts_arr = np.asarray(pts, dtype=np.float64)
            vtx_arr = np.asarray(vtx, dtype=np.int64)
            k = len(vtx_arr)
            if k == 0:
                continue
            pts_parts.append(pts_arr)
            vtx_parts.append(vtx_arr)
            touch_tag_parts.append(np.full(k, ti, dtype=np.int64))
            frame_tag_parts.append(np.full(k, fi, dtype=np.int64))

    if pts_parts:
        cp_pts_data = np.concatenate(pts_parts, axis=0)
        cp_vertex_data = np.concatenate(vtx_parts)
        cp_pts_frame_touch = np.concatenate(touch_tag_parts)
        cp_pts_frame_idx = np.concatenate(frame_tag_parts)
    else:
        cp_pts_data = np.empty((0, 3), dtype=np.float64)
        cp_vertex_data = np.empty(0, dtype=np.int64)
        cp_pts_frame_touch = np.empty(0, dtype=np.int64)
        cp_pts_frame_idx = np.empty(0, dtype=np.int64)

    try:
        np.savez_compressed(
            cache_path,
            touch_keys=touch_keys,
            gesture_types=gesture_types,
            frame_spikes_data=frame_spikes_data,
            frame_spikes_counts=frame_spikes_counts,
            cp_pts_data=cp_pts_data,
            cp_vertex_data=cp_vertex_data,
            cp_pts_frame_touch=cp_pts_frame_touch,
            cp_pts_frame_idx=cp_pts_frame_idx,
            forearm_vertices=data.session_data.forearm_vertices,
            tangent_rotation=data.session_data.tangent_rotation,
        )
    except Exception as exc:
        logger.warning(
            "_save_playback_cache: could not write cache %s — %s", cache_path, exc
        )


def _load_playback_cache(
    series_csv_path: Path,
    forearm_ply_path: Path,
) -> Optional[PlaybackData]:
    """Load the .npz sidecar cache for *series_csv_path* when it is fresh.

    Returns a ``PlaybackData`` on a valid cache hit, or ``None`` when:
    - the cache file does not exist,
    - the cache is older than either source file (mtime check).

    Raises ``ValueError`` on shape mismatches (corrupt cache) so that stale
    data does not silently propagate to the GUI.
    """
    cache_path = _playback_cache_path(series_csv_path)

    if not cache_path.exists():
        return None

    cache_mtime = cache_path.stat().st_mtime
    csv_mtime = series_csv_path.stat().st_mtime
    ply_mtime = forearm_ply_path.stat().st_mtime

    if cache_mtime < max(csv_mtime, ply_mtime):
        logger.debug(
            "_load_playback_cache: stale cache for %s — recomputing",
            series_csv_path.name,
        )
        return None

    npz = np.load(cache_path, allow_pickle=True)

    # Validate that all expected keys are present (old-format guard).
    required_keys = (
        "touch_keys", "gesture_types",
        "frame_spikes_data", "frame_spikes_counts",
        "cp_pts_data", "cp_vertex_data",
        "cp_pts_frame_touch", "cp_pts_frame_idx",
        "forearm_vertices", "tangent_rotation",
    )
    missing_keys = [k for k in required_keys if k not in npz]
    if missing_keys:
        logger.debug(
            "_load_playback_cache: old-format cache (missing %s) for %s — recomputing",
            missing_keys, series_csv_path.name,
        )
        return None

    touch_keys = npz["touch_keys"]
    gesture_types = npz["gesture_types"]
    frame_spikes_data = npz["frame_spikes_data"]
    frame_spikes_counts = npz["frame_spikes_counts"]
    cp_pts_data = npz["cp_pts_data"]
    cp_vertex_data = npz["cp_vertex_data"]
    cp_pts_frame_touch = npz["cp_pts_frame_touch"]
    cp_pts_frame_idx = npz["cp_pts_frame_idx"]
    forearm_vertices = npz["forearm_vertices"]
    tangent_rotation = npz["tangent_rotation"]

    # Shape validation — raise on corruption.
    if touch_keys.ndim != 2 or touch_keys.shape[1] != 2:
        raise ValueError(
            f"_load_playback_cache: 'touch_keys' has shape {touch_keys.shape} "
            f"(expected (n_touches, 2)): {cache_path}"
        )
    n_touches = len(touch_keys)

    if gesture_types.ndim != 1 or len(gesture_types) != n_touches:
        raise ValueError(
            f"_load_playback_cache: 'gesture_types' has shape {gesture_types.shape}, "
            f"expected (n_touches={n_touches},): {cache_path}"
        )
    if frame_spikes_counts.ndim != 1 or len(frame_spikes_counts) != n_touches:
        raise ValueError(
            f"_load_playback_cache: 'frame_spikes_counts' has shape "
            f"{frame_spikes_counts.shape}, expected ({n_touches},): {cache_path}"
        )
    expected_spikes_len = int(frame_spikes_counts.sum())
    if frame_spikes_data.ndim != 1 or len(frame_spikes_data) != expected_spikes_len:
        raise ValueError(
            f"_load_playback_cache: 'frame_spikes_data' has length "
            f"{len(frame_spikes_data)}, expected {expected_spikes_len}: {cache_path}"
        )
    total_pts = len(cp_pts_data)
    for arr_name, arr in [
        ("cp_vertex_data", cp_vertex_data),
        ("cp_pts_frame_touch", cp_pts_frame_touch),
        ("cp_pts_frame_idx", cp_pts_frame_idx),
    ]:
        if arr.ndim != 1 or len(arr) != total_pts:
            raise ValueError(
                f"_load_playback_cache: '{arr_name}' has shape {arr.shape}, "
                f"expected ({total_pts},): {cache_path}"
            )
    if cp_pts_data.ndim != 2 or cp_pts_data.shape[1] != 3:
        raise ValueError(
            f"_load_playback_cache: 'cp_pts_data' has shape {cp_pts_data.shape} "
            f"(expected (total_pts, 3)): {cache_path}"
        )
    if forearm_vertices.ndim != 2 or forearm_vertices.shape[1] != 3:
        raise ValueError(
            f"_load_playback_cache: 'forearm_vertices' has shape "
            f"{forearm_vertices.shape} (expected (N, 3)): {cache_path}"
        )
    if tangent_rotation.ndim != 2 or tangent_rotation.shape != (3, 3):
        raise ValueError(
            f"_load_playback_cache: 'tangent_rotation' has shape "
            f"{tangent_rotation.shape} (expected (3, 3)): {cache_path}"
        )

    # Reconstruct touch events.
    spikes_offset = 0
    touches_by_trial: dict[int, list[TouchEvent]] = {}

    for ti in range(n_touches):
        trial_id = int(touch_keys[ti, 0])
        single_touch_id = int(touch_keys[ti, 1])
        gesture = str(gesture_types[ti])
        n_frames = int(frame_spikes_counts[ti])
        spikes = frame_spikes_data[spikes_offset: spikes_offset + n_frames].astype(bool)
        spikes_offset += n_frames

        # Collect per-frame contact pts and vertex indices for this touch.
        touch_mask = cp_pts_frame_touch == ti
        frame_pts_list: list[np.ndarray] = [np.empty((0, 3), dtype=np.float64)] * n_frames
        frame_vtx_list: list[np.ndarray] = [np.empty(0, dtype=np.int64)] * n_frames
        if touch_mask.any():
            pts_for_touch = cp_pts_data[touch_mask]
            vtx_for_touch = cp_vertex_data[touch_mask].astype(np.int64)
            fidx_for_touch = cp_pts_frame_idx[touch_mask].astype(np.int64)
            for fi in range(n_frames):
                fi_mask = fidx_for_touch == fi
                frame_pts_list[fi] = pts_for_touch[fi_mask]
                frame_vtx_list[fi] = vtx_for_touch[fi_mask]

        event = TouchEvent(
            trial_id=trial_id,
            single_touch_id=single_touch_id,
            gesture_type=gesture,
            frame_contact_pts=frame_pts_list,
            frame_vertex_indices=frame_vtx_list,
            frame_spikes=spikes,
        )
        touches_by_trial.setdefault(trial_id, []).append(event)

    # Sort within each trial by single_touch_id.
    for tid in touches_by_trial:
        touches_by_trial[tid].sort(key=lambda e: e.single_touch_id)

    trial_ids = sorted(touches_by_trial.keys())

    session_data = PlaybackSessionData(
        forearm_vertices=forearm_vertices,
        tangent_rotation=tangent_rotation,
    )
    return PlaybackData(
        session_data=session_data,
        trial_ids=trial_ids,
        touches_by_trial=touches_by_trial,
    )


# ------------------------------------------------------------------
# Main loader
# ------------------------------------------------------------------

def load_playback_data(
    series_csv_path: Path,
    forearm_ply_path: Path,
) -> PlaybackData:
    """Load per-touch playback data from *series_csv_path* + *forearm_ply_path*.

    Groups rows by ``(trial_id, single_touch_id)``, deduplicates consecutive
    identical ``contact_points`` strings (30Hz Kinect frames), rotates contact
    points and forearm vertices to the tangent plane, and snaps contacts to the
    nearest forearm vertex via cKDTree.

    Returns a ``PlaybackData`` with one ``TouchEvent`` per ``(trial_id,
    single_touch_id)`` pair, with empty/distance-filtered touches silently
    skipped (warning logged).

    Raises ``ValueError`` on missing required columns, empty data after
    filtering, forearm loading failure, or rotation failure.
    """
    t_session = time.perf_counter()

    # --- Cache check ---
    cached = _load_playback_cache(series_csv_path, forearm_ply_path)
    if cached is not None:
        print(
            f"{_ts()} | [Touch Playback] [{series_csv_path.stem}]: cache hit "
            f"({time.perf_counter() - t_session:.2f}s)",
            flush=True,
        )
        return cached

    tag = series_csv_path.stem
    print(f"{_ts()} | [Touch Playback] [{tag}]: computing (cache miss)...", flush=True)

    # --- Read CSV ---
    t = time.perf_counter()
    df = pd.read_csv(series_csv_path)
    print(
        f"{_ts()} | [Touch Playback] [{tag}]   CSV read: {time.perf_counter() - t:.1f}s  "
        f"rows={len(df)}",
        flush=True,
    )

    # Validate required columns.
    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"load_playback_data: required column(s) missing from {series_csv_path}: "
            f"{missing_cols}"
        )

    # Forward-fill contact_points before grouping (avoids per-group ffill overhead).
    df["contact_points"] = df.groupby(
        ["trial_id", "single_touch_id"]
    )["contact_points"].ffill()

    # Filter rows: keep only real touches (trial_id > 0 AND single_touch_id > 0).
    df = df[(df["trial_id"] > 0) & (df["single_touch_id"] > 0)].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"load_playback_data: no rows with trial_id > 0 and single_touch_id > 0 "
            f"in {series_csv_path}"
        )

    print(
        f"{_ts()} | [Touch Playback] [{tag}]   after filter: {len(df)} rows",
        flush=True,
    )

    # --- Load forearm mesh ---
    t = time.perf_counter()
    print(f"{_ts()} | [Touch Playback] [{tag}]   loading forearm mesh...", flush=True)
    vertices = load_forearm_vertices(forearm_ply_path)
    if vertices is None:
        raise ValueError(
            f"load_playback_data: could not load forearm vertices from {forearm_ply_path}"
        )
    print(
        f"{_ts()} | [Touch Playback] [{tag}]   forearm mesh: {time.perf_counter() - t:.1f}s  "
        f"verts={len(vertices)}",
        flush=True,
    )

    # --- Compute tangent-plane rotation from ALL contact points ---
    t = time.perf_counter()
    cp_raw_all = df["contact_points"].fillna("[]").values
    # Quick parse to get a centroid for the rotation — parse every row here
    # since we need the global centroid, not just unique frames.
    centroid_pts: list[list[float]] = []
    for s in cp_raw_all:
        for m in _bracket_re.findall(str(s)):
            coords = m.split()
            if len(coords) == 3:
                try:
                    centroid_pts.append([float(coords[0]), float(coords[1]), float(coords[2])])
                except ValueError:
                    pass

    if not centroid_pts:
        raise ValueError(
            f"load_playback_data: no parseable contact points found in {series_csv_path}"
        )

    all_pts_for_centroid = np.array(centroid_pts, dtype=np.float64)
    nan_rows = np.any(np.isnan(all_pts_for_centroid), axis=1)
    all_pts_for_centroid = all_pts_for_centroid[~nan_rows]
    if len(all_pts_for_centroid) == 0:
        raise ValueError(
            f"load_playback_data: all contact points have NaN coordinates in {series_csv_path}"
        )

    contact_centroid = all_pts_for_centroid.mean(axis=0)
    rotation = compute_tangent_plane_rotation(vertices, contact_centroid)
    if rotation is None:
        raise ValueError(
            f"load_playback_data: compute_tangent_plane_rotation returned None "
            f"for {forearm_ply_path}"
        )

    rotated_vertices = (rotation @ vertices.T).T
    print(
        f"{_ts()} | [Touch Playback] [{tag}]   rotation: {time.perf_counter() - t:.1f}s",
        flush=True,
    )

    # --- Build KDTree over rotated vertices ---
    tree = cKDTree(rotated_vertices)

    # --- Group by (trial_id, single_touch_id) and build TouchEvents ---
    t = time.perf_counter()
    touches_by_trial: dict[int, list[TouchEvent]] = {}
    n_skipped_empty = 0
    n_skipped_distance = 0

    group_keys = ["trial_id", "single_touch_id"]
    for (trial_id, single_touch_id), group_df in df.groupby(group_keys, sort=True):
        trial_id = int(trial_id)
        single_touch_id = int(single_touch_id)

        # Deduplicate consecutive identical contact_points strings (30Hz frames).
        cp_strings = group_df["contact_points"].fillna("[]").values
        spikes_arr = group_df["Nerve_spike"].to_numpy(dtype=bool)
        gesture_type = str(group_df["gesture_type"].iloc[0])

        n_rows = len(cp_strings)
        # Build change mask for deduplication.
        change_mask = np.empty(n_rows, dtype=bool)
        change_mask[0] = True
        change_mask[1:] = cp_strings[1:] != cp_strings[:-1]
        unique_indices = np.where(change_mask)[0]
        n_unique = len(unique_indices)

        # Run lengths: how many 1kHz rows belong to each unique 30Hz frame.
        run_lengths = np.empty(n_unique, dtype=np.int64)
        run_lengths[:-1] = np.diff(unique_indices)
        run_lengths[-1] = n_rows - unique_indices[-1]

        unique_cp_strings = cp_strings[unique_indices]

        # Aggregate per-frame spikes: any spike in the 1kHz run for each frame.
        frame_spikes = np.empty(n_unique, dtype=bool)
        for fi in range(n_unique):
            start = unique_indices[fi]
            end = start + int(run_lengths[fi])
            frame_spikes[fi] = bool(spikes_arr[start:end].any())

        # Parse contact coords for each unique frame, rotate and distance-filter.
        frame_contact_pts: list[np.ndarray] = []
        frame_vertex_indices: list[np.ndarray] = []
        valid_frame_spikes: list[bool] = []

        for fi, s in enumerate(unique_cp_strings):
            matches = _bracket_re.findall(str(s))
            if not matches:
                continue

            raw_coords: list[list[float]] = []
            for m in matches:
                parts = m.split()
                if len(parts) == 3:
                    try:
                        raw_coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        pass

            if not raw_coords:
                continue

            pts = np.array(raw_coords, dtype=np.float64)
            nan_mask = np.any(np.isnan(pts), axis=1)
            pts = pts[~nan_mask]
            if len(pts) == 0:
                continue

            # Rotate contact points to tangent plane.
            rotated_pts = (rotation @ pts.T).T

            # Apply 15mm distance filter.
            distances, vertex_idx = tree.query(rotated_pts)
            bad = distances > _DISTANCE_THRESHOLD_MM
            if bad.any():
                n_skipped_distance += int(bad.sum())
                rotated_pts = rotated_pts[~bad]
                vertex_idx = vertex_idx[~bad]

            if len(rotated_pts) == 0:
                continue

            frame_contact_pts.append(rotated_pts)
            frame_vertex_indices.append(vertex_idx.astype(np.int64))
            valid_frame_spikes.append(bool(frame_spikes[fi]))

        if not frame_contact_pts:
            n_skipped_empty += 1
            logger.warning(
                "load_playback_data: touch (trial=%d, touch=%d) has zero valid frames "
                "after dedup+filtering — skipping",
                trial_id, single_touch_id,
            )
            continue

        event = TouchEvent(
            trial_id=trial_id,
            single_touch_id=single_touch_id,
            gesture_type=gesture_type,
            frame_contact_pts=frame_contact_pts,
            frame_vertex_indices=frame_vertex_indices,
            frame_spikes=np.array(valid_frame_spikes, dtype=bool),
        )
        touches_by_trial.setdefault(trial_id, []).append(event)

    print(
        f"{_ts()} | [Touch Playback] [{tag}]   grouping+parse: {time.perf_counter() - t:.1f}s",
        flush=True,
    )

    if n_skipped_empty > 0:
        logger.warning(
            "load_playback_data: skipped %d touch(es) with zero valid frames", n_skipped_empty
        )
    if n_skipped_distance > 0:
        logger.warning(
            "load_playback_data: dropped %d contact point(s) with nearest-vertex "
            "distance > %.0fmm",
            n_skipped_distance, _DISTANCE_THRESHOLD_MM,
        )

    if not touches_by_trial:
        raise ValueError(
            f"load_playback_data: no valid touch events remain after filtering "
            f"for {series_csv_path}"
        )

    # Sort within each trial by single_touch_id (groupby sort=True already orders
    # the keys, but we ensure the list is in order for deterministic access).
    for tid in touches_by_trial:
        touches_by_trial[tid].sort(key=lambda e: e.single_touch_id)

    trial_ids = sorted(touches_by_trial.keys())

    session_data = PlaybackSessionData(
        forearm_vertices=rotated_vertices,
        tangent_rotation=rotation,
    )

    result = PlaybackData(
        session_data=session_data,
        trial_ids=trial_ids,
        touches_by_trial=touches_by_trial,
    )

    # --- Save cache ---
    t = time.perf_counter()
    _save_playback_cache(series_csv_path, result)
    print(
        f"{_ts()} | [Touch Playback] [{tag}]   cache saved: {time.perf_counter() - t:.1f}s",
        flush=True,
    )

    total_touches = sum(len(v) for v in touches_by_trial.values())
    print(
        f"{_ts()} | [Touch Playback] [{tag}]: done in "
        f"{time.perf_counter() - t_session:.1f}s  "
        f"trials={len(trial_ids)}  touches={total_touches}",
        flush=True,
    )
    return result
