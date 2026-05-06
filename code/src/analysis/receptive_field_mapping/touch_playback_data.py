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

from .rf_data_loader import load_forearm_vertex_colors, load_forearm_vertices

logger = logging.getLogger(__name__)

_CACHE_SCHEMA_VERSION = 3

_REQUIRED_COLUMNS = (
    "contact_points",
    "block_order_id",
    "trial_id",
    "single_touch_id",
    "Nerve_spike",
    "Nerve_freq",
    "gesture_type",
)

_bracket_re = re.compile(r'\[([^\[\]]+)\]')


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ------------------------------------------------------------------
# Dataclasses
# ------------------------------------------------------------------

@dataclass
class PlaybackSessionData:
    forearm_vertices: np.ndarray              # (N, 3) raw coordinates
    forearm_vertex_colors: Optional[np.ndarray]  # (N, 3) uint8 RGB, or None


@dataclass
class TouchEvent:
    block_order_id: str
    trial_id: int
    single_touch_id: int
    gesture_type: str                       # 'tap', 'stroke_proximal', etc.
    # Per 1kHz row:
    frame_contact_pts: list                 # list of (K_i, 3) raw contact coords
    frame_vertex_indices: list              # list of (K_i,) nearest vertex per contact pt
    frame_spikes: np.ndarray               # (n_frames,) bool — per-row Nerve_spike
    frame_iff: np.ndarray                  # (n_frames,) float64 — per-row Nerve_freq (Hz)


@dataclass
class PlaybackData:
    session_data: PlaybackSessionData
    block_order_ids: list                   # sorted unique strings (by numeric value)
    trial_ids_by_block: dict                # block_order_id -> sorted list[int]
    touches_by_block_trial: dict            # (block_order_id, trial_id) -> sorted list[TouchEvent]


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

    # Collect all touches in a stable order (block → trial → touch_id).
    all_touches: list[TouchEvent] = []
    for bid in data.block_order_ids:
        for tid in data.trial_ids_by_block[bid]:
            all_touches.extend(data.touches_by_block_trial[(bid, tid)])

    n_touches = len(all_touches)

    touch_keys = np.array(
        [[t.trial_id, t.single_touch_id] for t in all_touches],
        dtype=np.int64,
    )  # (n_touches, 2)
    touch_block_ids = np.array([t.block_order_id for t in all_touches], dtype=str)

    gesture_types = np.array([t.gesture_type for t in all_touches], dtype=str)

    # frame_spikes: concatenate all (n_frames_i,) bool arrays.
    frame_spikes_counts = np.array([len(t.frame_spikes) for t in all_touches], dtype=np.int64)
    frame_spikes_data = np.concatenate(
        [t.frame_spikes for t in all_touches]
    ).astype(bool) if n_touches > 0 else np.array([], dtype=bool)

    # frame_iff: concatenate all (n_frames_i,) float64 arrays (same counts as frame_spikes).
    frame_iff_data = np.concatenate(
        [t.frame_iff for t in all_touches]
    ).astype(np.float64) if n_touches > 0 else np.array([], dtype=np.float64)

    # Contact points: deduplicated format.
    # The CSV parser already reuses the same ndarray object for consecutive
    # identical contact_points strings within a touch event.  We exploit
    # object identity (id()) to store each unique (pts, vtx) pair only once.
    #
    # cp_unique_pts     : (total_unique_pts, 3) float32 — unique contact coords
    # cp_unique_vtx     : (total_unique_pts,)   int32   — unique vertex indices
    # cp_unique_offsets : (n_groups + 1,)       int32   — cumulative group sizes
    # cp_frame_group    : (n_contact_frames,)   int32   — group index per frame
    # cp_frame_touch    : (n_contact_frames,)   int32   — touch index per frame
    # cp_frame_fi       : (n_contact_frames,)   int32   — frame-within-touch index
    seen: dict[int, int] = {}          # id(vtx_array) -> group index
    unique_pts_parts: list[np.ndarray] = []
    unique_vtx_parts: list[np.ndarray] = []
    group_sizes: list[int] = []
    frame_group_list: list[int] = []
    frame_touch_list: list[int] = []
    frame_fi_list: list[int] = []

    for ti, touch in enumerate(all_touches):
        for fi, (pts, vtx) in enumerate(zip(touch.frame_contact_pts, touch.frame_vertex_indices)):
            vtx_arr = np.asarray(vtx, dtype=np.int64)
            k = len(vtx_arr)
            if k == 0:
                continue
            obj_id = id(vtx)
            if obj_id not in seen:
                group_idx = len(seen)
                seen[obj_id] = group_idx
                unique_pts_parts.append(np.asarray(pts, dtype=np.float32))
                unique_vtx_parts.append(vtx_arr.astype(np.int32))
                group_sizes.append(k)
            else:
                group_idx = seen[obj_id]
            frame_group_list.append(group_idx)
            frame_touch_list.append(ti)
            frame_fi_list.append(fi)

    if unique_pts_parts:
        cp_unique_pts = np.concatenate(unique_pts_parts, axis=0)
        cp_unique_vtx = np.concatenate(unique_vtx_parts)
        cp_unique_offsets = np.zeros(len(group_sizes) + 1, dtype=np.int32)
        cp_unique_offsets[1:] = np.cumsum(group_sizes, dtype=np.int32)
        cp_frame_group = np.array(frame_group_list, dtype=np.int32)
        cp_frame_touch = np.array(frame_touch_list, dtype=np.int32)
        cp_frame_fi = np.array(frame_fi_list, dtype=np.int32)
    else:
        cp_unique_pts = np.empty((0, 3), dtype=np.float32)
        cp_unique_vtx = np.empty(0, dtype=np.int32)
        cp_unique_offsets = np.zeros(1, dtype=np.int32)
        cp_frame_group = np.empty(0, dtype=np.int32)
        cp_frame_touch = np.empty(0, dtype=np.int32)
        cp_frame_fi = np.empty(0, dtype=np.int32)

    optional_arrays: dict = {}
    if data.session_data.forearm_vertex_colors is not None:
        optional_arrays["forearm_vertex_colors"] = data.session_data.forearm_vertex_colors

    try:
        np.savez_compressed(
            cache_path,
            cache_schema_version=np.array(_CACHE_SCHEMA_VERSION, dtype=np.int64),
            touch_keys=touch_keys,
            touch_block_ids=touch_block_ids,
            gesture_types=gesture_types,
            frame_spikes_data=frame_spikes_data,
            frame_spikes_counts=frame_spikes_counts,
            frame_iff_data=frame_iff_data,
            cp_unique_pts=cp_unique_pts,
            cp_unique_vtx=cp_unique_vtx,
            cp_unique_offsets=cp_unique_offsets,
            cp_frame_group=cp_frame_group,
            cp_frame_touch=cp_frame_touch,
            cp_frame_fi=cp_frame_fi,
            forearm_vertices=data.session_data.forearm_vertices,
            **optional_arrays,
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

    # Validate schema version.
    if "cache_schema_version" not in npz or int(npz["cache_schema_version"]) != _CACHE_SCHEMA_VERSION:
        logger.debug(
            "_load_playback_cache: schema version mismatch for %s — recomputing",
            series_csv_path.name,
        )
        return None

    # Validate that all expected keys are present (old-format guard).
    required_keys = (
        "touch_keys", "touch_block_ids", "gesture_types",
        "frame_spikes_data", "frame_spikes_counts",
        "frame_iff_data",
        "cp_unique_pts", "cp_unique_vtx", "cp_unique_offsets",
        "cp_frame_group", "cp_frame_touch", "cp_frame_fi",
        "forearm_vertices",
    )
    missing_keys = [k for k in required_keys if k not in npz]
    if missing_keys:
        logger.debug(
            "_load_playback_cache: old-format cache (missing %s) for %s — recomputing",
            missing_keys, series_csv_path.name,
        )
        return None

    # Reject caches that still contain the old tangent_rotation key — they were
    # produced by the pre-refactor loader and must be regenerated.
    if "tangent_rotation" in npz:
        logger.debug(
            "_load_playback_cache: old-format cache (contains tangent_rotation) for %s "
            "— recomputing",
            series_csv_path.name,
        )
        return None

    touch_keys = npz["touch_keys"]
    touch_block_ids = npz["touch_block_ids"]
    gesture_types = npz["gesture_types"]
    frame_spikes_data = npz["frame_spikes_data"]
    frame_spikes_counts = npz["frame_spikes_counts"]
    frame_iff_data = npz["frame_iff_data"]
    cp_unique_pts = npz["cp_unique_pts"]
    cp_unique_vtx = npz["cp_unique_vtx"]
    cp_unique_offsets = npz["cp_unique_offsets"]
    cp_frame_group = npz["cp_frame_group"]
    cp_frame_touch = npz["cp_frame_touch"]
    cp_frame_fi = npz["cp_frame_fi"]
    forearm_vertices = npz["forearm_vertices"]
    # forearm_vertex_colors is optional — missing means PLY had no colours.
    forearm_vertex_colors: Optional[np.ndarray] = (
        npz["forearm_vertex_colors"] if "forearm_vertex_colors" in npz else None
    )

    # Shape validation — raise on corruption.
    if touch_keys.ndim != 2 or touch_keys.shape[1] != 2:
        raise ValueError(
            f"_load_playback_cache: 'touch_keys' has shape {touch_keys.shape} "
            f"(expected (n_touches, 2)): {cache_path}"
        )
    n_touches = len(touch_keys)

    if touch_block_ids.ndim != 1 or len(touch_block_ids) != n_touches:
        raise ValueError(
            f"_load_playback_cache: 'touch_block_ids' has shape {touch_block_ids.shape}, "
            f"expected ({n_touches},): {cache_path}"
        )

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
    if frame_iff_data.ndim != 1 or len(frame_iff_data) != expected_spikes_len:
        raise ValueError(
            f"_load_playback_cache: 'frame_iff_data' has length "
            f"{len(frame_iff_data)}, expected {expected_spikes_len}: {cache_path}"
        )
    if cp_unique_pts.ndim != 2 or cp_unique_pts.shape[1] != 3:
        raise ValueError(
            f"_load_playback_cache: 'cp_unique_pts' has shape {cp_unique_pts.shape} "
            f"(expected (total_unique_pts, 3)): {cache_path}"
        )
    total_unique_pts = len(cp_unique_pts)
    if cp_unique_vtx.ndim != 1 or len(cp_unique_vtx) != total_unique_pts:
        raise ValueError(
            f"_load_playback_cache: 'cp_unique_vtx' has shape {cp_unique_vtx.shape}, "
            f"expected ({total_unique_pts},): {cache_path}"
        )
    if cp_unique_offsets.ndim != 1 or len(cp_unique_offsets) < 1:
        raise ValueError(
            f"_load_playback_cache: 'cp_unique_offsets' has shape "
            f"{cp_unique_offsets.shape} (expected (n_groups+1,)): {cache_path}"
        )
    n_contact_frames = len(cp_frame_group)
    for arr_name, arr in [
        ("cp_frame_touch", cp_frame_touch),
        ("cp_frame_fi", cp_frame_fi),
    ]:
        if arr.ndim != 1 or len(arr) != n_contact_frames:
            raise ValueError(
                f"_load_playback_cache: '{arr_name}' has shape {arr.shape}, "
                f"expected ({n_contact_frames},): {cache_path}"
            )
    if forearm_vertices.ndim != 2 or forearm_vertices.shape[1] != 3:
        raise ValueError(
            f"_load_playback_cache: 'forearm_vertices' has shape "
            f"{forearm_vertices.shape} (expected (N, 3)): {cache_path}"
        )

    # Build a lookup: frame_lookup[ti][fi] = group_idx
    # so we can reconstruct per-frame pts/vtx slices efficiently.
    frame_lookup: dict[int, dict[int, int]] = {}
    for i in range(n_contact_frames):
        ti_val = int(cp_frame_touch[i])
        fi_val = int(cp_frame_fi[i])
        g_val = int(cp_frame_group[i])
        frame_lookup.setdefault(ti_val, {})[fi_val] = g_val

    # Reconstruct touch events.
    frames_offset = 0
    touches_by_block_trial: dict[tuple[str, int], list[TouchEvent]] = {}

    for ti in range(n_touches):
        block_order_id = str(touch_block_ids[ti])
        trial_id = int(touch_keys[ti, 0])
        single_touch_id = int(touch_keys[ti, 1])
        gesture = str(gesture_types[ti])
        n_frames = int(frame_spikes_counts[ti])
        spikes = frame_spikes_data[frames_offset: frames_offset + n_frames].astype(bool)
        iff = frame_iff_data[frames_offset: frames_offset + n_frames].astype(np.float64)
        frames_offset += n_frames

        # Reconstruct per-frame contact pts and vertex indices via group offsets.
        ti_lookup = frame_lookup.get(ti, {})
        frame_pts_list: list[np.ndarray] = []
        frame_vtx_list: list[np.ndarray] = []
        for fi in range(n_frames):
            if fi in ti_lookup:
                g = ti_lookup[fi]
                start = int(cp_unique_offsets[g])
                end = int(cp_unique_offsets[g + 1])
                frame_pts_list.append(cp_unique_pts[start:end].astype(np.float64))
                frame_vtx_list.append(cp_unique_vtx[start:end].astype(np.int64))
            else:
                frame_pts_list.append(np.empty((0, 3), dtype=np.float64))
                frame_vtx_list.append(np.empty(0, dtype=np.int64))

        event = TouchEvent(
            block_order_id=block_order_id,
            trial_id=trial_id,
            single_touch_id=single_touch_id,
            gesture_type=gesture,
            frame_contact_pts=frame_pts_list,
            frame_vertex_indices=frame_vtx_list,
            frame_spikes=spikes,
            frame_iff=iff,
        )
        touches_by_block_trial.setdefault((block_order_id, trial_id), []).append(event)

    for key in touches_by_block_trial:
        touches_by_block_trial[key].sort(key=lambda e: e.single_touch_id)

    trial_ids_by_block: dict[str, list[int]] = {}
    for bid, tid in touches_by_block_trial:
        trial_ids_by_block.setdefault(bid, []).append(tid)
    for bid in trial_ids_by_block:
        trial_ids_by_block[bid].sort()

    block_order_ids = sorted(trial_ids_by_block.keys(), key=int)

    session_data = PlaybackSessionData(
        forearm_vertices=forearm_vertices,
        forearm_vertex_colors=forearm_vertex_colors,
    )
    return PlaybackData(
        session_data=session_data,
        block_order_ids=block_order_ids,
        trial_ids_by_block=trial_ids_by_block,
        touches_by_block_trial=touches_by_block_trial,
    )


# ------------------------------------------------------------------
# Main loader
# ------------------------------------------------------------------

def load_playback_data(
    series_csv_path: Path,
    forearm_ply_path: Path,
) -> PlaybackData:
    """Load per-touch data at 1kHz frame rate from *series_csv_path* + *forearm_ply_path*.

    No coordinate transforms, no deduplication, no distance filtering.
    Matches preparation_viewer_data.py exactly: each CSV row becomes one frame,
    contact_points are forward-filled within each touch group, empty frames are
    kept as np.empty((0, 3)), and Nerve_spike is taken directly per row.

    Vertex snapping (cKDTree on raw forearm vertices) is applied for heatmap
    accumulation only.

    Raises ``ValueError`` on missing required columns, empty data after
    filtering, or forearm loading failure.
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

    nan_block_mask = df["block_order_id"].isna()
    if nan_block_mask.any():
        raise ValueError(
            f"load_playback_data: block_order_id is NaN for {nan_block_mask.sum()} rows "
            f"with trial_id > 0 and single_touch_id > 0 in {series_csv_path}"
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
    vertex_colors = load_forearm_vertex_colors(forearm_ply_path)
    print(
        f"{_ts()} | [Touch Playback] [{tag}]   forearm mesh: {time.perf_counter() - t:.1f}s  "
        f"verts={len(vertices)}  colors={'yes' if vertex_colors is not None else 'none'}",
        flush=True,
    )

    # --- Build KDTree over raw (unrotated) vertices ---
    tree = cKDTree(vertices)

    # --- Group by (block_order_id, trial_id, single_touch_id) and build TouchEvents ---
    t = time.perf_counter()
    touches_by_block_trial: dict[tuple[str, int], list[TouchEvent]] = {}

    group_keys = ["block_order_id", "trial_id", "single_touch_id"]
    for (block_order_id, trial_id, single_touch_id), group_df in df.groupby(group_keys, sort=True):
        block_order_id = str(block_order_id)
        trial_id = int(trial_id)
        single_touch_id = int(single_touch_id)

        cp_strings = group_df["contact_points"].values
        spikes_arr = group_df["Nerve_spike"].to_numpy(dtype=bool)
        iff_arr = group_df["Nerve_freq"].to_numpy(dtype=np.float64)
        gesture_type = str(group_df["gesture_type"].iloc[0])

        # Row-by-row parsing matching preparation_viewer_data.py exactly.
        # Optimization: reuse parsed result for consecutive identical strings.
        frame_contact_pts: list[np.ndarray] = []
        frame_vertex_indices: list[np.ndarray] = []
        frame_spikes: list[bool] = []
        frame_iff: list[float] = []

        prev_string: Optional[str] = None
        prev_pts: np.ndarray = np.empty((0, 3), dtype=np.float64)
        prev_vtx: np.ndarray = np.empty(0, dtype=np.int64)

        for row_idx, s in enumerate(cp_strings):
            s_str = str(s)
            if s_str == prev_string:
                # Reuse parsed result from previous identical string.
                frame_contact_pts.append(prev_pts)
                frame_vertex_indices.append(prev_vtx)
            else:
                matches = _bracket_re.findall(s_str)
                pts: list[list[float]] = []
                for m in matches:
                    parts = m.split()
                    if len(parts) == 3:
                        try:
                            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        except ValueError:
                            pass

                if pts:
                    pts_arr = np.array(pts, dtype=np.float64)
                    _, vtx_idx = tree.query(pts_arr)
                    prev_pts = pts_arr
                    prev_vtx = vtx_idx.astype(np.int64)
                else:
                    prev_pts = np.empty((0, 3), dtype=np.float64)
                    prev_vtx = np.empty(0, dtype=np.int64)

                prev_string = s_str
                frame_contact_pts.append(prev_pts)
                frame_vertex_indices.append(prev_vtx)

            frame_spikes.append(bool(spikes_arr[row_idx]))
            frame_iff.append(float(iff_arr[row_idx]))

        event = TouchEvent(
            block_order_id=block_order_id,
            trial_id=trial_id,
            single_touch_id=single_touch_id,
            gesture_type=gesture_type,
            frame_contact_pts=frame_contact_pts,
            frame_vertex_indices=frame_vertex_indices,
            frame_spikes=np.array(frame_spikes, dtype=bool),
            frame_iff=np.array(frame_iff, dtype=np.float64),
        )
        touches_by_block_trial.setdefault((block_order_id, trial_id), []).append(event)

    print(
        f"{_ts()} | [Touch Playback] [{tag}]   grouping+parse: {time.perf_counter() - t:.1f}s",
        flush=True,
    )

    if not touches_by_block_trial:
        raise ValueError(
            f"load_playback_data: no touch events found after filtering "
            f"for {series_csv_path}"
        )

    for key in touches_by_block_trial:
        touches_by_block_trial[key].sort(key=lambda e: e.single_touch_id)

    trial_ids_by_block: dict[str, list[int]] = {}
    for bid, tid in touches_by_block_trial:
        trial_ids_by_block.setdefault(bid, []).append(tid)
    for bid in trial_ids_by_block:
        trial_ids_by_block[bid].sort()

    block_order_ids = sorted(trial_ids_by_block.keys(), key=int)

    session_data = PlaybackSessionData(
        forearm_vertices=vertices,
        forearm_vertex_colors=vertex_colors,
    )

    result = PlaybackData(
        session_data=session_data,
        block_order_ids=block_order_ids,
        trial_ids_by_block=trial_ids_by_block,
        touches_by_block_trial=touches_by_block_trial,
    )

    # --- Save cache ---
    t = time.perf_counter()
    _save_playback_cache(series_csv_path, result)
    print(
        f"{_ts()} | [Touch Playback] [{tag}]   cache saved: {time.perf_counter() - t:.1f}s",
        flush=True,
    )

    total_touches = sum(len(v) for v in touches_by_block_trial.values())
    print(
        f"{_ts()} | [Touch Playback] [{tag}]: done in "
        f"{time.perf_counter() - t_session:.1f}s  "
        f"blocks={len(block_order_ids)}  touches={total_touches}",
        flush=True,
    )
    return result
