"""Per-touch population data model and loader for the Touch Population Explorer GUI.

Groups 1kHz frame-level contact points, IFF, and spike values by single touch,
applies tangent-plane rotation and KDTree vertex snapping, and stores the result
in flat numpy arrays for vectorised heatmap aggregation.
"""

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

_CACHE_SCHEMA_VERSION = 2

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
# Data model
# ------------------------------------------------------------------

@dataclass
class PopulationData:
    """All-numpy data model for the Touch Population Explorer.

    Per-touch arrays have length T (one entry per single touch).
    Per-contact-point arrays have length C (all touches concatenated).
    """

    # Session geometry
    forearm_vertices: np.ndarray        # (V, 3) rotated mesh
    tangent_rotation: np.ndarray        # (3, 3)

    # Per-touch arrays (length T = n_touches)
    gesture_types: np.ndarray           # (T,) object — gesture type strings
    spike_elicited: np.ndarray          # (T,) bool

    # Features from Stage 3 CSVs (optional; empty when not available)
    feature_names: list                  # list[str], length F
    feature_matrix: np.ndarray          # (T, F) float64

    # Per-contact-point flat arrays (all touches concatenated, length C)
    cp_vertex_idx: np.ndarray           # (C,) int64 — forearm vertex per contact pt
    cp_touch_idx: np.ndarray            # (C,) int64 — owning touch index (0..T-1)
    cp_iff: np.ndarray                  # (C,) float64 — IFF at that frame
    cp_spike: np.ndarray                # (C,) bool — spike at that frame

    def get_feature_array(self, name: str) -> np.ndarray:
        """Return the per-touch array for *name*.

        Names are looked up in ``feature_names`` (sourced from Stage 3 CSVs).
        Raises ``KeyError`` if *name* is not recognised.
        """
        if name in self.feature_names:
            col = self.feature_names.index(name)
            return self.feature_matrix[:, col]
        raise KeyError(
            f"get_feature_array: unknown feature '{name}'. "
            f"Available: {self.feature_names if self.feature_names else '(none — run Stage 3 feature extraction)'}"
        )


# ------------------------------------------------------------------
# Sidecar cache helpers
# ------------------------------------------------------------------

def _population_cache_path(series_csv_path: Path) -> Path:
    return series_csv_path.parent / f"{series_csv_path.stem}_population_cache.npz"


def _save_population_cache(series_csv_path: Path, data: PopulationData) -> None:
    """Persist *data* to a compressed .npz sidecar next to *series_csv_path*.

    gesture_types (object array of strings) are encoded as integer codes + labels
    so they survive the numpy .npz round-trip.  Write failures are logged as
    warnings and not raised so callers remain on the happy path.
    """
    cache_path = _population_cache_path(series_csv_path)

    unique_labels, codes = np.unique(data.gesture_types, return_inverse=True)
    feature_names_arr = np.array(data.feature_names, dtype=str)

    try:
        np.savez_compressed(
            cache_path,
            cache_schema_version=np.array(_CACHE_SCHEMA_VERSION, dtype=np.int64),
            forearm_vertices=data.forearm_vertices,
            tangent_rotation=data.tangent_rotation,
            gesture_type_codes=codes.astype(np.int32),
            gesture_type_labels=np.array(unique_labels, dtype=str),
            spike_elicited=data.spike_elicited,
            feature_names=feature_names_arr,
            feature_matrix=data.feature_matrix,
            cp_vertex_idx=data.cp_vertex_idx,
            cp_touch_idx=data.cp_touch_idx,
            cp_iff=data.cp_iff,
            cp_spike=data.cp_spike,
        )
    except Exception as exc:
        logger.warning(
            "_save_population_cache: could not write cache %s — %s", cache_path, exc
        )


def _load_population_cache(
    series_csv_path: Path,
    forearm_ply_path: Path,
) -> Optional[PopulationData]:
    """Load the .npz sidecar cache for *series_csv_path* when it is fresh.

    Returns a ``PopulationData`` on a valid cache hit, or ``None`` when:
    - the cache file does not exist,
    - the cache is older than either source file (mtime check),
    - the schema version does not match.

    Raises ``ValueError`` on shape mismatches (corrupt cache).
    """
    cache_path = _population_cache_path(series_csv_path)

    if not cache_path.exists():
        return None

    cache_mtime = cache_path.stat().st_mtime
    csv_mtime = series_csv_path.stat().st_mtime
    ply_mtime = forearm_ply_path.stat().st_mtime

    if cache_mtime < max(csv_mtime, ply_mtime):
        logger.debug(
            "_load_population_cache: stale cache for %s — recomputing",
            series_csv_path.name,
        )
        return None

    npz = np.load(cache_path, allow_pickle=True)

    if "cache_schema_version" not in npz or int(npz["cache_schema_version"]) != _CACHE_SCHEMA_VERSION:
        logger.debug(
            "_load_population_cache: schema version mismatch for %s — recomputing",
            series_csv_path.name,
        )
        return None

    required_keys = (
        "forearm_vertices", "tangent_rotation",
        "gesture_type_codes", "gesture_type_labels",
        "spike_elicited",
        "feature_names", "feature_matrix",
        "cp_vertex_idx", "cp_touch_idx", "cp_iff", "cp_spike",
    )
    missing = [k for k in required_keys if k not in npz]
    if missing:
        logger.debug(
            "_load_population_cache: old-format cache (missing %s) for %s — recomputing",
            missing, series_csv_path.name,
        )
        return None

    forearm_vertices = npz["forearm_vertices"]
    tangent_rotation = npz["tangent_rotation"]
    gesture_type_codes = npz["gesture_type_codes"]
    gesture_type_labels = npz["gesture_type_labels"]
    spike_elicited = npz["spike_elicited"]
    feature_names_arr = npz["feature_names"]
    feature_matrix = npz["feature_matrix"]
    cp_vertex_idx = npz["cp_vertex_idx"]
    cp_touch_idx = npz["cp_touch_idx"]
    cp_iff = npz["cp_iff"]
    cp_spike = npz["cp_spike"]

    if forearm_vertices.ndim != 2 or forearm_vertices.shape[1] != 3:
        raise ValueError(
            f"_load_population_cache: 'forearm_vertices' has shape "
            f"{forearm_vertices.shape} (expected (V, 3)): {cache_path}"
        )
    if tangent_rotation.ndim != 2 or tangent_rotation.shape != (3, 3):
        raise ValueError(
            f"_load_population_cache: 'tangent_rotation' has shape "
            f"{tangent_rotation.shape} (expected (3, 3)): {cache_path}"
        )

    T = len(spike_elicited)
    for arr_name, arr in [
        ("gesture_type_codes", gesture_type_codes),
        ("spike_elicited", spike_elicited),
    ]:
        if arr.ndim != 1 or len(arr) != T:
            raise ValueError(
                f"_load_population_cache: '{arr_name}' has shape {arr.shape}, "
                f"expected ({T},): {cache_path}"
            )

    C = len(cp_vertex_idx)
    for arr_name, arr in [
        ("cp_touch_idx", cp_touch_idx),
        ("cp_iff", cp_iff),
        ("cp_spike", cp_spike),
    ]:
        if arr.ndim != 1 or len(arr) != C:
            raise ValueError(
                f"_load_population_cache: '{arr_name}' has shape {arr.shape}, "
                f"expected ({C},): {cache_path}"
            )

    F = len(feature_names_arr)
    if feature_matrix.ndim != 2 or feature_matrix.shape != (T, F):
        raise ValueError(
            f"_load_population_cache: 'feature_matrix' has shape "
            f"{feature_matrix.shape} (expected ({T}, {F})): {cache_path}"
        )

    gesture_types = gesture_type_labels[gesture_type_codes]
    feature_names = [str(n) for n in feature_names_arr]

    return PopulationData(
        forearm_vertices=forearm_vertices.astype(np.float64),
        tangent_rotation=tangent_rotation.astype(np.float64),
        gesture_types=gesture_types.astype(object),
        spike_elicited=spike_elicited.astype(bool),
        feature_names=feature_names,
        feature_matrix=feature_matrix.astype(np.float64),
        cp_vertex_idx=cp_vertex_idx.astype(np.int64),
        cp_touch_idx=cp_touch_idx.astype(np.int64),
        cp_iff=cp_iff.astype(np.float64),
        cp_spike=cp_spike.astype(bool),
    )


# ------------------------------------------------------------------
# Stage 3 CSV merge
# ------------------------------------------------------------------

def _merge_stage3_features(
    touch_keys: np.ndarray,
    touch_features_dir: Path,
    series_csv_path: Path,
) -> tuple[list, np.ndarray]:
    """Scan *touch_features_dir* recursively for CSVs matching the session and merge on touch key.

    Returns ``(feature_names, feature_matrix)`` where
    ``feature_matrix`` has shape ``(T, F)`` with ``T = len(touch_keys)``
    and ``F = len(feature_names)``.

    Touch keys are matched on ``(trial_id, single_touch_id)`` columns.
    Rows in Stage 3 CSVs that do not match a touch key are silently skipped.
    On any error, logs a warning and returns an empty result rather than raising,
    because Stage 3 enrichment is explicitly optional.
    """
    session_stem = series_csv_path.stem
    T = len(touch_keys)

    candidates = list(touch_features_dir.rglob(f"{session_stem}*.csv"))
    if not candidates:
        logger.debug(
            "_merge_stage3_features: no Stage 3 CSVs matching '%s' in %s",
            session_stem, touch_features_dir,
        )
        return [], np.empty((T, 0), dtype=np.float64)

    merged_dfs = []
    for csv_path in candidates:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.warning(
                "_merge_stage3_features: could not read %s — %s", csv_path, exc
            )
            continue

        if "trial_id" not in df.columns or "single_touch_id" not in df.columns:
            logger.warning(
                "_merge_stage3_features: skipping %s — missing trial_id or single_touch_id",
                csv_path.name,
            )
            continue

        merged_dfs.append(df)

    if not merged_dfs:
        return [], np.empty((T, 0), dtype=np.float64)

    try:
        combined = pd.concat(merged_dfs, ignore_index=True)
    except Exception as exc:
        logger.warning("_merge_stage3_features: concat failed — %s", exc)
        return [], np.empty((T, 0), dtype=np.float64)

    key_cols = {"trial_id", "single_touch_id"}
    feature_cols = [c for c in combined.columns if c not in key_cols]
    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(combined[c])
    ]
    if not numeric_cols:
        return [], np.empty((T, 0), dtype=np.float64)

    # Build lookup: (trial_id, single_touch_id) -> row index in combined.
    combined = combined.drop_duplicates(subset=["trial_id", "single_touch_id"])
    combined = combined.set_index(["trial_id", "single_touch_id"])

    matrix = np.full((T, len(numeric_cols)), np.nan, dtype=np.float64)
    for ti in range(T):
        key = (int(touch_keys[ti, 0]), int(touch_keys[ti, 1]))
        if key in combined.index:
            row = combined.loc[key]
            for ci, col in enumerate(numeric_cols):
                val = row[col]
                if pd.notna(val):
                    matrix[ti, ci] = float(val)

    return numeric_cols, matrix


# ------------------------------------------------------------------
# Main loader
# ------------------------------------------------------------------

def load_population_data(
    series_csv_path: Path,
    forearm_ply_path: Path,
    touch_features_dir: Optional[Path] = None,
) -> PopulationData:
    """Load per-touch population data from *series_csv_path* + *forearm_ply_path*.

    Applies 30Hz-to-1kHz deduplication when parsing contact_points, then
    tangent-plane rotation and KDTree vertex snapping (15mm threshold) on
    the unique contact positions.  Per-touch scalars (spike_elicited) are
    derived from frame data within each touch group.

    When *touch_features_dir* is provided, scans recursively for matching
    Stage 3 CSVs and merges numeric feature columns into ``feature_matrix``.

    Results are cached in a ``.npz`` sidecar next to *series_csv_path*.
    Subsequent calls with an up-to-date cache skip all computation.

    Raises ``ValueError`` on missing required columns, empty data after
    filtering, or forearm loading failure.
    """
    t_session = time.perf_counter()

    cached = _load_population_cache(series_csv_path, forearm_ply_path)
    if cached is not None:
        print(
            f"{_ts()} | [Population] [{series_csv_path.stem}]: cache hit "
            f"({time.perf_counter() - t_session:.2f}s)",
            flush=True,
        )
        return cached

    tag = series_csv_path.stem
    print(f"{_ts()} | [Population] [{tag}]: computing (cache miss)...", flush=True)

    # --- Read CSV ---
    t = time.perf_counter()
    df = pd.read_csv(series_csv_path)
    print(
        f"{_ts()} | [Population] [{tag}]   CSV read: {time.perf_counter() - t:.1f}s  "
        f"rows={len(df)}",
        flush=True,
    )

    missing_cols = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"load_population_data: required column(s) missing from {series_csv_path}: "
            f"{missing_cols}"
        )

    # Filter rows: keep only real touches (trial_id > 0 AND single_touch_id > 0).
    df = df[(df["trial_id"] > 0) & (df["single_touch_id"] > 0)].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"load_population_data: no rows with trial_id > 0 and single_touch_id > 0 "
            f"in {series_csv_path}"
        )

    nan_block_mask = df["block_order_id"].isna()
    if nan_block_mask.any():
        raise ValueError(
            f"load_population_data: block_order_id is NaN for {nan_block_mask.sum()} rows "
            f"with trial_id > 0 and single_touch_id > 0 in {series_csv_path}"
        )

    print(
        f"{_ts()} | [Population] [{tag}]   after filter: {len(df)} rows", flush=True
    )

    # --- Load forearm mesh ---
    t = time.perf_counter()
    print(f"{_ts()} | [Population] [{tag}]   loading forearm mesh...", flush=True)
    vertices = load_forearm_vertices(forearm_ply_path)
    if vertices is None:
        raise ValueError(
            f"load_population_data: could not load forearm vertices from {forearm_ply_path}"
        )
    print(
        f"{_ts()} | [Population] [{tag}]   forearm mesh: {time.perf_counter() - t:.1f}s  "
        f"verts={len(vertices)}",
        flush=True,
    )

    # --- Group by (block_order_id, trial_id, single_touch_id) ---
    t = time.perf_counter()
    group_keys = ["block_order_id", "trial_id", "single_touch_id"]

    # Collect per-touch scalars and raw contact point strings.
    touch_records = []
    for (block_order_id, trial_id, single_touch_id), grp in df.groupby(group_keys, sort=True):
        cp_strings = grp["contact_points"].fillna("[]").values
        spikes_arr = grp["Nerve_spike"].to_numpy(dtype=bool)
        iff_arr = grp["Nerve_freq"].to_numpy(dtype=np.float64)
        gesture_type = str(grp["gesture_type"].iloc[0])

        touch_records.append({
            "trial_id": int(trial_id),
            "single_touch_id": int(single_touch_id),
            "gesture_type": gesture_type,
            "spike_elicited": bool(spikes_arr.any()),
            "cp_strings": cp_strings,
            "spikes_arr": spikes_arr,
            "iff_arr": iff_arr,
        })

    print(
        f"{_ts()} | [Population] [{tag}]   grouping: {time.perf_counter() - t:.1f}s  "
        f"touches={len(touch_records)}",
        flush=True,
    )

    if not touch_records:
        raise ValueError(
            f"load_population_data: no touch groups found after filtering "
            f"for {series_csv_path}"
        )

    T = len(touch_records)

    # --- Parse contact points with 30Hz deduplication (unique-frame-only parsing) ---
    # For each touch, deduplicate consecutive identical cp_strings (30Hz run-length
    # encoded within 1kHz rows), parse unique frames only, then record per-contact-point
    # IFF/spike by expanding back to all 1kHz rows.
    t = time.perf_counter()

    all_cp_vertex_idx_parts: list[np.ndarray] = []
    all_cp_touch_idx_parts: list[np.ndarray] = []
    all_cp_iff_parts: list[np.ndarray] = []
    all_cp_spike_parts: list[np.ndarray] = []

    # We collect all unique 3D contact points across all touches for a single
    # global KDTree query + tangent plane rotation (requires the centroid to be
    # known first).  We record offsets so we can scatter vertex indices back.
    global_unique_pts_parts: list[np.ndarray] = []    # (Ku, 3) per unique frame
    global_frame_touch_idx: list[int] = []             # touch index per unique frame
    global_frame_local_idx: list[int] = []             # frame index within touch
    global_pts_per_unique_frame: list[int] = []        # n_pts for each unique frame

    # Per-touch metadata needed later for contact-point expansion.
    per_touch_n_unique: list[int] = []
    per_touch_run_lengths: list[np.ndarray] = []
    per_touch_unique_frame_start: list[int] = []       # index into global_unique_* lists
    per_touch_unique_pts_per_frame: list[np.ndarray] = []

    global_unique_frame_count = 0

    for ti, rec in enumerate(touch_records):
        cp_strings = rec["cp_strings"]
        n_rows = len(cp_strings)

        # Deduplicate consecutive identical strings.
        change_mask = np.empty(n_rows, dtype=bool)
        change_mask[0] = True
        change_mask[1:] = cp_strings[1:] != cp_strings[:-1]
        unique_indices = np.where(change_mask)[0]
        n_unique = len(unique_indices)

        run_lengths = np.empty(n_unique, dtype=np.int64)
        run_lengths[:-1] = np.diff(unique_indices)
        run_lengths[-1] = n_rows - unique_indices[-1]

        unique_cp_raw = cp_strings[unique_indices]

        per_touch_n_unique.append(n_unique)
        per_touch_run_lengths.append(run_lengths)
        per_touch_unique_frame_start.append(global_unique_frame_count)

        unique_pts_per_frame = np.zeros(n_unique, dtype=np.int64)
        for ui, s in enumerate(unique_cp_raw):
            pts_list: list[list[float]] = []
            for m in _bracket_re.findall(str(s)):
                parts = m.split()
                if len(parts) == 3:
                    try:
                        pts_list.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        pass
            if pts_list:
                pts_arr = np.array(pts_list, dtype=np.float64)
                global_unique_pts_parts.append(pts_arr)
                n_pts = len(pts_arr)
            else:
                n_pts = 0
            unique_pts_per_frame[ui] = n_pts
            global_frame_touch_idx.append(ti)
            global_frame_local_idx.append(ui)
            global_pts_per_unique_frame.append(n_pts)
            global_unique_frame_count += 1

        per_touch_unique_pts_per_frame.append(unique_pts_per_frame)

    print(
        f"{_ts()} | [Population] [{tag}]   contact parse: {time.perf_counter() - t:.1f}s  "
        f"unique_frames={global_unique_frame_count}",
        flush=True,
    )

    if not global_unique_pts_parts:
        raise ValueError(
            f"load_population_data: no contact point coordinates parsed from {series_csv_path}"
        )

    all_pts = np.concatenate(global_unique_pts_parts, axis=0)

    nan_mask = np.any(np.isnan(all_pts), axis=1)
    if nan_mask.any():
        logger.warning(
            "load_population_data: dropping %d contact point(s) with NaN coordinates",
            nan_mask.sum(),
        )
        all_pts = all_pts[~nan_mask]
        if len(all_pts) == 0:
            raise ValueError(
                f"load_population_data: all contact points have NaN coordinates in "
                f"{series_csv_path}"
            )

    # --- Tangent plane rotation ---
    t = time.perf_counter()
    contact_centroid = all_pts.mean(axis=0)
    rotation = compute_tangent_plane_rotation(vertices, contact_centroid)
    if rotation is None:
        raise ValueError(
            f"load_population_data: compute_tangent_plane_rotation returned None "
            f"for {forearm_ply_path}"
        )
    rotated_vertices = (rotation @ vertices.T).T
    rotated_contacts = (rotation @ all_pts.T).T
    print(
        f"{_ts()} | [Population] [{tag}]   rotation: {time.perf_counter() - t:.1f}s",
        flush=True,
    )

    # --- KDTree vertex snapping with 15mm threshold ---
    t_kd = time.perf_counter()
    tree = cKDTree(rotated_vertices)
    distances, vertex_indices = tree.query(rotated_contacts)
    print(
        f"{_ts()} | [Population] [{tag}]   KDTree: {time.perf_counter() - t_kd:.1f}s  "
        f"queries={len(rotated_contacts)}",
        flush=True,
    )

    bad_pts = distances > 15.0
    if bad_pts.any():
        logger.warning(
            "load_population_data: dropping %d contact point(s) with nearest-vertex "
            "distance > 15mm (max=%.2f mm)",
            bad_pts.sum(), distances[bad_pts].max(),
        )

    # Build a per-unique-frame array mapping unique-frame index to its vertex block.
    # global_unique_pts_parts may exclude frames that had zero pts; we need to
    # reconstruct per-unique-frame vertex arrays from the flat vertex_indices,
    # accounting for NaN-dropped and distance-filtered points.
    # Strategy: walk the flat arrays using per-frame sizes (before filtering),
    # mark filtered points as invalid, then rebuild per-frame vertex lists.

    # Sizes of each unique frame (including ones with 0 pts).
    pts_per_uf = np.array(global_pts_per_unique_frame, dtype=np.int64)  # length = n_unique_frames

    # Cumulative sizes only for frames with pts > 0.
    sizes_with_pts = pts_per_uf[pts_per_uf > 0]
    offsets_in_flat = np.zeros(len(sizes_with_pts) + 1, dtype=np.int64)
    np.cumsum(sizes_with_pts, out=offsets_in_flat[1:])

    # For each unique frame (including zero-pt frames), record their vertex blocks.
    uf_vertex_blocks: list[Optional[np.ndarray]] = [None] * global_unique_frame_count
    flat_idx = 0
    for ufi in range(global_unique_frame_count):
        n_pts = int(pts_per_uf[ufi])
        if n_pts == 0:
            uf_vertex_blocks[ufi] = np.empty(0, dtype=np.int64)
            continue
        start = int(offsets_in_flat[flat_idx])
        end = int(offsets_in_flat[flat_idx + 1])
        vtx_block = vertex_indices[start:end].astype(np.int64)
        good_mask = ~bad_pts[start:end]
        uf_vertex_blocks[ufi] = vtx_block[good_mask]
        flat_idx += 1

    # --- Build flat cp_* arrays per touch ---
    t = time.perf_counter()

    gesture_types_arr = np.empty(T, dtype=object)
    spike_elicited_arr = np.empty(T, dtype=bool)

    for ti, rec in enumerate(touch_records):
        gesture_types_arr[ti] = rec["gesture_type"]
        spike_elicited_arr[ti] = rec["spike_elicited"]

    for ti, rec in enumerate(touch_records):
        n_unique = per_touch_n_unique[ti]
        run_lengths = per_touch_run_lengths[ti]
        unique_pts_per_frame = per_touch_unique_pts_per_frame[ti]
        uf_start = per_touch_unique_frame_start[ti]
        spikes_arr = rec["spikes_arr"]
        iff_arr = rec["iff_arr"]

        # Map each 1kHz row to a unique-frame index.
        # row_unique_frame[row] = unique frame index (local within touch)
        row_unique_frame = np.repeat(np.arange(n_unique, dtype=np.int64), run_lengths)

        # For each unique frame in this touch, expand its vertex block by run_length
        # and record IFF/spike at each cp from the corresponding 1kHz rows.
        touch_vtx_parts: list[np.ndarray] = []
        touch_iff_parts: list[np.ndarray] = []
        touch_spike_parts: list[np.ndarray] = []

        row_offset = 0
        for ui in range(n_unique):
            rl = int(run_lengths[ui])
            ufi = uf_start + ui
            vtx_block = uf_vertex_blocks[ufi]
            n_vtx = len(vtx_block)

            if n_vtx == 0:
                row_offset += rl
                continue

            # For each of the rl 1kHz rows in this run, emit n_vtx contact points.
            for row_in_run in range(rl):
                row_idx = row_offset + row_in_run
                spike_val = bool(spikes_arr[row_idx])
                iff_val = float(iff_arr[row_idx])
                touch_vtx_parts.append(vtx_block)
                touch_iff_parts.append(np.full(n_vtx, iff_val, dtype=np.float64))
                touch_spike_parts.append(np.full(n_vtx, spike_val, dtype=bool))

            row_offset += rl

        if touch_vtx_parts:
            all_cp_vertex_idx_parts.append(np.concatenate(touch_vtx_parts))
            all_cp_iff_parts.append(np.concatenate(touch_iff_parts))
            all_cp_spike_parts.append(np.concatenate(touch_spike_parts))
            all_cp_touch_idx_parts.append(
                np.full(len(all_cp_vertex_idx_parts[-1]), ti, dtype=np.int64)
            )

    print(
        f"{_ts()} | [Population] [{tag}]   cp arrays: {time.perf_counter() - t:.1f}s",
        flush=True,
    )

    if all_cp_vertex_idx_parts:
        cp_vertex_idx = np.concatenate(all_cp_vertex_idx_parts).astype(np.int64)
        cp_touch_idx = np.concatenate(all_cp_touch_idx_parts).astype(np.int64)
        cp_iff = np.concatenate(all_cp_iff_parts).astype(np.float64)
        cp_spike = np.concatenate(all_cp_spike_parts).astype(bool)
    else:
        raise ValueError(
            f"load_population_data: all contact points were dropped after distance "
            f"filtering for {series_csv_path}"
        )

    # --- Stage 3 feature merge (optional) ---
    touch_keys = np.array(
        [[rec["trial_id"], rec["single_touch_id"]] for rec in touch_records],
        dtype=np.int64,
    )

    if touch_features_dir is not None and touch_features_dir.is_dir():
        t = time.perf_counter()
        feature_names, feature_matrix = _merge_stage3_features(
            touch_keys, touch_features_dir, series_csv_path
        )
        print(
            f"{_ts()} | [Population] [{tag}]   stage3 merge: "
            f"{time.perf_counter() - t:.1f}s  features={len(feature_names)}",
            flush=True,
        )
    else:
        feature_names = []
        feature_matrix = np.empty((T, 0), dtype=np.float64)

    result = PopulationData(
        forearm_vertices=rotated_vertices,
        tangent_rotation=rotation,
        gesture_types=gesture_types_arr,
        spike_elicited=spike_elicited_arr,
        feature_names=feature_names,
        feature_matrix=feature_matrix,
        cp_vertex_idx=cp_vertex_idx,
        cp_touch_idx=cp_touch_idx,
        cp_iff=cp_iff,
        cp_spike=cp_spike,
    )

    t = time.perf_counter()
    _save_population_cache(series_csv_path, result)
    print(
        f"{_ts()} | [Population] [{tag}]   cache saved: {time.perf_counter() - t:.1f}s",
        flush=True,
    )

    print(
        f"{_ts()} | [Population] [{tag}]: done in "
        f"{time.perf_counter() - t_session:.1f}s  touches={T}  "
        f"contact_pts={len(cp_vertex_idx)}",
        flush=True,
    )
    return result
