"""
forearm_slim_uv.py
==================

Per-session SLIM UV precompute and cache I/O for the production RF mapping
pipeline.

The two public functions are:

precompute_forearm_slim_uv
    Build a SLIM UV map for a single session and write it to a ``.npz`` cache
    next to the forearm PLY.  Raises loudly on any failure — no silent
    fallbacks.

load_slim_uv_cache
    Load the ``.npz`` cache and optionally verify that the source PLY and
    spike CSV have not changed since the cache was written.

The cached data is described by :class:`SlimUvCache`.
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from .rf_surface_utils import load_or_build_forearm_mesh
from ._slim_helpers import clean_mesh, boundary_loop, flatten_slim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache schema
# ---------------------------------------------------------------------------

@dataclass
class SlimUvCache:
    V: np.ndarray          # (N_v, 3) float64 — cleaned mesh vertices
    F: np.ndarray          # (N_f, 3) int32   — face indices
    uv: np.ndarray         # (N_v, 2) float64 — SLIM UV per vertex
    center_vid: int        # interior vertex at UV origin
    boundary_vid: int      # boundary anchor vertex (boundary[0])
    ply_mtime: float       # PLY mtime at cache-write time
    ply_hash: str          # SHA-256 of first 4 kB of PLY
    spike_csv_mtime: float # spike_positions.csv mtime at cache-write time
    centroid_3d: np.ndarray  # (3,) float64 — spike-weighted centroid


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ply_hash(ply_path: Path) -> str:
    """SHA-256 hex digest of the first 4 kB of the PLY file."""
    with open(ply_path, 'rb') as f:
        data = f.read(4096)
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Precompute
# ---------------------------------------------------------------------------

def precompute_forearm_slim_uv(
    forearm_ply_path: Path,
    spike_positions_csv: Path,
    cache_path: Path | None = None,
    n_iter: int = 40,
) -> Path:
    """Build SLIM UV cache for a single session.  Raises on any failure.

    Parameters
    ----------
    forearm_ply_path:
        Path to the forearm point-cloud PLY file.
    spike_positions_csv:
        Path to the ``spike_positions.csv`` produced by
        ``map_receptive_fields_simple``.  Its rows must contain at least
        ``x``, ``y``, ``z`` columns.
    cache_path:
        Destination path for the ``.npz`` cache.  Defaults to
        ``<forearm_ply_stem>_slim_uv.npz`` in the same directory.
    n_iter:
        Number of SLIM iterations (default 40).

    Returns
    -------
    Path
        The path to the written ``.npz`` cache file.

    Raises
    ------
    ValueError
        If ``load_or_build_forearm_mesh`` returns ``None`` (degenerate PLY).
    FileNotFoundError
        If ``spike_positions_csv`` does not exist.
    ValueError
        If ``spike_positions_csv`` exists but contains no rows.
    ValueError
        If the spike-weighted centroid maps to a boundary vertex.
    RuntimeError
        If the cotangent-harmonic initialisation has flipped triangles
        (propagated from ``flatten_slim``).
    """
    # 1. Default cache path.
    if cache_path is None:
        cache_path = forearm_ply_path.with_name(
            forearm_ply_path.stem + "_slim_uv.npz"
        )

    # 2. Build mesh.
    raw_mesh = load_or_build_forearm_mesh(forearm_ply_path)
    if raw_mesh is None:
        raise ValueError(
            f"load_or_build_forearm_mesh returned None for {forearm_ply_path}. "
            "The PLY may be empty, too sparse, or not a forearm segmentation."
        )

    # 3. Clean mesh.
    V, F = clean_mesh(raw_mesh)
    logger.info(
        "Cleaned mesh: %d vertices, %d faces (source: %s)",
        V.shape[0], F.shape[0], forearm_ply_path.name,
    )

    # 4. Read spike_positions.csv.
    if not spike_positions_csv.exists():
        raise FileNotFoundError(
            f"spike_positions.csv not found: {spike_positions_csv}\n"
            "Run 'map_receptive_fields_simple' before 'precompute_forearm_slim_uv'."
        )

    spikes = pd.read_csv(spike_positions_csv)
    if spikes.empty:
        raise ValueError(
            f"No spikes in {spike_positions_csv} — cannot determine forearm "
            "hotspot centroid."
        )

    # 5. Compute spike-weighted centroid.
    centroid_3d = spikes[['x', 'y', 'z']].values.mean(axis=0)

    # 6. KDTree → nearest mesh vertex.
    tree = KDTree(V)
    _, center_vid = tree.query(centroid_3d)
    center_vid = int(center_vid)

    # 7. Boundary loop.
    bloop = boundary_loop(F)

    # 8. Fail-fast if centroid maps to a boundary vertex.
    boundary_set = set(int(v) for v in bloop)
    if center_vid in boundary_set:
        raise ValueError(
            f"Spike-weighted centroid maps to mesh boundary vertex {center_vid}. "
            "The forearm mesh boundary does not cover the neuron hotspot — "
            "consider re-extracting the forearm PLY with a larger skin region."
        )

    # 9. SLIM flattening.
    logger.info(
        "Running SLIM (n_iter=%d, center_vid=%d) for %s ...",
        n_iter, center_vid, forearm_ply_path.name,
    )
    uv = flatten_slim(V, F, bloop, center_vid=center_vid, n_iter=n_iter)

    # 10. Collect provenance.
    ply_mtime = forearm_ply_path.stat().st_mtime
    spike_csv_mtime = spike_positions_csv.stat().st_mtime
    phash = _ply_hash(forearm_ply_path)

    # 11. Write cache.
    np.savez(
        cache_path,
        V=V.astype(np.float64),
        F=F.astype(np.int32),
        uv=uv.astype(np.float64),
        center_vid=np.int32(center_vid),
        boundary_vid=np.int32(int(bloop[0])),
        ply_mtime=np.float64(ply_mtime),
        ply_hash=np.array(phash, dtype='U64'),
        spike_csv_mtime=np.float64(spike_csv_mtime),
        centroid_3d=centroid_3d.astype(np.float64),
    )

    logger.info("SLIM UV cache written → %s", cache_path)

    # 12. Return path.
    return cache_path


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_slim_uv_cache(
    cache_path: Path,
    *,
    forearm_ply_path: Path | None = None,
    spike_positions_csv: Path | None = None,
) -> SlimUvCache:
    """Load cached UV + mesh data.

    Parameters
    ----------
    cache_path:
        Path to the ``.npz`` cache file.
    forearm_ply_path:
        When provided, verify that the PLY mtime and hash match the cached
        values.  Raises :class:`RuntimeError` if stale.
    spike_positions_csv:
        When provided, verify that the spike CSV mtime matches the cached
        value.  Raises :class:`RuntimeError` if stale.

    Returns
    -------
    SlimUvCache

    Raises
    ------
    FileNotFoundError
        If the cache file does not exist.
    RuntimeError
        If mtime/hash staleness check fails (inputs changed since cache).
    """
    # 1. Guard: cache must exist.
    if not cache_path.exists():
        raise FileNotFoundError(
            f"SLIM UV cache not found: {cache_path}\n"
            "Run 'precompute_forearm_slim_uv' first."
        )

    # 2. Load arrays.
    data = np.load(cache_path, allow_pickle=False)

    # 3. Build dataclass.
    cache = SlimUvCache(
        V=data['V'],
        F=data['F'],
        uv=data['uv'],
        center_vid=int(data['center_vid']),
        boundary_vid=int(data['boundary_vid']),
        ply_mtime=float(data['ply_mtime']),
        ply_hash=str(data['ply_hash']),
        spike_csv_mtime=float(data['spike_csv_mtime']),
        centroid_3d=data['centroid_3d'],
    )

    # 4. Optional PLY staleness check.
    if forearm_ply_path is not None:
        current_mtime = forearm_ply_path.stat().st_mtime
        if current_mtime > cache.ply_mtime + 1e-3:
            raise RuntimeError(
                f"SLIM UV cache is stale: forearm PLY has been modified since the "
                f"cache was written.\n"
                f"  PLY mtime:   {current_mtime}\n"
                f"  Cache mtime: {cache.ply_mtime}\n"
                f"  Cache path:  {cache_path}\n"
                "Re-run 'precompute_forearm_slim_uv' to rebuild the cache."
            )
        current_hash = _ply_hash(forearm_ply_path)
        if current_hash != cache.ply_hash:
            raise RuntimeError(
                f"SLIM UV cache is stale: forearm PLY content has changed "
                f"(hash mismatch) since the cache was written.\n"
                f"  Current hash: {current_hash}\n"
                f"  Cached hash:  {cache.ply_hash}\n"
                f"  Cache path:   {cache_path}\n"
                "Re-run 'precompute_forearm_slim_uv' to rebuild the cache."
            )

    # 5. Optional spike CSV staleness check.
    if spike_positions_csv is not None:
        current_mtime = spike_positions_csv.stat().st_mtime
        if current_mtime > cache.spike_csv_mtime + 1e-3:
            raise RuntimeError(
                f"SLIM UV cache is stale: spike_positions.csv has been modified "
                f"since the cache was written.\n"
                f"  CSV mtime:   {current_mtime}\n"
                f"  Cache mtime: {cache.spike_csv_mtime}\n"
                f"  Cache path:  {cache_path}\n"
                "Re-run 'precompute_forearm_slim_uv' to rebuild the cache."
            )

    return cache


# ---------------------------------------------------------------------------
# Barycentric UV lookup
# ---------------------------------------------------------------------------

def barycentric_uv_lookup(
    cache: 'SlimUvCache',
    query_points_3d: np.ndarray,
) -> np.ndarray:
    """For each query point, find the nearest face and barycentric-interpolate UV.

    Parameters
    ----------
    cache:
        Loaded SLIM UV cache from ``load_slim_uv_cache()``.
    query_points_3d:
        (N, 3) array of 3D query positions in mesh coordinates.

    Returns
    -------
    np.ndarray
        (N, 2) float64 UV coordinates.
    """
    V = cache.V
    F = cache.F
    uv = cache.uv
    P = np.asarray(query_points_3d, dtype=np.float64)

    # Face centroids — KDTree finds the nearest face for each query point.
    face_centroids = V[F].mean(axis=1)          # (M, 3)
    tree = KDTree(face_centroids)
    _, face_idx = tree.query(P)                  # (N,)

    # Face vertices
    A = V[F[face_idx, 0]]                        # (N, 3)
    B = V[F[face_idx, 1]]                        # (N, 3)
    C = V[F[face_idx, 2]]                        # (N, 3)

    # Face normals (unnormalized)
    n = np.cross(B - A, C - A)                   # (N, 3)
    nn = np.einsum('ij,ij->i', n, n)             # (N,)
    degenerate = nn < 1e-20
    nn_safe = np.where(degenerate, 1.0, nn)
    n_hat = n / np.sqrt(nn_safe)[:, np.newaxis]  # (N, 3)

    # Project query points onto the face plane.
    PA = P - A
    d_plane = np.einsum('ij,ij->i', PA, n_hat)
    P_proj = P - d_plane[:, np.newaxis] * n_hat  # (N, 3)

    # Barycentric coordinates (Cramer's rule on the face plane).
    v0 = B - A
    v1 = C - A
    v2 = P_proj - A

    d00 = np.einsum('ij,ij->i', v0, v0)
    d01 = np.einsum('ij,ij->i', v0, v1)
    d11 = np.einsum('ij,ij->i', v1, v1)
    d20 = np.einsum('ij,ij->i', v2, v0)
    d21 = np.einsum('ij,ij->i', v2, v1)

    denom = d00 * d11 - d01 * d01
    bad = degenerate | (np.abs(denom) < 1e-20)
    denom_safe = np.where(bad, 1.0, denom)

    lam1 = (d11 * d20 - d01 * d21) / denom_safe
    lam2 = (d00 * d21 - d01 * d20) / denom_safe
    lam0 = 1.0 - lam1 - lam2

    # Interpolate UV.
    uv0 = uv[F[face_idx, 0]]                     # (N, 2)
    uv1 = uv[F[face_idx, 1]]                     # (N, 2)
    uv2 = uv[F[face_idx, 2]]                     # (N, 2)

    result = (
        lam0[:, np.newaxis] * uv0
        + lam1[:, np.newaxis] * uv1
        + lam2[:, np.newaxis] * uv2
    )

    # For degenerate faces, fall back to the face centroid UV.
    face_centroid_uv = (uv0 + uv1 + uv2) / 3.0
    result = np.where(bad[:, np.newaxis], face_centroid_uv, result)

    return result.astype(np.float64)
