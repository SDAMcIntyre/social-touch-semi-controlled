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
    single-touch RF maps NPZ have not changed since the cache was written.

The cached data is described by :class:`SlimUvCache`.
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

from analysis.receptive_field_mapping.data.rf_data_loader import load_forearm_vertices
from .rf_surface_utils import load_or_build_forearm_mesh
from .slim_helpers import clean_mesh, boundary_loop, flatten_slim

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
    rf_npz_mtime: float    # single-touch RF maps NPZ mtime at cache-write time
    centroid_3d: np.ndarray  # (3,) float64 — IFF-weighted centroid


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
    rf_maps_npz: Path,
    cache_path: Path | None = None,
    n_iter: int = 40,
) -> Path:
    """Build SLIM UV cache for a single session.  Raises on any failure.

    Parameters
    ----------
    forearm_ply_path:
        Path to the forearm point-cloud PLY file.
    rf_maps_npz:
        Path to the ``single_touch_rf_maps.npz`` produced by
        ``map_single_touch_rf``.  Its ``rf_data`` key must contain at least
        one touch with at least one contacted vertex carrying a nonzero IFF.
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
        If ``rf_maps_npz`` does not exist.
    ValueError
        If ``rf_maps_npz`` contains no touches (empty ``rf_data`` dict).
    ValueError
        If all aggregated per-vertex IFF values are zero (cannot compute
        weighted centroid).
    ValueError
        If the IFF-weighted centroid maps to a boundary vertex.
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

    # 4. Load single-touch RF maps NPZ.
    if not rf_maps_npz.exists():
        raise FileNotFoundError(
            f"Single-touch RF maps NPZ not found: {rf_maps_npz}\n"
            "Run 'map_single_touch_rf' before 'precompute_forearm_slim_uv'."
        )

    npz = np.load(rf_maps_npz, allow_pickle=True)
    rf_data: dict = npz["rf_data"].item()

    if not rf_data:
        raise ValueError(
            f"No touches in {rf_maps_npz} — cannot determine forearm "
            "hotspot centroid."
        )

    # 5. Load raw PLY vertices to resolve NPZ vertex indices to 3D positions.
    raw_verts = load_forearm_vertices(forearm_ply_path)
    if raw_verts is None:
        raise ValueError(
            f"load_forearm_vertices returned None for {forearm_ply_path}. "
            "The PLY may be empty or unreadable."
        )
    n_verts = len(raw_verts)

    # 6. Aggregate per-vertex mean IFF across all touches.
    #    Pattern mirrors touch_population_explorer.py:654-684.
    iff_sum = np.zeros(n_verts, dtype=np.float64)
    touch_count = np.zeros(n_verts, dtype=np.int64)

    for pairs in rf_data.values():
        for vertex_idx, mean_iff in pairs:
            idx = int(vertex_idx)
            if idx < 0 or idx >= n_verts:
                # Out-of-bounds index — skip silently (safety guard).
                continue
            iff_sum[idx] += float(mean_iff)
            touch_count[idx] += 1

    contacted_mask = touch_count > 0
    contacted_indices = np.where(contacted_mask)[0]

    if len(contacted_indices) == 0:
        raise ValueError(
            f"No contacted vertices found in {rf_maps_npz} — cannot determine "
            "forearm hotspot centroid."
        )

    per_vertex_mean_iff = iff_sum[contacted_indices] / touch_count[contacted_indices]

    # 7. Compute IFF-weighted 3D centroid.
    total_weight = per_vertex_mean_iff.sum()
    if total_weight == 0.0:
        raise ValueError(
            f"All aggregated per-vertex IFF values are zero in {rf_maps_npz}. "
            "Cannot compute IFF-weighted centroid — check that the neuron was "
            "responding during the recorded touches."
        )

    contacted_positions = raw_verts[contacted_indices]
    centroid_3d = np.average(contacted_positions, weights=per_vertex_mean_iff, axis=0)

    # 8. KDTree → nearest cleaned-mesh vertex.
    tree = KDTree(V)
    _, center_vid = tree.query(centroid_3d)
    center_vid = int(center_vid)

    # 9. Boundary loop.
    bloop = boundary_loop(F)

    # 10. Fail-fast if centroid maps to a boundary vertex.
    boundary_set = set(int(v) for v in bloop)
    if center_vid in boundary_set:
        raise ValueError(
            f"IFF-weighted centroid maps to mesh boundary vertex {center_vid}. "
            "The forearm mesh boundary does not cover the neuron hotspot — "
            "consider re-extracting the forearm PLY with a larger skin region."
        )

    # 11. SLIM flattening.
    logger.info(
        "Running SLIM (n_iter=%d, center_vid=%d) for %s ...",
        n_iter, center_vid, forearm_ply_path.name,
    )
    V, F, uv = flatten_slim(V, F, bloop, center_vid=center_vid, n_iter=n_iter)

    # 12. Re-derive center_vid and boundary after potential mesh trimming inside
    #     flatten_slim.  In the common (no-trim) case these are unchanged.
    _, center_vid = KDTree(V).query(centroid_3d)
    center_vid = int(center_vid)
    bloop = boundary_loop(F)

    # 13. Collect provenance.
    ply_mtime = forearm_ply_path.stat().st_mtime
    rf_npz_mtime = rf_maps_npz.stat().st_mtime
    phash = _ply_hash(forearm_ply_path)

    # 14. Write cache.
    np.savez(
        cache_path,
        V=V.astype(np.float64),
        F=F.astype(np.int32),
        uv=uv.astype(np.float64),
        center_vid=np.int32(center_vid),
        boundary_vid=np.int32(int(bloop[0])),
        ply_mtime=np.float64(ply_mtime),
        ply_hash=np.array(phash, dtype='U64'),
        rf_npz_mtime=np.float64(rf_npz_mtime),
        centroid_3d=centroid_3d.astype(np.float64),
    )

    logger.info("SLIM UV cache written → %s", cache_path)

    # 15. Save QC figures (300 DPI) next to the cache for visual verification.
    from .slim_qc_figures import save_slim_qc_figures
    qc_path, dist_path = save_slim_qc_figures(V, F, uv, center_vid, cache_path)
    logger.info("QC figures written → %s, %s", qc_path, dist_path)

    # 16. Return path.
    return cache_path


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

def load_slim_uv_cache(
    cache_path: Path,
    *,
    forearm_ply_path: Path | None = None,
    rf_maps_npz: Path | None = None,
) -> SlimUvCache:
    """Load cached UV + mesh data.

    Parameters
    ----------
    cache_path:
        Path to the ``.npz`` cache file.
    forearm_ply_path:
        When provided, verify that the PLY mtime and hash match the cached
        values.  Raises :class:`RuntimeError` if stale.
    rf_maps_npz:
        When provided, verify that the single-touch RF maps NPZ mtime matches
        the cached value.  Raises :class:`RuntimeError` if stale.

    Returns
    -------
    SlimUvCache

    Raises
    ------
    FileNotFoundError
        If the cache file does not exist.
    RuntimeError
        If the cache was written with the old ``spike_csv_mtime`` schema
        (pre-IFF-weighted centroid).  Delete the cache and re-run
        ``precompute_forearm_slim_uv`` to rebuild it.
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

    # 3. Old-cache guard: reject caches written before the IFF-weighted centroid
    #    migration.  A cache with spike_csv_mtime but without rf_npz_mtime was
    #    produced by the old CSV-based code.
    if "spike_csv_mtime" in data and "rf_npz_mtime" not in data:
        raise RuntimeError(
            f"Old-format SLIM UV cache detected at {cache_path}: contains "
            "'spike_csv_mtime' but not 'rf_npz_mtime'. "
            "Delete the cache and re-run 'precompute_forearm_slim_uv' to "
            "rebuild it with the IFF-weighted centroid schema."
        )

    # 4. Build dataclass.
    cache = SlimUvCache(
        V=data['V'],
        F=data['F'],
        uv=data['uv'],
        center_vid=int(data['center_vid']),
        boundary_vid=int(data['boundary_vid']),
        ply_mtime=float(data['ply_mtime']),
        ply_hash=str(data['ply_hash']),
        rf_npz_mtime=float(data['rf_npz_mtime']),
        centroid_3d=data['centroid_3d'],
    )

    # 5. Optional PLY staleness check.
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

    # 6. Optional RF maps NPZ staleness check.
    if rf_maps_npz is not None:
        current_mtime = rf_maps_npz.stat().st_mtime
        if current_mtime > cache.rf_npz_mtime + 1e-3:
            raise RuntimeError(
                f"SLIM UV cache is stale: single-touch RF maps NPZ has been "
                f"modified since the cache was written.\n"
                f"  NPZ mtime:   {current_mtime}\n"
                f"  Cache mtime: {cache.rf_npz_mtime}\n"
                f"  Cache path:  {cache_path}\n"
                "Re-run 'precompute_forearm_slim_uv' to rebuild the cache."
            )

    return cache


# ---------------------------------------------------------------------------
# UV → 3D world mapping
# ---------------------------------------------------------------------------

def uv_points_to_xyz(
    uv_points: np.ndarray,
    forearm_uv: np.ndarray,
    forearm_faces: np.ndarray,
    forearm_V: np.ndarray,
) -> np.ndarray:
    """Map UV points to 3D world (mm) via barycentric interpolation on the SLIM mesh.

    Uses matplotlib.tri.Triangulation + get_trifinder() to locate the containing
    triangle for each UV point, then interpolates forearm_V with barycentric weights.

    Raises ValueError if any UV point lies outside every triangle.
    """
    import matplotlib.tri as mtri

    uv_points = np.asarray(uv_points, dtype=np.float64)
    forearm_uv = np.asarray(forearm_uv, dtype=np.float64)
    forearm_V = np.asarray(forearm_V, dtype=np.float64)

    triang = mtri.Triangulation(forearm_uv[:, 0], forearm_uv[:, 1], forearm_faces)
    trifinder = triang.get_trifinder()

    tri_idx = trifinder(uv_points[:, 0], uv_points[:, 1])
    outside = tri_idx < 0
    if np.any(outside):
        n_out = int(outside.sum())
        raise ValueError(
            f"uv_points_to_xyz: {n_out} of {len(uv_points)} UV point(s) lie "
            "outside every triangle of the SLIM mesh. "
            "The boundary contour must stay within the UV domain."
        )

    face_vids = forearm_faces[tri_idx]  # (N, 3)
    A = forearm_uv[face_vids[:, 0]]
    B = forearm_uv[face_vids[:, 1]]
    C = forearm_uv[face_vids[:, 2]]

    v0 = B - A
    v1 = C - A
    v2 = uv_points - A

    d00 = np.einsum('ij,ij->i', v0, v0)
    d01 = np.einsum('ij,ij->i', v0, v1)
    d11 = np.einsum('ij,ij->i', v1, v1)
    d20 = np.einsum('ij,ij->i', v2, v0)
    d21 = np.einsum('ij,ij->i', v2, v1)

    denom = d00 * d11 - d01 * d01
    if np.any(np.abs(denom) < 1e-30):
        raise ValueError(
            "uv_points_to_xyz: degenerate triangle(s) in SLIM mesh (near-zero "
            "UV area). The SLIM UV map may be invalid."
        )

    lam1 = (d11 * d20 - d01 * d21) / denom
    lam2 = (d00 * d21 - d01 * d20) / denom
    lam0 = 1.0 - lam1 - lam2

    V0 = forearm_V[face_vids[:, 0]]
    V1 = forearm_V[face_vids[:, 1]]
    V2 = forearm_V[face_vids[:, 2]]

    xyz = (
        lam0[:, np.newaxis] * V0
        + lam1[:, np.newaxis] * V1
        + lam2[:, np.newaxis] * V2
    )
    return xyz.astype(np.float64)


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
