"""
_slim_helpers.py
================

Reusable primitives for SLIM-based forearm flattening.

Extracted from sandbox scripts so both the production pipeline and the
sandbox comparison harness can import a single authoritative copy.

Functions
---------
clean_mesh         — Keep largest component, fix winding, drop orphans,
                     remove non-manifold edges and pinch vertices.
boundary_loop      — Return ordered boundary vertex indices.
canonicalise_uv    — Rigid 2D similarity: centre at origin, boundary[0] on +x.
_has_flipped_triangles — Return True when the UV map has inconsistent triangle
                         orientation.
flatten_slim       — Symmetric-Dirichlet SLIM, initialised from a cotangent-
                     harmonic map.  **Fail-fast** — raises RuntimeError if the
                     harmonic init has flipped triangles.  No Tutte fallback.
"""

import numpy as np
import igl
import open3d as o3d
import trimesh
import trimesh.repair


# ---------------------------------------------------------------------------
# Mesh cleaning
# ---------------------------------------------------------------------------

def clean_mesh(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Keep the largest connected component, fix winding, drop orphan vertices.

    Parameters
    ----------
    mesh:
        Input trimesh (may have multiple components, flipped faces, or
        unreferenced vertices).

    Returns
    -------
    V : np.ndarray
        Vertex positions, shape (N, 3), float64.
    F : np.ndarray
        Face indices (into V), shape (M, 3), int64.

    Raises
    ------
    ValueError
        If the mesh has zero faces after cleaning (completely degenerate
        input or wrong PLY file).
    """
    # Split into connected components; keep the one with the most vertices.
    components = mesh.split(only_watertight=False)
    if len(components) == 0:
        raise ValueError(
            "mesh.split() returned no components — the mesh is empty or "
            "entirely degenerate."
        )

    largest = max(components, key=lambda m: len(m.vertices))

    # Fix face winding for consistent outward normals.
    trimesh.repair.fix_winding(largest)

    # Remove vertices that are not referenced by any face.
    largest.remove_unreferenced_vertices()

    # libigl LSCM/harmonic/ARAP require a strictly manifold mesh.
    # BPA output frequently has non-manifold edges; clean via Open3D.
    o3d_mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(largest.vertices),
        triangles=o3d.utility.Vector3iVector(largest.faces),
    )
    o3d_mesh.remove_duplicated_vertices()
    o3d_mesh.remove_duplicated_triangles()
    o3d_mesh.remove_degenerate_triangles()
    o3d_mesh.remove_non_manifold_edges()

    V = np.asarray(o3d_mesh.vertices, dtype=np.float64)
    F = np.asarray(o3d_mesh.triangles, dtype=np.int32)

    if F.shape[0] == 0:
        raise ValueError(
            "Zero faces remain after cleaning the mesh. "
            "Check that PLY_PATH points to a segmented forearm cloud, "
            "not a raw scan or an empty file."
        )

    # libigl LSCM fails on "pinch" boundary vertices — vertices that appear as
    # source in 2+ entries of igl.boundary_facets.  These arise from inconsistent
    # face winding near BPA seam edges.  Remove faces around pinch vertices and
    # re-take the largest connected component.  Small holes that remain are OK
    # for LSCM (it does not require single-loop boundary topology).
    import igl as _igl
    for _ in range(10):
        BF_pre, _, _ = _igl.boundary_facets(F.astype(np.int64))
        source_counts: dict = {}
        for _e in BF_pre:
            _u = int(_e[0])
            source_counts[_u] = source_counts.get(_u, 0) + 1
        pinch = np.array([u for u, c in source_counts.items() if c > 1], dtype=np.int32)
        if len(pinch) == 0:
            break
        bad_mask = np.any(np.isin(F, pinch), axis=1)
        F = F[~bad_mask]
        if F.shape[0] == 0:
            raise ValueError("Mesh became empty during pinch-vertex repair.")
        used = np.unique(F)
        remap = np.full(V.shape[0], -1, dtype=np.int32)
        remap[used] = np.arange(len(used), dtype=np.int32)
        V, F = V[used], remap[F]
        components = trimesh.Trimesh(vertices=V, faces=F, process=False).split(
            only_watertight=False
        )
        if components:
            lc = max(components, key=lambda m: len(m.vertices))
            V = np.asarray(lc.vertices, dtype=np.float64)
            F = np.asarray(lc.faces, dtype=np.int32)

    return V, F


# ---------------------------------------------------------------------------
# Boundary helpers
# ---------------------------------------------------------------------------

def boundary_loop(F: np.ndarray) -> np.ndarray:
    """Return the ordered boundary vertex indices for the mesh."""
    b = igl.boundary_loop(F)
    if b is None or len(b) == 0:
        raise RuntimeError(
            "No boundary loop found — the mesh appears to be a closed surface. "
            "Forearm meshes must have an open boundary for flattening."
        )
    return b


def canonicalise_uv(
    uv: np.ndarray,
    center_vid: int,
    boundary_vid: int,
) -> np.ndarray:
    # Rigid 2D similarity: translate uv[center_vid] -> origin, rotate so
    # uv[boundary_vid] lies on +x. Preserves scale and orientation, so it
    # cannot create or remove distortion — it only re-frames the layout.
    # The scale (and any per-method distortion) is left untouched.
    out = uv - uv[center_vid]
    anchor = out[boundary_vid]
    r = float(np.hypot(anchor[0], anchor[1]))
    if r < 1e-12:
        raise ValueError(
            f"Cannot canonicalise UV: boundary anchor vid={boundary_vid} "
            f"coincides with centre vid={center_vid} after translation."
        )
    cos_t, sin_t = anchor[0] / r, anchor[1] / r
    # Rotate by -theta so that the anchor lands on +x.
    R = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float64)
    return out @ R.T


# ---------------------------------------------------------------------------
# SLIM helpers
# ---------------------------------------------------------------------------

def _has_flipped_triangles(uv: np.ndarray, F: np.ndarray) -> bool:
    # Triangles agree on orientation iff all signed areas share one sign.
    p0, p1, p2 = uv[F[:, 0]], uv[F[:, 1]], uv[F[:, 2]]
    cross = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (
        p1[:, 1] - p0[:, 1]
    ) * (p2[:, 0] - p0[:, 0])
    return bool(np.any(cross > 0) and np.any(cross < 0))


def flatten_slim(
    V: np.ndarray,
    F: np.ndarray,
    boundary: np.ndarray,
    center_vid: int | None = None,
    n_iter: int = 40,
) -> np.ndarray:
    """Symmetric-Dirichlet SLIM flattening, initialised from a harmonic map.

    Production version — **no Tutte fallback**.  If the cotangent-harmonic
    initialisation has flipped triangles, raises ``RuntimeError`` immediately
    rather than silently degrading to a Tutte map.

    Mirrors libigl's ``tutorial/709_SLIM/param_2d_demo_iter.cpp``:

    * Initialise UV with the cotangent-harmonic map, **boundary-only**
      Dirichlet constraints (no interior pin — interior pins break the
      Rado-Kneser-Choquet bijectivity guarantee and create fold-overs).
    * If the harmonic init has any flipped triangles, raise immediately.
    * Run ``slim_precompute`` with **empty** ``b``/``bc`` and ``soft_p=0``
      — SLIM optimises symmetric Dirichlet over the whole map without any
      positional constraints.
    * If ``center_vid`` is given, apply a rigid 2D similarity transform
      after the solve to place the centre at the UV origin.

    Parameters
    ----------
    V:
        Vertex positions, shape (N, 3), float64.
    F:
        Face indices, shape (M, 3), int32.
    boundary:
        Ordered boundary vertex indices from ``boundary_loop(F)``.
    center_vid:
        Interior vertex index to place at the UV origin post-solve.
    n_iter:
        Number of SLIM iterations.

    Raises
    ------
    ValueError
        If ``center_vid`` is on the mesh boundary.
    RuntimeError
        If the cotangent-harmonic initialisation produces NaN values.
    RuntimeError
        If the cotangent-harmonic initialisation has flipped triangles —
        the mesh topology may have non-manifold edges or ill-conditioned
        geometry.  Re-mesh with a finer BPA radius or inspect the forearm PLY.
    RuntimeError
        If SLIM produces NaN values after the solve.
    """
    if center_vid is not None and int(center_vid) in set(int(v) for v in boundary):
        raise ValueError(
            f"center_vid {int(center_vid)} is on the mesh boundary — "
            "it must be an interior vertex for centre-aligned flattening."
        )

    # Boundary-only harmonic init (no interior pin).
    n_b = len(boundary)
    angles = np.linspace(0.0, 2.0 * np.pi, n_b, endpoint=False)
    boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)])
    b = boundary.astype(np.int32)
    uv_init = igl.harmonic(V, F, b, boundary_uv, 1).astype(np.float64)
    if np.any(np.isnan(uv_init)):
        raise RuntimeError(
            "Harmonic initialisation produced NaN — mesh may be degenerate."
        )

    # Fail-fast: no Tutte fallback in production.
    if _has_flipped_triangles(uv_init, F):
        raise RuntimeError(
            "Cotangent-harmonic initialisation produced flipped triangles — "
            "the mesh topology may have non-manifold edges or ill-conditioned "
            "geometry. Re-mesh with a finer BPA radius or inspect the forearm PLY."
        )

    # libigl 709 tutorial pattern: empty constraints, soft_p=0.
    empty_b = np.empty((0,), dtype=np.int32)
    empty_bc = np.empty((0, 2), dtype=np.float64)
    data = igl.slim_precompute(
        V,
        F,
        uv_init,
        igl.MappingEnergyType.SYMMETRIC_DIRICHLET,
        empty_b,
        empty_bc,
        0.0,
    )

    uv = uv_init
    for _ in range(n_iter):
        uv = igl.slim_solve(data, 1)

    if np.any(np.isnan(uv)):
        raise RuntimeError(
            "SLIM produced NaN values — mesh may be degenerate."
        )
    uv = uv.astype(np.float64)

    if center_vid is not None:
        uv = canonicalise_uv(uv, int(center_vid), int(boundary[0]))

    return uv
