"""
_slim_helpers.py
================

Reusable primitives for SLIM-based forearm flattening.

Extracted from sandbox scripts so both the production pipeline and the
sandbox comparison harness can import a single authoritative copy.

Functions
---------
clean_mesh             — Keep largest component, fix winding, drop orphans,
                         remove non-manifold edges and pinch vertices.
boundary_loop          — Return ordered boundary vertex indices.
canonicalise_uv        — Rigid 2D similarity: centre at origin, boundary[0] on +x.
_tutte_uniform_map     — Tutte (1963) uniform-weight Laplacian; provably bijective
                         on a convex boundary.  Used as a fallback for flatten_slim.
_has_flipped_triangles — Return True when the UV map has inconsistent triangle
                         orientation.
flatten_slim           — Symmetric-Dirichlet SLIM, initialised from a harmonic map.
                         Falls back to Tutte, then trims degenerate faces if needed.
                         Returns (V_out, F_out, uv) — V/F may be trimmed.
compute_face_distortion — Per-face conformal and area distortion via Jacobian SVD.
"""

import logging

import numpy as np
import igl
import open3d as o3d
import trimesh
import trimesh.repair
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mesh cleaning
# ---------------------------------------------------------------------------

def _find_boundary_loops(F: np.ndarray) -> list[list[int]]:
    """Trace all boundary loops from directed boundary edges.

    Returns a list of loops, each a list of vertex indices in traversal order.
    The loops are sorted longest-first.
    """
    BF, _, _ = igl.boundary_facets(F.astype(np.int64))
    if len(BF) == 0:
        return []

    adj: dict[int, list[int]] = {}
    for e in BF:
        u, v = int(e[0]), int(e[1])
        adj.setdefault(u, []).append(v)

    visited: set[int] = set()
    loops: list[list[int]] = []
    for start in list(adj.keys()):
        if start in visited:
            continue
        loop: list[int] = []
        cur = start
        while cur not in visited:
            visited.add(cur)
            loop.append(cur)
            nexts = [n for n in adj.get(cur, []) if n not in visited]
            if not nexts:
                break
            cur = nexts[0]
        if len(loop) >= 3:
            loops.append(loop)

    loops.sort(key=len, reverse=True)
    return loops


def _fill_interior_holes(
    V: np.ndarray,
    F: np.ndarray,
    max_passes: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill non-largest boundary loops with centroid fan triangulation.

    Interior holes left by ``remove_non_manifold_edges()`` break the
    single-boundary disk topology required by harmonic/Tutte parameterisation.
    This function closes each inner hole by inserting a centroid vertex and
    fanning triangles around the loop.  Winding is fixed afterwards.

    Iterates up to ``max_passes`` because filling can occasionally create
    new small boundary anomalies that require an additional pass.

    Returns ``(V, F)`` — may have new vertices appended to V.
    """
    total_holes = 0
    total_tris = 0

    for _pass in range(max_passes):
        loops = _find_boundary_loops(F)
        if len(loops) <= 1:
            break

        inner_loops = loops[1:]
        new_verts: list[np.ndarray] = []
        new_faces: list[list[int]] = []

        for loop in inner_loops:
            centroid = V[loop].mean(axis=0)
            new_vid = V.shape[0] + len(new_verts)
            new_verts.append(centroid)
            for i in range(len(loop)):
                j = (i + 1) % len(loop)
                new_faces.append([new_vid, loop[i], loop[j]])

        if not new_faces:
            break

        V = np.vstack([V, np.array(new_verts, dtype=np.float64)])
        F = np.vstack([F, np.array(new_faces, dtype=np.int32)])
        total_holes += len(inner_loops)
        total_tris += len(new_faces)

        tmp = trimesh.Trimesh(vertices=V, faces=F, process=False)
        trimesh.repair.fix_winding(tmp)
        V = np.asarray(tmp.vertices, dtype=np.float64)
        F = np.asarray(tmp.faces, dtype=np.int32)

    if total_holes > 0:
        logger.info(
            "Filled %d interior hole(s) with %d fan triangles.",
            total_holes, total_tris,
        )

    return V, F


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

    # Both remove_non_manifold_edges() and the pinch-vertex repair above can
    # leave interior holes that break the single-boundary disk topology
    # required for harmonic/Tutte UV initialisation.  Fill them with centroid
    # fan triangulation as the last cleanup step.
    V, F = _fill_interior_holes(V, F)

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


def _tutte_uniform_map(
    F: np.ndarray,
    n_vertices: int,
    boundary: np.ndarray,
    boundary_uv: np.ndarray,
) -> np.ndarray:
    """Tutte (1963) uniform-weight Laplacian with a convex boundary.

    Provably bijective when the boundary maps homeomorphically to a convex
    polygon (unit circle here).  Used as a fallback init for SLIM when the
    cotangent-harmonic init produces flipped triangles.
    """
    e_all = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    edges = np.unique(np.sort(e_all, axis=1), axis=0)
    a = edges[:, 0]
    b = edges[:, 1]
    n_e = len(edges)
    rows = np.concatenate([a, b, a, b])
    cols = np.concatenate([b, a, a, b])
    vals = np.concatenate([-np.ones(n_e), -np.ones(n_e), np.ones(n_e), np.ones(n_e)])
    L = coo_matrix((vals, (rows, cols)), shape=(n_vertices, n_vertices)).tolil()

    boundary = np.asarray(boundary, dtype=np.int64)
    for bi in boundary:
        L.rows[int(bi)] = [int(bi)]
        L.data[int(bi)] = [1.0]
    rhs = np.zeros((n_vertices, 2), dtype=np.float64)
    rhs[boundary] = boundary_uv

    return np.asarray(spsolve(L.tocsr(), rhs), dtype=np.float64)


def flatten_slim(
    V: np.ndarray,
    F: np.ndarray,
    boundary: np.ndarray,
    center_vid: int | None = None,
    n_iter: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric-Dirichlet SLIM flattening, initialised from a harmonic map.

    Mirrors libigl's ``tutorial/709_SLIM/param_2d_demo_iter.cpp``:

    * Initialise UV with the cotangent-harmonic map, **boundary-only**
      Dirichlet constraints (no interior pin — interior pins break the
      Rado-Kneser-Choquet bijectivity guarantee and create fold-overs).
    * If the harmonic init has any flipped triangles, fall back to the
      Tutte (uniform-weight Laplacian) map.  A ``logger.warning`` is
      emitted when this fallback is taken.
    * If the Tutte init still has ≤ ``_TRIM_FLIP_MAX`` flipped faces (a
      sign of degenerate sliver triangles from BPA meshing rather than
      global topology failure), the 1-ring neighbourhood of those faces is
      excised from the mesh, the boundary and ``center_vid`` are re-located
      via KDTree, and Tutte is recomputed on the trimmed mesh.  A second
      ``logger.warning`` is emitted.  If flips persist after the trim,
      ``RuntimeError`` is raised.
    * Run ``slim_precompute`` with **empty** ``b``/``bc`` and ``soft_p=0``.
    * If ``center_vid`` is given, apply a rigid 2D similarity transform
      after the solve to place the centre at the UV origin.

    Returns
    -------
    V_out : np.ndarray, shape (N', 3)
        Vertex positions of the (possibly trimmed) mesh.
    F_out : np.ndarray, shape (M', 3), int32
        Face indices of the (possibly trimmed) mesh.
    uv : np.ndarray, shape (N', 2), float64
        SLIM UV coordinates.

    In the common case (no trimming), ``V_out is V`` and ``F_out is F``.

    Raises
    ------
    ValueError
        If ``center_vid`` is on the mesh boundary.
    RuntimeError
        If the cotangent-harmonic initialisation produces NaN values.
    RuntimeError
        If the Tutte init still has flipped triangles after the 1-ring trim
        (the mesh topology is too complex for boundary-only parameterisation).
    RuntimeError
        If SLIM produces NaN values after the solve.
    """
    from scipy.spatial import KDTree as _KDTree

    # Save the 3D position of the centre so we can re-locate it after any
    # potential mesh trimming.
    _center_3d: np.ndarray | None = (
        V[int(center_vid)].copy() if center_vid is not None else None
    )

    if center_vid is not None and int(center_vid) in set(int(v) for v in boundary):
        raise ValueError(
            f"center_vid {int(center_vid)} is on the mesh boundary — "
            "it must be an interior vertex for centre-aligned flattening."
        )

    def _setup_boundary_uv(bdy: np.ndarray) -> np.ndarray:
        angles = np.linspace(0.0, 2.0 * np.pi, len(bdy), endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    # Boundary-only harmonic init (no interior pin).
    boundary_uv = _setup_boundary_uv(boundary)
    uv_init = igl.harmonic(V, F, boundary.astype(np.int32), boundary_uv, 1).astype(
        np.float64
    )
    if np.any(np.isnan(uv_init)):
        raise RuntimeError(
            "Harmonic initialisation produced NaN — mesh may be degenerate."
        )

    # Cotangent-harmonic can flip on ill-conditioned forearm topologies.
    # Fall back to the Tutte (uniform-weight) map.
    if _has_flipped_triangles(uv_init, F):
        logger.warning(
            "Cotangent-harmonic initialisation produced flipped triangles — "
            "falling back to Tutte (uniform-weight Laplacian) initialisation."
        )
        uv_init = _tutte_uniform_map(F, V.shape[0], boundary, boundary_uv)

        for _trim_round in range(_TRIM_MAX_ROUNDS):
            if not _has_flipped_triangles(uv_init, F):
                break

            p0, p1, p2 = uv_init[F[:, 0]], uv_init[F[:, 1]], uv_init[F[:, 2]]
            cross = (
                (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
            )
            flipped_fi = np.where(cross < 0)[0]
            n_flip = len(flipped_fi)

            if n_flip > _TRIM_FLIP_MAX:
                raise RuntimeError(
                    f"Tutte fallback has {n_flip} flipped triangle(s) — too many "
                    f"to repair by trimming (limit: {_TRIM_FLIP_MAX}).  The mesh "
                    "topology may have severe non-manifold structure; re-mesh with "
                    "a finer BPA radius or inspect the forearm PLY."
                )

            logger.warning(
                "Tutte init has %d flipped face(s) (round %d) — trimming "
                "1-ring of degenerate face vertices and retrying.",
                n_flip, _trim_round + 1,
            )

            bad_verts = set(int(v) for fi in flipped_fi for v in F[fi])
            ring_mask = np.any(np.isin(F, list(bad_verts)), axis=1)
            F_trim = F[~ring_mask]
            if F_trim.shape[0] == 0:
                raise RuntimeError(
                    "Mesh became empty while trimming degenerate flip-faces."
                )

            used = np.unique(F_trim)
            remap_arr = np.full(V.shape[0], -1, dtype=np.int32)
            remap_arr[used] = np.arange(len(used), dtype=np.int32)
            V = V[used]
            F = remap_arr[F_trim]

            comps = trimesh.Trimesh(
                vertices=V, faces=F, process=False
            ).split(only_watertight=False)
            if not comps:
                raise RuntimeError(
                    "No components remain after trimming degenerate flip-faces."
                )
            lc = max(comps, key=lambda m: len(m.vertices))
            V = np.asarray(lc.vertices, dtype=np.float64)
            F = np.asarray(lc.faces, dtype=np.int32)

            V, F = _fill_interior_holes(V, F)

            boundary = boundary_loop(F)
            boundary_uv = _setup_boundary_uv(boundary)

            if _center_3d is not None:
                _, center_vid = _KDTree(V).query(_center_3d)
                center_vid = int(center_vid)
                if center_vid in set(int(v) for v in boundary):
                    raise ValueError(
                        f"center_vid {center_vid} landed on the new mesh boundary "
                        "after trimming — the spike centroid is in the excised region."
                    )

            uv_init = _tutte_uniform_map(F, V.shape[0], boundary, boundary_uv)
        else:
            if _has_flipped_triangles(uv_init, F):
                raise RuntimeError(
                    f"Tutte fallback still has flipped triangles after "
                    f"{_TRIM_MAX_ROUNDS} trim rounds — the mesh topology is too "
                    "complex for boundary-only parameterisation.  Re-mesh with "
                    "a finer BPA radius or inspect the forearm PLY."
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

    return V, F, uv


# Maximum number of flipped faces that trigger the 1-ring trim fallback.
# Beyond this count the failure is global, not local, and trimming would
# remove too much of the mesh.
_TRIM_FLIP_MAX: int = 50
_TRIM_MAX_ROUNDS: int = 5


# ---------------------------------------------------------------------------
# Distortion metrics
# ---------------------------------------------------------------------------

def compute_face_distortion(
    V: np.ndarray,
    F: np.ndarray,
    uv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-face distortion via Jacobian SVD of the 3D-to-UV affine map.

    Returns
    -------
    conformal : (M,) float64 — σ_max/σ_min per face (1 = perfectly conformal).
    area : (M,) float64 — log2(det_J / median(det_J)); 0 = median area ratio.
    """
    e1 = V[F[:, 1]] - V[F[:, 0]]
    e2 = V[F[:, 2]] - V[F[:, 0]]

    t1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True).clip(1e-15)
    n = np.cross(e1, e2)
    n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-15)
    t2 = np.cross(n, t1)

    q1x = np.einsum("ij,ij->i", e1, t1)
    q1y = np.einsum("ij,ij->i", e1, t2)
    q2x = np.einsum("ij,ij->i", e2, t1)
    q2y = np.einsum("ij,ij->i", e2, t2)

    du1 = uv[F[:, 1]] - uv[F[:, 0]]
    du2 = uv[F[:, 2]] - uv[F[:, 0]]

    det = q1x * q2y - q2x * q1y
    degen = np.abs(det) < 1e-15
    det_safe = np.where(degen, 1.0, det)

    iq00 = q2y / det_safe
    iq01 = -q2x / det_safe
    iq10 = -q1y / det_safe
    iq11 = q1x / det_safe

    M_faces = len(F)
    J = np.empty((M_faces, 2, 2), dtype=np.float64)
    J[:, 0, 0] = du1[:, 0] * iq00 + du2[:, 0] * iq10
    J[:, 0, 1] = du1[:, 0] * iq01 + du2[:, 0] * iq11
    J[:, 1, 0] = du1[:, 1] * iq00 + du2[:, 1] * iq10
    J[:, 1, 1] = du1[:, 1] * iq01 + du2[:, 1] * iq11

    S = np.linalg.svd(J, compute_uv=False)
    s1 = np.maximum(S[:, 0], 1e-15)
    s2 = np.maximum(S[:, 1], 1e-15)
    s1[degen] = 1.0
    s2[degen] = 1.0

    conformal = s1 / s2
    det_J = s1 * s2
    area = np.log2(det_J / np.maximum(np.median(det_J), 1e-15))

    return conformal, area
