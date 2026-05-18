# Bug: SLIM UV Precompute Fails on Non-Manifold Forearm Mesh (ST13-02)

**Date:** 2026-05-18
**Status:** Resolved

---

## Symptom

Running `precompute_forearm_slim_uv_flow` (via the DAG launcher, all 12 sessions) crashed on
session `2022-06-14_ST13-02` with:

```
RuntimeError: Cotangent-harmonic initialisation produced flipped triangles —
the mesh topology may have non-manifold edges or ill-conditioned geometry.
Re-mesh with a finer BPA radius or inspect the forearm PLY.
```

Session `ST13-01` (processed first) succeeded; `ST13-02` was the first failure.

---

## Root Cause (three-layer failure)

### Layer 1 — Multiple boundary loops from non-manifold edge removal

`clean_mesh` uses Open3D's `remove_non_manifold_edges()` to remove triangles
touching non-manifold edges.  For the `ST13-02` forearm mesh, this created
**four small interior holes** (loop sizes 5, 8, 9, 13 vertices) in addition to
the legitimate outer boundary (383 vertices).  The cleaned mesh had:

```
Boundary loops: [383, 13, 9, 8, 5]
Euler characteristic: -4  (disk topology requires +1)
```

The four inner holes make the mesh topologically non-disk.

### Layer 2 — Harmonic init flips due to free inner-boundary vertices

The cotangent-harmonic map with boundary-only Dirichlet constraints
(`igl.harmonic`) maps all four inner-hole boundary vertices as free interior
vertices.  These free vertices cause the harmonic map to produce 3 flipped
triangles — the theoretical bijectivity guarantee (Rado-Kneser-Choquet) only
holds for genus-0, single-boundary meshes.

### Layer 3 — Tutte map produces a persistent sliver flip

After falling back to the Tutte (uniform-weight Laplacian) map — which is
also provably bijective only for disk topology — the map still produced
**1 flipped triangle** (face index 25836, vertices `[12988, 13035, 13036]`):

```
Signed UV area of flipped face: -6.5e-5  (nearly degenerate)
3D coords of vertices: [[25.2, 223.1, 655.9],
                         [27.1, 221.7, 655.7],
                         [27.0, 221.0, 656.6]]
```

This is a 2–3 mm sliver triangle adjacent to the outer boundary.  The free
inner-hole boundary vertices create a UV near-singularity that causes this
specific sliver triangle to report a tiny negative signed area.

SLIM cannot recover from this: symmetric Dirichlet energy diverges at
zero-area triangles, so initialising SLIM from this map makes the flip grow
from -6.5e-5 to -2.9 after 40 iterations.

---

## Fix

### `_slim_helpers.py::flatten_slim` — three-stage fallback

The function now returns `(V, F, uv)` instead of just `uv`, so callers always
see the mesh the UV was actually computed on (which may be slightly trimmed).

**Stage 1 — harmonic init** (unchanged).

**Stage 2 — Tutte fallback** (Tutte already existed): if harmonic flips,
fall back to Tutte with `logger.warning`.

**Stage 3 — 1-ring trim fallback** (new): if Tutte has ≤ `_TRIM_FLIP_MAX`
(10) flipped faces, excise the 1-ring neighbourhood of those face vertices:

```python
bad_verts = set(v for fi in flipped_fi for v in F[fi])
ring_mask = np.any(np.isin(F, list(bad_verts)), axis=1)
F_trim = F[~ring_mask]          # 17 faces removed for ST13-02
```

After excising, re-take the largest connected component, recompute the
boundary loop, relocate `center_vid` via KDTree on the new vertex set, and
recompute Tutte on the trimmed mesh.  If Tutte is still flipped after
trimming, raise `RuntimeError` (global topology failure, not patchable).

For `ST13-02` this removes 17 faces (0.07 % of the mesh) and produces a
flip-free Tutte initialisation.  SLIM then converges to a clean UV.

### `forearm_slim_uv.py::precompute_forearm_slim_uv`

Updated to unpack `V, F, uv = flatten_slim(...)` and re-derive `center_vid`
and `bloop` from the returned (possibly trimmed) mesh:

```python
V, F, uv = flatten_slim(V, F, bloop, center_vid=center_vid, n_iter=n_iter)
_, center_vid = KDTree(V).query(centroid_3d)
center_vid = int(center_vid)
bloop = boundary_loop(F)
```

### `compare_flattening_methods.py`

Updated to unpack the 3-tuple:

```python
_, _, uv_slim = flatten_slim(V, F, boundary, center_vid=center_vid)
```

---

## Verified

Both sessions tested end-to-end after the fix:

| Session | Result | Notes |
|---|---|---|
| `2022-06-14_ST13-01` | ✅ Pass | No fallbacks triggered (clean mesh) |
| `2022-06-14_ST13-02` | ✅ Pass | Harmonic→Tutte→trim→SLIM path taken |

The remaining 10 sessions (`ST13-03` through `ST18-04`) were **not run after the
fix** — they should be run via the DAG launcher to confirm no new failures.

---

## Known remaining work

1. **Run all 12 sessions** through `precompute_forearm_slim_uv` via the DAG
   launcher to confirm the fix generalises.  The two investigated sessions are
   the first two of the 12 listed in `configs/analyse_workflow_processing_dag.yaml`.

2. **`clean_mesh` does not fill inner holes** — it only removes non-manifold
   edges.  The four inner holes persist in the stored cache mesh (`V`, `F`).
   They are harmless for UV lookup (KDTree nearest-face handles them
   transparently), but they do mean the cache mesh is not a clean manifold.
   A future improvement could add explicit hole-filling (e.g., fan
   triangulation of loops shorter than N vertices) to `clean_mesh` so the
   stored mesh is topologically clean.

3. **`open3d.TriangleMesh.fill_holes` is not available** in the installed
   Open3D version — confirmed during investigation.  If upgrading Open3D,
   adding `fill_holes(hole_size=10.0)` (mm scale) in `clean_mesh` after
   `remove_non_manifold_edges` would be a clean upstream fix.

---

## Files changed

| File | Change |
|---|---|
| `code/src/analysis/receptive_field_mapping/_slim_helpers.py` | Added `_tutte_uniform_map`; added 3-stage fallback to `flatten_slim`; changed return type to `(V, F, uv)`; added `_TRIM_FLIP_MAX` constant; added `logging`, `scipy.sparse` imports |
| `code/src/analysis/receptive_field_mapping/forearm_slim_uv.py` | Updated to unpack `(V, F, uv)`; re-derives `center_vid`/`bloop` after flatten_slim |
| `code/scripts/compare_flattening_methods.py` | Removed local `_tutte_uniform_map` and `flatten_slim` wrapper (now redundant); updated call to `_, _, uv_slim = flatten_slim(...)` |

---

## Related notes

- [note-mesh-parameterization-interior-pin-foldovers.md](note-mesh-parameterization-interior-pin-foldovers.md) — why interior pins cause vortex foldovers; Tutte fallback design
- [note-igl-slim-api-version-mismatch.md](note-igl-slim-api-version-mismatch.md) — nanobind SLIM API (`slim_precompute` / `slim_solve`)
