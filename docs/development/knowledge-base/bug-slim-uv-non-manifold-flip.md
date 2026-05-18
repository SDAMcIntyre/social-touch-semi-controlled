# Bug: SLIM UV Precompute Fails on Non-Manifold Forearm Meshes

**Date:** 2026-05-18
**Status:** Resolved (all 12 sessions pass)

---

## Symptom

Running `precompute_forearm_slim_uv_flow` (via the DAG launcher, all 12 sessions) crashed on
multiple sessions with:

```
RuntimeError: Cotangent-harmonic initialisation produced flipped triangles —
the mesh topology may have non-manifold edges or ill-conditioned geometry.
Re-mesh with a finer BPA radius or inspect the forearm PLY.
```

The initial investigation focused on `ST13-02`; running all 12 sessions
revealed 3 more failures: `ST14-02`, `ST14-04`, `ST16-03` — all from the
same root cause (interior holes breaking disk topology).

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

## Fix (two-part)

### Part 1 — Interior hole filling in `clean_mesh`

Added `_find_boundary_loops()` and `_fill_interior_holes()` to
`_slim_helpers.py`.  `clean_mesh` now fills all non-largest boundary loops
with centroid fan triangulation as its final step:

1. **Detect** all boundary loops via `igl.boundary_facets` + directed-edge
   tracing.
2. **Identify** the largest loop (the real outer boundary).
3. **Fill** each smaller loop by inserting a centroid vertex and fanning
   triangles around the loop.
4. **Fix winding** via `trimesh.repair.fix_winding()`.
5. **Iterate** up to 5 passes — filling can occasionally create new small
   boundary anomalies that require an additional pass.

This restores disk topology before parameterisation:

| Session | Before | After |
|---|---|---|
| ST14-02 | 10 loops, χ=-5 | 1 loop, χ=1 |
| ST14-04 | 14 loops, χ=-16 | 1 loop, χ=-3 (genus) |
| ST16-03 | 2 loops, χ=0 | 1 loop, χ=1 |

### Part 2 — Iterative trim+fill in `flatten_slim`

The Tutte fallback now iterates up to `_TRIM_MAX_ROUNDS` (5) rounds of:
trim 1-ring → take largest component → fill holes → recompute Tutte.

This handles two scenarios:
- Fan triangles from hole filling that are poorly conditioned and cause
  Tutte flips (e.g., ST18-04: 17 flips resolved in 2 rounds).
- Trim creating new holes that need re-filling before the next attempt.

`_TRIM_FLIP_MAX` raised from 10 to 50.  Fan triangles from hole filling
can create a burst of degenerate slivers that are still local, not global.

### `flatten_slim` return type (prior fix, unchanged)

Returns `(V, F, uv)` — callers see the mesh the UV was computed on.

### `forearm_slim_uv.py::precompute_forearm_slim_uv`

Unpacks `V, F, uv = flatten_slim(...)` and re-derives `center_vid`
and `bloop` from the returned (possibly trimmed) mesh.

### `compare_flattening_methods.py`

Unpacks the 3-tuple: `_, _, uv_slim = flatten_slim(...)`.

---

## Verified

All 12 sessions tested end-to-end:

| Session | Result | Path taken |
|---|---|---|
| `2022-06-14_ST13-01` | ✅ Pass | Hole fill → harmonic → SLIM |
| `2022-06-14_ST13-02` | ✅ Pass | Hole fill → harmonic → Tutte → trim+fill (1 round) → SLIM |
| `2022-06-14_ST13-03` | ✅ Pass | Hole fill → harmonic → SLIM |
| `2022-06-15_ST14-01` | ✅ Pass | Hole fill → harmonic → Tutte → trim+fill (1 round) → SLIM |
| `2022-06-15_ST14-02` | ✅ Pass | Hole fill (10 holes, 178 tris) → harmonic → Tutte → SLIM |
| `2022-06-15_ST14-04` | ✅ Pass | Hole fill (13 holes) → harmonic → Tutte → trim+fill (1 round) → SLIM |
| `2022-06-16_ST15-01` | ✅ Pass | Hole fill → harmonic → Tutte → trim+fill (1 round) → SLIM |
| `2022-06-17_ST16-02` | ✅ Pass | Hole fill → harmonic → SLIM |
| `2022-06-17_ST16-03` | ✅ Pass | Hole fill (1 hole) → harmonic → SLIM (no flips) |
| `2022-06-17_ST16-05` | ✅ Pass | Hole fill → harmonic → SLIM |
| `2022-06-22_ST18-01` | ✅ Pass | Hole fill → harmonic → Tutte → trim+fill (1 round) → SLIM |
| `2022-06-22_ST18-04` | ✅ Pass | Hole fill → harmonic → Tutte → trim+fill (2 rounds) → SLIM |

---

## Files changed

| File | Change |
|---|---|
| `code/src/analysis/receptive_field_mapping/_slim_helpers.py` | Added `_find_boundary_loops`, `_fill_interior_holes`; `clean_mesh` calls `_fill_interior_holes` as final step; `flatten_slim` trim fallback now iterates (trim→fill→Tutte) up to `_TRIM_MAX_ROUNDS`; `_TRIM_FLIP_MAX` raised to 50; added `_TRIM_MAX_ROUNDS = 5` |
| `code/src/analysis/receptive_field_mapping/forearm_slim_uv.py` | Updated to unpack `(V, F, uv)`; re-derives `center_vid`/`bloop` after flatten_slim |
| `code/scripts/compare_flattening_methods.py` | Removed local `_tutte_uniform_map` and `flatten_slim` wrapper (now redundant); updated call to `_, _, uv_slim = flatten_slim(...)` |

---

## Related notes

- [note-mesh-parameterization-interior-pin-foldovers.md](note-mesh-parameterization-interior-pin-foldovers.md) — why interior pins cause vortex foldovers; Tutte fallback design
- [note-igl-slim-api-version-mismatch.md](note-igl-slim-api-version-mismatch.md) — nanobind SLIM API (`slim_precompute` / `slim_solve`)
