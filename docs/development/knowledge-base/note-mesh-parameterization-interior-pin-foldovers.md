# Note: Mesh-Parameterization Interior-Pin Foldovers ("Vortex" Bug)

**Date:** 2026-05-16
**Status:** Resolved

---

## Symptom

When running `compare_flattening_methods.py` with `CENTER_POINT` set (so an
interior vertex is being pinned to the UV origin), the **Harmonic** and
**SLIM** panels showed a spiral / "vortex" pattern around the centre with
the surface visibly folded onto itself.  LSCM looked clean; ARAP was
distorted but did not vortex as severely.

## Root cause

The Tutte / Rado-Kneser-Choquet (RKC) bijectivity guarantee for discrete
harmonic (and convex-combination) maps requires:

1. The **boundary** is mapped homeomorphically to the boundary of a
   **convex** planar region.
2. **All interior vertices are free** — each satisfies the
   convex-combination property (its UV is the weighted average of its
   neighbours).

The original `flatten_harmonic` added a second Dirichlet constraint:

```python
b = np.concatenate([boundary, [center_vid]])
bc = np.vstack([boundary_uv, [[0.0, 0.0]]])
uv = igl.harmonic(V, F, b, bc, 1)
```

That pinned the centre interior vertex to `(0, 0)` in addition to the
unit-circle boundary.  This violates the second RKC precondition.  On an
elongated, forearm-like patch the natural harmonic image of an off-axis
interior vertex is **far** from the origin, so yanking it back to
`(0, 0)` while every boundary vertex is also fixed forces the
neighbouring triangles to **flip** around the pin — the "vortex".

Verified on a synthetic forearm-shaped patch (213 vertices, 362 faces):
the old "boundary + interior pin" produced **11 flipped triangles**; the
fixed "boundary-only" map produced **0**.

**SLIM** inherited the foldover because `flatten_slim` was initialised
from the broken harmonic and additionally soft-constrained the centre
with `soft_p = 1e5` — the symmetric-Dirichlet energy cannot unflip an
already-flipped initial guess once the centre is essentially hard-pinned.

**LSCM** uses only two pins total (centre + one boundary vertex) and
leaves all other boundary vertices free, so it has the geometric slack
to wrap around the centre without flipping.

**ARAP** shares the bad init but its local-rigidity energy doesn't
aggressively re-twist a folded map, so it only looks distorted rather
than vortex-spiralled.

## Fix

Treat the centre point as a **post-process** rather than a constraint:

1. Run the parameterization with **boundary-only** Dirichlet constraints
   (LSCM keeps its 2-pin form — it's fine).
2. Apply a rigid 2D similarity transform after the solve so
   `uv[center_vid] = (0, 0)` and `uv[boundary[0]]` lies on `+x`.  Because
   the transform is rigid, it cannot create or remove distortion — it
   only re-frames the layout for visual consistency across methods.
3. For **SLIM**, follow the libigl tutorial `709_SLIM/param_2d_demo_iter.cpp`
   pattern:
   - Cotangent-harmonic init with boundary-only pin.
   - If the init has any flipped triangles, fall back to the **Tutte map**
     (uniform-weight Laplacian) which is provably bijective for a convex
     boundary.
   - Run `slim_precompute` with **empty** `b`/`bc` and `soft_p = 0`.
     SLIM optimises symmetric Dirichlet over the whole map without any
     positional constraints; the centre is placed post-hoc.

The shared post-process helper is `canonicalise_uv()` in
`flatten_forearm_sandbox.py`.

## Related files

- `code/scripts/flatten_forearm_sandbox.py` — `canonicalise_uv()`,
  `flatten_lscm`, `flatten_harmonic`, `flatten_arap`.
- `code/scripts/compare_flattening_methods.py` — `flatten_slim`,
  `_has_flipped_triangles`, `_tutte_uniform_map`.
- [note-igl-slim-api-version-mismatch.md](note-igl-slim-api-version-mismatch.md)
  — sibling note covering the nanobind SLIM API signature.

## References

- Tutte, *How to Draw a Graph*, 1963.
- Floater, *Parametrization and smooth approximation of surface
  triangulations*, 1997.
- libigl tutorial 709 SLIM:
  https://github.com/libigl/libigl/blob/main/tutorial/709_SLIM/param_2d_demo_iter.cpp
- Hormann et al., *Mesh Parameterization: Theory and Practice*
  (SIGGRAPH Asia 2008 course notes).
