# Plan: Configurable mesh method for SLIM UV pipeline

**Date:** 2026-05-22
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-24 18:03
**Base Branch:** `feature/slim-uv-boundary-gap-stitching`
**Branch:** `feature/slim-uv-configurable-mesh-method`

---

## Overview

Add a configurable `mesh_method` parameter (`"bpa"` or `"delaunay"`) to the
SLIM UV precompute pipeline so both mesh construction approaches can be
compared empirically using the interactive step-by-step viewer.  The current
pipeline hard-codes BPA; this change exposes the existing `build_delaunay_mesh`
function as an alternative production path.

## Problem Statement

The SLIM UV pipeline uses Ball Pivoting Algorithm (BPA) for mesh construction.
BPA produces non-manifold artifacts that require an 8-step cleaning pipeline
(`clean_mesh`).  The preprocessing pipeline uses 2.5D Delaunay triangulation
instead --- simpler, faster, and manifold by construction.

Investigation confirmed that RF-centering is translation-only (no rotation),
so the forearm surface remains a height field in XY at analysis time.  This
means 2.5D Delaunay is geometrically valid for the SLIM UV pipeline, and would
eliminate 3 of the 8 cleaning steps entirely (non-manifold removal, pinch
repair, sliver removal).

There is currently no way to toggle between methods in the production pipeline
to compare SLIM UV quality, distortion, and convergence.

## Goals

### In Scope
1. Make `mesh_method` configurable in the DAG config and threaded through the
   full call chain (DAG YAML -> flow -> precompute -> mesh builder)
2. Add `max_edge_mm` as an optional DAG parameter for Delaunay edge filtering
   (auto-computed from point density when unset)
3. Store `mesh_method` in the SLIM UV NPZ cache for provenance and staleness
   detection
4. Both methods viewable through the existing interactive step-by-step viewer

### Out of Scope
- Changing the `clean_mesh` function itself (BPA-specific steps become visible
  no-ops for Delaunay --- that is informative, not a bug)
- Modifying downstream SLIM UV cache consumers (cache filename unchanged)
- Benchmarking or recommending one method over the other (that is the user's
  empirical investigation)
- Removing or deprecating BPA

## Success Criteria

- [ ] Running with `mesh_method: delaunay` and `interactive: true` shows
      non-manifold removal, pinch repair, and sliver removal as no-ops
      (0 faces removed) in the step viewer
- [ ] Running with `mesh_method: bpa` produces identical output to the current
      pipeline (no regression)
- [ ] Switching `mesh_method` in the DAG config triggers automatic recompute
      (staleness detection) without needing `force_processing: true`
- [ ] Both `_mesh_bpa.obj` and `_mesh_delaunay.obj` cache files can coexist
      alongside the PLY

---

## Technical Design

### Approach

Thread a `mesh_method` parameter through the existing call chain with minimal
changes.  Reuse the existing `build_delaunay_mesh()` function for the Delaunay
path.  Keep `clean_mesh()` untouched --- the per-step toggles already exist,
and letting BPA-specific steps run as no-ops on Delaunay input is more
informative for comparison than skipping them.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Add `mesh_method` param to existing functions | Minimal code changes, reuses `build_delaunay_mesh` | Slightly wider function signatures | **Chosen** |
| Separate `precompute_forearm_slim_uv_delaunay` function | Clean separation | Code duplication, two flow registrations, two DAG tasks | Rejected |
| Auto-detect method from mesh topology | No config needed | Fragile heuristic, removes user control | Rejected |
| PCA projection for Delaunay (like sandbox) | More robust to arbitrary orientations | Unnecessary since RF-centering preserves XY alignment; adds complexity | Rejected |

### Architecture Changes

No new modules.  Changes are parameter additions to existing functions.

```
DAG config  (mesh_method: "bpa" | "delaunay")
    |
    v
analysis_workflow_processing.py::precompute_forearm_slim_uv_flow()
    |  + mesh_method, max_edge_mm params
    |  + staleness check (cached method != requested method)
    v
forearm_slim_uv.py::precompute_forearm_slim_uv()
    |  + mesh_method, max_edge_mm params
    |  + store mesh_method in NPZ
    v
rf_surface_utils.py::load_or_build_forearm_mesh()
    |  + mesh_method, max_edge_mm params
    |  + route to BPA or Delaunay branch
    v
build_delaunay_mesh()   [existing, line 167 --- reused as-is]
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| OBJ mesh cache | Separate files: `_mesh_bpa.obj`, `_mesh_delaunay.obj` | Both coexist on disk for comparison |
| NPZ SLIM cache | Single filename `_slim_uv.npz`, method stored inside | Avoids touching 6 downstream consumers |
| Cache staleness | Check stored `mesh_method` vs requested; force recompute on mismatch | Transparent method switching without `force_processing` |
| `max_edge_mm` default | `3.0 * avg_nn` (auto-computed) | Adapts to session point density; consistent with BPA radius strategy |
| Delaunay projection | XY (`vertices[:, :2]`) | RF-centering is translation-only; height-field property preserved |
| `clean_mesh` | Untouched | Per-step toggles exist; no-op steps are informative in viewer |

---

## Implementation Plan

### Phase 1: Mesh builder — Delaunay branch
**Started:** 2026-05-22
**Completed:** 2026-05-22
**Goal:** `load_or_build_forearm_mesh` supports both methods

- [x] Task 1.1 --- Add `mesh_method: str = "bpa"` and
      `max_edge_mm: float | None = None` to `load_or_build_forearm_mesh()`
      signature
- [x] Task 1.2 --- Add `mesh_method` validation (`raise ValueError` for
      unknown methods)
- [x] Task 1.3 --- Make cache suffix dynamic: `f"_mesh_{mesh_method}.obj"`
- [x] Task 1.4 --- Extract `_build_and_cache_delaunay()` private helper that
      loads PLY, computes `avg_nn`, defaults `max_edge_mm` to `3.0 * avg_nn`,
      calls `build_delaunay_mesh()`, logs stats, caches, and returns
- [x] Task 1.5 --- Route: `"delaunay"` calls new helper; `"bpa"` runs existing
      code unchanged

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/surface/rf_surface_utils.py` ---
  Add `mesh_method`/`max_edge_mm` params, Delaunay branch, validation

**Dependencies:** None

### Phase 2: SLIM precompute — thread parameter + cache provenance
**Started:** 2026-05-22
**Completed:** 2026-05-22
**Goal:** `precompute_forearm_slim_uv` accepts and persists `mesh_method`

- [x] Task 2.1 --- Add `mesh_method: str = "bpa"` and
      `max_edge_mm: float | None = None` to `precompute_forearm_slim_uv()`
- [x] Task 2.2 --- Add `mesh_method` validation
- [x] Task 2.3 --- Pass `mesh_method` and `max_edge_mm` to
      `load_or_build_forearm_mesh()` call (line 376)
- [x] Task 2.4 --- Store `mesh_method` in NPZ cache (line 535):
      `mesh_method=np.array(mesh_method, dtype='U16')`
- [x] Task 2.5 --- Add `mesh_method: str = "bpa"` field to `SlimUvCache`
      dataclass (line 46)
- [x] Task 2.6 --- Read `mesh_method` in `load_slim_uv_cache()` (line 672)
      with backward-compat default for old caches
- [x] Task 2.7 --- Add `mesh_method: str = "bpa"` kwarg to
      `_launch_slim_steps_viewer()` (line 88); change raw-mesh step label from
      `"Raw mesh (BPA)"` to `f"Raw mesh ({mesh_method.upper()})"`; update both
      call sites

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/surface/forearm_slim_uv.py` ---
  Thread param through precompute, cache, loader, viewer

**Dependencies:** Phase 1

### Phase 3: DAG wiring
**Started:** 2026-05-22
**Completed:** 2026-05-22
**Goal:** `mesh_method` configurable from DAG YAML

- [x] Task 3.1 --- Add `mesh_method: str = "bpa"` and
      `max_edge_mm: float | None = None` to
      `precompute_forearm_slim_uv_flow()` signature (line 147)
- [x] Task 3.2 --- Add mesh-method staleness check: after line 196, when cache
      exists and `force_processing` is false, load NPZ and compare stored
      `mesh_method` to requested; set `force_session = True` on mismatch
- [x] Task 3.3 --- Pass `mesh_method` and `max_edge_mm` to `_precompute()`
      call (line 208)
- [x] Task 3.4 --- Add `mesh_method` and `max_edge_mm` to the DAG params
      lambda (line 1145)
- [x] Task 3.5 --- Add `mesh_method: bpa` and commented-out `max_edge_mm` to
      `precompute_forearm_slim_uv.options` in the DAG YAML

**Files Modified:**
- `code/scripts/analysis_workflow_processing.py` --- Thread param through flow,
  staleness check, DAG params
- `configs/analyse_workflow_processing_dag.yaml` --- Add config options

**Dependencies:** Phase 2

---

## Testing Plan

### Manual Verification
- [ ] Run `precompute_forearm_slim_uv` with `mesh_method: delaunay`,
      `interactive: true` on one session --- verify the step viewer shows
      non-manifold/pinch/sliver steps as no-ops
- [ ] Run with `mesh_method: bpa`, `interactive: true` on the same session ---
      verify output matches previous pipeline results
- [ ] Run with `mesh_method: bpa`, then change to `delaunay` without
      `force_processing` --- verify staleness detection triggers recompute
- [ ] Verify both `_mesh_bpa.obj` and `_mesh_delaunay.obj` exist after running
      both methods

### Edge Cases
- [ ] Invalid `mesh_method` value (e.g. `"poisson"`) raises `ValueError`
- [ ] Old NPZ cache without `mesh_method` field loads correctly (defaults to
      `"bpa"`)
- [ ] Delaunay with very sparse point cloud (< 4 points) returns `None` and
      raises `ValueError` in precompute

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` --- mention `mesh_method` option in
      the SLIM UV section
- [ ] Inline YAML comments in DAG config (included in implementation)

---

## Rollback Plan

All changes are additive (new parameters with backward-compatible defaults).

1. Set `mesh_method: bpa` in DAG config to restore previous behavior
2. Old NPZ caches (without `mesh_method` field) load as `"bpa"` automatically
3. If needed, revert the commits; no data migration required

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Delaunay bridging triangles distort SLIM UV | Medium | Medium | `max_edge_mm` filtering removes bridging; user can tune threshold |
| `max_edge_mm` auto-default too aggressive/loose for some sessions | Low | Low | Exposed as DAG param for per-run override |
| Downstream cache consumers break on new NPZ field | Low | High | Field is additive; `load_slim_uv_cache` uses `dict.get` with default |
| Delaunay edge filtering creates too many holes | Medium | Medium | `clean_mesh` hole-filling still runs; visible in interactive viewer |

---

## References

- Knowledge base: `report-preprocessing-forearm-mesh-construction.md` --- why
  preprocessing uses 2.5D Delaunay
- Knowledge base: `report-slim-uv-mesh-cleaning-steps.md` --- the 8-step BPA
  cleanup chain
- Knowledge base: `bug-slim-uv-non-manifold-flip.md` --- BPA non-manifold
  consequences for SLIM
- Existing code: `rf_surface_utils.py:167` --- `build_delaunay_mesh()` function
- Existing code: `flatten_forearm_sandbox.py` --- sandbox with BPA/Delaunay
  toggle
