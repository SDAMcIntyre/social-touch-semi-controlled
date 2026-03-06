# Plan: Alternative Registration Methods for Forearm Point Clouds

**Date:** 2026-03-04
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/multi-snapshot-forearm-registration`

---

## Overview

The current `ForearmRegistrator` uses vanilla point-to-plane ICP which assigns
every source point its nearest neighbour in the target — including fringe points
whose true correspondences don't exist in the other cloud. This plan adds five
alternative registration methods (robust kernel, multiscale, generalized ICP,
trimmed, and global+local) so they can be compared on real session data and the
best-performing approach can be adopted.

## Problem Statement

When two forearm snapshots share ~85–95% overlap but each has a 5–15% "fringe"
of points visible only from its own viewpoint:

1. **Vanilla ICP** gives every source point equal weight — fringe points whose
   true correspondences don't exist in the target cloud get matched to their
   nearest neighbour, creating false correspondences that bias the rigid
   transform.
2. `max_correspondence_distance` provides a hard cutoff but does not downweight
   borderline matches.
3. The bias scales with fringe size relative to overlap: larger viewpoint
   changes produce more drift.

Additionally, the code defaults to `max_correspondence_distance=0.10` (10 cm)
while the knowledge-base note documents it as `0.01` (1 cm). This discrepancy
should be reconciled.

## Goals

### In Scope

1. Add 5 alternative registration methods as new methods on `ForearmRegistrator`
2. Add a `method` dispatch parameter to `register_all()` and
   `register_all_to_average()` for selecting the registration strategy
3. Thread the `registration_method` parameter through `register_session_forearms()`
4. Extract repeated normal-estimation logic into `_ensure_target_normals()`
5. Reconcile `max_correspondence_distance` discrepancy between code and docs
6. Update knowledge-base documentation with new methods and parameter guidance

### Out of Scope

- Automatic method selection based on fitness
- GPU-accelerated registration
- Changing the default method from `"vanilla"` (backward compatibility)
- Benchmarking harness or automated comparison tooling

## Success Criteria

- [ ] All 5 new methods callable via `ForearmRegistrator.register_<method>()`
- [ ] `register_all(method=...)` and `register_all_to_average(method=...)` dispatch correctly
- [ ] `register_session_forearms(registration_method=...)` threads through to registrator
- [ ] Default behaviour unchanged — `method="vanilla"` produces identical results to current code
- [ ] Each method runs without error on a real multi-snapshot session
- [ ] Knowledge-base note updated with method descriptions and parameter table

---

## Technical Design

### Approach

Implement each method as a separate `register_*()` method on `ForearmRegistrator`,
all sharing the return signature `Tuple[np.ndarray, float]` (4×4 transform,
fitness). A class-level `_METHOD_MAP` dict maps string keys to method names,
used by `register_all()` / `register_all_to_average()` for dispatch.

This approach keeps each method self-contained and testable in isolation, while
requiring minimal changes to the orchestration layer (one new parameter passed
through).

### Knowledge-Base Relevance Check

The existing `note-forearm-icp-registration.md` is directly relevant:

- **Constraints to respect:** normals must exist on target cloud; deterministic
  canonical reference (lowest frame ID); single-input no-op path.
- **Approaches previously rejected:** FPFH+RANSAC was rejected as overkill for
  small shifts but is now being added as an optional method for cases where
  identity initialization fails.
- **Patterns to reuse:** `_ensure_target_normals()` pattern (currently inlined
  in `register()`); fitness threshold logging at 0.9.
- **Discrepancy found:** `max_correspondence_distance` is `0.10` in code
  (line 47 of `forearm_registrator.py`) but `0.01` in the knowledge-base note
  (line 48 of `note-forearm-icp-registration.md`). Must reconcile.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Add methods to `ForearmRegistrator` | Single class, shared state, minimal API change | Class grows larger | **Chosen** — keeps all registration logic cohesive |
| Strategy pattern (separate classes per method) | Clean separation, testable | Over-engineered for 5 methods with identical signatures | Rejected |
| Configuration object instead of string dispatch | Type-safe, extensible | Adds complexity for a simple enum-like choice | Rejected |

### Architecture Changes

Modified class structure:

```
ForearmRegistrator
├── register()                        # existing — vanilla point-to-plane ICP
├── register_robust()                 # NEW — point-to-plane ICP + TukeyLoss
├── register_multiscale()             # NEW — coarse-to-fine ICP
├── register_generalized()            # NEW — Generalized ICP (GICP)
├── register_trimmed()                # NEW — overlap-aware trim → ICP
├── register_global_then_local()      # NEW — FPFH+RANSAC → ICP refinement
├── _ensure_target_normals()          # NEW — extracted from register()
├── _make_point_to_plane()            # NEW — helper for optional robust kernel
├── _compute_fpfh()                   # NEW — helper for global registration
├── register_all(method=...)          # MODIFIED — add dispatch
└── register_all_to_average(method=.) # MODIFIED — add dispatch
```

---

## Implementation Plan

### Phase 1: Refactor — Extract Shared Helpers

**Goal:** Factor out repeated logic into private helpers before adding methods.

- [ ] Task 1.1 — Extract `_ensure_target_normals()` from `register()`: checks
  `self._canonical.has_normals()`, estimates if missing
- [ ] Task 1.2 — Add `_make_point_to_plane(use_robust_kernel, kernel_k)` helper
  that returns `TransformationEstimationPointToPlane` with optional `TukeyLoss`
- [ ] Task 1.3 — Add `_compute_fpfh(cloud, voxel_size)` helper for FPFH feature
  extraction (downsample, estimate normals, compute features)
- [ ] Task 1.4 — Update `register()` to call `_ensure_target_normals()` instead
  of inlined normal estimation

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/registration/forearm_registrator.py`
  — extract helpers, refactor `register()`

**Dependencies:** None

### Phase 2: Add Alternative Registration Methods

**Goal:** Implement the 5 new `register_*()` methods.

- [ ] Task 2.1 — Add `register_robust()`: point-to-plane ICP with
  `TukeyLoss(k=kernel_k)`. Fringe-point residuals beyond ~3k are zeroed out
- [ ] Task 2.2 — Add `register_multiscale()`: coarse-to-fine ICP at 3
  resolution levels `[(0.01, 0.05), (0.005, 0.02), (0.002, 0.01)]` with
  optional robust kernel at each scale
- [ ] Task 2.3 — Add `register_generalized()`: Open3D's
  `registration_generalized_icp` with
  `TransformationEstimationForGeneralizedICP()`; models local surface covariance
  to implicitly downweight uncertain neighbourhoods
- [ ] Task 2.4 — Add `register_trimmed()`: pre-filter source points by
  nearest-neighbour distance to target (removing fringe), then run standard
  point-to-plane ICP on the trimmed cloud. Falls back to full cloud if
  < 100 points survive
- [ ] Task 2.5 — Add `register_global_then_local()`: FPFH feature extraction +
  RANSAC global alignment, followed by point-to-plane ICP refinement. Handles
  large displacements where identity initialization fails

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/registration/forearm_registrator.py`
  — 5 new public methods

**Dependencies:** Phase 1

### Phase 3: Method Dispatch

**Goal:** Wire method selection into existing orchestration.

- [ ] Task 3.1 — Add `_METHOD_MAP` class-level dict mapping string keys
  (`"vanilla"`, `"robust"`, `"multiscale"`, `"generalized"`, `"trimmed"`,
  `"global"`) to method names
- [ ] Task 3.2 — Add `method: str = "vanilla"` parameter to `register_all()`;
  dispatch to corresponding `register_*()` for each non-canonical cloud
- [ ] Task 3.3 — Add `method: str = "vanilla"` parameter to
  `register_all_to_average()`; pass through to internal `register_all()` call
- [ ] Task 3.4 — Add `registration_method: str = "vanilla"` parameter to
  `register_session_forearms()` in `register_session_forearms.py`; pass through
  to `registrator.register_all(..., method=registration_method)` and
  `registrator.register_all_to_average(..., method=registration_method)`

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/registration/forearm_registrator.py`
  — dispatch logic in `register_all` / `register_all_to_average`
- `code/src/preprocessing/forearm_extraction/registration/register_session_forearms.py`
  — thread `registration_method` parameter

**Dependencies:** Phase 2

### Phase 4: Documentation Updates

**Goal:** Reconcile parameter discrepancy and document new methods.

- [ ] Task 4.1 — Reconcile `max_correspondence_distance` discrepancy: update
  the knowledge-base note's parameter table to match the actual code default
  (`0.10`) and add a rationale note
- [ ] Task 4.2 — Add a new section to `note-forearm-icp-registration.md`
  describing the 5 alternative methods, when each is appropriate, and
  `TukeyLoss(k)` parameter guidance
- [ ] Task 4.3 — Update the "Out of Scope" bullet in the parent plan
  (`multi-snapshot-forearm-registration.md`) to reflect that FPFH+RANSAC is
  now implemented as `register_global_then_local()`

**Files Modified:**
- `docs/development/knowledge-base/note-forearm-icp-registration.md`
- `docs/development/plans/active/multi-snapshot-forearm-registration.md`

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests

- [ ] `register_robust()` returns valid 4×4 transform and fitness > 0
- [ ] `register_multiscale()` converges on two clouds with known offset
- [ ] `register_generalized()` returns valid transform on clouds with normals
- [ ] `register_trimmed()` falls back to full cloud when < 100 points survive trimming
- [ ] `register_global_then_local()` handles clouds with larger displacement
- [ ] `register_all(method="vanilla")` produces identical output to current code
- [ ] `register_all(method="invalid")` raises `ValueError`

### Integration Tests

- [ ] `register_session_forearms(registration_method="robust")` runs end-to-end
  on a real multi-snapshot session
- [ ] Single-forearm sessions skip registration regardless of `method` parameter
- [ ] Output transform JSON includes correct method metadata

### Manual Verification

- [ ] Run each method on a real session, compare fitness and RMSE across all 6
  (5 new + vanilla baseline)
- [ ] Visually inspect alignment in registration viewer for each method
- [ ] Confirm `register_all_to_average` works with each method

### Edge Cases

- [ ] `register_trimmed()` with fully overlapping clouds (no fringe) — behaves
  like vanilla
- [ ] `register_multiscale()` with very small clouds (< 100 points after
  coarse downsampling)
- [ ] `register_global_then_local()` when RANSAC fails to find good alignment

---

## Documentation Plan

- [ ] Update `note-forearm-icp-registration.md` with method descriptions and
  parameter guidance
- [ ] Reconcile `max_correspondence_distance` discrepancy in knowledge-base note
- [ ] Add inline docstrings to all new methods

---

## Rollback Plan

All changes are additive Python methods and documentation updates.

1. **Before deployment:** Changes are on the existing
   `feature/multi-snapshot-forearm-registration` branch.
2. **Data considerations:** No new artifacts; existing artifacts unchanged.
   The default method remains `"vanilla"`, so no output changes unless
   explicitly requested.
3. **Rollback procedure:** Revert the commits adding alternative methods. The
   vanilla `register()` path and all existing orchestration remain functional.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Open3D API differences across versions for GICP or FPFH | Low | Medium | Pin to Open3D 0.19.0 (already a project dependency); test on import |
| `register_multiscale()` too slow for interactive use | Medium | Low | Coarse-to-fine reduces total iterations; only used when explicitly selected |
| `register_trimmed()` removes too many points on low-overlap sessions | Low | Medium | Fallback to full cloud when < 100 points survive; log warning |
| `TukeyLoss(k)` parameter needs per-session tuning | Medium | Low | Default `k=0.01` based on typical forearm residuals; exposed as parameter for manual override |

---

## References

- Parent plan: `docs/development/plans/active/multi-snapshot-forearm-registration.md`
- Knowledge-base: `docs/development/knowledge-base/note-forearm-icp-registration.md`
- `code/src/preprocessing/forearm_extraction/registration/forearm_registrator.py`
- `code/src/preprocessing/forearm_extraction/registration/register_session_forearms.py`
- [Open3D Robust Kernels Tutorial](https://www.open3d.org/docs/release/tutorial/pipelines/robust_kernels.html)
- [Open3D ICP Registration](https://www.open3d.org/docs/release/tutorial/t_pipelines/t_icp_registration.html)
