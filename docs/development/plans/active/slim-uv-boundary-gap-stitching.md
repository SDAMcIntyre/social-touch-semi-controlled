# Plan: Boundary-Gap Stitching for Forearm SLIM UV

**Date:** 2026-05-21
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/slim-uv-delaunay-hole-filling`
**Branch:** `feature/slim-uv-boundary-gap-stitching`

---

## Overview

**What:** Add boundary-gap stitching to `clean_mesh()` so that narrow gaps at appendage junctions
(e.g., thumb–forearm) are welded rather than filled, preserving the appendage as part of the outer
boundary loop.

**Why:** Session ST14-02 has 10 interior holes created by non-manifold/pinch/sliver removal at the
thumb–forearm junction. `_fill_interior_holes()` fills all non-largest loops, inadvertently
enclosing the thumb — the boundary shortcuts across the thumb base. This produces extreme conformal
distortion (sigma1/sigma2 >> 3.0 vs ~1.175 in correctly-bounded sessions), flipped triangles,
and mesh trimming.

**How:** Before hole filling, identify secondary boundary loops whose vertices are close to the main
boundary (junction gaps), and weld those vertices to their nearest main-boundary counterparts. This
merges the secondary loop into the main boundary. Remaining true interior holes are then filled
normally.

## Problem Statement

`_fill_interior_holes()` classifies all non-largest boundary loops as interior holes and fills them
with Delaunay/fan triangulation. This is correct for true interior holes (created by non-manifold
removal deep inside the mesh) but incorrect for junction gaps — narrow strips of missing faces
where an appendage (thumb, finger) connects to the main forearm body.

When a junction gap is filled, the secondary loop that includes both the gap edges AND the
appendage perimeter is triangulated entirely. This converts the appendage perimeter from boundary
to interior, making the main boundary loop bypass the appendage.

**Downstream effects in ST14-02:**
- Step 3 diagnostic: boundary (yellow dots) skips the thumb
- Step 4: Tutte init required (harmonic produced flips)
- Step 5: mesh trimmed to remove flipped triangles
- Step 6: conformal distortion >> 3.0 in thumb region (normal is ~1.2)

**Comparison:** ST14-01 (same participant, different neuron) has no junction gaps by chance — its
boundary correctly traces around the thumb with max conformal distortion ~1.175.

## Goals

### In Scope
1. Weld secondary boundary loops adjacent to the main boundary (junction gaps) before hole filling
2. Preserve disk topology requirement (single boundary loop after stitching + filling)
3. Maintain backward compatibility — sessions without gaps are unaffected (no-op path)
4. Add unit tests for the stitching logic

### Out of Scope
- Pruning the thumb or other appendages from the mesh
- Modifying the upstream forearm PLY segmentation
- Changing `_fill_interior_holes()` logic (it continues to fill remaining true interior holes)
- Modifying diagnostic figure layout or step numbering
- Adjusting SLIM solver parameters or trim-retry thresholds

## Success Criteria

- [ ] ST14-02 step 3 diagnostic shows boundary loop tracing around the thumb
- [ ] ST14-02 step 6 conformal distortion in thumb region drops from >> 3.0 to normal range (~1.2)
- [ ] All 12 sessions run `precompute_forearm_slim_uv` to completion without error
- [ ] Sessions without junction gaps are unaffected (stitching is a no-op)
- [ ] All existing tests in `test_forearm_slim_uv.py` pass without modification
- [ ] New `TestStitchBoundaryGaps` tests pass

---

## Technical Design

### Approach

Vertex welding: for each secondary boundary loop, query a KDTree of main-boundary vertex positions.
If >= 2 secondary vertices are within `_BOUNDARY_GAP_PROXIMITY_MM` (5.0 mm) of the main boundary,
replace each close secondary vertex ID with its nearest main-boundary vertex ID in the face array.
This collapses the gap to zero width, merging the two loops into one. Degenerate faces (where 2+
vertices collapse to the same ID) are removed. The process iterates up to 3 passes because welding
can reveal new boundary configurations.

After stitching, `_fill_interior_holes()` fills any remaining true interior holes (those far from
the main boundary) as before.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Vertex welding (KDTree proximity) | Simple (~50 lines), no new faces, naturally merges loops | Shifts gap vertices by up to 5 mm | **Chosen** |
| Bridge triangulation (zig-zag across gap) | Preserves vertex positions, explicit gap filling | Complex pairing logic, new faces, winding issues | Rejected — higher complexity for marginal benefit |
| Skip filling perimeter-adjacent loops | Conceptually simple | Breaks disk topology (multiple boundary loops remain) | Rejected — violates Tutte/harmonic init requirement |
| Prune appendages from mesh | Removes root cause | User wants thumb in mesh; auto-detection complex | Rejected by user |

### Architecture Changes

No new modules or classes. Changes are localised to existing files:

```
surface/
  slim_helpers.py       -- new _stitch_boundary_gaps(); insert call in clean_mesh()
tests/
  test_forearm_slim_uv.py -- new TestStitchBoundaryGaps class
```

### Knowledge Base Constraints

From `bug-slim-uv-non-manifold-flip.md`:
- Disk topology (chi = 1) is mandatory for harmonic and Tutte init — stitching must preserve this
- Iterative passes may be needed (filling can create new anomalies) — stitching follows this pattern
- `_find_boundary_loops()` is the canonical loop detection function — reuse it
- `trimesh.repair.fix_winding()` must be called after topology changes

From `note-mesh-parameterization-interior-pin-foldovers.md`:
- No interior vertex pinning — stitching operates only on boundary vertices
- Single-boundary topology is a precondition for Rado-Kneser-Choquet bijectivity guarantee

---

## Implementation Plan

### Phase 1: Core stitching logic
**Goal:** Implement boundary-gap stitching and insert it into the `clean_mesh()` pipeline.
**Started:** 2026-05-21
**Completed:** 2026-05-21

- [x] Add `_BOUNDARY_GAP_PROXIMITY_MM = 5.0` constant to `slim_helpers.py`
- [x] Implement `_stitch_boundary_gaps(V, F, proximity_mm, max_passes)` (~50 lines):
  KDTree proximity query on main boundary, classify secondary loops as gap vs interior hole,
  weld close vertex pairs, remove degenerate faces, fix winding, take largest component
- [x] Insert `_stitch_boundary_gaps` call in `clean_mesh()` between `_remove_sliver_faces` and
  `_fill_interior_holes`
- [x] Add `TestStitchBoundaryGaps` class to `test_forearm_slim_uv.py`:
  `test_no_secondary_loops_noop`, `test_gap_welded_into_main_boundary`,
  `test_true_interior_hole_not_welded`, `test_degenerate_faces_removed`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/surface/slim_helpers.py` — new function + constant +
  call in `clean_mesh`
- `code/tests/test_forearm_slim_uv.py` — new test class

**Dependencies:** None (builds on existing `_find_boundary_loops`, KDTree, trimesh patterns)

### Phase 2: Validation
**Goal:** Verify correctness across all 12 sessions with no regressions.
**Started:** 2026-05-21

**Note:** Tasks 1–4 require manual pipeline runs with actual session data.

- [ ] Run `precompute_forearm_slim_uv` with `save_diagnostics: true, force_processing: true`
  on ST14-02 — step 3 should show boundary around thumb
- [ ] Compare step 6 distortion for ST14-02: conformal distortion in thumb region should drop
  from >> 3.0 to normal range
- [ ] Run all 12 sessions end-to-end — all must pass without error
- [ ] Verify sessions without junction gaps show no change (stitching is a no-op)
- [x] Run existing test suite: `pytest code/tests/test_forearm_slim_uv.py -v`

**Files Modified:** None (validation only)

**Dependencies:** Phase 1

### Phase 3: Documentation
**Goal:** Update knowledge base with the new stitching step.
**Started:** 2026-05-21
**Completed:** 2026-05-21

- [x] Update `docs/development/knowledge-base/bug-slim-uv-non-manifold-flip.md` — add note that
  boundary-gap stitching was added before hole filling to prevent enclosure of appendages

**Files Modified:**
- `docs/development/knowledge-base/bug-slim-uv-non-manifold-flip.md` — append stitching note

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] `test_no_secondary_loops_noop` — single-boundary disk mesh; V, F returned unchanged
- [ ] `test_gap_welded_into_main_boundary` — T-shaped mesh with narrow gap at junction; after
  stitching, single boundary loop traces around appendage; `_find_boundary_loops` returns 1 loop
- [ ] `test_true_interior_hole_not_welded` — mesh with interior hole far from boundary; hole
  remains after stitching (still 2 boundary loops); subsequently filled by `_fill_interior_holes`
- [ ] `test_degenerate_faces_removed` — welding that creates faces with 2+ identical vertices;
  degenerate faces are removed, mesh remains valid

### Integration Tests
- [ ] Existing `TestFlattenSlim` tests pass unchanged
- [ ] Existing `TestPrecomputeForearmSlimUv` tests pass unchanged
- [ ] Existing `TestFillInteriorHoles` tests pass unchanged
- [ ] Existing `TestSliverRemoval` tests pass unchanged

### Manual Verification
- [ ] ST14-02: step 3 boundary includes thumb (visual check of diagnostic PNG)
- [ ] ST14-02: step 6 conformal distortion reduced in thumb region (visual check)
- [ ] ST14-01: boundary still includes thumb (no regression)
- [ ] All 12 sessions: pipeline completes without error

### Edge Cases
- [ ] Session with no secondary loops (most sessions) — stitching is a no-op
- [ ] Gap with only 1 close vertex — not stitched (< 2 threshold); filled by `_fill_interior_holes`
- [ ] Gap wider than proximity threshold — not stitched; filled normally
- [ ] Welding fragments the mesh — largest connected component kept

---

## Documentation Plan

- [x] Update `docs/development/knowledge-base/bug-slim-uv-non-manifold-flip.md` with stitching note

---

## Rollback Plan

All changes are additive or internal refactors of private functions:

1. Remove `_stitch_boundary_gaps` function and its call from `clean_mesh()` in `slim_helpers.py`
2. Remove `TestStitchBoundaryGaps` class from `test_forearm_slim_uv.py`
3. Revert knowledge base note
4. Re-run `precompute_forearm_slim_uv` with `force_processing: true` to regenerate caches

No data migrations. No public API changes. Cached `.npz` files are regenerated on next run.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Welding creates non-manifold vertices | Low | Med | Degenerate face removal + winding fix + largest-component selection after each pass |
| Proximity threshold too aggressive (false welds on curved boundary) | Low | Med | 5 mm is conservative relative to BPA edge length (1-3 mm); `n_close >= 2` guard prevents single-vertex false positives |
| Welding distorts mesh at gap | Low | Low | Vertex shift <= 5 mm, comparable to BPA resolution |
| Stitching changes boundary for sessions that were previously correct | Low | High | Sessions without secondary loops are untouched (no-op); validate all 12 sessions |
| New non-manifold edges after welding | Low | Low | Handled by downstream `_fill_interior_holes` + `flatten_slim` trim-retry |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Core stitching + tests | ~45 min | None |
| Phase 2: Validation (12 sessions) | ~30 min | Phase 1 |
| Phase 3: Documentation | ~10 min | Phase 2 |

---

## References

- Knowledge base: `docs/development/knowledge-base/bug-slim-uv-non-manifold-flip.md`
- Knowledge base: `docs/development/knowledge-base/note-mesh-parameterization-interior-pin-foldovers.md`
- Related plan: `docs/development/plans/active/slim-uv-delaunay-hole-filling.md`
