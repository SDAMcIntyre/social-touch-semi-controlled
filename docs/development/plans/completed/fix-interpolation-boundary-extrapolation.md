# Plan: Fix Interpolation Boundary Extrapolation

**Date:** 2026-05-01
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-04 14:39
**Base Branch:** `feature/rf-explorer-independent-dag-flow`
**Branch:** `feature/rf-explorer-independent-dag-flow`

---

## Overview

The preparation stage's cubic NaN interpolation extrapolates beyond the forearm
surface at touch-event boundaries, producing contact locations up to 29mm from
the nearest mesh vertex.  This blocks the RF explorer, which enforces a 15mm
threshold.  The fix adds `limit_area='inside'` to restrict interpolation to
interior NaN gaps only — no boundary extrapolation.

## Problem Statement

`interpolate_touch_columns()` fills 1kHz NaN gaps between 30Hz Kinect samples
using cubic spline interpolation with `limit_direction='both'`.  At the edges of
each touch group (before the first Kinect sample, after the last), this
**extrapolates** the cubic spline outward.  Cubic extrapolation diverges rapidly,
pushing `contact_location_x/y/z` coordinates off the forearm surface.

248 of 213,937 frames in the prepared CSV exceed the 15mm nearest-vertex
threshold (max 29.34mm), raising a `ValueError` in `load_explorer_data` and
blocking the entire RF feature-space explorer.

The existing post-interpolation guard only protects `contact_depth` and
`contact_area` (clamped to >= 0).  Spatial coordinates have no guard.

See: `docs/development/knowledge-base/bug-rf-explorer-nearest-vertex-distance.md`

## Goals

### In Scope

1. Eliminate cubic extrapolation at touch-group boundaries
2. Update the knowledge-base bug report to reflect the fix

### Out of Scope

- Changing the interpolation method (cubic remains for interior gaps)
- Adding a spatial overshoot guard analogous to the depth/area clamp
- Modifying the 15mm threshold in the RF explorer data loader

## Success Criteria

- [ ] RF explorer runs without the 15mm `ValueError`
- [ ] `diagnose_rf_explorer_distance.py` reports zero frames exceeding 15mm in
      the prepared CSV
- [ ] No regression in downstream analysis stages (series transforms, feature
      extraction, clustering)

---

## Technical Design

### Approach

Add `limit_area='inside'` to the two `pd.Series.interpolate()` calls in
`_interpolate_group`.  This pandas parameter restricts interpolation to NaN
values that sit **between** two valid anchor values — no extrapolation at group
edges.  `limit_direction='both'` is retained (it controls direction within the
valid interior range).

Boundary frames (before first Kinect sample / after last in each touch group)
stay NaN.  These are naturally dropped by downstream NaN filters.  They represent
partial contact at onset/offset — the least reliable measurement frames.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `limit_area='inside'` | 1-param change, eliminates root cause, no false data | Loses ~248 boundary frames | **Chosen** |
| Linear interpolation for spatial columns | Reduces overshoot, no data loss | Still extrapolates at boundaries (less aggressively), loses smoothness | Rejected |
| Bounding-box clamp per group | Preserves cubic, no data loss | Creates flat segments, doesn't guarantee surface proximity | Rejected |
| `inside` + ffill/bfill at boundaries | No extrapolation, no data loss | Slight discontinuity at held-value boundary | Rejected (unnecessary complexity) |
| Surface-distance snap (nearest vertex) | Most physically accurate | Requires mesh access in preparation stage (currently mesh-agnostic) | Rejected (architectural change) |

### Architecture Changes

None.  Single parameter addition to two existing function calls.  No new modules,
classes, or interfaces.

---

## Implementation Plan

### Phase 1: Fix and Document

**Goal:** Stop boundary extrapolation and update the bug report.

**Started:** 2026-05-01
**Completed:** 2026-05-01

- [x] Task 1.1 — Add `limit_area='inside'` to the linear-fallback interpolation
      call (line 94)
- [x] Task 1.2 — Add `limit_area='inside'` to the primary method interpolation
      call (line 96)
- [x] Task 1.3 — Update `bug-rf-explorer-nearest-vertex-distance.md` Status
      section to reflect the fix applied

**Files Modified:**

- `code/src/analysis/touch_analytics/preparation/interpolation.py` — add
  `limit_area='inside'` parameter to lines 94 and 96
- `docs/development/knowledge-base/bug-rf-explorer-nearest-vertex-distance.md` —
  update Status section

**Dependencies:** None

---

## Testing Plan

### Manual Verification

- [ ] Re-run the analysis workflow with `explore_rf_feature_space` enabled —
      the `ValueError` about 248 frames exceeding 15mm must not appear
- [ ] Run `code/scripts/diagnose_rf_explorer_distance.py` against the freshly
      generated `_prepared.csv` — confirm zero frames exceed 15mm
- [ ] Spot-check that the prepared CSV row count is marginally lower (the ~248
      boundary frames are now NaN and filtered downstream)

### Edge Cases

- [ ] Touch groups with fewer than 4 valid Kinect samples (triggers linear
      fallback) — confirm `limit_area='inside'` works with the linear path
- [ ] Touch groups with exactly 1 valid sample (triggers ffill/bfill path) —
      this path is unaffected since it doesn't use `.interpolate()`

---

## Documentation Plan

- [ ] Update `bug-rf-explorer-nearest-vertex-distance.md` Status section

No CLAUDE.md or README changes needed — this is a bugfix with no API or
architecture change.

---

## Rollback Plan

Revert the single commit.  The two-parameter addition is fully isolated;
removing `limit_area='inside'` restores the previous behavior exactly.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Downstream stages expect boundary frames that are now NaN | Low | Med | Boundary frames were already unreliable; NaN filters exist at every stage boundary |
| `limit_area='inside'` interacts unexpectedly with `limit_direction='both'` | Low | Low | pandas docs confirm they compose correctly; `inside` restricts area, `both` controls direction within it |

---

## References

- Bug report: `docs/development/knowledge-base/bug-rf-explorer-nearest-vertex-distance.md`
- Diagnostic script: `code/scripts/diagnose_rf_explorer_distance.py`
