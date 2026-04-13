# Plan: RF Heatmap Vertex-Index Aggregation

**Date:** 2026-04-13
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/rf-heatmap-vertex-index-aggregation`

---

## Overview

Replace the exact-float coordinate groupby in the RF heatmap pipeline with integer vertex-index-based groupby. This eliminates scattered low-intensity points within high-intensity regions caused by serialization precision loss and cross-block coordinate instability.

## Problem Statement

The "map receptive fields simple" heatmap shows low-intensity (cold-colored) scatter points scattered **within** high-intensity (red) regions. Since touches are continuous, the coloring should be spatially continuous.

The root cause is the aggregation step in `rf_simple_pipeline.py:130-133`, which groups spike counts by exact `(x, y, z)` float coordinates parsed from `.1f`-rounded strings. Two mechanisms split counts for the same physical location into multiple entries:

1. **Serialization boundary straddling**: `serialize_contact_points()` (`csv_spatial_transformer.py:73`) rounds to `.1f` (0.1mm). A PLY vertex at `x=10.0500001` serializes as `10.1` in one block but after a slightly different PCA transform becomes `10.0499999` → `10.0`. The same physical vertex produces two entries with split spike counts.

2. **Cross-block vertex instability**: Different blocks use different forearm depth captures. After ICP registration + PCA calibration, the "same" physical location has slightly different coordinates. When projected to the reference PLY, points near a Voronoi boundary between two PLY vertices snap to different targets across blocks, splitting counts.

Both produce the observed symptom: spatially adjacent points with dramatically different spike counts in the rendered heatmap.

## Goals

### In Scope
1. Confirm the near-duplicate hypothesis with a diagnostic analysis
2. Fix spike count aggregation in `rf_simple_pipeline.py` to group by PLY vertex index
3. Apply the same fix to `rf_cluster_pipeline.py` for consistency

### Out of Scope
- Changing the `.1f` serialization precision in `serialize_contact_points()` (would require reprocessing all upstream data)
- Modifying the contact detection or projection pipeline
- Adding spatial smoothing or interpolation to the heatmap renderer

## Success Criteria

- [ ] Diagnostic confirms near-duplicate coordinate pairs with >5x spike count difference within 1mm radius
- [ ] Heatmap for affected sessions no longer shows scattered cold spots within hot regions
- [ ] Total spike count is preserved (counts merge, none are created or lost)
- [ ] Pipeline still works when forearm PLY is unavailable (graceful fallback to current behavior)

---

## Technical Design

### Approach

Snap each parsed spike position to its nearest forearm PLY vertex **by integer index** (KD-tree query), then group on that integer index instead of on float coordinates. Map vertex indices back to full-precision PLY coordinates for rendering.

This works because:
- Near-duplicates from serialization rounding snap to the **same** integer vertex index → counts merge correctly
- Cross-block coordinate instability is absorbed — same physical location → same nearest vertex
- No float comparison involved; groupby on integers is exact
- Rendering uses full-precision PLY coordinates (no `.1f` rounding artifacts)

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Vertex-index groupby (KD-tree snap) | Exact, no float comparison, reuses existing PLY | Requires PLY to be available | **Chosen** — PLY is already loaded for rendering |
| Increase serialization precision to `.3f` | Fixes boundary straddling at source | Requires reprocessing all upstream data; doesn't fix cross-block instability | Rejected |
| Radius-based spatial binning (grid) | Simple, no PLY dependency | Arbitrary bin size; bin boundaries create new artifacts | Rejected |
| Gaussian kernel smoothing on counts | Smooth visual result | Alters the data; harder to interpret quantitatively | Rejected |

### Architecture Changes

No new modules or classes. Changes are confined to the aggregation logic in two existing pipeline files. The forearm PLY (already resolved for rendering) is reused for vertex snapping.

---

## Implementation Plan

### Phase 1: Diagnostic
**Goal:** Confirm that the scattered cold spots are near-duplicates of hot spots

**Tasks:**
- [x] Task 1.1 — Add temporary diagnostic block in `rf_simple_pipeline.py` after `positions_df` is built (line 124): build KD-tree of unique positions, find neighbors within 1mm, flag pairs with >5x spike count difference, print summary
- [ ] Task 1.2 — Run the pipeline on an affected session, inspect diagnostic output

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_simple_pipeline.py` — temporary diagnostic code (removed in Phase 2)

**Dependencies:** None

### Phase 2: Fix simple pipeline
**Goal:** Replace exact-float groupby with vertex-index groupby

**Tasks:**
- [x] Task 2.1 — Move `resolve_forearm_ply()` call earlier (before aggregation, currently at line 135)
- [x] Task 2.2 — After building `positions_df`, load PLY vertices, build KD-tree, snap each position to nearest vertex index
- [x] Task 2.3 — Group by integer vertex index, then map back to full-precision PLY coordinates
- [x] Task 2.4 — Add fallback: if PLY unavailable or empty, use current exact-float groupby
- [x] Task 2.5 — Remove Phase 1 diagnostic code

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_simple_pipeline.py` — aggregation logic (lines ~124-134)

**Dependencies:** Phase 1 (confirms the hypothesis)

### Phase 3: Fix cluster pipeline
**Goal:** Apply the same vertex-index snapping to the cluster-based pipeline

**Tasks:**
- [ ] Task 3.1 — Identify the equivalent aggregation point in `rf_cluster_pipeline.py`
- [ ] Task 3.2 — Apply the same KD-tree snap + vertex-index groupby pattern
- [ ] Task 3.3 — Add the same PLY-unavailable fallback

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — aggregation logic

**Dependencies:** Phase 2

### Phase 4: Verification
**Goal:** Confirm the fix resolves the visual anomaly

**Tasks:**
- [ ] Task 4.1 — Re-run `map_receptive_fields_simple` on the affected session
- [ ] Task 4.2 — Visually compare old vs new heatmap — cold spots within hot regions should disappear
- [ ] Task 4.3 — Verify total spike count is preserved (sum of spike_count before and after should match)

**Dependencies:** Phase 2

---

## Testing Plan

### Manual Verification
- [ ] Run pipeline on a session exhibiting scattered cold spots; confirm they disappear
- [ ] Compare total spike count before/after fix (must be identical)
- [ ] Run on a session without a forearm PLY; confirm fallback to current behavior with no crash
- [ ] Run cluster pipeline on a session with clusters; confirm heatmaps are similarly improved

### Edge Cases
- [ ] Session with no forearm PLY — fallback to exact-float groupby
- [ ] Session with very few spike positions (< 5) — should still render correctly
- [ ] Session where all positions snap to the same PLY vertex — single point with total count

---

## Documentation Plan

- [ ] No external documentation changes needed (internal pipeline improvement)
- [ ] Inline comments explaining the vertex-index snapping rationale

---

## Rollback Plan

1. Revert the changes to `rf_simple_pipeline.py` and `rf_cluster_pipeline.py`
2. No data migrations or breaking changes — the fix only affects in-memory aggregation, not persisted files
3. `spike_positions.csv` output is unchanged (raw positions, not aggregated)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Forearm PLY has very different vertex set than contact points (e.g., wrong coordinate space) | Low | High — all points snap to wrong vertices | Validate by checking mean snap distance; warn if > 2mm |
| KD-tree query adds noticeable processing time | Low | Low — PLY has ~50k vertices, query is O(n log n) | Acceptable for offline analysis pipeline |
| Some sessions lack forearm PLY entirely | Medium | None — graceful fallback | Fallback to current exact-float groupby |

---

## References

- Knowledge base: `docs/development/knowledge-base/note-forearm-icp-registration.md` (cross-block registration)
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md` (mm units)
- Serialization: `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py:59-74`
- Projection: `code/scripts/_5_postprocessing/project_contacts_onto_forearm.py`
