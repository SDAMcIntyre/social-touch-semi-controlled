# Plan: RF Center — Top Spike Count Filter

**Date:** 2026-04-17
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/rf-center-top-spike-filter`

---

## Overview

The RF center is currently computed as the selectivity-weighted centroid of all points in the dominant DBSCAN cluster. This change narrows the centroid calculation to only the top 10% of points by raw spike count, weighted by spike count. This focuses the RF center on the most active sub-region of the receptive field rather than diluting it with low-activity peripheral points.

## Problem Statement

The current approach includes every point that passed the selectivity threshold (0.3) and landed in the largest DBSCAN cluster. Points at the cluster periphery with few spikes pull the centroid away from the true hotspot. For neurons with spatially extended but unevenly active receptive fields, this produces an RF center that may not correspond to the peak-activity region.

## Goals

### In Scope
1. Filter dominant DBSCAN cluster points to the top 10% by raw spike count before computing the RF center
2. Weight the centroid by raw spike count (not selectivity)
3. Record the filtering parameters in the output metadata JSON

### Out of Scope
- Modifying the DBSCAN clustering logic or selectivity threshold
- Changing the `RFCluster` dataclass or `RFMappingEngine`
- Applying this filter to the RF heatmap visualisation pipeline
- Making the 10% threshold configurable via DAG YAML config

## Success Criteria

- [ ] RF center is computed from only the top 10% spike count points in the dominant cluster
- [ ] `rf_center_origin.json` includes `rf_center_top_fraction` and `rf_center_top_n_points` fields
- [ ] Graceful fallback: if all top spike counts are zero, unweighted mean is used
- [ ] Existing pass-through behaviour on DBSCAN failure is unchanged

---

## Technical Design

### Approach

After DBSCAN selects the dominant cluster, look up the raw spike count for each cluster point from the `gsd.spike_counts` Counter (already in scope in `_compute_rf_center()`). Sort by spike count, keep the top 10% (minimum 1 point), and compute the centroid weighted by raw spike count.

This is a localised change — the `gsd.spike_counts` Counter is available at the exact location where the centroid is computed (line 117), so no data needs to be threaded through new parameters or dataclass fields.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Filter in `_compute_rf_center()` using existing `gsd.spike_counts` | Zero blast radius, no API changes, Counter already in scope | Filtering logic is script-specific, not reusable | **Chosen** — reusability not needed |
| Add `spike_counts` array to `RFCluster` dataclass | Reusable by other consumers | Wider blast radius, no other consumer needs it | Rejected |
| Replace DBSCAN entirely with top-10% global filter | Simpler pipeline | Loses spatial coherence, may include scattered outliers | Rejected |
| Use selectivity instead of spike count for filtering | Consistent with existing weighting | Doesn't match user intent (focus on raw activity) | Rejected |

### Architecture Changes

None. Single function modification within existing script. No new modules, classes, or interfaces.

### Knowledge Base Constraints

- **Units (mm):** RF center coordinates remain in mm. No conversion needed.
- **Spatial alignment pipeline:** RF centering is the final transform (Space 4 → Space 5). This change only affects centroid point selection, not the transform itself.
- **Metadata structure:** Existing JSON fields preserved; two new fields added.
- **Graceful no-op:** Pass-through fallback on DBSCAN failure remains unchanged.

---

## Implementation Plan

### Phase 1: Modify RF Center Computation
**Goal:** Filter to top 10% spike count points and compute spike-count-weighted centroid.

**Tasks:**
- [ ] Task 1.1 — Replace centroid computation (lines 117–120) with spike-count filtering + weighted average
- [ ] Task 1.2 — Add `rf_center_top_fraction` and `rf_center_top_n_points` to metadata dict (lines 122–131)
- [ ] Task 1.3 — Update function docstring (lines 40–55) to describe new behaviour

**Files Modified:**
- `code/scripts/_5_postprocessing/center_on_receptive_field.py` — lines 40–55 (docstring), 117–131 (centroid + metadata)

**Dependencies:** None

**Code change (lines 117–120 replaced with):**

```python
TOP_FRACTION = 0.1

dominant = max(rf_result.clusters, key=lambda c: c.point_count)

spike_counts_arr = np.array([
    gsd.spike_counts.get(tuple(pt), 0)
    for pt in dominant.points
])

n_top = max(1, int(np.ceil(len(spike_counts_arr) * TOP_FRACTION)))
top_indices = np.argsort(spike_counts_arr)[-n_top:]

top_points = dominant.points[top_indices]
top_weights = spike_counts_arr[top_indices]

if top_weights.sum() > 0:
    rf_center = np.average(top_points, weights=top_weights, axis=0)
else:
    rf_center = np.mean(top_points, axis=0)
```

**Edge cases:**

| Scenario | Handling |
|----------|----------|
| Cluster has < 10 points (10% < 1) | `max(1, ceil(N * 0.1))` guarantees at least 1 point |
| All top spike counts are 0 | Fallback to unweighted mean (extremely unlikely after selectivity > 0.3 filter) |
| Ties at the 10% boundary | `np.argsort` breaks ties by array order — acceptable since tied points have equal spike counts |
| Single-point cluster | `n_top = 1`, center is that point |

---

## Testing Plan

### Manual Verification
- [ ] Run postprocessing pipeline on a test session
- [ ] Inspect `rf_center_origin.json` — confirm `rf_center_top_fraction: 0.1` and `rf_center_top_n_points` fields present
- [ ] Compare old vs new RF center coordinates — new center should shift toward the highest-activity sub-region
- [ ] Visually inspect RF heatmap to confirm center aligns with the spike count hotspot
- [ ] Verify pass-through behaviour still works when DBSCAN finds no clusters (test with a session that has sparse spike data)

### Edge Cases
- [ ] Session with very small dominant cluster (< 10 points) — verify at least 1 point is used
- [ ] Session where RF estimation fails — verify unchanged pass-through and metadata

---

## Documentation Plan

- [ ] No external documentation changes needed — this is an internal algorithmic refinement
- [ ] The `rf_center_origin.json` metadata self-documents the new parameters

---

## Rollback Plan

1. Revert the single commit on `center_on_receptive_field.py`
2. Re-run the postprocessing pipeline to regenerate RF-centered outputs with the old centroid logic
3. No data migration needed — outputs are regenerated from upstream CSVs

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Top 10% filter is too aggressive for small clusters | Low | Low | `max(1, ...)` guarantees at least 1 point; DBSCAN `min_cluster_points=5` ensures minimum cluster size |
| Spike count lookup misses points due to float tuple key mismatch | Low | Med | Points in `dominant.points` originated from `gsd.spike_counts` keys via the same parsing path — keys match by construction |
| RF center shifts significantly, affecting downstream analysis | Med | Low | Expected behaviour; compare before/after on a few sessions to validate the shift is toward the hotspot |

---

## References

- Related file: `code/scripts/_5_postprocessing/center_on_receptive_field.py`
- Knowledge base: `docs/development/knowledge-base/note-spatial-alignment-pipeline.md`
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`

---
