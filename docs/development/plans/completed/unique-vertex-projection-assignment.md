# Plan: Unique Vertex Projection Assignment

**Created:** 2026-04-13
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/unique-vertex-projection-assignment`

---

## Overview

**What:** Enforce one-to-one mapping between contact points and forearm vertices during the projection step, so that no two contact points within the same row snap to the same forearm vertex.

**Why:** The current nearest-neighbor projection allows multiple contact points to collapse onto a single vertex, which is physically incorrect — each contact location on the forearm should correspond to a distinct surface point.

**How:** Replace the independent per-point KD-tree query with a `scipy.optimize.linear_sum_assignment` call on a small candidate cost matrix built from k-nearest neighbors.

## Problem Statement

In `project_contacts_onto_forearm.py`, Stage 4 of the postprocessing pipeline, each contact point in a row's `contact_points` field is independently snapped to its nearest forearm PLY vertex via `kdtree.query(query)`. When two or more contact points are spatially close on the forearm surface, they can snap to the **same** vertex. This produces duplicate vertex coordinates in the output, which:

- Misrepresents the spatial extent of the contact area
- Causes downstream mean-location (`contact_location_x/y/z`) to collapse toward a single vertex
- Can bias receptive field heatmaps by over-counting a single vertex

## Goals

### In Scope
1. Enforce unique vertex assignment within each row of `_project_single_csv()`
2. Use `scipy.optimize.linear_sum_assignment` for optimal assignment
3. Preserve the existing interface contract (`distances`, `indices` arrays with same shape/semantics)

### Out of Scope
- Cross-row uniqueness (different frames can legitimately touch the same vertex)
- Changes to `serialize_contact_points()` or its `.1f` rounding
- Changes to the outer `project_contacts_onto_forearm()` function
- Changes to downstream analysis pipelines (handled separately by the vertex-index aggregation plan)

## Success Criteria

- [ ] No two contact points in the same output row share the same forearm vertex index
- [ ] Single-point rows (M=1) produce identical output to the current implementation
- [ ] Projection stats (`projection_stats.csv`) show negligibly different distances (collisions are rare)
- [ ] Pipeline runs without errors on all existing sessions

---

## Technical Design

### Approach

For each row with M > 1 contact points:
1. Query `k = M` nearest neighbors per point via `kdtree.query(query, k=M)` — returns `(M, M)` distance and index arrays
2. Collect the union of all candidate vertex indices (`np.unique`)
3. Compute the full distance matrix from each point to each candidate vertex via `scipy.spatial.distance.cdist`
4. Solve the optimal assignment via `scipy.optimize.linear_sum_assignment(cost)` — returns row and column indices minimizing total distance
5. Map column indices back to vertex indices

For M = 1: use the existing `kdtree.query(query)` path unchanged (no collision possible).

This guarantees the **globally optimal** one-to-one assignment (minimum total displacement) using a well-tested scipy routine.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `linear_sum_assignment` on k-nearest candidate cost matrix | Optimal total distance, clean code (~6 lines), well-tested scipy routine | 2 additional scipy imports | **Chosen** |
| Greedy closest-first (sort by distance, assign greedily) | No extra imports, simple loop | Near-optimal but not guaranteed optimal; more code (~15 lines) | Rejected |
| Hungarian on full (M × N) cost matrix | Truly global optimal | Impractical: N ≈ 50k vertices, cost matrix too large | Rejected |
| Iterative re-query (query k=1, detect collision, re-query with k=2, ...) | No extra imports | Multiple KDTree calls, awkward loop termination, unclear correctness | Rejected |

### Architecture Changes

No new modules or classes. Changes confined to one function in one file:

```
code/scripts/_5_postprocessing/
└── project_contacts_onto_forearm.py  — modify _project_single_csv() lines 84–87
```

Two new imports at the top of the file:
- `from scipy.optimize import linear_sum_assignment`
- `from scipy.spatial.distance import cdist`

---

## Implementation Plan

### Phase 1: Core Change
**Goal:** Replace independent nearest-neighbor with unique assignment in `_project_single_csv()`
**Started:** 2026-04-13
**Completed:** 2026-04-13

**Tasks:**
- [x] Task 1.1 — Add imports: `from scipy.optimize import linear_sum_assignment` and `from scipy.spatial.distance import cdist`
- [x] Task 1.2 — Replace lines 84–87 in `_project_single_csv()`: add M=1 fast path (existing behavior) and M>1 path using `kdtree.query(query, k=M)` + `cdist` + `linear_sum_assignment`
- [x] Task 1.3 — Verify `distances` and `indices` arrays have the same shape and semantics as before (no downstream changes needed)

**Files Modified:**
- `code/scripts/_5_postprocessing/project_contacts_onto_forearm.py` — add 2 imports, replace ~4 lines with ~12 lines in `_project_single_csv()`

**Dependencies:** None

### Phase 2: Verification
**Goal:** Confirm the fix works on real data
**Started:** —
**Completed:** —

**Tasks:**
- [ ] Task 2.1 — Run the postprocessing pipeline on a session with multi-point contact rows
- [ ] Task 2.2 — Inspect output CSVs: verify no duplicate vertex coordinates within any row's `contact_points`
- [ ] Task 2.3 — Compare `projection_stats.csv` before/after — confirm distances are nearly identical

**Files Modified:** None (verification only)

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Run `project_contacts_onto_forearm` on a session known to have multi-point contact rows
- [ ] Inspect output CSV: parse `contact_points` per row, confirm all coordinates are unique
- [ ] Compare projection stats before/after — mean displacement should be nearly unchanged

### Edge Cases
- [ ] Single contact point per row (M=1) — must produce identical output to current code
- [ ] Two contact points with the same nearest vertex — one gets nearest, other gets second-nearest
- [ ] Row with empty `contact_points` — passes through unchanged (guard at line 77)
- [ ] All M points share the same nearest vertex — each assigned a distinct vertex from the candidate pool

---

## Documentation Plan

- [ ] No external documentation changes needed (internal pipeline improvement)
- [ ] Inline comment explaining why `linear_sum_assignment` is used instead of plain `kdtree.query`

---

## Rollback Plan

1. Revert the single commit modifying `project_contacts_onto_forearm.py`
2. No data migration needed — output CSV format is unchanged
3. No interface changes — downstream consumers are unaffected

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `k=M` nearest neighbors don't include enough distinct vertices for all points | Very Low | Med — assignment would fail | With M ≤ 5 and ~50k vertices, k=M always provides enough unique candidates. Could increase k as a safety margin if needed. |
| `linear_sum_assignment` adds noticeable overhead | Very Low | Low — offline pipeline | Cost matrix is at most (5 × 25); assignment is microseconds |
| `cdist` import adds unexpected dependency | None | None | `scipy.spatial.distance` is already available alongside `scipy.spatial.KDTree` |

---

## References

- Completed plan: `docs/development/plans/completed/contact-point-forearm-projection.md` (original projection implementation)
- Active plan: `docs/development/plans/active/rf-heatmap-vertex-index-aggregation.md` (downstream aggregation fix)
- Knowledge base: `docs/development/knowledge-base/note-spatial-alignment-pipeline.md` (spatial pipeline reference)
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md` (mm units)
- scipy docs: `scipy.optimize.linear_sum_assignment`
