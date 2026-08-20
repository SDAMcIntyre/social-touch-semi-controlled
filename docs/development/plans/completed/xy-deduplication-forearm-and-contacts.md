# Plan: XY-Deduplication for Forearm and Contact Points

**Date:** 2026-05-05
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-08-20 14:39
**Started:** 2026-05-05
**Base Branch:** `feature/contact-projection-first-architecture`
**Branch:** `feature/contact-projection-first-architecture`

---

## Overview

Add a new initial postprocessing step that deduplicates both the forearm
pointcloud vertices and the per-frame contact_points by (x,y) position.
When multiple points share the same (x,y) within epsilon=0.1mm but have
different z values, only the point with the lowest z (closest to the Kinect
sensor — the outermost forearm surface) is kept.

## Problem Statement

The Kinect depth sensor can produce layered points at the same (x,y) surface
location with slightly different depths.  These duplicate-in-xy points:

1. In the **forearm pointcloud**: create ambiguous KDTree targets — a
   contact point may snap to an inner-layer vertex instead of the true
   surface.
2. In the **contact_points**: since they are selected from forearm mesh
   vertices that penetrate the hand (`objects_interaction_processor.py:175`),
   the same (x,y) duplication can propagate into the contact point set.

The lowest-z point is the geometrically correct outer surface (Kinect z-axis =
depth from sensor; lower = closer = outermost).

## Goals

### In Scope
1. Deduplicate forearm PLY vertices by (x,y) within epsilon, keeping lowest z.
2. Deduplicate contact_points in merged CSVs per frame, same logic.
3. Integrate as the first step (Step 0) of the postprocessing pipeline, before
   the existing projection step.
4. Log statistics on how many points were removed.

### Out of Scope
- Modifying the projection algorithm (already refactored by the
  contact-projection-first plan).
- Changing how `objects_interaction_processor.py` generates contact_points
  (upstream preprocessing).
- Configurable epsilon per session (fixed at 0.1mm for now).

## Success Criteria

- [ ] No forearm PLY in `forearm_deduped/` contains two vertices with
  `|(x1,y1) - (x2,y2)| < 0.1mm` and different z values.
- [ ] No `contact_points` cell in `blocks_merged_deduped/` contains two points
  with `|(x1,y1) - (x2,y2)| < 0.1mm` and different z values.
- [ ] `assert len(deduped) + n_removed == len(original)` holds for every call.
- [ ] Full postprocess pipeline completes without error after integration.
- [ ] Stats logging reports removal counts (informational — any value is valid).

---

## Technical Design

### Approach

**Core algorithm — `deduplicate_xy`:**
1. Quantize (x,y) to bins of size epsilon: `bin_xy = np.round(points[:, :2] / epsilon).astype(int64)`
2. Group points by `(bin_x, bin_y)` key
3. For each group with >1 point, keep the point with `argmin(z)`
4. Return `(deduped_points, n_removed)`

This is applied twice per block:
- On the forearm PLY vertices (thousands of points)
- On each row's `contact_points` in the merged CSV (typically 5–50 points per frame)

**Pipeline integration:**
```
blocks_merged/ ──[Step 0: deduplicate_xy]──► blocks_merged_deduped/
per-block forearm PLY ──[dedup]──► forearm_deduped/
                                         │
                   [Step 1: project onto deduped forearm]
                                         │
                                blocks_merged_projected/
                                         │
                                   ICP → PCA → ...
```

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Quantize+group per-point (proposed) | O(N), simple, deterministic | Bin-boundary edge cases (mitigated by rounding) | **Chosen** |
| KDTree radius query to find duplicates | Exact within epsilon | O(N log N), complex merging logic, overkill | Rejected |
| Filter at source in `objects_interaction_processor.py` | Fixes root cause | Upstream change; doesn't fix forearm PLY | Rejected |

### Architecture Changes

- **New module:** `code/scripts/_5_postprocessing/deduplicate_xy_points.py`
- **New output directories:** `blocks_merged_deduped/`, `forearm_deduped/`
- **Pipeline step insertion:** New Step 0 before projection in `pipeline_stages`
- **Step 1 (projection) change:** Receives deduped CSVs and deduped forearm PLYs
  from context instead of loading raw forearms directly.

### Knowledge-Base Constraints

- **Units:** All coordinates in mm (note-somatosensory-units-and-calculations);
  epsilon=0.1mm is sub-vertex-spacing.
- **ICP:** Runs after dedup; no interaction (note-forearm-icp-registration).
- No other knowledge-base notes apply.

---

## Implementation Plan

### Phase 1: Core dedup utility

**Goal:** Implement the reusable deduplication algorithm and its two wrappers.

- [x] Task 1.1 — Implement `deduplicate_xy(points: np.ndarray, epsilon: float = 0.1) -> Tuple[np.ndarray, int]`
  - Quantize (x,y) to epsilon bins
  - Group by bin key, keep `argmin(z)` per group
  - Assert `len(result) + n_removed == len(input)`
- [x] Task 1.2 — Implement `deduplicate_forearm_ply(input_ply: Path, output_ply: Path, epsilon: float = 0.1) -> dict`
  - Load PLY via open3d, run `deduplicate_xy`, write deduped PLY
  - Return `{n_original, n_deduped, n_removed}`
- [x] Task 1.3 — Implement `deduplicate_contact_points_csv(input_csv: Path, output_csv: Path, epsilon: float = 0.1) -> dict`
  - For each row with non-empty `contact_points`: parse → dedup → serialize
  - Recompute `contact_location_x/y/z` as mean of deduped points
  - Return `{n_rows_processed, total_points_before, total_points_after}`

**Files Modified:**
- `code/scripts/_5_postprocessing/deduplicate_xy_points.py` — NEW

**Dependencies:** None

### Phase 2: Pipeline flow and integration

**Goal:** Wire the utility into the postprocessing workflow as Step 0.

- [x] Task 2.1 — Add import of `deduplicate_xy_points` module in
  `postprocess_workflow_kinect_auto.py`
- [x] Task 2.2 — Create `deduplicate_xy_flow()` Prefect flow function:
  - For each (config, input_csv) pair:
    - Load per-block forearm (`_load_per_block_forearm_ply` pattern)
    - `deduplicate_forearm_ply()` → save to `forearm_deduped/`
    - `deduplicate_contact_points_csv()` → save to `blocks_merged_deduped/`
  - Log per-block stats at INFO level
  - Return `(deduped_csv_paths, deduped_forearm_ply_paths)`
- [x] Task 2.3 — Insert Step 0 in `pipeline_stages` before the projection step
- [x] Task 2.4 — Update Step 1 (projection flow) to consume
  `context["deduped_source_files"]` and `context["deduped_forearm_plys"]`
  instead of loading raw forearms itself

**Files Modified:**
- `code/scripts/postprocess_workflow_kinect_auto.py` — new flow, pipeline_stages update,
  projection step input change

**Dependencies:** Phase 1

### Phase 3: DAG config

**Goal:** Register the new task in the DAG so it can be enabled/disabled.

- [x] Task 3.1 — Add `deduplicate_xy` task entry in
  `configs/postprocess_workflow_kinect_auto_dag.yaml` before
  `project_contacts_onto_forearm`, with `enabled: true` and `depends_on: []`
- [x] Task 3.2 — Update `project_contacts_onto_forearm` to
  `depends_on: [deduplicate_xy]`

**Files Modified:**
- `configs/postprocess_workflow_kinect_auto_dag.yaml`

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] `deduplicate_xy` with no duplicates → returns same array, n_removed=0
- [ ] `deduplicate_xy` with 3 points at same (x,y), different z → returns 1
  point with lowest z, n_removed=2
- [ ] `deduplicate_xy` with points at (x,y) differing by exactly epsilon → no
  dedup (they land in different bins)
- [ ] `deduplicate_xy` with empty array → returns empty, n_removed=0

### Manual Verification
- [ ] Run full postprocess pipeline on a session; confirm `blocks_merged_deduped/`
  and `forearm_deduped/` directories are created with correct file counts
- [ ] Compare contact_points counts before/after in deduped CSVs — verify
  reasonable reduction (not 0%, not >50%)
- [ ] Verify downstream steps (projection → ICP → PCA → RF center) complete

### Edge Cases
- [ ] Block where forearm metadata is missing — dedup skips forearm, still
  deduplicates contact_points in CSV
- [ ] Frame with all contact_points at unique (x,y) — passes through unchanged
- [ ] Frame with empty contact_points — row passes through unchanged

---

## Documentation Plan

- [ ] No README changes (internal pipeline step)
- [ ] No CLAUDE.md changes (architecture unchanged at high level)
- [ ] Update the contact-projection-first-architecture plan to note this
  prerequisite step

---

## Rollback Plan

1. Remove the `deduplicate_xy` entry from `pipeline_stages` in
   `postprocess_workflow_kinect_auto.py`
2. Revert Step 1 (projection) to read from `context["source_files"]` and load
   raw forearms via `_load_per_block_forearm_ply`
3. Remove `deduplicate_xy` from the DAG config
4. The module file can remain (unused) or be deleted
5. Re-run pipeline with `force_processing=True` to regenerate from raw merged data

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Bin-boundary artifact: two points at distance < epsilon land in adjacent bins | Low | Low | Acceptable — the 0.1mm tolerance means worst case is a ~0.14mm gap at bin corners; negligible at mesh resolution |
| Forearm PLY dedup removes too many points (thin forearm viewed at steep angle) | Low | Medium | Log stats; if >20% removed, investigate per-session |
| Breaking change to projection flow interface | Low | Low | Single branch; both changes committed together |

---

## References

- Parent plan: `docs/development/plans/active/contact-projection-first-architecture.md`
- contact_points origin: `code/src/preprocessing/motion_analysis/tactile_quantification/model/objects_interaction_processor.py:174-175`
- Forearm loading: `code/scripts/postprocess_workflow_kinect_auto.py:_load_per_block_forearm_ply()`
- Projection function: `code/scripts/_5_postprocessing/project_contacts_onto_forearm.py`
- Units reference: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
