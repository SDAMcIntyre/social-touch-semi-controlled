# Plan: XY-Deduplication — Cluster-Based Algorithm

**Created:** 2026-05-05 09:30
**Approved:** —
**Completed:** 2026-05-05 10:13
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/stage-viewer-dedup-and-click-navigate`

---

## Overview

Replace the bin-quantize-and-group algorithm inside `deduplicate_xy()` with a
single-linkage cluster algorithm. Two points whose 2-D (x, y) distance is
within `epsilon` end up in the same cluster (transitively); within each
cluster, the single survivor is the point with the lowest z. This eliminates
the bin-boundary artefact in the current implementation, where two points
within ε of each other can fall into adjacent bins and both be kept.

## Problem Statement

The active plan
[`xy-deduplication-forearm-and-contacts.md`](../active/xy-deduplication-forearm-and-contacts.md)
chose a binning approach (`np.round(xy / eps).astype(int64)` then group by bin
key, keep `argmin(z)`). The risk table acknowledged a bin-boundary artefact as
"Low/Low — acceptable", but this assumption does not hold in practice.

Concrete failure observed by the user with the new interactive monitor at
`epsilon = 0.378` mm:

| Point | x | y | bin_x = round(x / ε) | bin_y = round(y / ε) |
|-------|----|----|----------------------|----------------------|
| A     | 35.0 | −21.731 | 93 | −57 |
| B     | 35.0 | −22.000 | 93 | −58 |

Distance(A, B) = 0.269 mm < ε = 0.378 mm — yet A and B fall into different
bins (the y axis bin boundary at −57.5 × ε ≈ −21.735 mm separates them) and
both points survive. In the worst case the algorithm preserves point pairs as
close as `~ε / √2` apart on bin diagonals.

This contradicts the visual semantics communicated by the slider in the
interactive monitor (a circle of radius ε around each kept point) and produces
fewer dedup-survivor reductions than expected on real forearm scans.

## Goals

### In Scope
1. Replace the algorithm inside `deduplicate_xy()` so any two points within
   `epsilon` (Euclidean, on x-y only) end up represented by a single survivor.
2. Survivor selection rule unchanged: lowest z within the cluster
   (deterministic stable tie-break by original input order).
3. Public function signature, return type, and surrounding wrappers
   (`deduplicate_forearm_ply`, `deduplicate_contact_points_csv`,
   `monitor_deduplicate_xy_interactive`, `deduplicate_xy_flow`) are unchanged.
4. Live slider in `monitor_deduplicate_xy_interactive()` remains responsive
   (≤ ~250 ms re-cluster on a representative forearm cloud).

### Out of Scope
- Changing the dedup unit (still `(x, y)` only — z is the tie-breaker).
- Configurable cluster method (single-linkage / radius-graph CC is the
  decision; alternatives are noted but not exposed).
- Changing the default `epsilon` value or the slider range.
- Modifying upstream contact_points generation in
  `objects_interaction_processor.py`.
- Centroid-based survivors (e.g. cluster mean) — survivor must remain an
  existing input point so contact-point semantics are preserved exactly.

## Success Criteria

- [ ] For every pair of points in the **input** with `dist(xy_i, xy_j) ≤ ε`,
  at most one of them appears in the output.
- [ ] User's reported case reproduces the fix:
  `deduplicate_xy([[35.0, -21.731, z1], [35.0, -22.0, z2]], 0.378)` returns
  exactly 1 point — the one with the lower z.
- [ ] Survivor selection is deterministic and equal to "lowest z, ties broken
  by lower original index".
- [ ] `assert len(deduped) + n_removed == len(input)` continues to hold.
- [ ] Empty-input case returns `(empty, 0)` (unchanged).
- [ ] Postprocessing pipeline runs end-to-end on
  `valid_configs_ST13-02/.../block-order03.yaml` with `monitor: true`.
- [ ] Slider drag in the monitor remains usable (no multi-second freeze) on a
  forearm pointcloud with ~50k points at ε up to 5 mm.

---

## Technical Design

### Approach

Use `sklearn.cluster.DBSCAN(eps=epsilon, min_samples=1)` on the 2-D `(x, y)`
sub-array. With `min_samples=1` every point is a core point, so DBSCAN
degenerates to **single-linkage clustering at radius ε** — exactly the
"connected components on the ε-radius graph" semantics that match what the
slider visualises.

For each cluster label, pick the index of the point with the lowest z (ties
broken by lower original index, achieved with a stable sort). Reorder by
original index so the output preserves relative input order, matching the
current contract.

```python
from sklearn.cluster import DBSCAN

def deduplicate_xy(points: np.ndarray, epsilon: float = .75) -> tuple[np.ndarray, int]:
    if len(points) == 0:
        return points, 0

    labels = DBSCAN(eps=epsilon, min_samples=1).fit_predict(points[:, :2])

    # Stable sort: primary key = z ascending; secondary key = original index ascending.
    order = np.lexsort((np.arange(len(points)), points[:, 2]))

    seen: set[int] = set()
    kept: list[int] = []
    for i in order:
        lbl = int(labels[i])
        if lbl in seen:
            continue
        seen.add(lbl)
        kept.append(int(i))
    kept.sort()  # restore input order

    deduped = points[kept]
    n_removed = len(points) - len(deduped)
    assert len(deduped) + n_removed == len(points), (
        f"Deduplication invariant violated: {len(deduped)} + {n_removed} != {len(points)}"
    )
    return deduped, n_removed
```

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `DBSCAN(eps, min_samples=1)` | sklearn already a project dep; one-line clustering; matches slider semantics; KDTree-backed (fast in 2-D) | Slightly heavier than hand-rolled bin grid | **Chosen** |
| `scipy.spatial.KDTree.query_ball_tree` + union-find | Lower-level; explicit control | More code; reinvents what DBSCAN already does | Rejected |
| 3×3 neighbour-bin check on top of current binning | Keeps grid structure; O(N) | Tightens but does not eliminate the boundary artefact (still grid-based, distance guarantee = ε only inside cells); adds bookkeeping for marginal gain | Rejected |
| Iterative greedy: sort by z, walk in order, drop within ε of kept set | Deterministic; no dependency change | Not single-linkage — anchor points lock in early and break transitive closure (chain effect lost); semantics differ from slider visualisation | Rejected |
| Replace dedup by centroid per cluster (mean of cluster) | Smoother surface | Survivor stops being an existing input point — would invalidate contact-point semantics that need a real on-surface point | Rejected |

### Architecture Changes

- **Single function body change** in
  `code/scripts/_5_postprocessing/deduplicate_xy_points.py::deduplicate_xy`.
  No signature or return-type changes. Wrappers
  (`deduplicate_forearm_ply`, `deduplicate_contact_points_csv`) call into it
  unchanged.
- **No DAG, workflow, or config changes.**
- **New import:** `from sklearn.cluster import DBSCAN` at the top of the
  module. (sklearn is already imported in
  `postprocess_workflow_kinect_auto.py`.)

### Knowledge-Base Constraints

- **Units:** Coordinates are in mm (`note-somatosensory-units-and-calculations`).
  ε remains in mm. No conversion.
- **Fail-fast:** Existing assertion on the dedup invariant is retained. No
  silent fallback added.
- **CuPy import order:** Not affected — sklearn is CPU-only and unrelated to
  the CuPy/preprocessing constraint
  (`note-cupy-import-order`).

---

## Implementation Plan

### Phase 1: Algorithm replacement
**Goal:** Swap the bin-grid implementation for the DBSCAN-based one.
**Started:** 2026-05-05 10:15
**Completed:** 2026-05-05 10:18

- [x] Task 1.1 — Add `from sklearn.cluster import DBSCAN` import in
  `deduplicate_xy_points.py`.
- [x] Task 1.2 — Replace the body of `deduplicate_xy()` with the cluster-based
  implementation above. Keep signature, return shape, docstring updated to
  describe the new semantics ("any two points within ε on (x, y) are merged
  into a single survivor — the lowest-z point in the cluster").
- [x] Task 1.3 — Update the existing inline assertion / docstring wording so
  it states the radius-based guarantee, not the bin guarantee.

**Files Modified:**
- `code/scripts/_5_postprocessing/deduplicate_xy_points.py` — body of
  `deduplicate_xy()`, top-level import, docstring.

**Dependencies:** None.

### Phase 2: Verification
**Goal:** Confirm correctness on the user's reported case and end-to-end on a
real session.
**Started:** 2026-05-05 14:30
**Completed:** —

- [x] Task 2.1 — Add a `pytest`-style unit test (or extend `code/tests/`) for:
  the user's reported A/B example, an empty input, a 3-co-located-point case
  (deduped to 1 with min-z), a long chain to confirm single-linkage transitive
  closure, and a no-duplicates case (passthrough).
- [ ] Task 2.2 — Run the postprocess pipeline on
  `valid_configs_ST13-02/kinect_config_2022-06-14_ST13-02_semicontrolled_block-order03.yaml`
  with `monitor: true`, confirm visually in the monitor that no two surviving
  red points lie within an ε-radius circle.
- [ ] Task 2.3 — Smoke-test the slider: drag from 0.05 mm to ~3 mm and confirm
  no perceptible freeze on a representative forearm cloud. If a freeze is
  observed, document it (mitigation belongs to a follow-up plan, not this
  one).

**Files Modified:**
- `code/tests/test_deduplicate_xy_points.py` — NEW (small). Contains 27 unit tests covering:
  - Basic scenarios (empty input, single point, user's reported case)
  - Co-located points (same x/y, different z)
  - Single-linkage transitive closure (chained clusters)
  - Tie-breaking with equal z values
  - Output order preservation
  - Edge cases (zero epsilon boundary, large epsilon, negative z, etc.)
  - Stress tests (grid of clusters, random cloud with noise)
  - API contract verification

**Dependencies:** Phase 1.

### Manual Verification Steps (Tasks 2.2 & 2.3)

**Task 2.2** and **2.3** are manual user verification tasks, not automated tests:

- **Task 2.2 (Visual verification on real session):** User runs the postprocess pipeline on the specified ST13-02 session with `monitor: true` and visually inspects the interactive viewer to confirm:
  - No two surviving (red) points lie within an ε-radius circle when viewed from the top-down (z-axis view).
  - At the chosen epsilon, the visual representation matches the expected clustering.

- **Task 2.3 (Slider responsiveness):** User drags the epsilon slider from 0.05 mm to approximately 3 mm and confirms:
  - No multi-second freezes occur during slider interaction on a representative forearm cloud (~50k points).
  - If a freeze is observed, it will be documented and addressed in a follow-up plan (not in scope for this plan).

These steps are meant to be performed by the user after the automated unit tests pass.

---

## Testing Plan

### Unit Tests

- [ ] **User's reported case** —
  `deduplicate_xy(np.array([[35.0, -21.731, 0.0], [35.0, -22.0, 1.0]]), epsilon=0.378)`
  returns 1 point: `(35.0, -21.731, 0.0)` (the lower-z survivor).
- [ ] **Empty input** — `deduplicate_xy(np.empty((0, 3)), 0.5)` returns
  `(empty, 0)`.
- [ ] **No duplicates** — three points spaced > ε apart pass through unchanged
  with `n_removed == 0`.
- [ ] **Co-located points (different z)** — three points at the same (x, y)
  with z = `[2.0, 0.5, 1.0]` collapse to the z = 0.5 survivor.
- [ ] **Transitive chain** — points along a line at spacing 0.5 ε form a
  single cluster (single-linkage) and collapse to the lowest-z survivor.
- [ ] **Stable tie-break** — two points with identical z and within ε keep
  the one at the lower input index.
- [ ] **Output order** — surviving rows appear in their original relative
  order in the output array.

### Integration / Manual Verification

- [ ] **End-to-end pipeline** — postprocess run on the configured ST13-02
  block completes; `blocks_merged_deduped/` and `forearm_deduped/` populate
  as before; downstream projection/ICP/PCA produce no errors.
- [ ] **Monitor visual check** — at the user's chosen ε, surviving (red)
  points are visibly spaced ≥ ε apart on top-down ("View Z") view.
- [ ] **Slider responsiveness** — slider drag stays interactive at all ε
  values in the slider range.

### Edge Cases

- [ ] Single point input — returns the same point, `n_removed == 0`.
- [ ] All points identical in (x, y) — collapse to 1, `n_removed == N - 1`.
- [ ] Very large ε exceeding cloud extent — collapse to 1 point.
- [ ] All points have identical z — survivor is the lowest-input-index member
  per cluster.

---

## Documentation Plan

- [ ] Update the docstring of `deduplicate_xy()` to reflect the
  radius-clustering semantics.
- [ ] Append a short "Algorithm refinement" note at the bottom of the active
  plan
  [`xy-deduplication-forearm-and-contacts.md`](../active/xy-deduplication-forearm-and-contacts.md)
  pointing to this plan.
- [ ] No CLAUDE.md / README updates (internal pipeline detail).

---

## Rollback Plan

The change is contained to a single function body and one import. To revert:

1. `git revert` the implementation commit, OR
2. Restore the previous `deduplicate_xy()` body (bin-quantize-and-group) and
   remove the `DBSCAN` import.

Re-run the postprocess pipeline with `force_processing: true` on the
`deduplicate_xy` task to regenerate `blocks_merged_deduped/` and
`forearm_deduped/` with the prior algorithm.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| DBSCAN runtime regression in the live monitor at large ε | Med | Low | sklearn uses KDTree by default in 2-D; profile in Phase 2 task 2.3. If it freezes, follow-up plan to debounce slider callbacks (not in this plan's scope). |
| Single-linkage chain effect: a long ribbon of evenly spaced points collapses to one survivor | Low | Low | Matches the user-stated semantics ("clusters of points within ε"). Documented behaviour, not a bug. |
| Slight reduction in deduped point count compared to prior runs that already shipped | Low | Low | Re-running with `force_processing: true` regenerates downstream outputs. No on-disk format change. |
| sklearn import order interacting with the project's CuPy constraint | Low | Med | sklearn is CPU-only and is already imported elsewhere in the postprocess workflow (`postprocess_workflow_kinect_auto.py`). No new ordering risk. |

---

## References

- Predecessor plan (active): `docs/development/plans/active/xy-deduplication-forearm-and-contacts.md`
- Algorithm location: `code/scripts/_5_postprocessing/deduplicate_xy_points.py::deduplicate_xy`
- Interactive monitor where the bug is observed: same file,
  `monitor_deduplicate_xy_interactive` and `_monitor_viewer_process`.
- DBSCAN reference:
  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
