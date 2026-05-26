# Plan: RF Center Proximal vs Distal Comparison

**Date:** 2026-05-25
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `dev`
**Branch:** `feature/rf-center-proximal-distal-comparison`

---

## Overview

Add a new DAG task that compares population response field centers between
`stroke_proximal` and `stroke_distal` gestures on a per-session basis,
using the `all` gesture center as a reference.  The task reads existing
per-session NPZ files (already containing boundary centroids in SLIM UV
space), produces per-session center-marked heatmap PNGs and a
cross-session aggregate plot showing the directional shift between
proximal and distal RF centers.

## Problem Statement

The current pipeline extracts population RF heatmaps and inflection
boundary contours per gesture type, but the only downstream comparison
(`compare_session_rf_boundaries`) focuses on boundary shape metrics
(area, perimeter, circularity, PCA axes).  There is no visualisation
that isolates and compares the **center position** of the response field
between stroke directions — a key question for understanding whether
stroke direction modulates the spatial tuning of mechanoreceptive
afferents.

The centroid data already exists in the per-session NPZ
(`boundary_centroid_uv_{gtype}`), so the missing piece is purely
rendering and aggregation.

## Goals

### In Scope

1. Per-session, per-gesture-type center-marked heatmap PNGs (like
   existing standalone interpolated PNGs but with a center marker
   instead of the boundary contour).
2. Cross-session aggregate 2D UV plot showing proximal/distal centroid
   offsets from the per-session `all` center, with dot (proximal) and
   diamond (distal) markers connected by a line.
3. Summary CSV with raw centroid coordinates, offsets, and
   proximal-distal distance per session.
4. DAG task wired into the analysis workflow, runnable via the pipeline
   GUI.

### Out of Scope

- Statistical testing on the proximal-distal shift (t-test, permutation,
  etc.) — future work once enough sessions are available.
- 3D visualisation of centers on the forearm mesh — the 2D UV
  representation is sufficient for this comparison.
- Tap gesture center comparison — only proximal/distal strokes are
  compared in this task.
- Modifications to the upstream
  `extract_population_rf_response_field_boundaries` pipeline.

  **Note (2026-05-26):** The `feature/rf-hotspot-center` branch (plan:
  `docs/development/plans/active/rf-hotspot-center.md`) introduced upstream
  modifications that are additive to this plan's scope: `InflectionBoundary`
  now carries a `peak_uv` field (hotspot center), and the per-session NPZ
  stores `boundary_peak_uv_{gtype}` / `boundary_peak_xyz_{gtype}` alongside
  the centroid keys.  The comparison pipeline (`rf_proximal_distal_center_pipeline.py`)
  was extended to load hotspot UV, render separate hotspot PNGs, and add
  hotspot columns to the summary CSV.

## Success Criteria

- [ ] Running `compare_rf_center_proximal_distal` produces per-session
      directories under `4_analysed/rf_center_proximal_distal/<session_id>/`
      with one center-marked heatmap PNG per gesture type that has a
      boundary centroid.
- [ ] A cross-session aggregate PNG
      `rf_center_proximal_distal_aggregate.png` is produced showing all
      sessions' proximal/distal pairs relative to origin.
- [ ] A summary CSV `rf_center_proximal_distal_summary.csv` is produced
      with correct centroid coordinates and offsets.
- [ ] The task is idempotent via sentinel JSON and respects
      `force_processing`.
- [ ] Sessions missing a centroid for `all`, `stroke_proximal`, or
      `stroke_distal` are skipped with a logged warning (no crash).
- [ ] Center-marked heatmaps use global colour scale and UV limits
      across sessions, consistent with existing standalone interpolated
      PNGs.

---

## Technical Design

### Approach

Read pre-computed centroid UV coordinates from the per-session
`{session_id}_population_response_fields.npz` (produced by
`extract_population_rf_response_field_boundaries`).  No re-computation
or re-projection is needed — the NPZ already stores
`boundary_centroid_uv_{gtype}` for each gesture type in SLIM UV space.

Rendering follows the same two-output pattern as
`compare_session_rf_boundaries`: per-session detail figures plus a
cross-session summary figure.  The pipeline structure mirrors
`rf_session_boundary_comparison_pipeline.py` (NPZ loading →
per-session extraction → global limits → render → sentinel).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| New standalone pipeline + renderer (this plan) | Clean separation; no risk to existing code; mirrors existing comparison task | Two new files | **Chosen** |
| Extend `compare_session_rf_boundaries` with center plots | Fewer files | Bloats an already-focused pipeline; mixes boundary-shape and center-position concerns | Rejected |
| Add centroid marker option to existing `render_population_rf_standalone_interpolated` | Reuses existing renderer | Adds optional parameters to a stable function used by another pipeline; risk of subtle regressions | Rejected |

### Architecture Changes

Two new modules, no changes to existing module signatures:

```
code/src/analysis/receptive_field_mapping/
├── pipelines/
│   └── rf_proximal_distal_center_pipeline.py   ← NEW
└── rendering/
    └── rf_center_comparison_renderer.py        ← NEW
```

Integration points (modifications to existing files):
- `configs/analyse_workflow_processing_dag.yaml` — new task entry
- `code/scripts/analysis_workflow_processing.py` — new `@flow` function
- `code/src/analysis/CLAUDE.md` — document the new task

DAG dependency graph (new task is a sibling of the existing comparison):

```
extract_population_rf_response_field_boundaries
    ├── compare_session_rf_boundaries           (existing)
    └── compare_rf_center_proximal_distal       (NEW)
```

---

## Implementation Plan

### Phase 1: Renderer
**Goal:** Create rendering functions for both output types.

- [x] Task 1.1 — Create `render_center_marked_heatmap()`: single-panel
      interpolated heatmap with a `+` marker at the centroid (violet,
      consistent with existing `_draw_inflection_boundary` style).
      Pattern-match figure setup, pcolormesh, and black-background style
      from `render_population_rf_standalone_interpolated`.
- [x] Task 1.2 — Create `render_proximal_distal_aggregate()`:
      white-background 2D scatter plot.  Origin crosshairs at (0, 0).
      Per-session: circle marker (proximal), diamond marker (distal),
      connecting line.  `tab20` colormap per session (consistent with
      `rf_boundary_comparison_renderer.py`).  Legend with session IDs.
      Equal aspect ratio.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_center_comparison_renderer.py` — new file (~120 lines)

**Dependencies:** None

### Phase 2: Pipeline
**Goal:** Create the pipeline that reads NPZ data and drives rendering.

- [x] Task 2.1 — Create `run_proximal_distal_center_comparison()`
      following the structure of
      `rf_session_boundary_comparison_pipeline.py`: iterate sessions,
      load NPZ, extract centroids, compute offsets, collect data for
      global limits.
- [x] Task 2.2 — Per-session: load grid data from NPZ, render
      center-marked heatmap PNGs with global vmin/vmax and UV limits.
      Reuse `compute_standalone_figwidth` from
      `rf_population_map_renderer.py` for consistent figure sizing.
- [x] Task 2.3 — Cross-session: build summary DataFrame, write CSV,
      render aggregate plot.
- [x] Task 2.4 — Sentinel JSON for idempotency.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_center_pipeline.py` — new file (~180 lines)

**Dependencies:** Phase 1

### Phase 3: DAG + Workflow Integration
**Goal:** Wire the new pipeline into the DAG config and Prefect workflow.

- [x] Task 3.1 — Add `compare_rf_center_proximal_distal` task entry to
      `configs/analyse_workflow_processing_dag.yaml` after
      `compare_session_rf_boundaries`, with `depends_on:
      [extract_population_rf_response_field_boundaries]` and options for
      `force_processing`, `heatmap_space`, and `cmap`.
- [x] Task 3.2 — Add `compare_rf_center_proximal_distal_flow` Prefect
      `@flow` function to
      `code/scripts/analysis_workflow_processing.py`, following the
      pattern of `compare_session_rf_boundaries_flow`.
- [x] Task 3.3 — Update `code/src/analysis/CLAUDE.md` to document the
      new task under the "Population response fields" section.

**Files Modified:**
- `configs/analyse_workflow_processing_dag.yaml` — new task block (~10 lines)
- `code/scripts/analysis_workflow_processing.py` — new flow function (~20 lines)
- `code/src/analysis/CLAUDE.md` — add documentation entry

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests

- [ ] Synthetic NPZ with known centroids for all/proximal/distal →
      verify CSV output columns and values match expected offsets and
      Euclidean distance.
- [ ] Synthetic NPZ missing `stroke_distal` boundary → verify session
      is skipped with warning log, no crash, no row in CSV.

### Integration Tests

- [ ] Run the full DAG task on existing sessions → verify per-session
      directories created, aggregate PNG exists, CSV has one row per
      session with valid values.
- [ ] Verify the task respects `force_processing: false` (sentinel
      skip) and `force_processing: true` (re-renders).

### Manual Verification

- [ ] Open a center-marked heatmap and visually compare with the
      existing boundary-contour heatmap for the same session/gesture —
      heatmap background should be identical, marker should sit at the
      centroid position visible in the boundary version.
- [ ] Open the aggregate PNG and verify each session shows two markers
      connected by a line, with the legend matching session IDs.
- [ ] Launch the pipeline GUI and confirm the new task appears and can
      be toggled.

### Edge Cases

- [ ] Session where `stroke_proximal` has touches but
      `stroke_distal` does not (or vice versa) — graceful skip.
- [ ] Session where `all` gesture has no boundary (upstream detection
      failed) — skip entire session.
- [ ] Single-session run — aggregate plot should still render with one
      pair.

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — add entry under
      "Population response fields" for the new task and its outputs.
- [ ] No README.md or user guide changes needed — the task is
      auto-discovered by the DAG launcher GUI.
- [ ] No changelog entry required — small additive feature.

---

## Rollback Plan

The feature is entirely additive (two new files, a new DAG task key,
a new flow function).  No existing code signatures or outputs are
modified.

1. **Rollback procedure:**
   - Delete the two new source files.
   - Remove the DAG YAML task entry.
   - Remove the flow function from the workflow script.
   - Revert the CLAUDE.md documentation addition.
2. **Data considerations:**
   - Output directory `4_analysed/rf_center_proximal_distal/` can be
     deleted entirely — no other task reads from it.
   - No upstream data is modified.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Some sessions lack centroid for one or more gesture types (upstream boundary detection failed) | Medium | Low | Skip session with logged warning; do not crash. Aggregate plot and CSV simply omit that session. |
| SLIM UV units differ across sessions (different mesh scale) | Low | Low | Offsets are relative to each session's own `all` center, so absolute UV scale cancels out in the aggregate plot. |
| Global vmin/vmax computation fails if all sessions lack valid heatmap data | Very Low | Medium | Raise `ValueError` with descriptive message (fail-fast convention). This would indicate a broader upstream problem. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 (renderer) | ~1 hour | None |
| Phase 2 (pipeline) | ~1.5 hours | Phase 1 |
| Phase 3 (integration) | ~30 minutes | Phase 2 |

Total: ~3 hours.

---

## References

- Upstream pipeline: `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py`
- Comparison pipeline to pattern-match: `code/src/analysis/receptive_field_mapping/pipelines/rf_session_boundary_comparison_pipeline.py`
- Existing renderer to pattern-match: `code/src/analysis/receptive_field_mapping/rendering/rf_population_map_renderer.py`
- Boundary comparison renderer (tab20 color pattern): `code/src/analysis/receptive_field_mapping/rendering/rf_boundary_comparison_renderer.py`
- DAG config: `configs/analyse_workflow_processing_dag.yaml`
- Workflow script: `code/scripts/analysis_workflow_processing.py`
- NPZ data format: documented in `code/src/analysis/CLAUDE.md` under "Population response fields"

---
