# Plan: Visualize Population RF Grid Metrics Heatmaps

**Date:** 2026-05-06
**Author:** Basil Duvernoy
**Status:** Completed
**Base Branch:** `feature/reduce-population-rf-grid-metrics`
**Branch:** `feature/reduce-population-rf-grid-metrics`

---

## Overview

Add a DAG-integrated visualization task that reads population RF grid metrics
CSVs and renders one heatmap PNG per IFF-specific metric per session+gesture
type. Each heatmap shows the metric value across the 2D feature space
(x = velocity, y = pressure), enabling visual inspection of how RF
characteristics change with stimulus parameters.

## Problem Statement

`reduce_population_rf_grid` produces CSVs with ~40 scalar metrics per grid
cell, but there is no visual output. Inspecting metric variation across the
velocity x pressure grid requires loading CSVs and writing ad-hoc plotting
code. A batch rendering step would let researchers open a metric folder
(e.g. `max_iff/`) and immediately compare heatmaps across sessions and
gesture types.

## Goals

### In Scope
1. New renderer module producing one heatmap PNG per IFF metric
2. Output organized by metric folder (not by session) for cross-session
   comparison
3. DAG task registration with `depends_on: [reduce_population_rf_grid]`
4. Idempotent execution via sentinel JSON

### Out of Scope
- Rendering spatial/RFMetrics-derived columns (convex_hull_area, Gaussian fit, etc.)
- Interactive viewer or GUI
- Cross-session statistical overlays
- Log-scale axes (grid is pre-binned in linear space)

## Success Criteria

- [ ] Task runs end-to-end, producing 15 PNGs per session per gesture type
- [ ] PNGs are organized as `<metric>/<session_id>_<gesture_type>.png`
- [ ] NaN cells (empty grid cells) are visually distinct (white/gray background)
- [ ] Axes labeled with cleaned feature names, colorbar present, title shows
      session ID + gesture type + metric name
- [ ] Task is skippable via `enabled: false` in DAG config
- [ ] Re-running without force skips already-rendered outputs (idempotency)
- [ ] Existing pipeline tasks are unaffected

---

## Technical Design

### Approach

Single renderer module with two functions:

1. **`render_grid_metric_heatmap()`** — Takes a DataFrame (one CSV), pivots
   the metric column to 2D via `df.pivot(index=pressure_col, columns=velocity_col,
   values=metric)`, renders with `plt.pcolormesh()`, saves PNG.

2. **`run_population_rf_grid_metrics_visualization()`** — Batch orchestrator:
   iterates sessions, discovers CSVs, iterates the 15 IFF metrics, calls the
   renderer for each, writes sentinel JSON.

Feature columns are discovered dynamically from `*_center` columns in the CSV.
The first `_center` column maps to x-axis (velocity), the second to y-axis
(pressure), matching insertion order from the upstream NPZ.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| One PNG per metric per session+gesture | Easy cross-session comparison per metric | More files (15 x sessions x gestures) | **Chosen** |
| Multi-panel figure (3x5 subplots) | Fewer files, see all metrics at once | Small subplots, hard to compare across sessions | Rejected |
| Seaborn `sns.heatmap()` | Nicer defaults, annotation support | Less control over NaN display, pcolormesh is standard for gridded data | Rejected |
| Output organized by session | Groups all metrics per session | Hard to compare one metric across sessions | Rejected |

### Architecture Changes

**New file:**
```
code/src/analysis/receptive_field_mapping/
    rf_population_grid_metrics_renderer.py   -- rendering + batch orchestration
```

**Output directory** (new, alongside existing metrics directory):
```
4_analysed/population_rf_grid_metrics_heatmaps/
    max_iff/
        <session_id>_<gesture_type>.png
        ...
    mean_iff/
        ...
    ...  (15 metric folders)
    heatmaps_summary.json                    (sentinel)
```

### Key Functions to Reuse

| Function | Location | Purpose |
|----------|----------|---------|
| `should_process_task()` | `utils/should_process_task.py` | Idempotency check |

### Rendering Conventions

Following existing codebase patterns (from `reporting.py`, `rf_2d_renderer.py`):

- Backend: `matplotlib.use('Agg')` (headless)
- DPI: 150
- Save: `savefig(..., bbox_inches='tight')`
- Colormap: `viridis` (sequential, perceptually uniform)
- NaN handling: `pcolormesh` with `set_bad()` on colormap for NaN cells
- Axis labels: feature names with `_center` suffix removed, underscores
  replaced with spaces

---

## Implementation Plan

### Phase 1: Renderer Module
**Goal:** Implement rendering function and batch orchestrator
**Started:** 2026-05-06

**Tasks:**
- [x] Task 1.1 -- Create `rf_population_grid_metrics_renderer.py` with the
      15-metric constant list `IFF_METRICS`
- [x] Task 1.2 -- Implement `render_grid_metric_heatmap(df, metric_name,
      x_feature_col, y_feature_col, output_path, session_id, gesture_type)`:
      pivot DataFrame to 2D, render with `pcolormesh`, add colorbar/title/labels,
      save PNG
- [x] Task 1.3 -- Implement `run_population_rf_grid_metrics_visualization(
      input_items, output_dir, force)`:
      iterate sessions and CSVs, discover feature columns, iterate metrics,
      call renderer, write sentinel JSON

**Files:**
- `code/src/analysis/receptive_field_mapping/rf_population_grid_metrics_renderer.py` -- new

**Dependencies:** None

### Phase 2: DAG Integration
**Goal:** Wire the task into the pipeline
**Started:** 2026-05-06

**Tasks:**
- [x] Task 2.1 -- Add import to `__init__.py`:
      `run_population_rf_grid_metrics_visualization`
- [x] Task 2.2 -- Add `visualize_population_rf_grid_metrics_flow` to
      `analysis_workflow.py`. For each session: resolve `metrics_dir` =
      `db_path / '4_analysed' / 'population_rf_grid_metrics' / session_id`,
      validate it exists with at least one CSV. Set `output_dir` =
      `db_path / '4_analysed' / 'population_rf_grid_metrics_heatmaps'`
- [x] Task 2.3 -- Register
      `("visualize_population_rf_grid_metrics", visualize_population_rf_grid_metrics_flow)`
      in `available_tasks` after `reduce_population_rf_grid`
- [x] Task 2.4 -- Add DAG config entry to
      `configs/analyse_workflow_processing_dag.yaml`:
      ```yaml
      visualize_population_rf_grid_metrics:
        category: processing
        enabled: true
        options:
          force_processing: false
        depends_on: [reduce_population_rf_grid]
      ```

**Files:**
- `code/src/analysis/receptive_field_mapping/__init__.py` -- modify
- `code/scripts/analysis_workflow.py` -- modify
- `configs/analyse_workflow_processing_dag.yaml` -- modify

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Run the task on a real session with multiple gesture types
- [ ] Open a metric folder (e.g. `max_iff/`) and verify all session+gesture
      PNGs are present
- [ ] Verify axes: x = velocity, y = pressure, colorbar present
- [ ] Verify NaN cells are visually distinct (white/light gray)
- [ ] Verify title contains session ID, gesture type, and metric name
- [ ] Spot-check a heatmap value against the source CSV
- [ ] Re-run without force: confirm task is skipped (sentinel check)

### Edge Cases
- [ ] Session with only one gesture type: produces PNGs in one gesture only
- [ ] All cells NaN for a metric: heatmap renders but shows no data
- [ ] Grid with extreme value ranges: colorbar auto-scales appropriately

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` receptive field mapping section
      with new task description

---

## Rollback Plan

1. Remove task from `available_tasks` in `analysis_workflow.py`
2. Remove DAG config entry from `analyse_workflow_processing_dag.yaml`
3. Remove import from `__init__.py`
4. Delete `rf_population_grid_metrics_renderer.py`
5. No data migrations -- output is additive (new PNG files in new directory)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Large number of PNGs (15 x sessions x gestures) | Certain | Low -- disk space | PNGs are small (~50-100 KB each); total ~50 MB for 10 sessions |
| Matplotlib memory leak on many figures | Low | Medium | Explicit `plt.close()` after each save |
| Feature column order assumption (velocity first, pressure second) | Low | Medium | Validated by upstream NPZ feature insertion order; fail-fast if not 2 feature columns |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Renderer Module | ~100 lines | None |
| Phase 2: DAG Integration | ~50 lines | Phase 1 |

---

## References

- Upstream task: `reduce_population_rf_grid` in `rf_population_grid_metrics_pipeline.py`
- Upstream grid task: `map_population_rf_grid` in `rf_population_grid_pipeline.py`
- Active plan: `docs/development/plans/active/reduce-population-rf-grid-metrics.md`
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
