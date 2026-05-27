# Plan: Cross-Session Touch Feature Comparison

**Date:** 2026-05-27
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `refactor/analysis-output-dirs`
**Branch:** `feature/stimulus-compare-sessions`

---

## Overview

Add a new analysis pipeline task (`stimulus_compare_sessions`) that pools per-session touch feature summary CSVs and renders cross-session comparison plots with all sessions on a single figure. This fills a gap: no existing task plots raw touch feature distributions (velocity, contact area, depth, etc.) across sessions side-by-side.

## Problem Statement

The pipeline can extract per-session touch features (`stimulus_extract_features`) and render per-session radar plots (`stimulus_render_radar`), but there is no task that places all sessions on the same axes for direct visual comparison of feature distributions. Researchers currently have to open individual session PNGs side-by-side or build ad-hoc notebooks. A built-in cross-session comparison task would make inter-session variability immediately visible and reproducible.

## Goals

### In Scope
1. Pool touch feature summary CSVs across all sessions into a single DataFrame
2. Render per-feature, per-gesture comparison plots (box + strip) with every session on one figure
3. Render a multi-panel summary grid per gesture type for at-a-glance comparison
4. Export a `pooled_feature_summary.csv` for downstream ad-hoc analysis
5. Support configurable feature selection reusing the same `features` schema as `stimulus_render_radar`
6. Support multiple plot types: box+strip (default), violin, bar+error

### Out of Scope
- Statistical testing across sessions (e.g., Kruskal-Wallis) — future enhancement
- Interactive GUI viewer for cross-session features
- Neural response (IFF/spike) comparison — already handled by `cross_render_sessions`

## Success Criteria

- [ ] Running the pipeline with `stimulus_compare_sessions` enabled produces one PNG per feature per gesture type, plus summary grids
- [ ] All sessions appear on the same x-axis in each plot
- [ ] `pooled_feature_summary.csv` contains all sessions with correct `session_id` and `gesture_type` columns
- [ ] Sentinel-based idempotency works: re-run without `force_processing` skips the task
- [ ] Axis labels use correct physical units (mm/s, mm, mm²) — not the known cm² mislabeling

---

## Technical Design

### Approach

Reuse the CSV-loading and column-resolution helpers already factored out in `rf_touch_feature_radar_pipeline.py` (`_find_feature_csv`, `_load_and_merge_feature_csvs`, `_resolve_radar_columns`, `_resolve_required_aggregation_folders`). These handle multi-aggregation CSV merging and data-type-to-column-name mapping. The new pipeline imports them directly — no duplication.

The pipeline follows the two-pass architecture used by `rf_touch_feature_radar_pipeline` and `rf_session_boundary_comparison_pipeline`:
1. **Pass 1 (load):** Iterate sessions, load feature CSVs, concatenate into a pooled DataFrame, compute global y-axis limits per feature.
2. **Pass 2 (render):** For each gesture type and feature column, call the renderer.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| New pipeline importing radar helpers | No duplication, consistent column resolution | Imports private functions across modules | **Chosen** — helpers are well-factored and stable |
| Extend `stimulus_render_radar` with a cross-session mode | Single module | Mixes per-session radar logic with cross-session box plots, harder to maintain | Rejected |
| Duplicate CSV-loading logic | No cross-module imports | Code duplication, divergence risk | Rejected |
| Extract shared helpers to `analysis/pipeline/` | Cleanest long-term | Over-engineering for 2 consumers | Deferred — do this if a 3rd consumer appears |

### Architecture Changes

New modules follow existing naming and placement conventions:

```
code/src/analysis/receptive_field_mapping/
    pipelines/
        rf_stimulus_session_comparison_pipeline.py   # NEW — orchestrator
    rendering/
        rf_stimulus_session_comparison_renderer.py   # NEW — pure matplotlib rendering
```

Integration points:
- Imports `_find_feature_csv`, `_load_and_merge_feature_csvs`, `_resolve_required_aggregation_folders`, `_resolve_radar_columns` from `rf_touch_feature_radar_pipeline`
- Imports `DATA_TYPE_TO_COLUMNS` from `clustering_pipeline` (via the radar pipeline's column resolution)
- Imports `_DISPLAY_NAMES` from `rf_touch_feature_radar_pipeline` for axis labels
- Uses `GESTURE_TYPES`, `TOUCH_ID_COLS_WITH_SESSION`, `session_id_from_path` from `analysis.pipeline.shared_constants`

---

## Implementation Plan

### Phase 1: Infrastructure
**Goal:** Add the output directory constant and DAG config entry.
**Started:** 2026-05-27
**Completed:** 2026-05-27

- [x] Add `STIMULUS_COMPARE_SESSIONS = "stimulus_compare_sessions"` to `output_dirs.py`
- [x] Add task entry in `analyse_workflow_processing_dag.yaml` after `stimulus_render_radar`

**Files Modified:**
- `code/src/analysis/pipeline/output_dirs.py` — add constant in the Stimulus sensitivity section
- `configs/analyse_workflow_processing_dag.yaml` — add task with `comparison_groups` config

**Dependencies:** None

### Phase 2: Renderer
**Goal:** Implement pure rendering functions (no I/O except PNG writing).
**Started:** 2026-05-27
**Completed:** 2026-05-27

- [x] Create `rf_stimulus_session_comparison_renderer.py`
- [x] Implement `render_feature_session_comparison()` — single feature, all sessions on one axes
- [x] Implement `render_feature_summary_grid()` — multi-panel grid (3 columns) of all features for one gesture
- [x] Implement `assign_session_colors()` — `tab20` colormap mapping
- [x] Support 3 plot types: `box_strip` (box + jittered dots), `violin`, `bar_error` (mean +/- std)
- [x] Dark theme consistent with existing renderers (`#1a1a1a` background)
- [x] Figure width scales with session count: `max(8, 1.2 * n_sessions)`

**Files Created:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_stimulus_session_comparison_renderer.py`

**Dependencies:** None

### Phase 3: Pipeline Orchestrator
**Goal:** Implement the two-pass load-then-render pipeline.
**Started:** 2026-05-27
**Completed:** 2026-05-27

- [x] Create `rf_stimulus_session_comparison_pipeline.py`
- [x] Implement `run_stimulus_session_comparison()` as main entry point
- [x] Pass 1: iterate sessions, load CSVs via radar helpers, build pooled DataFrame with `session_id`
- [x] Compute global y-axis limits per feature column (with 5% margin, anchor at 0 for positive-only)
- [x] Write `pooled_feature_summary.csv`
- [x] Pass 2: for each gesture type (tap, stroke_proximal, stroke_distal, all) and feature column, call renderer
- [x] Render summary grid per gesture type
- [x] Write sentinel JSON per comparison group
- [x] Handle empty gesture types gracefully (skip session in plot, don't crash)

**Files Created:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_stimulus_session_comparison_pipeline.py`

**Dependencies:** Phase 2

### Phase 4: Integration
**Goal:** Wire into the Prefect flow system and package exports.
**Started:** 2026-05-27
**Completed:** 2026-05-27

- [x] Add `run_stimulus_session_comparison` to `receptive_field_mapping/__init__.py` imports and `__all__`
- [x] Add `stimulus_compare_sessions_flow` Prefect flow in `analysis_workflow_processing.py`
- [x] Register in `_build_pipeline_stages` after `stimulus_render_radar`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/__init__.py` — add import and `__all__` entry
- `code/scripts/analysis_workflow_processing.py` — add flow function and stage registration

**Dependencies:** Phase 3

---

## Testing Plan

### Unit Tests
- [ ] `_resolve_radar_columns` produces expected column names for the configured features (already tested via radar pipeline — verify no regression)
- [ ] `assign_session_colors` returns unique colors for up to 20 sessions
- [ ] `render_feature_session_comparison` produces a PNG file without error for each plot type

### Integration Tests
- [ ] Full pipeline run with 2+ sessions: verify output PNGs exist in expected directory structure
- [ ] Sentinel file prevents re-run when `force_processing=False`
- [ ] `force_processing=True` overwrites existing outputs

### Manual Verification
- [ ] Inspect output PNGs: all sessions appear on x-axis, box plots show median/quartiles
- [ ] Verify axis labels use correct units (mm/s for velocity, mm² for area, mm for depth)
- [ ] Check `pooled_feature_summary.csv` has expected columns and row count
- [ ] Run pipeline GUI — new task appears and can be toggled

### Edge Cases
- [ ] Session with zero touches of a gesture type — should be omitted from that gesture's plot, not crash
- [ ] Single session — should still render (degenerate but valid)
- [ ] Feature CSV missing for one session — should raise `ValueError` (fail-fast)

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — add `stimulus_compare_sessions` to the `stimulus_sensitivity` category list in the Orchestration section

---

## Rollback Plan

All changes are additive (new files + new entries in existing files). Rollback is straightforward:

1. Remove the 2 new files
2. Revert additions to `output_dirs.py`, `__init__.py`, `analysis_workflow_processing.py`, and the DAG YAML
3. No data migrations or breaking changes to existing outputs

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Importing private (`_`-prefixed) functions from radar pipeline breaks if they're refactored | Low | Med | These functions are stable and well-tested; if a 3rd consumer appears, extract to shared module |
| Large session count (>15) makes x-axis labels unreadable | Low | Low | Figure width scales with session count; labels rotated 45 degrees |
| Inconsistent aggregation configs between `stimulus_render_radar` and `stimulus_compare_sessions` | Med | Low | Both use identical `features` schema — document that they should stay in sync |

---

## DAG Config

```yaml
stimulus_compare_sessions:
  category: stimulus_sensitivity
  enabled: true
  options:
    force_processing: false
    plot_type: box_strip
    comparison_groups:
      canonical_iff:
        enabled: true
        features:
          contact_area:
          - mean_during_iff
          contact_depth:
          - mean_during_iff
          hand_velocity_amplitude:
          - mean_during_iff
          pressure:
          - mean_during_iff
  depends_on: [stimulus_extract_features]
```

---

## Output Structure

```
4_analysed/stimulus_compare_sessions/
  {group_name}/
    {gesture_type}/
      {feature_column}_session_comparison.png
    {gesture_type}_feature_grid.png
    pooled_feature_summary.csv
    session_comparison_sentinel.json
```

---

## References

- Radar pipeline (reused helpers): `code/src/analysis/receptive_field_mapping/pipelines/rf_touch_feature_radar_pipeline.py`
- Session boundary comparison (cross-session pattern): `code/src/analysis/receptive_field_mapping/pipelines/rf_session_boundary_comparison_pipeline.py`
- Knowledge base: `note-somatosensory-units-and-calculations.md` (unit labels: mm, mm/s, mm²)
