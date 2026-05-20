# Plan: Touch Feature Radar Plots

**Date:** 2026-05-20
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/compare-session-rf-boundaries`
**Branch:** `feature/touch-feature-radar-plots`

---

## Overview

Add per-session radar/spider plots that visualize the distribution of touch gesture characteristics. Each session produces an overall radar (all touches), one radar per gesture type (tap, stroke_proximal, stroke_distal), and a composite overlay for direct comparison. The feature axes match the canonical contact features used in touch clustering.

## Problem Statement

The feature extraction pipeline produces rich per-touch statistics (depth, area, velocity, pressure, MOS features), but there is no visualization showing what the overall stimulus space looks like per session. Researchers cannot quickly compare the touch characteristic distributions across gesture types or across sessions without manually inspecting raw CSVs.

## Goals

### In Scope
1. Render radar/spider plots with 9 contact feature axes per session
2. Show median + IQR (25th–75th percentile) per gesture type
3. Produce individual per-gesture-type PNGs, an overall PNG, and a composite overlay PNG
4. Integrate as a new DAG task with configurable aggregation (default: `mean_during_iff`)
5. Follow existing pipeline conventions (idempotency, fail-fast, dark theme)

### Out of Scope
- Cross-session radar comparison (single figure with multiple sessions)
- Interactive radar plot GUI/viewer
- Radar plots for clustering results (per-cluster distributions)
- Custom feature selection via DAG config (fixed to the 9 canonical features)

## Success Criteria

- [ ] Each session produces 5 PNGs: `_radar_all`, `_radar_tap`, `_radar_stroke_proximal`, `_radar_stroke_distal`, `_radar_composite`
- [ ] Radar axes display 9 features matching `DATA_TYPE_TO_COLUMNS` scalar features
- [ ] All axes normalized 0–1 (session-wide min-max) so gesture types are comparable
- [ ] Composite overlay shows all gesture types with distinct colors + IQR shading
- [ ] DAG task `render_touch_feature_radar` runs after `touch_feature_extraction`
- [ ] Sentinel JSON enables idempotency (re-run skips up-to-date sessions)
- [ ] Sessions with <2 touches for a gesture type skip that type gracefully

---

## Technical Design

### Approach

Create a pure renderer (matplotlib polar axes, Agg backend) and a pipeline orchestrator that loads feature extraction CSVs, computes per-gesture-type statistics, normalizes within session, and calls the renderer. The 9 features are the scalar subset of `DATA_TYPE_TO_COLUMNS` from the clustering pipeline — the same features researchers use for touch clustering.

Normalization uses session-wide min-max so all gesture-type subsets share the same scale and are directly comparable within a session.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Min-max normalization (session-wide) | Intuitive 0–1 range; gesture types comparable within session | Cross-session comparison misleading | **Chosen** — per-session comparison is the primary use case |
| StandardScaler (z-scores) | Matches clustering normalization | Negative values on radar are confusing; radar polygon can collapse | Rejected |
| Percentile/rank normalization | Robust to outliers | Loses magnitude information; all distributions look similar | Rejected |
| Configurable feature set via DAG | Flexible | Over-engineering; the 9 canonical features are the standard set | Rejected for now (out of scope) |

### Architecture Changes

Two new modules, three modified files. No new dependencies — uses only matplotlib (already required) and numpy/pandas.

```
code/src/analysis/receptive_field_mapping/
  rendering/
    rf_touch_feature_radar_renderer.py   # NEW — pure rendering
  pipelines/
    rf_touch_feature_radar_pipeline.py   # NEW — orchestrator
  __init__.py                            # MODIFIED — add export
```

---

## Implementation Plan

### Phase 1: Renderer
**Goal:** Pure matplotlib radar plot rendering with no I/O dependencies
**Started:** 2026-05-20
**Completed:** 2026-05-20

- [x] Task 1.1 — Create `rf_touch_feature_radar_renderer.py` with dark theme constants (`_BG = "#1a1a1a"`)
- [x] Task 1.2 — Implement `render_gesture_radar()` for single-gesture radar (polar axes, median line, IQR fill)
- [x] Task 1.3 — Implement `render_gesture_radar_composite()` for multi-gesture overlay on one radar
- [x] Task 1.4 — Define gesture color palette: tap=`#00bcd4`, stroke_proximal=`#ff9800`, stroke_distal=`#e040fb`, all=`#78909c`

**Files Created:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_touch_feature_radar_renderer.py`

**Dependencies:** None

### Phase 2: Pipeline
**Goal:** Orchestrator that loads CSVs, computes statistics, and drives the renderer
**Started:** 2026-05-20
**Completed:** 2026-05-20

- [x] Task 2.1 — Create `rf_touch_feature_radar_pipeline.py` with `RADAR_FEATURE_COLUMNS` and `RADAR_DISPLAY_LABELS`
- [x] Task 2.2 — Implement CSV discovery: `{db}/4_analysed/touch_features/{aggregation}/{session_id}_*_touch_summary.csv`
- [x] Task 2.3 — Implement per-session statistics: session-wide min-max, per-gesture median + IQR
- [x] Task 2.4 — Implement `run_touch_feature_radar()` with idempotency via `should_process_task()` + sentinel JSON
- [x] Task 2.5 — Handle edge cases: missing columns (raise `ValueError`), <2 touches per gesture (skip with log)

**Files Created:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_touch_feature_radar_pipeline.py`

**Dependencies:** Phase 1

### Phase 3: DAG Integration
**Goal:** Wire the pipeline into the analysis workflow
**Started:** 2026-05-20
**Completed:** 2026-05-20

- [x] Task 3.1 — Add `run_touch_feature_radar` to `receptive_field_mapping/__init__.py` exports
- [x] Task 3.2 — Add `render_touch_feature_radar` task to `analyse_workflow_processing_dag.yaml` (depends_on: `touch_feature_extraction`, default aggregation: `mean_during_iff`)
- [x] Task 3.3 — Add `render_touch_feature_radar_flow` Prefect flow to `analysis_workflow_processing.py`
- [x] Task 3.4 — Add stage entry in `_build_pipeline_stages()` after `touch_feature_extraction`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/__init__.py` — add import + `__all__` entry
- `configs/analyse_workflow_processing_dag.yaml` — add task definition
- `code/scripts/analysis_workflow_processing.py` — add flow function + stage registration

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] Test `render_gesture_radar()` produces a PNG file at the specified path with synthetic data (9 features, known medians/IQR)
- [ ] Test `render_gesture_radar_composite()` produces a PNG with 3 gesture types overlaid
- [ ] Test normalization: verify 0–1 range, verify session-wide min-max is used for all subsets

### Integration Tests
- [ ] Test `run_touch_feature_radar()` end-to-end with a real session's feature extraction CSV
- [ ] Test idempotency: second run with unchanged inputs skips processing
- [ ] Test force_processing=True re-generates outputs

### Manual Verification
- [ ] Enable `touch_feature_extraction` (with `mean_during_iff` enabled) and `render_touch_feature_radar` in DAG config
- [ ] Run pipeline for at least 2 sessions
- [ ] Verify 5 PNGs per session in `4_analysed/touch_feature_radar/{session_id}/`
- [ ] Verify radar axes show 9 labeled features
- [ ] Verify composite overlay has distinct colors per gesture type with visible IQR shading
- [ ] Verify sentinel JSON is written

### Edge Cases
- [ ] Session with no stroke_distal touches — should skip that type with a log message, produce 4 PNGs
- [ ] Session where a feature column has zero variance — normalized values should all be 0 (not NaN/crash)
- [ ] Feature extraction CSV missing (task not run) — should raise `ValueError` with clear message

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — add radar pipeline to orchestration section and task list
- [ ] No user guide needed — output is self-explanatory PNG files

---

## Rollback Plan

1. Remove the two new files (renderer + pipeline)
2. Revert changes to `__init__.py`, DAG config, and workflow script
3. No data migration — output PNGs can be deleted from `4_analysed/touch_feature_radar/`

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `touch_feature_extraction` disabled in DAG — no input CSVs | High | Medium | DAG dependency will surface the issue; document in DAG config comment |
| `mean_during_iff` aggregation not enabled in feature extraction config | Medium | Medium | Fail-fast with clear error naming the missing aggregation directory |
| Radar labels overlap at 9 axes | Low | Low | Use moderate figure size (8x8), small font, abbreviations for display labels |
| Zero-variance feature produces NaN after normalization | Low | Medium | Guard division: if max == min, set normalized value to 0.0 |

---

## References

- Approved Claude plan: `.claude/plans/analyse-the-map-single-compiled-crown.md`
- Canonical feature set: `code/src/analysis/touch_analytics/clustering_pipeline.py` lines 58–74 (`DATA_TYPE_TO_COLUMNS`)
- Gesture types: `code/src/analysis/pipeline/shared_constants.py` line 21 (`GESTURE_TYPES`)
- Rendering patterns: `code/src/analysis/receptive_field_mapping/rendering/rf_simple_diagnostics.py` (dark theme)
- Pipeline patterns: `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py` (sentinel idempotency)
