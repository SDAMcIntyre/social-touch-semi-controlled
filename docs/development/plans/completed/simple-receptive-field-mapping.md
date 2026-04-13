# Plan: Simple Per-Neuron Receptive Field Mapping

**Date:** 2026-04-13
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/simple-receptive-field-mapping`

---

## Overview

Add a lightweight receptive field mapping task that runs early in the analysis workflow (before feature extraction). For each neuron/session, it collects all XYZ contact positions where `Nerve_spike == 1`, saves them as a CSV, and renders the spike positions overlaid on the forearm PLY. This gives an immediate spatial overview of each neuron's receptive field, independent of downstream feature extraction or clustering choices.

## Problem Statement

The only existing RF mapping (`map_receptive_fields_clustered`) depends on the full feature extraction and clustering pipeline. There is no way to get a quick, per-neuron spatial view of where spikes occur on the forearm without running the entire analysis chain first.

## Goals

### In Scope
1. New DAG task `map_receptive_fields_simple` positioned after `summarize_session_blocks`, before `touch_feature_extraction`
2. Per-session CSV output with raw spike-associated XYZ contact positions (from multi-vertex `contact_points` column)
3. Per-session PNG rendering of spike positions on the forearm PLY (reusing existing `render_forearm_heatmap`)

### Out of Scope
- Selectivity scoring or DBSCAN clustering of spike positions
- Interactive GUI / camera angle picker for simple RF maps
- Aggregated cross-session RF maps
- Any changes to the existing cluster-based RF mapping pipeline

## Success Criteria

- [ ] `map_receptive_fields_simple` task appears in DAG config and runs before `touch_feature_extraction`
- [ ] Per-session `spike_positions.csv` saved with columns `x`, `y`, `z` (one row per parsed contact vertex per spike frame)
- [ ] Per-session `<session_id>_rf_simple.png` renders forearm PLY in grey with spike positions overlaid
- [ ] Task is skippable via `enabled: false` in DAG config and respects `force_processing`

---

## Technical Design

### Approach

Create a single new module `rf_simple_pipeline.py` that reuses existing parsing and rendering utilities. The pipeline reads each session's aggregated CSV, applies the established 30Hz-to-1kHz forward-fill on `contact_points`, filters spike rows, parses multi-vertex contact strings, and outputs raw positions + a static heatmap PNG.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Use `contact_points` (multi-vertex) | Denser point cloud, matches cluster-based pipeline | Requires string parsing | Chosen |
| Use `contact_location_x/y/z` (centroid) | No parsing, simpler code | Single point per frame, less spatial detail | Rejected |
| Extend `rf_cluster_pipeline.py` with a "no-cluster" mode | No new file | Overcomplicates existing pipeline with conditional logic | Rejected |

### Architecture Changes

New module `rf_simple_pipeline.py` in the existing `receptive_field_mapping` package. No new packages or architectural patterns introduced.

Reuses:
- `parse_contact_points()` from `rf_data_loader.py`
- `render_forearm_heatmap()` from `rf_cluster_visualizer.py`
- `session_id_from_path()` from `pipeline_shared.py`

---

## Implementation Plan

### Phase 1: Pipeline module + workflow integration
**Goal:** Working simple RF mapping task in the analysis workflow

**Started:** 2026-04-13
**Completed:** 2026-04-13

**Tasks:**
- [x] Task 1.1 — Create `rf_simple_pipeline.py` with `run_simple_rf_mapping(input_items, output_dir, force)` function
- [x] Task 1.2 — Export `run_simple_rf_mapping` from `receptive_field_mapping/__init__.py`
- [x] Task 1.3 — Add `map_receptive_fields_simple_flow` Prefect flow and register in `available_tasks` (position 1, after `summarize_session_blocks`)
- [x] Task 1.4 — Add `map_receptive_fields_simple` task entry to `analyse_workflow_dag.yaml` (between `summarize_session_blocks` and `touch_feature_extraction`, `depends_on: []`)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_simple_pipeline.py` — **New**: core pipeline function
- `code/src/analysis/receptive_field_mapping/__init__.py` — Add export
- `code/scripts/analysis_workflow.py` — New flow function + task registration
- `configs/analyse_workflow_dag.yaml` — New task entry

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Run analysis workflow with only `map_receptive_fields_simple` enabled
- [ ] Verify `4_analysed/receptive_field_maps_simple/<session_id>/spike_positions.csv` exists per session
- [ ] Verify CSV has `x`, `y`, `z` columns and plausible row counts (non-zero for sessions with spikes)
- [ ] Verify `<session_id>_rf_simple.png` renders forearm in grey with coloured spike dots
- [ ] Verify task runs before `touch_feature_extraction` in pipeline log output
- [ ] Re-run with `force_processing: false` and verify existing output is skipped

### Edge Cases
- [ ] Session with zero spikes — CSV should be empty (header only), PNG should be skipped gracefully
- [ ] Session with missing forearm PLY — CSV should still be saved, PNG rendering logs warning and continues
- [ ] Session with missing `contact_points` or `Nerve_spike` column — skip with warning

---

## Documentation Plan

- No external documentation changes needed (internal pipeline addition)

---

## Rollback Plan

1. Revert the single commit on `feature/simple-receptive-field-mapping`
2. Remove the `map_receptive_fields_simple` entry from `analyse_workflow_dag.yaml`
3. No data migrations — output CSVs/PNGs are generated on each run and can be deleted

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Large CSV output for sessions with many spikes | Low | Low | Raw positions are lightweight (3 float columns); disk usage is negligible |
| `render_forearm_heatmap` slow with many points | Low | Low | Existing function already handles large point sets with subsampled forearm; spike positions are typically sparse |
| Missing forearm PLY for some sessions | Med | Low | Pipeline saves CSV regardless; logs warning and skips PNG render |

---

## References

- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — Existing cluster-based RF mapping (forward-fill + spike extraction pattern)
- `code/src/analysis/receptive_field_mapping/rf_data_loader.py` — `parse_contact_points()` utility
- `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py` — `render_forearm_heatmap()` rendering function
