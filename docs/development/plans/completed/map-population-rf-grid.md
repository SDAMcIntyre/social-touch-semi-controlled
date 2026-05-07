# Plan: Systematic RF Population Mapping via Feature-Space Grid Sweep

**Date:** 2026-05-06
**Author:** Basil Duvernoy
**Status:** Completed
**Base Branch:** `dev`
**Branch:** `feature/map-population-rf-grid`

---

## Overview

Add a new analysis processing task (`map_population_rf_grid`) that
systematically maps receptive fields across a multi-dimensional feature-space
grid. For each session, it sweeps through combinations of touch-level features
(e.g., velocity, pressure) and at each grid cell averages the single-touch RFs
of all touches whose feature values fall within that cell's window — producing
a dense RF map per cell with vertex-level thresholding.

## Problem Statement

The touch-population-explorer GUI allows interactive exploration of how RF shape
varies across feature space (via the draggable filter rectangle). However, this
exploration is manual — the researcher drags a rectangle and visually inspects
one region at a time. There is no way to systematically sweep the full feature
space to produce a complete set of RF maps that can be analysed downstream
(e.g., how does RF shape change as velocity increases?).

A batch processing task that automates this sweep would:
- Produce reproducible RF maps at every grid point in feature space
- Enable downstream statistical analysis of RF shape vs. feature values
- Eliminate manual effort and subjective positioning of the filter window

## Goals

### In Scope
1. New pipeline module that constructs an N-dimensional grid over configurable
   features, filters touches per cell, averages their single-touch RFs, and
   applies vertex threshold
2. DAG config entry with per-feature `min`, `max`, `step`, `span` and global
   `neuron_mode`, `per_gesture_type`, `vertex_threshold_ratio`
3. Single `.npz` output per session (per gesture type if configured) containing
   all grid-cell RF maps and metadata
4. Flow registration in `analysis_workflow.py`

### Out of Scope
- Companion viewer/explorer for the grid output (future enhancement)
- GPU acceleration of the grid sweep
- Adaptive grid (non-uniform step sizes)
- Automatic feature selection (user specifies features in config)

## Success Criteria

- [ ] Task runs end-to-end for a session with 2 features configured
- [ ] Output NPZ contains correct `rf_maps`, `grid_centers`, `touch_counts`,
      `touch_ids`, and metadata
- [ ] Vertex threshold correctly masks vertices contacted by fewer than
      `ratio × n_cell_touches` touches
- [ ] `per_gesture_type: true` produces separate outputs per gesture type
- [ ] `per_gesture_type: false` pools all gestures into a single output
- [ ] Grid cells with 0 matching touches produce NaN RF maps and count=0
- [ ] Touches with NaN feature values are excluded from all cells
- [ ] Task is skippable via `enabled: false` in DAG config
- [ ] Existing pipeline tasks are unaffected

---

## Technical Design

### Approach

Reuse the existing single-touch RF data (`single_touch_rf_maps.npz`) and touch
feature CSVs as inputs. The grid sweep is a pure post-processing step that
filters and averages pre-computed data — no new contact-point parsing or
KDTree snapping needed.

The core algorithm mirrors what the touch-population-explorer does
interactively (filter touches → average RFs → apply vertex threshold), but
replaces the manual rectangle with a systematic grid.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Reuse `PopulationData` / `PopulationRFData` loaders | Proven code, aligned data model | Loads all contact points (large); we only need RF maps + features | **Chosen** — the loaders handle alignment and caching well |
| Re-parse CSVs from scratch | Could load only needed columns | Duplicates complex parsing logic; fragile | Rejected |
| Compute RFs on-the-fly from raw contact points | Avoids dependency on `map_single_touch_rf` | Much slower; duplicates `_compute_touch_rf()` | Rejected |

### Architecture Changes

**New file:**
- `code/src/analysis/receptive_field_mapping/rf_population_grid_pipeline.py`
  — grid construction, sweep logic, NPZ output

**Modified files:**
- `code/scripts/analysis_workflow.py` — add `@flow` function and register in
  `available_tasks`
- `configs/analyse_workflow_processing_dag.yaml` — add task entry

**No changes to:**
- `touch_population_data.py` (reuse loaders as-is)
- `touch_population_explorer.py` (GUI unchanged)
- `rf_single_touch_pipeline.py` (upstream unchanged)

### Key Functions to Reuse

| Function | Location | Purpose |
|----------|----------|---------|
| `load_population_data()` | `touch_population_data.py:394` | Load touch features + contact data |
| `load_population_rf_data()` | `touch_population_data.py:791` | Load single-touch RF sparse maps |
| `_compute_rf_heatmap()` logic | `touch_population_explorer.py:582` | RF averaging via `np.add.at()` |
| `_compute_unique_touch_count()` logic | `touch_population_explorer.py:755` | Vertex overlap counting |
| `_effective_threshold()` logic | `touch_population_explorer.py:788` | Ratio → absolute threshold conversion |
| `load_forearm_vertices()` | `rf_data_loader.py` | Forearm mesh vertex loading |

---

## Implementation Plan

### Phase 1: Core Pipeline Module
**Started:** 2026-05-06
**Completed:** 2026-05-06
**Goal:** Implement the grid sweep algorithm in a standalone module

**Tasks:**
- [x] Task 1.1 — Create `rf_population_grid_pipeline.py` with a
      `PopulationRFGridConfig` dataclass holding feature definitions
      (name, min, max, step, span), neuron_mode, vertex_threshold_ratio,
      per_gesture_type flag
- [x] Task 1.2 — Implement `build_feature_grid(config)` → returns array of
      grid centers `(G, N_features)` as Cartesian product of per-feature
      `np.arange(min, max, step)` ranges
- [x] Task 1.3 — Implement `filter_touches_for_cell(feature_matrix, center, spans)`
      → boolean mask of touches where all features fall within
      `[center_i - span_i/2, center_i + span_i/2]`
- [x] Task 1.4 — Implement `compute_cell_rf(rf_data, touch_mask, n_vertices,
      threshold_ratio)` → dense `(n_vertices,)` array using `np.add.at()`
      averaging + vertex threshold masking (NaN for below-threshold)
- [x] Task 1.5 — Implement `run_population_rf_grid(input_items, output_dir,
      config)` — main entry point: iterate sessions, load data, build grid,
      sweep cells, save NPZ + sentinel JSON
- [x] Task 1.6 — Handle gesture type splitting: if `per_gesture_type`, filter
      `PopulationData` by gesture type and produce separate NPZ per type

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_population_grid_pipeline.py` — new file

**Dependencies:** None

### Phase 2: Pipeline Registration
**Started:** 2026-05-06
**Completed:** 2026-05-06
**Goal:** Wire the new task into the DAG and analysis workflow

**Tasks:**
- [x] Task 2.1 — Add `map_population_rf_grid` entry to
      `configs/analyse_workflow_processing_dag.yaml` with category, depends_on,
      and options (features, neuron_mode, per_gesture_type,
      vertex_threshold_ratio)
- [x] Task 2.2 — Add `map_population_rf_grid_flow` function to
      `analysis_workflow.py` following existing `@flow` pattern
- [x] Task 2.3 — Register `("map_population_rf_grid",
      map_population_rf_grid_flow)` in `available_tasks` list

**Files Modified:**
- `configs/analyse_workflow_processing_dag.yaml` — add task entry
- `code/scripts/analysis_workflow.py` — add flow function + registration

**Dependencies:** Phase 1

### Phase 3: Testing
**Started:** 2026-05-06
**Completed:** 2026-05-06
**Goal:** Validate grid construction, filtering, and RF averaging

**Tasks:**
- [x] Task 3.1 — Unit test for `build_feature_grid()`: verify grid shape,
      step coverage, single-feature and multi-feature cases
- [x] Task 3.2 — Unit test for `filter_touches_for_cell()`: overlapping
      windows, boundary values, NaN exclusion
- [x] Task 3.3 — Unit test for `compute_cell_rf()`: correct averaging,
      vertex threshold masking, empty-cell NaN output
- [x] Task 3.4 — Integration test: run on a session with known feature
      distribution, verify output NPZ structure and touch_ids correctness

**Files Modified:**
- `code/tests/test_rf_population_grid_pipeline.py` — new file

**Dependencies:** Phase 1

---

## Output Specification

### NPZ File Structure

One `.npz` per session (or per session × gesture_type), saved under
`4_analysed/population_rf_grid/<session_id>/`:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `grid_centers` | `(G, F)` | float64 | Center coordinates of each grid cell |
| `feature_names` | `(F,)` | str | Feature column names, in order |
| `feature_configs` | dict | — | Per-feature min/max/step/span |
| `rf_maps` | `(G, V)` | float64 | Averaged RF heatmap per cell (NaN = uncontacted or below threshold) |
| `touch_counts` | `(G,)` | int64 | Number of touches in each cell |
| `touch_ids` | list of G arrays | int64 | Per-cell `(block_id, trial_id, single_touch_id)` triples |
| `gesture_type` | str or None | — | Which gesture type (None if pooled) |
| `neuron_mode` | str | — | "iff" or "spike" |
| `vertex_threshold_ratio` | float | — | Ratio used for vertex masking |

Where `G` = total grid cells, `F` = number of features, `V` = number of
forearm vertices.

### Sentinel File

`population_rf_grid_summary.json`:
```json
{
    "session_id": "ST13-03",
    "feature_names": ["hand_velocity_amplitude_mean_during_iff", "pressure_mean_during_iff"],
    "grid_shape": [20, 10],
    "total_cells": 200,
    "non_empty_cells": 142,
    "neuron_mode": "iff",
    "vertex_threshold_ratio": 0.25
}
```

### DAG Config Entry

```yaml
map_population_rf_grid:
  category: processing
  enabled: true
  depends_on: [map_single_touch_rf, touch_feature_extraction]
  options:
    neuron_mode: "iff"
    per_gesture_type: true
    vertex_threshold_ratio: 0.25
    features:
      hand_velocity_amplitude_mean_during_iff:
        min: 0
        max: 200
        step: 10
        span: 20
      pressure_mean_during_iff:
        min: 0
        max: 5
        step: 0.5
        span: 1.0
```

---

## Testing Plan

### Unit Tests
- [ ] `build_feature_grid()` with 1 feature → 1D array of centers
- [ ] `build_feature_grid()` with 3 features → correct Cartesian product shape
- [ ] `filter_touches_for_cell()` includes touches at window boundary
- [ ] `filter_touches_for_cell()` excludes touches with NaN feature values
- [ ] `compute_cell_rf()` averages correctly with 2 overlapping sparse RFs
- [ ] `compute_cell_rf()` returns all-NaN for empty touch set
- [ ] Vertex threshold masks vertices below `ratio × n_touches`

### Integration Tests
- [ ] Full sweep on synthetic data: 10 touches, 2 features, 3×3 grid → verify
      NPZ structure and correctness

### Manual Verification
- [ ] Run on a real session (e.g., ST13-03), load output NPZ, spot-check a
      cell with known touches against the explorer GUI
- [ ] Verify `per_gesture_type: true` produces 3 separate NPZ files

### Edge Cases
- [ ] All touches have identical feature values → single non-empty cell
- [ ] Feature span > step → overlapping windows, same touch in multiple cells
- [ ] Feature span = 0 → degenerate (should raise or warn)
- [ ] Grid produces >10,000 cells → log warning

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` with new task description
- [ ] Update `CLAUDE.md` architecture table if needed
- [ ] Add inline docstring to the flow function explaining config schema

---

## Rollback Plan

1. Remove task from `available_tasks` list in `analysis_workflow.py`
2. Remove DAG config entry from `analyse_workflow_processing_dag.yaml`
3. Delete `rf_population_grid_pipeline.py` and test file
4. No database or data migrations to reverse — output is additive (new NPZ
   files under a new directory)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Combinatorial explosion with many features (3+ dimensions) | Medium | High — OOM or very slow | Log warning if total cells > 10,000; document recommended limits |
| Feature column names in config don't match CSV headers | Medium | High — silent empty results | Validate feature names against loaded CSV columns; raise on mismatch |
| Large NPZ files for high-resolution grids | Low | Medium — disk space | Dense `(G, V)` array; consider sparse storage if G × V > threshold |
| Touch count per cell too low for meaningful RF | Low | Low — noisy maps | Output `touch_counts` so downstream can filter; no silent suppression |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Core Pipeline | ~200 lines | None |
| Phase 2: Registration | ~50 lines | Phase 1 |
| Phase 3: Testing | ~150 lines | Phase 1 |

---

## References

- Related GUI: `code/src/analysis/receptive_field_mapping/gui/touch_population_explorer.py`
- Upstream task: `map_single_touch_rf` in `rf_single_touch_pipeline.py`
- Feature source: `touch_feature_extraction` in `extraction_pipeline.py`
- Prompt refinement discussion: `.claude/plans/let-s-discuss-about-how-refactored-simon.md`
