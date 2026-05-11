# Plan: RF Simple Pipeline — Use PopulationData Instead of Inline Parsing

**Date:** 2026-05-11
**Author:** Basil Duvernoy
**Status:** In Progress
**Started:** 2026-05-11
**Completed (Phase 1):** 2026-05-11
**Base Branch:** `feature/replace-rf-3d-renderer-with-pyvista`
**Branch:** `refactor/rf-simple-use-population-data`

---

## Overview

Refactor `map_receptive_fields_simple` to read the series-augmented CSV (same source as the RF Camera Settings viewer) via `load_population_data()`, removing all inline data transformation (CSV parsing, forward-filling, KD-tree snapping). The pipeline becomes a pure consumer of pre-processed data plus a 2D projection renderer.

## Problem Statement

`run_simple_rf_mapping()` in `rf_simple_pipeline.py` duplicates ~120 lines of data transformation logic that `load_population_data()` in `touch_population_data.py` already handles:

- Contact-point string parsing (regex extraction of `[x y z]` triples)
- Forward-filling `contact_points` within touch groups (30 Hz → 1 kHz alignment)
- KD-tree vertex snapping to forearm PLY mesh

This duplication means:
1. Two code paths with different KD-tree thresholds (2 mm vs 15 mm) and different fallback strategies
2. The simple pipeline reads a different CSV (aggregated session) than every other RF viewer/pipeline (series-augmented), creating inconsistency
3. Bug fixes to contact-point parsing must be applied in two places

## Goals

### In Scope
1. Switch input from aggregated session CSV to series-augmented CSV
2. Replace all inline parsing/snapping with a single `load_population_data()` call
3. Delete `_aggregate_spike_counts()` and all related dead code
4. Maintain identical output structure: `spike_positions.csv`, heatmap PNG, sentinel JSON

### Out of Scope
- Changing the 2D projection or rendering logic (`render_forearm_heatmap()`)
- Modifying `load_population_data()` itself
- Changing the DAG task name, config keys, or flow function signature
- Changing downstream consumers of the output files

## Success Criteria

- [x] `rf_simple_pipeline.py` no longer imports `open3d`, `scipy.spatial`, or `parse_contact_points`
- [x] No contact-point parsing, forward-filling, or KD-tree logic remains in the file
- [x] Pipeline reads `4_analysed/series_transforms/{session_id}_series_augmented.csv` (not the aggregated CSV)
- [x] Output directory structure and file names unchanged
- [x] `spike_positions.csv` columns are `x, y, z` with PLY vertex coordinates
- [ ] Heatmap PNG renders correctly with camera settings applied
- [x] Idempotency check references series-augmented CSV as input

---

## Technical Design

### Approach

Replace all data-loading code in `run_simple_rf_mapping()` with a single call to `load_population_data(series_csv_path, forearm_ply_path)`. This function already handles CSV reading, contact-point parsing, forward-filling, KD-tree snapping (15 mm threshold), and caches results in an NPZ sidecar. The simple pipeline then extracts spike data via pure array indexing on the returned `PopulationData`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Use `load_population_data()` | Single source of truth; NPZ cache reuse; consistent threshold | Loads full PopulationData (more than needed) | **Chosen** — cache makes this negligible |
| Use `load_viewer_session_data()` | Lighter (only vertices + heatmap) | No spike/non-spike distinction; would need a new function | Rejected |
| Keep inline parsing, just switch CSV | Minimal change | Still duplicates transformation logic | Rejected |

### Architecture Changes

No new modules. One file is substantially rewritten:

```
rf_simple_pipeline.py
  BEFORE: reads aggregated CSV → parses → forward-fills → KD-tree snaps → renders
  AFTER:  calls load_population_data() → indexes arrays → renders
```

**Knowledge base constraints applied:**
- Camera settings JSON included in `should_process_task` input paths for staleness tracking (per `note-rf-camera-settings-connections.md`)
- Forearm PLY is mandatory (no fallback) — consistent with the fail-fast convention
- Upstream 15 mm KD-tree threshold already filters boundary artifacts (per `bug-rf-explorer-nearest-vertex-distance.md`)

---

## Implementation Plan

### Phase 1: Refactor `rf_simple_pipeline.py`
**Goal:** Replace inline transformation with `load_population_data()` call

**Tasks:**
- [x] Task 1.1 — Remove imports: `open3d`, `scipy.spatial.cKDTree`, `parse_contact_points`
- [x] Task 1.2 — Add import: `load_population_data` from `touch_population_data`
- [x] Task 1.3 — Delete `_KEY_COLS`, `_DATA_COLS`, `_NEEDED_COLS` constants
- [x] Task 1.4 — Delete `_aggregate_spike_counts()` function entirely
- [x] Task 1.5 — Rewrite `run_simple_rf_mapping()` body:
  - Resolve `series_csv_path` from `database_path` (same pattern as `_resolve_explorer_session_paths()`)
  - Resolve `forearm_ply_path` via `resolve_forearm_ply()`, raise if None
  - Idempotency check with `series_csv_path` + camera settings JSON
  - Call `load_population_data(series_csv_path, forearm_ply_path)`
  - Extract spike vertices: `pop_data.cp_vertex_idx[pop_data.cp_spike]`
  - Save `spike_positions.csv` from `forearm_vertices[spike_vertex_indices]`
  - Build `spike_counts_df` via `np.bincount()` on spike vertex indices
  - Build `RFRenderContext` from PopulationData fields
  - Render heatmap (unchanged call)
  - Save sentinel JSON
- [x] Task 1.6 — Update module docstring

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_simple_pipeline.py` — major rewrite

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Run pipeline on a session with saved camera settings — verify PNG renders correctly
- [ ] Run with `force_processing: false` twice — first reprocesses (new input path), second skips
- [ ] Run on a session with zero spikes — verify "no spikes" message and header-only CSV
- [ ] Inspect `spike_positions.csv` — columns are `x, y, z` with PLY vertex coordinates
- [ ] Verify `rf_simple_summary.json` sentinel contains expected fields
- [ ] Confirm `open3d` and `scipy.spatial` no longer imported in `rf_simple_pipeline.py`

### Edge Cases
- [ ] Series-augmented CSV missing (touch_series_transforms not run) — should raise immediately
- [ ] Forearm PLY missing — should raise ValueError before attempting load
- [ ] Camera settings missing for session — should raise ValueError at render time

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — simple pipeline now reads series-augmented CSV, not aggregated CSV

---

## Rollback Plan

1. `git revert` the single commit touching `rf_simple_pipeline.py`
2. No data migration needed — output files are regenerated on next run
3. Existing sentinel files will be stale (different input path) so the pipeline will re-run automatically

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| NPZ cache not yet built on first run | Low | Low | `load_population_data()` builds and caches automatically on first call |
| Series-augmented CSV not available | Low | Low | DAG dependency chain ensures `touch_series_transforms` runs first; fail-fast if missing |
| KD-tree threshold change (2 mm → 15 mm) produces different spike counts | Med | Low | 15 mm is the standard threshold used by all other RF pipelines; more consistent |
