# Plan: IFF Metric Selection (Mean vs Max)

**Date:** 2026-05-28
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/stimulus-iff-tuning-curves`
**Branch:** `feature/iff-metric-mean-max-selection`

---

## Overview

Add an `iff_metric` option (`"mean"` | `"max"`) to the analysis pipeline so that both mean and max IFF are produced as separate outputs. Currently, only mean IFF is used everywhere downstream. Both metrics are neuroscientifically meaningful: mean IFF captures average neural response intensity; max IFF captures peak response.

## Problem Statement

The pipeline hardcodes mean IFF as the sole neural response metric at two independent levels:

1. **Stimulus path**: The IFF tuning curves pipeline (`rf_iff_tuning_pipeline.py`) hardcodes `_IFF_COL = "Nerve_freq_mean"` and reads only from the `mean/` aggregation CSV. `Nerve_freq_max` already exists in the `max/` CSV but is unused.

2. **Spatial path**: The single-touch RF pipeline (`rf_single_touch_pipeline.py`) computes only per-vertex mean IFF (`val_sum / contact_count`). All downstream consumers (population heatmaps, boundary extraction, session comparison, viewers) inherit this mean-only data.

Researchers need both mean and max IFF outputs to compare neural response profiles across touch conditions.

## Goals

### In Scope
1. `spatial_map_single_touch` produces two distinct NPZ files per session (mean + max)
2. A user-selectable `iff_metric` dropdown in the DAG GUI launcher for consumer tasks
3. IFF tuning curves render with either `Nerve_freq_mean` or `Nerve_freq_max` as Y-axis
4. Spatial RF heatmaps, boundaries, and viewers use either mean or max per-vertex data
5. Backward compatibility with existing NPZ files (legacy fallback for mean)

### Out of Scope
- Adding other IFF aggregations (median, std, etc.) beyond mean and max
- Changing the `neuron_mode` (IFF vs spike) mechanism — orthogonal to `iff_metric`
- Adding `iff_metric` to `spatial_map_single_touch` itself (always produces both)
- Per-grid-group `iff_metric` selection in `cross_map_feature_grid` (task-level only)

## Success Criteria

- [ ] `spatial_map_single_touch` writes `single_touch_rf_maps_mean.npz` and `single_touch_rf_maps_max.npz` per session
- [ ] DAG GUI shows `iff_metric` dropdown on `stimulus_iff_tuning_curves`, `spatial_extract_boundaries`, `cross_render_sessions`, `cross_map_feature_grid`, `explore_single_touch_rf`, `explore_touch_population`
- [ ] IFF tuning curves with `iff_metric: max` produce PNGs with Y-axis "Max IFF (Hz)" using `Nerve_freq_max`
- [ ] `spatial_extract_boundaries` with `iff_metric: max` produces population heatmaps from per-touch max IFF data
- [ ] Legacy `single_touch_rf_maps.npz` works for `iff_metric: mean` with a deprecation warning
- [ ] Existing tests pass without regression

---

## Technical Design

### Approach

**Two-file NPZ strategy**: `spatial_map_single_touch` always produces both `_mean.npz` and `_max.npz` in a single pass (both accumulators run simultaneously). Consumer tasks select which file to read based on `iff_metric`. This keeps the producer simple (no option needed) and the consumers decoupled (each picks its metric independently).

**Stimulus path merge**: When `iff_metric="max"`, the tuning curves pipeline reads X-axis features from the `mean/` CSV (e.g., `contact_area_mean`) and Y-axis IFF from the `max/` CSV (`Nerve_freq_max`), merging on touch ID columns. This preserves the semantic meaning of feature binning.

**Metric-agnostic downstream**: `compute_rf_heatmap()` and all renderers are unchanged — they operate on whatever values the NPZ contains. The metric choice is encapsulated at the data-loading boundary.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Two distinct NPZ files (mean + max) | Clean separation; backward compatible; no format change per file | Slight disk overhead (two files per session) | **Chosen** |
| Single NPZ with both `rf_data` and `rf_data_max` keys | Single file; atomic read | Format change; every loader needs updating; backward compat harder | Rejected |
| Parameterize `_compute_touch_rf` to produce only the selected metric | Smaller output; simple | Must re-run for each metric; no simultaneous availability | Rejected |

### Architecture Changes

**New constants** in `shared_constants.py`:
- `IFF_METRICS = ('mean', 'max')` — validates `iff_metric` values
- `single_touch_npz_filename(iff_metric)` — returns `f"single_touch_rf_maps_{iff_metric}.npz"`
- `resolve_single_touch_npz(session_dir, iff_metric)` — resolves path with legacy fallback

**Modified function signatures** (add `iff_metric: str = "mean"` parameter):
- `run_population_response_field_extraction()`
- `launch_single_touch_rf_explorer()`
- `launch_touch_population_explorer()`
- All affected flow functions in `analysis_workflow_processing.py` and `analysis_workflow_viewers.py`

**Unchanged modules** (metric-agnostic by design):
- `compute_rf_heatmap()` — averages whatever `rf_values` it receives
- `load_population_rf_data()` — receives `npz_path` as parameter
- `load_single_touch_rf_data()` — receives `npz_path` as parameter
- All renderers and boundary/comparison pipelines

---

## Implementation Plan

### Phase 1: Shared Constants and NPZ Path Helpers
**Started:** 2026-05-28  **Completed:** 2026-05-28
**Goal:** Single source of truth for IFF metric validation and NPZ path resolution

- [x] Add `IFF_METRICS` constant tuple
- [x] Add `single_touch_npz_filename()` helper with validation
- [x] Add `resolve_single_touch_npz()` with legacy fallback and deprecation warning

**Files Modified:**
- `code/src/analysis/pipeline/shared_constants.py` — add constant, two functions

**Dependencies:** None

### Phase 2: Core Computation — Dual NPZ Production
**Started:** 2026-05-28  **Completed:** 2026-05-28
**Goal:** `spatial_map_single_touch` produces both mean and max NPZ files per session

- [x] Add `val_max` accumulator with `np.maximum.at` in `_compute_touch_rf()`
- [x] Return `(mean_pairs, max_pairs)` tuple from `_compute_touch_rf()`
- [x] Collect `rf_data_mean` and `rf_data_max` dicts in `run_single_touch_rf_mapping()`
- [x] Write two NPZ files per session in a loop (identical structure, different values)
- [x] Update `produced` list and skip-path to reference `_mean.npz`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_single_touch_pipeline.py` — dual accumulator + dual NPZ output

**Dependencies:** Phase 1

### Phase 3: GUI Dropdown and DAG Config
**Started:** 2026-05-28  **Completed:** 2026-05-28
**Goal:** User can select `iff_metric` via dropdown in the DAG launcher

- [x] Add `"iff_metric"` to `_OPTION_ENUMS` in `task_detail_panel.py`
- [x] Add `iff_metric: mean` to processing DAG config for: `stimulus_iff_tuning_curves`, `spatial_extract_boundaries`, `cross_render_sessions`, `cross_map_feature_grid`
- [x] Add `iff_metric: mean` to viewers DAG config for: `explore_single_touch_rf`, `explore_touch_population`

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — add enum entry
- `configs/analyse_workflow_processing_dag.yaml` — add option to 4 tasks
- `configs/analyse_workflow_viewers_dag.yaml` — add option to 2 tasks

**Dependencies:** None (can run parallel with Phase 2)

### Phase 4: Stimulus Path — IFF Tuning Curves
**Started:** 2026-05-28  **Completed:** 2026-05-28
**Goal:** Tuning curves support both `Nerve_freq_mean` and `Nerve_freq_max` as Y-axis

- [x] Remove hardcoded `_IFF_COL` and `_AGGREGATION` module constants
- [x] Extract `iff_metric` from options in `run_iff_tuning_curves()`; derive `iff_col` and aggregation source
- [x] Implement CSV merge logic: when `iff_metric="max"`, read `mean/` CSV for features + `max/` CSV for `Nerve_freq_max`, merge on `TOUCH_ID_COLS`
- [x] Parameterize `_compute_global_iff_ylim()` and `_compute_global_count_max()` to accept `iff_col`
- [x] Add `iff_ylabel` parameter to `render_session_tuning_curve()` and `render_overlay_tuning_curve()`
- [x] Replace hardcoded "Mean IFF (Hz)" Y-axis label with parameterized `iff_ylabel`
- [x] Pass `iff_ylabel` from pipeline to renderer

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_iff_tuning_pipeline.py` — parameterize IFF column, CSV source, merge logic
- `code/src/analysis/receptive_field_mapping/rendering/rf_iff_tuning_renderer.py` — parameterize Y-axis label

**Dependencies:** Phase 1

### Phase 5: Spatial Path — Thread `iff_metric` to NPZ Consumers
**Started:** 2026-05-28  **Completed:** 2026-05-28
**Goal:** Downstream consumers select mean or max NPZ file based on `iff_metric`

- [x] Add `iff_metric` parameter to `run_population_response_field_extraction()`, use `single_touch_npz_filename(iff_metric)` for NPZ path
- [x] Add `iff_metric` parameter to `launch_single_touch_rf_explorer()` and `launch_touch_population_explorer()`, use in NPZ path
- [x] Hardcode `single_touch_npz_filename("mean")` in SLIM UV config viewer (line ~483 of `rf_cluster_gui_launchers.py`)
- [x] Update `spatial_precompute_slim_uv_flow` NPZ path to use `single_touch_npz_filename("mean")`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py` — add `iff_metric` param, update NPZ path
- `code/src/analysis/receptive_field_mapping/pipelines/rf_cluster_gui_launchers.py` — add `iff_metric` param to launchers, hardcode mean for SLIM UV

**Dependencies:** Phase 1, Phase 2

### Phase 6: Flow Function Wiring
**Started:** 2026-05-28  **Completed:** 2026-05-28
**Goal:** `iff_metric` flows from DAG config through flow functions to pipeline calls

- [x] Add `iff_metric` param to `stimulus_iff_tuning_curves_flow()`, pass in options dict
- [x] Add `iff_metric` param to `spatial_extract_boundaries_flow()`, pass to pipeline
- [x] Add `iff_metric` param to `cross_render_sessions_flow()`, use in NPZ path
- [x] Add `iff_metric` param to `cross_map_feature_grid_flow()`, use in NPZ path
- [x] Update `spatial_precompute_slim_uv_flow()` NPZ path to `single_touch_npz_filename("mean")`
- [x] Add `iff_metric` extraction to each task's DAG dispatch lambda
- [x] Add `iff_metric` param to `explore_single_touch_rf_flow()` and `explore_touch_population_flow()` in viewers script
- [x] Add `iff_metric` extraction to viewer DAG dispatch lambdas

**Files Modified:**
- `code/scripts/analysis_workflow_processing.py` — 5 flow functions + dispatch lambdas
- `code/scripts/analysis_workflow_viewers.py` — 2 flow functions + dispatch lambdas

**Dependencies:** Phase 3, Phase 4, Phase 5

---

## Testing Plan

### Unit Tests
- [ ] `_compute_touch_rf` returns valid `(mean_pairs, max_pairs)` where max >= mean per vertex
- [ ] `single_touch_npz_filename("mean")` returns `"single_touch_rf_maps_mean.npz"`
- [ ] `single_touch_npz_filename("invalid")` raises `ValueError`
- [ ] `resolve_single_touch_npz` falls back to legacy NPZ for `iff_metric="mean"`
- [ ] `resolve_single_touch_npz` raises `FileNotFoundError` for `iff_metric="max"` on legacy data

### Integration Tests
- [ ] `run_single_touch_rf_mapping` produces both `_mean.npz` and `_max.npz` per session
- [ ] Both NPZ files have identical structure (`touch_id_map`, `rf_data`, `neuron_mode`)
- [ ] IFF tuning curves with `iff_metric="max"` correctly merges mean/ and max/ CSVs

### Manual Verification
- [ ] Run `spatial_map_single_touch` → verify both NPZ files appear per session
- [ ] Run `stimulus_iff_tuning_curves` with `iff_metric: max` → verify Y-axis label reads "Max IFF (Hz)"
- [ ] Run `spatial_extract_boundaries` with `iff_metric: max` → verify heatmaps differ from mean
- [ ] Open DAG GUI → verify `iff_metric` dropdown appears on correct tasks only

### Edge Cases
- [ ] Session with all-zero `Nerve_freq` → both mean and max NPZ have empty RF data
- [ ] `iff_metric="max"` with `neuron_mode="spike"` → per-vertex max is always 0 or 1 (binary spike)
- [ ] Legacy `single_touch_rf_maps.npz` without `_mean`/`_max` suffix → fallback works for mean, clear error for max

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — document `iff_metric` option in orchestration section and shared constants table
- [ ] Update active plan `docs/development/plans/active/stimulus-iff-tuning-curves.md` — note that tuning curves now support iff_metric selection

---

## Rollback Plan

1. **NPZ files**: Old `single_touch_rf_maps.npz` files remain on disk. `resolve_single_touch_npz` provides legacy fallback. Rolling back the code restores the old loader that reads the legacy filename.
2. **DAG config**: `iff_metric: mean` is the default. Removing the option key from YAML returns to default behavior.
3. **No database migrations**: All changes are file-based (NPZ, PNG, YAML). No schema changes.
4. **Git revert**: Single feature branch; `git revert` or branch deletion is clean.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SLIM UV cache invalidation on first run (NPZ filename changed) | High | Low | Expected behavior — SLIM UV rebuilds automatically. Document in commit message. |
| `np.maximum.at` with `-np.inf` init when all frames are NaN | Low | Medium | Share the existing NaN filter mask from mean computation — if mean is NaN, max is also NaN. Add unit test. |
| `max/` CSV missing when `iff_metric="max"` for tuning curves | Low | High | `max` aggregation is already enabled in DAG config. Fail-fast with clear error if CSV not found. |
| Merge on touch ID columns produces fewer rows (inner join) | Low | Low | Touch IDs are identical across aggregation CSVs — they come from the same groupby. Validate row count after merge. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Constants + helpers | Small | None |
| Phase 2: Dual NPZ production | Medium | Phase 1 |
| Phase 3: GUI + config | Small | None |
| Phase 4: Stimulus path | Medium | Phase 1 |
| Phase 5: Spatial path consumers | Small | Phase 1, 2 |
| Phase 6: Flow wiring | Medium | Phase 3, 4, 5 |

---

## References

- Related Plan: `docs/development/plans/active/stimulus-iff-tuning-curves.md`
- Knowledge Base: `docs/development/knowledge-base/note-qt-itemchanged-signal-recursion.md` (Qt signal guard pattern)
- Knowledge Base: `docs/development/knowledge-base/note-rf-camera-settings-connections.md` (parameter threading pattern)
- Knowledge Base: `docs/development/knowledge-base/note-analysis-pipeline-coordinate-spaces.md` (neuron-centroid invariant)
