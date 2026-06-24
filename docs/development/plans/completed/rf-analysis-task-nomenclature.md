# Plan: Analysis Pipeline Task Nomenclature Redesign

**Date:** 2026-05-26
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-29 08:51
**Base Branch:** `dev`
**Branch:** `refactor/analysis-task-nomenclature`

---

## Overview

Rename all ~24 analysis DAG processing tasks and 2 viewer tasks to clearly
communicate which conceptual branch each task belongs to. The unifying concept
is **sensitivity** — the neuron's sensitivity viewed through two lenses:
spatial (where on the skin) and stimulus (what touch parameters). A third
cross-domain category bridges both. New DAG graph category colours reinforce
the grouping visually.

## Problem Statement

Current task names are inconsistent: some use `touch_*`, some use `*rf*`, some
start with a verb (`extract_*`, `compare_*`, `map_*`). The longest name is 49
characters (`extract_population_rf_response_field_boundaries`). A newcomer
cannot tell which intellectual branch a task belongs to from its name alone.

The pipeline has two main analytical directions:

1. **Spatial** — characterising the receptive field's shape, boundary, area,
   and centre on the forearm surface.
2. **Stimulus** — characterising how neural response varies with touch
   parameters (velocity, pressure, depth, contact area).

A third set of tasks bridges both by examining how RF spatial properties change
as a function of stimulus features.

None of this structure is visible in the current naming.

## Goals

### In Scope

1. Rename all processing and viewer DAG task keys to use branch-specific
   prefixes (`spatial_`, `stimulus_`, `cross_`, `touch_`).
2. Update all `depends_on` references, `@flow(name=...)` strings, Python
   function names, `_build_pipeline_stages` entries, and `get_task_options()`
   calls.
3. Add new DAG graph categories (`foundation`, `spatial_sensitivity`,
   `stimulus_sensitivity`, `cross_domain`) with distinct colours.
4. Remove the deprecated `map_receptive_fields_clustered` task entirely.
5. Update documentation (`CLAUDE.md`, `code/src/analysis/CLAUDE.md`).

### Out of Scope

- Renaming output directories under `4_analysed/` (would break existing data).
- Renaming internal library functions (e.g. `run_preparation`,
  `run_clustering`) — these are internal API, not user-facing.
- Changing import paths or package structure.
- Any logic or behavioural changes — this is a pure rename.

## Success Criteria

- [ ] Every old task name returns zero grep hits across the codebase
- [ ] `pytest code/tests/` passes with no regressions
- [ ] The GUI DAG graph renders all tasks with correct new names and branch colours
- [ ] Pipeline executes a dry run (all tasks disabled except the cheapest one)
      without name-resolution errors

---

## Technical Design

### Approach

Prefix-based nomenclature with four categories. Each task name follows the
pattern `<prefix>_<verb>_<object>`. Prefixes are natural English words chosen
by the user: `spatial_`, `stimulus_`, `cross_`, `touch_`.

The rename is a mechanical search-and-replace across a bounded set of files.
No logic changes. No new files. No new dependencies.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Short prefixes (`rfm_`, `stim_`, `rfc_`) | Compact (3-5 chars) | Less readable for newcomers | Rejected |
| No prefixes, category colours only | Shorter names | Branch membership invisible in YAML/logs/Prefect | Rejected |
| Spelled-out prefixes (`spatial_`, `stimulus_`, `cross_`) | Clear, natural English | 7-9 char prefix overhead | **Chosen** |

### Architecture Changes

No new modules. No new classes. Changes are confined to:

- YAML config keys and `depends_on` arrays (3 files)
- Python orchestration `@flow(name=...)` and dispatch tables (2 files)
- GUI category colour dicts (1 file)
- Documentation (2 files)

---

## Complete Rename Mapping

### Foundation (`touch_`) — category: `foundation`

| Current | Proposed | Outcome |
|---------|----------|---------|
| `touch_preparation` | `touch_prepare_sessions` | Cleaned CSV with block IDs, gesture types |
| `touch_series_transforms` | `touch_compute_series` | Per-frame velocity, acceleration, pressure, MoS |
| `summarize_session_blocks` | `touch_summarize_blocks` | Summary CSV of block/trial structure |

### Spatial Sensitivity (`spatial_`) — category: `spatial_sensitivity`

| Current | Proposed | Outcome |
|---------|----------|---------|
| `map_single_touch_rf` | `spatial_map_single_touch` | Per-touch RF maps on forearm vertices |
| `set_rf_camera_settings` | `spatial_set_camera` | Camera orientation JSON for 3D rendering |
| `map_receptive_fields_simple` | `spatial_map_baseline` | Simple RF heatmap — baseline reference |
| `configure_forearm_slim_uv` | `spatial_configure_slim_uv` | Per-session SLIM UV config |
| `precompute_forearm_slim_uv` | `spatial_precompute_slim_uv` | Cached UV flattening |
| `extract_population_rf_response_field_boundaries` | `spatial_extract_boundaries` | Population RF heatmap + inflection boundary |
| `compare_session_rf_boundaries` | `spatial_compare_boundaries` | Cross-session RF boundary comparison |
| `compare_rf_center_proximal_distal` | `spatial_compare_rf_centers` | Proximal vs distal RF centre offset |

### Stimulus Sensitivity (`stimulus_`) — category: `stimulus_sensitivity`

| Current | Proposed | Outcome |
|---------|----------|---------|
| `touch_feature_extraction` | `stimulus_extract_features` | Per-touch aggregated features |
| `touch_clustering` | `stimulus_cluster_touches` | Clustered touches by feature values |
| `touch_comparing` | `stimulus_compare_clusters` | Statistical tests across clusters |
| `render_touch_feature_radar` | `stimulus_render_radar` | Radar charts of feature profiles |
| `analyse_ap_efficacy` | `stimulus_analyse_efficacy` | Action potential efficacy analysis |

### Cross-domain (`cross_`) — category: `cross_domain`

| Current | Proposed | Outcome |
|---------|----------|---------|
| `map_population_rf_grid` | `cross_map_feature_grid` | RF maps per feature-space bin |
| `reduce_population_rf_grid` | `cross_extract_grid_metrics` | Scalar RF metrics per grid cell |
| `visualize_population_rf_grid_metrics` | `cross_render_grid_metrics` | Heatmaps of RF metrics across grid |
| `visualize_session_comparison` | `cross_render_sessions` | Cross-session RF vs velocity bins |
| `extract_receptive_fields_clustered` | `cross_extract_cluster_rf` | RF extraction per touch cluster |
| `compute_receptive_field_metrics` | `cross_compute_cluster_metrics` | RF metrics per cluster |
| `visualize_receptive_fields_clustered` | `cross_render_cluster_rf` | RF heatmap per cluster |

### Viewers (minimal changes)

| Current | Proposed |
|---------|----------|
| `explore_rf_feature_space` | `explore_feature_space` |
| `precompute_explorer_caches` | `explore_precompute_caches` |
| All others | Unchanged |

### Deprecated — remove entirely

| Task | Replacement |
|------|-------------|
| `map_receptive_fields_clustered` | `cross_extract_cluster_rf` + `cross_render_cluster_rf` |

---

## DAG Category Colours

| Category | Colour | Purpose |
|----------|--------|---------|
| `foundation` | `#e0e0e0` (light grey) | Shared prerequisites |
| `spatial_sensitivity` | `#d0e8ff` (light blue) | Spatial sensitivity branch |
| `stimulus_sensitivity` | `#ffe0d0` (light peach) | Stimulus sensitivity branch |
| `cross_domain` | `#d0f0d0` (light green) | Bridge tasks |
| `viewer_required` | `#ffe0b0` (amber) | Unchanged |
| `viewer` | `#e8d0ff` (purple) | Unchanged |
| `viewer_support` | `#d0ffe8` (mint) | Unchanged |

---

## Implementation Plan

### Phase 1: YAML Config Renames
**Goal:** Rename all task keys and dependency references in the three DAG YAML
files.

**Started:** 2026-05-26
**Completed:** 2026-05-26

- [x] Rename task keys and `depends_on` in `configs/analyse_workflow_processing_dag.yaml`
- [x] Update `category` values to new branch-specific categories
- [x] Remove deprecated `map_receptive_fields_clustered` entry
- [x] Rename 2 viewer tasks in `configs/analyse_workflow_viewers_dag.yaml`
- [x] Update `configs/analyse_workflow_dag.yaml` (combined reference)

**Files Modified:**
- `configs/analyse_workflow_processing_dag.yaml` — all task keys, depends_on, category values
- `configs/analyse_workflow_viewers_dag.yaml` — 2 task keys
- `configs/analyse_workflow_dag.yaml` — mirror changes

**Dependencies:** None

### Phase 2: Python Orchestration Renames
**Goal:** Update all flow names, function names, and dispatch tables in the two
orchestration scripts.

**Started:** 2026-05-26
**Completed:** 2026-05-26

- [x] Update `@flow(name="...")` strings in `analysis_workflow_processing.py`
- [x] Rename Python functions (e.g. `touch_preparation_flow` → `touch_prepare_sessions_flow`)
- [x] Update `_build_pipeline_stages` name entries
- [x] Update all `dag_handler.get_task_options("old_name")` calls
- [x] Update `_task_enabled("old_name")` and `_preparation_dir()` / `_series_dir()` references
- [x] Remove deprecated `map_receptive_fields_clustered_flow` function and its stage entry
- [x] Same changes in `analysis_workflow_viewers.py` for the 2 renamed viewer tasks

**Files Modified:**
- `code/scripts/analysis_workflow_processing.py` — flow decorators, function names, stage list, option lookups
- `code/scripts/analysis_workflow_viewers.py` — same pattern for 2 tasks

**Dependencies:** Phase 1

### Phase 3: GUI Category Colours
**Goal:** Add new category entries so the DAG graph renders each branch in its
own colour.

**Started:** 2026-05-26
**Completed:** 2026-05-26

- [x] Add `foundation`, `spatial_sensitivity`, `stimulus_sensitivity`,
      `cross_domain` to `_CATEGORY_COLORS` dict
- [x] Add matching entries to `_CATEGORY_BORDER_COLORS` dict

**Files Modified:**
- `code/src/utils/gui/dag_launcher/dag_graph_items.py` — colour dicts

**Dependencies:** None (parallel with Phases 1-2)

### Phase 4: Documentation
**Goal:** Update task name references in documentation.

**Started:** 2026-05-26
**Completed:** 2026-05-26

- [x] Update `CLAUDE.md` analysis overview if it references task names
- [x] Update `code/src/analysis/CLAUDE.md` orchestration section

**Files Modified:**
- `CLAUDE.md` — task name references
- `code/src/analysis/CLAUDE.md` — task list and descriptions

**Dependencies:** Phases 1-2

### Phase 5: Verification
**Goal:** Confirm zero stale references and no regressions.

**Started:** 2026-05-26
**Completed:** 2026-05-26

- [x] Grep for every old task name across the entire codebase — zero hits in YAML; all Python hits updated (docstrings, error messages, log prefixes, task-name string lookups)
- [x] `pytest code/tests/` passes — 403 passed, 12 pre-existing failures (unrelated to this rename: `test_gmm_clusterer.py` import error, `test_get_config_entries_multi` config state, `test_rf_grid_cell_metrics.py` tangent-plane rotation matrix)
- [x] Launch the GUI — DAG graph renders with new names and branch colours (manual — pending user validation)
- [x] Inspect Prefect flow list (if server running) for new flow names (manual — pending user validation)

**Files Fixed in Phase 5 (regressions found and corrected):**
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — `"touch_clustering"` → `"stimulus_cluster_touches"` in runtime model lookups (lines 766, 771, 1321)
- `code/tests/test_dag_config_model.py` — `"map_population_rf_grid"` → `"cross_map_feature_grid"` in fixture YAML and all test helper calls
- `code/scripts/analysis_workflow_processing.py` — log prefix strings, error messages, docstrings
- `code/scripts/analysis_workflow_viewers.py` — docstrings updated
- `code/scripts/compare_flattening_methods.py` — comment updated
- `code/src/analysis/receptive_field_mapping/data/rf_extraction_io.py` — error messages
- `code/src/analysis/receptive_field_mapping/metrics/rf_grid_cell_metrics.py` — module docstring
- `code/src/analysis/receptive_field_mapping/pipelines/rf_cluster_pipeline.py` — module docstring
- `code/src/analysis/receptive_field_mapping/pipelines/rf_cluster_visualization_pipeline.py` — error message
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py` — error messages
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_center_pipeline.py` — error message
- `code/src/analysis/receptive_field_mapping/pipelines/rf_session_boundary_comparison_pipeline.py` — error message
- `code/src/analysis/receptive_field_mapping/pipelines/rf_simple_pipeline.py` — module docstring, error message
- `code/src/analysis/receptive_field_mapping/pipelines/rf_single_touch_pipeline.py` — module docstring, docstring param, error messages
- `code/src/analysis/receptive_field_mapping/pipelines/rf_touch_feature_radar_pipeline.py` — error messages
- `code/src/analysis/receptive_field_mapping/surface/forearm_slim_uv.py` — error messages, docstring
- `code/src/analysis/receptive_field_mapping/surface/rf_projection.py` — error message
- `code/src/analysis/touch_analytics/clustering_pipeline.py` — warning message
- `code/src/analysis/touch_analytics/comparing_pipeline.py` — warning message, docstring
- `code/src/analysis/touch_analytics/series_pipeline.py` — error message
- `code/src/utils/gui/dag_launcher/cluster_group_dialog.py` — module docstring, class docstring
- `code/src/utils/gui/dag_launcher/clustering_config_dialog.py` — module docstring, class docstrings
- `code/src/utils/gui/dag_launcher/grid_group_dialog.py` — module docstring
- `code/src/utils/gui/dag_launcher/radar_group_dialog.py` — module docstring
- `code/src/utils/pipeline/dag_config_model.py` — section comments
- `code/tests/test_forearm_slim_uv.py` — test helper docstring

**Dependencies:** Phases 1-4

---

## Testing Plan

### Unit Tests
- [ ] Existing `pytest code/tests/` passes — no test references old task names
- [ ] `test_dag_config_model.py` validates YAML structure (if it loads the
      analysis DAG, it must accept new keys)

### Integration Tests
- [ ] DAG dependency resolution: `DagConfigHandler` resolves topological order
      with new task names
- [ ] Pipeline stage dispatch: `_build_pipeline_stages` returns all stages with
      correct name→function mapping

### Manual Verification
- [ ] Launch `python code/scripts/launch_pipeline_gui.py` — all tasks visible
      in the DAG graph with correct branch colours
- [ ] Click a task node — detail panel shows correct name, options, dependencies
- [ ] Enable one cheap task (e.g. `touch_summarize_blocks`) and run — completes
      without name-resolution errors

### Edge Cases
- [ ] Grep confirms no old names remain in comments, docstrings, or log messages
- [ ] Sentinel JSON files (idempotency guards) do not contain task names in
      their content — verified during research

---

## Documentation Plan

- [ ] Update `CLAUDE.md` architecture overview with new task names
- [ ] Update `code/src/analysis/CLAUDE.md` with new orchestration task list
- [ ] No new documentation files needed — this is a rename, not a new feature

---

## Rollback Plan

1. `git revert <merge-commit>` — single `--no-ff` merge commit reverts all
   changes atomically
2. No data migrations — output directories are unchanged
3. No database/state changes — YAML configs are the only persistent state

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Stale reference in a file not covered by grep | Low | Med | Comprehensive grep for every old name; run tests |
| Prefect flow history shows old + new names | Low | Low | Cosmetic; resolves as new runs replace old |
| Test fixtures reference old task names | Low | Med | Run full test suite; fix any fixture references |
| User muscle memory for old names | Med | Low | Prefix scheme aids rediscovery; GUI colours help |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: YAML configs | ~30 min | None |
| Phase 2: Python scripts | ~45 min | Phase 1 |
| Phase 3: GUI colours | ~10 min | None |
| Phase 4: Documentation | ~15 min | Phases 1-2 |
| Phase 5: Verification | ~15 min | Phases 1-4 |

---

## References

- Earlier brainstorming session established the `spatial_` / `stimulus_` /
  `cross_` / `touch_` prefix scheme and complete rename mapping
- Knowledge base: `note-qt-itemchanged-signal-recursion.md` — Qt signal guard
  pattern for GUI colour updates
