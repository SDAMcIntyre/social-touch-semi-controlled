# Plan: Named Grid Groups for Population RF Grid Pipeline

**Date:** 2026-05-11
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-12 10:42
**Base Branch:** `feature/session-comparison-heatmap`
**Branch:** `feature/rf-population-grid-groups`

---

## Overview

**What:** Introduce named `grid_groups` to the `map_population_rf_grid` task and propagate group names through the downstream chain (`reduce_population_rf_grid`, `visualize_population_rf_grid_metrics`), mirroring the existing `cluster_groups` pattern in `touch_clustering`.

**Why:** The current `map_population_rf_grid` task has a flat config — a single set of features, neuron_mode, etc. Only one grid configuration can run at a time. Multiple grid configurations (e.g., fine 2D velocity+pressure, coarse 1D velocity) cannot coexist.

**How:** Wrap the flat config into a `grid_groups` dict in the YAML. Add a `group_name` parameter to the three underlying pipeline functions for path construction. Flow functions iterate over enabled groups. DAG wiring extracts `_grid_group_defs` and forwards them to downstream tasks, exactly as `_cluster_group_defs` is forwarded from `touch_clustering`.

## Problem Statement

- The population RF grid pipeline produces one grid configuration per run — to try different feature combinations or bin sizes, the researcher must edit the YAML and re-run, overwriting previous outputs
- The `touch_clustering` pipeline solved this problem with `cluster_groups`: multiple named configurations coexist, each with its own output directory, and group names propagate to all downstream tasks
- The RF grid pipeline lacks this capability, forcing serial experimentation instead of parallel comparison

## Goals

### In Scope
1. Introduce `grid_groups` dict in `map_population_rf_grid` options, where each group carries `enabled`, `features`, `neuron_mode`, `per_gesture_type`, `vertex_threshold_ratio`, `compute_baseline`
2. Add `group_name` parameter to `run_population_rf_grid`, `run_population_rf_grid_metrics`, and `run_population_rf_grid_metrics_visualization` for path construction
3. Update flow functions to iterate over enabled groups
4. Propagate `_grid_group_defs` from `map_population_rf_grid` options to `reduce_population_rf_grid` and `visualize_population_rf_grid_metrics` via DAG wiring in `run_batch_analysis`
5. Backward compatibility: when `grid_groups` is absent, fall back to flat config with `group_name=None` (original path structure)

### Out of Scope
- Modifying `visualize_session_comparison` — it calls the underlying functions directly with its own 1D config and is not downstream of `map_population_rf_grid`
- GUI viewers for group selection (future enhancement)
- Multi-group cross-comparison metrics or reports

## Success Criteria

- [ ] Running the pipeline with one enabled grid group produces outputs at `4_analysed/population_rf_grid/<group_name>/<session_id>/`, `population_rf_grid_metrics/<group_name>/<session_id>/`, and `population_rf_grid_metrics_heatmaps/<group_name>/<metric>/`
- [ ] Adding a second enabled group produces a second set of outputs in parallel (both group directories coexist)
- [ ] Disabling all groups causes a clean skip (no error)
- [ ] `visualize_session_comparison` produces unchanged outputs at `4_analysed/session_comparison/` (backward compatibility)
- [ ] Old flat YAML config (without `grid_groups`) still works via the backward-compat fallback

---

## Technical Design

### Approach

Add an optional `group_name: str | None = None` parameter to the three underlying pipeline functions. When `group_name is not None`, a `<group_name>` directory segment is inserted between the task subdirectory and `<session_id>`. Flow functions resolve enabled groups from `grid_groups` (or wrap flat params into `{None: {...}}` for backward compat) and iterate. DAG wiring extracts `_grid_group_defs` and forwards to downstream tasks.

This mirrors the `cluster_groups` pattern:
- `_cluster_group_defs = options.get("cluster_groups") or {}` (line 1067)
- forwarded to `touch_comparing`, `extract_receptive_fields_clustered`, etc. (lines 1138-1146)

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `group_name` parameter on underlying functions | Minimal change; backward compatible; clean path structure | 3 functions need a new parameter | **Chosen** |
| Adjust `output_dir` at flow level to include group | No changes to underlying functions | Creates `4_analysed/<group>/population_rf_grid/` — mixes group names with other dirs, or doubles the `population_rf_grid/` segment | Rejected |
| Separate task per group | DAG-level parallelism | Explosive YAML growth; breaks DAG dependency model | Rejected |

### Architecture Changes

No new modules or classes. Changes are parameter additions and iteration logic.

**Output directory structure change:**
```
Before:
  4_analysed/population_rf_grid/<session_id>/
  4_analysed/population_rf_grid_metrics/<session_id>/
  4_analysed/population_rf_grid_metrics_heatmaps/<metric>/

After (with group "velocity_pressure_2d"):
  4_analysed/population_rf_grid/velocity_pressure_2d/<session_id>/
  4_analysed/population_rf_grid_metrics/velocity_pressure_2d/<session_id>/
  4_analysed/population_rf_grid_metrics_heatmaps/velocity_pressure_2d/<metric>/

After (backward compat, group_name=None):
  4_analysed/population_rf_grid/<session_id>/               # unchanged
  4_analysed/population_rf_grid_metrics/<session_id>/       # unchanged
  4_analysed/population_rf_grid_metrics_heatmaps/<metric>/  # unchanged
```

---

## Implementation Plan

### Phase 1: Add group_name to underlying functions
**Goal:** Add backward-compatible `group_name` parameter to the three pipeline functions for path construction. All callers still pass `None` implicitly — nothing breaks.
**Started:** 2026-05-11
**Completed:** 2026-05-11

- [x] Task 1.1 — Add `group_name: str | None = None` to `run_population_rf_grid` signature; insert `group_name` segment into `session_out_dir` when not None (line 283)
- [x] Task 1.2 — Add `group_name: str | None = None` to `run_population_rf_grid_metrics` signature; insert `group_name` segment into `output_session_dir` when not None (line 217). Camera settings path (`output_dir / 'rf_camera_settings'` line 220) unchanged — per-database, not per-group
- [x] Task 1.3 — Add `metrics_base_dir: Path | None = None` to `run_population_rf_grid_metrics_visualization` signature; when `None`, fall back to existing `output_dir.parent / "population_rf_grid_metrics"` (line 194)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_population_grid_pipeline.py` — 1 param + 3-line path conditional
- `code/src/analysis/receptive_field_mapping/rf_population_grid_metrics_pipeline.py` — 1 param + 3-line path conditional
- `code/src/analysis/receptive_field_mapping/rf_population_grid_metrics_renderer.py` — 1 param + 2-line fallback

**Dependencies:** None

### Phase 2: Update flow functions
**Goal:** Flow functions iterate over enabled groups and pass `group_name` to underlying functions.
**Started:** 2026-05-11
**Completed:** 2026-05-11

- [x] Task 2.1 — `map_population_rf_grid_flow`: add `grid_groups: dict = None` parameter. If present, filter to enabled groups; if absent, wrap flat params into `{None: {...}}` for backward compat. Replace single `run_population_rf_grid` call with loop over `enabled_groups.items()`
- [x] Task 2.2 — `reduce_population_rf_grid_flow`: add `grid_group_defs: dict = None` parameter. Resolve enabled group names (or `[None]`). Wrap existing `resolved_items` + `run_population_rf_grid_metrics` call in group loop. Construct `grid_dir` with group_name segment when not None
- [x] Task 2.3 — `visualize_population_rf_grid_metrics_flow`: add `grid_group_defs: dict = None` parameter. Resolve enabled group names (or `[None]`). Loop over groups, constructing `output_dir = .../population_rf_grid_metrics_heatmaps/<group_name>` and `metrics_base_dir = .../population_rf_grid_metrics/<group_name>`. Pass both to renderer

**Files Modified:**
- `code/scripts/analysis_workflow.py` — 3 flow function signature + body changes (~40 lines net)

**Dependencies:** Phase 1

### Phase 3: DAG wiring and YAML config
**Goal:** Extract `_grid_group_defs` in `run_batch_analysis` and forward to downstream tasks. Migrate YAML config to `grid_groups` structure.
**Started:** 2026-05-11
**Completed:** 2026-05-11

- [x] Task 3.1 — In `run_batch_analysis`, after `_cluster_group_defs` extraction (line 1067), add `_grid_group_defs` extraction: `_grid_group_options = dag_handler.get_task_options("map_population_rf_grid") or {}; _grid_group_defs = _grid_group_options.get("grid_groups") or {}`
- [x] Task 3.2 — Forward `grid_groups` to `map_population_rf_grid`: add `if "grid_groups" in options: kwargs["grid_groups"] = options["grid_groups"]` before existing legacy param forwarding (lines 1115-1123)
- [x] Task 3.3 — Forward `_grid_group_defs` to downstream tasks: add `if task_name in ("reduce_population_rf_grid", "visualize_population_rf_grid_metrics"): if _grid_group_defs: kwargs["grid_group_defs"] = _grid_group_defs`
- [x] Task 3.4 — Transform `analyse_workflow_processing_dag.yaml` (lines 152-172): wrap flat `neuron_mode`, `per_gesture_type`, `vertex_threshold_ratio`, `compute_baseline`, `features` under `grid_groups → velocity_pressure_2d → {enabled: true, ...}`

**Files Modified:**
- `code/scripts/analysis_workflow.py` — ~10 lines in `run_batch_analysis`
- `configs/analyse_workflow_processing_dag.yaml` — restructure ~15 lines under `map_population_rf_grid.options`

**Dependencies:** Phase 2

---

## Testing Plan

### Manual Verification
- [ ] Enable `velocity_pressure_2d` group, run pipeline — verify `4_analysed/population_rf_grid/velocity_pressure_2d/<session>/` with NPZs
- [ ] Verify `4_analysed/population_rf_grid_metrics/velocity_pressure_2d/<session>/` with CSVs
- [ ] Verify `4_analysed/population_rf_grid_metrics_heatmaps/velocity_pressure_2d/<metric>/` with PNGs
- [ ] Run `visualize_session_comparison` — verify unchanged output at `4_analysed/session_comparison/`
- [ ] Add a second group (e.g., `velocity_only_1d`), re-run — verify both group directories appear side by side
- [ ] Re-run with `force_processing: false` — verify idempotency (skips re-computation)

### Edge Cases
- [ ] All groups `enabled: false` — clean skip with log message, no error
- [ ] `grid_groups` absent AND flat `features` absent — `ValueError` (same as today)
- [ ] Both `grid_groups` and flat `features` present — `grid_groups` takes precedence

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — add `grid_groups` to orchestration section describing the population RF grid pipeline
- [ ] Add inline YAML comments in `analyse_workflow_processing_dag.yaml` describing the grid_groups structure

---

## Rollback Plan

1. Revert the 5 file changes (3 underlying functions + 1 workflow + 1 YAML)
2. No data migration: output artifacts in `<group_name>/` subdirectories are write-only NPZ/CSV/PNG files that can be safely deleted
3. Existing `4_analysed/session_comparison/` outputs are unaffected and need no cleanup

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Old flat YAML breaks silently when `grid_groups` is absent | Low | Med | Backward compat: flow wraps flat params into `{None: {...}}`, producing original path structure |
| `visualize_session_comparison` regresses | Low | Med | Session comparison calls underlying functions without `group_name` kwarg — default `None` preserves original behavior. Verified by manual test |
| Large number of groups creates excessive output | Low | Low | Each group's grid computation is independent; user controls which groups are enabled |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Underlying functions | ~15 min | None |
| Phase 2: Flow functions | ~30 min | Phase 1 |
| Phase 3: DAG wiring + YAML | ~15 min | Phase 2 |

---

## References

- Pattern source: `touch_clustering.cluster_groups` in `configs/analyse_workflow_processing_dag.yaml` (lines 238-334)
- DAG wiring reference: `_cluster_group_defs` extraction and forwarding in `code/scripts/analysis_workflow.py` (lines 1065-1146)
- Underlying functions: `rf_population_grid_pipeline.py::run_population_rf_grid`, `rf_population_grid_metrics_pipeline.py::run_population_rf_grid_metrics`, `rf_population_grid_metrics_renderer.py::run_population_rf_grid_metrics_visualization`
- Related active plan: `docs/development/plans/active/session-comparison-heatmap.md`

---
