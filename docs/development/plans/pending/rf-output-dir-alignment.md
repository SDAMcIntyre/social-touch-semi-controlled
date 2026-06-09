# Plan: Analysis Output Directory Alignment with Task Nomenclature

**Date:** 2026-05-27
**Author:** Basil Duvernoy
**Status:** Pending (revised 2026-05-27)
**Base Branch:** `refactor/analysis-task-nomenclature`
**Branch:** `refactor/analysis-output-dirs`

---

## Overview

Rename the 20 output subdirectories under `4_analysed/` to mirror the new
task naming system (`spatial_`, `stimulus_`, `cross_`, `touch_` prefixes).
Simultaneously, extract all hardcoded path strings into a single constants
module so the scattered-string problem cannot recur. Normalize the
`output_dir` passing convention so all batch pipeline functions receive the
full path from the workflow script. A migration script renames existing
on-disk data to the new layout.

## Problem Statement

After the task nomenclature redesign (`refactor/analysis-task-nomenclature`),
output directory names still use the old, unprefixed, domain-based scheme
(`preparation`, `touch_features`, `receptive_field_maps_simple`, …). A
developer looking at `4_analysed/` cannot tell which analytical branch
produced a given folder. The disconnect between task names and folder names
also makes log messages confusing: a task named
`stimulus_extract_features` writes to `touch_features/`.

Additionally, path strings are hardcoded in ~16 files with no central
definition, so any future rename requires another multi-file grep-and-replace.
The `output_dir` parameter passing convention is also inconsistent: some
pipeline functions receive the full output path, others receive only the
`4_analysed/` root and append the subdirectory name internally, and others
construct the full path from `database_path` without an `output_dir` parameter
at all.

## Goals

### In Scope

1. Rename all 20 output subdirectories to match the new task prefix scheme.
2. Introduce `code/src/analysis/pipeline/output_dirs.py` — a single module
   of path constants that all pipeline code must import.
3. Replace every hardcoded `"4_analysed/<subfolder>"` string in Python source
   with a reference to the constant (both output and input path references).
4. Normalize all batch pipeline functions to receive `output_dir` as a
   parameter (pattern A) — functions no longer construct their own output
   directory from `database_path`.
5. Update docstrings that reference old directory names.
6. Provide a `scripts/migrate_output_dirs.py` migration script that renames
   existing on-disk directories and invalidates stale sentinel files.
7. Update `code/src/analysis/CLAUDE.md` with the new directory layout.

### Out of Scope

- Renaming files *inside* the output directories (e.g. `session_rf_boundary_comparison_done.json` sentinel filenames).
- Changing the `4_analysed/` top-level name.
- Any logic or algorithm changes — pure rename + centralization.
- Viewer task output paths (viewers write to the same dirs as processing tasks;
  covered by the constants module automatically).

## Design Decisions (resolved)

### Q1 — Shared neutral names ✓

For tasks that share an output directory, use shared neutral names:

| Tasks sharing a dir | Shared dir name |
|---------------------|-----------------|
| `spatial_configure_slim_uv` + `spatial_precompute_slim_uv` | `spatial_slim_uv` |
| `cross_extract_cluster_rf` + `cross_compute_cluster_metrics` + `cross_render_cluster_rf` | `cross_cluster_rf` |

### Q2 — Migration script ✓

`scripts/migrate_output_dirs.py` renames existing directories and invalidates
sentinel JSONs. Existing computed data survives (recomputing is expensive).

### Q3 — Normalize output_dir passing to pattern A ✓

All batch pipeline functions receive the full `output_dir` from the workflow
script. Functions no longer construct their own output directory from
`database_path`. GUI launchers remain as-is (they access multiple directories
and use constants for the names).

Seven functions are normalized:
- `run_population_response_field_extraction`
- `run_session_rf_boundary_comparison`
- `run_proximal_distal_center_comparison`
- `run_touch_feature_radar`
- `run_population_rf_grid`
- `run_population_rf_grid_metrics`
- `run_session_comparison_visualization`

### Q4 — Update docstrings ✓

Docstrings referencing old directory names are updated alongside code.

---

## Directory Rename Mapping (20 entries)

| Category | Current dir | New dir |
|----------|-------------|--------------|
| Foundation | `preparation` | `touch_prepare_sessions` |
| Foundation | `series_transforms` | `touch_compute_series` |
| Foundation | `session_summary` | `touch_summarize_blocks` |
| Spatial | `single_touch_rf_maps` | `spatial_map_single_touch` |
| Spatial | `rf_camera_settings` | `spatial_set_camera` |
| Spatial | `receptive_field_maps_simple` | `spatial_map_baseline` |
| Spatial | `forearm_slim_uv` | `spatial_slim_uv` |
| Spatial | `population_response_fields` | `spatial_extract_boundaries` |
| Spatial | `session_rf_boundary_comparison` | `spatial_compare_boundaries` |
| Spatial | `rf_center_proximal_distal` | `spatial_compare_rf_centers` |
| Stimulus | `touch_features` | `stimulus_extract_features` |
| Stimulus | `touch_clusters` | `stimulus_cluster_touches` |
| Stimulus | `touch_comparisons` | `stimulus_compare_clusters` |
| Stimulus | `touch_feature_radar` | `stimulus_render_radar` |
| Stimulus | `ap_efficacy` | `stimulus_analyse_efficacy` |
| Cross | `population_rf_grid` | `cross_map_feature_grid` |
| Cross | `population_rf_grid_metrics` | `cross_extract_grid_metrics` |
| Cross | `population_rf_grid_metrics_heatmaps` | `cross_render_grid_metrics` |
| Cross | `session_comparison` | `cross_render_sessions` |
| Cross | `receptive_field_maps_clustered` | `cross_cluster_rf` |

---

## Success Criteria

- [ ] Zero grep hits for old directory name strings in Python source
- [ ] `pytest code/tests/` passes with no regressions
- [ ] `python scripts/migrate_output_dirs.py --dry-run` reports all 20
      renames without errors on a real database path
- [ ] Pipeline runs a dry run (one cheap task enabled) and writes output to
      the new directory name
- [ ] `code/src/analysis/CLAUDE.md` reflects the new directory layout

---

## Technical Design

### Approach

**Centralized constants module** (`output_dirs.py`) defines all 20 path
segment names as module-level string constants.
All pipeline files import from it. This is the same single-path pattern used
for Kinect depth access (see `note-kinect-depth-access-single-path.md`).

The rename in pipeline source is then mechanical: replace hardcoded string
literals with constant references. The migration script handles on-disk data.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Centralized constants + migration script | No future scatter; data preserved | More files to change upfront | **Chosen** |
| Simple grep-replace, no constants module | Faster | Scatter problem recurs on next rename | Rejected |
| Split shared dirs into per-task dirs | Cleaner 1:1 mapping | Downstream readers need to find multiple dirs | Rejected |
| Accept recomputation, no migration | Zero migration risk | Expensive; invalidates months of compute | Rejected |

### Architecture Changes

**New file:**
- `code/src/analysis/pipeline/output_dirs.py` — string constants for all
  output subdirectory names; no logic, no imports.

**Modified files (code path replacement + normalization + docstrings):**

Workflow scripts:
- `code/scripts/analysis_workflow_processing.py`
- `code/scripts/analysis_workflow_viewers.py`
- `code/scripts/diagnose_stroke_direction.py`
- `code/scripts/inspect_touch_feature_ranges.py`

Pipeline functions (code + docstrings):
- `code/src/analysis/receptive_field_mapping/pipelines/rf_simple_pipeline.py`
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_response_field_pipeline.py`
- `code/src/analysis/receptive_field_mapping/pipelines/rf_session_boundary_comparison_pipeline.py`
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_center_pipeline.py`
- `code/src/analysis/receptive_field_mapping/pipelines/rf_cluster_pipeline.py`
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_grid_pipeline.py`
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_grid_metrics_pipeline.py`
- `code/src/analysis/receptive_field_mapping/pipelines/rf_touch_feature_radar_pipeline.py`
- `code/src/analysis/receptive_field_mapping/pipelines/rf_cluster_gui_launchers.py`
- `code/src/analysis/receptive_field_mapping/rendering/rf_session_comparison_renderer.py`
- `code/src/analysis/receptive_field_mapping/rendering/rf_population_grid_metrics_renderer.py`
- `code/src/analysis/touch_analytics/gui/preparation_viewer_data.py`

Docstring-only:
- `code/src/analysis/touch_analytics/clustering_pipeline.py`

**New file:**
- `code/scripts/migrate_output_dirs.py` — standalone migration script;
  renames `4_analysed/<old>` → `4_analysed/<new>`, deletes sentinel JSONs
  inside renamed dirs so the pipeline re-validates them on next run.

**Modified documentation:**
- `code/src/analysis/CLAUDE.md` — update directory layout section

---

## Implementation Plan

### Phase 1: Constants Module
**Goal:** Create `output_dirs.py` with all directory name constants.

- [ ] Create `code/src/analysis/pipeline/output_dirs.py` with 20 constants
      plus `RENAME_MAPPING` dict (old → new)

**New file:** `code/src/analysis/pipeline/output_dirs.py`

**Dependencies:** None

### Phase 2: Normalize batch pipeline functions to pattern A
**Goal:** Refactor 7 functions that construct their own output directory to
accept `output_dir: Path` instead, and update their callers.

- [ ] `run_population_response_field_extraction` — accept `output_dir`
- [ ] `run_session_rf_boundary_comparison` — accept `output_dir`
- [ ] `run_proximal_distal_center_comparison` — accept `output_dir`
- [ ] `run_touch_feature_radar` — accept `output_dir`
- [ ] `run_population_rf_grid` — receive full path, remove internal append
- [ ] `run_population_rf_grid_metrics` — receive full path, remove internal append
- [ ] `run_session_comparison_visualization` / renderer — accept `output_dir`
- [ ] Update all callers in `analysis_workflow_processing.py`

**Dependencies:** Phase 1

### Phase 3: Replace all hardcoded path strings
**Goal:** Replace every remaining hardcoded `4_analysed/<subdir>` string with
a constant import, including input paths and docstrings.

- [ ] Workflow scripts (4 files)
- [ ] Pipeline functions — input/sibling path references (12 files)
- [ ] Docstrings (`clustering_pipeline.py` + files with mixed refs)
- [ ] Grep for residual strings — zero hits outside migration script

**Dependencies:** Phases 1-2

### Phase 4: Migration Script
**Goal:** Write and validate the on-disk migration script.

- [ ] `code/scripts/migrate_output_dirs.py` — import `RENAME_MAPPING`,
      `--database-path`, `--dry-run`, idempotent, deletes sentinel JSONs

**New file:** `code/scripts/migrate_output_dirs.py`

**Dependencies:** Phase 1

### Phase 5: Documentation
**Goal:** Update CLAUDE.md to reflect the new directory layout.

- [ ] `code/src/analysis/CLAUDE.md` — new names + `output_dirs.py` reference

**Dependencies:** Phases 1-3

### Phase 6: Verification
**Goal:** Confirm zero stale references and no regressions.

- [ ] Grep for every old directory name — zero hits in Python source
- [ ] `pytest code/tests/` passes
- [ ] Migration `--dry-run` reports all 20 renames
- [ ] Pipeline end-to-end: output lands in new directory

**Dependencies:** Phases 1-5

---

## Testing Plan

### Unit Tests
- [ ] `test_dag_config_model.py` — no directory name references, passes unchanged
- [ ] Any test that constructs a `4_analysed/<subdir>` path must be updated
      to use the new constant

### Integration Tests
- [ ] Migration script dry-run produces exactly 20 rename operations on a
      populated database, no errors
- [ ] Pipeline stage writes to the new directory name on a fresh run

### Manual Verification
- [ ] `python code/scripts/migrate_output_dirs.py --dry-run --database-path <path>`
      prints all 20 renames
- [ ] After running migration (live), old dirs are gone, new dirs exist, data
      is intact
- [ ] Sentinel files inside migrated dirs are absent (deleted by script),
      confirming pipeline will re-validate

### Edge Cases
- [ ] Migration on a database with only some dirs present — missing old dirs
      silently skipped
- [ ] Migration run twice — second run is a no-op (idempotent)
- [ ] New dir already exists when migration runs — raise with a clear message
      (fail-fast)

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — directory layout section
- [ ] No new knowledge-base note needed (pattern already covered by
      `note-kinect-depth-access-single-path.md`)

---

## Rollback Plan

1. `git revert <merge-commit>` — reverts all Python source changes atomically.
2. Run migration script in reverse (swap old↔new in the mapping) — or
   restore from a pre-migration directory backup.
3. No database schema changes; only filesystem directory names change.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| A path string missed by grep stays hardcoded | Low | Med | Grep audit in Phase 5; pipeline run writes to new dir or fails loudly |
| Migration script renames wrong dir on a typo | Low | High | `--dry-run` mandatory first; raise on unexpected state |
| New dir already exists (partial prior migration) | Low | High | Script raises immediately — fail-fast, user inspects |
| Sentinel deletion causes full recompute of a session | Med | Low | Expected behaviour; recompute is correct after a dir rename |
| Tests reference old dir names in fixtures | Low | Med | pytest run in Phase 6 catches this |
| Normalization changes function signatures | Low | Med | Each caller is in the same workflow script — lockstep update |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Constants module | ~15 min | None |
| Phase 2: Normalize to pattern A | ~45 min | Phase 1 |
| Phase 3: Replace path strings | ~60 min | Phases 1-2 |
| Phase 4: Migration script | ~30 min | Phase 1 |
| Phase 5: Documentation | ~10 min | Phases 1-3 |
| Phase 6: Verification | ~20 min | Phases 1-5 |

---

## References

- Predecessor plan: `docs/development/plans/active/rf-analysis-task-nomenclature.md`
- Knowledge base: `docs/development/knowledge-base/note-kinect-depth-access-single-path.md` — single-path + allowlist pattern
- Knowledge base: `docs/development/knowledge-base/note-rf-cluster-visualization-overview.md` — sentinel/idempotency pattern
