# Plan: Analysis Workflow — Processing vs. Viewer Separation

**Date:** 2026-05-04
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-05 07:26
**Base Branch:** `tweak/viewer-speed-33x`
**Branch:** `feature/analysis-workflow-processing-viewer-separation`

---

## Overview

The analysis workflow mixes processing tasks (produce files, run unattended) with viewer tasks (launch interactive PyQt5 GUIs that block the pipeline) in a single flat loop. This plan separates them so processing can run unattended and viewers are launched only on demand.

## Problem Statement

`run_batch_analysis()` in `code/scripts/analysis_workflow.py` iterates 15 tasks in a flat list. GUI tasks (e.g., `explore_preparation`, `explore_touch_playback`) call `app.exec_()` and block the entire pipeline until the user closes the window. The task order also interleaves viewers between processing steps — `explore_preparation` sits between `touch_preparation` and `touch_series_transforms` — so processing cannot complete unattended.

Additionally, `visualize_receptive_fields_clustered` is a hybrid: its primary job is rendering heatmap PNGs (processing), but it optionally launches the `RFClusterGalleryViewer` GUI when `gallery_viewer: true`, blocking the pipeline mid-processing.

## Goals

### In Scope
1. Classify all 15 analysis tasks into processing, viewer-support, or viewer categories
2. Reorder the task execution list so all processing completes before any viewer launches
3. Add a `--mode` CLI flag (`all` / `processing` / `viewers`) for selective execution
4. Extract the gallery viewer from `visualize_receptive_fields_clustered` into a standalone `explore_rf_gallery` viewer task
5. Add a `category` field to each task in the DAG YAML for documentation and filtering

### Out of Scope
- Splitting into two separate DAG YAML files (one config is simpler to manage)
- Parallelizing processing tasks (Prefect concurrency is a separate concern)
- Changing the merging visualisation workflow (`merging_pipeline_neuron_to_kinect_visualisation.py` — already separate)
- Removing the legacy `map_receptive_fields_clustered` task (just document it as deprecated)

## Success Criteria

- [ ] `--mode processing` runs all 10 processing tasks without opening any GUI window
- [ ] `--mode viewers` runs only viewer + viewer-support tasks (assumes processing outputs exist on disk)
- [ ] `--mode all` (default) preserves current behaviour: processing first, then viewers
- [ ] `explore_rf_gallery` launches the gallery viewer as a standalone task after `visualize_receptive_fields_clustered`
- [ ] `gallery_viewer` option in `visualize_receptive_fields_clustered` emits a deprecation warning directing to `explore_rf_gallery`
- [ ] `pytest code/tests/` passes with no regressions

---

## Technical Design

### Task Classification

**Processing** (10 tasks — produce files, run unattended):

| Task | Output |
|------|--------|
| `summarize_session_blocks` | `session_block_summary.csv` |
| `map_receptive_fields_simple` | `spike_positions.csv` + heatmap PNGs |
| `touch_preparation` | `<session>_prepared.csv` |
| `touch_series_transforms` | `<session>_series_augmented.csv` |
| `touch_feature_extraction` | per-feature `touch_summary.csv` |
| `touch_clustering` | `pooled_touch_summary_clustered.csv` + metadata JSON |
| `touch_comparing` | comparison result JSONs |
| `analyse_ap_efficacy` | `batch_ap_efficacy_matrix.csv` + heatmap PNGs |
| `extract_receptive_fields_clustered` | `spike_counts.csv` + extraction artifacts |
| `visualize_receptive_fields_clustered` | RF heatmap PNGs + `rf_metrics_summary.csv` |

Notes:
- `analyse_ap_efficacy` has `show=True` but is NOT blocking — `VisualReportingStrategy` sets `plt.switch_backend('Agg')`.
- `map_receptive_fields_simple` has `show_interactive` option (default `false`). When `false`, pure processing.

**Viewer Support** (1 task — utility for viewer tasks only):

| Task | Output | Consumed by |
|------|--------|-------------|
| `precompute_explorer_caches` | `.npz` sidecar caches | `explore_rf_feature_space` only |

**Viewer** (4 tasks — launch interactive PyQt5 GUIs):

| Task | GUI |
|------|-----|
| `explore_preparation` | `TouchPreparationViewer` |
| `explore_rf_feature_space` | `RFFeatureSpaceExplorer` |
| `explore_touch_playback` | `TouchPlaybackExplorer` |
| `explore_rf_gallery` (NEW) | `RFClusterGalleryViewer` |

**Legacy** (1 task — deprecated):

| Task | Notes |
|------|-------|
| `map_receptive_fields_clustered` | Comment says "use extract + visualize instead" |

### Approach

Minimal changes that preserve backward compatibility:
1. Reorder the existing `available_tasks` list (zero-risk, same tasks, same dependency system)
2. Add a `category` metadata field to the DAG YAML (`DagConfigHandler` ignores unknown keys)
3. Filter tasks by category when `--mode` is specified
4. Extract gallery viewer into its own flow function using the existing `launch_gallery_viewer()` function

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Single DAG + `category` field + `--mode` flag | Simple, one config file, backward-compatible | Category field is just metadata (not enforced) | Chosen |
| Two separate DAG files (processing.yaml + viewers.yaml) | Hard separation | Duplicates `kinect_configs` parameter, two files to maintain, breaks existing launcher | Rejected |
| Separate Python scripts (processing_workflow.py + viewer_workflow.py) | Complete isolation | Large refactor, duplicates session resolution and boilerplate | Rejected |

### Architecture Changes

No new modules. Changes are within existing files:

- `analysis_workflow.py`: reorder task list, add `explore_rf_gallery_flow`, add `--mode` filtering
- `analyse_workflow_dag.yaml`: add `category` fields, add `explore_rf_gallery` entry, reorder sections
- `rf_cluster_pipeline.py`: deprecation warning when `gallery_viewer=True` in `run_cluster_rf_visualization`

Existing function to reuse:
- `launch_gallery_viewer()` at `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py:891` — already cleanly factored with state persistence (JSON sidecars) and deferred PyVista initialization

---

## Implementation Plan

### Phase 1: Reorder Tasks + Add Category Metadata
**Goal:** Processing runs before viewers; each task is labeled by category.

- [x] Task 1.1 — Reorder `available_tasks` list: processing first, then viewer-support, then viewers
- [x] Task 1.2 — Add `category` field (`processing` / `viewer_support` / `viewer`) to every task in DAG YAML
- [x] Task 1.3 — Restructure DAG YAML with section comments separating processing and viewer blocks
- [x] Task 1.4 — Add YAML comment on `map_receptive_fields_simple.options.show_interactive` clarifying it is a debugging toggle
- [x] Task 1.5 — Add YAML comment on `map_receptive_fields_clustered` marking it as deprecated

**Started:** 2026-05-04

**Files Modified:**
- `code/scripts/analysis_workflow.py` — reorder lines 583-599
- `configs/analyse_workflow_dag.yaml` — add `category` fields, reorder sections, add comments

**Dependencies:** None

### Phase 2: Add `--mode` CLI Flag
**Goal:** Selective execution — run only processing, only viewers, or both.

- [x] Task 2.1 — Add `--mode {all,processing,viewers}` argument to `main()` argparse
- [x] Task 2.2 — Pass mode to `run_batch_analysis()` as a new parameter
- [x] Task 2.3 — In the task loop, read each task's `category` from the DAG config and skip when mode does not match
- [x] Task 2.4 — Log which tasks are skipped due to mode filtering (distinct from "disabled in DAG")

**Started:** 2026-05-04
**Completed:** 2026-05-04

**Files Modified:**
- `code/scripts/analysis_workflow.py` — `main()` (argparse), `run_batch_analysis()` (filtering logic)

**Dependencies:** Phase 1

### Phase 3: Extract Gallery Viewer Task
**Goal:** Gallery viewer becomes a standalone viewer task; processing task no longer launches GUIs.

- [x] Task 3.1 — Create `explore_rf_gallery_flow` in `analysis_workflow.py`: iterate enabled `cluster_groups`, call `launch_gallery_viewer()` for each combo/clusterer
- [x] Task 3.2 — Add `explore_rf_gallery` entry to DAG YAML with `category: viewer`, `depends_on: [visualize_receptive_fields_clustered]`
- [x] Task 3.3 — Add `launch_gallery_viewer` to the imports from `analysis.receptive_field_mapping`
- [x] Task 3.4 — In `run_cluster_rf_visualization()`, emit a deprecation warning when `gallery_viewer=True` directing to `explore_rf_gallery`
- [x] Task 3.5 — Change `gallery_viewer` default to `false` in the DAG YAML for `visualize_receptive_fields_clustered`
- [x] Task 3.6 — Forward `cluster_group_defs` to `explore_rf_gallery_flow` in `run_batch_analysis()`

**Started:** 2026-05-04
**Completed:** 2026-05-04

**Files Modified:**
- `code/scripts/analysis_workflow.py` — new flow (~30 lines), add to `available_tasks`, forward kwargs
- `configs/analyse_workflow_dag.yaml` — new `explore_rf_gallery` entry, change `gallery_viewer` default
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — deprecation warning in `run_cluster_rf_visualization`
- `code/src/analysis/receptive_field_mapping/__init__.py` — export `launch_gallery_viewer` if not already exported

**Dependencies:** Phase 1, Phase 2

---

## Testing Plan

### Unit Tests
- [ ] Existing `pytest code/tests/` passes without regressions
- [ ] `DagConfigHandler` correctly ignores the new `category` field (no parsing errors)

### Integration Tests
- [ ] `--mode processing` with all viewers enabled in DAG: no GUI windows open, all processing tasks run
- [ ] `--mode viewers` with processing outputs on disk: only viewer tasks execute
- [ ] `--mode all` (default): same behaviour as before this change

### Manual Verification
- [ ] Run `python code/scripts/analysis_workflow.py --dag-config configs/analyse_workflow_dag.yaml --mode processing` — completes without opening any window
- [ ] Run with `--mode viewers` — viewer tasks launch their GUIs
- [ ] Run with `explore_rf_gallery` enabled — gallery viewer opens after RF visualization data exists
- [ ] Confirm `gallery_viewer: true` on `visualize_receptive_fields_clustered` emits deprecation warning in the log

### Edge Cases
- [ ] `--mode viewers` when processing outputs do not exist on disk — tasks fail with clear error, not silent corruption
- [ ] All viewer tasks disabled in DAG + `--mode all` — no GUI windows, processing runs normally
- [ ] Legacy `map_receptive_fields_clustered` still works if re-enabled (backward compat)

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — add task category classification and `--mode` flag documentation
- [ ] Update `CLAUDE.md` root — mention the `--mode` flag in the Development Commands section
- [ ] DAG YAML comments serve as inline documentation for task categories

---

## Rollback Plan

All changes are additive and backward-compatible:

1. **`category` field:** `DagConfigHandler` ignores unknown keys — removing the field has no effect
2. **`--mode` flag:** defaults to `all`, which is current behaviour — removing the flag reverts cleanly
3. **`explore_rf_gallery` task:** a new optional task — disabling or removing it restores previous behaviour
4. **Gallery viewer deprecation:** the warning is non-blocking — removing it is a one-line revert

No data migrations, no breaking changes, no schema modifications.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Task reorder changes execution order for edge-case dependency chains | Low | Low | Dependency system (`DagConfigHandler`) already handles ordering; reorder only affects default iteration sequence |
| `--mode viewers` fails because processing outputs are missing | Medium | Low | Tasks already raise on missing inputs (fail-fast convention); add a clear log message suggesting `--mode processing` first |
| `explore_rf_gallery` does not find extraction artifacts | Low | Low | Depends on `visualize_receptive_fields_clustered` which itself depends on extraction; DAG dependency chain covers this |
| `launch_gallery_viewer` not exported from `__init__.py` | Low | Low | Check export in Phase 3; add if missing |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Reorder + category metadata | Small (reorder list, add YAML fields) | None |
| Phase 2: `--mode` CLI flag | Small (argparse + filter logic) | Phase 1 |
| Phase 3: Extract gallery viewer | Medium (new flow + deprecation + DAG entry) | Phase 1, Phase 2 |

---

## References

- Investigation notes: `.claude/plans/investigate-the-flows-in-indexed-waffle.md`
- Knowledge base: `docs/development/knowledge-base/note-rf-cluster-gallery-gui-components.md`
- Knowledge base: `docs/development/knowledge-base/note-rf-gui-comparison-gallery-vs-explorer.md`
