# Plan: Cluster-Based Receptive Field Mapping

**Created:** 2026-03-17 00:30
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/cluster-based-rf-mapping`

---

## Overview

**What:** Add a new receptive field mapping flow that groups touches by `cluster_label` (assigned by the unified touch analysis pipeline's clustering phase) instead of the current `type_metadata`/`direction` discretization, with per-session and per-cluster output isolation.

**Why:** The current `map_receptive_fields` flow uses hardcoded grouping columns (`type_metadata`, `direction`) and writes all output to a flat directory. The unified pipeline now assigns data-driven cluster labels to each touch; RF mapping should leverage these clusters to discover receptive fields that correspond to mechanically meaningful touch categories rather than experimenter-defined labels.

**How:** Add `session_id` to the unified pipeline output so pooled CSVs retain session identity, create a cluster-aware data loader, and implement a new Prefect flow that iterates all extraction_profile x clustering_method combinations per session.

## Problem Statement

- The unified pipeline assigns `cluster_label` to touches based on extracted features, but the RF mapping flow ignores these labels and uses its own discretization-based grouping.
- RF mapping outputs go to a flat `receptive_field_maps/` directory, making it hard to compare results across sessions, extraction profiles, and clustering methods.
- The pooled `pooled_touch_summary_clustered.csv` loses session identity after `pd.concat`, preventing per-session filtering downstream.

## Goals

### In Scope
1. Add `session_id` column to unified pipeline per-session CSVs (propagates to pooled clustered CSV)
2. Add a cluster-aware data loader that builds group lookups from `cluster_label`
3. Add a new Prefect flow `map_receptive_fields_clustered` that runs RF mapping per session, per cluster, for each extraction_profile x clustering_method combination
4. Per-session, per-cluster output directory structure

### Out of Scope
- Modifying the existing `map_receptive_fields` flow (it continues to work as-is)
- Adding new extraction or clustering methods
- Interactive Open3D visualization (reuse existing `RFVisualizer` as-is)
- Comparing RF maps across clusters or sessions (future work)

## Success Criteria

- [ ] `session_id` column present in all per-session and pooled clustered CSVs
- [ ] Existing `map_receptive_fields` flow produces identical output (no regression)
- [ ] New flow produces output at `4_analysed/receptive_field_maps/<profile>/<clusterer>/<session_id>/`
- [ ] Each session subdirectory contains per-cluster CSVs, PNGs, and a `rf_mapping_summary.json`
- [ ] Sessions with zero touches for a cluster produce no empty files for that cluster
- [ ] Idempotency: re-running with unchanged inputs skips processing

---

## Technical Design

### Approach

Keep RF mapping as a **separate Prefect flow** (not embedded in `unified_pipeline.py`). The unified pipeline is a pure data-processing module with no Open3D or visualization dependencies; mixing RF mapping into it would bloat its dependency surface. Instead, add a new flow `map_receptive_fields_clustered_flow()` that reads the unified pipeline's clustered CSV output and runs the existing `RFMappingEngine` per cluster per session.

The key prerequisite is adding a `session_id` column to the unified pipeline's per-session CSVs so that session identity survives `pd.concat` during clustering. This is a small, backward-compatible addition.

For the data loader, add a new `_build_cluster_group_lookup()` function rather than modifying the existing `_build_group_lookup()`, since the two strategies are fundamentally different (one reads discretized metadata, the other reads pre-assigned labels). The common raw-CSV iteration loop is refactored into a shared `_accumulate_spatial_data()` helper.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Separate flow + cluster-aware loader | Clean separation of concerns; existing flow untouched; reuses RF engine/visualizer | New flow function to maintain | **Chosen** |
| Embed RF mapping as Phase 3 in `unified_pipeline.py` | Single orchestration point | Mixes Open3D/viz deps into data pipeline; breaks module boundaries | Rejected |
| Add conditional branch to existing `map_receptive_fields_flow` | No new flow | Combinatorial parameter complexity; harder to reason about | Rejected |

### Architecture Changes

No new modules. Changes are additive to existing files:

```
code/src/analysis/touch_analytics/unified_pipeline.py
  └── Add 'session_id' to _SHARED_COLUMNS and per-session CSV output

code/src/analysis/receptive_field_mapping/rf_data_loader.py
  ├── Refactor: _accumulate_spatial_data() (extracted from existing loop)
  ├── New: _build_cluster_group_lookup()
  └── New: load_grouped_spatial_data_by_cluster()

code/src/analysis/receptive_field_mapping/__init__.py
  └── Export load_grouped_spatial_data_by_cluster

code/scripts/analysis_workflow.py
  └── New: map_receptive_fields_clustered_flow() + registration in available_tasks

configs/analyse_workflow_dag.yaml
  └── New task: map_receptive_fields_clustered
```

Output directory structure:

```
4_analysed/receptive_field_maps/
  max/
    kmeans/
      ST13-01/
        rf_map_cluster_0.csv
        rf_map_cluster_0.png
        rf_map_cluster_1.csv
        rf_map_cluster_1.png
        rf_mapping_summary.json
      ST13-02/
        ...
    dbscan/
      ST13-01/
        ...
  statistical/
    kmeans/
      ...
```

---

## Implementation Plan

### Phase 1: Add `session_id` to Unified Pipeline
**Goal:** Ensure session identity survives pooling so downstream consumers can filter by session.
**Started:** —
**Completed:** —

- [ ] Add `'session_id'` to `_SHARED_COLUMNS` list
- [ ] In `_extract_session()`, insert `session_id` column into `summary_df` using the `prefix` variable (already computed at line 162)

**Files Modified:**
- `code/src/analysis/touch_analytics/unified_pipeline.py` — Add `session_id` to shared columns and per-session output

**Dependencies:** None

### Phase 2: Cluster-Aware Data Loader
**Goal:** Add a new loader path that builds group lookups from `cluster_label` instead of discretization.
**Started:** —
**Completed:** —

- [ ] Refactor raw-CSV iteration loop (lines 216-276 of `rf_data_loader.py`) into `_accumulate_spatial_data(group_lookup, raw_csv_paths, config)` shared helper
- [ ] Update `load_grouped_spatial_data()` to use `_accumulate_spatial_data()` (preserving identical behavior)
- [ ] Add `_build_cluster_group_lookup(clustered_csv_path, session_id, trial_id_col, touch_id_col)` — reads pooled CSV, filters by `session_id`, returns `{(trial_id, touch_id): "cluster_<N>"}` dict
- [ ] Add `load_grouped_spatial_data_by_cluster(raw_csv_paths, clustered_csv_path, session_id, config)` — calls `_build_cluster_group_lookup()` + `_accumulate_spatial_data()`
- [ ] Export `load_grouped_spatial_data_by_cluster` from `__init__.py`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_data_loader.py` — Refactor loop, add cluster-aware functions
- `code/src/analysis/receptive_field_mapping/__init__.py` — Export new function

**Dependencies:** Phase 1

### Phase 3: New Flow and DAG Integration
**Goal:** Wire cluster-based RF mapping into the Prefect workflow and DAG config.
**Started:** —
**Completed:** —

- [ ] Add `map_receptive_fields_clustered_flow()` to `analysis_workflow.py`:
  - Accepts `input_items`, `force_processing`, `monitor`, optional `extraction_profiles`/`clustering_profiles`
  - Defaults profiles from unified pipeline's `_DEFAULT_OPTIONS`
  - For each `(input_file, database_path)` x `(profile, clusterer)`:
    - Locates `pooled_touch_summary_clustered.csv` (skips if missing)
    - Creates output dir: `4_analysed/receptive_field_maps/<profile>/<clusterer>/<session_id>/`
    - Idempotency via `should_process_task()`
    - Calls `load_grouped_spatial_data_by_cluster()`
    - Per cluster: `compute_selectivity()` + `cluster_receptive_field()` + save CSV/PNG/JSON
- [ ] Register `map_receptive_fields_clustered` in `available_tasks` dict
- [ ] Add `map_receptive_fields_clustered` task to `configs/analyse_workflow_dag.yaml` with `depends_on: ["unified_touch_analysis"]`

**Files Modified:**
- `code/scripts/analysis_workflow.py` — Add new flow function and register in available_tasks
- `configs/analyse_workflow_dag.yaml` — Add task config entry

**Dependencies:** Phase 2

---

## Testing Plan

### Integration Tests
- [ ] Re-run unified pipeline with `force_processing: true`; verify `session_id` column in per-session and pooled CSVs
- [ ] Run `map_receptive_fields_clustered` task; verify output directory structure matches spec
- [ ] Run existing `map_receptive_fields` flow; verify identical output (no regression from `session_id` addition)

### Manual Verification
- [ ] Inspect per-cluster CSVs: columns are `[x, y, z, selectivity, cluster_id]`, values are plausible
- [ ] Inspect per-cluster PNGs: visualizations render correctly
- [ ] Inspect `rf_mapping_summary.json`: contains correct `extraction_profile`, `clustering_method`, `session_id`, and per-cluster stats

### Edge Cases
- [ ] Session with all touches in one cluster — produces a single `rf_map_cluster_X.csv/png`
- [ ] Session with no touches in a cluster — that cluster is skipped, no empty files
- [ ] Missing `pooled_touch_summary_clustered.csv` for a profile/clusterer combo — logs warning, skips gracefully

---

## Documentation Plan

- [ ] Update inline comments in modified files
- [ ] Update unified pipeline docstring to document `session_id` column

---

## Rollback Plan

1. Revert the commits on `feature/cluster-based-rf-mapping`
2. The `session_id` column addition is backward-compatible; existing consumers ignore it, but it can be reverted by removing it from `_SHARED_COLUMNS` and the `insert()` call
3. No data migrations; output files are additive (new directory structure alongside existing)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Trial ID collisions across sessions in pooled CSV | Med | High | `session_id` column resolves this; `_build_cluster_group_lookup` filters by session first |
| Cluster labels change on re-run (K-means non-determinism) | High | Low | Inherent to clustering; `cluster_metadata.json` provides traceability; documented, not a bug |
| Large number of profile x clusterer combos (4x2=8) | Low | Low | Each combo is fast (DBSCAN on small spatial point sets); clear logging of progress |
| Empty clusters for some sessions | Med | Low | Flow skips empty groups gracefully; no empty files created |

---

## References

- Related Plans: `docs/development/plans/pending/unified-touch-analysis-pipeline.md`
- Knowledge Base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md` (spatial data is in mm from Kinect SDK)

---
