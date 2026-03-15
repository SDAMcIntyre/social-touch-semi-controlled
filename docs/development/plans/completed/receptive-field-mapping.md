# Plan: Receptive Field Mapping via Selectivity + DBSCAN

**Date:** 2026-03-12
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/receptive-field-mapping`
**Supersedes:** `receptive-field-filtering.md` (archived)

---

## Overview

Migrate receptive field (RF) determination from the post-processing pipeline into the analysis pipeline, rename it from "receptive field filtering" to **"receptive field mapping"**, and upgrade the algorithm from naive exact-point matching to selectivity-scoring + DBSCAN clustering. The new module produces one RF map per category combination, where categories are provided externally by the caller (group-agnostic design). This resolves the architectural tension where RF computation needed category information that only existed downstream in the analysis pipeline.

## Problem Statement

The current `determine_receptive_field.py` (post-processing, stage 5) has three problems:

1. **Wrong pipeline stage.** RF mapping needs per-touch categories (depth, velocity, type, direction) to produce per-category RF maps. These categories are computed in the analysis pipeline (stage 6), which runs *after* post-processing. The data flow is backwards.

2. **Naive algorithm.** The current approach tags a row as "on RF" if any of its `contact_points` appear in the set of spike-associated points (exact float-tuple matching). This conflates true RF hotspots with co-touched bystander areas — when the RF is touched, surrounding non-RF skin is touched simultaneously and falsely appears spike-associated.

3. **No per-category RF.** The current implementation computes a single global RF. In reality, RF boundaries vary by touch characteristics (e.g., light/slow touches may activate a smaller RF than deep/fast touches). Per-category RF maps are needed.

Additionally, `filter_by_receptive_field` (the downstream consumer in post-processing) is a stub that returns `[]`, making the post-processing RF pipeline incomplete.

## Goals

### In Scope
1. Create a group-agnostic RF mapping engine under `code/src/analysis/receptive_field_mapping/` that receives pre-grouped spatial data and produces RF maps via selectivity + DBSCAN
2. Integrate as a new Prefect flow in the analysis workflow (`map_receptive_fields`), consuming both unified touch summaries (for grouping) and raw merged CSVs (for spatial data)
3. Output one RF map per category combination: cluster definitions, selectivity scores, and stats
4. Visualize RF clusters per group via Open3D
5. Support configurable grouping columns via DAG options (the module doesn't know what groups mean)
6. Deprecate `determine_receptive_field` and `filter_by_receptive_field` in the post-processing DAG

### Out of Scope
- Defining a fixed set of categories — the caller chooses grouping columns at runtime
- Dropping non-RF rows from CSVs (output is RF maps, not filtered data)
- GPU acceleration of DBSCAN
- Migrating `set_xyz_reference_from_gestures` out of post-processing (captured as separate idea)
- Row-level `on_receptive_field` tagging in the analysis pipeline (the output is spatial maps, not tagged CSVs)

## Success Criteria

- [ ] RF mapping engine produces correct selectivity scores and DBSCAN clusters on a known session
- [ ] Per-group RF maps differ meaningfully (e.g., tap RF vs stroke RF show different spatial extent)
- [ ] New selectivity-based approach excludes bystander points that the old exact-matching included
- [ ] Engine is fully group-agnostic — no hardcoded knowledge of category names or semantics
- [ ] Analysis workflow runs `map_receptive_fields` after `process_unified_touches` via DAG dependency
- [ ] Open3D visualization shows RF clusters colored by membership with selectivity brightness
- [ ] Algorithm parameters (eps, min_samples, selectivity_threshold) configurable via DAG options
- [ ] Old post-processing RF tasks disabled in DAG config
- [ ] Idempotency preserved (`should_process_task`)

---

## Technical Design

### Approach

**Two-source data join + per-group selectivity + DBSCAN:**

1. **Grouping** (data loader): Read the unified touch summary CSV to get per-touch kinematics. Discretize continuous grouping columns into bins. Assign each `(trial_id, single_touch_id)` a group label.

2. **Spatial aggregation** (data loader): Iterate raw merged CSV frames. For each frame, look up its touch's group. Accumulate per-group `spike_counts` and `total_counts` Counters for each unique 3D contact point.

3. **Selectivity scoring** (engine): For each group, compute `selectivity = spike_count / total_touch_count` per point. The existing `active_points_counter` and `global_points_counter` pattern from `determine_receptive_field.py` provides exactly these inputs — the new version adds division rather than set membership.

4. **DBSCAN clustering** (engine): Filter points above selectivity threshold, run `sklearn.cluster.DBSCAN` on survivors to find spatially coherent RF clusters. Discard noise and micro-clusters.

5. **Output**: Per-group `RFMapResult` containing clusters (point arrays + stats), selectivity scores, and metadata. Saved as CSV + JSON per group.

**Coordinates:** All spatial data in mm (Azure Kinect SDK convention). Uses `contact_points_transformed` column via `resolve_column()`. DBSCAN `eps` is in mm.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Selectivity + DBSCAN in analysis pipeline (group-agnostic) | Resolves data flow issue; leverages discriminating signal; per-category RF; clean engine API | Requires two-source data join; more complex loader | **Chosen** |
| Keep RF in post-processing, import categories | Minimal pipeline changes | Circular dependency (post-proc needs analysis output); categories not available yet | Rejected — data flow is backwards |
| Convex hull of spike points per group | Simple geometry | Includes co-touched bystanders; single contiguous region only | Rejected — doesn't solve core problem |
| Raw DBSCAN on spike points (no selectivity) | Simpler | Clusters include bystander points; misses discriminating non-spike signal | Rejected — misses key insight |
| Standalone RF pipeline (separate from analysis) | Full independence | Duplicates infrastructure (Prefect flows, DAG handling, config resolution) | Rejected — unnecessary overhead |

### Architecture Changes

New module under `analysis` package:

```
code/src/analysis/receptive_field_mapping/
    __init__.py                    # re-exports public API
    rf_mapping_config.py           # config + result dataclasses
    rf_mapping_engine.py           # compute_selectivity(), cluster_receptive_field()
    rf_data_loader.py              # load & join unified summary + raw CSVs
    rf_visualizer.py               # Open3D cluster visualization
```

Follows the engine + config + visualizer pattern from `code/src/postprocessing/xyz_reference_from_gestures/` (pure logic with no I/O in the engine).

**Integration points:**
- `analysis.touch_analytics.touch_config.DISCRETIZATION_CONFIG` — reused for continuous variable binning in the data loader
- `preprocessing.forearm_extraction.registration.csv_spatial_transformer.resolve_column()` — column resolution for transformed coordinates
- `utils.gui.visualize_point_cloud_comparison()` — Open3D side-by-side visualization
- `utils.should_process_task` — idempotency
- `preprocessing.forearm_extraction.ForearmCatalog` — forearm point cloud loading for visualization

### Knowledge Base Constraints

From `note-cupy-import-order.md`: Not directly applicable — the new module uses sklearn (CPU), not CuPy. No preprocessing imports in the engine itself.

From `note-forearm-icp-registration.md`: RF visualization consumes already-transformed CSVs. Single-forearm sessions have no transforms JSON — visualizer handles `forearm_pcd=None` gracefully.

From `note-somatosensory-units-and-calculations.md`: All coordinates in mm. DBSCAN `eps` is in mm. Selectivity is unitless (0–1).

---

## Implementation Plan

### Phase 1: Configuration & Result Types
**Goal:** Define all data structures for the module

- [ ] Create `rf_mapping_config.py` with `SelectivityDBSCANConfig` (eps, min_samples, selectivity_threshold, min_cluster_points), `RFMappingColumnConfig` (touch_id, points, spike column names), `RFMappingConfig` (combines both), `RFCluster` (cluster_id, points array, mean_selectivity, point_count), `RFMapResult` (group_label, clusters list, selectivity scores dict, stats), `GroupedSpatialData` (group_label, spike_counts Counter, total_counts Counter, touch_count)
- [ ] Create `__init__.py` with re-exports of public API

**Files Created:**
- `code/src/analysis/receptive_field_mapping/__init__.py`
- `code/src/analysis/receptive_field_mapping/rf_mapping_config.py`

**Dependencies:** None

### Phase 2: Core Selectivity + DBSCAN Engine
**Goal:** Implement pure computational logic (no I/O)

- [ ] Implement `compute_selectivity(spike_counts: Counter, total_counts: Counter) -> dict[tuple, float]` — returns per-point selectivity ratio
- [ ] Implement `cluster_receptive_field(selectivity_scores: dict, config: SelectivityDBSCANConfig, group_label: str) -> RFMapResult` — filters by threshold, runs `sklearn.cluster.DBSCAN`, builds `RFCluster` objects, discards micro-clusters
- [ ] Log warnings for edge cases: too few points above threshold, no clusters found, very sparse data

**Files Created:**
- `code/src/analysis/receptive_field_mapping/rf_mapping_engine.py`

**Dependencies:** Phase 1

### Phase 3: Data Loader
**Goal:** Handle two-source join between unified summary and raw CSVs

- [ ] Implement `load_grouped_spatial_data(raw_csv_paths, summary_csv_path, grouping_columns, config, use_transformed) -> dict[str, GroupedSpatialData]`
  - Read unified summary → discretize continuous grouping columns using `DISCRETIZATION_CONFIG` from `touch_config.py` → build group label per `(trial_id, single_touch_id)`
  - Iterate raw CSV frames → parse `contact_points` → look up group for this touch → accumulate per-group spike/total Counters
- [ ] Copy `parse_contact_points()` from `determine_receptive_field.py` as canonical location in this module
- [ ] Reuse `resolve_column()` from `csv_spatial_transformer.py` for transformed column resolution

**Files Created:**
- `code/src/analysis/receptive_field_mapping/rf_data_loader.py`

**Dependencies:** Phase 1

### Phase 4: Visualization
**Goal:** Open3D display of per-group RF clusters on the forearm

- [ ] Implement `visualize_rf_map(rf_result: RFMapResult, forearm_pcd=None)` — left panel: forearm reference point cloud; right panel: RF points colored by cluster membership, brightness scaled by selectivity
- [ ] Reuse `visualize_point_cloud_comparison()` from `utils.gui`
- [ ] Handle `forearm_pcd=None` gracefully (empty placeholder)

**Files Created:**
- `code/src/analysis/receptive_field_mapping/rf_visualizer.py`

**Dependencies:** Phase 2

### Phase 5: Analysis Workflow Integration
**Goal:** Wire RF mapping into the analysis pipeline as a new Prefect flow

- [ ] Add `map_receptive_fields_flow` to `code/scripts/analysis_workflow.py`:
  - For each `(input_file, database_path)` pair: locate unified summary CSV, call `load_grouped_spatial_data()`, run `compute_selectivity()` + `cluster_receptive_field()` per group
  - Save outputs to `database_path / "4_analysed" / "receptive_field_maps/"`: per-group CSV (points + selectivity + cluster assignment) + summary JSON (config, cluster stats)
  - Optionally call `visualize_rf_map()` when `monitor=True`
- [ ] Register `("map_receptive_fields", map_receptive_fields_flow)` in `run_batch_analysis`'s `available_tasks` list
- [ ] Forward DAG options: `grouping_columns`, `monitor`, `use_transformed`, `force_processing`
- [ ] Add import of new module
- [ ] Idempotency via `should_process_task`

**Files Modified:**
- `code/scripts/analysis_workflow.py` — add new flow, register in available_tasks, add imports

**Dependencies:** Phases 2–4

### Phase 6: DAG Config & Package Wiring
**Goal:** Enable the new task in the analysis DAG

- [ ] Add `map_receptive_fields` task to `configs/analyse_workflow_dag.yaml` with `depends_on: [process_unified_touches]` and options: `grouping_columns`, `use_transformed`, `monitor`, `force_processing`
- [ ] Update `code/src/analysis/__init__.py` to expose `receptive_field_mapping` subpackage if needed
- [ ] Update `code/src/analysis/receptive_field_mapping/__init__.py` with final re-exports after implementation

**Files Modified:**
- `configs/analyse_workflow_dag.yaml`
- `code/src/analysis/__init__.py`
- `code/src/analysis/receptive_field_mapping/__init__.py`

**Dependencies:** Phase 5

### Phase 7: Post-processing Deprecation
**Goal:** Disable old RF code in post-processing (after validation)

- [ ] Set `determine_receptive_field: enabled: false` in `configs/postprocess_workflow_kinect_auto_dag.yaml`
- [ ] Set `filter_by_receptive_field: enabled: false` in same file
- [ ] Add deprecation docstring to `code/scripts/_5_postprocessing/determine_receptive_field.py` pointing to `analysis.receptive_field_mapping`

**Files Modified:**
- `configs/postprocess_workflow_kinect_auto_dag.yaml`
- `code/scripts/_5_postprocessing/determine_receptive_field.py`

**Dependencies:** Successful validation of Phase 6

---

## Testing Plan

### Manual Verification
- [ ] Run on a known session with a clear RF (e.g., ST13-02) — verify per-group RF maps are produced
- [ ] Compare old `on_receptive_field` tagging vs new selectivity-based clusters — verify the new approach excludes bystander points
- [ ] Inspect per-group selectivity CSV: verify gradient from high (near RF) to low (periphery)
- [ ] Visually confirm DBSCAN clusters in Open3D viewer align with expected RF location
- [ ] Verify RF maps differ between groups (e.g., tap RF vs stroke RF should show different spatial extent)
- [ ] Re-run to confirm idempotency (skips when outputs up-to-date)
- [ ] Run with different `grouping_columns` configurations to verify group-agnostic behavior

### Edge Cases
- [ ] Session with very few trials (sparse data) — verify `min_cluster_points` catches this and logs warning
- [ ] Session with no spikes — verify empty result per group, no crash
- [ ] Session with uniform selectivity (no clear RF) — verify sensible output (no clusters or all-noise)
- [ ] Single-forearm session (no registration transforms JSON) — verify graceful fallback in visualization
- [ ] Group with < 20 touches — verify warning logged, result still produced
- [ ] Missing unified summary CSV (process_unified_touches not run) — verify graceful skip with warning

---

## Documentation Plan

- [ ] Inline docstrings on all public functions in the new module
- [ ] Update CLAUDE.md if architectural patterns change
- [ ] Add knowledge base note if novel patterns emerge during implementation

---

## Rollback Plan

1. The new `code/src/analysis/receptive_field_mapping/` module can be deleted without affecting other code
2. The `map_receptive_fields` flow registration in `analysis_workflow.py` can be removed (single list entry + imports)
3. The `map_receptive_fields` task in `analyse_workflow_dag.yaml` can be removed or set `enabled: false`
4. The old `determine_receptive_field.py` remains in git history and can be re-enabled in the post-processing DAG
5. No database or external state changes — purely file-based outputs (CSVs, JSONs in `4_analysed/receptive_field_maps/`)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| DBSCAN `eps` too large — merges distinct RF regions | Medium | High | Default 5 mm is conservative for forearm scale (~300 mm); configurable via DAG options per session |
| Many group combinations produce sparse data per group | Medium | Medium | Log warning when group has < 20 touches; `min_cluster_points` filter discards unreliable clusters |
| Selectivity threshold too aggressive — drops true RF points | Medium | Medium | Default 0.3 is moderate; stats CSV preserves all scores for post-hoc tuning |
| Missing unified summary (process_unified_touches not run) | Low | High | `depends_on` in DAG config prevents this; flow also checks and skips gracefully |
| Memory pressure from many groups x many sessions | Low | Medium | Process one session at a time, streaming into per-group Counters |
| `parse_contact_points` duplication (old + new) | Low | Low | Canonical version in RF mapping module; old copy remains until consolidated in follow-up |

---

## DBSCAN Parameter Defaults

| Parameter | Default | Unit | Rationale |
|-----------|---------|------|-----------|
| `selectivity_threshold` | 0.3 | unitless | Points must fire on >=30% of touches to be RF candidates |
| `dbscan_eps` | 5.0 | mm | Conservative neighborhood for forearm-scale coordinates |
| `dbscan_min_samples` | 3 | count | Standard DBSCAN minimum for core points |
| `min_cluster_points` | 5 | count | Discard spurious micro-clusters |

All overridable via `SelectivityDBSCANConfig` dataclass and DAG options.

---

## References

- Superseded plan: `docs/development/plans/archived/receptive-field-filtering.md`
- Existing implementation: `code/scripts/_5_postprocessing/determine_receptive_field.py`
- Analysis module pattern: `code/src/analysis/touch_analytics/`
- Engine pattern: `code/src/postprocessing/xyz_reference_from_gestures/calibration_pca_engine.py`
- Discretization config: `code/src/analysis/touch_analytics/touch_config.py`
- Spatial transform utilities: `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py`
- Knowledge base: `note-cupy-import-order.md`, `note-forearm-icp-registration.md`, `note-somatosensory-units-and-calculations.md`
