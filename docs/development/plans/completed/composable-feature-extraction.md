# Plan: Composable Feature Extraction

**Created:** 2026-03-20
**Approved:** —
**Completed:** 2026-03-20 19:51
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/composable-feature-extraction`

---

## Overview

**What:** Refactor the analysis pipeline into a two-stage architecture: (1) independent feature extraction where each feature type produces its own output folder, and (2) downstream stages that define named feature combinations and run cross-products with their method profiles.

**Why:** Currently each extraction profile runs exactly one extractor and produces one CSV. There is no way to combine features from different extractors (e.g., kinematic stats + temporal) without creating redundant profiles. `MaxExtractor` is a strict subset of `StatisticalExtractor`. Downstream stages are locked to single-profile silos.

**How:** Make every feature type (max, mean, min, temporal, mechanics_of_solids, ...) an independently toggleable peer that extracts to its own folder. Downstream tasks (clustering, comparing, etc.) define named feature combinations that merge outputs from multiple feature folders, then run the cross-product of enabled combinations x enabled methods.

## Problem Statement

- `MaxExtractor` produces 4 columns that are a strict subset of `StatisticalExtractor` with `max` aggregation — pure redundancy.
- Each extraction profile can only run **one** extractor. To combine kinematic stats AND temporal features, users must create separate profiles producing separate CSVs that downstream stages cannot merge.
- Downstream tasks (clustering, comparing) operate per-profile in silos — there is no cross-feature-type merging.
- Heatmap and discretization code hardcodes `max_*` column names, preventing use of other aggregations.
- The term "extraction profile" conflates what features to extract with how to aggregate them, making the config unintuitive.

## Goals

### In Scope
1. Delete `MaxExtractor` — each kinematic aggregation (max, mean, min, etc.) becomes a first-class feature using `StatisticalExtractor` internally
2. Each feature type is independently toggleable and extracts to its own output folder
3. Downstream tasks define named `feature_combinations` that merge multiple feature folders
4. Downstream tasks run the cross-product of enabled feature combinations x enabled method profiles
5. Update all hardcoded `max_*` column references to `{signal}_{aggregation}` convention
6. Backward-compatibility: old `extraction_profiles` / `method:` configs translate to new format
7. GUI shows feature checkboxes for extraction and feature combination management for downstream tasks

### Out of Scope
- Adding new aggregation functions beyond the existing 7
- Modifying temporal or MoS extractor internals
- Changing clustering or comparing algorithm logic (they are already column-agnostic)
- Parameterizing heatmap column selection in the GUI (future work)
- Regenerating existing output CSVs (users re-run with `force_processing: true`)

## Success Criteria

- [ ] `MaxExtractor` is deleted; `features: {max: {enabled: true}}` produces `touch_features/max/` with columns `depth_max, area_max, velocity_max, acceleration_max`
- [ ] Multiple enabled features produce separate folders (e.g., `touch_features/max/`, `touch_features/mean/`)
- [ ] Clustering with `feature_combinations: {full: {features: [max, mean]}}` merges CSVs from both folders into one pooled DataFrame
- [ ] Cross-product works: 2 enabled combinations x 2 enabled clusterers = 4 output directories
- [ ] Old config format (`extraction_profiles: {max: {method: max}}`) still works via backward-compat translation
- [ ] `matrix_generation.py` and `touch_config.py` use `{signal}_{aggregation}` column names
- [ ] Full pipeline runs end-to-end: extraction -> clustering -> comparing without errors
- [ ] GUI shows per-feature checkboxes for extraction and feature combination editor for downstream tasks

---

## Technical Design

### Approach

Two-stage architecture with independent feature extraction and composable downstream consumption:

**Stage 1 — Feature Extraction:** Each feature type is a peer. Kinematic aggregations (`max`, `mean`, `min`, `median`, `std`, `range`, `skewness`) each run `StatisticalExtractor` with a single aggregation. Whole-touch feature sets (`temporal`, `mechanics_of_solids`) run their own extractor. Each enabled feature produces its own output folder with per-session CSVs.

**Stage 2 — Downstream Consumption:** Each downstream task (clustering, comparing, AP efficacy, receptive fields) defines its own `feature_combinations` — named groups that reference which feature folders to merge. The task runs the cross-product of enabled combinations x enabled method profiles.

### New Config Format

#### Feature Extraction
```yaml
touch_feature_extraction:
  enabled: true
  options:
    force_processing: true
    features:
      max:
        enabled: true
      mean:
        enabled: true
      min:
        enabled: false
      median:
        enabled: false
      std:
        enabled: false
      range:
        enabled: false
      skewness:
        enabled: false
      temporal:
        enabled: false
      mechanics_of_solids:
        enabled: false
        youngs_modulus_kpa: 100.0
        poissons_ratio: 0.45
        skin_thickness_mm: 1.5
  depends_on: []
```

Each feature entry has `enabled` plus optional parameters. Simple features (kinematic aggregations, temporal) have only `enabled`. Parameterized features (mechanics_of_solids) have `enabled` plus their configuration keys.

#### Downstream Tasks (Clustering Example)
```yaml
touch_clustering:
  enabled: true
  options:
    force_processing: true
    feature_combinations:
      basic:
        enabled: true
        features: [max]
      kinematic_full:
        enabled: true
        features: [max, mean, std]
      all_features:
        enabled: false
        features: [max, mean, temporal, mechanics_of_solids]
    clustering_profiles:
      binning:
        enabled: true
        method: binning
        n_bins: 20
        bin_method: equal_width
      kmeans:
        enabled: false
        method: kmeans
        min_touches_per_cluster: 30
      hierarchical:
        enabled: false
        method: hierarchical
        min_instances_per_sensor: 5
        min_sensor_types: 2
        sensor_col: session_id
  depends_on: [touch_feature_extraction]
```

Cross-product: every enabled `feature_combination` x every enabled `clustering_profile`. Same pattern for comparing, AP efficacy, and receptive fields.

#### Full Downstream Config (Comparing, AP Efficacy, Receptive Fields)
```yaml
touch_comparing:
  enabled: false
  options:
    force_processing: false
    feature_combinations:
      basic:
        enabled: true
        features: [max]
    clustering_profiles:
      hierarchical:
        method: hierarchical
    comparing_profiles:
      bias:
        method: bias
        measurement_col: spike_elicited
        sensor_col: session_id
        alpha: 0.05
      precision:
        method: precision
        measurement_col: spike_elicited
        sensor_col: session_id
      distribution:
        method: distribution
        measurement_col: spike_elicited
        sensor_col: session_id
    min_instances_per_sensor: 5
    min_sensor_types: 2
  depends_on: [touch_clustering]

analyse_ap_efficacy:
  enabled: false
  options:
    force_processing: false
    feature_combinations:
      basic:
        enabled: true
        features: [max]
  depends_on: [touch_feature_extraction]

map_receptive_fields:
  enabled: false
  options:
    grouping_columns: ['type_metadata', 'direction']
    monitor: false
    force_processing: false
    feature_combinations:
      basic:
        enabled: true
        features: [max]
  depends_on: [touch_feature_extraction]
```

### Output Structure
```
touch_features/                              # Stage 1 output
  max/
    <session>_touch_summary.csv              # shared cols + depth_max, area_max, velocity_max, acceleration_max
  mean/
    <session>_touch_summary.csv              # shared cols + depth_mean, area_mean, ...
  temporal/
    <session>_touch_summary.csv              # shared cols + duration_frames, auc_depth, ...
  mechanics_of_solids/
    <session>_touch_summary.csv              # shared cols + strain_max, stress_mean, ...

touch_clusters/                              # Stage 2 output (clustering)
  basic/                                     # feature combination name
    binning/                                 # clustering method name
      pooled_touch_summary_clustered.csv
      cluster_metadata.json
      heatmaps/
  kinematic_full/
    binning/
      pooled_touch_summary_clustered.csv
      ...
```

### Column Naming Convention Change

| Old (MaxExtractor) | New ({signal}_{aggregation}) |
|---|---|
| `max_depth` | `depth_max` |
| `max_contact_area` | `area_max` |
| `max_velocity` | `velocity_max` |
| `max_acceleration` | `acceleration_max` |

### Merge Logic

For a feature combination like `kinematic_full: [max, mean, std]`:
1. For each session, read CSVs from `touch_features/max/`, `touch_features/mean/`, `touch_features/std/`
2. Join on shared columns (touch identity: `block_order_id`, `trial_id`, `single_touch_id`, `session_id`)
3. Pool all sessions into one DataFrame
4. Pass to clustering/comparing

### Backward Compatibility

If config contains old `extraction_profiles` key instead of `features`:
- `{max: {method: max}}` -> `features: {max: {enabled: true}}`
- `{statistical: {method: statistical, aggregations: [mean, std]}}` -> `features: {mean: {enabled: true}, std: {enabled: true}}`
- `{temporal: {method: temporal}}` -> `features: {temporal: {enabled: true}}`
- `{mos: {method: mechanics_of_solids, ...params}}` -> `features: {mechanics_of_solids: {enabled: true, ...params}}`

If downstream tasks have old `extraction_profiles` key instead of `feature_combinations`:
- Each profile name becomes a combination: `{max: {method: max}}` -> `feature_combinations: {max: {features: [max]}}`

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Merge features at extraction time (one CSV per profile) | Simpler extraction code | No feature reuse; must re-extract if you want a different combination | Rejected (original plan) |
| Keep MaxExtractor alongside StatisticalExtractor | No refactoring | Redundant code, no composability | Rejected |
| Feature-per-folder + feature_combinations at downstream | Each feature extracted once, freely combinable, clean separation | Merge logic needed at downstream stage | **Chosen** |

### Architecture Changes

```
feature_extraction/
  __init__.py             -- delete MaxExtractor, add get_feature_extractor(), AGGREGATION_NAMES
  base.py                 -- unchanged
  max_extractor.py        -- DELETE
  statistical_extractor.py -- unchanged
  temporal_extractor.py    -- unchanged
  mos_extractor.py         -- unchanged
  kinematics.py            -- unchanged
```

Key new function in `__init__.py`:
- `get_feature_extractor(feature_name, feature_config) -> FeatureExtractor` — if name is a known aggregation, returns `StatisticalExtractor(aggregations=[name])`; else looks up in `EXTRACTOR_REGISTRY`

Key new function in `clustering_pipeline.py`:
- `_merge_feature_csvs(feature_names, extraction_dir) -> pd.DataFrame` — reads per-session CSVs from each feature folder, joins on shared columns, pools all sessions

### Knowledge Base

- `note-somatosensory-units-and-calculations.md` — documents depth (mm), area (mm^2), velocity (mm/frame). Relevant context for signal naming but no constraints on this refactor.
- No other notes applicable.

---

## Implementation Plan

### Phase 1: Registry Refactor and MaxExtractor Removal
**Goal:** Delete `MaxExtractor`, make each kinematic aggregation a first-class feature name.
**Started:** 2026-03-20
**Completed:** 2026-03-20

- [x] Delete `code/src/analysis/touch_analytics/feature_extraction/max_extractor.py`
- [x] Update `code/src/analysis/touch_analytics/feature_extraction/__init__.py`:
  - Remove `MaxExtractor` import and `'max'` registry entry
  - Define `AGGREGATION_NAMES = frozenset({'max', 'min', 'mean', 'median', 'std', 'range', 'skewness'})`
  - Add `get_feature_extractor(feature_name: str, feature_config: dict) -> FeatureExtractor`:
    - If `feature_name` in `AGGREGATION_NAMES`: return `StatisticalExtractor()` configured with `aggregations: [feature_name]`
    - Else: look up in `EXTRACTOR_REGISTRY`, instantiate with `feature_config`
    - Raise `KeyError` if unknown
  - Keep `get_extractor()` temporarily with deprecation warning for backward compat
  - Update `__all__`

**Files Modified:**
- `code/src/analysis/touch_analytics/feature_extraction/max_extractor.py` — DELETE
- `code/src/analysis/touch_analytics/feature_extraction/__init__.py` — registry refactor

**Dependencies:** None

### Phase 2: Extraction Pipeline — Feature-per-Folder
**Goal:** Rewrite extraction to iterate over enabled features, saving each to its own folder.
**Started:** 2026-03-20
**Completed:** 2026-03-20

- [x] Change `run_feature_extraction()` signature: `extraction_profiles: dict` -> `features: dict`
- [x] Filter enabled features via existing `filter_enabled_profiles()` (works as-is: `{name: {enabled: bool, ...}}`)
- [x] For each enabled feature: call `get_feature_extractor(name, config)`, run on all sessions, save to `output_dir / feature_name / csv_stem`
- [x] Update `_extract_session()`: iterate features instead of profiles (rename variables, same structure)
- [x] Add backward-compat translation: if options contain `extraction_profiles` instead of `features`, convert old format to new (see Backward Compatibility section)

**Files Modified:**
- `code/src/analysis/touch_analytics/extraction_pipeline.py` — feature-per-folder iteration

**Dependencies:** Phase 1

### Phase 3: Update Hardcoded Column References
**Goal:** Align downstream code with `{signal}_{aggregation}` naming convention.
**Started:** 2026-03-20
**Completed:** 2026-03-20

- [x] Update `touch_config.py` `DISCRETIZATION_CONFIG`: `max_depth` -> `depth_max`, `max_contact_area` -> `area_max`, `max_velocity` -> `velocity_max`, `max_acceleration` -> `acceleration_max`
- [x] Update `matrix_generation.py`: `hierarchy_order` list (~line 118), `heat_x`/`heat_y1`/`heat_y2` (~lines 189-191)
- [x] Update `touch_analysis.py`: output dict keys (~lines 103-106), plot references (~lines 138, 145)

**Files Modified:**
- `code/src/analysis/touch_analytics/touch_config.py` — column name updates
- `code/src/analysis/touch_analytics/matrix_generation.py` — column name updates
- `code/src/analysis/touch_analytics/touch_analysis.py` — column name updates

**Dependencies:** Phase 1

### Phase 4: Clustering Pipeline — Feature Combinations and Cross-Product
**Goal:** Rewrite clustering to accept `feature_combinations`, merge CSVs from multiple feature folders, run cross-product with clustering methods.
**Started:** 2026-03-20
**Completed:** 2026-03-20

- [x] Change `run_clustering()` signature: replace `extraction_profiles: dict` with `feature_combinations: dict`
- [x] Add `_merge_feature_csvs(feature_names: list[str], extraction_dir: Path) -> pd.DataFrame`:
  - Discover session CSVs in each feature folder via glob `*_touch_summary.csv`
  - For each session found across all feature folders, read and join on shared columns
  - Pool all sessions, return DataFrame with all feature columns merged
  - Log warning for missing feature folders or sessions with incomplete feature sets
- [x] Rewrite main loop: outer = enabled `feature_combinations`, inner = enabled `clustering_profiles` (cross-product)
- [x] Output path: `output_dir / combination_name / clusterer_name /`
- [x] Backward-compat: if `extraction_profiles` key found, convert each to `feature_combinations: {name: {features: [name]}}`

**Files Modified:**
- `code/src/analysis/touch_analytics/clustering_pipeline.py` — feature combination merge + cross-product

**Dependencies:** Phase 2

### Phase 5: Comparing Pipeline and Other Downstream Tasks
**Goal:** Apply same `feature_combinations` + cross-product pattern to comparing, AP efficacy, receptive fields.
**Started:** 2026-03-20
**Completed:** 2026-03-20

- [x] Update `comparing_pipeline.py`: replace `extraction_profiles` with `feature_combinations`, iterate `combinations x clustering_profiles x comparing_profiles`
- [x] Update `analysis_workflow.py` flow functions: pass `features` to extraction, `feature_combinations` to downstream tasks
- [ ] Update AP efficacy flow to accept and use `feature_combinations` (not applicable — AP efficacy uses matrix generation, not feature folders directly)
- [ ] Update receptive field mapping flow to accept and use `feature_combinations` (not applicable — RF mapping scans touch_features subfolders generically)

**Files Modified:**
- `code/src/analysis/touch_analytics/comparing_pipeline.py` — feature_combinations support
- `code/scripts/analysis_workflow.py` — orchestrator parameter mapping

**Dependencies:** Phase 4

### Phase 6: Config and Defaults Update
**Goal:** Switch DAG YAML and code defaults to new format.
**Started:** 2026-03-20
**Completed:** 2026-03-20

- [x] Rewrite `configs/analyse_workflow_dag.yaml` to new format with `features` dict and `feature_combinations` in each downstream task
- [x] Update `pipeline_shared.py`:
  - `DEFAULT_EXTRACTION_OPTIONS` -> `{'force_processing': False, 'features': {'max': {'enabled': True}}}`
  - Add `DEFAULT_FEATURE_COMBINATIONS = {'basic': {'enabled': True, 'features': ['max']}}`
  - Update `DEFAULT_CLUSTERING_OPTIONS` and `DEFAULT_COMPARING_OPTIONS` to use `feature_combinations`
- [x] Update `analysis_workflow.py` fallback defaults

**Files Modified:**
- `configs/analyse_workflow_dag.yaml` — new config structure
- `code/src/analysis/touch_analytics/pipeline_shared.py` — updated defaults
- `code/scripts/analysis_workflow.py` — fallback defaults

**Dependencies:** Phase 2, Phase 4, Phase 5

### Phase 7: GUI Update
**Goal:** Feature checkboxes for extraction, feature combination management for downstream tasks.
**Started:** 2026-03-20
**Completed:** 2026-03-20

- [x] Update `_is_profile_dict()` in `task_panel.py` to recognize `features` dict (no `method` key) and `feature_combinations` dict
- [x] Render flat feature checkboxes for extraction task — handled by existing profile rendering now that `_is_profile_dict` recognises the new format
- [x] Render `feature_combinations` in downstream tasks: enable/disable per named combination — handled by existing profile checkbox rendering
- [ ] Features with params (e.g., MoS `youngs_modulus_kpa`) — edit via YAML editor (complex cell, future dedicated UI)
- [ ] Click to edit the `features: [...]` list within a combination — edit via YAML editor (future dedicated UI)
- [x] `DagConfigModel` already supports new format generically via `get_profile_enabled`/`set_profile_enabled`

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_panel.py` — rendering logic for new config shape
- `code/src/utils/pipeline/dag_config_model.py` — model support (if needed)

**Dependencies:** Phase 6

---

## Testing Plan

### Manual Verification
- [ ] Run extraction with `features: {max: {enabled: true}}` — verify `touch_features/max/` contains CSVs with `depth_max, area_max, velocity_max, acceleration_max` columns
- [ ] Run extraction with `features: {max: {enabled: true}, mean: {enabled: true}}` — verify two separate folders with correct columns
- [ ] Run extraction with `features: {temporal: {enabled: true}}` — verify temporal-only output (no kinematic columns)
- [ ] Run extraction with `features: {mechanics_of_solids: {enabled: true, youngs_modulus_kpa: 100.0}}` — verify MoS-only output
- [ ] Run extraction with legacy `extraction_profiles: {max: {method: max}}` config — verify backward-compat translation
- [ ] Run clustering with `feature_combinations: {basic: {features: [max]}, full: {features: [max, mean]}}` and `clustering_profiles: {binning: {enabled: true}}` — verify 2 output directories (`basic/binning/`, `full/binning/`)
- [ ] Verify merged CSV in `full/binning/` has both `depth_max` and `depth_mean` columns
- [ ] Run full pipeline end-to-end: extraction -> clustering -> verify no errors
- [ ] Run AP efficacy matrix generation — verify uses updated column names

### GUI Verification
- [ ] Open DAG launcher with updated `analyse_workflow_dag.yaml` — extraction task shows per-feature checkboxes
- [ ] Toggle individual feature checkboxes — `features` dict in model updates, save persists to YAML
- [ ] MoS checkbox with params — edit button opens YAML dialog for MoS-specific configuration
- [ ] Clustering task shows feature_combinations — enable/disable per combination, edit features list
- [ ] Adding/removing feature combinations works from GUI

### Edge Cases
- [ ] Feature combination references a feature that wasn't extracted — log warning, skip that combination
- [ ] Only temporal features extracted, no kinematic — clustering still works (column-agnostic)
- [ ] Profile with `enabled: false` — skipped entirely
- [ ] Empty `features: {}` (all disabled) — log warning, produce no output
- [ ] Session has CSV in one feature folder but not another — inner join on shared columns, log warning

---

## Documentation Plan

- [ ] Update plan document with completion timestamps per phase
- [ ] Add inline comments in `__init__.py` documenting `get_feature_extractor()` contract
- [ ] Update DAG config comments to describe new `features` and `feature_combinations` format
- [ ] Update CLAUDE.md memory with new architecture (feature-per-folder, feature_combinations)

---

## Rollback Plan

1. Restore `max_extractor.py` from git
2. Revert `__init__.py` registry to re-add `MaxExtractor` and `'max'` entry
3. Revert `extraction_pipeline.py` to profile-based iteration
4. Revert `clustering_pipeline.py` to `extraction_profiles` based scanning
5. Revert column name changes in `touch_config.py`, `matrix_generation.py`, `touch_analysis.py`
6. Revert DAG config to old format
7. Revert GUI changes in `task_panel.py`

All changes are in code files with no data migration — rollback is a clean `git revert`.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Existing output CSVs have old column names | Certain | Low | Users re-run with `force_processing: true` |
| Feature combination merge produces duplicate columns | Very Low | Medium | Each feature produces disjoint column names by design; log warning if `merge()` detects duplicates |
| Session has CSV in one feature folder but not another | Low | Medium | `_merge_feature_csvs()` uses inner join on sessions present in ALL feature folders; logs warning for incomplete sessions |
| Legacy configs in user workflows use `method: max` | Medium | Low | Backward-compat translation in extraction and clustering pipelines |
| `touch_analysis.py` still used by other scripts | Low | Medium | Update column names in Phase 3; remains functional |
| GUI `_is_profile_dict` heuristic breaks for new format | Medium | Medium | Phase 7 explicitly replaces detection logic for new config shape |

---

## References

- Related files: `code/src/analysis/touch_analytics/feature_extraction/` (extractor package)
- Related files: `code/src/analysis/touch_analytics/extraction_pipeline.py` (orchestrator)
- Related files: `code/src/analysis/touch_analytics/clustering_pipeline.py` (clustering)
- Related files: `code/src/analysis/touch_analytics/comparing_pipeline.py` (comparing)
- Related files: `code/src/utils/gui/dag_launcher/task_panel.py` (GUI task options panel)
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
