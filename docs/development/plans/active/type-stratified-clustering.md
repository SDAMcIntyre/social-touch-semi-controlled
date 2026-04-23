# Plan: Type-Stratified Clustering for Touch Data

**Created:** 2026-04-23 14:30
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/type-stratified-clustering`

---

## Overview

Implement type-stratified clustering (split-by-type, cluster independently, merge results) to ensure RF clusters are type-homogeneous: each cluster contains only one touch type (tap, stroke-proximal, or stroke-distal). This solves the mixed-type cluster problem by treating type as a **grouping constraint** rather than a clustering feature. It is introduced as a new `type_stratified` clustering profile that runs **alongside** the existing `binning` profile — outputs land in parallel `<combination>/type_stratified/` folders, leaving `binning/` outputs untouched.

## Problem Statement

When clustering touches with both pressure/velocity and type information, the BinningClusterer treats type columns (is_tap, is_stroke, dir_proximal, dir_distal) as numeric features to be binned alongside pressure/velocity. Since binary columns have low variance, type doesn't control cluster assignment — clustering is driven purely by pressure/velocity ranges, resulting in clusters spanning multiple types. This violates the requirement that each cluster must be type-homogeneous.

**Impact:** RF clusters are inconsistent (mixed type/direction populations), making it difficult to interpret whether receptive field properties are driven by touch mechanics, type, or direction.

## Goals

### In Scope
1. **Create TypeStratifiedClusterer** — Wrapper clusterer that splits by type, delegates to a base clusterer per group, and merges results with type-prefixed string labels
2. **Pipeline injection of type/direction labels** — Add a `type_col` → `_type_labels`/`_direction_labels` injection block in `clustering_pipeline._cluster_combination`, mirroring the existing `sensor_col` pattern
3. **Register in factory** — Add `"type_stratified"` to `CLUSTERER_REGISTRY`
4. **Enable as a global profile** — Add a `type_stratified` profile to both `touch_clustering` and `map_receptive_fields_clustered` sections of the DAG, running alongside `binning` for every enabled combination
5. **Preserve cluster-description richness** — Update `rf_cluster_pipeline._build_cluster_description` so `bin_range` still resolves for stratified string labels via nested `per_type` metadata
6. **Verify type homogeneity** — Confirm output clusters contain only single types per cluster label

### Out of Scope
- Changing other clustering methods (BinningClusterer, KMeans, etc. remain unchanged — used as delegates)
- Modifying touch_category_extractor or touch_analytics extraction
- Adding per-combination clusterer overrides to the DAG schema (kept for a follow-up if the universal-wrapper approach proves too noisy)
- Creating separate RF visualizations per type (type info preserved in existing cluster descriptions)
- Handling edge cases where a type is missing from data (gracefully skip with warnings)

## Success Criteria

- [ ] TypeStratifiedClusterer class created and passes unit tests
- [ ] Output CSV in every `<combination>/type_stratified/` folder contains cluster labels prefixed by type (e.g. `tap_00`, `stroke_distal_02`)
- [ ] Each cluster label contains touches of only one type (spot-checked on 5+ clusters across ≥ 2 combinations)
- [ ] Cluster descriptions show 100% of one type per cluster (`type_distribution` → single type)
- [ ] Cluster descriptions still contain a `bin_range` field for stratified labels (Phase 4 regression guard)
- [ ] RF cluster pipeline processes stratified output without errors
- [ ] No regressions in existing `<combination>/binning/` outputs — files are byte-identical to a pre-change baseline for at least one session (the change is additive; binning profile is untouched)

---

## Technical Design

### Approach

**Type-Stratified Clustering Algorithm:**

1. **Split by type:** Group touches into three buckets: `{tap, stroke_proximal, stroke_distal}`
   - Taps ignore direction (direction is meaningless for a tap)
   - Strokes with no resolved direction (`direction` missing / null) fall back to `stroke_distal`, matching `touch_category_extractor`'s `end_y <= start_y → distal` rule
   - Each group clusters independently

   **Why 3-way and not 2-way?** Proximal vs. distal strokes are kinematically distinct (opposite afferent recruitment patterns); mixing them inside a stroke cluster recreates the same interpretability problem we're solving for tap/stroke. The cost is up to 2× stroke clusters, which is acceptable — downstream tools key on the string label, not cluster count.

2. **Cluster each type-group:** For each non-empty group, instantiate the base clusterer (via `get_clusterer(base_method)`) and delegate `fit_predict` on that group's feature slice.

3. **Merge with type prefix:** Combine results into a single `labels` array of strings: `f"{type}_{cluster_idx:02d}"` (e.g. `tap_00`, `stroke_proximal_00`, `stroke_distal_05`).

4. **Handle empty / under-sized groups:** If a type has zero touches, log a warning and skip. If a group has too few touches for the base clusterer's own minimum, let the base clusterer decide (do not second-guess it here).

5. **Metadata:** Emit a nested per-type metadata dict (see "Metadata schema" below), not a flat one — each type-group has its own `primary_feature` / `bin_edges`.

**Why this approach:**
- Type homogeneity guaranteed by construction (no post-processing needed)
- Transparent: cluster labels show type in name
- Preserves existing BinningClusterer logic; no complex modifications
- Supports all underlying clusterers (BinningClusterer, KMeans, DBSCAN, Hierarchical)

### How type labels reach the clusterer (critical)

`type_metadata` and `direction` are in `SHARED_COLUMNS` (`pipeline_shared.py:14-20`) and are therefore **filtered out** of `feature_df` before `fit_predict` is called (`clustering_pipeline.py:327-331`). The stratifier cannot read type from `feature_df`.

The pipeline already has a precedent for injecting non-feature columns into a clusterer: `HierarchicalClusterer` receives sensor labels via the `_sensor_labels` key on the config dict (`clustering_pipeline.py:346-348`). We follow the same pattern:

```python
# clustering_pipeline.py — new block, next to the existing sensor_col injection
type_col = clusterer_config.get('type_col')
if type_col and type_col in pooled.columns:
    clusterer_config = {
        **clusterer_config,
        '_type_labels': pooled.loc[valid_idx, type_col].values,
        '_direction_labels': (
            pooled.loc[valid_idx, 'direction'].values
            if 'direction' in pooled.columns else None
        ),
    }
```

TypeStratifiedClusterer reads `_type_labels` (+ optional `_direction_labels`) from the config and builds the 3-way grouping internally.

### Metadata schema

Each per-type base-clusterer call returns its own metadata. We preserve them nested rather than flattening, so downstream tools (e.g. `rf_cluster_pipeline._build_cluster_description`) can look up the right `bin_edges`/`primary_feature` from the string cluster label:

```json
{
  "algorithm": "type_stratified",
  "base_algorithm": "binning",
  "params": { ... profile config ... },
  "per_type": {
    "tap":              { "primary_feature": "...", "bin_edges": {...}, "n_clusters": N, ... },
    "stroke_proximal":  { ... },
    "stroke_distal":    { ... }
  },
  "extra_columns": { "bin_<col>": np.ndarray (concatenated, per-row) }
}
```

`extra_columns` is concatenated in the row-index order of the returned `labels` so `clustering_pipeline._cluster_combination` can assign bin columns without change.

### Scope restriction: which combinations get stratified?

The DAG runs **every enabled feature_combination × every enabled clustering_profile** — there is no per-combination clusterer override today. Three ways to handle this; we choose (a):

- **(a) CHOSEN — stratifier is a universal wrapper:** `type_stratified` is enabled globally. When run against a combination whose pooled DataFrame still carries `type_metadata` (it always does — shared column), stratification applies uniformly. For combinations like `only_mean` this produces `tap_00` / `stroke_proximal_00` / `stroke_distal_00` labels alongside the existing flat `0..19` binning output (different output folder: `only_mean/type_stratified/` vs `only_mean/binning/`). Parallel outputs, no regression.
- **(b)** Add per-combination `clustering_profiles:` override to the DAG schema — schema change, out of scope for this plan.
- **(c)** Only enable type_stratified profile, disable binning for the one combination — can't express in current YAML schema.

Consequence of (a): the user reads type-stratified results for every enabled combination, not just `pressure_velocity_mean_and_type`. This is a feature, not a bug — it lets you compare type-stratified vs. type-blind clusters on the same features. If undesirable later, mitigate by disabling unwanted combinations or adding per-combination overrides in a follow-up plan.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **A: Type-stratified clustering** | Type-homogeneous by construction; clean separation; transparent labeling | May create many clusters if types imbalanced | ✅ **Chosen** |
| B: Constraint-based clustering | Single pass; architecturally elegant | Complex to implement; bin edges may not align with type boundaries | Rejected |
| C: Post-process clustering | Minimal code changes | Fragmented clusters; less meaningful statistics | Rejected |
| D: Type-aware features | Semantically clear | Requires feature engineering changes; complex | Rejected |

### Architecture Changes

**New file:**
```
code/src/analysis/touch_analytics/clustering/type_stratified_clusterer.py
```

TypeStratifiedClusterer:
- Implements `fit_predict(feature_df, config) -> (labels, metadata)` — same signature as every other `TouchClusterer`.
- Reads the base clusterer name from `config['base_method']` and delegates per-type via `get_clusterer(base_method)`.
- Reads `_type_labels` and `_direction_labels` injected by `clustering_pipeline` (see "How type labels reach the clusterer").
- Returns `labels: np.ndarray` of `object` dtype (strings like `"tap_00"`), and nested metadata (see "Metadata schema").

**Modified files:**
```
code/src/analysis/touch_analytics/clustering/__init__.py
  — Add TypeStratifiedClusterer to CLUSTERER_REGISTRY under key "type_stratified"

code/src/analysis/touch_analytics/clustering_pipeline.py
  — Add type_col / _type_labels / _direction_labels injection block next to
    the existing sensor_col block (~line 346-348)

configs/analyse_workflow_dag.yaml
  — Add "type_stratified" profile to touch_clustering.clustering_profiles
  — Add same profile to map_receptive_fields_clustered.clustering_profiles
```

**YAML profile schema:**
```yaml
type_stratified:
  method: type_stratified
  base_method: binning       # delegate clusterer; any method in CLUSTERER_REGISTRY
  type_col: type_metadata    # column in pooled df used for tap/stroke split
  # Base-clusterer params are read from the same dict:
  n_bins: 20
  bin_method: equal_width
```

**Files that need light adjustment (not full rewrite):**
- `rf_cluster_pipeline.py:263-275` — the `bin_range` lookup does `int(cluster_label)` against flat `bin_edges`. With stratified string labels (`tap_00`) it silently falls through and `bin_range` is omitted from cluster descriptions. Fix: parse `{type}_{idx}` labels and look up `bin_edges` from the nested `per_type` metadata. Small, local change.

**No changes needed:**
- BinningClusterer, KMeans, DBSCAN, HierarchicalClusterer (used as delegates, unchanged)
- `_format_cluster_folder` in `rf_cluster_pipeline.py:214-219` — already falls through to `f'cluster_{cluster_label}'` for non-integer labels, producing e.g. `cluster_tap_00/`

---

## Implementation Plan

### Phase 1: Pipeline type-label injection
**Goal:** Make type/direction labels reachable from a clusterer without breaking existing clusterers
**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Add `type_col` injection block to `clustering_pipeline._cluster_combination`
  - Insert next to existing `sensor_col` block (~line 346-348)
  - Read `type_col` from `clusterer_config`; inject `_type_labels` + `_direction_labels` into config
  - Guard: only inject if `type_col` is set and present in `pooled.columns`
  - Verify existing clusterers ignore unknown `_type_labels` key (they take `config: dict` — unused keys are harmless)

**Files Modified:**
- `code/src/analysis/touch_analytics/clustering_pipeline.py` — add injection block

**Dependencies:** None

### Phase 2: TypeStratifiedClusterer implementation
**Goal:** Implement the stratification logic and register it in the factory
**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Create `type_stratified_clusterer.py` with TypeStratifiedClusterer class
  - Read `_type_labels` / `_direction_labels` from `config` (error clearly if missing)
  - Build 3-way group key per row: `tap` / `stroke_proximal` / `stroke_distal`
    - Strokes with null/missing direction → `stroke_distal` (matches `touch_category_extractor` fallback)
  - For each non-empty group: instantiate base clusterer via `get_clusterer(config['base_method'])`, slice `feature_df` by group-row-mask, call `fit_predict(sliced_df, config)`
  - Build string labels `f"{group}_{idx:02d}"`, place back at original row positions
  - Concatenate per-type `extra_columns['bin_<col>']` arrays in original row order
  - Return `labels` (object dtype) + nested `per_type` metadata

- [x] Write unit tests: `code/src/analysis/touch_analytics/clustering/test_type_stratified_clusterer.py`
  - Test single type present (tap only) → only `tap_XX` labels
  - Test multiple types (tap + stroke with both directions) → three-way split
  - Test direction fallback: stroke with null direction → `stroke_distal`
  - Test label format regex: `^(tap|stroke_proximal|stroke_distal)_\d{2}$`
  - Test missing `_type_labels` in config → raises clear error
  - Test empty group handling (no taps in data) → warning logged, no `tap_*` labels in output
  - Test row-order preservation: output `labels[i]` corresponds to `feature_df.iloc[i]`

- [x] Register in clustering factory: `clustering/__init__.py`
  - Import `TypeStratifiedClusterer`
  - Add `'type_stratified': TypeStratifiedClusterer` to `CLUSTERER_REGISTRY`

**Files Modified:**
- `code/src/analysis/touch_analytics/clustering/type_stratified_clusterer.py` — **New**
- `code/src/analysis/touch_analytics/clustering/test_type_stratified_clusterer.py` — **New**
- `code/src/analysis/touch_analytics/clustering/__init__.py` — Add factory registration

**Dependencies:** Phase 1

### Phase 3: DAG Configuration Integration
**Goal:** Enable type-stratified clustering as a parallel profile alongside existing ones
**Started:** 2026-04-23
**Completed:** 2026-04-23

Important: the current DAG schema runs every enabled combination × every enabled clusterer. Enabling `type_stratified` globally means all combinations get a stratified output folder in addition to their binning output — see "Scope restriction" in Technical Design. We accept this in scope (a).

- [x] Add `type_stratified` profile to `touch_clustering.clustering_profiles`
  ```yaml
  type_stratified:
    method: type_stratified
    base_method: binning
    type_col: type_metadata
    n_bins: 20
    bin_method: equal_width
  ```

- [x] Add the same profile to `map_receptive_fields_clustered.clustering_profiles` so RF mapping consumes the stratified output

- [x] Leave all `feature_combinations` as-is (per decision above — no per-combination override in this plan)

- [ ] Test configuration parsing: ensure `DagConfigModel` round-trips the YAML without errors

**Files Modified:**
- `configs/analyse_workflow_dag.yaml` — Add type_stratified profile to two sections

**Dependencies:** Phase 2

### Phase 4: Downstream `bin_range` compatibility
**Goal:** Preserve cluster-description richness when labels are stratified strings
**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Update `rf_cluster_pipeline._build_cluster_description` (lines 263-275)
  - Parse string labels of the form `{type}_{idx:02d}` (regex or split on last `_`)
  - When `algorithm == 'type_stratified'` in metadata, look up `bin_edges` from `metadata['per_type'][type_key]` instead of flat top-level
  - Preserve existing integer-label behavior for non-stratified clusterers (backward-compatible branch)

- [x] Add test: stratified cluster description contains `bin_range` for a `tap_03` label

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — `_build_cluster_description` only

**Dependencies:** Phase 2

### Phase 5: Verification & Testing
**Goal:** Verify type homogeneity and RF pipeline compatibility
**Started:** —
**Completed:** —

- [ ] Run single test session through clustering pipeline with type_stratified enabled
  - Inspect output `pooled_touch_summary_clustered.csv`
  - Verify cluster labels follow `{type}_{idx:02d}` format
  - Verify each cluster contains only one type

- [ ] Run RF cluster mapping on stratified output
  - Confirm no errors in rf_cluster_pipeline.py
  - Inspect cluster descriptions: verify type_distribution shows 100% of one type

- [ ] Spot-check 5+ clusters for type consistency
  - Load clustered CSV, group by cluster_label
  - Print unique type_metadata values per cluster
  - Verify exactly one type per cluster

- [ ] Run full pipeline on 2-3 sessions for regression testing
  - Only_mean combination should be identical to baseline
  - pressure_velocity_mean, pressure_velocity_max unchanged
  - pressure_velocity_mean_and_type should now be type-stratified

**Files Modified:** None (testing only)

**Dependencies:** Phase 4

---

## Testing Plan

### Unit Tests
- [ ] Test type extraction: verify correct grouping by `type_metadata` + direction
- [ ] Test single-type clustering: DataFrame with only taps → labels `tap_00`, `tap_01`, etc.
- [ ] Test multi-type clustering: mixed taps/strokes → `tap_XX`, `stroke_proximal_XX`, `stroke_distal_XX`
- [ ] Test direction inference: strokes with direction → separate proximal/distal groups
- [ ] Test metadata preservation: input feature columns preserved in output
- [ ] Test empty type groups: missing type → warning logged, skipped gracefully
- [ ] Test cluster label format: all labels match `{type}_{index:02d}` regex
- [ ] Test zero-padded indices: up to 99 clusters per type handled correctly

### Integration Tests
- [ ] Touch clustering pipeline processes stratified output without errors for every enabled combination
- [ ] RF cluster mapping reads stratified cluster CSV correctly
- [ ] Cluster descriptions show single type per cluster (100% in type_distribution)
- [ ] Cluster descriptions for stratified labels still contain `bin_range` (Phase 4 check)
- [ ] No regression in `<combination>/binning/` outputs — byte-compare one session vs. baseline
- [ ] kmeans + hierarchical profiles still run correctly when the `_type_labels` injection block is present but the profile doesn't use it

### Manual Verification
- [ ] Run full analysis pipeline on one session with type_stratified enabled
- [ ] Visual inspection: cluster folder names show type prefixes (e.g., `cluster_tap_00/`, `cluster_stroke_proximal_01/`)
- [ ] Spot-check RF metrics per cluster: confirm spike distributions are reasonable
- [ ] Compare RF metrics before/after: pressure-based clusters should have similar structure, just type-split

### Edge Cases
- [ ] All touches are same type (only taps) → still creates clusters correctly
- [ ] Single touch per type → creates cluster_0 for each type
- [ ] Very imbalanced types (99% taps, 1% strokes) → handles both without error
- [ ] No type_metadata column → falls back gracefully (or errors with clear message)
- [ ] Directory with no touches of a type → skips type silently

---

## Documentation Plan

- [ ] Add docstring to `TypeStratifiedClusterer` class explaining algorithm
- [ ] Document cluster label format in clustering pipeline docs
- [ ] Update `docs/development/knowledge-base/` if new patterns emerge
- [ ] Add comment in `analyse_workflow_dag.yaml` explaining type_stratified purpose
- [ ] Update memory: mark analysis_pipeline_issue_mixed_types_in_clusters.md as resolved

---

## Rollback Plan

The change is **additive** in YAML: `type_stratified` sits alongside `binning` without replacing it. Rollback is therefore cheap.

1. **Disable at the config level (no code revert needed):**
   - Set `enabled: false` on the `type_stratified` profile in both `touch_clustering` and `map_receptive_fields_clustered` sections of `analyse_workflow_dag.yaml`
   - Pipeline resumes producing only `binning/` outputs

2. **If a deeper problem surfaces (e.g. the injection block breaks other clusterers):**
   - `git revert` the pipeline-injection commit (Phase 1)
   - TypeStratifiedClusterer code stays on the feature branch; registry entry becomes dormant

3. **If stratified outputs produced during testing need to be cleared:**
   - Delete `<output_dir>/*/type_stratified/` folders (per-combination subdirs)
   - `binning/` outputs are untouched

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `_type_labels` missing (pipeline injection skipped or misconfigured `type_col`) | Med | High | Clusterer errors with an explicit message referencing the YAML key; Phase 1 test covers the injection path end-to-end |
| Cluster label collision (same idx for tap and stroke) | Low | High | Type prefix in label by construction; regex assertion in unit tests |
| `bin_range` disappears from cluster descriptions for stratified labels | High (default behavior) | Low | Addressed directly in Phase 4 |
| Severe type imbalance creates many single-cluster types | Low | Med | Accept — clustering still valid; metadata makes imbalance visible |
| Performance regression (N sub-clusterings instead of 1) | Low | Low | Total work is unchanged (sum of rows = N); overhead is N Python calls, negligible |
| Stratification applied to combinations where it's not meaningful | Med | Low | Intentional per "Scope restriction (a)"; each combination gets both `binning/` and `type_stratified/` folders — user picks at read time |
| Other clusterers (kmeans/dbscan/hierarchical) regress from the pipeline injection block | Low | High | Injected keys start with `_` and are ignored by clusterers that don't read them; smoke-test kmeans/hierarchical on one session in Phase 5 |
| Existing `binning/` outputs change unexpectedly | Low | High | Phase 5 byte-compares one session's binning output vs. pre-change baseline |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Pipeline type-label injection | 30 min | None |
| Phase 2: TypeStratifiedClusterer + tests | 2–3 h | Phase 1 |
| Phase 3: DAG integration | 30 min | Phase 2 |
| Phase 4: Downstream `bin_range` fix | 45 min | Phase 2 |
| Phase 5: Verification | 1 h | Phase 4 |
| **Total** | **~5 h** | — |

---

## References

- **Analysis pipeline issue:** `memory/analysis_pipeline_issue_mixed_types_in_clusters.md`
- **Clustering pipeline:** `code/src/analysis/touch_analytics/clustering_pipeline.py`
- **BinningClusterer:** `code/src/analysis/touch_analytics/clustering/binning_clusterer.py`
- **RF cluster pipeline:** `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py`
- **DAG config:** `configs/analyse_workflow_dag.yaml`

---
