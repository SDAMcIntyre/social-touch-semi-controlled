# Plan: Global Gesture Type Categorization in Preparation Stage

**Date:** 2026-04-29
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `dev`
**Branch:** `feature/gesture-type-preparation`

---

## Overview

**What:** Move gesture type categorization (tap / stroke_proximal / stroke_distal) from an opt-in clustering concern into a first-class column computed during touch preparation (Stage 1).

**Why:** The three-way gesture type only exists as a synthesized grouping key inside `TypeStratifiedClusterer._build_group_keys()`. The same direction-inference logic is duplicated in 4+ locations, and non-stratified clusterers are completely blind to gesture type. Making it a global column ensures every downstream stage inherits a single, authoritative categorization.

**How:** Create a new `preparation/gesture_type.py` module with the canonical classification function. Call it in `preparation_pipeline.py` after interpolation so that `gesture_type` appears in `_prepared.csv` and flows through all subsequent stages. Remove duplicate logic from extraction, clustering, and touch_analysis. Remove the `direction` column entirely (subsumed by `gesture_type`).

## Problem Statement

- The three-way gesture type (tap / stroke_proximal / stroke_distal) does not exist as a column anywhere in the pipeline. It is only synthesized at clustering time inside `_build_group_keys()` in `type_stratified_clusterer.py`.
- Direction inference (`infer_direction()`) is called in 3 separate locations: `extraction_pipeline.py` (twice -- statistical batch at line 453 and per-touch loop at line 496) and `touch_category.py` (line 46). A 4th duplicate exists inline in `touch_analysis.py:77-83`.
- `TypeStratifiedClusterer` is the only consumer of the three-way split, and it is opt-in via YAML config. All other clusterers and all visualization/reporting code must independently decide how to handle gesture types.
- The `direction` column (proximal/distal/static) is not computed until extraction (Stage 2b), too late for preparation-level validation or series-level use.

## Goals

### In Scope
1. New `gesture_type` column (`'tap'`, `'stroke_proximal'`, `'stroke_distal'`) computed once during preparation
2. Single source of truth -- one function (`classify_gesture_type`), one call site (preparation), one column
3. All downstream stages read the column instead of recomputing
4. Remove `direction` column from shared columns (fully subsumed by `gesture_type`)
5. Deprecate `representation/series_level/direction.py`

### Out of Scope
- Changing the proximal/distal classification rule (end_y > start_y)
- Adding new gesture types beyond the current three
- Removing `type_metadata` column (still useful as the two-way tap/stroke split)
- Deleting `TouchCategoryExtractor` (it still produces useful one-hot features)

## Success Criteria

- [ ] `gesture_type` column present in every `_prepared.csv` with valid values
- [ ] `gesture_type` flows through `_series_augmented.csv` -> per-feature CSVs -> clustering
- [ ] `TypeStratifiedClusterer` reads `gesture_type` directly (no more `_build_group_keys` synthesis)
- [ ] `infer_direction()` no longer called from extraction or clustering code
- [ ] `direction` column removed from `SHARED_COLUMNS`
- [ ] All existing tests pass (updated as needed)
- [ ] New unit tests for `classify_gesture_type()` and `assign_gesture_type()`
- [ ] Full pipeline runs end-to-end (preparation -> extraction -> clustering -> RF mapping)

---

## Technical Design

### Approach

Create `preparation/gesture_type.py` with two functions:
- `classify_gesture_type(group: pd.DataFrame) -> str` -- per-touch-group classification (single source of truth)
- `assign_gesture_type(df: pd.DataFrame) -> pd.DataFrame` -- DataFrame-level convenience that groups, classifies, and broadcasts

Call `assign_gesture_type()` in `_prepare_session()` after `interpolate_touch_columns()`, which guarantees forward-filled `type_metadata` and interpolated `sticker_blue_position_y`. The function follows the project's fail-fast convention: raises `ValueError` if `type_metadata` is missing, unrecognized, or if `sticker_blue_position_y` is absent for a stroke.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Compute in preparation (Stage 1) | Earliest possible; in `_prepared.csv`; available to all downstream stages | Adds per-touch iteration to a stage that currently only does DataFrame-level ops | **Chosen** |
| Compute in series transforms (Stage 2a) | Already iterates per touch group; natural fit | Only in `_series_augmented.csv`, not in `_prepared.csv` | Rejected |
| Keep in extraction (Stage 2b) | Minimal change | Status quo fragmentation; computed too late | Rejected |

### Architecture Changes

New module:
```
preparation/
  gesture_type.py    -- classify_gesture_type(), assign_gesture_type()  [NEW]
  __init__.py        -- add exports
  loader.py
  block_id.py
  interpolation.py
  grouping.py
```

Key files modified:
- `preparation_pipeline.py` -- call `assign_gesture_type()` after interpolation
- `pipeline_shared.py` -- add `gesture_type` to `SHARED_COLUMNS`, remove `direction`
- `extraction_pipeline.py` -- remove `infer_direction()` calls; read `gesture_type` from DataFrame
- `clustering/base.py` -- add `gesture_type_labels` to `ClusteringContext`
- `clustering_pipeline.py` -- populate `gesture_type_labels` from column
- `type_stratified_clusterer.py` -- read `gesture_type_labels` directly; simplify `_build_group_keys`
- `statistical.py` -- add `gesture_type` to `_EXCLUDE_FROM_AGGREGATION`, remove `direction`
- `touch_category.py` -- read `gesture_type` column instead of calling `infer_direction()`
- `touch_analysis.py` -- remove inline duplicate; import `classify_gesture_type` for compute-on-the-fly
- `reporting.py` -- update heatmap type iteration to three-way split
- `configs/analyse_workflow_dag.yaml` -- update clustering profile keys

### Knowledge Base Constraints

- **Single source of truth** (from `note-kinect-depth-access-single-path.md`): compute `gesture_type` in one place only; downstream must read the column, never recompute.
- **No silent type conversions** (from `note-somatosensory-units-and-calculations.md`): keep `gesture_type` as string throughout CSV round-trips.
- **Fail-fast convention** (from `CLAUDE.md`): raise on missing or invalid `type_metadata`, not silent fallback.

---

## Implementation Plan

### Phase 1: Foundation -- new module + preparation integration
**Goal:** `gesture_type` column exists in `_prepared.csv`
**Started:** 2026-04-29
**Completed:** 2026-04-29

- [x] Task 1.1 -- Create `preparation/gesture_type.py` with `classify_gesture_type()` and `assign_gesture_type()`
- [x] Task 1.2 -- Call `assign_gesture_type()` in `preparation_pipeline.py:_prepare_session()` after `interpolate_touch_columns()`
- [x] Task 1.3 -- Add `'gesture_type'` to `SHARED_COLUMNS` in `pipeline_shared.py`
- [x] Task 1.4 -- Add `'gesture_type'` to `_EXCLUDE_FROM_AGGREGATION` in `statistical.py`
- [x] Task 1.5 -- Write unit tests for `classify_gesture_type()` and `assign_gesture_type()`

**Files Modified:**
- `code/src/analysis/touch_analytics/preparation/gesture_type.py` -- **NEW**: classification logic
- `code/src/analysis/touch_analytics/preparation/__init__.py` -- add exports
- `code/src/analysis/touch_analytics/preparation_pipeline.py` -- call `assign_gesture_type()`
- `code/src/analysis/touch_analytics/pipeline_shared.py` -- update `SHARED_COLUMNS`
- `code/src/analysis/touch_analytics/representation/feature_characterization/statistical.py` -- update exclusion set

**Dependencies:** None

### Phase 2: Downstream consumers read `gesture_type`
**Goal:** Extraction and clustering use the pre-computed column; duplicate logic removed
**Started:** 2026-04-29
**Completed:** 2026-04-29

- [x] Task 2.1 -- Update `extraction_pipeline.py`: remove `infer_direction()` calls (lines 453 and 496), read `gesture_type` from DataFrame via `meta_spec['gesture_type'] = 'first'`, remove `direction` from output
- [x] Task 2.2 -- Add `gesture_type_labels` field to `ClusteringContext` in `clustering/base.py`
- [x] Task 2.3 -- Update `clustering_pipeline.py`: populate `gesture_type_labels` from pooled `gesture_type` column (lines 587-595)
- [x] Task 2.4 -- Rewrite `type_stratified_clusterer.py`: read `context.gesture_type_labels` directly; simplify `_build_group_keys()` to validation-only (no more combining `type_metadata` + `direction`)
- [x] Task 2.5 -- Update `TouchCategoryExtractor` in `touch_category.py` to read `gesture_type` column instead of calling `infer_direction()`
- [x] Task 2.6 -- Update `touch_analysis.py`: remove inline direction logic (lines 77-83), import `classify_gesture_type` for compute-on-the-fly (raw CSVs lack `gesture_type`)
- [x] Task 2.7 -- Update `test_type_stratified_clusterer.py` for new `gesture_type_labels` interface

**Files Modified:**
- `code/src/analysis/touch_analytics/extraction_pipeline.py` -- remove `infer_direction` import + calls; read `gesture_type`
- `code/src/analysis/touch_analytics/clustering/base.py` -- add `gesture_type_labels` field
- `code/src/analysis/touch_analytics/clustering_pipeline.py` -- populate new context field
- `code/src/analysis/touch_analytics/clustering/type_stratified_clusterer.py` -- simplify to read `gesture_type_labels`
- `code/src/analysis/touch_analytics/representation/feature_characterization/touch_category.py` -- read `gesture_type` instead of calling `infer_direction`
- `code/src/analysis/touch_analytics/touch_analysis.py` -- remove inline direction logic; import `classify_gesture_type`
- `code/src/analysis/touch_analytics/clustering/test_type_stratified_clusterer.py` -- update tests

**Dependencies:** Phase 1

### Phase 3: Cleanup -- remove `direction` column, update visualization, deprecate old module
**Goal:** `direction` column fully removed (subsumed by `gesture_type`); heatmaps use three-way split
**Started:** 2026-04-29
**Completed:** 2026-04-29

- [x] Task 3.1 -- Remove `'direction'` from `SHARED_COLUMNS`, `_EXCLUDE_FROM_AGGREGATION`, and all config references
- [x] Task 3.2 -- Remove `type_labels` and `direction_labels` from `ClusteringContext` (both subsumed by `gesture_type_labels`)
- [x] Task 3.3 -- Update `reporting.py` heatmap iteration: change from `['tap', 'stroke']` to three-way `gesture_type` split
- [x] Task 3.4 -- Update `clustering_pipeline.py` heatmap logic (lines 810-820) to iterate gesture types instead of `type_metadata`
- [x] Task 3.5 -- Update `rf_cluster_pipeline.py` to use `gesture_type` in cluster descriptions
- [x] Task 3.6 -- Deprecate `representation/series_level/direction.py` (add deprecation docstring, remove all imports)
- [x] Task 3.7 -- Update `configs/analyse_workflow_dag.yaml`: replace `type_col: type_metadata` with `gesture_type_col: gesture_type` in clustering profiles
- [x] Task 3.8 -- Update `code/src/analysis/CLAUDE.md` if stage descriptions reference `direction`

**Files Modified:**
- `code/src/analysis/touch_analytics/pipeline_shared.py` -- remove `direction`
- `code/src/analysis/touch_analytics/clustering/base.py` -- remove `type_labels`, `direction_labels`
- `code/src/analysis/touch_analytics/reporting.py` -- three-way heatmap split
- `code/src/analysis/touch_analytics/clustering_pipeline.py` -- heatmap and context cleanup
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` -- cluster description update
- `code/src/analysis/touch_analytics/representation/series_level/direction.py` -- deprecation notice
- `configs/analyse_workflow_dag.yaml` -- config key update

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] `classify_gesture_type()` returns `'tap'` for tap groups
- [ ] `classify_gesture_type()` returns `'stroke_proximal'` when end_y > start_y
- [ ] `classify_gesture_type()` returns `'stroke_distal'` when end_y <= start_y
- [ ] `classify_gesture_type()` raises `ValueError` on missing `type_metadata` column
- [ ] `classify_gesture_type()` raises `ValueError` on unknown `type_metadata` value (not tap/stroke)
- [ ] `classify_gesture_type()` raises `ValueError` on stroke with missing `sticker_blue_position_y`
- [ ] `classify_gesture_type()` returns `'stroke_distal'` for single-frame stroke (end_y == start_y)
- [ ] `assign_gesture_type()` adds column to DataFrame with correct values per group
- [ ] `assign_gesture_type()` skips `single_touch_id == 0` groups
- [ ] `TypeStratifiedClusterer` reads `gesture_type_labels` and produces correct three-way split
- [ ] `TypeStratifiedClusterer` raises on invalid `gesture_type` values

### Integration Tests
- [ ] Preparation pipeline produces `_prepared.csv` with `gesture_type` column
- [ ] Feature extraction preserves `gesture_type` in per-feature CSVs
- [ ] Clustering reads `gesture_type` from pooled DataFrame and stratifies correctly

### Manual Verification
- [ ] Run full pipeline: preparation -> series -> extraction -> clustering -> RF mapping
- [ ] Verify `_prepared.csv` contains `gesture_type` column with valid values
- [ ] Verify heatmaps render with three-way gesture type split
- [ ] Verify RF cluster descriptions include `gesture_type` distribution

### Edge Cases
- [ ] Session with only taps (no strokes) -- `stroke_proximal`/`stroke_distal` groups empty
- [ ] Session with only strokes (no taps) -- `tap` group empty
- [ ] Single-frame stroke touch (start_y == end_y -> stroke_distal)
- [ ] Old `_prepared.csv` without `gesture_type` column -- pipeline should raise clearly

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` with gesture_type column in stage descriptions
- [ ] Update root `CLAUDE.md` if the architecture overview references direction inference

---

## Rollback Plan

1. Revert the feature branch merge commit (`git revert --no-ff`)
2. No database or state changes -- pipeline outputs are regenerated from source CSVs
3. Old `_prepared.csv` files without `gesture_type` still work with the pre-refactor code

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Old `_prepared.csv` files lack `gesture_type` | High | Med | Fail-fast with clear message: "re-run preparation with force_processing: true" |
| Heatmap layout changes (2-col to 3-col) | Med | Low | Visual change only; verify manually |
| `touch_analysis.py` reads raw CSV (no `gesture_type`) | Med | Med | Import `classify_gesture_type` for compute-on-the-fly |
| Downstream code still references `direction` column | Low | Low | Grep for all `'direction'` references and update in Phase 3 |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Foundation | Small (~100 lines new + tests) | None |
| Phase 2: Downstream | Medium (~200 lines changed across 7 files) | Phase 1 |
| Phase 3: Cleanup | Medium (~150 lines changed across 7 files) | Phase 2 |

---

## References

- Completed plan: `docs/development/plans/completed/type-stratified-clustering.md`
- Knowledge base: `docs/development/knowledge-base/note-kinect-depth-access-single-path.md` (single-source-of-truth pattern)
- Current direction module: `code/src/analysis/touch_analytics/representation/series_level/direction.py`
- Current three-way split: `code/src/analysis/touch_analytics/clustering/type_stratified_clusterer.py:104-123`
