# Plan: Touch Category Features

**Date:** 2026-04-23
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/geometric-pressure-proxy`

---

## Overview

Add a `TouchCategoryExtractor` that encodes touch type (`tap` / `stroke`) and stroke direction (`proximal` / `distal`) as one-hot binary columns. These columns become first-class clustering dimensions, allowing the BinningClusterer and other clusterers to separate cluster axes by stimulus modality and direction — complementing continuous features like `geo_pressure_mean`.

## Problem Statement

The existing feature set captures kinematics (depth, area, velocity, pressure proxy) but treats touch type and stroke direction as metadata-only fields (`type_metadata`, `direction` in shared columns). They are displayed in cluster descriptions but never enter the clustering feature space. This means clusters that are kinematically similar but mechanistically distinct (e.g., a slow proximal stroke vs a tap at the same depth) are merged. Adding binary encodings lets the clusterer — and the RF mapping downstream — distinguish these categories without any pipeline changes.

## Goals

### In Scope
1. `TouchCategoryExtractor` class implementing `FeatureExtractor`, outputting four binary columns
2. Registry entry `touch_category` in `EXTRACTOR_REGISTRY`
3. YAML config entry under `touch_feature_extraction.options.features`
4. Feature combinations for clustering that include `touch_category` alone and paired with `pressure_velocity_mean`

### Out of Scope
- Multi-class ordinal encoding (binary one-hot only in this phase)
- Encoding additional categorical fields (e.g., `session_id`, `block_order_id`)
- Changes to the clustering, comparing, or RF mapping pipelines
- GUI changes beyond what the existing feature toggle grid already provides

## Success Criteria

- [ ] `touch_category` appears as a toggleable feature in the DAG launcher GUI feature grid
- [ ] Extraction produces a CSV with columns `is_tap`, `is_stroke`, `dir_proximal`, `dir_distal` (0/1 integers)
- [ ] Tap rows have `is_tap=1`, `is_stroke=0`, `dir_proximal=0`, `dir_distal=0`
- [ ] Proximal-stroke rows have `is_tap=0`, `is_stroke=1`, `dir_proximal=1`, `dir_distal=0`
- [ ] Distal-stroke rows have `is_tap=0`, `is_stroke=1`, `dir_proximal=0`, `dir_distal=1`
- [ ] Clustering pipeline accepts `touch_category` in a feature combination without modification
- [ ] RF mapping pipeline accepts combinations that include `touch_category` without modification
- [ ] Missing `type_metadata` column produces all-zero row without error

---

## Technical Design

### Approach

`TouchCategoryExtractor` reads `type_metadata` from `group.iloc[0]` and replicates the direction-inference logic from `extraction_pipeline._extract_all_touches` (lines 296–299). It returns four integer (0/1) columns. Because the clustering pipeline (`clustering_pipeline._cluster_combination`, line 327) selects all numeric non-shared columns as feature dimensions, these columns are consumed automatically with no pipeline changes.

Direction-inference logic (canonical, from extraction_pipeline.py):
```python
if touch_type == 'stroke':
    start_y = group['sticker_blue_position_y'].iloc[0]
    end_y   = group['sticker_blue_position_y'].iloc[-1]
    direction = 'proximal' if end_y > start_y else 'distal'
else:
    direction = 'static'
```

The extractor duplicates this 4-line block because it receives the raw per-frame group DataFrame — not the summary row where `direction` is already resolved.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| One-hot binary (4 columns) | No implied metric distance between categories; each category contributes variance independently; can include subsets in combinations | Adds 4 columns even if only 2 are needed | **Chosen** |
| Ordinal encoding (1 column: tap=0, proximal=1, distal=2) | Fewer columns | Implies tap < proximal < distal — meaningless ordering; distorts BinningClusterer variance axis | Rejected |
| Extend shared columns into clustering pipeline | No new extractor needed | Couples clustering pipeline to categorical shared columns; non-numeric; requires pipeline changes | Rejected |
| Separate extraction runs per touch type | Fine-grained RF maps per modality | Halves the touch count per run; requires new orchestration logic; combinatorial explosion of combinations | Rejected |

### Architecture Changes

**New file:**
- `code/src/analysis/touch_analytics/feature_extraction/touch_category_extractor.py` — `TouchCategoryExtractor` class

**Modified files:**
- `code/src/analysis/touch_analytics/feature_extraction/__init__.py` — import + registry entry `touch_category`
- `configs/analyse_workflow_dag.yaml` — feature entry + feature combinations in `touch_clustering` and `map_receptive_fields_clustered`

**Unchanged:** `extraction_pipeline.py`, `clustering_pipeline.py`, `rf_cluster_pipeline.py`, all clusterers, all RF renderers.

### Output Columns

| Column | Value | Condition |
|--------|-------|-----------|
| `is_tap` | 1 | `type_metadata == 'tap'` |
| `is_stroke` | 1 | `type_metadata == 'stroke'` |
| `dir_proximal` | 1 | stroke and `end_y > start_y` |
| `dir_distal` | 1 | stroke and `end_y <= start_y` |

All other cases produce 0. Missing `type_metadata` column produces 0 for all four columns.

### BinningClusterer interaction

Binary columns (variance ≤ 0.25) will typically have lower variance than continuous features like `geo_pressure_mean` (variance often > 1). When combined in a multi-feature combination, the continuous feature will be selected as the primary bin axis. Using `touch_category` as a standalone combination produces bins that are effectively cluster splits on the dominant binary dimension — practically separating tap from stroke (and further splitting by direction within strokes).

---

## Implementation Plan

### Phase 1: Extractor (single phase)
**Goal:** Working `touch_category` feature, end-to-end from YAML toggle to output CSV and clustering.

**Tasks:**
- [x] Task 1.1 — Create `touch_category_extractor.py`: `TouchCategoryExtractor` implementing `FeatureExtractor`, reading `type_metadata` and computing direction from `sticker_blue_position_y`
- [x] Task 1.2 — Update `__init__.py`: import `TouchCategoryExtractor`, add `'touch_category': TouchCategoryExtractor` to `EXTRACTOR_REGISTRY`, add to `__all__`
- [x] Task 1.3 — Update `configs/analyse_workflow_dag.yaml`: add `touch_category: {enabled: false}` under `touch_feature_extraction.options.features`; add combinations `touch_category` and `pressure_and_category` (disabled by default) in `touch_clustering` and `map_receptive_fields_clustered`

**Files Modified:**
- `code/src/analysis/touch_analytics/feature_extraction/touch_category_extractor.py` — new file
- `code/src/analysis/touch_analytics/feature_extraction/__init__.py` — import + registry
- `configs/analyse_workflow_dag.yaml` — feature toggle + combinations

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Enable `touch_category` in DAG YAML, run extraction on one session, verify CSV columns `is_tap`, `is_stroke`, `dir_proximal`, `dir_distal` appear with 0/1 values
- [ ] Confirm `is_tap + is_stroke == 1` for every row (no touch is both)
- [ ] Confirm `dir_proximal + dir_distal == is_stroke` for every row (direction only set for strokes)
- [ ] Add `touch_category` to a feature combination, run clustering, verify it merges and clusters correctly
- [ ] Open DAG launcher GUI — confirm `touch_category` checkbox appears in the features grid
- [ ] Toggle checkbox, save config, verify YAML round-trips correctly

### Edge Cases
- [ ] Touch where `type_metadata` column is absent — all four columns return 0, no error
- [ ] Touch where `type_metadata` is neither 'tap' nor 'stroke' (e.g., 'static') — all four columns return 0
- [ ] Single-frame stroke touch — `start_y == end_y` → `end_y <= start_y` → `dir_distal=1` (acceptable; document in docstring)
- [ ] `sticker_blue_position_y` absent in session — `KeyError` propagated, caught by `_extract_all_touches` try/except as with all other extractors

---

## Documentation Plan

- [ ] Column definitions documented in `TouchCategoryExtractor` docstring
- [ ] Knowledge base note `note-somatosensory-units-and-calculations.md` — add `touch_category` columns to the metrics table

---

## Rollback Plan

1. Remove `touch_category: {enabled: false}` from DAG YAML (and any combinations referencing it)
2. Remove registry entry from `__init__.py`
3. Delete `touch_category_extractor.py`
4. No database or state changes — extraction outputs are new files that can be deleted

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Binary columns have near-zero variance in unbalanced sessions (all taps) | Medium | Low | BinningClusterer degrades gracefully to a single bin; warning logged; RF map still renders |
| Direction sign convention is anatomically inverted (proximal/distal depends on forearm orientation) | Medium | Low | Column names `dir_proximal`/`dir_distal` follow existing shared-column convention; verify against one known session before enabling in combinations |
| Duplicate direction-inference logic diverges from extraction_pipeline | Low | Low | Both are 4 lines; flag in docstring to keep in sync if extraction_pipeline logic changes |

---

## References

- Existing extractors: `pressure_extractor.py`, `temporal_extractor.py`, `statistical_extractor.py`
- Direction inference source: `extraction_pipeline._extract_all_touches` lines 296–299
- Clustering feature selection: `clustering_pipeline._cluster_combination` line 327
- Feature extraction registry: `feature_extraction/__init__.py`
