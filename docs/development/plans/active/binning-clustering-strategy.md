# Plan: Binning Clustering Strategy

**Date:** 2026-03-17
**Author:** Basil Duvernoy
**Status:** Active
**Branch:** `feature/unified-touch-analysis-pipeline`

---

## Overview

Add a new **binning** clustering strategy to the unified touch analysis pipeline
that divides each numeric feature column independently into N equal-width bins
(default 20, customizable). This provides a simple, non-algorithmic way to
discretize continuous features for downstream analysis, complementing the
existing kmeans and DBSCAN strategies.

## Problem Statement

The current pipeline only offers algorithmic clusterers (kmeans, DBSCAN) that
produce a single global cluster label from all features simultaneously. For
exploratory analysis it is useful to bin each feature independently into
fixed-width intervals — e.g. "touches in the top 5 % of depth" — without the
complexity or assumptions of a multi-dimensional clustering algorithm.

## Goals

### In Scope
1. New `BinningClusterer` class following the existing `TouchClusterer` strategy pattern
2. Per-feature bin columns (`bin_<feature>`) in the output CSV
3. Configurable number of bins (`n_bins`, default 20)
4. Configurable binning method: equal-width (`pd.cut`) or equal-frequency (`pd.qcut`)
5. YAML configuration entry in `analyse_workflow_dag.yaml`

### Out of Scope
- Multi-dimensional composite binning (cross-feature bin combinations)
- Binning-specific visualizations or heatmaps (existing heatmap generator is reused)
- Changes to the `TouchClusterer` abstract base class

## Success Criteria

- [ ] `BinningClusterer` registered and selectable via `method: binning` in YAML
- [ ] Output CSV contains `bin_<col>` columns with integer indices in [0, n_bins-1]
- [ ] `cluster_metadata.json` lists binned features and algorithm params (no numpy arrays)
- [ ] Existing kmeans/DBSCAN outputs are byte-identical (zero behavioral change)
- [ ] `n_bins` and `bin_method` are configurable from YAML

---

## Technical Design

### Approach

Use the existing `TouchClusterer` interface unchanged. The binning clusterer
returns the standard `(np.ndarray, dict)` tuple:

- **`labels`**: 1-D array — bin indices of the highest-variance feature (used as
  `cluster_label` for backward compatibility with heatmap generation and
  downstream consumers).
- **`metadata`**: dict including a special `"extra_columns"` key mapping
  `"bin_<col>"` names to their per-row bin arrays. The pipeline pops this key
  before JSON serialization and adds the columns to `result_df`.

This keeps the base class and all existing clusterers untouched.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Metadata `extra_columns` dict | No base-class change; backward compatible; 4-line pipeline patch | Slightly unconventional return-channel | **Chosen** |
| Change `fit_predict` to return DataFrame | Clean multi-column return | Breaks `TouchClusterer` contract; requires updating kmeans & DBSCAN | Rejected |
| One clusterer run per feature (loop in pipeline) | No interface change at all | Multiplies output directories; changes pipeline control flow significantly | Rejected |

### Architecture Changes

- **New module**: `clustering/binning_clusterer.py` — `BinningClusterer(TouchClusterer)`
- **Modified**: `clustering/__init__.py` — registry entry
- **Modified**: `unified_pipeline.py` — 4-line `extra_columns` handling in `_cluster_profile()`
- **Modified**: `configs/analyse_workflow_dag.yaml` — add `binning` profile

### Knowledge-Base Constraints

- Feature values are in native Kinect units (mm, mm/frame, mm^2). Equal-width
  bins operate on raw values, so bin edges are in the same units — no conversion
  needed (ref: `note-somatosensory-metric-units.md`).

---

## Implementation Plan

### Phase 1: Binning Clusterer
**Goal:** Implement and register the new strategy.

- [ ] Create `code/src/analysis/touch_analytics/clustering/binning_clusterer.py`
  - Inherit `TouchClusterer`, implement `fit_predict`
  - `pd.cut` (equal-width) / `pd.qcut` (equal-frequency) per feature column
  - Skip constant columns (variance == 0)
  - Fill NaN bins with -1
  - Return primary labels (highest-variance feature) + `extra_columns` in metadata
- [ ] Register in `code/src/analysis/touch_analytics/clustering/__init__.py`
  - Import `BinningClusterer`
  - Add `'binning': BinningClusterer` to `CLUSTERER_REGISTRY`
  - Add to `__all__`

**Files Modified:**
- `code/src/analysis/touch_analytics/clustering/binning_clusterer.py` — New file (~50 lines)
- `code/src/analysis/touch_analytics/clustering/__init__.py` — Add import + registry entry

**Dependencies:** None

### Phase 2: Pipeline Integration
**Goal:** Wire extra columns into the output CSV.

- [ ] In `_cluster_profile()` (unified_pipeline.py), after `result_df['cluster_label'] = labels`:
  ```python
  extra_cols = metadata.pop('extra_columns', None)
  if extra_cols:
      for col_name, col_values in extra_cols.items():
          result_df[col_name] = col_values
  ```
  `pop` removes numpy arrays before `json.dump`.

**Files Modified:**
- `code/src/analysis/touch_analytics/unified_pipeline.py` — 4 lines added after line 407

**Dependencies:** Phase 1

### Phase 3: Configuration
**Goal:** Enable binning in the analysis DAG config.

- [ ] Add `binning` entry under `clustering_profiles` in `configs/analyse_workflow_dag.yaml`:
  ```yaml
  binning:
    method: binning
    n_bins: 20
    bin_method: equal_width
  ```

**Files Modified:**
- `configs/analyse_workflow_dag.yaml` — 4 lines added under `clustering_profiles`

**Dependencies:** Phase 2

---

## Testing Plan

### Manual Verification
- [ ] Run analysis workflow with `binning` enabled alongside kmeans/dbscan
- [ ] Confirm `binning/` subdirectory created under each extraction profile's `clustering/` folder
- [ ] Open `pooled_touch_summary_clustered.csv` and verify `bin_*` columns exist with values in [0, 19]
- [ ] Open `cluster_metadata.json` and verify: `algorithm: "binning"`, `binned_features` list present, no raw arrays
- [ ] Compare kmeans and dbscan output CSVs to a prior run — should be unchanged

### Edge Cases
- [ ] Extraction profile with a single feature column — only one `bin_*` column produced
- [ ] Feature column with zero variance (constant) — column skipped, warning logged
- [ ] `n_bins` larger than unique values — `pd.cut` still produces valid bins (some empty)
- [ ] `bin_method: equal_frequency` with heavily skewed data — `duplicates='drop'` handles gracefully

---

## Documentation Plan

- [ ] No external docs needed — the YAML config is self-documenting
- [ ] Inline docstring on `BinningClusterer` class explains config keys

---

## Rollback Plan

1. Remove `binning` entry from `analyse_workflow_dag.yaml`
2. Remove `'binning'` from `CLUSTERER_REGISTRY` in `__init__.py`
3. Delete `binning_clusterer.py`
4. Remove `extra_columns` handling lines from `_cluster_profile()`

All changes are additive; removing them restores prior behavior exactly.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `bin_*` columns picked up as features by downstream code re-reading the CSV | Med | Low | `feature_cols` is computed before clustering runs, so current pipeline is safe. Document `bin_` prefix convention in metadata. |
| `pd.cut` producing all-NaN for constant columns | Low | Low | Skip columns with `nunique() < 2` and log a warning |
| `extra_columns` metadata key collides with future clusterer | Low | Low | Key name is specific; document convention in `BinningClusterer` docstring |

---

## Output CSV Structure

For extraction profile `max` with features `max_depth`, `max_velocity`, `max_contact_area`, `max_acceleration`:

| Column | Description |
|--------|-------------|
| *(all shared + feature columns)* | Unchanged from other clusterers |
| `cluster_label` | Bin index of highest-variance feature (0..19) |
| `bin_max_depth` | Equal-width bin index (0..19) |
| `bin_max_velocity` | Equal-width bin index (0..19) |
| `bin_max_contact_area` | Equal-width bin index (0..19) |
| `bin_max_acceleration` | Equal-width bin index (0..19) |

Output path: `4_analysed/unified_touches/<profile>/clustering/binning/pooled_touch_summary_clustered.csv`
