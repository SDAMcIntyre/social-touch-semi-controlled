# Plan: Touch Population Explorer — Use Stage 3 Features

**Date:** 2026-05-05
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-11 07:31
**Base Branch:** `feature/touch-population-explorer`
**Branch:** `feature/touch-population-explorer` (continuation — no new branch)

---

## Overview

**What:** Remove the built-in per-touch feature aggregation (`pressure_mean`, `velocity_amplitude_mean`) from the Touch Population Explorer's data loader, and instead source all scatter-plot axes from Stage 3 touch feature CSVs.
**Why:** The current `.mean()` aggregation propagates NaN from any single frame, causing all 826 touch means to be NaN — resulting in an empty scatter plot and a grey 3D mesh (no RF). Stage 3 already extracts per-touch features with proper NaN handling and a richer feature set.
**How:** Strip the built-in aggregation from `PopulationData`, fix the Stage 3 CSV discovery (flat glob → recursive), wire `touch_features_dir` through the launcher, and handle empty features gracefully in the GUI.

## Problem Statement

The Touch Population Explorer loads frame-level data successfully (826 touches, 69M contact points) but displays:
1. An empty 2D scatter plot (no dots)
2. A grey 3D forearm mesh (no RF heatmap)
3. A `RuntimeWarning: All-NaN slice encountered` from `np.nanpercentile`

**Root cause chain:**
- `touch_population_data.py:464-465` computes `pressure_arr.mean()` and `np.abs(velocity_arr).mean()` — numpy's `.mean()` returns NaN if *any* element is NaN
- The `pressure` and/or `hand_velocity_signed` CSV columns contain sporadic NaN values (frames where tracking failed)
- All-NaN feature arrays cascade: empty scatter (matplotlib ignores NaN) → `_init_filter_rect()` gets NaN percentile bounds → `_apply_filter_update()` creates all-False touch mask (NaN comparisons return False) → heatmap recomputed with zero contact points → all vertices NaN → grey mesh

**Design issue:** Re-deriving per-touch features from frame-level data duplicates work that Stage 3 feature extraction already does correctly. The frame-level CSV is still needed for contact-point parsing (3D heatmap), but scatter features should come from the proper extraction pipeline.

**Additional bugs:**
- `launch_touch_population_explorer` never passes `touch_features_dir` to `load_population_data()`, so Stage 3 features are never loaded
- `_merge_stage3_features()` uses a flat glob that misses CSVs in `touch_features/<feature>/` subdirectories

## Goals

### In Scope
1. Remove built-in `pressure_mean` and `velocity_amplitude_mean` aggregation from `PopulationData` and its loader
2. Fix Stage 3 CSV discovery: recursive glob to find CSVs in feature subdirectories
3. Wire `touch_features_dir` from launcher through to `load_population_data()`
4. Handle empty features gracefully: informative message in scatter area, 3D heatmap still renders
5. Invalidate stale caches via schema version bump

### Out of Scope
- Changing what Stage 3 feature extraction produces (that pipeline is correct)
- Adding new feature extractors
- Modifying the per-frame RF Feature Space Explorer (separate viewer)
- Making Stage 3 extraction run automatically before the viewer launches

## Success Criteria

- [ ] Scatter plot shows one dot per touch with Stage 3 feature axes (not `pressure_mean`/`velocity_amplitude_mean`)
- [ ] 3D forearm heatmap shows coloured RF (not all grey)
- [ ] X/Y axis combos list all numeric Stage 3 features
- [ ] If Stage 3 features are unavailable: scatter shows informative message, 3D heatmap still renders with all contacts
- [ ] No `RuntimeWarning: All-NaN slice encountered` during normal operation
- [ ] Old `.npz` caches auto-invalidate (schema version bump)

---

## Technical Design

### Approach

Keep reading the frame-level CSV for contact-point parsing (the 3D heatmap needs per-frame spatial data with IFF/spike values), but remove all per-touch scalar aggregation from the loader. All scatter-plot features come exclusively from Stage 3 CSVs via `_merge_stage3_features()`.

The `PopulationData` dataclass loses its `pressure_mean` and `velocity_amplitude_mean` fields. The existing `extra_feature_names` / `extra_feature_matrix` fields become the sole source for scatter axes — rename them to `feature_names` / `feature_matrix` to reflect they are no longer "extra".

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Fix `.mean()` → `np.nanmean()` for built-in features | Minimal change, keeps working without Stage 3 | Duplicates Stage 3 work; limited to 2 features; masks the design issue | Rejected |
| Remove built-in features, source all from Stage 3 | Single source of truth; richer feature set; proper NaN handling | Requires Stage 3 extraction to have been run | Chosen |
| Embed a mini feature extractor in the viewer | Self-contained, no Stage 3 dependency | Duplicates extraction pipeline; maintenance burden | Rejected |

### Architecture Constraints (from knowledge base)

- **Empty-data fallback paths are mandatory** for PyVista viewers — check for empty arrays before rendering, provide explicit fallback (note: `note-rf-cluster-gallery-gui-components.md`)
- **Deferred PyVista init** via `showEvent()` + `QTimer.singleShot(0, ...)` — already in place, must be preserved (note: `note-rf-feature-space-explorer-gui-components.md`)
- **Signal blocking** with `blockSignals(True/False)` during programmatic combo updates — already in place in `_populate_axis_combos()` (note: `note-qt-itemchanged-signal-recursion.md`)
- **All geometric features in mm** — no unit conversion in the pipeline (note: `note-somatosensory-units-and-calculations.md`)

### Architecture Changes

No new modules. Changes confined to three existing files:

```
code/src/analysis/receptive_field_mapping/
    touch_population_data.py        # Remove built-in features; fix glob; rename extra→feature
    gui/
        touch_population_explorer.py # Remove _BUILTIN_FEATURES; handle empty features
    rf_cluster_pipeline.py           # Wire touch_features_dir in launcher
```

---

## Implementation Plan

### Phase 1: Data Model Cleanup
**Started:** 2026-05-05
**Completed:** 2026-05-05
**Goal:** Remove built-in feature aggregation from `PopulationData` and fix Stage 3 CSV discovery.

- [x] Remove `pressure_mean` and `velocity_amplitude_mean` fields from `PopulationData` dataclass
- [x] Rename `extra_feature_names` → `feature_names`, `extra_feature_matrix` → `feature_matrix`
- [x] Update `get_feature_array()` to only look in `feature_names` (remove built-in name branches)
- [x] Remove per-touch mean computation from groupby loop (lines 464-465)
- [x] Remove `pressure_mean_arr` and `velocity_amplitude_mean_arr` construction and population (lines 652-653, 658-659)
- [x] Remove from cache save keys (lines 129, 131) and load keys (lines 201, 206, 221-226, 259-260)
- [x] Fix `_merge_stage3_features()` glob: `touch_features_dir.glob(...)` → `touch_features_dir.rglob(...)` to search subdirectories
- [x] Bump `_CACHE_SCHEMA_VERSION` from 1 → 2

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/touch_population_data.py` — Remove fields, rename extra→feature, fix glob, bump schema

**Dependencies:** None

### Phase 2: Pipeline Wiring
**Started:** 2026-05-05
**Completed:** 2026-05-05
**Goal:** Pass `touch_features_dir` from launcher to loader so Stage 3 features are actually loaded.

- [x] In `launch_touch_population_explorer._load_one()`, derive `touch_features_dir` from `series_csv` path: `series_csv.parent.parent / 'touch_features'`
- [x] Pass `touch_features_dir` to `load_population_data()`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — 2-line change in `_load_one()`

**Dependencies:** Phase 1

### Phase 3: GUI Hardening
**Started:** 2026-05-05
**Completed:** 2026-05-05
**Goal:** Update the GUI to work with Stage 3 features only and handle empty feature sets gracefully.

- [x] Remove `_BUILTIN_FEATURES` constant
- [x] Update `_populate_axis_combos()` to use `self._data.feature_names` directly
- [x] Guard `_deferred_start()`: if no features available, show message in scatter axes ("No touch features — run feature extraction first"), skip `_draw_scatter()` / `_init_filter_rect()`, still call `_render_3d()` with all contacts
- [x] Guard `_init_filter_rect()`: if percentiles return NaN, log warning and skip rect initialization
- [x] Update all references from `extra_feature_names` → `feature_names`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/touch_population_explorer.py` — Remove built-in features, add empty-data guards

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Run Stage 3 feature extraction for the test session (enable `touch_feature_extraction` in DAG if needed)
- [ ] Launch `explore_touch_population` — scatter shows dots with Stage 3 feature axes
- [ ] Confirm 3D mesh shows coloured RF heatmap (not all grey)
- [ ] Switch X/Y axis combos — scatter redraws with different feature dimensions
- [ ] Drag filter rectangle — heatmap updates, touch count label reflects filtered count
- [ ] Switch heatmap modes (spike density / mean IFF / cumulative IFF) — visible spatial variation
- [ ] Toggle gesture checkboxes — both scatter and heatmap update

### Edge Cases
- [ ] Stage 3 features not available: scatter shows informative message, 3D heatmap renders with all contacts, no crash
- [ ] Stage 3 CSVs exist but contain no numeric columns: same graceful fallback
- [ ] Session with no spikes: heatmap all dark blue (spike density = 0), no crash
- [ ] Single touch in session: scatter shows one dot, rectangle defaults gracefully

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — note that Touch Population Explorer requires Stage 3 feature extraction

---

## Rollback Plan

1. **All changes are on the existing feature branch** — no new branch to clean up
2. **Rollback procedure:** `git revert` the fix commit(s); old cached `.npz` files will be recomputed on next load (schema version mismatch triggers recompute)
3. **Data considerations:** `.npz` sidecar caches are inert — stale caches auto-invalidate via schema version check

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Stage 3 features not extracted for test sessions | Medium | Medium | Document the dependency; show clear message in GUI when features are missing |
| Session stem mismatch between series CSV and Stage 3 CSV filenames | Low | High | The extraction pipeline uses `input_file.stem` to name outputs, and the series CSV is the input — stems will match |
| `rglob` picks up unrelated CSVs in nested directories | Low | Low | The session stem prefix is specific enough to avoid false matches |

---

## References

- Parent plan: `docs/development/plans/active/touch-population-explorer.md`
- Stage 3 extraction output: `code/src/analysis/touch_analytics/extraction_pipeline.py` (line 13: `<feature_name>/<session>_touch_summary.csv`)
- Knowledge base: `docs/development/knowledge-base/note-rf-feature-space-explorer-gui-components.md`
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
