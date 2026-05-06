# Plan: IFF-Windowed Mean Features

**Date:** 2026-05-05
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/touch-population-explorer`
**Branch:** `feature/touch-population-explorer`

---

## Overview

**What:** Add two independently toggleable feature extractors that compute temporally-windowed means anchored to neural firing activity (IFF) within each single touch: `mean_during_iff` and `mean_before_iff`.

**Why:** The existing `StatisticalExtractor` computes means over the entire touch duration, treating all frames equally regardless of neural activity. Separating stimulus characteristics *during* neural firing from those *just before* firing onset enables direct comparison of what drives neural responses versus the baseline stimulus.

**How:** Two extractor classes (`MeanDuringIffExtractor`, `MeanBeforeIffExtractor`) sharing a common base in `iff_windowed_mean.py`, each registered separately in `EXTRACTOR_REGISTRY`. Each appears as its own feature entry in the DAG YAML — two checkboxes, two output folders.

## Problem Statement

- Current feature extraction aggregates over the full touch duration, mixing pre-firing, firing, and post-firing frames
- Researchers need to isolate stimulus conditions during neural firing versus the baseline just before firing onset
- No temporal windowing mechanism exists in the extraction pipeline — all aggregations treat every frame equally

## Goals

### In Scope

1. Two extractor classes sharing activity-mask logic via a common base
2. Two-tier activity mask: primary `Nerve_freq > 0`, fallback `contact_detected == 1`
3. `mean_during_iff` → `<col>_mean_during_iff` columns; `mean_before_iff` → `<col>_mean_before_iff` columns
4. Configurable pre-IFF window size on `mean_before_iff` (default 250ms)
5. Two separate `EXTRACTOR_REGISTRY` entries and DAG YAML feature entries
6. Each produces its own output folder under `4_analysed/touch_features/`
7. Unit tests

### Out of Scope

- Other aggregations (std, max, min) on the windowed frames — future enhancement
- First-contiguous-burst-only mode — future config option if needed
- Registering existing unregistered extractors (PressureExtractor, MoS, TouchCategory) — separate task
- Modifications to `extraction_pipeline.py` or `base.py`

## Success Criteria

- [ ] `MeanDuringIffExtractor` produces correct `_mean_during_iff` values for touches with IFF activity
- [ ] `MeanBeforeIffExtractor` produces correct `_mean_before_iff` values with a window up to 250ms before the first activity frame
- [ ] Both extractors fall back to `contact_detected` when `Nerve_freq` is all-zero
- [ ] All features are NaN when both `Nerve_freq` and `contact_detected` are all-zero
- [ ] Pipeline produces two separate output folders: `4_analysed/touch_features/mean_during_iff/` and `4_analysed/touch_features/mean_before_iff/`
- [ ] Each feature can be enabled/disabled independently in the DAG config
- [ ] All unit tests pass

---

## Technical Design

### Approach

A single file containing:
- `_IffWindowedBase(FeatureExtractor)` — shared logic: activity mask computation (two-tier `Nerve_freq > 0` → `contact_detected == 1`) and numeric column discovery (reuses `_EXCLUDE_FROM_AGGREGATION` from `statistical.py`)
- `MeanDuringIffExtractor(_IffWindowedBase)` — computes mean over all activity-masked frames
- `MeanBeforeIffExtractor(_IffWindowedBase)` — computes mean over the window before the first active frame

Two registry entries: `'mean_during_iff'` and `'mean_before_iff'`. The existing per-touch loop path in `extraction_pipeline.py` handles everything — no orchestrator changes needed.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Two classes with shared base | Independent toggles; own output folders; clean separation | Activity mask computed twice if both enabled | **Chosen** |
| Single extractor, single YAML key | Computes mask once | Cannot toggle features independently; one output folder | Rejected |
| Single class registered twice, mode via config | Less code duplication | Implicit coupling via config key; less clear | Rejected |

### Architecture Changes

New file in existing extractor directory:

```
representation/feature_characterization/
├── base.py                  # Unchanged
├── statistical.py           # Unchanged (imports _EXCLUDE_FROM_AGGREGATION)
├── pressure.py              # Unchanged
├── mos.py                   # Unchanged
├── touch_category.py        # Unchanged
└── iff_windowed_mean.py     # NEW — _IffWindowedBase, MeanDuringIffExtractor, MeanBeforeIffExtractor
```

Integration points:
- `__init__.py` — import both + register both in `EXTRACTOR_REGISTRY`
- `feature_extraction/__init__.py` (shim) — re-export both class names
- `configs/analyse_workflow_processing_dag.yaml` — two feature entries

Output layout:
```
4_analysed/touch_features/
├── mean_during_iff/
│   └── <session>_touch_summary.csv    # <col>_mean_during_iff columns
└── mean_before_iff/
    └── <session>_touch_summary.csv    # <col>_mean_before_iff columns
```

### Algorithm Detail

**Shared (_IffWindowedBase):**
```
1. Validate 'Nerve_freq' column exists (raise ValueError if missing)
2. Auto-discover numeric columns (same filter as StatisticalExtractor)
3. Build activity mask:
   a. activity_mask = Nerve_freq > 0
   b. IF not activity_mask.any():
        IF 'contact_detected' in columns:
          activity_mask = contact_detected == 1
        ELSE:
          raise ValueError (fail-fast)
4. Return (activity_mask, numeric_cols)
```

**MeanDuringIffExtractor:**
```
1. Get (activity_mask, numeric_cols) from base
2. IF not activity_mask.any(): RETURN all NaN
3. For each numeric column:
     <col>_mean_during_iff = mean where activity_mask is True
```

**MeanBeforeIffExtractor:**
```
1. Get (activity_mask, numeric_cols) from base
2. IF not activity_mask.any(): RETURN all NaN
3. first_active_idx = first True index in activity_mask
4. IF first_active_idx == 0: RETURN all NaN
5. window_start = max(0, first_active_idx - pre_iff_window_ms)
6. before_slice = group.iloc[window_start : first_active_idx]
7. For each numeric column:
     <col>_mean_before_iff = mean in before_slice
```

---

## Implementation Plan

### Phase 1: Extractor + Registration
**Started:** 2026-05-05
**Completed:** 2026-05-05
**Goal:** Create the extractor classes, register them, and add DAG config entries

- [x] Create `code/src/analysis/touch_analytics/representation/feature_characterization/iff_windowed_mean.py` with `_IffWindowedBase`, `MeanDuringIffExtractor`, `MeanBeforeIffExtractor`
- [x] Update `representation/feature_characterization/__init__.py` — import both, register both in `EXTRACTOR_REGISTRY`, add to `__all__`
- [x] Update `feature_extraction/__init__.py` (shim) — re-export both classes
- [x] Add `mean_during_iff` and `mean_before_iff` entries to `configs/analyse_workflow_processing_dag.yaml` under `touch_feature_extraction.options.features`

**Files Modified:**
- `code/src/analysis/touch_analytics/representation/feature_characterization/iff_windowed_mean.py` — **NEW** (~70 lines)
- `code/src/analysis/touch_analytics/representation/feature_characterization/__init__.py` — import + two registry entries
- `code/src/analysis/touch_analytics/feature_extraction/__init__.py` — re-export
- `configs/analyse_workflow_processing_dag.yaml` — two feature config entries

**Dependencies:** None

### Phase 2: Tests
**Started:** 2026-05-05
**Completed:** 2026-05-05
**Goal:** Unit tests covering all paths for both extractors

- [x] Create `code/tests/test_iff_windowed_mean.py` with test cases
- [x] Run tests and verify all pass

**Files Modified:**
- `code/tests/test_iff_windowed_mean.py` — **NEW** (~150 lines)

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests — MeanDuringIffExtractor

- [x] `test_during_basic` — Group with mixed zero/non-zero `Nerve_freq`; verify means match manual computation on non-zero frames only
- [x] `test_during_fallback_contact` — `Nerve_freq` all zero, `contact_detected` has non-zero frames; verify means use contact frames
- [x] `test_during_both_zero` — Both signals all zero; verify all features NaN
- [x] `test_during_all_active` — `Nerve_freq > 0` for every frame; verify means equal overall mean

### Unit Tests — MeanBeforeIffExtractor

- [x] `test_before_full_window` — First IFF frame at index 300; verify mean computed over frames 50–299 (250-frame window)
- [x] `test_before_clipped_window` — First IFF frame at index 100; verify mean computed over frames 0–99
- [x] `test_before_first_frame_active` — Activity starts at frame 0; verify all features NaN
- [x] `test_before_custom_window` — `pre_iff_window_ms: 100` in config; verify 100-frame window
- [x] `test_before_fallback_contact` — `Nerve_freq` all zero; verify before window uses `contact_detected` onset

### Unit Tests — Shared

- [x] `test_missing_nerve_freq_raises` — No `Nerve_freq` column; verify `ValueError` raised
- [x] `test_excludes_orchestrator_columns` — Verify `block_order_id`, `trial_id`, `Nerve_spike` not in output keys
- [x] `test_registry_lookup_during` — `get_feature_extractor('mean_during_iff', {})` returns `MeanDuringIffExtractor`
- [x] `test_registry_lookup_before` — `get_feature_extractor('mean_before_iff', {})` returns `MeanBeforeIffExtractor`

### Manual Verification

- [ ] Enable both features in DAG config, run pipeline on a session with neural data
- [ ] Verify two separate output folders: `mean_during_iff/` and `mean_before_iff/`
- [ ] Spot-check a `spike_elicited=1` touch: during values should differ from overall mean; before values should reflect pre-firing baseline

### Edge Cases

- [ ] Touch with only 1 frame where `Nerve_freq > 0`
- [ ] Touch with NaN values in numeric columns during the windowed frames

---

## Documentation Plan

- [ ] No README/CLAUDE.md updates needed (internal pipeline addition, follows existing patterns)
- [ ] DAG config YAML comments describe each feature

---

## Rollback Plan

1. Remove `iff_windowed_mean.py` and test file
2. Revert `__init__.py` changes (registry entries + imports)
3. Remove YAML config entries
4. No data migration — output CSVs are regenerated by the pipeline

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Sessions without `Nerve_freq` column (no neural data) | Low | Med | Fail-fast `ValueError` — session must have neural data to use these extractors |
| Column count (~20 new columns per extractor per touch) | Low | Low | Consistent with StatisticalExtractor pattern; downstream clustering selects specific columns |
| Activity mask computed twice when both features enabled | Low | Low | Mask computation is trivial (boolean comparison); negligible overhead vs. I/O |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Extractors + Registration | ~30 min | None |
| Phase 2: Tests | ~25 min | Phase 1 |

---

## References

- Key files: `code/src/analysis/touch_analytics/representation/feature_characterization/` (extractor directory)
- Knowledge base: No applicable notes
