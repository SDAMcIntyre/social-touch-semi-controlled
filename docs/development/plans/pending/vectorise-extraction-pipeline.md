# Plan: Vectorise Feature Extraction Pipeline

**Created:** 2026-04-24 00:00
**Approved:** 2026-04-24 00:00
**Completed:** —
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/refactor-series-transforms-sticker-selection`

---

## Overview

The touch feature extraction pipeline runs a per-touch Python loop once per statistical
aggregation feature, each with its own `groupby()` materialisation. Replacing these with
a single vectorised `groupby().agg()` call per session eliminates the dominant CPU cost.
Non-statistical extractors keep the existing per-touch loop but share one groupby.

## Problem Statement

`_extract_session()` iterates over every enabled feature (typically 7 statistical
aggregations). For each it calls `_extract_all_touches()`, which:
1. Materialises `list(df.groupby(TOUCH_KEYS))` — 7× per session
2. Loops over every touch group in Python
3. Calls `StatisticalExtractor.extract()` per touch, which re-discovers numeric columns
   and calls scalar aggregation functions per column

For 15 sessions × 1 000 touches × 7 features × ~40 columns this is ~4 200 000 scalar
Python aggregation calls and 105 redundant groupby materialisations. Everything that
`StatisticalExtractor` does can be expressed as a single `df.groupby().agg()` call.

## Goals

### In Scope
1. Add `_extract_statistical_batch()`: one vectorised groupby pass for all statistical aggregations
2. Refactor `_extract_session()` to use the batch path for aggregation features; keep the
   per-touch loop only for non-statistical extractors (`temporal`, `mechanics_of_solids`)
3. Share the groupby materialisation across non-statistical extractors
4. Hoist `scipy.stats.skew` import to module level in `statistical.py`

### Out of Scope
- Vectorising non-statistical extractors (`temporal`, `mechanics_of_solids`)
- Parallelising across sessions
- Changes to the series pipeline or clustering pipeline
- Any change to the output CSV schema

## Success Criteria

- [ ] Output CSVs are numerically identical to those produced by the old path (within
      float tolerance; skewness may differ by biased-vs-unbiased estimator)
- [ ] Extraction of 15 sessions with 7 stat features runs at least 5× faster
- [ ] Idempotency (per-feature up-to-date checks) behaves identically
- [ ] Non-statistical extractors (`temporal`, `mechanics_of_solids`) are unaffected

---

## Technical Design

### Approach

A new private function `_extract_statistical_batch(df, enabled_aggregations, has_nerve_data, session_id)`
computes all enabled statistical aggregations in three pandas operations:
1. `groupby().agg([native_aggs])` — mean, median, std, min, max
2. Post-agg arithmetic for `range` (max − min)
3. `groupby().skew()` for `skewness`

Shared metadata (type, contact location, spike_elicited) is computed vectorially with a
second agg call. Direction is computed with one `groupby().apply(infer_direction)` — still
a Python apply, but executed once per session instead of once per (session × feature).

`_extract_session()` separates features into `stat_features` (those in `AGGREGATION_NAMES`)
and `other_features`. Stat features use the batch path; other features use the existing
`_extract_all_touches()` loop, but the groupby is materialised once and shared.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Vectorised batch (chosen) | 10–30× speedup, no schema change | Skewness switches from biased to pandas unbiased estimator | Chosen |
| Parallelise across sessions | Large speedup | Complex, risky, out of scope | Rejected |
| Keep per-touch loop, hoist column discovery | Tiny win | Doesn't address the fundamental O(n) Python loop | Rejected |

### Architecture Changes

No new modules. Only two files change:

- `code/src/analysis/touch_analytics/extraction_pipeline.py` — add
  `_extract_statistical_batch()`, refactor `_extract_session()`, add `_groups` param to
  `_extract_all_touches()`
- `code/src/analysis/touch_analytics/representation/feature_characterization/statistical.py`
  — hoist scipy import

---

## Implementation Plan

### Phase 1: Batch statistical extraction
**Goal:** Replace the per-touch stat loop with a vectorised groupby pass
**Started:** 2026-04-24
**Completed:** —

- [ ] Add `TOUCH_KEYS` constant and `_EXCLUDE_FROM_AGGREGATION` import to `extraction_pipeline.py`
- [ ] Implement `_extract_statistical_batch()`
- [ ] Refactor `_extract_session()` to use the batch path for stat features and share groupby for other features
- [ ] Add `_groups` optional param to `_extract_all_touches()`

**Files Modified:**
- `code/src/analysis/touch_analytics/extraction_pipeline.py`

**Dependencies:** None

### Phase 2: Scipy import hygiene
**Goal:** Move scipy import to module level in statistical.py
**Started:** —
**Completed:** —

- [ ] Replace inline `from scipy.stats import skew` with a module-level try/except import

**Files Modified:**
- `code/src/analysis/touch_analytics/representation/feature_characterization/statistical.py`

**Dependencies:** None (independent of Phase 1)

---

## Testing Plan

### Manual Verification
- [ ] Run extraction on one session; diff output CSVs before/after (all stat features)
- [ ] Verify `temporal` and `mechanics_of_solids` CSVs are byte-for-byte identical
- [ ] Mark one stat feature output as up-to-date; confirm it is skipped by the batch
- [ ] Time full 15-session run before and after

### Edge Cases
- [ ] Session where all touches have `single_touch_id == 0` (no valid touches)
- [ ] Session missing `contact_location_*` columns
- [ ] Session missing `Nerve_spike` column
- [ ] Only one aggregation feature enabled (e.g. only `skewness`)
- [ ] Only `range` enabled (tests auxiliary min/max computation)

---

## Rollback Plan

`_extract_all_touches()` is kept intact. To revert: restore `_extract_session()` to call
`_extract_all_touches()` for all features (remove stat/other split and batch call).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Skewness biased vs unbiased difference | Med | Low | Document; values are equivalent for large N |
| Column suffix collision assertion fires on real data | Low | Low | Assertion raises early with clear message; fix the column name |
| `groupby().apply(infer_direction)` pandas 2.x deprecation warning | Med | Low | Suppress or add `include_groups=False` if needed |
