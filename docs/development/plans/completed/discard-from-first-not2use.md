# Plan: Discard From First Not2Use Option

**Date:** 2026-03-10
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/discard-from-first-not2use`

---

## Overview

Add a `discard_from_first_not2use` boolean option (default `True`) to `filter_block_by_neural_quality()`. When enabled, all trials from the first Not2Use trial onward are discarded — the file is truncated at that point. This simplifies the current behaviour where scattered Not2Use trials are removed individually while trailing suffixes are truncated, unifying both cases into a single "truncate at first bad trial" strategy by default.

## Problem Statement

Currently, `filter_block_by_neural_quality()` has two code paths:

1. **Trailing suffix** — if all Not2Use trials form a contiguous range to trial 12 (e.g., {7..12}), the file is truncated at the first Not2Use row.
2. **Scattered** — if Not2Use trials are non-contiguous (e.g., {3, 7}), only those specific trial rows are removed while keeping later trials.

In practice, once a Not2Use trial appears, all subsequent trials in that block are typically unreliable (e.g., electrode drift, subject discomfort). The scattered-removal path can produce datasets with gaps that misrepresent recording continuity. A "discard everything from the first bad trial" default better matches the experimental reality.

## Goals

### In Scope
1. Add `discard_from_first_not2use: bool = True` parameter to `filter_block_by_neural_quality()`
2. When `True`: truncate the block at the first row of the minimum Not2Use trial (unified behaviour, replaces both current paths)
3. When `False`: preserve the current scattered-removal behaviour (selective removal of individual Not2Use trials, with trailing-suffix optimisation)
4. Thread the option through from the DAG YAML config to the function call

### Out of Scope
- Changes to `parse_neural_quality_xlsx()` or the Excel format
- Changes to trial_id calculation or forward-fill logic
- GUI controls for this option (future work)
- Modifying the analysis/summary code downstream

## Success Criteria

- [ ] `filter_block_by_neural_quality()` accepts `discard_from_first_not2use` kwarg (default `True`)
- [ ] When `True`, the function truncates at the first row of `min(not2use_trials)` regardless of trial distribution
- [ ] When `False`, the function behaves identically to the current implementation (scattered removal + trailing-suffix optimisation)
- [ ] The option is configurable via `filter_by_neural_quality.options.discard_from_first_not2use` in the DAG YAML
- [ ] Logging output indicates which mode was used

---

## Technical Design

### Approach

Add a single boolean parameter to the existing function. When `True`, the function always uses the truncation path (current `is_trailing_suffix` branch). When `False`, it uses the current two-path logic. This is minimal, backward-compatible (default preserves the desired new behaviour), and easy to revert by passing `False`.

The existing `is_trailing_suffix` detection and truncation code already implements the core logic — the change is to make truncation the default regardless of whether trials form a contiguous suffix.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Boolean param on existing function | Minimal change, backward-compatible, easy to configure | Slightly overloads function | **Chosen** |
| Separate function (`truncate_from_first_not2use`) | Clean separation of concerns | Duplication of setup/parsing logic; two call sites to maintain | Rejected |
| Config-only (no function param) | No API change | Harder to test, couples function to config format | Rejected |

### Architecture Changes

No new modules or classes. Changes are localised to:
- The filter function signature and body
- The pipeline script that calls it
- The DAG YAML config schema

**Knowledge base check:** No applicable notes. CuPy import order is not relevant (this module does not use CuPy).

---

## Implementation Plan

### Phase 1: Function Parameter
**Goal:** Add the option to the filtering function

- [ ] Task 1.1 — Add `discard_from_first_not2use: bool = True` keyword argument to `filter_block_by_neural_quality()`
- [ ] Task 1.2 — Refactor the filtering logic: when `True`, always truncate at `min(not2use_trials)` (reuse existing truncation code); when `False`, keep the current `is_trailing_suffix` branching
- [ ] Task 1.3 — Update log messages to indicate the active mode (e.g., "truncated from trial X onward" vs "removed trials {X, Y}")

**Files Modified:**
- `code/scripts/_4_merging/filter_merged_by_neural_quality.py` — add param, refactor conditional block (lines 78–159)

**Dependencies:** None

### Phase 2: Pipeline Integration
**Goal:** Thread the option from YAML config through the pipeline script

- [ ] Task 2.1 — Read `discard_from_first_not2use` from task options in the pipeline script and pass to `filter_block_by_neural_quality()`
- [ ] Task 2.2 — Add `discard_from_first_not2use: true` to the DAG YAML config under `filter_by_neural_quality.options`

**Files Modified:**
- `code/scripts/merging_pipeline_neuron_to_kinect_auto.py` — pass option through to filter call
- `configs/merging_pipeline_neuron_to_kinect_auto_dag.yaml` — add default option value

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Run pipeline on a block with scattered Not2Use (e.g., {3, 7}) with `discard_from_first_not2use: true` — verify truncation at trial 3
- [ ] Run same block with `discard_from_first_not2use: false` — verify only trials 3 and 7 are removed (current behaviour)
- [ ] Run on a block with trailing-suffix Not2Use (e.g., {9..12}) with both settings — verify identical truncation result
- [ ] Run on a block with no Not2Use trials — verify file copied as-is regardless of setting
- [ ] Check log output shows correct mode indication

### Edge Cases
- [ ] Single Not2Use trial (e.g., {1}) with option `True` — should truncate everything from trial 1
- [ ] All 12 trials marked Not2Use — should produce an empty (header-only) output
- [ ] Option missing from YAML — should default to `True`

---

## Documentation Plan

- [ ] Update docstring of `filter_block_by_neural_quality()` to document new parameter
- [ ] Add inline comment in DAG YAML explaining the option

---

## Rollback Plan

1. Set `discard_from_first_not2use: false` in YAML to restore previous behaviour without code changes
2. If code revert needed: revert the single commit on `feature/discard-from-first-not2use`

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Default `True` changes output for blocks with scattered Not2Use that were previously selectively filtered | Med | Med | Explicit `false` option available; re-run affected sessions if needed |
| Downstream analysis expects specific trial structure | Low | Low | Filtering only removes rows; downstream already handles variable trial counts |

---

## References

- Filter module: `code/scripts/_4_merging/filter_merged_by_neural_quality.py`
- Pipeline script: `code/scripts/merging_pipeline_neuron_to_kinect_auto.py`
- DAG config: `configs/merging_pipeline_neuron_to_kinect_auto_dag.yaml`
- Original feature: `docs/development/plans/completed/filter-merged-neural-quality.md` (if exists)
