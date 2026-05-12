# Plan: Fix build_metrics_dataframe Empty-Cell Filter

**Date:** 2026-05-07
**Author:** Basil Duvernoy
**Status:** Draft
**Base Branch:** `dev`
**Branch:** `feature/fix-build-metrics-dataframe-empty-cell-filter`

---

## Overview

**What:** Remove a two-line early-exit guard in `build_metrics_dataframe()` that silently drops grid cells with zero touches from the output DataFrame.
**Why:** `compute_grid_cell_metrics()` already handles empty cells correctly (returns NaN metrics and `n_active_vertices=0` via its internal `_empty_row_dict()`), making the guard both wrong and redundant. It causes 7 test failures in `TestBuildMetricsDataframe`.
**How:** Delete lines 115–116 (`if touch_counts[g] == 0: continue`) in `rf_population_grid_metrics_pipeline.py`.

## Problem Statement

`build_metrics_dataframe()` loops over all `G` grid cells but skips any with `touch_counts[g] == 0`. This means downstream consumers receive a sparse DataFrame indexed 0…K (K < G) instead of the full G-row grid structure. Any code that aligns the output to grid position by row index silently produces wrong results. The 7 failing tests in `TestBuildMetricsDataframe` (added in commit `3cfc95b`) document the correct full-grid contract.

## Goals

### In Scope
1. Remove the zero-touch filter so all G grid cells appear as rows in the output
2. All 7 `TestBuildMetricsDataframe` tests pass
3. All 48 currently-passing tests continue to pass

### Out of Scope
- Fixing the centroid-mismatch bug documented in `investigate-rf-projection.md`
- Changing metric computation logic in `compute_grid_cell_metrics()`
- Adding new tests beyond the existing 7

## Success Criteria

- [ ] `pytest code/tests/test_rf_grid_cell_metrics.py` — all 55 tests pass (0 failures)
- [ ] `pytest code/tests/test_rf_population_grid_pipeline.py` — no regressions
- [ ] `build_metrics_dataframe()` returns exactly G rows for any input

---

## Technical Design

### Approach

Delete the two-line guard at lines 115–116 of `rf_population_grid_metrics_pipeline.py`:

```python
# Remove these two lines:
if touch_counts[g] == 0:
    continue
```

`compute_grid_cell_metrics()` already branches on whether the rf_map contains any active vertices. When `touch_counts[g] == 0` the rf_map is all-NaN and the function returns `_empty_row_dict()` with `n_active_vertices=0` and all metric fields set to NaN — exactly what the tests expect.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Delete the guard | Single change, downstream always gets full grid | None | Chosen |
| Keep guard, fix tests to not expect empty rows | Fewer rows to process | Breaks grid-structure contract, hides sparse output | Rejected |
| Replace guard with explicit empty-row insertion | Explicit control flow | More code, redundant since `compute_grid_cell_metrics` already handles it | Rejected |

### Architecture Changes

None. Single 2-line deletion in one function.

---

## Implementation Plan

### Phase 1: Fix
**Goal:** Delete the filter, verify tests pass.

- [ ] Delete lines 115–116 in `rf_population_grid_metrics_pipeline.py`
- [ ] Run `pytest code/tests/test_rf_grid_cell_metrics.py code/tests/test_rf_population_grid_pipeline.py`
- [ ] Commit

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_population_grid_metrics_pipeline.py` — delete 2 lines (115–116)

**Dependencies:** None

---

## Testing Plan

### Unit Tests
- [ ] All 55 tests in `test_rf_grid_cell_metrics.py` pass
- [ ] All tests in `test_rf_population_grid_pipeline.py` pass (no regressions)

### Manual Verification
- [ ] None required — the test suite fully covers the contract

---

## Documentation Plan

- [ ] None — this is a bug fix with no API or behaviour change visible to callers

---

## Rollback Plan

Revert is a single `git revert` of the fix commit.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Downstream consumer assumed sparse rows (relies on implicit dense-cell filtering) | Low | Medium | Search callers of `build_metrics_dataframe` before committing; none found in current codebase |

---

## References

- Failing tests: `code/tests/test_rf_grid_cell_metrics.py::TestBuildMetricsDataframe` (7 tests)
- Fix location: `code/src/analysis/receptive_field_mapping/rf_population_grid_metrics_pipeline.py:115–116`
- Related pending work: `docs/development/plans/pending/investigate-rf-projection.md`
