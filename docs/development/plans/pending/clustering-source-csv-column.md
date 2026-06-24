# Plan: Add `source_csv` Column to Clustered Output

**Date:** 2026-03-23
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/clustering-source-csv-column`

---

## Overview

The clustering pipeline's `pooled_touch_summary_clustered.csv` output is missing a column that identifies which source CSV each row was loaded from. This fix adds a `source_csv` column (first column) containing the CSV stem and ensures `session_id` is the second column.

## Problem Statement

When `_merge_feature_csvs()` in `clustering_pipeline.py` pools individual session CSVs, it discards the source filename. Users opening `pooled_touch_summary_clustered.csv` cannot trace rows back to their originating CSV file. The `session_id` column exists but is not positioned prominently either.

## Goals

### In Scope
1. Add `source_csv` column containing the full CSV stem (e.g. `ST13-01_semicontrolled_touch_summary`)
2. Ensure column ordering: `source_csv` first, `session_id` second, then remaining columns

### Out of Scope
- Changes to the extraction pipeline's CSV output format
- Adding source tracking to comparing pipeline outputs
- Modifying `SHARED_COLUMNS` or merge key logic

## Success Criteria

- [ ] `pooled_touch_summary_clustered.csv` has `source_csv` as its first column
- [ ] `session_id` is the second column
- [ ] `source_csv` values match the stem of each loaded `*_touch_summary.csv` file
- [ ] No regressions in clustering output (same row count, same cluster assignments)

---

## Technical Design

### Approach

Add the `source_csv` column inside the existing per-session loop in `_merge_feature_csvs()`, then reorder columns before returning the concatenated DataFrame. This is a minimal, two-line change in a single function.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Tag in `_merge_feature_csvs()` | Minimal change, single file | None | Chosen |
| Tag in extraction pipeline (earlier) | Available to all downstream | Would require re-running extraction; changes CSV schema | Rejected |

### Architecture Changes

None. Single function modified, no new modules or interfaces.

---

## Implementation Plan

### Phase 1: Add column and reorder
**Goal:** `source_csv` column present and correctly positioned in output

**Tasks:**
- [ ] Task 1.1 — In `_merge_feature_csvs()`, add `merged['source_csv'] = stem` after the per-session merge loop body (before `all_sessions.append(merged)`, line 165)
- [ ] Task 1.2 — Replace `return pd.concat(all_sessions, ignore_index=True)` with column-reordered version: `source_csv` first, `session_id` second

**Files Modified:**
- `code/src/analysis/touch_analytics/clustering_pipeline.py` — `_merge_feature_csvs()` function (lines 146–169)

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Run clustering pipeline on existing dataset
- [ ] Open `pooled_touch_summary_clustered.csv` and verify `source_csv` is column 1, `session_id` is column 2
- [ ] Verify `source_csv` values match the stems of files in the extraction output folders
- [ ] Verify row count matches previous output (no data loss)

### Edge Cases
- [ ] Single session — column still present with one unique value
- [ ] Multiple feature combinations — each output CSV has the column

---

## Documentation Plan

- No documentation changes needed (internal pipeline fix)

---

## Rollback Plan

1. Revert the single commit on `feature/clustering-source-csv-column`
2. No data migrations — output CSVs are regenerated on each run

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Downstream code expects specific column order | Low | Low | No downstream code indexes by position; all use column names |

---

## References

- `code/src/analysis/touch_analytics/clustering_pipeline.py` — `_merge_feature_csvs()` (line 78)
- `code/src/analysis/touch_analytics/pipeline_shared.py` — `SHARED_COLUMNS` definition
- `code/src/analysis/touch_analytics/matrix_generation.py` — reference pattern using `source_file_id`
