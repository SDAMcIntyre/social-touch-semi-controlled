# Plan: Add `block_order_id` Column During CSV Aggregation

**Date:** 2026-03-23
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-03-23 17:15
**Branch:** `feature/aggregation-block-order-id-column`

---

## Overview

**What:** Add a `block_order_id` column to the aggregated session CSV produced by `aggregate_session_blocks()`.
**Why:** The numeric block order is currently re-extracted from `source_block_file` by 3 independent downstream consumers using the same regex. Producing it at the aggregation point eliminates duplication and makes the CSV immediately usable.
**How:** Extract `block_order_id` from each block filename during aggregation, then simplify downstream consumers to prefer the pre-existing column.

## Problem Statement

`aggregate_session_blocks()` adds a `source_block_file` column containing the raw filename (e.g., `ST13-01_semicontrolled_block-order-03_merged_data.csv`), but not the parsed `block_order_id`. Three downstream modules independently re-extract it with the same regex pattern `block-order-(\d+)`:

- `extraction_pipeline.py:183-189`
- `touch_analysis.py:47-50`
- `session_summary.py:52-61`

This is redundant, fragile (if the filename pattern changes, all 3 must be updated), and means the aggregated CSV requires post-hoc parsing before it can be used for groupby operations.

## Goals

### In Scope

1. Add `block_order_id` column during CSV aggregation in `aggregate_blocks_session.py`
2. Simplify 3 downstream consumers to use the pre-existing column when available
3. Maintain backward compatibility with older aggregated CSVs that lack the column

### Out of Scope

- Changing the `filter_merged_by_neural_quality.py` extraction (operates on per-block filenames, not aggregated CSVs)
- Consolidating the regex pattern into a shared constant (minimal benefit for 1 remaining usage)
- Re-running aggregation on existing data

## Success Criteria

- [ ] Aggregated session CSVs contain a `block_order_id` column with correct numeric string values
- [ ] Downstream consumers (`extraction_pipeline.py`, `touch_analysis.py`, `session_summary.py`) use the column when present and fall back to extraction when absent
- [ ] No change in analysis output when processing the same data

---

## Technical Design

### Approach

Extract `block_order_id` from `filename.name` using `re.search(r'block-order-(\d+)', ...)` immediately after the existing `source_block_file` assignment on line 60 of `aggregate_blocks_session.py`. Store as a string (consistent with pandas `.str.extract()` output used downstream). Downstream consumers check for the column's presence before attempting extraction.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Extract at aggregation + simplify downstream | Single source of truth, backward compatible | Requires touching 4 files | Chosen |
| Extract at aggregation only (no downstream changes) | Minimal change (1 file) | Downstream still re-extracts, column may conflict | Rejected |
| Shared regex constant in `pipeline_shared.py` | DRY pattern | Doesn't solve the repeated extraction, adds an import dependency | Rejected |

### Architecture Changes

No new modules or classes. Minor modifications to 4 existing files:

- `aggregate_blocks_session.py` — add `import re`, add 2 lines extracting `block_order_id`
- `extraction_pipeline.py` — guard extraction with column-existence check
- `touch_analysis.py` — guard extraction with column-existence check
- `session_summary.py` — guard extraction with column-existence check

---

## Implementation Plan

### Phase 1: Add column and simplify consumers

**Goal:** Produce `block_order_id` at aggregation time and simplify downstream extraction.

**Tasks:**

- [x] Task 1.1 — Add `import re` to `aggregate_blocks_session.py`
- [x] Task 1.2 — After line 60 (`df['source_block_file'] = filename.name`), extract `block_order_id` from `filename.name` using `re.search(r'block-order-(\d+)', filename.name)` and assign to `df['block_order_id']`
- [x] Task 1.3 — In `extraction_pipeline.py` (lines 183-189), wrap extraction in `if 'block_order_id' not in df.columns`
- [x] Task 1.4 — In `touch_analysis.py` (lines 47-50), wrap extraction in `if 'block_order_id' not in df.columns`
- [x] Task 1.5 — In `session_summary.py` (lines 52-61), use `block_order_id` column if present instead of extracting into `_block_id`

**Files Modified:**

- `code/scripts/_4_merging/aggregate_blocks_session.py` — add `import re`, add `block_order_id` extraction (~2 lines)
- `code/src/analysis/touch_analytics/extraction_pipeline.py` — guard existing extraction with column check
- `code/src/analysis/touch_analytics/touch_analysis.py` — guard existing extraction with column check
- `code/src/analysis/touch_analytics/session_summary.py` — prefer `block_order_id` column over `_block_id` extraction

**Dependencies:** None

---

## Testing Plan

### Manual Verification

- [ ] Run postprocess pipeline on one session with `force_processing: true` for `aggregate_session`
- [ ] Open resulting `*_aggregated_session.csv` and verify `block_order_id` column exists with correct values (e.g., `01`, `02`, etc.)
- [ ] Run analysis workflow (`touch_feature_extraction` task) and confirm it picks up the pre-existing column (no extraction log/warning)
- [ ] Delete `block_order_id` column from a CSV manually, re-run analysis, and confirm fallback extraction still works

### Edge Cases

- [ ] Block filename without `block-order-` pattern — `block_order_id` should be `None` for those rows
- [ ] Older aggregated CSV without `block_order_id` column — downstream consumers should fall back to regex extraction

---

## Documentation Plan

- [ ] No external documentation needed (internal pipeline change)

---

## Rollback Plan

1. Revert the 4 file changes — downstream consumers retain their own extraction logic so the column's absence is harmless
2. No data migration needed — the column is purely additive

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Filename pattern lacks `block-order-` | Low | Low | Produces `None`, same as current downstream behavior |
| Older CSVs break downstream | Low | Med | Fallback extraction preserved in all 3 consumers |

---

## References

- Related file: `code/scripts/_4_merging/aggregate_blocks_session.py`
- Related pattern: `code/src/analysis/touch_analytics/extraction_pipeline.py:183-189`
