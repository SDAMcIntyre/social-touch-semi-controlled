# Plan: Fast skip-check — avoid CSV load for up-to-date sessions in extraction pipeline

**Created:** 2026-04-24
**Approved:** —
**Completed:** 2026-04-30
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/fast-skip-check-extraction-pipeline`

---

## Overview

The extraction pipeline takes several minutes to iterate over all sessions even
when every output already exists, because it unconditionally loads each session CSV
before checking whether work is needed.  Moving the should-process check above the
CSV read eliminates this cost and brings the "nothing to do" path from minutes to
~2–5 seconds.

## Problem Statement

`_extract_session()` (`extraction_pipeline.py`) follows this order:

```
1. resolve source_file path           (fast — path ops only)
2. pd.read_csv(source_file)           ← ALWAYS, line 199
3. shared preprocessing (block_order_id, interpolation)
4. for each feature:
       compute csv_stem / output_path
       should_process_task()          ← check happens here, after the read
       if skip → continue
```

`csv_stem` (line 230) is derived solely from `input_file.name` and does not depend on
CSV content.  All output paths can therefore be determined before the read.

For 465 already-processed sessions × 7 features the unconditional read fires 465 times,
reading large CSVs into RAM only to discard them.  This is the dominant wall-clock cost
of a "nothing to do" pipeline run.

## Goals

### In Scope
1. Hoist `csv_stem` computation above `pd.read_csv` in `_extract_session()`
2. Add a pre-check loop that tests all feature outputs before loading the CSV
3. Return early (no CSV load) when every feature is up-to-date
4. Leave the rest of `_extract_session()` unchanged so partial-staleness handling is unaffected

### Out of Scope
- Speeding up the clustering or series pipelines (different bottlenecks)
- Parallelising the session loop
- Changing any output CSV schema or feature logic

## Success Criteria

- [ ] A full DAG run where all outputs exist completes the extraction stage in ≤ 10 s
- [ ] Deleting one session's outputs and re-running reprocesses only that session correctly
- [ ] Running with `--force` bypasses the pre-check and processes every session
- [ ] Output CSVs produced on a forced re-run are byte-for-byte identical to those from the original run

---

## Technical Design

### Approach

Restructure `_extract_session()` in three minimal steps:

**Step 1 — hoist `csv_stem` above `pd.read_csv`**

Lines 230–235 compute `csv_stem` from `input_file.name` / `input_file.stem`.  Move this
block to immediately after the `source_file` resolution block (line 196).  Also move the
`stat_features` / `other_features` split (lines 237–238) to the same location — these
derive from the `features` parameter, not the CSV.

**Step 2 — add a pre-check block**

Insert between the `source_file` resolution and `pd.read_csv`:

```python
# Fast pre-check: skip CSV load if every feature output is already current
if not force:
    pre_results: dict[str, Path] = {}
    needs_work = False
    for feature_name in features:
        out = output_dir / feature_name / csv_stem
        if not out.exists():
            needs_work = True
            break
        try:
            if should_process_task(input_paths=[source_file],
                                   output_paths=[out], force=False):
                needs_work = True
                break
        except FileNotFoundError:
            needs_work = True
            break
        pre_results[feature_name] = out

    if not needs_work:
        for feature_name, out in pre_results.items():
            if progress is not None:
                progress.set_postfix_str(f"extract: {session_id}/{feature_name}",
                                         refresh=False)
                progress.update(1)
            print(f"  [extract] {session_id} / {feature_name} — up to date",
                  flush=True)
        return pre_results
```

The loop breaks on the first stale feature, keeping the fast path O(1) in the common
case where the very first feature is stale.

**Step 3 — leave existing logic intact**

The body of `_extract_session()` from `pd.read_csv` onward is unchanged.  It already
handles partial staleness correctly (some features skip, others run).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Hoist check + early exit (chosen) | Minimal diff, no schema change, O(stat_calls) skip path | None | Chosen |
| Batch-stat all sessions before any check | Could overlap I/O and compute | Complex restructuring, out of scope | Rejected |
| Cache CSV on first load, reuse across features | Zero extra reads within a session | Already the case — one read per session, not per feature | N/A |

### Architecture Changes

One file changes.  `should_process_task` is already imported; no new imports needed.

---

## Implementation Plan

### Phase 1: Hoist csv_stem and add pre-check
**Goal:** Eliminate unconditional CSV reads for sessions whose every feature output is current

- [x] Move `csv_stem` derivation block (lines 230–235) above `pd.read_csv` (line 199)
- [x] Move `stat_features` / `other_features` split (lines 237–238) to the same location
- [x] Insert pre-check block (see Technical Design) between `source_file` resolution and `pd.read_csv`

**Files Modified:**
- `code/src/analysis/touch_analytics/extraction_pipeline.py` — hoist + pre-check only

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Run extraction stage on full dataset where all outputs exist; confirm it completes in ≤ 10 s
- [ ] Confirm every session/feature prints "— up to date"
- [ ] Delete one session's feature outputs; re-run; confirm only that session is reprocessed
      and output CSVs are identical to originals
- [ ] Run with `--force`; confirm every session is fully reprocessed (no early exit)

### Edge Cases
- [ ] Session where `source_file` is missing — `should_process_task` raises `FileNotFoundError`;
      pre-check sets `needs_work = True` and falls through to CSV load (which will error loudly, as designed)
- [ ] Only one feature enabled — pre-check loop iterates once, behaves identically
- [ ] Mix of stale and current features — first stale feature breaks the loop, CSV is loaded,
      existing per-feature logic handles the rest

---

## Rollback Plan

The change is a pure code reordering within `_extract_session()`.  To revert: move
`csv_stem` and `stat_features` / `other_features` back to their original positions and
delete the pre-check block.  No data migrations, no schema changes.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `source_file` resolution result differs between pre-check and later code | Low | High | Same resolution block runs once before the pre-check; `source_file` is reused identically below |
| `mkdir(parents=True, exist_ok=True)` skipped for up-to-date features | None | None | Directories already exist if outputs exist |
| Progress bar count mismatches on early exit | Low | Low | Explicit `progress.update(1)` per feature in early-exit branch mirrors existing skip path |
