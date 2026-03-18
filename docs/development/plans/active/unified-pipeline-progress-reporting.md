# Plan: Unified Pipeline Progress Reporting

**Date:** 2026-03-17
**Author:** Basil Duvernoy
**Status:** Active
**Branch:** `feature/unified-touch-analysis-pipeline`

---

## Overview

Add percentage-based progress reporting to the unified touch analysis pipeline so the user can monitor execution status. Currently the pipeline runs silently, with only scipy `RuntimeWarning` messages appearing in the terminal, giving no indication of progress or estimated remaining time.

## Problem Statement

When running the unified touch analysis pipeline (`run_unified_touch_analysis`), the terminal shows only noisy scipy warnings like `RuntimeWarning: Precision loss occurred in moment calculation due to catastrophic cancellation`. There is no indication of:
- Which session is being processed
- Which extraction profile is running
- How many touches have been processed vs. total
- Overall pipeline completion percentage

This makes it impossible to estimate remaining time or detect stalls.

## Goals

### In Scope
1. Add two-level `tqdm` progress bars: outer (overall pipeline %) and inner (per-touch extraction)
2. Suppress noisy `RuntimeWarning` messages from scipy during pipeline execution
3. Show descriptive labels indicating current session, profile, and phase

### Out of Scope
- Progress reporting for other analysis workflows (AP efficacy, RF mapping)
- GUI-based progress (this is terminal/CLI only)
- Persistent progress logging to file
- Changes to the public API of `run_unified_touch_analysis`

## Success Criteria

- [ ] Outer progress bar shows overall % across all sessions x profiles + clustering
- [ ] Inner progress bar shows per-touch % within each extraction, disappears when done
- [ ] Scipy `RuntimeWarning` spam no longer clutters the terminal
- [ ] Existing `logging.info` messages still appear between progress updates
- [ ] No changes to the public API signature of `run_unified_touch_analysis`

---

## Technical Design

### Approach

Use `tqdm` (already a project dependency) for two-level progress bars. Suppress scipy warnings with `warnings.catch_warnings()` context manager scoped to the pipeline run only. All changes confined to `unified_pipeline.py`.

- **Outer bar:** Total units = `(N_sessions x M_extraction_profiles) + (M_profiles x K_clusterers)`. Increments after each session-profile extraction and each clustering step. Postfix shows current phase (e.g., `"extract: ST13-01/max"` or `"cluster: max/kmeans"`).
- **Inner bar:** Wraps the `df.groupby(...)` loop in `_extract_all_touches()`. Uses `leave=False` so it disappears after each extraction, keeping output clean. This is where most wall-clock time is spent.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Two-level tqdm (outer + inner) | Shows both overall % and per-touch detail; clean output with `leave=False` | Requires materializing groupby into list for tqdm total | **Chosen** |
| Single outer bar only | Simplest change | Too coarse — single session x 4 profiles can take minutes with no feedback | Rejected |
| Logging-based progress (no tqdm) | No new patterns | No progress bar, no %, clutters log output | Rejected |
| Three+ nested bars | Maximum detail | Visual clutter, terminal rendering issues with deeply nested tqdm | Rejected |

### Architecture Changes

No new modules or classes. Internal signature changes only:

| Function | New parameter |
|----------|---------------|
| `_extract_session` | `progress: tqdm = None` |
| `_extract_all_touches` | `desc: str = ""` |

---

## Implementation Plan

### Phase 1: Add progress reporting
**Goal:** Add tqdm progress bars and warning suppression

- [ ] Add `import warnings` and `from tqdm import tqdm` to imports
- [ ] Wrap `run_unified_touch_analysis` body in `warnings.catch_warnings()` with `warnings.filterwarnings("ignore", category=RuntimeWarning)`
- [ ] Create outer tqdm bar in `run_unified_touch_analysis` with calculated total units
- [ ] Pass outer bar to `_extract_session` via new `progress` parameter; update after each profile
- [ ] Wrap groupby loop in `_extract_all_touches` with inner tqdm bar (`leave=False`)
- [ ] Update outer bar after each clustering step in Phase 2 loop
- [ ] Handle edge cases: failed CSV loads, skipped (idempotent) profiles, empty clustering input — always increment bar to keep total accurate

**Files Modified:**
- `code/src/analysis/touch_analytics/unified_pipeline.py` — All changes (imports, 4 functions modified)

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Run pipeline: `python code/scripts/analysis_workflow.py --dag-config configs/analyse_workflow_dag.yaml`
- [ ] Confirm outer bar shows overall % and updates per extraction profile
- [ ] Confirm inner bar shows per-touch progress within each extraction, disappears after
- [ ] Confirm no scipy `RuntimeWarning` messages appear in terminal
- [ ] Confirm existing `logging.info` messages (e.g., `"Saved N touches → ..."`) still appear
- [ ] Confirm pipeline produces identical output files as before (no functional changes)

### Edge Cases
- [ ] Session with CSV load failure — outer bar still advances correctly
- [ ] Session skipped by idempotency — outer bar still advances correctly
- [ ] Empty clustering input (no session CSVs) — outer bar still advances correctly

---

## Documentation Plan

- [ ] No documentation updates needed (internal terminal UX improvement only)

---

## Rollback Plan

1. Revert the single commit touching `unified_pipeline.py`
2. No data migrations or breaking changes — purely additive terminal output

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Materializing groupby list uses extra memory | Low | Low | GroupBy result is views over existing DataFrame, cheap to list |
| tqdm output conflicts with Prefect logging | Low | Low | tqdm writes to stderr by default, Prefect uses stdout; tested in existing codebase modules |
| Warning suppression hides legitimate errors | Low | Medium | Scoped to `RuntimeWarning` only, inside `catch_warnings()` context manager |

---

## References

- Existing tqdm usage pattern: `code/src/preprocessing/motion_analysis/tactile_quantification/core/objects_interaction_controller.py`
- Warning suppression pattern: `code/src/preprocessing/led_analysis/core/led_blinking_analyzer.py`
- Follow-up: `docs/development/plans/active/persistent-pipeline-progress.md` — adds permanent milestone lines and heatmap tracking
