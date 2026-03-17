# Plan: Remove Deprecated Analysis Workflow Tasks

**Date:** 2026-03-17
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/unified-touch-analysis-pipeline`

---

## Overview

Remove the two deprecated Prefect flow functions (`summarize_touches_per_session` and `analyse_number_single_touches`) from the analysis workflow. These flows were superseded by the `unified_touch_analysis` pipeline and are already disabled in the DAG config, but their code and config entries remain, adding maintenance burden and risk of accidental re-enablement.

## Problem Statement

`code/scripts/analysis_workflow.py` contains three flows that process touch data:

1. `unified_touch_analysis_flow` (active replacement)
2. `summarize_touches_per_session_flow` (deprecated, emits `DeprecationWarning`)
3. `analyse_number_single_touches_flow` (deprecated, emits `DeprecationWarning`)

All three are registered in the `available_tasks` list and will execute if enabled in the DAG config. The deprecated flows call different underlying functions (`generate_unified_summary`, `generate_touch_summary_matrix`) than the unified pipeline, meaning enabling all three would process the same data redundantly. The deprecated tasks are currently `enabled: false` in the DAG YAML, but their presence is confusing and error-prone.

## Goals

### In Scope
1. Delete the two deprecated flow functions from `analysis_workflow.py`
2. Remove their entries from `available_tasks` registration
3. Remove their task definitions from `configs/analyse_workflow_dag.yaml`
4. Clean up imports that become unused after removal
5. Update downstream DAG `depends_on` references that still name the old tasks

### Out of Scope
- Removing the underlying library functions (`generate_unified_summary`, `generate_touch_summary_matrix`) from `analysis.touch_analytics` — they remain exported for potential external use
- Modifying the unified pipeline itself
- Changing downstream flows (`analyse_ap_efficacy`, `map_receptive_fields`)

## Success Criteria

- [ ] `summarize_touches_per_session_flow` and `analyse_number_single_touches_flow` no longer exist in `analysis_workflow.py`
- [ ] Neither task appears in `available_tasks` or `configs/analyse_workflow_dag.yaml`
- [ ] No unused imports remain in `analysis_workflow.py`
- [ ] Downstream tasks (`analyse_ap_efficacy`, `map_receptive_fields`) still find their input files when run after `unified_touch_analysis`
- [ ] The workflow runs end-to-end with the updated DAG config without errors

---

## Technical Design

### Approach

Straight deletion of the deprecated code paths and their config entries. The unified pipeline already writes backward-compatible output to `4_analysed/unified_touches/<session>_touch_summary.csv` (via dual-write in `unified_pipeline.py` lines 211-218), so downstream flows that read from that path are unaffected.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Delete deprecated flows entirely | Clean codebase, no dead code | Irreversible (mitigated by git history) | **Chosen** |
| Keep flows but remove from `available_tasks` | Flows remain callable programmatically | Dead code stays, confusing | Rejected |
| Convert deprecated flows to thin wrappers around unified pipeline | Backward compat for any external callers | Over-engineering; no external callers exist | Rejected |

### Architecture Changes

No new modules or classes. Pure removal of ~100 lines of dead code and ~15 lines of DAG config.

---

## Implementation Plan

### Phase 1: Remove deprecated flows from workflow script
**Goal:** Delete dead code and clean up imports

**Tasks:**
- [ ] Task 1.1 — Delete `summarize_touches_per_session_flow` (lines 101-151)
- [ ] Task 1.2 — Delete `analyse_number_single_touches_flow` (lines 154-200)
- [ ] Task 1.3 — Remove both entries from `available_tasks` list (lines 477-478)
- [ ] Task 1.4 — Remove `generate_unified_summary` and `generate_touch_summary_matrix` from the import block (lines 31-32) since they are no longer used in this file

**Files Modified:**
- `code/scripts/analysis_workflow.py` — Remove ~100 lines of deprecated flow code, 2 task registrations, 2 unused imports

**Dependencies:** None

### Phase 2: Update DAG configuration
**Goal:** Remove stale task definitions and fix dependency references

**Tasks:**
- [ ] Task 2.1 — Delete the `summarize_touches_per_session` task block (currently `enabled: false`) from DAG YAML
- [ ] Task 2.2 — Delete the `analyse_number_single_touches` task block (currently `enabled: false`) from DAG YAML
- [ ] Task 2.3 — Update `depends_on` in `analyse_ap_efficacy` and `map_receptive_fields` tasks: replace `"summarize_touches_per_session"` with `"unified_touch_analysis"`

**Files Modified:**
- `configs/analyse_workflow_dag.yaml` — Remove ~15 lines of disabled task config, update 2 dependency references

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Run `python code/scripts/analysis_workflow.py --dag-config configs/analyse_workflow_dag.yaml` — confirm it starts without import errors and processes the enabled tasks
- [ ] Verify `analyse_ap_efficacy` task finds unified touch summaries at `4_analysed/unified_touches/` after `unified_touch_analysis` completes
- [ ] Verify `map_receptive_fields` task finds unified touch summaries at the same path
- [ ] Confirm no Python warnings or deprecation messages appear for the removed tasks

### Edge Cases
- [ ] Run with all optional tasks disabled except `unified_touch_analysis` — confirm clean execution
- [ ] Run with `analyse_ap_efficacy` enabled but `unified_touch_analysis` disabled — confirm graceful "no input files" warning (existing behavior via `_collect_unified_files`)

---

## Documentation Plan

- [ ] Update `docs/development/plans/pending/unified-touch-analysis-pipeline.md` to note that deprecated task removal is complete (or move relevant section to completed)

---

## Rollback Plan

1. `git revert <commit>` — all changes are in a single feature branch
2. No data migrations or external state changes involved
3. The removed code is preserved in git history indefinitely

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| External script calls deprecated flow by name | Low | Low | Flows are already disabled in config and emit deprecation warnings; no external references found in codebase |
| Downstream tasks can't find input files | Low | Med | Verified that `unified_pipeline.py` dual-writes to the same root path used by `_collect_unified_files()` |

---

## References

- Related Plan: `docs/development/plans/pending/unified-touch-analysis-pipeline.md`
- Backward-compat dual-write: `code/src/analysis/touch_analytics/unified_pipeline.py` lines 211-218
- Completed cleanup: `docs/development/plans/completed/clean-touch-analytics-dead-code.md`
