# Plan: Rename Analysis Flow Functions for Clarity

**Date:** 2026-03-16
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/rename-analysis-flow-functions`

---

## Overview

Two analysis pipeline flow functions (`generate_session_summary_flow` and `process_unified_touches_flow`) have vague names that make the pipeline hard to understand for new readers. This plan renames them to action+output-focused names that clearly describe what each flow produces.

## Problem Statement

The current flow names use generic verbs ("generate", "process") that don't convey what artifact each step produces. A new reader encountering `process_unified_touches_flow` cannot tell whether it processes raw touch data, merges touch records, or produces summary statistics. Clearer names reduce onboarding friction and make the DAG configuration self-documenting.

## Goals

### In Scope
1. Rename `generate_session_summary_flow` to `summarize_session_blocks_flow`
2. Rename `process_unified_touches_flow` to `summarize_touches_per_session_flow`
3. Update all references: Prefect `@flow(name=...)`, DAG YAML keys, `available_tasks` entries, log messages, and comments

### Out of Scope
- Renaming the three downstream flows (`analyse_number_single_touches_flow`, `analyse_ap_efficacy_flow`, `map_receptive_fields_flow`) — can be a follow-up
- Changing any behavior or function signatures
- Renaming the imported library functions (`generate_unified_summary`, `generate_session_summary`)

## Success Criteria

- [ ] Both functions and their Prefect flow names use the new names
- [ ] DAG YAML config keys and all `depends_on` references updated
- [ ] `available_tasks` registration uses new string keys and function references
- [ ] No stale references to old names remain in the codebase
- [ ] `python code/scripts/analysis_workflow.py --help` runs without errors

---

## Technical Design

### Approach

Straightforward rename across two files. No logic changes, no new modules, no API changes. The rename touches the function definitions, their Prefect decorator names, the DAG YAML task keys, dependency lists, and a log warning message.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Action + output names (chosen) | Describes what each flow produces; consistent with each other | Slightly longer names | Chosen |
| Verb + domain noun (`tabulate_block_trials`, `extract_touch_kinematics`) | Domain-precise | Less intuitive for non-domain readers | Rejected |
| Pipeline-step names (`reduce_sessions_to_touch_summaries`) | Emphasizes data transformation | Verbose; mixes abstraction levels | Rejected |

### Architecture Changes

None. This is a pure rename with no structural changes.

---

## Implementation Plan

### Phase 1: Rename (single phase)
**Goal:** Update all names and references

- [ ] Rename function + decorator in `analysis_workflow.py` (2 functions)
- [ ] Update `available_tasks` list entries (string keys + function refs)
- [ ] Update log warning referencing old name
- [ ] Rename DAG YAML task keys and update `depends_on` lists
- [ ] Update YAML comments referencing old names

**Files Modified:**
- `code/scripts/analysis_workflow.py` — Rename 2 function definitions, 2 `@flow(name=...)` decorators, 2 `available_tasks` entries, 1 log warning
- `configs/analyse_workflow_dag.yaml` — Rename 2 task keys, update 3 `depends_on` entries, update 2 comments

**Dependencies:** None

### Detailed Changes

#### `code/scripts/analysis_workflow.py`

| Location | Old | New |
|----------|-----|-----|
| Line 43 decorator | `@flow(name="generate_session_block_summary")` | `@flow(name="summarize_session_blocks")` |
| Line 44 function | `def generate_session_summary_flow(` | `def summarize_session_blocks_flow(` |
| Line 79 decorator | `@flow(name="process_unified_touches")` | `@flow(name="summarize_touches_per_session")` |
| Line 80 function | `def process_unified_touches_flow(` | `def summarize_touches_per_session_flow(` |
| ~Line 440 tasks | `("generate_session_summary", generate_session_summary_flow)` | `("summarize_session_blocks", summarize_session_blocks_flow)` |
| ~Line 440 tasks | `("process_unified_touches", process_unified_touches_flow)` | `("summarize_touches_per_session", summarize_touches_per_session_flow)` |
| ~Line 162 warning | `Run 'process_unified_touches' first.` | `Run 'summarize_touches_per_session' first.` |

#### `configs/analyse_workflow_dag.yaml`

| Location | Old | New |
|----------|-----|-----|
| Task key | `generate_session_summary:` | `summarize_session_blocks:` |
| Task key | `process_unified_touches:` | `summarize_touches_per_session:` |
| Comment (line 41) | `process_unified_touches` | `summarize_touches_per_session` |
| Comment (line 48) | `process_unified_touches` | `summarize_touches_per_session` |
| 3x depends_on | `["process_unified_touches"]` | `["summarize_touches_per_session"]` |

---

## Testing Plan

### Manual Verification
- [ ] Run `python code/scripts/analysis_workflow.py --help` — confirms no import/syntax errors
- [ ] Grep entire codebase for old names — confirms no stale references
- [ ] Verify DAG config loads correctly by inspecting `DagConfigHandler` key matching against `available_tasks`

### Edge Cases
- [ ] Confirm no other YAML files reference these task keys (already verified: only `analyse_workflow_dag.yaml`)

---

## Documentation Plan

- [ ] No external documentation updates needed (this is an internal rename)
- [ ] Plan document serves as the change record

---

## Rollback Plan

1. Revert the single commit (`git revert <sha>`)
2. No data migrations or breaking changes — fully reversible

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Missed reference to old name | Low | Low | Grep codebase post-rename to verify |
| DAG config key mismatch with `available_tasks` | Low | Med | Verify key strings match exactly between YAML and Python |

---

## References

- Analysis workflow script: `code/scripts/analysis_workflow.py`
- DAG config: `configs/analyse_workflow_dag.yaml`
