# Plan: Fix DAG Launcher Config-to-Script Name Mismatch

**Date:** 2026-03-06
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/dag-workflow-category-grouping` (current branch)

---

## Overview

The DAG launcher GUI fails to resolve the script for the forearm extraction workflow because the config filename does not match the script filename. Renaming the config file to follow the project's existing naming convention fixes the bug with minimal changes.

## Problem Statement

When a user selects the forearm extraction workflow in the DAG launcher GUI, the "Run" button is disabled and shows "Script not found". This is caused by a naming mismatch:

- **Config file**: `configs/preprocess_forearm_manual_dag.yaml` (stem: `preprocess_forearm_manual`)
- **Script file**: `code/scripts/preprocess_pipeline_extract_forearm_manual.py` (stem: `preprocess_pipeline_extract_forearm_manual`)

The `_resolve_script()` method in `launcher_window.py:260-268` strips the `_dag` suffix from the config stem, then looks for `{stem}.py` — yielding `preprocess_forearm_manual.py`, which does not exist.

A secondary effect: `_ORDERED_STEMS` in `workflow_selector.py:20-30` lists `preprocess_pipeline_extract_forearm_manual` (the script stem), but config discovery produces stem `preprocess_forearm_manual` — so the button ordering fails and the workflow appears at the bottom of the list instead of in its intended position within the Preprocess group.

## Goals

### In Scope
1. Fix the script resolution so the GUI can launch the forearm extraction workflow.
2. Fix the button ordering so the workflow appears in its correct position.
3. Maintain the project's Convention over Configuration naming pattern.

### Out of Scope
- Fixing the existing `analyse_workflow` / `analysis_workflow` spelling mismatch (already handled by `_SCRIPT_OVERRIDES`).
- Refactoring `_resolve_script()` or introducing a new config schema field.
- Changes to the forearm pipeline logic itself.

## Success Criteria

- [ ] Selecting the forearm workflow in the GUI resolves to `code/scripts/preprocess_pipeline_extract_forearm_manual.py`.
- [ ] The "Run" button is enabled when the forearm workflow is selected.
- [ ] The forearm workflow button appears in the Preprocess category group (2nd position, after Primary).
- [ ] Running the script directly (`python code/scripts/preprocess_pipeline_extract_forearm_manual.py`) still loads the correct config.

---

## Technical Design

### Approach

Rename the config file to match the script, following the Convention over Configuration pattern that 7 of 9 DAG configs already use (`{STEM}_dag.yaml` maps to `{STEM}.py`). This eliminates the mismatch at its source and requires zero GUI code changes.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Rename config** to match script | Convention-consistent; zero GUI changes; `_ORDERED_STEMS` already correct | Slightly longer filename (50 chars) | **Chosen** |
| **Add `_SCRIPT_OVERRIDES` entry** | Two one-line edits, no file rename | Adds maintenance burden; 2 of 9 configs needing overrides is a smell | Rejected |
| **Add `script:` field to YAML schema** | Most robust long-term | Disproportionate to problem; changes schema for all 9 configs; YAGNI | Rejected |

### Architecture Changes

No architectural changes. The fix aligns the config filename with the existing naming convention. The resolution logic in `_resolve_script()` is unchanged.

---

## Implementation Plan

### Phase 1: Rename and update reference
**Goal:** Align config filename with the script filename.

**Tasks:**
- [ ] Task 1.1 — `git mv configs/preprocess_forearm_manual_dag.yaml configs/preprocess_pipeline_extract_forearm_manual_dag.yaml`
- [ ] Task 1.2 — Update hardcoded path in `code/scripts/preprocess_pipeline_extract_forearm_manual.py` line 67: change `"preprocess_forearm_manual_dag.yaml"` to `"preprocess_pipeline_extract_forearm_manual_dag.yaml"`
- [ ] Task 1.3 — Grep for `preprocess_forearm_manual_dag` across the codebase to confirm no remaining references to the old filename.

**Files Modified:**
- `configs/preprocess_forearm_manual_dag.yaml` — Renamed to `configs/preprocess_pipeline_extract_forearm_manual_dag.yaml`
- `code/scripts/preprocess_pipeline_extract_forearm_manual.py` — Line 67: update config filename string

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Launch the DAG GUI and confirm the "Preprocess Pipeline Extract Forearm Manual" button appears in the **Preprocess** category group (after Primary).
- [ ] Click the button — confirm the run bar shows the resolved script path.
- [ ] Confirm the Run button is **enabled** (not grayed out).
- [ ] Optionally click Run to confirm the subprocess launches correctly.
- [ ] Run `python code/scripts/preprocess_pipeline_extract_forearm_manual.py` directly and confirm it finds its config file.

### Edge Cases
- [ ] Confirm all other workflows still resolve correctly (no regressions from the rename).

---

## Documentation Plan

- [ ] No external documentation changes needed (bugfix, not a new feature).

---

## Rollback Plan

Revert the rename and the one-line edit:
```
git mv configs/preprocess_pipeline_extract_forearm_manual_dag.yaml configs/preprocess_forearm_manual_dag.yaml
git checkout HEAD -- code/scripts/preprocess_pipeline_extract_forearm_manual.py
```

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Other tools or docs reference old config filename | Low | Low | Grep confirms no other references; completed plan docs are historical records |
| Longer filename is unwieldy | Low | Low | Only referenced in one line of code; comparable to existing filenames (e.g. `preprocess_workflow_kinect_visualisation_dag.yaml` is 48 chars) |
