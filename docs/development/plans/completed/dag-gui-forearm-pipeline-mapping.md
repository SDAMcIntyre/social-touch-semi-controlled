# Plan: DAG GUI Forearm Pipeline Mapping

**Date:** 2026-03-06
**Author:** Basil Duvernoy
**Status:** Completed
**Completion Date:** 2026-03-06
**Branch:** `feature/multi-snapshot-forearm-registration` (existing)

---

## Overview

The DAG Launcher GUI already discovers `preprocess_forearm_manual_dag.yaml` and displays it in the workflow selector, but clicking "Run" shows "Script not found" because the automatic naming convention fails to resolve the actual script. A one-line override fixes the mapping.

## Problem Statement

The GUI maps DAG config stems to scripts by stripping the `_dag` suffix and looking for `code/scripts/{stem}.py`. For the forearm pipeline:

- **Expected:** `code/scripts/preprocess_forearm_manual.py`
- **Actual:**   `code/scripts/preprocess_pipeline_extract_forearm_manual.py`

The Run button is disabled and displays "Script not found" for this workflow.

## Goals

### In Scope
1. Make the forearm manual pipeline launchable from the DAG Launcher GUI

### Out of Scope
- Renaming the existing script (would break existing usage / muscle memory)
- Adding GUI-specific forearm configuration panels (e.g. a forearm config directory selector)
- Modifying the forearm pipeline script itself

## Success Criteria

- [x] Selecting "Preprocess Forearm Manual" in the GUI shows the resolved script path (not "Script not found")
- [x] The Run button is enabled and launches the forearm pipeline subprocess
- [x] The task panel displays all 6 tasks from `preprocess_forearm_manual_dag.yaml`

---

## Technical Design

### Approach

Add one entry to the existing `_SCRIPT_OVERRIDES` dictionary in `launcher_window.py`. This is the same mechanism already used to map `analyse_workflow` to `analysis_workflow`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Add `_SCRIPT_OVERRIDES` entry | Minimal change, uses existing pattern, no side effects | Another manual override to maintain | **Chosen** |
| Rename script to match convention | No override needed | Breaks existing invocations and habits | Rejected |
| Rename DAG YAML to match script | Script resolves automatically | Long YAML name, breaks existing config references | Rejected |

### Architecture Changes

None. Reuses the existing `_SCRIPT_OVERRIDES` pattern in `launcher_window.py`.

---

## Implementation Plan

### Phase 1: Add Script Override
**Goal:** Map `preprocess_forearm_manual_dag.yaml` to `preprocess_pipeline_extract_forearm_manual.py`

- [x] Add entry `"preprocess_forearm_manual": "preprocess_pipeline_extract_forearm_manual"` to `_SCRIPT_OVERRIDES`

**Files Modified:**
- `code/src/utils/gui/dag_launcher/launcher_window.py` (line ~31) — Add one dict entry

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [x] Run `python code/scripts/launch_dag_config_gui.py`
- [x] Click "Preprocess Forearm Manual" in the workflow selector
- [x] Confirm the run bar shows the resolved script path and Run button is enabled
- [x] Confirm the task panel shows the 6 tasks (extract, curate, clean, normals, mesh, register)
- [x] Click Run and verify the subprocess launches (check console output)

---

## Documentation Plan

- [x] No documentation changes needed (internal wiring fix)

---

## Rollback Plan

1. Remove the single dict entry from `_SCRIPT_OVERRIDES`
2. No data, config, or state changes to revert

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Typo in override key or value | Low | Low | Verified both stems from actual filenames |

---

## References

- Existing override pattern: `code/src/utils/gui/dag_launcher/launcher_window.py:31-33`
- DAG config: `configs/preprocess_forearm_manual_dag.yaml`
- Pipeline script: `code/scripts/preprocess_pipeline_extract_forearm_manual.py`
