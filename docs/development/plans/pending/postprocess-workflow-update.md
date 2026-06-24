# Plan: Update Postprocess Workflow Kinect Auto

**Date:** 2026-03-10 (revised 2026-03-16)
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `fix/postprocess-workflow-update`

---

## Overview

`code/scripts/postprocess_workflow_kinect_auto.py` has drifted from the reference
patterns established in `preprocess_workflow_kinect_auto.py`. It contains a critical bug
(output storage inside executor context) and is missing standard infrastructure (dashboard
monitor, error-skip loop, timestamped reports). This plan brings it up to date without
changing the postprocessing logic itself.

## Problem Statement

The postprocess workflow script was written before recent refactors and has not been
updated to match the conventions now used across all other workflow scripts. There is a
runtime bug where output storage occurs inside the executor context manager (risking
partial results on failure), a misnamed `@flow` decorator, and structural gaps that make
monitoring and debugging difficult.

## Goals

### In Scope
1. Fix critical bugs (output storage placement, flow name mismatch)
2. Align structural patterns with `preprocess_workflow_kinect_auto.py` (setup, monitor, error handling)
3. Clean up dead code (unused imports, stale comments)

### Out of Scope
- Adding new postprocessing stages
- Modifying the underlying `_5_postprocessing` functions
- Adding CuPy support (not needed — no preprocessing imports)
- Implementing parallel execution logic (only wiring the flag from config)

## Success Criteria

- [ ] Script passes `python -m py_compile` without errors
- [ ] PipelineMonitor dashboard appears on launch (matching preprocess behavior)
- [ ] `parallel_execution` parameter is read from DAG YAML (not hardcoded)
- [ ] On stage failure, remaining tasks are marked SKIPPED in the monitor report
- [ ] All unused imports removed; no stale comments remain

---

## Technical Design

### Approach

Align the postprocess script with the proven patterns in `preprocess_workflow_kinect_auto.py`.
Each fix is a targeted, isolated change — no architectural redesign.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Targeted fixes (this plan) | Minimal risk, focused changes | Doesn't restructure session-grouping | **Chosen** |
| Full rewrite to match reference exactly | Complete alignment | High risk for a script with session-grouping logic that differs from per-block preprocess | Rejected |
| Extract shared base class for workflows | DRY, future-proof | Over-engineering for 2 scripts with different iteration models | Rejected |

### Architecture Changes

No new modules. Single file modified: `code/scripts/postprocess_workflow_kinect_auto.py`.

### Knowledge Base Constraints

- **Somatosensory units** (`note-somatosensory-units-and-calculations.md`): postprocess
  stages consuming merged CSVs must preserve mm-based units. No unit conversion is
  introduced by this plan.
- **CuPy import order** (`note-cupy-import-order.md`): not applicable — script has no
  preprocessing imports. If added in future, apply the guarded import pattern.

---

## Implementation Plan

### Phase 1: Critical Bug Fixes

**Goal:** Eliminate runtime bugs that produce incorrect output or mask errors.

- [ ] **C2** — Move output-storage block (lines 256-264) outside `with executor:` block (line 224). Guard with `if not executor.error_msg:` to prevent storing partial results. Reference pattern: `preprocess_workflow_kinect_auto.py` lines 534-544.
- [ ] **C5** — Fix `@flow` name at line 60: `"analyze_pca_components"` -> `"set_xyz_reference_from_gestures"` to match the function name `set_xyz_reference_from_gestures_flow`.

**Files Modified:**
- `code/scripts/postprocess_workflow_kinect_auto.py`

**Dependencies:** None

### Phase 2: Structural Alignment

**Goal:** Match the reference script's infrastructure patterns for monitoring, error handling, and setup.

- [ ] **M1** — Extract `setup_environment()` function returning `(project_data_root, configs_dir, report_file_path)`. Add timestamp to report filename using `"%Y-%m-%d_%H-%M"` format: `f"{timestamp}_postprocess_status.xlsx"`. Reference: preprocess lines 604-616.
- [ ] **M2** — In `main()`, create `PipelineMonitor` with `live_plotting=True` and call `show_dashboard()`. After batch completes, `sleep(10)` then `close_dashboard(block=True)`. Pass `main_monitor.queue` as `monitor_queue` to `run_batch_postprocessing()`. Reference: preprocess lines 635-654.
- [ ] **M4** — Replace hardcoded `parallel=False` (line 359) with `main_dag_handler.get_parameter('parallel_execution', False)`.
- [ ] **M5** — Remove duplicate `DagConfigHandler(dag_config_path)` at line 285 (the instantiation at line 308 is the one used before the session loop; line 285 is immediately overwritten).
- [ ] **M6** — After `if executor.error_msg:` check (line 266), add loop over remaining `pipeline_stages` to mark each as `SKIPPED` in the monitor before returning. Reference: preprocess lines 548-550.

**Files Modified:**
- `code/scripts/postprocess_workflow_kinect_auto.py`

**Dependencies:** Phase 1

### Phase 3: Cleanup

**Goal:** Remove dead code and align style.

- [ ] **m1** — Remove stale comment at line 23: `"Mocking utils for template completeness - Replace with actual imports"` (imports are real, not mocked).
- [ ] **m2** — Change `from utils import path_tools` (line 24) to `import utils.path_tools as path_tools` to match reference script.
- [ ] **m3** — Remove unused imports: `os` (line 2), `traceback` (line 8), `pd` (line 13), `np` (line 14), `PCA` (line 16), `Any` (line 10). Keep `shutil`, `time`, `datetime` (used by `setup_environment()`).

**Files Modified:**
- `code/scripts/postprocess_workflow_kinect_auto.py`

**Dependencies:** Phase 2

---

## Revision Notes (2026-03-16)

Items removed from the original 2026-03-10 plan after codebase audit:

| Original Item | Reason Removed |
|---------------|----------------|
| **C1** — double-nested output dir | Already fixed; no `output_dir / "session_xyz_reference_from_gestures"` line exists |
| **C3** — `filter_by_receptive_field` stub return | Function does not exist in the script |
| **C4** — `determine_receptive_field_flow` type annotation | Function does not exist in the script |
| **M3** — try/except for FileNotFoundError | Already present at lines 297-303 (catches generic `Exception`) |
| **Optional DAG YAML update** — disable `filter_by_receptive_field` | Task does not exist in the DAG config |

Line numbers updated to match current file state (script is 363 lines).

---

## Testing Plan

### Manual Verification
- [ ] `python -m py_compile code/scripts/postprocess_workflow_kinect_auto.py` passes
- [ ] Launch via GUI (`configs/launcher.yaml` > Postprocess > Kinect [Auto]) — verify dashboard appears
- [ ] Run on a single session — verify output storage occurs only on success
- [ ] Intentionally disable a dependency task — verify downstream tasks are marked SKIPPED in report
- [ ] Set `parallel_execution: true` in DAG YAML — verify it is read (no crash; parallel may not fully work but should not be hardcoded off)

### Edge Cases
- [ ] Empty `kinect_configs` list — script should log warning and exit gracefully
- [ ] Missing DAG config file — should print error message and exit (not unhandled traceback)

---

## Documentation Plan

- [ ] No external documentation changes needed (internal script fix)

---

## Rollback Plan

1. `git revert <commit>` — all changes are in a single script file
2. No data migrations, no schema changes, no external state affected

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Moving output storage outside `with executor:` changes error flow | Low | Medium | Match exact pattern from preprocess reference; test with intentional failure |
| Monitor dashboard may conflict with session-grouped iteration | Low | Low | Same `PipelineMonitor` class used by preprocess; session_id used as block_name |

---

## References

- Reference script: `code/scripts/preprocess_workflow_kinect_auto.py`
- PipelineMonitor: `code/src/utils/pipeline/monitoring/pipeline_monitor.py`
- DagConfigHandler: `code/src/utils/pipeline/pipeline_config_manager.py`
- DAG config: `configs/postprocess_workflow_kinect_auto_dag.yaml`
- Underlying functions: `code/scripts/_5_postprocessing/`
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
