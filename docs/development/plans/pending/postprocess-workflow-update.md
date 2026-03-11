# Plan: Update Postprocess Workflow Kinect Auto

**Date:** 2026-03-10
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `fix/postprocess-workflow-update`

---

## Overview

`code/scripts/postprocess_workflow_kinect_auto.py` has drifted from the reference
patterns established in `preprocess_workflow_kinect_auto.py`. It contains critical bugs
(double-nested output directory, output storage inside executor context) and is missing
standard infrastructure (dashboard monitor, error handling, timestamped reports). This
plan brings it up to date without changing the postprocessing logic itself.

## Problem Statement

The postprocess workflow script was written before recent refactors and has not been
updated to match the conventions now used across all other workflow scripts. While all
imports and underlying APIs are still valid, there are runtime bugs that produce
incorrect output paths, and structural gaps that make monitoring and debugging difficult.

## Goals

### In Scope
1. Fix all critical bugs (double-nested dir, output storage, type annotations, flow name)
2. Align structural patterns with `preprocess_workflow_kinect_auto.py` (setup, monitor, error handling)
3. Clean up dead code (unused imports, stale comments)

### Out of Scope
- Implementing the `filter_by_receptive_field` function body (it remains a stub)
- Adding new postprocessing stages
- Modifying the underlying `_5_postprocessing` functions
- Adding CuPy support (not needed — no preprocessing imports)

## Success Criteria

- [ ] Script passes `python -m py_compile` without errors
- [ ] No double-nested output directories when PCA stage runs
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
| Targeted fixes (this plan) | Minimal risk, focused changes | Doesn't address `filter_by_receptive_field` stub | **Chosen** |
| Full rewrite to match reference exactly | Complete alignment | High risk for a script with session-grouping logic that differs from per-block preprocess | Rejected |
| Extract shared base class for workflows | DRY, future-proof | Over-engineering for 2 scripts with different iteration models | Rejected |

### Architecture Changes

No new modules. Single file modified plus optional DAG YAML tweak.

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

- [ ] **C1** — Remove double-nested output dir in `set_xyz_reference_from_gestures_flow` (delete line 51: `output_dir = output_dir / "session_xyz_reference_from_gestures"`)
- [ ] **C2** — Move output-storage block (lines 209-217) outside `with executor:`, guard with `if not executor.error_msg:`
- [ ] **C3** — Fix `filter_by_receptive_field` stub to `return []` instead of bare `return`
- [ ] **C4** — Correct return type annotation on `determine_receptive_field_flow`: `Tuple[List[Path], List[Path]]` -> `Tuple[List[Path], Path]`
- [ ] **C5** — Fix `@flow` name: `"analyze_pca_components"` -> `"set_xyz_reference_from_gestures"`

**Files Modified:**
- `code/scripts/postprocess_workflow_kinect_auto.py` — All five fixes

**Dependencies:** None

### Phase 2: Structural Alignment

**Goal:** Match the reference script's infrastructure patterns for monitoring, error handling, and setup.

- [ ] **M1** — Extract `setup_environment()` function; add timestamp to report filename
- [ ] **M2** — Create `PipelineMonitor` in `main()` with `live_plotting=True`; call `show_dashboard()` and `close_dashboard()`; pass `main_monitor.queue` to batch
- [ ] **M3** — Add try/except for `FileNotFoundError` around config loading in `main()`
- [ ] **M4** — Read `parallel_execution` from DAG handler instead of hardcoding `False`
- [ ] **M5** — Remove duplicate `DagConfigHandler` instantiation in `run_batch_postprocessing` (line 238)
- [ ] **M6** — Add skip-remaining-tasks loop on failure (mark downstream tasks as `SKIPPED` in monitor)

**Files Modified:**
- `code/scripts/postprocess_workflow_kinect_auto.py` — All six changes

**Dependencies:** Phase 1

### Phase 3: Cleanup

**Goal:** Remove dead code and align style.

- [ ] **m1** — Remove stale comment: `"Mocking utils for template completeness"`
- [ ] **m2** — Change `from utils import path_tools` to `import utils.path_tools as path_tools`
- [ ] **m3** — Remove unused imports: `pd`, `np`, `PCA`, `os`, `traceback`, `Dict`, `Any`; keep `shutil`, `time`, `datetime` (used by `setup_environment()`)

**Files Modified:**
- `code/scripts/postprocess_workflow_kinect_auto.py` — Import and comment cleanup

**Dependencies:** Phase 2

### Optional: DAG YAML Update

- [ ] Set `filter_by_receptive_field.enabled: false` in `configs/postprocess_workflow_kinect_auto_dag.yaml` until the function body is implemented

**Files Modified:**
- `configs/postprocess_workflow_kinect_auto_dag.yaml` — One-line change

**Dependencies:** None (independent)

---

## Testing Plan

### Manual Verification
- [ ] `python -m py_compile code/scripts/postprocess_workflow_kinect_auto.py` passes
- [ ] Launch via GUI (`configs/launcher.yaml` > Postprocess > Kinect [Auto]) — verify dashboard appears
- [ ] Run on a single session — verify output directories are flat (no double-nesting)
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
| `filter_by_receptive_field` enabled in DAG but still a stub | Medium | Low | Return `[]` prevents crash; optional DAG YAML change disables it |
| Monitor dashboard may conflict with session-grouped iteration | Low | Low | Same `PipelineMonitor` class used by preprocess; session_id used as block_name |

---

## References

- Reference script: `code/scripts/preprocess_workflow_kinect_auto.py`
- DAG config: `configs/postprocess_workflow_kinect_auto_dag.yaml`
- Underlying functions: `code/scripts/_5_postprocessing/`
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
