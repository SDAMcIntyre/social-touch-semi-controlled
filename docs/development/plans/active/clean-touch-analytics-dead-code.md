# Plan: Clean Dead Code in touch_analytics Package

**Date:** 2026-03-16
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `chore/clean-touch-analytics-dead-code`

---

## Overview

The `touch_analytics` package contains dead stub functions and stale exports left over
from an earlier refactor that moved the real matrix-generation implementations into
`matrix_generation.py`. This plan removes the stubs, restores the missing idempotency
checks, and cleans the public API.

## Problem Statement

When the matrix-generation logic was extracted into `matrix_generation.py`, the original
functions in `touch_analysis.py` were left behind as placeholders (writing dummy CSV
headers). The `__init__.py` imports the real implementations from `matrix_generation.py`,
so the stubs are never called. However:

1. The stubs contained the `should_process_task` idempotency checks, which were **not
   copied** to the new implementations — so idempotency is now silently broken for both
   `generate_touch_summary_matrix` and `generate_ap_efficacy_matrix`.
2. `__all__` lists two names (`analyse_number_single_touches`,
   `analyse_ap_generation_efficacy`) that are Prefect flow names and do not exist in the
   package.
3. The dead stubs are confusing — a reader sees two definitions of the same function with
   different signatures and different behaviour.

## Goals

### In Scope
1. Remove dead stub functions from `touch_analysis.py`
2. Add `should_process_task` idempotency to the real implementations in `matrix_generation.py`
3. Clean stale entries from `__all__` in `__init__.py`

### Out of Scope
- Renaming Prefect flow functions (covered by the separate `rename-analysis-flow-functions` plan)
- Changing the matrix-generation algorithm or visualisation logic
- Adding unit tests (no test infrastructure exists for this package yet)

## Success Criteria

- [ ] `touch_analysis.py` contains no matrix-generation functions (only `generate_unified_summary` and its internals)
- [ ] `matrix_generation.py:generate_touch_summary_matrix` skips processing when outputs are up-to-date (unless `force=True`)
- [ ] `matrix_generation.py:generate_ap_efficacy_matrix` same behaviour
- [ ] `__all__` in `__init__.py` lists only names that are actually defined in the package
- [ ] All three files pass `python -m py_compile`

---

## Technical Design

### Approach

Targeted deletion and minimal additions — no new modules, no signature changes, no
architectural redesign.

### Separation of Concerns (current architecture)

The analysis pipeline has two layers:

| Layer | Location | Responsibility |
|-------|----------|----------------|
| **Flow** (orchestration) | `code/scripts/analysis_workflow.py` | Prefect `@flow` functions: collect inputs, set up output dirs, delegate to library, catch errors |
| **Library** (computation) | `code/src/analysis/touch_analytics/` | Pure functions: load CSVs, encode variables, build matrices, visualise, save |

The flow `analyse_number_single_touches_flow` calls `generate_touch_summary_matrix`.
The flow `analyse_ap_efficacy_flow` calls `generate_ap_efficacy_matrix`.
Neither flow performs idempotency checks — that responsibility belongs to the library layer.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Delete stubs, add idempotency to real impl (this plan) | Minimal change, fixes the bug | None | **Chosen** |
| Move idempotency to the flow layer instead | Keeps library functions pure | Inconsistent with `generate_unified_summary` and `generate_session_summary` which check internally | Rejected |

### Knowledge Base Constraints

- **Somatosensory units** (`note-somatosensory-units-and-calculations.md`): no unit
  calculations are being changed — only control flow (idempotency) is added.
- **CuPy import order** (`note-cupy-import-order.md`): `matrix_generation.py` does not
  import from `preprocessing`, so no constraint applies. `touch_analysis.py` already
  imports from `preprocessing` correctly.

---

## Implementation Plan

### Phase 1: Remove dead stubs from `touch_analysis.py`

**Goal:** Eliminate the placeholder functions that are shadowed by `matrix_generation.py`.

- [ ] Delete `generate_touch_summary_matrix` (lines 38-80)
- [ ] Delete `generate_ap_efficacy_matrix` (lines 82-118)
- [ ] Verify no remaining code in the file references these functions

**Files Modified:**
- `code/src/analysis/touch_analytics/touch_analysis.py` — delete two functions (~80 lines)

**Dependencies:** None

### Phase 2: Add idempotency to `matrix_generation.py`

**Goal:** Restore the `should_process_task` checks that were lost when stubs were created.

- [ ] Add `from utils.should_process_task import should_process_task` import
- [ ] In `generate_touch_summary_matrix`: add idempotency check before calling `_generate_matrix_internal`, early-return `output_file` if up-to-date
- [ ] In `generate_ap_efficacy_matrix`: same treatment

Pattern to follow (from `session_summary.py:31-37`):
```python
if not should_process_task(
    input_paths=input_files,
    output_paths=[output_file],
    force=force,
):
    logging.info(f"Skipping ... (up-to-date): {output_file.name}")
    return output_file
```

**Files Modified:**
- `code/src/analysis/touch_analytics/matrix_generation.py` — add import + two guard blocks

**Dependencies:** None (independent of Phase 1)

### Phase 3: Clean `__init__.py`

**Goal:** Remove stale names from `__all__` that don't correspond to anything in the package.

- [ ] Remove `"analyse_number_single_touches"` from `__all__`
- [ ] Remove `"analyse_ap_generation_efficacy"` from `__all__`

**Files Modified:**
- `code/src/analysis/touch_analytics/__init__.py` — edit `__all__` list

**Dependencies:** None (independent of Phase 1 and 2)

---

## Testing Plan

### Manual Verification
- [ ] `python -m py_compile code/src/analysis/touch_analytics/touch_analysis.py`
- [ ] `python -m py_compile code/src/analysis/touch_analytics/matrix_generation.py`
- [ ] `python -m py_compile code/src/analysis/touch_analytics/__init__.py`
- [ ] Grep codebase for any import of the deleted stubs to confirm nothing breaks
- [ ] Run analysis workflow on one session with `force_processing: false` twice — second run should skip matrix generation (idempotency works)

### Edge Cases
- [ ] `force_processing: true` — should regenerate even when outputs exist
- [ ] Missing input files — should log warning and return early (existing behaviour in `_generate_matrix_internal`)

---

## Documentation Plan

- [ ] No external documentation changes needed (internal cleanup)

---

## Rollback Plan

1. `git revert <commit>` — all changes are in three files within a single package
2. No data migrations, no schema changes, no external state affected

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Something imports the stubs from `touch_analysis` directly (bypassing `__init__.py`) | Low | Medium | Grep for `from .touch_analysis import generate_touch_summary_matrix` and `from analysis.touch_analytics.touch_analysis import` before deleting |
| `should_process_task` import fails in `matrix_generation.py` | Very Low | Low | Same import already works in `touch_analysis.py` and `session_summary.py` in the same package |

---

## References

- Dead stubs: `code/src/analysis/touch_analytics/touch_analysis.py` lines 38-118
- Real implementations: `code/src/analysis/touch_analytics/matrix_generation.py` lines 12-43
- Idempotency pattern: `code/src/analysis/touch_analytics/session_summary.py` lines 31-37
- Flow callers: `code/scripts/analysis_workflow.py` lines 125-202
- DAG config: `configs/analyse_workflow_dag.yaml`
