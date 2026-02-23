# Plan: Fix GLFW Cleanup Warnings in Forearm Batch Pipeline

**Date:** 2026-02-22
**Completed:** 2026-02-23
**Author:** Claude Code
**Status:** Completed
**Branch:** `fix/forearm-batch-glfw-cleanup`

---

## Overview

The forearm extraction batch pipeline emits repeated GLFW warnings at the end of processing, cluttering terminal output. This plan fixes the issue by disabling mesh visualization in batch mode (where it's not needed) while preserving interactive visualization for single-session workflows.

---

## Problem Statement

When `code/scripts/preprocess_pipeline_extract_forearm_manual.py` completes batch processing, the terminal displays 6+ spurious GLFW errors:

```
[Open3D WARNING] GLFW Error: The GLFW library is not initialized
[error] GLFW error: The GLFW library is not initialized
```

The script completes successfully, but the warnings indicate improper GLFW cleanup in Open3D's visualization system. The root cause is that batch mode shouldn't require interactive visualization windows—they slow down processing and leave GLFW in a bad state.

**Why it matters:** Terminal output pollution makes it harder to spot real errors or warnings. Batch processing should be silent and efficient.

---

## Goals

### In Scope
1. Eliminate GLFW warnings from batch pipeline execution
2. Maintain mesh visualization for single-session/interactive workflows
3. Make visualization mode explicit and configurable (not hardcoded)
4. Improve batch processing performance by skipping unnecessary window management

### Out of Scope
- Fixing Open3D's GLFW handling upstream
- Implementing alternative visualization libraries
- Adding progress UI or progress bars during batch processing

---

## Success Criteria

- [ ] Batch pipeline runs with `show=False` and produces no GLFW warnings
- [ ] All mesh files are correctly saved (`*.obj` format)
- [ ] Single-session/interactive mode still supports mesh visualization
- [ ] Code is clear about when visualization is enabled vs. disabled
- [ ] No hardcoded boolean flags; use function parameters instead

---

## Technical Design

### Approach

Introduce an `interactive` parameter to the batch execution pipeline:

1. **Add parameter to `execute_frame_batch()`** — Accept `interactive: bool = True` parameter
2. **Pass parameter through to `define_forearm_mesh()`** — Use `show=interactive` instead of hardcoded `show=True`
3. **Call batch executor with `interactive=False`** — In batch mode, disable visualization
4. **Preserve default `interactive=True`** — Maintains backward compatibility for any direct calls

This is the **minimal change** approach: we don't fix Open3D's GLFW cleanup, we just avoid triggering the problematic code path in batch mode.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Add `interactive` parameter (chosen)** | Minimal change, explicit, backward compatible, addresses root cause | Requires two params to thread through | ✅ Chosen |
| **Force GLFW shutdown between iterations** | Keeps visualization, may prevent context corruption | Fragile, relies on internal GLFW bindings, platform-specific | ❌ Rejected |
| **Suppress stderr during visualization** | Very minimal code change | Hides legitimate errors, poor engineering | ❌ Rejected |
| **Use Open3D's Visualizer class** | More control over lifecycle | Major refactoring, requires rewriting interaction loop | ❌ Deferred (future improvement) |

### Architecture Changes

**No architectural changes.** This is a simple parameter threading change:

```
execute_frame_batch(batch: FrameBatch, interactive: bool = True)
  → extract_forearm(..., interactive=interactive)       # prevents ArmSegmentation GUI in batch
  → define_forearm_mesh(..., show=interactive)          # PRIMARY FIX: prevents GLFW visualizer
```

The failed cleanup block in `_execute_all_batches()` (a `draw_geometries([])` no-op) is also removed.

**Affected files:**
- `code/scripts/preprocess_pipeline_extract_forearm_manual.py` — Three call sites + one dead block removed

**Integration:** No new dependencies or external integrations required.

---

## Implementation Plan

### Phase 1: Parameter Threading
**Goal:** Add `interactive` parameter to the execution pipeline

**Tasks:**
- [ ] **1.1** — Modify `execute_frame_batch(batch, interactive=True)` signature
  - Add parameter to function definition
  - Update docstring to document parameter
- [ ] **1.2** — Update both visualization calls within `execute_frame_batch()`
  - Change `extract_forearm(..., interactive=True)` → `extract_forearm(..., interactive=interactive)`
    - In batch mode this prevents the `ArmSegmentation` GUI (skin-colour/clustering sliders) from reopening for every frame
    - Note: this visualizer uses `gui.Application.instance.run()` (newer framework), so it does **not** cause the reported GLFW warnings — but it is logically inconsistent with headless batch processing
  - Change `define_forearm_mesh(..., show=True)` → `define_forearm_mesh(..., show=interactive)`
    - This is the **primary fix**: `show=True` calls `draw_geometries_with_key_callbacks()` which is the legacy GLFW-based visualizer and the confirmed source of the spurious warnings
- [ ] **1.3** — Modify `_execute_all_batches(batches, interactive=False)` signature
  - Add parameter with default False (batch mode)
  - Update docstring
- [ ] **1.4** — Thread parameter through call: `_execute_all_batches()` → `execute_frame_batch()`
  - Update the loop: `execute_frame_batch(batch, interactive=interactive)`
- [ ] **1.5** — Remove the dead cleanup block from `_execute_all_batches()`
  - The `draw_geometries([])` call (lines ~298–303) was a failed prior fix attempt; it is a no-op that does not reset the GLFW context
  - Once visualization is disabled in batch mode it becomes unreachable dead code

**Files Modified:**
- `code/scripts/preprocess_pipeline_extract_forearm_manual.py`
  - Line ~148 — `execute_frame_batch()` definition and calls to `extract_forearm()` and `define_forearm_mesh()`
  - Line ~205 — `run_session()` call to `_execute_all_batches()`
  - Line ~289 — `_execute_all_batches()` definition, internal call, and dead cleanup block

**Dependencies:** None

---

## Testing Plan

### Unit Tests
- [ ] **Manual test — Batch mode:** Run script in batch mode (`show=False`), verify no GLFW warnings appear
- [ ] **Manual test — Interactive mode:** Run single session with visualization enabled, verify windows open/close properly

### Integration Tests
- [ ] **End-to-end batch:** Process 2-3 sample sessions, verify all `.obj` mesh files created correctly
- [ ] **Mesh file integrity:** Spot-check generated meshes are valid OBJ format (not corrupted)

### Manual Verification
- [ ] Run full batch pipeline: `python code/scripts/preprocess_pipeline_extract_forearm_manual.py`
  - Scroll to end of output
  - Confirm no GLFW warnings appear
  - Verify `.SUCCESS` flags written correctly

- [ ] Test single-session interactive mode (if available)
  - Verify mesh visualization window appears
  - Verify window can be closed without errors
  - Note: GLFW warnings acceptable in interactive mode (not a batch failure)

### Edge Cases
- [ ] **Empty batch list:** `_execute_all_batches([])` — Should skip processing and cleanup without error
- [ ] **Single batch:** Single frame processed, no GLFW warnings
- [ ] **Exception in frame batch:** Verify `interactive` parameter is still honored if a batch fails

---

## Documentation Plan

- [ ] Update `execute_frame_batch()` docstring to document `interactive` parameter
- [ ] Update `_execute_all_batches()` docstring to document `interactive` parameter
- [ ] Update bug report: `docs/development/knowledge-base/bug-glfw-cleanup-forearm-batch-pipeline.md`
  - Add "Status: Fixed"
  - Link to this plan
- [ ] Optionally add inline comment in `_execute_all_batches()` explaining why `interactive=False`:
  ```python
  # Batch mode: disable mesh visualization (no interactive inspection needed)
  # and avoid GLFW context corruption that causes spurious warnings at exit.
  _execute_all_batches(batches, interactive=False)
  ```

---

## Rollback Plan

If this change causes unexpected issues:

1. **Revert commits:** `git revert [commit-hash]`
2. **Restore to default:** All parameters default to `interactive=True`, maintaining pre-fix behavior
3. **No data loss:** This is a UI-only change; no data or state is affected

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Parameter threading mistake** | Low | Medium | Review parameter names and call sites carefully; run tests on edge cases (empty batch, exception) |
| **Backward compatibility break** | Very Low | Low | Use default parameter `interactive=True` to preserve existing behavior if code calls `execute_frame_batch()` directly |
| **Mesh files not saved** | Very Low | High | Test that `.obj` files are created even with `show=False`; verify `output_path` is still passed through |
| **Interactive mode broken** | Low | Medium | Manually test single-session workflow with visualization enabled before merging |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 (Parameter threading) | 15 minutes | None |
| Testing | 20 minutes | Phase 1 |
| Documentation | 10 minutes | Testing |
| **Total** | **~45 minutes** | — |

---

## References

- **Bug report:** `docs/development/knowledge-base/bug-glfw-cleanup-forearm-batch-pipeline.md`
- **Affected script:** `code/scripts/preprocess_pipeline_extract_forearm_manual.py`
- **Mesh generation:** `code/scripts/_3_preprocessing/_3_forearm_extraction/define_forearm_mesh.py`

---

## Sign-Off

Ready for implementation review. Awaiting approval before proceeding with Phase 1.

