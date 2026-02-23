# Plan: Explicit `remap_lowest_to_zero` parameter in `get_forearms_with_fallback`

**Date:** 2026-02-23
**Status:** Shipped

---

## Overview

**What:** Add an explicit `remap_lowest_to_zero` keyword parameter to `get_forearms_with_fallback()`.
**Why:** The lowest `representative_frame_id` key is currently silently remapped to `0` — this is intentional for the somatosensory pipeline but implicit and non-obvious to callers.
**How:** Add a boolean keyword parameter (default `False`), update the existing call site to pass `True`.

## Problem Statement

In `get_forearms_with_fallback()`, the lowest dict key is unconditionally remapped to `0` before returning. This forces the first forearm reference to be active from frame 0 of the hand-motion sequence. The behaviour is correct for the somatosensory pipeline, but it is hidden — callers cannot opt out, and reading the function signature gives no hint that keys will be mutated.

## Goals

### In Scope
1. Add `remap_lowest_to_zero: bool = False` keyword parameter to `get_forearms_with_fallback()`
2. Update the single existing call site to pass `True` to preserve current behaviour

### Out of Scope
- Changing `ObjectsInteractionController` frame-iteration logic
- Changing `ForearmCatalog` internals

## Success Criteria

- [ ] `get_forearms_with_fallback(..., remap_lowest_to_zero=False)` returns keys unchanged
- [ ] `get_forearms_with_fallback(..., remap_lowest_to_zero=True)` remaps lowest key to 0 (current behaviour)
- [ ] Existing call site in `compute_somatosensory_characteristics.py` passes `True` — runtime behaviour unchanged

---

## Technical Design

### Approach

Wrap the existing key-remapping block in a conditional on the new parameter. Default to `False` (no mutation) so the function is transparent by default; the somatosensory call site opts in explicitly.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Keyword parameter (default `False`) | Explicit, backward-compatible at call site, zero risk | Requires updating one call site | **Chosen** |
| Remove remapping entirely | Simpler function | Would change runtime behaviour, needs deeper controller audit | Rejected |

### Architecture Changes

None — signature-level change only.

---

## Implementation Plan

### Phase 1 (single phase)

**Goal:** Make the key-remapping opt-in.

**Tasks:**
- [ ] Add `remap_lowest_to_zero: bool = False` to `get_forearms_with_fallback()` signature
- [ ] Wrap lines 244–248 in `if remap_lowest_to_zero:`
- [ ] Update call in `compute_somatosensory_characteristics.py` to pass `remap_lowest_to_zero=True`
- [ ] Grep for any other call sites and update if needed

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/models/forearm_catalog.py` — add parameter, conditionalise remapping
- `code/scripts/_3_preprocessing/_4_somatosensory_quantification/compute_somatosensory_characteristics.py` — pass `remap_lowest_to_zero=True`

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Grep all call sites of `get_forearms_with_fallback` — confirm none are missed
- [ ] Read modified function to confirm `False` skips remapping and `True` preserves it

---

## Rollback Plan

Revert the two file edits — the change is purely additive (new keyword with default preserving old external behaviour for any hypothetical callers not passing the flag).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Missed call site | Low | Med | Grep verification step |
