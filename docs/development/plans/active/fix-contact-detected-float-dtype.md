# Plan: Fix contact_detected Float Dtype

**Date:** 2026-03-19
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/fix-contact-detected-float-dtype`

---

## Overview

The `contact_detected` column in merged CSV files contains float values (e.g., `0.0`, `1.0`, `NaN`) instead of the intended integer `0`/`1`. This plan fixes the dtype coercion that occurs during the Kinect-to-nerve upsampling step and ensures integer output in the final CSV.

## Problem Statement

During the merging pipeline (`merge_neural_and_kinect_data.py`), Kinect data is upsampled to match nerve signal sampling rate. This creates a sparse DataFrame with NaN gaps. When `ffill()` is applied, pandas converts integer columns (including `contact_detected`) to float64 — because NaN is only representable in float dtype. Downstream consumers (analysis, feature extraction) then read `contact_detected` as float, which is semantically incorrect for a binary flag.

## Goals

### In Scope
1. Ensure `contact_detected` is written as integer `0`/`1` in merged CSV output
2. Apply forward-fill to `contact_detected` during upsampling (it is a state flag — the contact state persists between samples)
3. Cast `contact_detected` to int after forward-fill, before CSV write

### Out of Scope
- Changing `contact_detected` semantics (it remains binary 0/1)
- Fixing other columns that may also suffer from NaN-induced float coercion (e.g., `led_on` is already handled separately)
- Re-processing existing merged CSVs (users can re-run the merging pipeline)

## Success Criteria

- [ ] `contact_detected` column in newly generated merged CSVs contains only `0` and `1` (integer, no decimals)
- [ ] No `NaN` values remain in `contact_detected` after forward-fill
- [ ] Existing pipeline behavior is unchanged for all other columns

---

## Technical Design

### Root Cause

```
objects_interaction_processor.py  →  contact_detected = 0 or 1 (int)
        ↓ written to Kinect CSV
merge_neural_and_kinect_data.py   →  upsampling creates NaN gaps
        ↓ ffill() coerces int → float64
merged CSV                        →  contact_detected = 0.0, 1.0, or NaN
```

### Approach

After the forward-fill step in `merge_neural_and_kinect_data.py`, explicitly cast `contact_detected` (and `led_on`, which has the same issue in the `scaling_nofilling=False` path) to integer dtype. Use `fillna(0)` before casting to handle any remaining leading NaN values (frames before the first contact state is known).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Cast to int after ffill in merging script | Simple, targeted, fixes at the source | Only applies to new runs | **Chosen** |
| Use `pd.Int64` nullable integer dtype throughout | Handles NaN natively | Adds complexity, may break downstream `==` comparisons | Rejected |
| Fix at CSV write time with `float_format` | No code logic change | Masks the problem, doesn't fix in-memory dtype | Rejected |

### Architecture Changes

No new modules. Single targeted edit in `merge_neural_and_kinect_data.py`.

### Knowledge Base

No applicable notes. The somatosensory units note (`note-somatosensory-units-and-calculations.md`) documents `objects_interaction_processor.py` but not this dtype issue.

---

## Implementation Plan

### Phase 1: Fix dtype coercion
**Goal:** Ensure `contact_detected` is integer in the output DataFrame.

**Tasks:**
- [ ] Task 1.1 — After the `ffill()` block (line ~154), add explicit forward-fill and int cast for `contact_detected`:
  ```python
  kinect_scaled["contact_detected"] = kinect_scaled["contact_detected"].ffill().fillna(0).astype(int)
  ```
- [ ] Task 1.2 — Verify `led_on` similarly gets int treatment (it already has special handling at line 152, but confirm it doesn't remain float in the `scaling_nofilling=False` path)

**Files Modified:**
- `code/scripts/_4_merging/merge_neural_and_kinect_data.py` — Add dtype cast after ffill block (~line 155)

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Run the merging pipeline on a sample session
- [ ] Open the output CSV and confirm `contact_detected` contains only `0` and `1` (no `.0` suffix, no `NaN`)
- [ ] Confirm `led_on` column is also integer
- [ ] Run the downstream analysis pipeline to verify no breakage (comparisons like `contact_detected == 1` work identically for int and float, so no issues expected)

### Edge Cases
- [ ] Session where contact is never detected (all zeros) — should produce all `0`, no NaN
- [ ] Session where the first few upsampled frames have no source value (leading NaN before first ffill) — `fillna(0)` handles this

---

## Documentation Plan

- [ ] No documentation changes needed (internal pipeline fix)

---

## Rollback Plan

1. Revert the single commit on the feature branch
2. No data migration needed — re-run merging pipeline to regenerate CSVs

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Downstream code relies on float dtype of contact_detected | Low | Low | `== 1` and `== 1.0` are equivalent in pandas; no code uses `.dtype` checks |
| Leading NaN filled with 0 is incorrect (contact was actually happening) | Very Low | Low | First Kinect frames before any detection are always no-contact; 0 is correct default |
