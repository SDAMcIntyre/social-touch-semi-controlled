# Plan: Fix postprocess visualization ICP transform type mismatch

**Date:** 2026-03-14
**Completed:** 2026-03-14
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/postprocessing-visualization-pipeline`

---

## Overview

Fix a runtime crash in the postprocessing visualization pipeline where `_load_icp_schedule()` passes tuple-format transform data to `get_transform_schedule()`, which expects raw JSON dict data. The mismatch causes `tuple indices must be integers or slices, not str` during session initialisation.

## Problem Statement

When launching the postprocessing visualiser on a session with ICP registration data, the flow fails with:

```
Failed to initialise session kinect_config_2022-06-14_ST13-01_semicontrolled_block-order01.yaml:
  tuple indices must be integers or slices, not str
```

The error originates from a data-format contract violation between two APIs:

- `ForearmCatalog.load_registration_transforms()` returns `Dict[str, Tuple[np.ndarray, float]]` — each value is a **tuple** `(matrix, fitness)`
- `get_transform_schedule()` accesses `entry["matrix_4x4"]` — expecting each value to be a **dict**

The `_load_icp_schedule()` function in `postprocess_visualization.py` bridges these two APIs without converting the format, causing the crash.

## Goals

### In Scope
1. Fix the type mismatch so postprocess visualization sessions initialise correctly
2. Remove unnecessary dependencies (`ForearmCatalog`, `ForearmFrameParametersFileHandler`) from `_load_icp_schedule()`

### Out of Scope
- Changing the return type of `ForearmCatalog.load_registration_transforms()` (other callers depend on tuple format)
- Changing the input format of `get_transform_schedule()` (other callers pass raw JSON dicts)

## Success Criteria

- [ ] Postprocess visualization flow initialises sessions with ICP data without errors
- [ ] ICP transforms are correctly applied in the viewer (hand transforms to PCA space)
- [ ] No regression in `merging_pipeline_neuron_to_kinect_visualisation.py` (uses tuple format)
- [ ] No regression in `apply_icp_registration.py` (uses raw JSON dict format)

---

## Technical Design

### Approach

Rewrite `_load_icp_schedule()` to load the registration transforms JSON file directly (as raw dicts), bypassing `ForearmCatalog.load_registration_transforms()`. This matches the established pattern used by `apply_icp_registration.py:89` which also passes raw JSON data to `get_transform_schedule()`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Load raw JSON in `_load_icp_schedule()` | Matches existing pattern in `apply_icp_registration.py`; removes unnecessary dependencies | Duplicates the file-path construction logic | **Chosen** |
| Change `get_transform_schedule()` to accept tuples | Single change point | Breaks `apply_icp_registration.py` which passes raw dicts; diverges from documented data flow | Rejected |
| Change `load_registration_transforms()` to return dicts | Aligns catalog with `get_transform_schedule()` | Breaks `merging_pipeline_neuron_to_kinect_visualisation.py` which unpacks tuples | Rejected |
| Add format conversion in `_load_icp_schedule()` | Minimal change | Wasteful: converts JSON→tuple→dict when we can just use JSON directly | Rejected |

### Architecture Changes

No new modules. Single function rewrite in an existing script.

### Knowledge Base Constraints

Per `note-forearm-icp-registration.md` (Section 7 — Data Flow): `get_transform_schedule()` is designed to consume raw JSON dict data directly. `ForearmCatalog.load_registration_transforms()` is a higher-level accessor that wraps data in tuples for caching/analysis use cases. These are separate pipeline stages and should not be mixed.

---

## Implementation Plan

### Phase 1: Fix `_load_icp_schedule()`
**Goal:** Eliminate the type mismatch by loading raw JSON directly

**Tasks:**
- [ ] Task 1.1 — Rewrite `_load_icp_schedule()` to open the `_registration_transforms.json` file directly and pass `raw["transforms"]` to `get_transform_schedule()`
- [ ] Task 1.2 — Remove `ForearmCatalog` and `ForearmFrameParametersFileHandler` imports if no longer used elsewhere in the file
- [ ] Task 1.3 — Remove now-unused `forearm_metadata_path` parameter from `_load_icp_schedule()` signature and update callers

**Files Modified:**
- `code/scripts/postprocess_visualization.py` — rewrite `_load_icp_schedule()` (lines 114-144), update imports (lines 51-52), update callers

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Run postprocess visualization flow on `kinect_config_2022-06-14_ST13-01_semicontrolled_block-order01.yaml` — session should initialise without error
- [ ] Verify ICP transforms are visually applied (hand mesh transforms to PCA space in viewer)
- [ ] Run on a session without registration data — should gracefully skip (return None)

### Edge Cases
- [ ] Missing `_registration_transforms.json` file — should return None, no crash
- [ ] Malformed JSON file — should catch exception and print warning

---

## Documentation Plan

- [ ] No external documentation changes needed (bug fix, no API changes)

---

## Rollback Plan

1. Revert the single commit modifying `postprocess_visualization.py`
2. No data migrations or breaking changes involved

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| File-path construction diverges from `ForearmCatalog` logic | Low | Med | Path pattern `{session_id}_registration_transforms.json` is simple and already used by the catalog; verify against `forearm_catalog.py:186` |
| Callers of `_load_icp_schedule()` pass stale `forearm_metadata_path` | Low | Low | Update all call sites when removing the parameter |

---

## References

- Knowledge base: `docs/development/knowledge-base/note-forearm-icp-registration.md` (Section 7 — Data Flow)
- Related pattern: `code/scripts/_5_postprocessing/apply_icp_registration.py:89` (correct raw-JSON usage)
- Error location: `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py:219,248,255`

---
