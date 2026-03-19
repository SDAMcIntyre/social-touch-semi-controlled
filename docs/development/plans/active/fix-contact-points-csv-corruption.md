# Plan: Fix contact_points CSV Corruption

**Date:** 2026-03-19
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/fix-contact-detected-float-dtype` (extends existing branch)

---

## Overview

The `contact_points` column in `_contact_and_kinematic_data.csv` files stores Python list-of-lists whose comma-separated `str()` representation collides with CSV column delimiters, causing data to spill across columns and rows. This plan fixes the serialization to use a CSV-safe, space-separated format that downstream parsers already expect.

## Problem Statement

In `objects_interaction_processor.py`, contact points are stored as `contact_points.tolist()` — a nested Python list like `[[76.0, 61.19, 617.14], [75.0, 62.0, 123.45]]`. When pandas writes this to CSV via `to_csv()`, the commas inside the list representation are interpreted as column delimiters, splitting the data across multiple columns and eventually wrapping to subsequent rows, corrupting the entire CSV structure.

Additionally, `empty_structure()` omits `contact_points` from the data dict, so no-contact frames get `NaN` instead of the expected `"[]"` string.

**User impact:** Generated CSV files are unreadable — every row with contact detected becomes corrupted, breaking all downstream processing (merging, postprocessing, analysis, visualization).

## Goals

### In Scope
1. Serialize `contact_points` to a CSV-safe string format at the source
2. Ensure no-contact frames produce `"[]"` instead of `NaN`
3. Maintain backward compatibility with all downstream parsers

### Out of Scope
- Re-processing existing corrupted CSVs (users re-run the preprocessing pipeline)
- Changing the `[[x y z] [x y z]]` format convention (it is already the project standard)
- Adding CSV quoting changes (the fix eliminates the need for special quoting)

## Success Criteria

- [ ] `contact_points` column in newly generated CSVs contains properly formatted strings (`[[x y z] ...]` or `[]`)
- [ ] No commas appear inside `contact_points` cell values
- [ ] No-contact frames show `[]` (not `NaN`)
- [ ] All downstream consumers (postprocessing, analysis, viewers) process the files without errors
- [ ] In-memory visualization pipeline (Open3D) continues to receive raw numpy arrays

---

## Technical Design

### Root Cause

```
objects_interaction_processor.py
  → contact_points.tolist()           # Python list: [[76.0, 61.19, 617.14], ...]
  → stored in DataFrame cell           # object dtype, raw list
  → to_csv() calls str()              # "[[76.0, 61.19, 617.14], [75.0, 62.0, ...]]"
  → commas clash with CSV delimiter    # cell splits across columns/rows
```

### Approach

Serialize `contact_points` to the numpy-style string format `[[x y z] [x y z]]` (space-separated, no commas) at the point of creation in the processor. Reuse the existing `serialize_contact_points()` utility from `csv_spatial_transformer.py`, which already produces this exact format and is used throughout the postprocessing pipeline.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Serialize at source using existing utility | Clean, DRY, consistent with rest of pipeline | Cross-package import (forearm_extraction → motion_analysis) | **Chosen** |
| Serialize at CSV write time in `compute_somatosensory_characteristics.py` | Keeps processor output as raw data | Adds column-specific logic to a generic script; fragile | Rejected |
| Add `quoting=csv.QUOTE_ALL` to `to_csv()` | No format change | Masks the problem; still stores commas; Excel may still misbehave | Rejected |
| Inline the 2-line serializer | No cross-package import | Duplicates code; divergence risk | Rejected |

### Architecture Changes

No new modules or classes. Single file modified. The `serialize_contact_points` function is already exported from `preprocessing.forearm_extraction` and used in `csv_spatial_transformer.py:283,367` with the identical pattern.

### Knowledge Base

No directly applicable notes. `note-somatosensory-units-and-calculations.md` documents the processor but not serialization.

---

## Implementation Plan

### Phase 1: Fix serialization (single phase)
**Goal:** Eliminate CSV corruption by pre-serializing contact_points to string.

**Tasks:**
- [x] Task 1.1 — Add import `from preprocessing.forearm_extraction import serialize_contact_points` to `objects_interaction_processor.py`
- [x] Task 1.2 — Replace `contact_points.tolist()` (line 178) with `serialize_contact_points([tuple(pt) for pt in contact_points])`
- [x] Task 1.3 — Add `contact_data["contact_points"] = "[]"` to `empty_structure()` (after line 224)

**Files Modified:**
- `code/src/preprocessing/motion_analysis/tactile_quantification/model/objects_interaction_processor.py` — 3 edits: import, serialize in contact case, add to empty case

**Dependencies:** None

---

## Downstream Compatibility Audit

All 12 files referencing `contact_points` were audited. Every parser handles both comma-separated and space-separated formats (via `replace(",", " ").split()`). No downstream changes needed:

| Consumer | Type | Compatible |
|----------|------|-----------|
| `csv_spatial_transformer.py` (parse/serialize/transform) | Postprocessing | Yes |
| `rf_data_loader.py` (parse) | Analysis | Yes |
| `project_contacts_onto_forearm.py` (parse→project→serialize) | Postprocessing | Yes |
| `set_xyz_reference_from_gestures.py` (parse→PCA→serialize) | Postprocessing | Yes |
| `postprocessed_scene_viewer.py` (parse for GUI) | Visualization | Yes |
| `neural_kinect_scene_viewer.py` (parse for GUI) | Visualization | Yes |
| `objects_interaction_visualizer.py` (raw ndarray) | In-memory viz | N/A (uses viz dict) |
| `debug_visualiser.py` (raw ndarray) | In-memory viz | N/A (uses viz dict) |

---

## Testing Plan

### Manual Verification
- [ ] Re-run `compute_somatosensory_characteristics` on a session
- [ ] Open the output `_contact_and_kinematic_data.csv` in Excel — confirm `contact_points` stays in its column with no spill
- [ ] Verify no-contact rows show `[]` in the `contact_points` column
- [ ] Run the downstream postprocessing pipeline (`apply_registration_transform`, `project_contacts_onto_forearm`) to confirm no errors
- [ ] Open a merged CSV in the neural-kinect viewer to confirm visualization works

### Edge Cases
- [ ] Frame with a single contact point — `[[x y z]]`
- [ ] Frame with many contact points (100+) — long string but no commas
- [ ] Frame with no contact — `[]`
- [ ] Session where no frames have contact — all rows show `[]`

---

## Documentation Plan

- [ ] No documentation changes needed (internal pipeline fix, format convention unchanged)

---

## Rollback Plan

1. Revert the commit on the feature branch
2. No data migration — re-run preprocessing to regenerate CSVs

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Cross-package import creates coupling | Low | Low | Function is a pure string utility; already exported publicly |
| Existing corrupted CSVs not automatically fixed | N/A | Med | Users must re-run preprocessing; document in commit message |
| `tuple(pt)` conversion fails on edge data | Very Low | Low | `contact_points` is always a numpy `(N,3)` float array; conversion is safe |
