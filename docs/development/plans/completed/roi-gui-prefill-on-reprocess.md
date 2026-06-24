# Plan: ROI GUI pre-fill on reprocess

**Date:** 2026-04-14
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-04-17 15:34
**Branch:** `feature/roi-gui-prefill-on-reprocess`

---

## Overview

In the "extract forearm manual" pipeline, the ROI rectangle-drawing GUI
(`FrameROIRotatable`) should pre-fill the previously saved rotated
rectangle when a session is re-processed (`force_processing: true`), so the
user can **review-and-keep** or **review-and-tweak** prior work instead of
redrawing from scratch. Today the pre-fill path exists but silently shows a
wrong-shape rectangle because the save/reload round-trip collapses the
rotated rectangle to its axis-aligned bounding box and loses the original
centre/size. The fix persists the original centre-based rectangle alongside
the AABB.

## Problem Statement

The first task of `configs/preprocess_pipeline_extract_forearm_manual_dag.yaml`
(`extract_forearm`) drives two interactive GUIs via
`_annotate_forearm_roi()` in `code/scripts/preprocess_pipeline_extract_forearm_manual.py`:
the frame-group selector and the ROI rectangle drawer.

When `force_processing=true` and metadata from a previous run exists
(`<session_id>_arm_roi_metadata.json`), the user reports that the
previously-drawn rectangle does not appear as a starting point in the ROI
GUI — so they cannot quickly confirm prior work or make small adjustments.

Code path:

- `code/scripts/_3_preprocessing/_3_forearm_extraction/define_extraction_parameters.py`
  attempts to pre-fill via `_find_existing_roi_for_group()` (line ~163),
  which returns `{x, y, width, height, angle_deg}` derived from
  `RegionOfInterest.top_left_corner` / `bottom_right_corner` (the stored
  AABB).
- `FrameROIRotatable.__init__` (`code/src/preprocessing/common/gui/frame_roi_rotatable.py:68-89`)
  then applies `angle_deg` on top of that AABB — producing the rotated
  AABB, **not** the user's original rotated rectangle.

Root cause: the save path (`_build_forearm_parameters`, line ~228 of
`define_extraction_parameters.py`) discards the original centre-based
rectangle after computing the AABB. Information loss is structural, not a
branch bug — the `RegionOfInterest` dataclass has no field for it.

For `angle_deg ≈ 0` the pre-fill is correct by accident; for any meaningful
rotation the wrong shape makes "review and tweak" unusable, which is what
the user observes as "no prior data shown."

## Goals

### In Scope
1. Persist the original centre-based rotated rectangle (`cx`, `cy`,
   `width`, `height`) in the forearm ROI metadata alongside the existing
   AABB, so save → reload → GUI pre-fill is lossless.
2. Pre-fill the ROI drawing GUI with the exact prior rotated rectangle when
   `force_processing=true` (or when the DAG otherwise re-runs
   `extract_forearm`), allowing edit-in-place.
3. Keep existing metadata files (without the new fields) loadable and
   usable — fall back to the current AABB-based pre-fill in that case.

### Out of Scope
- Pre-filling the frame-group selector (`MultiVideoFramesSelector`) — the
  user confirmed only the ROI GUI is affected.
- Changing DAG force-processing semantics or `should_process_task`.
- Backfilling rotated rectangles for already-saved metadata files (the
  original centre data was never stored; legacy files correctly fall back
  to AABB pre-fill).
- Modifying the downstream 3D cuboid filter (`get_3d_cuboid_from_roi` in
  `extract_participant_forearm.py`) — it continues to read the AABB.

## Success Criteria

- [ ] `RegionOfInterest` persists `center_x`, `center_y`, `width`,
      `height` fields alongside `top_left_corner`, `bottom_right_corner`,
      `angle_deg`.
- [ ] A round-trip test (build → save → load → read) on a rotated
      rectangle (`angle_deg=30°`) preserves `cx, cy, width, height,
      angle_deg` exactly.
- [ ] `_find_existing_roi_for_group` returns a centre-format dict
      (`{cx, cy, width, height, angle_deg}`) when the new fields are
      present, and the legacy AABB-format dict otherwise.
- [ ] End-to-end manual run: after a first save with the new code,
      rerunning the pipeline with `force_processing: true` shows the
      previously-drawn rotated rectangle pre-drawn in the GUI and it is
      editable in place.
- [ ] Legacy metadata files (missing the new fields) still load and still
      open the GUI without error, falling back to today's AABB-based
      pre-fill behaviour.
- [ ] No changes required to `extract_participant_forearm.py` or any
      downstream extraction code — verified by greps and by a successful
      end-to-end run.

---

## Technical Design

### Approach

Extend `RegionOfInterest` with **optional** centre-based fields
(`center_x`, `center_y`, `width`, `height` — all `Optional[float]`,
default `None`). `_build_forearm_parameters` populates them from the raw
GUI output before computing the AABB. `_find_existing_roi_for_group`
prefers these fields when available and falls back to the current AABB
reconstruction otherwise. The JSON file handler serialises/deserialises
the new fields, treating missing fields as `None` on load
(backward-compatible with existing `_arm_roi_metadata.json`).

Downstream code keeps reading the AABB (`top_left_corner`,
`bottom_right_corner`) — no change required. `FrameROIRotatable` already
accepts both the centre-format (`cx`, `cy`, …) and corner-format (`x`,
`y`, …) pre-fill dicts (see `frame_roi_rotatable.py:68-89`), so the GUI
itself needs no change.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Add optional centre fields to `RegionOfInterest` (chosen) | Lossless round-trip; backward-compatible on load; no downstream changes; GUI already accepts both formats | Model gains four nullable fields; slight redundancy with AABB | **Chosen** |
| Replace AABB with centre+angle, recompute AABB at read-time | Single source of truth | Breaking change for all downstream readers (`get_3d_cuboid_from_roi`, saved JSONs); much larger blast radius | Rejected |
| Store a parallel `rotated_rect` sub-object | Cleaner separation | Extra nested type; more plumbing; forces a schema decision about which is canonical | Rejected |
| Solve in the GUI (derive original centre from AABB+angle) | Zero model change | Mathematically impossible without the original dimensions — AABB+angle is not invertible | Rejected |

### Architecture Changes

- **Model change** — extend `RegionOfInterest` dataclass.
- **Serialisation change** — handle the new optional fields in the JSON
  file handler with missing-field fallback.
- **Annotation script change** — populate the new fields on save; prefer
  them on reload.
- **No new modules, no new classes, no new abstractions.**

No integration surface outside the forearm-extraction package changes.

### Knowledge base relevance check

Reviewed `docs/development/knowledge-base/`:

- `note-azure-kinect-rgb-depth-parallax.md` / `note-kinect-depth-access-single-path.md`
  — not relevant (this change never touches raw depth).
- `bug-hue-circle-drag-not-working.md`, `note-qt-itemchanged-signal-recursion.md`
  — GUI-quirk notes but unrelated (ROI GUI is OpenCV, not Qt/Open3D).
- `bug-glfw-cleanup-forearm-batch-pipeline.md` — operates on the batch
  loop, not the annotation stage.
- Remaining notes (cupy, mesh, ICP, units) — unrelated.

No prior note constrains the chosen approach. No new note is required
(this is a model/serialisation refinement, not a reusable engineering
lesson).

---

## Implementation Plan

### Phase 1: Model + serialisation

**Goal:** Make `RegionOfInterest` capable of storing the original rotated
rectangle, and persist it through the JSON round-trip.

**Tasks:**
- [x] 1.1 — Add `center_x: Optional[float] = None`, `center_y: Optional[float] = None`,
  `width: Optional[float] = None`, `height: Optional[float] = None` to
  `RegionOfInterest` in `forearm_parameters.py`. Import `Optional` from `typing`.
- [x] 1.2 — Extend the JSON serialiser in
  `ForearmFrameParametersFileHandler` to emit the new fields when set.
- [x] 1.3 — Extend the JSON deserialiser to read the new fields when
  present and default to `None` otherwise (preserves backward compatibility
  with existing `_arm_roi_metadata.json` files).
- [x] 1.4 — Add a unit test (save → load) covering both the new path
  (centre fields set) and the legacy path (centre fields absent).

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/models/forearm_parameters.py`
  — add optional centre-based fields to `RegionOfInterest`.
- `code/src/preprocessing/forearm_extraction/data_access/forearm_frame_parameters_filehandler.py`
  — serialise/deserialise the new optional fields with missing-field
  fallback.
- `code/tests/` — new round-trip test (location aligned with existing
  forearm-extraction tests, e.g. alongside `test_ellipse_depth_extractor.py`).

**Dependencies:** None.

### Phase 2: Annotation save + reload

**Goal:** Populate the new fields on save; read them back for GUI pre-fill.

**Tasks:**
- [x] 2.1 — In `_build_forearm_parameters`
  (`define_extraction_parameters.py` line ~228), set
  `region_of_interest.center_x/center_y/width/height` from the raw GUI
  `roi_data` **before** collapsing to the AABB. The AABB computation
  remains unchanged.
- [x] 2.2 — In `_find_existing_roi_for_group` (line ~163), if the
  centre-based fields are present on the loaded `RegionOfInterest`, return
  `{cx, cy, width, height, angle_deg}`; otherwise return the current
  AABB-format dict (backward-compatible fallback).
- [x] 2.3 — Verify (grep + test run) that `FrameROIRotatable` correctly
  routes the centre-format dict through its existing
  `'cx' in predefined_roi` branch (lines 68-89).

**Files Modified:**
- `code/scripts/_3_preprocessing/_3_forearm_extraction/define_extraction_parameters.py`
  — populate and read the new centre-based fields.

**Dependencies:** Phase 1.

### Phase 3: End-to-end verification

**Goal:** Confirm the GUI pre-fill works on a real session, including the
legacy-metadata fallback.

**Tasks:**
- [ ] 3.1 — Run `preprocess_pipeline_extract_forearm_manual.py` against an
  existing session with legacy metadata (no centre fields). Confirm the
  GUI opens with AABB-based pre-fill (today's behaviour) and saves cleanly
  to the new format.
- [ ] 3.2 — Rerun the same session with `force_processing: true`. Confirm
  the previously-saved rotated rectangle is pre-drawn exactly, is editable
  in place, and saves correctly after an edit.
- [ ] 3.3 — Grep-verify no downstream code was affected:
  - `get_3d_cuboid_from_roi` in
    `extract_participant_forearm.py` still uses only
    `top_left_corner` / `bottom_right_corner` — no change needed.

**Files Modified:** None (verification only).

**Dependencies:** Phases 1 & 2.

---

## Testing Plan

### Unit Tests
- [ ] `RegionOfInterest` round-trip with centre fields populated —
      `cx, cy, width, height, angle_deg` preserved exactly.
- [ ] `RegionOfInterest` round-trip with centre fields absent (legacy JSON)
      — loads without error, centre fields read as `None`.
- [ ] `_find_existing_roi_for_group` returns centre-format dict when centre
      fields present; returns AABB-format dict when absent.

### Integration Tests
- [ ] Full `define_rois_for_frame_groups` call with `force_processing=True`
      and existing `saved_parameters` — verifies the correct
      `predefined_roi` is passed to `_select_roi_for_group`.

### Manual Verification
- [ ] Fresh session, draw a rotated rectangle (angle ~30°), save, set
      `force_processing: true`, rerun — the same rotated rectangle is
      pre-drawn and corner/angle handles work as expected.
- [ ] Legacy metadata (pre-change) — rerunning loads the AABB pre-fill
      (today's behaviour) without error; saving writes the new fields.

### Edge Cases
- [ ] `angle_deg == 0` — centre-format and AABB-format are geometrically
      equivalent; both paths produce the same rectangle.
- [ ] Extreme angles (e.g. 89°) and near-square rectangles — confirm no
      numerical surprises in the round-trip.
- [ ] Multiple groups per video where some have legacy metadata and some
      have new-format metadata in the same JSON — each group pre-fills
      correctly via its own branch.

---

## Documentation Plan

- [ ] Update inline docstring of `RegionOfInterest` to describe the new
      optional centre-based fields and when each set is authoritative.
- [ ] Update docstring of `_find_existing_roi_for_group` to document the
      centre-vs-AABB return shape.
- [ ] Add a short note to the JSON schema comment (top of the metadata
      file handler) describing the optional `center_*`/`width`/`height`
      fields.

No README.md, CLAUDE.md, user-guide, or changelog changes are required
(internal data-model refinement, no user-facing or convention change).

---

## Rollback Plan

1. **Before deployment:** All changes are local to the forearm-extraction
   package. Revert the commits on `feature/roi-gui-prefill-on-reprocess`.

2. **Data considerations:** New metadata files written with the new code
   contain extra optional fields. Older code paths that load them will
   simply ignore the unknown fields **if** the deserialiser uses a strict
   allow-list — confirm in Phase 1.3 that the existing handler tolerates
   extra keys; if not, include forward-compat behaviour (ignore unknown
   keys) in Phase 1 so rolled-back code still loads new-format files.

3. **Rollback procedure:**
   - Revert commits on the feature branch / delete the branch.
   - No database or filesystem cleanup required — new-format JSON files
     remain valid input for the rolled-back code (optional fields are
     simply ignored).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Deserialiser rejects new-format JSON when rolled back | Low | Medium | In Phase 1.3, verify the loader ignores unknown keys; if it doesn't, make it tolerant as part of this change |
| Downstream reader accidentally prefers centre fields over AABB | Low | High | Phase 3.3 grep check; no reader outside the annotation script is intended to touch centre fields |
| Multiple groups per video silently collide on representative-frame match | Low | Medium | Behaviour unchanged vs. today — matching logic in `_find_existing_params_for_group` is untouched |
| Rotated-rectangle numerical drift through JSON (float precision) | Very Low | Low | Assert equality within `1e-6` tolerance in the round-trip test |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 (Model + serialisation) | ~1h | None |
| Phase 2 (Annotation save + reload) | ~30m | Phase 1 |
| Phase 3 (Verification) | ~30m | Phases 1 & 2 |

---

## References

- DAG config: `configs/preprocess_pipeline_extract_forearm_manual_dag.yaml`
- Orchestrator: `code/scripts/preprocess_pipeline_extract_forearm_manual.py` (`_annotate_forearm_roi`, lines 306-353)
- Annotation: `code/scripts/_3_preprocessing/_3_forearm_extraction/define_extraction_parameters.py` (`_build_forearm_parameters` line ~228; `_find_existing_roi_for_group` line ~163)
- Model: `code/src/preprocessing/forearm_extraction/models/forearm_parameters.py`
- GUI: `code/src/preprocessing/common/gui/frame_roi_rotatable.py` (lines 68-89 already accept both pre-fill shapes)
- Downstream AABB reader (unchanged): `code/scripts/_3_preprocessing/_3_forearm_extraction/extract_participant_forearm.py` (`get_corners_from_roi`, `get_3d_cuboid_from_roi`)
