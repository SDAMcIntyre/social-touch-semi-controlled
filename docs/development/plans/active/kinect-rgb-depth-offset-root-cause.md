# Plan: Kinect RGB↔Depth Offset — Root-Cause Investigation

**Date:** 2026-04-14
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/kinect-rgb-depth-offset-root-cause`

---

## Overview

**What:** Extend the existing audit widget (`KinectRgbDepthViewer`) with a
paired-point Δ readout and a depth-source toggle, then run a short
empirical protocol to confirm whether the observed RGB↔depth offset is a
factory calibration defect of our Azure Kinect unit(s).

**Why:** The audit from
`docs/development/plans/active/kinect-rgb-depth-registration-audit.md`
surfaced a clearly visible >5 px RGB↔depth shift. The user has already
confirmed the shift appears on **stationary non-finger objects**, which
rules out temporal lag and finger-specific parallax. We now need to
quantify the offset's range-dependence before picking a remediation.

**How:** Two small additions to the existing viewer, a 5-feature ×
2-frame × ≥1-session measurement protocol, and a decision table that
routes the follow-up into one of three remediation paths.

## Problem Statement

- **Current limitation.** The Phase 4 "verdict" in the active audit plan
  cannot be written yet because we don't know *where* the offset comes
  from. Without range-dependence data we cannot choose between a global
  uniform shift (cheap), a per-device OpenCV recalibration (correct but
  costly), or documenting a tolerance band and moving on.
- **Why it matters.** Every downstream spatial pipeline (contact point,
  forearm registration, depth-weighted XYZ aggregation) inherits this
  offset. A wrong fix in the data-access layer would silently bias
  every subsequent analysis.
- **Supporting evidence from public trackers.** The symptom matches a
  well-documented Azure Kinect factory-calibration defect:
  - **microsoft/Azure-Kinect-Sensor-SDK #1058** — 0.75" (~19 mm)
    misalignment at 6–7 ft on a stationary cube. Microsoft's team:
    *"a few color pixels of transformation reprojection error could be
    expected"*. Range-dependent (worse at far distances). Workaround:
    custom OpenCV recalibration of color↔IR extrinsics.
  - **#1201** — dev team concluded *"we believe this is a hardware
    issue"* for a similar few-pixel misalignment; advised RMA.
  - **Depthkit verification procedure** — at 6 ft, quality bands are
    Good <0.5", Moderate 0.5–1", Bad ≥1". Our observed magnitude
    (>5 px) likely places us in Moderate.

## Goals

### In Scope

1. Paired-point Δ readout in `KinectRgbDepthViewer`: click RGB feature,
   click matching depth feature, display `(Δu, Δv, |Δ|, z_mm)` in a
   status bar.
2. Depth-source combobox toggling between
   `transformed_depth_point_cloud[:, :, 2]` and raw `transformed_depth`,
   to isolate whether any bug lives in our point-cloud wrapper.
3. Measurement protocol run on ST14-01 / block-order-07 and at least
   one other session if available; raw `(z, |Δ|)` table pasted into the
   active audit plan's Phase 4 section.
4. Decision-table outcome that selects one of three remediation paths
   (B1 / B2 / B3 below) as a *separate follow-up plan*.
5. Knowledge-base note
   `docs/development/knowledge-base/note-azure-kinect-factory-calibration-defect.md`
   recording symptom, upstream references, our measurements, and the
   chosen remediation.

### Out of Scope

- Any registration *fix* (checkerboard recalibration, uniform shift,
  tolerance-band documentation). Those are Phase-B work and will live
  in a separate plan conditioned on the Phase-A outcome.
- Any change to sticker tracking, xyz extractors, or aggregation
  viewers.
- Hardware RMA decisions. This plan only produces measurements to
  support that decision.
- Temporal lag audit — covered by
  `docs/development/plans/ideas/rgb-d-frame-lag-compensation.md`
  (explicitly ruled out by user observation for this issue).
- GPU / CuPy paths. Viewer is CPU-only.

## Success Criteria

- [ ] Paired-point Δ mode reports `(Δu, Δv, |Δ|, z_mm)` in the viewer's
      status bar for any two clicks.
- [ ] Depth-source combobox redraws the depth panel without reopening
      the MKV; both sources look geometrically identical on forearm
      geometry (or disagreement is flagged as a separate bug).
- [ ] At least 10 `(z, |Δ|)` measurements logged across ≥2 frames on
      the ST14-01 session, pasted into the active plan's Phase 4
      section.
- [ ] Active-plan verdict filled in with one of the four decision-table
      outcomes; knowledge-base note committed.
- [ ] No file under `code/src/preprocessing/stickers_analysis/` is
      modified by this plan.

---

## Technical Design

### Approach

Reuse the existing widget plumbing. The viewer already owns the
`KinectMKV` handle, both matplotlib axes, an imshow for depth, and a
click handler. Three small additions:

1. **Paired-point Δ mode.** Add a `QPushButton` "Measure offset"
   toggling a mode flag. In that mode, the existing
   `_on_canvas_click` branches: first click stashes `(u_rgb, v_rgb)`
   from `ax_rgb`; second click stashes `(u_depth, v_depth)` from
   `ax_depth`; status label shows `Δu, Δv, |Δ|` and `z` read from
   `transformed_depth_point_cloud[v_depth, u_depth, 2]`. Visual markers
   (one cross per axis) are drawn alongside the existing crosshair.
2. **Depth-source combobox.** A `QComboBox` with two items. The
   renderer `_render_depth_panel` is refactored to accept the chosen
   2D array instead of always slicing the point cloud. `transformed_depth`
   is already exposed at
   `code/src/preprocessing/common/data_access/kinect_mkv_manager.py:55`,
   so no data-access changes are needed.
3. **Status label.** A `QLabel` between the scrubber row and the
   Canny-controls row; shows paired-point readouts and a persistent
   `device | resolution | mode` summary.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Extend existing `KinectRgbDepthViewer` | Minimum code, minimum files, tool stays one launcher | Mixes two modes (crosshair + paired-point) in one widget | **Chosen** |
| New sibling widget `KinectRgbDepthOffsetMeter` | Clean separation | Duplicates the MKV handling, extent, and mouse wiring | Rejected |
| Jupyter notebook with ad-hoc clicks | Fastest to write | Not reusable; clicks are imprecise on static images | Rejected |
| Write Phase 4 verdict from eyeballed observation | Fastest | User has already requested root cause, not an opinion | Rejected |

### Architecture Changes

**Modified files (Phase A):**
- `code/src/preprocessing/common/gui/kinect_rgb_depth_viewer.py` —
  add paired-point Δ mode, depth-source combobox, status label. Target
  ~100 additional lines.

**Unchanged but referenced:**
- `code/src/preprocessing/common/data_access/kinect_mkv_manager.py` —
  already exposes both `transformed_depth` (line 55) and
  `transformed_depth_point_cloud` (line 62).
- `code/src/preprocessing/common/data_access/kinect_pointcloud_wrapper.py` —
  examined only if depth-source toggle reveals disagreement.

**New files (Phase A):**
- `docs/development/knowledge-base/note-azure-kinect-factory-calibration-defect.md`
  (written at the end of Phase A with measurements and chosen path).

---

## Implementation Plan

### Phase 1: Viewer instrumentation
**Goal:** Add paired-point Δ mode, depth-source combobox, and status
label to the existing audit widget.

**Tasks:**
- [x] 1.1 — Add a `QLabel` status bar to the widget layout (between
      scrubber and Canny rows).
- [x] 1.2 — Add a `QComboBox` for depth source; refactor
      `_render_depth_panel` to take a precomputed 2D array; wire the
      combobox to call `_display_frame(current)` on change.
- [x] 1.3 — Add a "Measure offset" `QPushButton` that toggles a
      `_measure_mode` flag. When on, left-clicks feed a 2-step state
      machine: (a) record RGB click, (b) record depth click, compute
      Δ, update status label. Draw a small "+" marker per click that
      clears on mode exit or on "Clear crosshair".
- [x] 1.4 — Read `z_mm = point_cloud[v_depth, u_depth, 2]` (handle
      NaN / zero gracefully in the label).

**Files Modified:**
- `code/src/preprocessing/common/gui/kinect_rgb_depth_viewer.py` —
  additions described in Technical Design.

**Dependencies:** None (builds on completed Phases 1–3 of the active
audit plan).

### Phase 2: Measurement protocol
**Goal:** Produce a `(z, |Δ|)` table on ST14-01 / block-order-07 and
≥1 other session (different Kinect unit if possible).

**Tasks:**
- [ ] 2.1 — Pick a frame with stationary geometry (table edge, board,
      rig element). Measure 5 features spanning near/mid/far depth.
- [ ] 2.2 — Repeat on a second frame in the same session; confirm the
      offset is time-invariant.
- [ ] 2.3 — Repeat on a second session. Note the device serial or
      recording date to identify units.
- [ ] 2.4 — Toggle depth-source combobox on a forearm frame; confirm
      the two sources are geometrically identical (else flag
      `kinect_pointcloud_wrapper.py` for a separate plan).
- [ ] 2.5 — Optional: Depthkit gridded-sheet test at 6 ft to place the
      device on the Good/Moderate/Bad scale.

**Files Modified:**
- `docs/development/plans/active/kinect-rgb-depth-registration-audit.md`
  — Phase 4 "Verdict" section filled with the measurement table and
  decision-table outcome.

**Dependencies:** Phase 1.

### Phase 3: Verdict, knowledge-base note, follow-up plan
**Goal:** Commit the finding and route the next fix to a new pending
plan.

**Tasks:**
- [ ] 3.1 — Write
      `docs/development/knowledge-base/note-azure-kinect-factory-calibration-defect.md`:
      symptom, upstream references (#1058, #1201, Depthkit), our raw
      measurements, chosen remediation path.
- [ ] 3.2 — Move the active audit plan to `completed/` once its
      verdict is in.
- [ ] 3.3 — Based on the decision table, create a new pending plan
      for Phase B (remediation). Do not start Phase B in this plan.

**Decision table routing Phase 3 → Phase B pending plan:**

| Observation | Interpretation | New pending plan |
|-------------|---------------|------------------|
| `|Δ|` grows with z (far points worse) | Factory miscalibration, #1058 profile | `kinect-opencv-color-ir-recalibration.md` (Phase B1) |
| `|Δ|` roughly constant across z | Uniform 2D offset | `kinect-uniform-offset-stopgap.md` (Phase B3) |
| `|Δ|` varies per session / unit | Per-device problem | `kinect-per-device-calibration.md` (Phase B1 per-device) |
| `transformed_depth` and point-cloud-Z disagree | Wrapper bug | `kinect-pointcloud-wrapper-bug.md` |
| Forearm edges align, finger offset alone | Reclassify as parallax — not a calibration issue after all | Knowledge-base note only; no remediation |

**Files Modified:**
- `docs/development/knowledge-base/note-azure-kinect-factory-calibration-defect.md`
  — new.
- `docs/development/plans/active/kinect-rgb-depth-registration-audit.md`
  → move to `completed/`.
- `docs/development/plans/pending/<chosen-follow-up>.md` — new.

**Dependencies:** Phase 2.

---

## Testing Plan

### Unit Tests

- [ ] Paired-point Δ computation: given synthetic clicks `(10, 20)` and
      `(14, 23)`, the status label reads `Δu=4, Δv=3, |Δ|≈5.0`.
- [ ] Depth-source toggle: switching the combobox re-renders the depth
      axis with a different backing array; `imshow` artist count
      remains 1.
- [ ] Measure-mode clear: exiting "Measure offset" removes the "+"
      markers from both axes.

### Integration Tests

- [ ] Smoke test: opening the launcher on a fixture MKV and toggling
      through Measure-mode, depth-source, Canny, and Clear does not
      raise. Skipped if no fixture MKV is available.

### Manual Verification

- [ ] On a forearm-only frame with a known edge, Canny edges align
      with the valid-depth silhouette under both depth sources.
- [ ] 5 features at distinct `z` values yield a consistent Δ direction
      across two frames (time-invariance).
- [ ] Second session produces similar |Δ| → rules out per-unit
      variation; or visibly different → confirms it.

### Edge Cases

- [ ] Depth click lands on NaN — status label shows `z = NaN`, not a
      crash.
- [ ] Only one click made, then mode toggled off — the "+" marker is
      removed and state resets.
- [ ] Depth-source combobox switched mid-measurement — measurement
      state resets to "awaiting RGB click".

---

## Documentation Plan

- [ ] Update
      `docs/development/plans/active/kinect-rgb-depth-registration-audit.md`
      Phase 4 section with measurement table + verdict.
- [ ] Create
      `docs/development/knowledge-base/note-azure-kinect-factory-calibration-defect.md`.
- [ ] Add upstream references (issues #1058, #1201, Depthkit) in the
      knowledge-base note.
- [ ] Inline docstring on the new `KinectRgbDepthViewer` methods.
- [ ] No CLAUDE.md change (no new project-wide convention).

---

## Rollback Plan

1. **Before deployment:**
   - Revert the widget changes in
     `code/src/preprocessing/common/gui/kinect_rgb_depth_viewer.py`.
   - Delete the knowledge-base note if it was committed.

2. **Data considerations:**
   - None. The tool is read-only against MKVs; it produces measurements
     pasted into a plan document, not data artefacts.

3. **Rollback procedure:**
   - `git revert` the feature branch commits. The active audit plan
     stays in `active/` until its verdict is written.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Click accuracy is the limiting factor — sub-pixel precision is impossible | High | Medium | Accept ±1 px per click; measure ≥5 features per frame and look at trends, not individual values. |
| All available sessions use the same Kinect unit — no way to confirm per-device variance | Medium | Low | Document as a known gap in the verdict; per-device variance can be checked later with any new recording. |
| `transformed_depth` and point-cloud-Z actually disagree — investigation pivots | Low | Medium | The decision table already routes this case to a separate plan against `kinect_pointcloud_wrapper.py`. |
| Observed offset disappears on stationary *geometry* and only survives on soft targets (skin) | Low | Medium | Protocol explicitly measures on rigid stationary geometry first; skin-only offset would reclassify back to a parallax / multipath issue. |
| The user upgrades pyk4a / Azure Kinect firmware mid-investigation | Low | High | Pin pyk4a version in the measurement log; re-run if updated. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 | 0.5 day | None |
| Phase 2 | 0.5 day (data-dependent) | Phase 1 |
| Phase 3 | 0.5 day | Phase 2 |

---

## References

- **Parent plan (active):**
  `docs/development/plans/active/kinect-rgb-depth-registration-audit.md`
- **Related ideas:**
  - `docs/development/plans/ideas/rgb-d-frame-lag-compensation.md`
  - `docs/development/plans/ideas/contact-point-forearm-spatial-discrepancy.md`
- **Upstream references:**
  - [Azure Kinect color-depth misalignment — Depthkit Docs](https://docs.depthkit.tv/docs/azure-kinect-alignment-verification/)
  - [Depth & color alignment inaccuracy — microsoft/Azure-Kinect-Sensor-SDK #1058](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/1058)
  - [Imperfect alignment in IR&depth to Color transformation — #1201](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/1201)
  - [Use Azure Kinect Sensor SDK image transformations — Microsoft Learn](https://learn.microsoft.com/en-us/azure/kinect-dk/use-image-transformation)
- **Data access:**
  `code/src/preprocessing/common/data_access/kinect_mkv_manager.py`
  (`KinectFrame.transformed_depth` line 55,
  `.transformed_depth_point_cloud` line 62)
- **Widget being extended:**
  `code/src/preprocessing/common/gui/kinect_rgb_depth_viewer.py`
- **Planning procedure:** `docs/development/planning-procedure.md`
