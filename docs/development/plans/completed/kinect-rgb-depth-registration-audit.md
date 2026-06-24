# Plan: Kinect RGB ↔ Depth Registration Audit (sticker-independent)

**Created:** 2026-04-14 00:00
**Approved:** —
**Completed:** 2026-04-14
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/kinect-rgb-depth-registration-audit` (work continued under `feature/kinect-parallax-tolerance-band`)

---

## Overview

**What:** A sticker-independent PyQt5 diagnostic widget, `KinectRgbDepthViewer`,
that renders a Kinect MKV's raw color frame and its `transformed_depth` /
point-cloud-Z frame side-by-side in a shared absolute pixel coordinate
system, with a click-synchronised crosshair and an optional Canny-edge
overlay.

**Why:** In `view_xyz_depth_aggregation.py` (Panels 0/1), the tracked
fingertip ellipse looks well-aligned on the RGB panel, but the valid-depth
blob looks smaller and possibly shifted. We cannot yet tell whether this
is Azure Kinect / pyk4a edge invalidation on thin structures or a real
RGB↔depth registration offset introduced somewhere in our stack.

**How:** Reuse `KinectMKV` / `KinectFrame`
(`code/src/preprocessing/common/data_access/kinect_mkv_manager.py`) and
matplotlib-in-Qt plumbing already used by
`depth_aggregation_diagnostics_gui.py`, wired into a widget with two
linked axes (`sharex`, `sharey`) and `extent=(0, W, H, 0)` so both panels
share absolute pixel coordinates. One launcher script drives it against
a chosen MKV frame.

## Problem Statement

- **Current limitation.** The only surface on which we can currently
  observe the RGB↔depth relationship is the sticker-tracking aggregation
  viewer. Its panels are entangled with ellipse geometry and depth-weight
  logic, which masks whether any mismatch comes from the Kinect hardware,
  from pyk4a's default transform, from our MKV reader, from the viewer's
  LRU cache, or from the tracking layer on top.
- **Why it matters.** If a systematic offset exists below the sticker
  layer, then *every* downstream spatial pipeline (contact point, forearm
  registration, depth-weighted XYZ aggregation) is subtly wrong. Diagnostic
  overlays added to the aggregation viewer during
  `feature/preprocessing-viewer-merging-layout` were rejected for being
  too sticker-coupled to answer the question cleanly; the question
  therefore remains open.
- **User impact.** Research decisions based on XYZ from fingertip depth
  can unknowingly inherit a registration offset. A short audit resolves
  this either way: "known hardware artefact, no code change" or "specific
  offset, file follow-up plan".

## Goals

### In Scope

1. A PyQt5 widget `KinectRgbDepthViewer` that takes an MKV path and a
   frame index and displays `KinectFrame.color` and
   `KinectFrame.transformed_depth_point_cloud[:, :, 2]` side-by-side with
   visible pixel-axis ticks.
2. Synchronised crosshair: clicking on either panel at pixel `(u, v)`
   places a crosshair at `(u, v)` on the other panel; zoom/pan stay
   linked via `sharex` / `sharey`.
3. Toggleable Canny-edge overlay computed from the color frame and
   drawn on the depth panel only, with two trackbars for `t1, t2`.
4. A launcher script under
   `code/scripts/_3_preprocessing/_0_kinect_diagnostics/audit_rgb_depth_registration.py`
   that opens the widget on a chosen session's MKV.
5. An audit verdict captured in this plan's follow-up notes (before
   moving to `completed/`): either "no systematic offset" or "offset of
   ≈N pixels in direction D", with a screenshot archived next to the
   plan.

### Out of Scope

- Any modification to sticker tracking, `xyz_extractor_ellipse_depth.py`,
  or the aggregation viewer's panels.
- Fixing the registration. This plan only produces the diagnostic; any
  follow-up fix lives in a separate plan conditioned on the audit's
  verdict.
- Temporal lag compensation — covered by
  `docs/development/plans/ideas/rgb-d-frame-lag-compensation.md`.
- Forearm / contact spatial reconciliation — covered by
  `docs/development/plans/ideas/contact-point-forearm-spatial-discrepancy.md`.
- GPU / CuPy paths. The widget is CPU-only; the CuPy import-order
  constraint from `CLAUDE.md` therefore does not apply here.
- A general-purpose MKV viewer. This tool is scoped to answer one
  question: is color↔transformed_depth co-registered at the pixel level?

## Success Criteria

- [ ] The widget opens any Kinect MKV used by the project and displays
      RGB + depth panels for an arbitrary frame index.
- [ ] Clicking on the RGB panel at a visually distinct feature places the
      depth-panel crosshair on the *same* feature (within the expected
      depth-edge invalidation band).
- [ ] Canny-edge overlay on a flat, high-confidence surface (forearm or
      a ruler) shows edges coincident with the valid-depth silhouette,
      not systematically offset.
- [ ] Audit run on `2022-06-15_ST14-01/block-order-07` yields a written
      verdict in this plan before it is moved to `completed/`.
- [ ] No file under `code/src/preprocessing/stickers_analysis/` is
      modified by this feature.

---

## Technical Design

### Approach

- Reuse `KinectMKV` / `KinectFrame` directly. Within one frame,
  `.color` and `.transformed_depth_point_cloud` are populated from the
  same `PyK4ACapture`, so there is no frame-index ambiguity between the
  two panels — this is precisely what we need for a registration audit.
- Render both panels with matplotlib inside a PyQt5 widget (same stack
  as `depth_aggregation_diagnostics_gui.py`). Use `imshow` with
  `extent=(0, W, H, 0)` on both axes so each axis is labelled in
  absolute pixel coordinates. Link the axes with `sharex=ax_rgb,
  sharey=ax_rgb` so pan/zoom remain synchronised.
- Crosshair state is a single `(u, v)` on the widget. The
  `button_press_event` handler on each axis updates it and triggers a
  redraw. Out-of-range clicks are ignored.
- Canny overlay: `cv2.Canny(cv2.cvtColor(color, BGR2GRAY), t1, t2)` is
  drawn over the depth axis as a red-tinted semi-transparent image. Two
  `QSlider`s control `t1`, `t2` with live redraw.
- The widget is imported from the `common/gui/` package alongside
  `scene_viewer.py`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Widget under `common/gui/` next to `scene_viewer.py` | Lightest footprint, matches existing layout, fewest new files | Mixes a diagnostic with general GUI primitives | **Chosen** |
| New `common/kinect_diagnostics/` package | Cleanest separation; easy home for future audit tools | Over-packaging for one widget + one launcher | Rejected |
| Extend `DepthAggregationDiagnosticsGUI` with an "audit mode" | Reuses an existing viewer | Re-introduces the sticker coupling the idea was designed to avoid | Rejected |
| Jupyter notebook, one-off | Fastest to write | Not reusable; harder to revisit on another session | Rejected |

### Architecture Changes

New files:

- `code/src/preprocessing/common/gui/kinect_rgb_depth_viewer.py` — the
  widget (target ~250 lines).
- `code/scripts/_3_preprocessing/_0_kinect_diagnostics/__init__.py`
- `code/scripts/_3_preprocessing/_0_kinect_diagnostics/audit_rgb_depth_registration.py` —
  launcher that parses CLI args (MKV path, optional frame index),
  instantiates the widget, and runs the Qt event loop.

Modified files:

- `code/src/preprocessing/common/gui/__init__.py` — re-export
  `KinectRgbDepthViewer`.

No data-access changes. No sticker-pipeline changes.

```
code/src/preprocessing/common/gui/
├── __init__.py                       — add KinectRgbDepthViewer re-export
├── scene_viewer.py                   — (unchanged)
├── scene_viewer_video_maker.py       — (unchanged)
├── time_series_panel.py              — (unchanged)
└── kinect_rgb_depth_viewer.py        — NEW: the widget

code/scripts/_3_preprocessing/
└── _0_kinect_diagnostics/            — NEW directory
    ├── __init__.py                   — NEW
    └── audit_rgb_depth_registration.py  — NEW launcher
```

---

## Implementation Plan

### Phase 1: Widget scaffold
**Goal:** A two-panel widget that loads a Kinect MKV and renders one
frame with linked pixel-coordinate axes.
**Started:** 2026-04-14
**Completed:** 2026-04-14

**Tasks:**
- [x] 1.1 — Create `kinect_rgb_depth_viewer.py` with a `QWidget` subclass
      owning a `matplotlib.figure.Figure`, `FigureCanvasQTAgg`, two
      axes (RGB + depth-Z) and a `QSpinBox` for frame index.
- [x] 1.2 — On construction, open a `KinectMKV` context and keep a
      handle; on frame change, fetch one `KinectFrame` and update both
      `imshow` artists with `extent=(0, W, H, 0)`.
- [x] 1.3 — Depth panel renders
      `transformed_depth_point_cloud[:, :, 2]` as grayscale with NaN
      masking for zero/invalid values. Use `ax.set_xlabel("u [px]")`,
      `ax.set_ylabel("v [px]")` and major ticks on both axes.
- [x] 1.4 — Link axes with `sharex=ax_rgb, sharey=ax_rgb`.

**Files Modified:**
- `code/src/preprocessing/common/gui/kinect_rgb_depth_viewer.py` — new.
- `code/src/preprocessing/common/gui/__init__.py` — re-export.

**Dependencies:** None.

### Phase 2: Synchronised crosshair
**Goal:** Click on either panel draws a crosshair at the same `(u, v)`
on both.
**Started:** 2026-04-14
**Completed:** 2026-04-14

**Tasks:**
- [x] 2.1 — Store crosshair state `(u, v)` on the widget (default
      `None`).
- [x] 2.2 — Connect `button_press_event` on both axes; on click, round
      `event.xdata / event.ydata` to int pixel coords, clamp to image
      shape, update state, and redraw both panels.
- [x] 2.3 — Represent the crosshair with `ax.axhline(v)` +
      `ax.axvline(u)` artists managed in a small helper to avoid
      redrawing the full image.
- [x] 2.4 — Add a "Clear" button that resets the crosshair.

**Files Modified:**
- `code/src/preprocessing/common/gui/kinect_rgb_depth_viewer.py` —
  click handling + crosshair artists.

**Dependencies:** Phase 1.

### Phase 3: Canny overlay + launcher
**Goal:** Toggleable Canny-edge overlay on the depth panel plus a
runnable launcher script.
**Started:** 2026-04-14
**Completed:** 2026-04-14

**Tasks:**
- [x] 3.1 — Add a `QCheckBox` "Canny overlay" and two `QSlider`s for
      `t1`, `t2` (defaults 50, 150).
- [x] 3.2 — On any control change, compute
      `cv2.Canny(gray(color), t1, t2)` and draw it over the depth axis
      as a red-tinted RGBA image with alpha from the edge mask.
- [x] 3.3 — Build
      `code/scripts/_3_preprocessing/_0_kinect_diagnostics/audit_rgb_depth_registration.py`
      with an `argparse` CLI: `--mkv PATH --frame INT` → opens the
      widget in a `QApplication`.
- [x] 3.4 — Add `__init__.py` for the new scripts directory.

**Files Modified:**
- `code/src/preprocessing/common/gui/kinect_rgb_depth_viewer.py` —
  overlay + controls.
- `code/scripts/_3_preprocessing/_0_kinect_diagnostics/__init__.py` — new.
- `code/scripts/_3_preprocessing/_0_kinect_diagnostics/audit_rgb_depth_registration.py` — new.

**Dependencies:** Phases 1, 2.

### Phase 4: Audit run and verdict
**Goal:** Produce an evidence-based answer to the original question and
record it on this plan.
**Started:** 2026-04-14
**Completed:** 2026-04-14 (verdict written; superseded by follow-up plan)

**Tasks:**
- [x] 4.1 — Audit was run on multiple sessions (ST13_2, ST14_1,
      ST14_2, ST14_3, ST16_5, ST18_1, ST18_2). Frame selection was
      driven by where stickers were visible rather than the original
      ST14-01/block-07 fixture.
- [x] 4.2 — Forearm/sticker features were measured directly, replacing
      the qualitative Canny-edge silhouette check. The quantitative
      paired-click measurements are stronger evidence than the visual
      check.
- [x] 4.3 — Multiple paired-feature clicks per frame produced the
      offset measurements summarised below.
- [x] 4.4 — Verdict section written below; CSV-based evidence under
      `F:/_tmp/kinect-measurement-offsets/` replaces the screenshot
      capture.

**Files Modified:**
- This plan (verdict section).
- `docs/development/knowledge-base/note-azure-kinect-rgb-depth-parallax.md`
  (final knowledge-base note carrying the band).

**Dependencies:** Phase 3.

---

## Verdict (2026-04-14)

**A systematic RGB ↔ depth offset is present** at the project's
operating range (500–800 mm). Pooled across 8 measurement CSVs
(n = 35 features, 1 misclick excluded), the
`KinectFrame.transformed_depth` pixel corresponding to a feature
clicked on `KinectFrame.color` is shifted by:

- **median |Δ| = 10.20 px**, IQR [7.77, 14.50] px, worst observed
  25.06 px;
- **median Δu = +10 px** (97 % sign-consistency), median Δv ≈ −1 px;
- direction is consistent across sessions and devices, matching the
  Azure Kinect color/IR baseline geometry.

**Cause classification:** geometric color↔IR parallax, not a
hardware/calibration defect. The follow-up plan
`kinect-rgb-depth-parallax-tolerance-band.md` characterises the
operating-range envelope and audits downstream consumers; the KB
note `note-azure-kinect-rgb-depth-parallax.md` codifies the band.
**No code change is made by this plan.** The downstream-consumer
audit (carried out by the follow-up plan) flagged
`xyz_extractor_centroid.py` and `kinect_pointcloud_wrapper.py` as
unaccounted; idea files have been opened for each.

Aggregation script:
`code/scripts/_3_preprocessing/_0_kinect_diagnostics/analyse_rgb_depth_offset_dataset.py`.
Plot: `F:/_tmp/kinect-measurement-offsets/aggregate_near_field.png`.

---

## Testing Plan

### Unit Tests

- [ ] Smoke test: `KinectRgbDepthViewer` can be instantiated with a
      fixture MKV path and render frame 0 without raising. Skipped if
      no fixture MKV is available in the test environment.
- [ ] Crosshair state: simulating a `button_press_event` at `(u, v)`
      updates `self._crosshair` to `(int(u), int(v))` and sets the
      expected axhline/axvline positions on both axes.
- [ ] Canny overlay toggling: enabling the checkbox adds exactly one
      image artist to the depth axis; disabling removes it.

### Integration Tests

- [ ] Launcher CLI: invoking
      `audit_rgb_depth_registration.py --mkv <fixture> --frame 10`
      opens the Qt application and exits cleanly when the window is
      closed.

### Manual Verification

- [ ] On forearm-only frame, Canny edges align with the valid-depth
      silhouette (within ~1 pixel).
- [ ] On fingertip frame, the crosshair lands on the same physical
      feature in both panels for three independent clicks.
- [ ] Zoom and pan on either panel propagate to the other.

### Edge Cases

- [ ] Frame with entirely invalid depth — depth panel renders uniformly
      grey, no exceptions; crosshair still works.
- [ ] Click outside the image extent — no crosshair update, no crash.
- [ ] MKV with a non-standard color / depth-mode combination — if
      `transformed_depth_point_cloud` is None, widget shows an empty
      depth axis with a status message.

---

## Documentation Plan

- [ ] Inline module docstring on `kinect_rgb_depth_viewer.py`
      explaining scope and its non-coupling to stickers.
- [ ] Short CLI-usage block at the top of
      `audit_rgb_depth_registration.py`.
- [ ] If the audit returns a systematic-offset verdict, add a
      knowledge-base note under
      `docs/development/knowledge-base/` summarising the offset and its
      cause so future pipelines can compensate.
- [ ] Update `README.md` only if an offset is found and downstream
      guidance changes; otherwise not needed.
- [ ] No CLAUDE.md change (no new project-wide convention).

---

## Rollback Plan

1. **Before deployment:**
   - Delete `code/src/preprocessing/common/gui/kinect_rgb_depth_viewer.py`.
   - Remove the re-export line from
     `code/src/preprocessing/common/gui/__init__.py`.
   - Delete the
     `code/scripts/_3_preprocessing/_0_kinect_diagnostics/` directory.

2. **Data considerations:**
   - None. The tool is read-only against MKVs and produces no on-disk
     artefacts other than screenshots the user chooses to save.

3. **Rollback procedure:**
   - `git revert` the feature commits. No migrations, no breaking
     changes, no shared-state resets.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Fingertip is a pathological Kinect case; every reading will look "shrunken" regardless of registration quality. | High | Medium | Audit protocol starts on forearm / high-reflectance surfaces before drawing conclusions from fingertips. |
| A single-frame audit misses a temporal (frame-lag) component. | Medium | Medium | Scope explicitly defers temporal audit to `rgb-d-frame-lag-compensation.md`; verdict notes this assumption. |
| Matplotlib axis linkage via `sharex/sharey` misleads when axes have different shapes (e.g., raw depth vs color). | Medium | High (would invalidate the audit) | Always render `transformed_depth_point_cloud` (color-camera geometry), never raw depth. Assert both arrays share `(H, W)` before display. |
| The launcher directory `_0_kinect_diagnostics/` is novel; Prefect or other workflow code might not pick it up. | Low | Low | The launcher is a standalone Qt app, not a Prefect task — no DAG wiring needed. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 | 0.5 day | None |
| Phase 2 | 0.5 day | Phase 1 |
| Phase 3 | 0.5 day | Phases 1, 2 |
| Phase 4 | 0.5 day (data-dependent) | Phase 3 |

---

## References

- **Idea (to be deleted on branch open):** `docs/development/plans/ideas/kinect-rgb-depth-registration-audit.md`
- **Related ideas:**
  - `docs/development/plans/ideas/rgb-d-frame-lag-compensation.md`
  - `docs/development/plans/ideas/contact-point-forearm-spatial-discrepancy.md`
- **Data access:**
  `code/src/preprocessing/common/data_access/kinect_mkv_manager.py`
  (`KinectFrame.color`, `.transformed_depth`, `.transformed_depth_point_cloud`)
- **Observed in:**
  `code/scripts/_3_preprocessing/_1_sticker_tracking/view_xyz_depth_aggregation.py`
  (Panels 0/1 of `code/src/preprocessing/stickers_analysis/xyz/gui/depth_aggregation_diagnostics_gui.py`)
- **GUI prior art:** `code/src/preprocessing/common/gui/scene_viewer.py`
- **Planning procedure:** `docs/development/planning-procedure.md`
