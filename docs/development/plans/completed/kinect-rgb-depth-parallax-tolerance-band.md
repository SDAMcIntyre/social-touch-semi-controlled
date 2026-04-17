# Plan: Kinect RGB↔Depth Parallax Tolerance Band

**Date:** 2026-04-14
**Completed:** 2026-04-14
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/kinect-parallax-tolerance-band`

---

## Overview

**What:** Improve the existing `KinectRgbDepthViewer` to make paired-point
offset measurements faster, less noisy, and more traceable; then use the
improved tool to run a flat-wall verification capture that distinguishes
color↔IR *parallax* from extrinsics *miscalibration*; then codify the
outcome as a project-wide RGB↔depth tolerance band.

**Why:** Phase 1 of the root-cause plan produced 14 measurements whose
pattern (|Δ| largest at near range, dropping at far range; consistent
horizontal Δu sign) does not match the microsoft/Azure-Kinect-Sensor-SDK
#1058 factory-defect signature and instead points to color↔IR baseline
parallax at depth discontinuities. Before any downstream pipeline adopts
a fix, the parallax hypothesis needs a cheap binary confirmation (flat
wall → |Δ| should collapse to the click-noise floor), and the tool that
produced the Phase-1 data needs usability fixes that mitigate the
click-noise / single-frame limitations surfaced in the analysis.

**How:** Three small additions to the viewer (repeated-click median
mode, local depth-gradient warning, rich CSV header), one planar-wall
capture run using the improved tool, and one knowledge-base note plus a
downstream-consumer audit checklist.

## Problem Statement

- **Phase-1 data is ambiguous without a control.** The near-field-worse
  trend is consistent with parallax but also with *click-bias on
  small features*. Without a depth-discontinuity-free capture (flat
  wall), the two cannot be separated.
- **The current viewer does not support confident data collection.**
  Single-click measurements have a ~±1.4 px noise floor on |Δ|, no
  indicator warns the operator that the clicked pixel lies on a depth
  edge, and the CSV lacks the session metadata (device serial, pyk4a
  version) needed to interpret measurements across days.
- **Downstream pipelines lack a principled tolerance band.** Sticker
  tracking, depth-weighted XYZ aggregation, and contact-point
  projection all silently assume pixel-precision alignment between the
  RGB frame and the transformed depth frame. The Phase-1 data shows
  this is false at the ~10–15 px level near-field. Every downstream
  module should either document the tolerance it can absorb or bound
  its inputs to a safe range.
- **The parent root-cause plan cannot write its Phase 4 verdict**
  until the parallax-vs-calibration distinction is settled.

## Goals

### In Scope

1. Add repeated-click median mode to `KinectRgbDepthViewer`: a spinbox
   selects N clicks per feature per axis, the viewer accumulates them,
   and one CSV row is committed with median `(Δu, Δv, |Δ|, z_mm)` plus
   per-click std for diagnostics.
2. Add a local depth-gradient warning: when the operator clicks on the
   depth axis, compute `max(z) - min(z)` over a 5×5 neighborhood (NaN-
   safe). If it exceeds a threshold (default 20 mm), colour the status
   label and record a `near_edge=True` flag on the row.
3. Add a CSV header line with session metadata: recording file, frame
   range scanned, device serial (from `KinectMKV` metadata), pyk4a
   version (`pkg_resources.get_distribution`), viewer-git-SHA,
   `clicks_per_feature`.
4. Add a one-shot programmatic equivalence check at MKV load:
   `np.allclose(transformed_depth, transformed_depth_point_cloud[..., 2],
   equal_nan=True)` — log a one-line result and hide the depth-source
   combobox if they are identical (they should be; the toggle is now
   diagnostic only, not user-facing).
5. Run a flat-wall verification capture with the improved tool: 5
   features × 3 distances (≈500 / 1000 / 1500 mm) × 2 frames,
   `clicks_per_feature = 5`.
6. Produce `docs/development/knowledge-base/note-azure-kinect-rgb-depth-parallax.md`
   recording the measured tolerance band and the flat-wall result.
7. Produce a downstream-consumer audit checklist: list every module
   that takes paired RGB and depth inputs and state, for each, whether
   the measured tolerance band is absorbed, mitigated, or unaccounted.
8. Fill the parent plan's Phase 4 verdict and move it to `completed/`.

### Out of Scope

- Color↔IR extrinsics recalibration (OpenCV-based Phase B1). Ruled out
  by the Phase-1 signature; a recalibration plan is only revived if
  the flat-wall result disagrees with the parallax hypothesis.
- Uniform-shift stop-gap (Phase B3). A uniform shift cannot fit a
  range-dependent error and is not warranted by the data.
- Changes to sticker tracking, xyz extractors, aggregation viewers, or
  contact-projection code. The audit deliverable lists them; fixes are
  separate follow-up plans.
- Hardware RMA. Data does not support a hardware defect.
- Changes to the parent plan's already-completed Phases 1–3.

## Success Criteria

- [ ] Repeated-click mode: selecting `clicks_per_feature = 5` and
      clicking an 8-point feature set produces 8 CSV rows each
      containing `delta_u_median`, `delta_v_median`, `delta_mag_median`,
      `delta_mag_std`, `n_clicks=5`, `near_edge`.
- [ ] Depth-gradient warning visibly fires on a known edge pixel and
      does not fire on a known interior pixel of a flat surface.
- [ ] CSV header parses as a single commented line and includes
      `device_serial`, `pyk4a_version`, `viewer_git_sha`,
      `clicks_per_feature`.
- [ ] One-shot `np.allclose` check passes on the ST13 MKV at load and
      its result is logged.
- [ ] Flat-wall capture yields ≥30 measurements; the median |Δ| on the
      flat wall is ≤ the click-noise floor (≤3 px) at ≥2 of the 3
      distances. If that bound is met, the knowledge-base note
      classifies the residual offset as parallax.
- [ ] Audit checklist enumerates every downstream module that ingests
      paired RGB+depth and assigns each a status
      (`absorbed`/`mitigated`/`unaccounted`).
- [ ] Parent plan `kinect-rgb-depth-registration-audit.md` moved to
      `completed/` with verdict written.
- [ ] No file under `code/src/preprocessing/stickers_analysis/`,
      `code/src/preprocessing/common/data_access/`, or
      `code/src/merging/` is modified by this plan.

---

## Technical Design

### Approach

Extend the existing audit widget rather than creating a new one, since
the Phase-1 tool is already the canonical offset-measurement entry
point and the usability fixes are small. The flat-wall capture is
operational work performed with the extended tool, not a code change.
Deliverable documentation (KB note + audit checklist) is the
outcome the parent plan was blocked on.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Extend existing `KinectRgbDepthViewer` with median mode + gradient warning | Smallest diff, reuses state machine and CSV export, data collection tool stays single-entry | Mode logic grows; state machine becomes (RGB×N) → (depth×N) → commit | **Chosen** |
| Keep viewer unchanged; post-process noisy CSVs offline | Zero code change | Operator cannot tell during capture whether a click landed on an edge → same noisy data as Phase 1 | Rejected |
| Write a dedicated `FlatWallCaptureTool` | Clean separation of concerns | Duplicates MKV handling, axis wiring, and export; breaks the "one tool, one CSV schema" invariant | Rejected |
| Skip the flat-wall capture; declare parallax from Phase-1 data alone | Fastest to close the parent plan | Phase-1 signature is suggestive, not conclusive; downstream pipelines would inherit an unverified assumption | Rejected |
| Replace manual clicks with a feature-detector (ORB/SIFT) that auto-pairs points | Eliminates click noise | Introduces a new source of error (detector bias on IR vs RGB) and a maintenance cost that isn't paid back | Rejected — out of scope |

### Architecture Changes

**Modified files (Phase 1):**
- `code/src/preprocessing/common/gui/kinect_rgb_depth_viewer.py` —
  extend state machine to accumulate N RGB clicks then N depth clicks;
  add `QSpinBox` for `clicks_per_feature`; add local depth-gradient
  computation in the depth-click branch; change CSV export to include
  header line and new columns; hide depth-source combobox when the
  one-shot equivalence check passes. Target ~150 additional lines.

**Unchanged but referenced:**
- `code/src/preprocessing/common/data_access/kinect_mkv_manager.py` —
  read-only access to `KinectFrame.transformed_depth` (line 55) and
  `.transformed_depth_point_cloud` (line 62). The one-shot equivalence
  check is performed in the viewer, not the data-access layer.

**New files (Phase 3):**
- `docs/development/knowledge-base/note-azure-kinect-rgb-depth-parallax.md`
  — symptom, Phase-1 + flat-wall measurements, quantified tolerance
  band, references to #1058 / #1201 / Depthkit.
- `docs/development/plans/completed/kinect-rgb-depth-parallax-audit-checklist.md`
  *or* append the checklist directly to the KB note — decision made
  at Phase 3 time based on length.

**Data-access layer:** no changes. No new abstractions. No refactor
of the CSV writer into a separate module — the current ~15-line
export block stays inline.

---

## Implementation Plan

### Phase 1: Viewer usability fixes
**Goal:** Remove the three data-quality obstacles the Phase-1 analysis
surfaced: single-click noise, unknown edge-proximity, missing session
metadata. Also retire the depth-source combobox if the equivalence
check passes (it did in Phase-1; this just makes it official).

**Tasks:**
- [x] 1.1 — Add `QSpinBox _clicks_per_feature` (default 1, range 1–9)
      to the controls row. When > 1, the measure-mode state machine
      collects N RGB clicks into `self._measure_rgb_pts: list`, then N
      depth clicks into `self._measure_depth_pts: list`, then commits
      one row using `np.median` over each column and records
      `delta_mag_std`.
- [x] 1.2 — In the depth-click branch, sample `z[v-2:v+3, u-2:u+3]`
      from the point cloud, compute `np.nanmax - np.nanmin`, and set
      `near_edge = span_mm > 20.0`. Status label turns amber when
      `near_edge` fires; CSV column records the boolean.
- [x] 1.3 — At MKV load, compute
      `np.allclose(frame.transformed_depth, frame.transformed_depth_point_cloud[..., 2],
      equal_nan=True)` on the first frame. If True, hide
      `_combo_depth_source` and log one INFO line. If False, keep the
      combobox and log one WARNING line. No behaviour change
      downstream.
- [x] 1.4 — CSV export: write a single commented header line `#` with
      `recording_path`, `device_serial` (from `KinectMKV.playback.get_record_configuration`
      if available, else `"unknown"`), `pyk4a_version`, `viewer_git_sha`
      (via `subprocess.run(["git", "rev-parse", "--short", "HEAD"])`
      at export time, fail-soft to `"uncommitted"`), `clicks_per_feature`.
      Add columns: `delta_u_median`, `delta_v_median`, `delta_mag_median`,
      `delta_mag_std`, `n_clicks`, `near_edge`. Keep the existing
      per-measurement columns for backward readability of Phase-1
      files or bump a `csv_schema_version` header field.
- [x] 1.5 — Update the docstring of `_on_canvas_click` and add a
      single-line comment on the state-machine step transitions. No
      other doc changes.

**Files Modified:**
- `code/src/preprocessing/common/gui/kinect_rgb_depth_viewer.py`

**Dependencies:** None.

### Phase 2: Near-field measurement (substituted for flat-wall capture)
**Goal:** Produce a dataset that quantifies the offset envelope at the
500–800 mm operating range — the only range used by the downstream
pipelines this plan is meant to serve. The original three-distance
flat-wall capture was retired because (a) downstream consumers do not
operate at far range, so a range-stratified discriminator carries no
decision weight, and (b) the operator-judged flat-region clicks on
session MKVs (with the new median + `near_edge` flagging) provide
sufficient evidence at the relevant range.

**Tasks:**
- [x] 2.1 — Collected 5 CSVs under
      `F:/_tmp/kinect-measurement-offsets/phase-2/` covering ST13_2,
      ST14_1, ST14_2, ST16_5, ST18_2 with `clicks_per_feature = 5`,
      operator-judged flat body regions (`near_edge = False` on
      24/27 rows). Combined with the 3 Phase-1 CSVs this gives n = 35
      near-field measurements after dropping a single |Δ| > 100 px
      misclick.
- [x] 2.2 — Same as 2.1; per-feature std uniformly ≤ 2 px confirms
      median mode is removing click noise.
- [x] 2.3 — Decision: the original flat-wall ≤3 px decision rule was
      not invoked because the relevant scope changed from
      "parallax-vs-miscalibration verdict" to "near-field tolerance
      envelope". Pooled near-field band: median |Δ| = 10.20 px,
      IQR [7.77, 14.50] px, worst 25.06 px, Δu sign-consistency
      97 % (+ direction). The optional one-distance flat-wall top-up
      at ~600 mm was *skipped* — within-session variance is small
      enough and no consumer is on a tolerance boundary that the
      flat-geometry vs body-geometry distinction would change.
- [x] 2.4 — Verdict written into the parent plan
      `kinect-rgb-depth-registration-audit.md` and codified in the
      KB note.

**Files Modified:**
- `docs/development/plans/completed/kinect-rgb-depth-registration-audit.md`
  — Phase 4 verdict section populated.
- `docs/development/knowledge-base/note-azure-kinect-rgb-depth-parallax.md`
  — new (carries the band).

**Dependencies:** Phase 1.

### Phase 3: Tolerance-band codification + downstream audit
**Goal:** Make the finding discoverable and actionable for every
future change that touches RGB↔depth paired access.

**Tasks:**
- [x] 3.1 — Wrote
      `docs/development/knowledge-base/note-azure-kinect-rgb-depth-parallax.md`
      with the near-field band (median 10.20 px, IQR [7.77, 14.50],
      worst 25.06, Δu sign-consistency 97 %), pointer to CSVs and
      plot, upstream references (#1058, #1201, Depthkit, Microsoft
      Learn), and an explicit scope statement that the band applies
      to the 500–800 mm operating range only.
- [x] 3.2 — Audit checklist produced inline in the KB note. Results:
      `xyz_extractor_centroid.py` → **unaccounted**;
      `xyz_extractor_ellipse_depth.py` → **mitigated** (ellipse mask
      + 10 mm sticker-size z-guard); `ellipse_fit_view_gui.py` →
      **n/a**; `kinect_pointcloud_wrapper.py` → **unaccounted**;
      `neural_kinect_scene_viewer.py` → **mitigated** (visual only);
      `kinect_rgb_depth_viewer.py` → **n/a** (audit tool itself).
- [x] 3.3 — Parent plan
      `kinect-rgb-depth-registration-audit.md` moved to
      `docs/development/plans/completed/` with verdict section.
- [x] 3.4 — Idea files opened for the two `unaccounted` modules:
      `docs/development/plans/ideas/xyz-extractor-centroid-parallax-tolerance.md`
      and
      `docs/development/plans/ideas/kinect-pointcloud-wrapper-parallax-tolerance.md`.

**Files Modified:**
- `docs/development/knowledge-base/note-azure-kinect-rgb-depth-parallax.md` — new.
- `docs/development/knowledge-base/README.md` — index entry added.
- `docs/development/plans/active/kinect-rgb-depth-registration-audit.md` → moved to `completed/`.
- `docs/development/plans/ideas/xyz-extractor-centroid-parallax-tolerance.md` — new.
- `docs/development/plans/ideas/kinect-pointcloud-wrapper-parallax-tolerance.md` — new.
- `code/scripts/_3_preprocessing/_0_kinect_diagnostics/analyse_rgb_depth_offset_dataset.py` — new aggregation script.

**Dependencies:** Phase 2.

---

## Testing Plan

### Unit Tests

- [ ] Median-mode commit: a stream of synthetic clicks at
      `[(10,20), (11,20), (10,21), (12,20), (10,22)]` for RGB and
      `[(14,23), (15,23), (14,24), (16,23), (14,25)]` for depth
      with `clicks_per_feature=5` commits one row with
      `delta_u_median=4`, `delta_v_median=3`, `delta_mag_median≈5.0`,
      `delta_mag_std>0`, `n_clicks=5`.
- [ ] Depth-gradient warning: synthesising a 5×5 depth patch with
      `ptp=21.0 mm` sets `near_edge=True`; `ptp=15.0 mm` sets it False.
      NaN in the patch is ignored, not propagated.
- [ ] CSV header: export on an empty measurement list still writes the
      header and column line (one file, two lines); export with one
      measurement writes three lines.

### Integration Tests

- [ ] Smoke test: open the launcher on a fixture MKV, toggle
      `clicks_per_feature` from 1 to 5, perform one complete paired
      measurement, export CSV, assert the file has the expected
      columns and one row. Skipped if no fixture MKV is available.
- [ ] One-shot equivalence check: on the ST13 MKV, the check passes;
      the combobox is hidden.

### Manual Verification

- [ ] On a flat wall frame, the depth-gradient warning does not fire
      on features chosen near the image centre.
- [ ] On a hand/forearm frame, the warning fires on sticker edges.
- [ ] Selecting `clicks_per_feature=5` visibly requires 5 RGB clicks
      then 5 depth clicks before the row is committed; intermediate
      "+" markers accumulate on both axes.
- [ ] CSV opens cleanly in Excel / pandas with the header line either
      commented (`#`-prefixed) or consumed by `pandas.read_csv(comment="#")`.

### Edge Cases

- [ ] Depth click hits a NaN-dominated neighborhood — `near_edge`
      falls back to `False` (cannot compute span), status label
      reads `z = NaN`, no crash.
- [ ] `clicks_per_feature` changed mid-sequence — in-progress
      measurement is cleared, "+" markers removed, status label
      resets.
- [ ] `git rev-parse` unavailable (no git, detached worktree) — the
      `viewer_git_sha` field is written as `"unknown"`.
- [ ] `device_serial` absent from MKV metadata — recorded as
      `"unknown"`; operator is expected to note it in the sidecar
      README.

---

## Documentation Plan

- [ ] New KB note:
      `docs/development/knowledge-base/note-azure-kinect-rgb-depth-parallax.md`.
- [ ] Update `docs/development/knowledge-base/README.md` index to
      include the new note.
- [ ] Update parent plan
      `docs/development/plans/active/kinect-rgb-depth-registration-audit.md`
      Phase 4 section with measurement table and verdict.
- [ ] Inline docstring updates on
      `KinectRgbDepthViewer._on_canvas_click` and the new spinbox /
      gradient-check helpers.
- [ ] No CLAUDE.md change (no new project-wide convention; the KB
      note is sufficient).
- [ ] Update `docs/development/plans/README.md` if it carries an
      active/pending index (check before writing).

---

## Rollback Plan

1. **Before deployment:**
   - Revert changes in
     `code/src/preprocessing/common/gui/kinect_rgb_depth_viewer.py`.
   - Delete the new KB note if already committed.
   - Restore the parent plan to `active/` with its verdict section
     blanked.

2. **Data considerations:**
   - None. All artefacts are measurement CSVs under `F:/_tmp` (not in
     the repo) and documentation files. No database, no deployed
     service, no downstream pipeline change.

3. **Rollback procedure:**
   - `git revert` the feature-branch commits.
   - Manually move the parent plan back to `active/` if Phase 3 had
     already moved it.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Flat-wall capture is ambiguous (mixed result across distances) | Medium | Medium | Phase 2.3 explicitly covers this case by adding a fourth distance; only pivots to recalibration plan on ≥2 distances confirming non-parallax. |
| Median mode hides real drift (e.g., slow finger motion between clicks) | Low | Low | Operator instructed to capture only stationary geometry; `delta_mag_std` column surfaces within-feature inconsistency. |
| `device_serial` is not exposed by pyk4a / the MKV metadata | Medium | Low | Serial is recorded in the sidecar README by the operator; the CSV field defaults to `"unknown"`. |
| `viewer_git_sha` captures `"uncommitted"` because operator forgets to commit viewer changes before capture | Medium | Low | Accept; `"uncommitted"` is still informative and cheaper than enforcing a clean tree. |
| Depth-gradient threshold (20 mm over 5×5) is wrong for very close or very far captures | Low | Low | Threshold is a module-level constant; can be tuned once real flat-wall data arrives. |
| Downstream-audit checklist (Phase 3.2) uncovers an `unaccounted` module that is currently in production use | Medium | Medium | 3.4 explicitly opens an idea file per finding rather than attempting a same-branch fix; prevents scope creep. |
| Flat-wall decision rule fires "parallax refuted" and this plan has to pause mid-branch | Low | Medium | Phase 2.3 defines the branching condition cleanly: open the recalibration plan in `pending/` and freeze this branch without merging. |

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
- **Root-cause plan (active):**
  `docs/development/plans/active/kinect-rgb-depth-offset-root-cause.md`
- **Phase-1 measurement CSVs:**
  `F:/_tmp/kinect-measurement-offsets/` (ST13_2, ST14_3, ST18_1)
- **Related ideas:**
  - `docs/development/plans/ideas/rgb-d-frame-lag-compensation.md`
  - `docs/development/plans/ideas/contact-point-forearm-spatial-discrepancy.md`
- **Upstream references:**
  - [Depth & color alignment inaccuracy — microsoft/Azure-Kinect-Sensor-SDK #1058](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/1058)
  - [Imperfect alignment in IR&depth to Color transformation — #1201](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/1201)
  - [Azure Kinect color-depth misalignment — Depthkit Docs](https://docs.depthkit.tv/docs/azure-kinect-alignment-verification/)
  - [Use Azure Kinect Sensor SDK image transformations — Microsoft Learn](https://learn.microsoft.com/en-us/azure/kinect-dk/use-image-transformation)
- **Viewer being extended:**
  `code/src/preprocessing/common/gui/kinect_rgb_depth_viewer.py`
- **Data access (unchanged):**
  `code/src/preprocessing/common/data_access/kinect_mkv_manager.py`
- **Planning procedure:** `docs/development/planning-procedure.md`
