# Plan: RF Inflection Basin — Morphological Opening

**Date:** 2026-05-20
**Author:** Basil Duvernoy
**Status:** Draft
**Base Branch:** `feature/inflection-boundary-extrapolation-snapshots`
**Branch:** `feature/rf-inflection-basin-opening`

---

## Overview

Add an optional morphological-opening step to the negative-Laplacian
flood-fill basin used by the population-RF inflection-boundary detector,
so that thin finger-like protrusions in the basin mask are removed before
the contour is extracted. Controlled by a new `basin_opening_iterations`
parameter (default `0` = disabled, preserves current behaviour) threaded
from the DAG YAML down to `compute_inflection_boundary`.

## Problem Statement

The detector in
`code/src/analysis/receptive_field_mapping/metrics/rf_inflection_boundary.py`
produces an inflection contour by flood-filling the connected
negative-Laplacian region containing the IFF peak. Because the
connectivity used by `scipy.ndimage.label` is 4-connected and the
binarization is a strict `lap < 0` threshold, narrow necks of
weakly-concave pixels can fuse the main basin to small secondary
patches. The marching-squares contour then traces a thin appendage out
to the secondary patch and back.

A concrete example is the `stroke_distal` gesture for session
`2022-06-14_ST13-03`, where the step-4 contour PNG shows a clear
downward-pointing finger on the lower part of the contour.

This biases the geometric metrics:

- **Perimeter** ↑ (long thin extensions add lots of perimeter for
  little area).
- **Circularity** ↓ (4π·area/perimeter² heavily penalises non-compact
  shapes).
- **PCA orientation** can swing if the protrusion is long enough.
- **Centroid** drifts slightly toward the protrusion.

Increasing `gaussian_sigma` would suppress these necks, but at the cost
of enlarging the whole basin and risking merging into neighbouring
structure. A morphological opening surgically removes thin appendages
without changing the bulk of the basin.

## Goals

### In Scope

1. Add `basin_opening_iterations: int = 0` parameter to
   `compute_inflection_boundary`.
2. Apply `scipy.ndimage.binary_opening` to the selected basin mask
   between component selection and `find_contours`, using the default
   scipy cross structuring element (matches the 4-connectivity used
   for labelling).
3. Preserve both raw and opened basin masks and render them
   side-by-side in the Step-3 diagnostic snapshot
   (`inflection_*_step3_basin.png`) so users can verify the chosen
   iteration count is appropriate.
4. Thread the parameter through
   `rf_population_map_pipeline.run_population_rf_maps` (both
   per-gesture and composite call sites).
5. Expose the parameter in
   `configs/analyse_workflow_processing_dag.yaml` under
   `visualize_population_rf_maps.options`, defaulting to `0` so
   existing outputs are unaffected.
6. Add focused unit tests covering: default-off behaviour,
   protrusion removal at `iterations=1`, and the
   over-aggressive case where opening eats the whole basin.

### Out of Scope

- Replacing the `< 0` Laplacian threshold with a sub-zero level
  (`< -ε`) — separable change, can be added later if necks persist
  after opening.
- Polyline smoothing of the extracted contour (Chaikin / spline) —
  cosmetic concern, orthogonal to basin shape correction.
- Making the structuring-element shape configurable (disk vs cross) —
  always 4-connected cross for now.
- Re-tuning the existing `gaussian_sigma` default.

## Success Criteria

- [ ] `compute_inflection_boundary` accepts `basin_opening_iterations`
      and defaults to `0`.
- [ ] When `basin_opening_iterations == 0`, all current unit tests
      pass unchanged and existing snapshot PNGs are visually
      identical (regression).
- [ ] When `basin_opening_iterations == 1` on a synthetic basin with
      a single-pixel neck, the resulting contour has measurably lower
      perimeter and higher circularity than with `0`.
- [ ] Step-3 snapshot is a two-panel figure showing raw vs opened
      basin, with the iteration count in the suptitle.
- [ ] Re-running `visualize_population_rf_maps` on session
      `2022-06-14_ST13-03` with `basin_opening_iterations: 1`
      eliminates the downward protrusion seen in
      `inflection_stroke_distal_step4_contour.png`.
- [ ] DAG YAML carries the new option and is loaded by the Prefect
      flow without error.

---

## Technical Design

### Approach

Apply `scipy.ndimage.binary_opening(mask, iterations=N)` to the
peak-component mask in `_select_peak_basin_contour` between component
selection (line 163) and contour extraction (line 165). The default
scipy structuring element is the 4-connected cross, which is the same
connectivity used at line 157 to build the component — so a single
iteration removes pixels whose only connection to the basin is through
a 1-pixel-wide isthmus.

After opening, re-validate:
1. The peak is still inside the opened mask (it could be eroded if the
   basin is tiny and iterations is large).
2. The longest contour of the opened mask still has ≥ 4 points.

If either check fails, return `None` for the boundary but still hand
back the raw mask so the diagnostic snapshot can be rendered to show
what happened.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Morphological opening on basin mask (this plan) | Surgically removes thin appendages without changing main basin; controllable per session | Adds one tunable; requires snapshot upgrade | **Chosen** |
| Increase `gaussian_sigma` | Already implemented; one knob | Enlarges entire basin; risks merging with neighbours | Rejected — wrong-scale solution |
| Sub-zero Laplacian threshold (`< -ε`) | Removes weakly-concave fringe pixels including necks | Needs careful ε normalisation across sessions; more global effect than necessary | Deferred — option 3 in follow-up work |
| Polyline smoothing of contour | Cosmetically clean | Doesn't fix the underlying basin shape; metrics still biased | Rejected — addresses symptom not cause |
| Largest-area component filter post-opening | Cheap, handles disconnection if opening splits basin | Adds complexity for an unlikely edge case (opening rarely splits) | Deferred until observed |

### Architecture Changes

- **No new modules** — change is contained within
  `metrics/rf_inflection_boundary.py` (logic), with parameter threading
  through `pipelines/rf_population_map_pipeline.py` and a single new
  key in the DAG YAML.
- **One internal function signature change**:
  `_select_peak_basin_contour` returns
  `(contour, opened_mask, raw_mask)` instead of `(contour, mask)`.
  This function is private (`_`-prefixed) and called only from
  `compute_inflection_boundary` in the same file — no external break.
- **Snapshot renderer extension**: `_render_step_basin` becomes a
  two-panel layout. Existing single-panel callers do not exist (the
  function is private to this module).
- **Parameter naming**: `basin_opening_iterations` (pipeline-level and
  YAML-level). Internal `_select_peak_basin_contour` uses the shorter
  `opening_iterations`. This mirrors the existing
  `inflection_sigma`/`gaussian_sigma` naming pair.

---

## Implementation Plan

### Phase 1: Core logic + snapshot upgrade
**Goal:** Add morphological opening inside the boundary detector and
make the diagnostic step-3 snapshot a two-panel view.

- [ ] Add `opening_iterations: int = 0` parameter to
      `_select_peak_basin_contour`.
- [ ] Apply `scipy.ndimage.binary_opening` between component
      selection and `find_contours`; skip when `opening_iterations <= 0`.
- [ ] Preserve raw mask; re-check peak containment and contour size
      against the opened mask.
- [ ] Change return signature to `(contour, opened_mask, raw_mask)`.
      Return raw mask even on failure paths.
- [ ] Convert `_render_step_basin` to two-panel layout (raw |
      opened); include iteration count in suptitle; show peak `rx`
      marker on both panels; render `FAILED` overlay when opened mask
      lost the peak.
- [ ] Extend `_save_inflection_snapshots` signature with
      `component_mask_raw` and `opening_iterations` parameters.
- [ ] Add public `basin_opening_iterations: int = 0` parameter to
      `compute_inflection_boundary` with docstring entry. Wire it
      through `_select_peak_basin_contour` and both
      `_save_inflection_snapshots` calls (success and failure paths).

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/metrics/rf_inflection_boundary.py`
  — all changes confined to this file.

**Dependencies:** None.

### Phase 2: Pipeline threading + DAG exposure
**Goal:** Make the new parameter reachable from the DAG config.

- [ ] Add `basin_opening_iterations: int = 0` to
      `run_population_rf_maps` signature; document under `Parameters`.
- [ ] Pass through to both `compute_inflection_boundary` call sites:
      per-gesture pass (~line 267) and composite pass (~line 369).
- [ ] Add `basin_opening_iterations: 0` under
      `visualize_population_rf_maps.options` in the DAG YAML.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_map_pipeline.py`
  — signature + two call sites.
- `configs/analyse_workflow_processing_dag.yaml` — one new option key.

**Dependencies:** Phase 1.

### Phase 3: Tests
**Goal:** Lock in the new behaviour with focused unit tests.

- [ ] `TestBasinOpening::test_opening_zero_matches_default` —
      regression guard: a clean circular Gaussian with the parameter
      omitted produces identical area/perimeter to the call with
      `basin_opening_iterations=0`.
- [ ] `TestBasinOpening::test_opening_removes_thin_protrusion` —
      synthetic two-peak landscape connected by a single-pixel ridge;
      check that `iterations=1` lowers perimeter and raises
      circularity vs `iterations=0`.
- [ ] `TestBasinOpening::test_opening_too_aggressive_returns_none` —
      small basin, `iterations=5`, expect `None` return and a warning
      log entry mentioning the peak being eroded.
- [ ] `TestSnapshotOutput`-style file-existence check: step-3 PNG
      is produced for both `iterations=0` and `iterations=1` runs.

**Files Modified:**
- `code/tests/test_rf_inflection_boundary.py` — one new test class.

**Dependencies:** Phase 1.

---

## Testing Plan

### Unit Tests

- [ ] `test_opening_zero_matches_default` — `basin_opening_iterations=0`
      reproduces baseline metrics on a circular Gaussian.
- [ ] `test_opening_removes_thin_protrusion` — synthetic basin with
      single-pixel neck; perimeter strictly lower and circularity
      strictly higher at `iterations=1`.
- [ ] `test_opening_too_aggressive_returns_none` — small basin +
      `iterations=5` → `None` returned, raw mask still passed to
      snapshot renderer.
- [ ] Step-3 snapshot file exists for both `iterations=0` and
      `iterations=1`.

### Integration Tests

- [ ] All existing tests in `test_rf_inflection_boundary.py` pass
      unchanged (no behaviour change at default).
- [ ] Running the population RF maps Prefect flow end-to-end on a
      tiny synthetic session produces snapshots with the new step-3
      layout and no errors when the option is set in YAML.

### Manual Verification

- [ ] In `analyse_workflow_processing_dag.yaml`, set
      `basin_opening_iterations: 1`. Run
      `visualize_population_rf_maps` against session
      `2022-06-14_ST13-03` with `force_processing: true`. Confirm:
      - `inflection_stroke_distal_step3_basin.png` shows the
        protrusion in the left (raw) panel and a clean basin in the
        right (opened) panel.
      - `inflection_stroke_distal_step4_contour.png` shows a contour
        without the downward finger.
- [ ] Same session with `basin_opening_iterations: 2` — confirm the
      right panel is visibly over-eroded relative to the left,
      validating the two-panel diagnostic value.
- [ ] Pick a clean session (no protrusion in current outputs), rerun
      with the default `basin_opening_iterations: 0`, and confirm
      step-4 and step-5 PNGs are visually identical to the prior
      outputs.

### Edge Cases

- [ ] Peak within 2 pixels of basin edge — opening must not erode
      across the peak.
- [ ] Basin is exactly the size of the structuring element — opening
      with `iterations >= 1` removes everything; expect
      well-handled `None` return.
- [ ] All-NaN or flat Laplacian input — should short-circuit before
      reaching the opening step (existing guards at lines 593-604).

---

## Documentation Plan

- [ ] Update the docstring of `compute_inflection_boundary` to
      describe the new parameter and recommend `1` as a starting
      value when protrusions are visible in step-3 snapshots.
- [ ] Update the docstring of `run_population_rf_maps` in
      `rf_population_map_pipeline.py` with the new parameter.
- [ ] Add a one-line comment near the new YAML key explaining its
      effect (consistent with surrounding `inflection_sigma` comment
      style if any).
- [ ] No README/CLAUDE.md changes needed — this is a tunable inside
      an already-documented task.
- [ ] No changelog entry required (small enhancement, no breaking
      change).

---

## Rollback Plan

The change is opt-in by default (`basin_opening_iterations: 0`
preserves existing behaviour bit-for-bit), so a rollback only matters
if the new code introduces a defect when the parameter is non-zero or
if the snapshot layout change breaks downstream consumers.

1. **Before deployment:**
   - Verify default-off behaviour via the regression unit test and
     the clean-session manual check above.
   - Keep the change in a dedicated feature branch off
     `feature/inflection-boundary-extrapolation-snapshots`; merge
     only after manual verification on the problem session.

2. **Data considerations:**
   - No migrations. Snapshot PNGs are regenerated on every run; old
     PNGs from prior runs remain on disk under prior filenames.
   - The vertex-data NPZ artefact format is unchanged.

3. **Rollback procedure:**
   - Revert the merge commit on
     `feature/inflection-boundary-extrapolation-snapshots`.
   - Remove the new YAML key from
     `configs/analyse_workflow_processing_dag.yaml`.
   - No code state to restore beyond the two files
     (`rf_inflection_boundary.py`,
     `rf_population_map_pipeline.py`) and the tests file.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Opening removes the peak in an unusually small basin, breaking boundary detection | Medium | Low | Re-check peak containment after opening; return `None` with a logged warning. Default `0` means opt-in. |
| Two-panel step-3 snapshot is misinterpreted (e.g. user confuses raw/opened) | Low | Low | Panel titles are explicit (`Basin (raw)` vs `Basin (after opening)`); suptitle carries iteration count. |
| Opening on the binary mask happens to split the basin into two components when iterations > 1 | Low | Medium | Marching squares returns the longest contour anyway; impact is at most that we pick the larger of two pieces. If this becomes a problem, add a largest-area filter (deferred). |
| Performance regression on large grids | Very Low | Low | `binary_opening` is O(N·iterations) and basins are ≤ 150×150; runtime is negligible compared to the surrounding Laplacian computation. |
| Breaking the private `_select_peak_basin_contour` return signature affects another caller | Very Low | Medium | Function is `_`-prefixed and called from one place in the same module — verified during exploration. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 (core + snapshot) | ~1–2 hours | None |
| Phase 2 (threading + YAML) | ~30 minutes | Phase 1 |
| Phase 3 (tests) | ~1 hour | Phase 1 |

Total: half a day's work.

---

## References

- Source file: `code/src/analysis/receptive_field_mapping/metrics/rf_inflection_boundary.py`
- Caller: `code/src/analysis/receptive_field_mapping/pipelines/rf_population_map_pipeline.py`
- DAG config: `configs/analyse_workflow_processing_dag.yaml`
- Tests: `code/tests/test_rf_inflection_boundary.py`
- Triggering example: session `2022-06-14_ST13-03`, gesture
  `stroke_distal`, file
  `inflection_stroke_distal_step4_contour.png`.
- Related plan: `docs/development/plans/completed/cluster-based-rf-mapping.md`
  (original receptive-field-mapping work).
- Earlier predecessor work on this branch: commit `b497caa`
  (replaced basin dilation with extrapolation; introduced the
  current step-3 snapshot).

---
