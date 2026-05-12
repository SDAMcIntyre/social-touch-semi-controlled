# Plan: Lock Hand-Mesh Scale to Session Median

> **Superseded** by `docs/development/plans/pending/stabilise-handmesh-pose.md` (2026-05-12).
> Both scale locking and rotation smoothing are merged into a single post-generation
> pipeline step to ensure translation is always re-derived from the anchor constraint.

**Created:** 2026-05-12 22:00
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** Draft
**Base Branch:** `fix/contact-detection-false-positives`
**Branch:** `fix/lock-handmesh-scale`

---

## Overview

The Procrustes alignment in `HandMotionManager` computes a per-frame uniform
scale factor from 3 noisy XYZ sticker anchors. Since the hand does not
physically change size, all scale variation is depth-sensor noise. This plan
locks the scale to a single robust estimate (session median) computed after
all frames are processed, and recomputes per-frame translations so that the
anchor sticker vertex still maps exactly to its measured 3D position.

## Problem Statement

`_calculate_alignment_procrustes()` derives scale each frame from a
least-squares ratio of rotated-source to target dot products. Even after the
upstream 6 Hz Butterworth filter on sticker XYZ, residual depth noise
propagates into this ratio, causing the hand mesh to visibly "breathe" in
size across rendered frames. There is currently no mechanism to enforce
constant scale.

## Goals

### In Scope
1. Lock all per-frame scales to the session median of valid scale values.
2. Recompute per-frame translations to preserve anchor-sticker alignment
   after the scale change.
3. Filter out invalid scales (non-positive or outside plausible range) before
   computing the median.
4. Log raw scale diagnostics (min, max, std, n_valid) before locking.
5. Fail-fast if no valid scales remain (upstream data quality problem).

### Out of Scope
- Temporal smoothing of rotation or translation (handled by
  `smooth-handmesh-rotation.md`).
- Changing the upstream sticker Butterworth filter cutoff.
- Per-subject MANO mesh calibration or rigid (no-scale) Procrustes.
- New pipeline steps or DAG config entries — this is a correction to the
  existing `generate_3d_hand_in_motion` step.

## Success Criteria

- [ ] All entries in the NPZ `scales` array are identical after generation.
- [ ] The CSV `scale` column is constant for a given session.
- [ ] Hand mesh no longer visibly changes size between frames in overlay videos.
- [ ] Anchor sticker vertex still maps to the measured sticker position after
      scale locking (translation corrected).
- [ ] Raw NPZ vertices are unchanged (scale is applied at render time, not baked in).
- [ ] Existing tests pass; new tests cover median locking and translation correction.
- [ ] `smooth-handmesh-rotation.md` updated to drop scale from its scope.

---

## Technical Design

### Approach

After the frame loop completes in `generate_3d_hand_in_motion` but before
`manager.save()`, call a new `lock_scale_to_median()` method on
`HandMotionManager`. This method locks scale AND corrects translations so
that the anchor sticker vertex still maps exactly to its measured position.

**Why translation must be corrected:**

The Procrustes alignment computes translation as (line 338):
```
t = t0 - s_frame * R @ s0
```
where `t0` is the measured sticker position and `s0` is the anchor vertex on
the untransformed mesh. This guarantees `s_frame * R @ s0 + t = t0`. If we
replace `s_frame` with `s_median` without adjusting `t`, the anchor drifts by
`(s_median - s_frame) * R @ s0` — up to several millimetres per frame.

**Correction formula:**
```
t_corrected = t_old + (s_frame - s_median) * R @ s0
```
This preserves the original anchor mapping: `s_median * R @ s0 + t_corrected = t0`.

**Method steps:**

1. Converts `self.scales` to a numpy array.
2. Masks invalid entries: `(scale <= 0) | (scale < _SCALE_MIN) | (scale > _SCALE_MAX)`.
3. Computes `np.median()` of valid entries — robust to outliers.
4. Raises `ValueError` if zero valid scales remain.
5. Logs raw scale statistics and the locked value via `logging.info()`.
6. For each frame, corrects translation:
   - `s0 = self.vertices_sequence[i][self._sticker_vertex_indices[0]]`
   - `R = Rotation.from_quat(self.rotations[i]).as_matrix()`
   - `self.translations[i] += (self.scales[i] - median) * (R @ s0)`
7. Replaces `self.scales` with `[median] * N`.

**New stored state:** `self._sticker_vertex_indices` — set once in
`process_frame()` on the first call. These indices are constant across frames
(same MANO mesh topology). Required by `lock_scale_to_median()` to look up
the anchor vertex per frame.

Scale is stored separately in the NPZ (`scales` key) and applied at render
time in `__getitem__()` via `matrix[:3, :3] *= scale`. Locking the array
and correcting translations propagates to all downstream consumers without
touching vertices.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Locked median scale | Simplest; physically correct; robust to outliers | Loses per-frame scale signal (but that signal is pure noise) | Chosen |
| Temporal smoothing of scale | Reduces jitter while preserving drift | Scale shouldn't drift at all; over-complex for a constant quantity | Rejected (rotation plan handles rotation/translation) |
| Per-session calibration window | More robust to mid-session tracking failures | Extra complexity; median already handles this via outlier robustness | Rejected |
| Rigid Procrustes (no scale DOF) | Most principled | Requires per-subject MANO calibration step that doesn't exist | Rejected |

### Architecture Changes

**Modified files:**
```
code/src/preprocessing/motion_analysis/hand_tracking/models/hand_motion_manager.py
    — promote _SCALE_MIN/_SCALE_MAX to class constants
    — store self._sticker_vertex_indices in process_frame() (first call)
    — add lock_scale_to_median() method (locks scale + corrects translations)

code/scripts/_3_preprocessing/_2_hand_tracking/generate_3d_hand_in_motion.py
    — single-line call to manager.lock_scale_to_median()

docs/development/plans/pending/smooth-handmesh-rotation.md
    — update out-of-scope note, remove scale from smoothing scope

code/tests/test_hand_motion_manager.py
    — add TestLockScaleToMedian class
```

No new files. No new dependencies. No DAG config changes.

---

## Implementation Plan

### Phase 1: Core method and pipeline integration
**Goal:** Add `lock_scale_to_median()` and wire it into the generation script.
**Started:** —
**Completed:** —

- [ ] Promote `_SCALE_MIN` / `_SCALE_MAX` from local variables (line 318) to
      class-level constants in `HandMotionManager`.
- [ ] Update existing references in `_calculate_alignment_procrustes()` to use
      the class constants.
- [ ] Store `sticker_vertex_indices` as `self._sticker_vertex_indices` in
      `process_frame()` on the first call (indices are constant across frames).
- [ ] Add `lock_scale_to_median()` method after `save()` (line 147), before
      the Playback section. Method must:
      - Compute median of valid scales
      - Correct per-frame translations: `t += (s_frame - s_median) * R @ s0`
      - Replace all scales with median
- [ ] Insert `manager.lock_scale_to_median()` call in
      `generate_3d_hand_in_motion.py` between line 112 (end of frame loop)
      and line 114 (`print("Step 3: Saving data...")`).

**Files Modified:**
- `code/src/preprocessing/motion_analysis/hand_tracking/models/hand_motion_manager.py` — class constants + new method
- `code/scripts/_3_preprocessing/_2_hand_tracking/generate_3d_hand_in_motion.py` — single-line call

**Dependencies:** None

### Phase 2: Tests and plan update
**Goal:** Add test coverage and update the rotation smoothing plan.
**Started:** —
**Completed:** —

- [ ] Add `TestLockScaleToMedian` class to `test_hand_motion_manager.py` with
      6 test cases (see Testing Plan below).
- [ ] Update `smooth-handmesh-rotation.md`: out-of-scope note (line 48),
      technical design text (lines 85-86), Phase 1 function signature (line 148).

**Files Modified:**
- `code/tests/test_hand_motion_manager.py` — new test class
- `docs/development/plans/pending/smooth-handmesh-rotation.md` — scope updates

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests

Add `TestLockScaleToMedian` to `code/tests/test_hand_motion_manager.py`:

- [ ] `test_lock_replaces_all_with_median` — 5 known scales `[1.0, 1.05, 0.95, 1.02, 0.98]`,
      assert all entries == median (1.0), assert return value == 1.0
- [ ] `test_lock_corrects_translation` — set up 2 frames with known
      `(vertices, rotations, translations, scales, _sticker_vertex_indices)`,
      call `lock_scale_to_median()`, verify that for each frame:
      `s_median * R @ s0 + t_corrected == s_frame * R @ s0 + t_old`
      (anchor sticker position preserved)
- [ ] `test_lock_filters_out_of_range` — scales `[1.0, 0.2, 1.05, 6.0, 0.98]`,
      assert median computed from valid-only `[1.0, 1.05, 0.98]`
- [ ] `test_lock_filters_non_positive` — scales `[1.0, -0.5, 0.0, 1.02, 0.98]`,
      assert 0 and negative excluded
- [ ] `test_lock_raises_on_empty` — no scales, expect `ValueError`
- [ ] `test_lock_raises_on_all_invalid` — scales `[0.0, -1.0, 0.1, 7.0]`,
      expect `ValueError`
- [ ] `test_lock_single_frame` — single valid scale `[1.05]`, locked to itself

### Manual Verification

- [ ] Run `generate_3d_hand_in_motion` on a real session — confirm output scales
      are all identical in the NPZ.
- [ ] Render handmesh overlay video — confirm mesh no longer "breathes" in size.
- [ ] Verify mesh stays aligned with blue stickers (anchor not drifting).
- [ ] Inspect CSV output — `scale` column should be constant.

### Edge Cases

- [ ] Session with some degenerate sticker frames (scale <= 0 from Procrustes
      guard) — fallback scales are valid copies of prior frames, should pass filter.
- [ ] Session with single frame — `np.median` of one element returns itself.
- [ ] Session where all scales are invalid — `ValueError` raised immediately.

---

## Documentation Plan

- [ ] No CLAUDE.md or README changes required — internal pipeline correction.
- [ ] `smooth-handmesh-rotation.md` updated to reflect scale is now handled upstream.

---

## Rollback Plan

The change is confined to `lock_scale_to_median()` and a single call-site.
To rollback: remove the call in `generate_3d_hand_in_motion.py` and
re-generate the NPZ. No data migrations, no schema changes.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Median skewed by many degenerate frames | Low | Med | Filter invalid scales before median; log valid/invalid counts |
| Session with legitimately different hand scale (different subject mid-session) | Very Low | Low | Not a real scenario in this protocol — single subject per session |
| Existing downstream code assumes per-frame scale variation | Low | Low | No downstream consumer uses scale variation as signal; all treat it as a transform parameter |
| Translation correction accumulates float error | Very Low | Low | Correction is a single multiply-add per frame; float32 precision is ample for mm-scale coordinates |

---

## References

- Related plan: `docs/development/plans/pending/smooth-handmesh-rotation.md`
- `HandMotionManager`: `code/src/preprocessing/motion_analysis/hand_tracking/models/hand_motion_manager.py`
- Scale computation: `_calculate_alignment_procrustes()` lines 310-316
- Scale application: `__getitem__()` line 223 (`matrix[:3, :3] *= scale`)
- Existing tests: `code/tests/test_hand_motion_manager.py::TestProcrustesScaleGuard`
