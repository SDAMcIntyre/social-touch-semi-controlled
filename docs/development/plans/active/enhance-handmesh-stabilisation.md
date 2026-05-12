# Plan: Enhance Hand-Mesh Pose Stabilisation — Anchor Smoothing & Filter Tuning

**Created:** 2026-05-12
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/stabilise-handmesh-pose`
**Branch:** `feature/stabilise-handmesh-pose` (existing — continuation of parent plan)

---

## Overview

Adds anchor position smoothing to the existing 4-step pose stabilisation and
tunes three filter cutoff frequencies to further reduce visible hand-mesh jitter.
The dominant remaining noise source — translation jitter from unsmoothed sticker
positions — is addressed by filtering the reconstructed `t0` before re-deriving
translation. Two YAML-only cutoff changes provide additional noise reduction with
zero code impact.

## Problem Statement

After the current stabilisation (scale locking + rotation smoothing), the formula
`t_new = t0 - s_stable * R_smooth @ s0` causes `t_new` to inherit `t0`'s noise
directly: `R_smooth` and `s_stable` are smooth, so translation tracks the raw
sticker measurement almost exactly. The upstream sticker XYZ filter uses a 10 Hz
cutoff (`configs/preprocess_workflow_kinect_auto_dag.yaml`, line 61), so
significant noise below 10 Hz flows through Procrustes alignment into `t0`. This
is the dominant remaining source of visible jitter in rendered handmesh overlays.

A secondary source is the rotation filter at 5 Hz, which passes through
low-frequency rotational noise that could be further attenuated with a lower
cutoff.

## Goals

### In Scope

1. Smooth `t0` (reconstructed sticker anchor positions) before re-deriving
   translation, so the mesh glides instead of jittering.
2. Make anchor smoothing configurable from the DAG YAML (`smooth_anchor` on/off
   + separate `anchor_filter_params` with independent cutoff).
3. Lower the rotation filter cutoff from 5 Hz to 3 Hz.
4. Lower the upstream sticker XYZ correction cutoff from 10 Hz to 6 Hz.

### Out of Scope

- Per-vertex (MANO articulation) smoothing — future improvement.
- One-Euro adaptive filter implementation.
- Changes to the Procrustes alignment algorithm itself.
- Rigid (no-scale) Procrustes or per-subject MANO calibration.

## Success Criteria

- [ ] Stabilised hand mesh shows visibly less translational jitter in overlay
      video compared to the current stabilisation.
- [ ] Anchor smoothing is configurable: `smooth_anchor: true` (default) with
      `anchor_filter_params` parameter in DAG YAML.
- [ ] Rotation cutoff lowered to 3 Hz; upstream sticker cutoff lowered to 6 Hz.
- [ ] All existing tests pass; `test_anchor_preserved` updated for smoothed
      anchor semantics.
- [ ] Raw NPZ remains byte-for-byte unchanged.

---

## Technical Design

### Approach

Three independent changes, all building on existing `MotionFilterFactory`
infrastructure:

**Change 1 — Smooth t0 in `PoseStabilisation.stabilise()`:**

Insert anchor position filtering between Step 1 (reconstruct t0) and Step 4
(re-derive translation). A separate filter instance is constructed with its own
cutoff parameter to allow independent tuning of anchor vs rotation smoothing.

```python
# New Step 1.5 — After t0 reconstruction, before translation re-derivation:
if smooth_anchor:
    anchor_filt = MotionFilterFactory.get_filter(filter_method, anchor_filter_params)
    for axis in range(3):
        t0[:, axis] = anchor_filt.filter(t0[:, axis], fps)
```

The anchor invariant changes semantics: `anchor_world == t0_smooth` instead of
`anchor_world == t0_raw`. This is a deliberate trade-off — the smoothed position
is closer to the true physical position than the noisy raw measurement.

**Change 2 — Lower rotation cutoff (YAML only):**

`stabilise_hand_motion.filter_params.butterworth.cutoff_hz`: `5.0` → `3.0`.
At 30 fps, 3 Hz retains most voluntary hand rotation while further attenuating
depth-sensor noise. Combined with zero-phase `filtfilt`, the group delay is
negligible.

**Change 3 — Lower upstream sticker cutoff (YAML only):**

`correct_xyz_stickers_motion.filter_params.butterworth.cutoff_hz`: `10.0` → `6.0`.
At 30 fps (Nyquist 15 Hz), 6 Hz retains voluntary hand-speed motion (~1–4 Hz)
while removing most depth-sensor noise. This is the `ButterworthFilter` class
default.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Smooth t0 with same filter as rotation | Minimal code | Can't tune anchor independently of rotation | Rejected |
| Smooth t0 with separate filter + cutoff | Independent tuning, ~10 lines | Slight API growth | **Chosen** |
| Skip t0 smoothing, only lower cutoffs | Zero code changes | Misses the biggest noise source | Rejected |
| One-Euro adaptive filter for t0 | Better fast-motion preservation | No existing infrastructure; designed for real-time, not offline batch | Deferred |
| Per-vertex MANO articulation smoothing | Addresses finger jitter | Larger scope; return type change | Deferred to separate plan |

### Architecture Changes

**Modified files only — no new files:**

```
code/src/preprocessing/motion_analysis/hand_tracking/
└── pose_stabilisation.py          — add Step 1.5 (t0 smoothing) + new params

code/scripts/_3_preprocessing/_2_hand_tracking/
└── stabilise_hand_motion.py       — forward new params from DAG config

code/scripts/
└── preprocess_workflow_kinect_auto.py  — read + forward new DAG options

configs/
└── preprocess_workflow_kinect_auto_dag.yaml  — tune 3 cutoff values + add anchor params

code/tests/
└── test_hand_motion_manager.py    — update anchor test + add smooth_anchor=False test
```

**Reused infrastructure:**
- `MotionFilterFactory.get_filter()` — `code/src/preprocessing/stickers_analysis/xyz/motion_correction/motion_filter_factory.py`
- `ButterworthFilter` — same package (`butterworth_filter.py`)
- Existing `_min_frames_required()` logic in `PoseStabilisation`

---

## Implementation Plan

### Phase 1: Add anchor smoothing to PoseStabilisation

**Goal:** Filter the reconstructed t0 positions before re-deriving translation.
**Started:** 2026-05-12
**Completed:** 2026-05-12

- [x] Add `smooth_anchor: bool = True` and `anchor_filter_params: dict | None = None`
      parameters to `PoseStabilisation.stabilise()`.
- [x] Default `anchor_filter_params` to `{"butterworth": {"order": 2, "cutoff_hz": 5.0}}`
      when `None` and `smooth_anchor` is `True`.
- [x] Between Step 1 (reconstruct t0) and Step 4 (re-derive translation), construct
      a second filter instance via `MotionFilterFactory.get_filter(filter_method, anchor_filter_params)`
      and filter each of the 3 t0 components.
- [x] Update the docstring to document the new parameters and the changed anchor
      invariant.

**Files Modified:**
- `code/src/preprocessing/motion_analysis/hand_tracking/pose_stabilisation.py` — add ~10 lines

**Dependencies:** None

### Phase 2: Forward parameters and tune DAG YAML

**Goal:** Wire the new anchor params through the pipeline script and tune all
three cutoff frequencies.
**Started:** 2026-05-12
**Completed:** 2026-05-12

- [x] In `stabilise_hand_motion.py`: add `smooth_anchor` and `anchor_filter_params`
      keyword arguments, forward them to `PoseStabilisation.stabilise()`.
- [x] In `preprocess_workflow_kinect_auto.py`: read `smooth_anchor` and
      `anchor_filter_params` from the DAG task options and forward them to the
      flow/script.
- [x] In `preprocess_workflow_kinect_auto_dag.yaml`, update `stabilise_hand_motion`:
      ```yaml
      stabilise_hand_motion:
        enabled: true
        options:
          force_processing: true
          filter_method: butterworth
          smooth_anchor: true
          filter_params:
            butterworth:
              order: 2
              cutoff_hz: 3.0
          anchor_filter_params:
            butterworth:
              order: 2
              cutoff_hz: 5.0
        depends_on: [generate_3d_hand_in_motion]
      ```
- [x] In the same YAML, update `correct_xyz_stickers_motion.filter_params.butterworth.cutoff_hz`
      from `10.0` to `6.0`.

**Files Modified:**
- `code/scripts/_3_preprocessing/_2_hand_tracking/stabilise_hand_motion.py` — forward new params
- `code/scripts/preprocess_workflow_kinect_auto.py` — read + forward new DAG options
- `configs/preprocess_workflow_kinect_auto_dag.yaml` — tune cutoffs + add anchor params

**Dependencies:** Phase 1

### Phase 3: Update tests

**Goal:** Update the anchor invariant test for smoothed t0 semantics.
**Started:** 2026-05-12
**Completed:** 2026-05-12

- [x] Update `test_anchor_preserved` to verify anchor world position matches the
      *smoothed* t0 (apply the same filter to t0 in the test, then compare).
- [x] Add `test_anchor_smoothing_disabled` — pass `smooth_anchor=False`, verify
      original invariant `anchor_world == t0_raw` still holds.
- [x] Verify all existing tests pass with the new parameter defaults.

**Files Modified:**
- `code/tests/test_hand_motion_manager.py` — update + add tests

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests

- [ ] `test_anchor_preserved` (updated) — anchor world position matches smoothed
      t0 to < 1e-5 per frame.
- [ ] `test_anchor_smoothing_disabled` (new) — `smooth_anchor=False` preserves
      raw t0 exactly (original invariant).
- [ ] All existing tests pass unchanged (scale locking, rotation smoothing,
      sign-fixup, short session, no valid scales, NaN quaternions).

### Manual Verification

- [ ] Regenerate stabilised NPZ for a real session with the new parameters.
- [ ] Render handmesh overlay video — compare translational jitter with and
      without anchor smoothing.
- [ ] Use `view_hand_mesh_comparison.py` to verify the smoothed mesh tracks
      the hand correctly without visible lag.
- [ ] Verify raw NPZ is byte-for-byte unchanged after re-running.

### Edge Cases

- [ ] Session with very few frames (near `_min_frames_required`): anchor filter
      uses the same minimum as rotation filter — verify both are checked.
- [ ] `smooth_anchor=False` with lowered rotation cutoff: translation still
      inherits raw t0 noise but rotation is smoother — verify no regression.

---

## Documentation Plan

- [ ] No CLAUDE.md or README changes — internal parameter tuning and small code change.
- [ ] Inline comment on the t0 smoothing rationale (why smoothed anchor is preferred
      over raw measurement fidelity).

---

## Rollback Plan

1. Set `smooth_anchor: false` in DAG YAML to disable anchor smoothing without
   reverting code.
2. Restore cutoff values (`cutoff_hz: 5.0` for rotation, `10.0` for upstream
   sticker) in the DAG YAML.
3. Re-run the pipeline to regenerate the stabilised NPZ with original parameters.

No schema changes, no new files, no downstream format changes.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Anchor smoothing causes visible lag during fast hand translations | Low | Med | 5 Hz cutoff with zero-phase `filtfilt` gives negligible lag; tunable via `anchor_filter_params` |
| Lowering upstream sticker cutoff to 6 Hz over-smooths fast stroking | Low | Med | 6 Hz is the `ButterworthFilter` default; review sticker trajectories visually before committing |
| Lowering rotation cutoff to 3 Hz over-smooths rapid wrist rotation | Med | Med | Verify on sessions with fast repositioning; can raise to 4 Hz if needed |
| Downstream consumers expect raw sticker-level anchor precision | Very Low | Low | `smooth_anchor` is configurable; no other consumer reads the stabilised NPZ |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — Anchor smoothing | ~10 lines | None |
| Phase 2 — Parameter wiring + YAML tuning | ~15 lines + YAML edits | Phase 1 |
| Phase 3 — Test updates | ~30 lines | Phase 1 |

---

## References

- Parent plan: `docs/development/plans/active/stabilise-handmesh-pose.md`
- `PoseStabilisation`: `code/src/preprocessing/motion_analysis/hand_tracking/pose_stabilisation.py`
- Pipeline script: `code/scripts/_3_preprocessing/_2_hand_tracking/stabilise_hand_motion.py`
- Pipeline flow: `code/scripts/preprocess_workflow_kinect_auto.py`
- Filter infrastructure: `code/src/preprocessing/stickers_analysis/xyz/motion_correction/`
- DAG config: `configs/preprocess_workflow_kinect_auto_dag.yaml` (lines 52–71 upstream sticker, 110–119 stabilisation)
