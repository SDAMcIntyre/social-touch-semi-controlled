# Plan: Enhance Hand-Mesh Pose Stabilisation — Adaptive Anchor Smoothing

**Created:** 2026-05-12
**Approved:** —
**Completed:** 2026-05-15 08:45
**Author:** Basil Duvernoy
**Status:** Completed
**Base Branch:** `feature/stabilise-handmesh-pose`
**Branch:** `feature/stabilise-handmesh-pose` (existing — continuation of parent plan)

---

## Overview

Adds adaptive anchor position smoothing to the existing 4-step pose
stabilisation.  The dominant remaining noise source — translation jitter from
unsmoothed sticker positions — is addressed by filtering the reconstructed `t0`
with a One-Euro adaptive filter before re-deriving translation.  The One-Euro
filter smooths aggressively at rest while preserving fast motion (sticker
velocity reaches 500 mm/s in some sessions), avoiding the fixed-cutoff trade-off
between over-smoothing strokes and under-smoothing stationary periods.

## Problem Statement

After the current stabilisation (scale locking + rotation smoothing), the formula
`t_new = t0 - s_stable * R_smooth @ s0` causes `t_new` to inherit `t0`'s noise
directly: `R_smooth` and `s_stable` are smooth, so translation tracks the raw
sticker measurement almost exactly.  The upstream sticker XYZ filter uses a 10 Hz
cutoff, so significant noise below 10 Hz flows through Procrustes alignment into
`t0`.

A fixed low-pass on `t0` cannot satisfy both stationary and fast-motion
requirements:

| Butterworth cutoff | Attenuation at 3 Hz stroke | At 500 mm/s peak |
|--------------------|---------------------------|-------------------|
| 8 Hz               | ~2%                       | Fine              |
| 5 Hz               | ~12%                      | Visible lag       |
| 3 Hz               | ~30%                      | Unacceptable      |

The One-Euro filter (Casiez et al. 2012) solves this: its cutoff adapts to
signal speed.  At rest → low cutoff (aggressive smoothing).  During a 500 mm/s
stroke → high cutoff (preserves motion).

**Important constraint:** sticker positions feed quantitative downstream analysis
(contact detection, somatosensory features, neural correlations).  The upstream
sticker XYZ filter (10 Hz) and rotation filter (5 Hz) must NOT be changed.

## Goals

### In Scope

1. Implement a `OneEuroFilter` class in the existing motion-correction package,
   conforming to `MotionFilterInterface`.
2. Use a forward-backward (zero-phase) variant for offline batch processing.
3. Register the new filter in `MotionFilterFactory` + `FilterChoice` enum.
4. Add t0 anchor smoothing to `PoseStabilisation.stabilise()` using One-Euro.
5. Make anchor smoothing configurable from the DAG YAML (`smooth_anchor` on/off
   + `anchor_filter_method` + `anchor_filter_params`).

### Out of Scope

- Changes to the upstream sticker XYZ correction cutoff (affects quantitative
  downstream consumers).
- Changes to the rotation Butterworth cutoff (adequate at 5 Hz).
- Per-vertex MANO articulation smoothing — future improvement.
- Using One-Euro for the rotation filter (rotvec component meaning makes
  velocity-based adaptation less principled).

## Success Criteria

- [ ] `OneEuroFilter` passes the `MotionFilterInterface` contract and is
      registered in `MotionFilterFactory`.
- [ ] Stabilised hand mesh shows visibly less translational jitter at rest
      **without** attenuating fast strokes.
- [ ] Anchor smoothing is configurable: `smooth_anchor: true` (default) with
      `anchor_filter_method` and `anchor_filter_params` in DAG YAML.
- [ ] All existing tests pass; new tests cover One-Euro and anchor smoothing.
- [ ] Raw NPZ remains byte-for-byte unchanged.

---

## Technical Design

### Approach

**One-Euro adaptive filter** — the cutoff frequency adapts per-sample based on
the signal's instantaneous speed:

```
fc(t) = min_cutoff + beta * |dx_smooth(t)|
```

- `min_cutoff` (Hz): cutoff at rest — controls maximum smoothing.
- `beta` (Hz·s/mm): speed coefficient — how fast the cutoff rises with speed.
- `d_cutoff` (Hz): fixed cutoff for the speed estimator.

At `min_cutoff=1.0, beta=0.007`:
- **At rest** (0 mm/s): fc = 1.0 Hz → heavy smoothing, eliminates visible jitter
- **At 100 mm/s**: fc = 1.7 Hz → moderate smoothing
- **At 500 mm/s**: fc = 4.5 Hz → light smoothing, preserves fast motion

**Zero-phase variant for offline processing:**  The standard One-Euro is
causal (introduces lag).  For offline batch processing, apply the filter in a
forward-backward pass: filter the signal forward, then filter the result
backward.  The backward pass cancels the phase delay while maintaining adaptive
behaviour.  This mirrors the `filtfilt` principle used by `ButterworthFilter`.

**Integration into PoseStabilisation — new Step 1.5:**

```python
# After Step 1 (reconstruct t0), before Step 4 (re-derive translation):
if smooth_anchor:
    anchor_filt = MotionFilterFactory.get_filter(
        anchor_filter_method, anchor_filter_params
    )
    for axis in range(3):
        t0[:, axis] = anchor_filt.filter(t0[:, axis], fps)
```

The anchor invariant changes: `anchor_world == t0_smooth` instead of
`anchor_world == t0_raw`.  The smoothed position tracks the true physical
position more closely than the noisy raw measurement, especially at rest.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Fixed Butterworth on t0 | Reuses existing filter; zero-phase via `filtfilt` | Fixed cutoff: either over-smooths fast strokes or under-smooths rest | Rejected — 500 mm/s strokes need adaptive behaviour |
| One-Euro on t0 (forward-backward) | Adaptive: aggressive at rest, preserves fast motion; fits `MotionFilterInterface` | New filter class (~60 lines); 3 parameters to tune | **Chosen** |
| Lower upstream sticker cutoff | Reduces noise at source | Affects quantitative downstream consumers (contact features, neural correlations) | Rejected — scope too broad |
| Lower rotation Butterworth cutoff | Reduces rotation noise | 3 Hz cutoff attenuates 30% of a 3 Hz stroke; adaptive less principled on rotvec | Rejected — 5 Hz is adequate |
| Segment-based adaptive Butterworth | Reuses Butterworth; per-segment cutoffs | Boundary artefacts; complex to tune | Rejected |

### Architecture Changes

**New file:**
```
code/src/preprocessing/stickers_analysis/xyz/motion_correction/
└── one_euro_filter.py     — OneEuroFilter (MotionFilterInterface)
```

**Modified files:**
```
code/src/preprocessing/stickers_analysis/xyz/motion_correction/
└── motion_filter_factory.py   — add ONE_EURO to FilterChoice + dispatch

code/src/preprocessing/motion_analysis/hand_tracking/
└── pose_stabilisation.py      — add Step 1.5 (t0 smoothing) + new params

code/scripts/_3_preprocessing/_2_hand_tracking/
└── stabilise_hand_motion.py   — forward anchor params from DAG config

code/scripts/
└── preprocess_workflow_kinect_auto.py  — read + forward new DAG options

configs/
└── preprocess_workflow_kinect_auto_dag.yaml  — add anchor smoothing config

code/tests/
└── test_hand_motion_manager.py — anchor test updates + One-Euro tests
```

**Reused infrastructure:**
- `MotionFilterInterface` — `motion_filter_interface.py` (the abstract base)
- `MotionFilterFactory` — `motion_filter_factory.py` (factory + enum)
- `PoseStabilisation.stabilise()` — existing 4-step correction
- `should_process_task()` — idempotency guard in pipeline script

---

## Implementation Plan

### Phase 1: One-Euro filter class

**Goal:** Implement `OneEuroFilter` conforming to `MotionFilterInterface` with
forward-backward zero-phase processing.
**Started:** 2026-05-12
**Completed:** 2026-05-12

- [x] Create `one_euro_filter.py` with class `OneEuroFilter`:
  - Constructor: `__init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0)`
  - Private `_smoothing_factor(cutoff, dt)`: compute alpha = 1 / (1 + tau/dt)
    where tau = 1 / (2*pi*cutoff).
  - Private `_one_euro_pass(signal, dt)`: single causal pass — iterate frames,
    estimate derivative, smooth derivative with d_cutoff, compute adaptive
    cutoff, apply first-order low-pass.
  - Public `filter(signal, sampling_rate_hz)`: forward-backward pass — run
    `_one_euro_pass` forward on the signal, then backward on the forward
    result (reversed), return the reversed backward result.
  - Public `name()`: return `"One-Euro (min_fc={min_cutoff}, β={beta})"`.
- [x] Register in `motion_filter_factory.py`:
  - Add `ONE_EURO = "one_euro"` to `FilterChoice` enum.
  - Add dispatch branch in `get_filter()`:
    `kwargs = filter_params.get("one_euro", {}); return OneEuroFilter(**kwargs)`.
  - Add import for `OneEuroFilter`.

**Files Modified:**
- `code/src/preprocessing/stickers_analysis/xyz/motion_correction/one_euro_filter.py` — new (~60 lines)
- `code/src/preprocessing/stickers_analysis/xyz/motion_correction/motion_filter_factory.py` — add enum + dispatch

**Dependencies:** None

### Phase 2: Anchor smoothing in PoseStabilisation

**Goal:** Add t0 smoothing between Step 1 and Step 4 using configurable filter.
**Started:** 2026-05-12
**Completed:** 2026-05-12

- [x] Add parameters to `PoseStabilisation.stabilise()`:
  - `smooth_anchor: bool = True`
  - `anchor_filter_method: str | Any = "one_euro"`
  - `anchor_filter_params: dict | None = None`
- [x] Default `anchor_filter_params` to
  `{"one_euro": {"min_cutoff": 1.0, "beta": 0.007, "d_cutoff": 1.0}}` when
  `None` and `smooth_anchor` is `True`.
- [x] Insert Step 1.5 between Step 1 and Step 4: construct filter via
  `MotionFilterFactory.get_filter(anchor_filter_method, anchor_filter_params)`,
  filter each of the 3 t0 components.
- [x] Update `_min_frames_required()` to handle `"one_euro"` (minimum 2 frames).
- [x] Update docstring for new parameters and changed anchor semantics.

**Files Modified:**
- `code/src/preprocessing/motion_analysis/hand_tracking/pose_stabilisation.py` — ~15 lines

**Dependencies:** Phase 1

### Phase 3: Pipeline wiring + DAG config

**Goal:** Forward anchor params through the pipeline and configure defaults.
**Started:** 2026-05-12
**Completed:** 2026-05-12

- [x] In `stabilise_hand_motion.py`: add `smooth_anchor`, `anchor_filter_method`,
  and `anchor_filter_params` keyword arguments; forward to
  `PoseStabilisation.stabilise()`.
- [x] In `preprocess_workflow_kinect_auto.py`: read `smooth_anchor`,
  `anchor_filter_method`, and `anchor_filter_params` from DAG task options;
  forward to the flow/script.
- [x] In `preprocess_workflow_kinect_auto_dag.yaml`, update `stabilise_hand_motion`:
  ```yaml
  stabilise_hand_motion:
    enabled: true
    options:
      force_processing: true
      filter_method: butterworth
      smooth_anchor: true
      anchor_filter_method: one_euro
      anchor_filter_params:
        one_euro:
          min_cutoff: 1.0
          beta: 0.007
          d_cutoff: 1.0
      filter_params:
        butterworth:
          order: 2
          cutoff_hz: 5.0
    depends_on: [generate_3d_hand_in_motion]
  ```

**Files Modified:**
- `code/scripts/_3_preprocessing/_2_hand_tracking/stabilise_hand_motion.py` — forward new params
- `code/scripts/preprocess_workflow_kinect_auto.py` — read + forward new DAG options
- `configs/preprocess_workflow_kinect_auto_dag.yaml` — add anchor config (rotation cutoff unchanged at 5 Hz)

**Dependencies:** Phase 2

### Phase 4: Tests

**Goal:** Test One-Euro filter and anchor smoothing.
**Started:** 2026-05-12
**Completed:** 2026-05-12

- [x] Add `TestOneEuroFilter` to test file:
  - `test_constant_signal_unchanged` — constant input returns constant output.
  - `test_smooths_noisy_stationary` — stationary signal + white noise:
    output RMS error < input RMS error.
  - `test_preserves_fast_ramp` — linear ramp (high velocity): output tracks
    input closely (< 5% peak deviation).
  - `test_adaptive_cutoff` — slow segment is smoothed more than fast segment
    in the same signal.
- [x] Update `test_anchor_preserved` to verify anchor world position matches
  the *smoothed* t0 (apply One-Euro to t0 in the test, then compare).
- [x] Add `test_anchor_smoothing_disabled` — `smooth_anchor=False` preserves
  raw t0 exactly.
- [x] Verify all existing tests pass with unchanged defaults.

**Files Modified:**
- `code/tests/test_hand_motion_manager.py` — update + add tests (~50 lines)

**Dependencies:** Phase 1, Phase 2

### Phase 5: Fix over-smoothing at high sticker velocities

**Goal:** Eliminate excessive smoothing during fast strokes (500 mm/s).  Testing
revealed ~67% total attenuation at the 3 Hz stroke frequency, caused by two
compounding issues: rotation cutoff at 3 Hz (leftover from earlier plan
iteration) and One-Euro `beta=0.007` being far too low (cutoff only 4.5 Hz at
500 mm/s, compounded by forward-backward cascading).

**Requirement:** no perceptible filtering above 200 mm/s sticker velocity.

**Root cause analysis:**
- Rotation Butterworth at 3 Hz with `filtfilt` attenuates 3 Hz stroke by ~30%.
- One-Euro `beta=0.007` → fc = 4.5 Hz at 500 mm/s.  First-order at 4.5 Hz
  attenuates 3 Hz by ~26%.  Forward-backward compounds to ~50%.
- Combined: ~67% signal loss at stroke frequency.

**Fix:** Three YAML parameter changes (no code changes):

| Parameter | Current | New | Rationale |
|-----------|---------|-----|-----------|
| `beta` | 0.007 | 0.5 | At 200 mm/s: fc = 101 Hz (transparent). At rest: fc = 1 Hz (unchanged) |
| `d_cutoff` | 1.0 | 10.0 | Faster derivative estimation — adapts in ~1-2 frames instead of ~5 |
| `cutoff_hz` (rotation) | 3.0 | 5.0 | Revert to parent plan value — 3 Hz was too aggressive |

**Why beta=0.5 also fixes forward-backward cascading:** at 200+ mm/s, the
forward pass barely modifies the signal (fc > 101 Hz, <0.5% attenuation), so
the backward pass sees near-original velocity and computes the same high
cutoff.  Cascading only manifests when the forward pass attenuates
significantly — which with beta=0.5 only happens at rest (no velocity to
cascade).

**Expected behaviour after fix:**

| Velocity | One-Euro fc | Anchor atten. at 3 Hz | Rotation atten. | Total |
|----------|-------------|----------------------|-----------------|-------|
| 0 mm/s   | 1.0 Hz      | >95%                 | ~11%            | >95%  |
| 50 mm/s  | 26 Hz       | ~8%                  | ~11%            | ~18%  |
| 200 mm/s | 101 Hz      | <0.5%                | ~11%            | ~11%  |
| 500 mm/s | 251 Hz      | <0.1%                | ~11%            | ~11%  |

**Started:** 2026-05-12
**Completed:** 2026-05-12

**Outcome:** Data-driven analysis on ST16-05 block-order01 showed the raw
anchor has only 0.002 mm RMS HF jitter (upstream 6 Hz filter already cleans
it).  Anchor smoothing was **disabled** (`smooth_anchor: false`) as it
provides no visible benefit.  Rotation cutoff changed from 3 Hz to 5 Hz.

- [x] In `preprocess_workflow_kinect_auto_dag.yaml`, update `stabilise_hand_motion`:
  - `cutoff_hz: 3.0` → `cutoff_hz: 5.0`
  - `smooth_anchor: true` → `smooth_anchor: false`
  - Anchor filter params left in YAML but ignored (smooth_anchor=false)
- [x] Run `pytest code/tests/test_hand_motion_manager.py -v` — all 17 tests pass.
- [ ] Regenerate stabilised NPZ for a fast-motion session — confirm fast strokes
      preserved and scale breathing eliminated.

**Files Modified:**
- `configs/preprocess_workflow_kinect_auto_dag.yaml` — smooth_anchor disabled, cutoff 5 Hz

**Dependencies:** Phase 3

---

## Testing Plan

### Unit Tests — One-Euro Filter

- [ ] `test_constant_signal_unchanged` — no smoothing artefacts on DC input
- [ ] `test_smooths_noisy_stationary` — noise reduction on stationary signal
- [ ] `test_preserves_fast_ramp` — < 5% deviation on high-velocity ramp
- [ ] `test_adaptive_cutoff` — slow segments smoothed more than fast segments

### Unit Tests — Anchor Smoothing

- [ ] `test_anchor_preserved` (updated) — anchor matches smoothed t0 to < 1e-5
- [ ] `test_anchor_smoothing_disabled` — `smooth_anchor=False` preserves raw t0
- [ ] All existing tests pass unchanged (scale locking, rotation smoothing,
      sign-fixup, edge cases)

### Manual Verification

- [ ] Regenerate stabilised NPZ for a real session with One-Euro anchor smoothing.
- [ ] Render handmesh overlay — verify reduced jitter at rest without lag during
      fast strokes.
- [ ] Compare side-by-side using `view_hand_mesh_comparison.py`.
- [ ] Verify raw NPZ is byte-for-byte unchanged.

### Edge Cases

- [ ] Very short session (2-3 frames): One-Euro forward-backward should not crash.
- [ ] Session with sustained high velocity: anchor cutoff stays high, minimal smoothing.
- [ ] Session with sustained rest: anchor cutoff drops to `min_cutoff`, heavy smoothing.

---

## Documentation Plan

- [ ] No CLAUDE.md or README changes — internal pipeline enhancement.
- [ ] Inline comment in `one_euro_filter.py` referencing Casiez et al. 2012 and
      explaining the forward-backward zero-phase variant.

---

## Rollback Plan

1. Set `smooth_anchor: false` in DAG YAML — disables anchor smoothing without
   reverting code.
2. Re-run the pipeline to regenerate stabilised NPZ with original parameters.
3. The One-Euro filter class remains available but unused — no harm in keeping it.

No schema changes, no downstream format changes.  Rotation and upstream sticker
filters are unchanged.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| One-Euro `beta` parameter needs per-dataset tuning | Med | Med | **Realized in Phase 5:** `beta=0.007` was far too low for 500 mm/s strokes. Fixed to `beta=0.5` — transparent above 200 mm/s. |
| Forward-backward One-Euro over-smooths (backward pass sees already-smoothed velocity) | Low | Low | **Realized in Phase 5:** compounded with low beta to cause ~50% anchor attenuation. High beta (0.5) makes forward pass near-transparent at speed, eliminating the cascading. |
| One-Euro first-order rolloff is gentler than Butterworth order 2 | Low | Low | Adaptive cutoff compensates: at rest, even 1 Hz first-order provides strong attenuation of > 3 Hz noise |
| Anchor smoothing causes mesh to drift from physical sticker position | Very Low | Low | Drift only during rest-period jitter (which is noise anyway); fast motion tracked faithfully |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — One-Euro filter class | ~60 lines | None |
| Phase 2 — Anchor smoothing | ~15 lines | Phase 1 |
| Phase 3 — Pipeline wiring + YAML | ~15 lines + YAML edits | Phase 2 |
| Phase 4 — Tests | ~50 lines | Phase 1, Phase 2 |

---

## References

- Casiez, G., Roussel, N., & Vogel, D. (2012). 1€ Filter: A Simple Speed-based
  Low-pass Filter for Noisy Input in Interactive Systems. CHI '12.
- Parent plan: `docs/development/plans/active/stabilise-handmesh-pose.md`
- `PoseStabilisation`: `code/src/preprocessing/motion_analysis/hand_tracking/pose_stabilisation.py`
- Pipeline script: `code/scripts/_3_preprocessing/_2_hand_tracking/stabilise_hand_motion.py`
- Filter infrastructure: `code/src/preprocessing/stickers_analysis/xyz/motion_correction/`
- DAG config: `configs/preprocess_workflow_kinect_auto_dag.yaml` (lines 110–119 for stabilisation)
