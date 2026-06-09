# Investigation: Hand-Mesh Stabilisation Over-Smoothing

**Date:** 2026-05-12
**Status:** Resolved — root cause identified, YAML fix applied
**Branch:** `feature/stabilise-handmesh-pose`
**Related plan:** `docs/development/plans/active/enhance-handmesh-stabilisation.md`

---

## Symptom

After enabling One-Euro adaptive anchor smoothing (Step 1.5 in
`PoseStabilisation.stabilise()`), the stabilised hand mesh is **visibly
over-smoothed during fast strokes** — motion amplitude is noticeably reduced.
This was confirmed on sessions where sticker velocity reaches 500 mm/s.

A YAML-only parameter change (`beta: 0.007 → 0.5`, `d_cutoff: 1.0 → 10.0`,
rotation `cutoff_hz: 3.0 → 5.0`) was proposed but initially reported as
ineffective.

## Root cause

**The YAML parameter changes were never written to the config file.** The
stabilised NPZ was regenerated with the original aggressive parameters
(`beta=0.007`, `d_cutoff=1.0`, `cutoff_hz=3.0`).  This was confirmed by
reading `configs/preprocess_workflow_kinect_auto_dag.yaml` — all three values
were still at their original values.

Two compounding issues at the original parameters:

1. **One-Euro at beta=0.007 is effectively a fixed ~1.3 Hz low-pass** — the
   velocity contribution to cutoff is `0.007 * speed`, yielding only +1.4 Hz
   at 200 mm/s.  The derivative smoother at d_cutoff=1.0 (alpha_d=0.173 at
   30 fps) further dampens the velocity estimate, keeping the effective cutoff
   around 1.3 Hz regardless of motion speed.

2. **Butterworth rotation cutoff at 3.0 Hz** (leftover from an earlier plan
   iteration; should have been 5.0 Hz per the parent plan) — with filtfilt
   (effective order 4), this attenuates 50% at the cutoff frequency.

## Quantitative evidence (ST16-05 block-order01, 3156 frames, 30 fps)

### Measured attenuation (old parameters)

| Frames    | Raw t0 speed | Stab t0 speed | Anchor ratio | Rot ratio |
|-----------|-------------|---------------|-------------|-----------|
| 1500-1700 | 198 mm/s    | 149 mm/s      | 0.753       | 0.639     |
| 1700-1900 | 286 mm/s    | 70 mm/s       | 0.243       | 0.499     |
| 2100-2300 | 199 mm/s    | 51 mm/s       | 0.253       | 0.448     |
| 2500-2700 | 192 mm/s    | 30 mm/s       | 0.157       | 0.354     |
| 2900-3100 | 251 mm/s    | 36 mm/s       | 0.142       | 0.313     |

**Average (frames 1500-3000):** anchor retains only 28% of raw speed;
rotations retain only 45%.

The attenuation worsens toward the recording end because the backward pass
starts at frame 3156 with `dx_smooth = 0.0` and processes the
already-attenuated forward output.  At beta=0.007, the forward pass
significantly reduces velocity, so the backward pass computes even lower
cutoffs — cascading the attenuation.

### One-Euro cutoff analysis (old vs fixed)

With `beta=0.007`: average forward cutoff 1.3-1.6 Hz (barely above
`min_cutoff=1.0`).

With `beta=0.5, d_cutoff=10.0`: average forward cutoff 30-38 Hz during
motion.  Forward and backward passes produce nearly identical cutoffs — the
cascade vanishes because the forward pass barely modifies the signal.

### Verified fix (computed on raw data, not yet regenerated)

| Frames    | Raw t0 speed | Old (beta=0.007) | Fixed (beta=0.5) |
|-----------|-------------|-------------------|-------------------|
| 1500-1700 | 198 mm/s    | 149 mm/s (0.753)  | 197 mm/s (0.992)  |
| 1700-1900 | 286 mm/s    | 70 mm/s (0.243)   | 281 mm/s (0.983)  |
| 2500-2700 | 192 mm/s    | 30 mm/s (0.157)   | 187 mm/s (0.974)  |
| 2900-3100 | 251 mm/s    | 36 mm/s (0.142)   | 243 mm/s (0.968)  |

Rotation Butterworth at 5 Hz: attenuation drops from 55% to 21%.

## Resolution

### Step 1: Fix rotation cutoff (YAML)

`cutoff_hz: 3.0 → 5.0` — rotation attenuation drops from 55% to 21%.

### Step 2: Disable anchor smoothing (YAML)

`smooth_anchor: true → false`.

After applying `beta=0.5` to fix the over-smoothing, the anchor filter became
essentially transparent (<3% attenuation).  Data-driven analysis confirmed
that the raw anchor signal has only 0.002 mm RMS jitter above 3 Hz — the
upstream `correct_xyz_stickers_motion` (Butterworth 6 Hz) already removes all
high-frequency noise before Procrustes alignment.  The One-Euro anchor filter
was solving a non-existent problem.

No code changes required.  Regenerate stabilised NPZ with `force_processing: true`.

## Corrections to earlier analysis

1. **Upstream sticker filter**: the signal chain section previously stated
   "Butterworth 10 Hz, order 2" — the actual DAG config value is **6 Hz**
   (line 61 of `preprocess_workflow_kinect_auto_dag.yaml`).

## Signal chain (corrected)

```
Raw sticker XYZ
    → correct_xyz_stickers_motion (Butterworth 6 Hz, order 2, filtfilt)
    → Procrustes alignment (per-frame SVD → R, s, t)
    → PoseStabilisation.stabilise():
        Step 1:   Reconstruct t0 = t + s * R @ s0
        Step 1.5: (disabled) Smooth t0 — upstream 6 Hz filter sufficient
        Step 2:   Smooth rotations (Butterworth 5 Hz, order 2, filtfilt on rotvec)
        Step 3:   Lock scale to session median
        Step 4:   Re-derive t_new = t0 - s_stable * R_smooth @ s0
```

## Key files

| File | Role |
|------|------|
| `code/src/preprocessing/stickers_analysis/xyz/motion_correction/one_euro_filter.py` | One-Euro filter (forward-backward) |
| `code/src/preprocessing/motion_analysis/hand_tracking/pose_stabilisation.py` | 4-step + Step 1.5 anchor smoothing |
| `code/scripts/_3_preprocessing/_2_hand_tracking/stabilise_hand_motion.py` | Pipeline entry point |
| `configs/preprocess_workflow_kinect_auto_dag.yaml` | All filter parameters (lines 52-71 sticker, 110-126 stabilisation) |
| `code/scripts/_3_preprocessing/_2_hand_tracking/view_hand_mesh_comparison.py` | Side-by-side raw vs stabilised viewer |
| `code/tests/test_hand_motion_manager.py` | Unit tests for PoseStabilisation |

## Future considerations (not required for this fix)

1. **Forward-backward One-Euro initialization**: the backward pass could
   initialize `dx_smooth` from the forward pass's final velocity instead of
   zero.  At beta=0.5 this is moot (<3% difference), but would improve
   robustness at low beta values.
2. **Rotation cutoff at 8 Hz**: would reduce rotation attenuation from 21%
   to 4%.  Trade-off: more high-frequency rotation noise passes through.
   The parent plan chose 5 Hz as adequate.
