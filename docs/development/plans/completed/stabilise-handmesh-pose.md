# Plan: Stabilise Hand-Mesh Pose Stream

**Created:** 2026-05-12 23:00
**Approved:** —
**Completed:** 2026-05-15 08:45
**Author:** Basil Duvernoy
**Status:** Completed
**Base Branch:** `dev`
**Branch:** `feature/stabilise-handmesh-pose`

---

## Overview

Adds a single post-generation pipeline step `stabilise_hand_motion` that corrects two
artefacts in the MANO hand-mesh pose stream — size breathing and rotational jitter — both
caused by Procrustes alignment sensitivity to depth-sensor noise in the 3 sticker anchors.
The step applies scale locking and rotation smoothing in a single ordered pass that
re-derives translation from the anchor constraint, preserving the exact world position of
the blue sticker vertex.

## Problem Statement

`HandMotionManager` aligns the mesh frame-by-frame using only 3 sticker anchor points.
Even after the upstream 6 Hz Butterworth filter on sticker XYZ, residual depth noise
propagates through the Procrustes SVD and produces two visible artefacts in rendered
handmesh overlay videos:

1. **Size breathing** — the per-frame uniform scale factor varies even though the hand
   does not physically change size. All scale variation is pure noise.
2. **Rotational jitter** — per-frame rotation vibrates between adjacent frames, degrading
   visual quality of rendered overlays.

A third constraint governs both corrections: the Procrustes translation is derived from
the anchor formula `t = t0 − s × R @ s0`, where `t0` is the measured sticker position
and `s0` is the blue sticker vertex on the untransformed mesh. Any correction that changes
`s` or `R` without re-deriving `t` causes the blue sticker vertex to drift from its
measured position. Scale locking and rotation smoothing must therefore be applied together
in a single pass where translation is always the last derived quantity.

## Goals

### In Scope
1. Lock all per-frame scales to the session median of valid scale values.
2. Smooth the rotation stream via rotation-vector space (handles quaternion double-cover).
3. Re-derive per-frame translations from the anchor constraint after scale and rotation
   are finalised — never filter translations directly.
4. Write `*_handmodel_motion_stabilised.npz` alongside the unchanged raw NPZ.
5. Save `sticker_vertex_indices` to the raw NPZ so the post-generation step can
   reconstruct the measured sticker positions without reading the CSV.
6. Wire `stabilise_hand_motion` into the preprocess auto DAG; renderer and visualisation
   prefer the stabilised NPZ when it exists.
7. Filter method, cutoff, and order configurable from the DAG YAML.
8. Fail-fast if no valid scales remain or session is too short for the chosen filter.

### Out of Scope
- Changes to the upstream sticker Butterworth filter cutoff.
- Smoothing mesh vertices directly.
- Manual DAG (`preprocess_workflow_kinect_manual_dag.yaml`).
- Contact detection or somatosensory pipeline.
- Per-subject MANO mesh calibration or rigid (no-scale) Procrustes.

## Success Criteria

- [ ] All entries in `scales` in the stabilised NPZ are identical (median-locked).
- [ ] Rotational jitter is visibly gone in handmesh overlay video rendered from stabilised NPZ.
- [ ] Blue sticker vertex world position is preserved: `‖s_stable × R_smooth @ s0 + t_new − t0‖ < 1e-5` per frame.
- [ ] Raw NPZ is byte-for-byte unchanged after the step runs.
- [ ] `stabilise_hand_motion` appears in the pipeline GUI under the Preprocess [Auto] workflow.
- [ ] `force_processing=False` skips when the stabilised NPZ already exists.
- [ ] Filter method, cutoff, and order are configurable from the DAG YAML.
- [ ] Existing tests pass; new tests cover the anchor invariant, scale locking, rotation smoothing, and all edge cases.

---

## Technical Design

### Approach

A single post-generation step reads the raw NPZ, applies all corrections in a fixed
4-step order, and writes a stabilised NPZ. The raw file is never modified. The ordering
is mandatory: translation must be the last quantity derived, from the anchor constraint,
not filtered independently.

**4-Step Correction:**

**Step 1 — Reconstruct measured sticker positions** (`t0` is not stored in the NPZ;
derived from the stored data using the inverse of the Procrustes translation formula):
```python
# anchor_idx = sticker_vertex_indices[0]  (blue sticker vertex index)
t0[i] = translations[i] + scales[i] * (R_orig[i] @ vertices[i][anchor_idx])
```

**Step 2 — Smooth rotations** (rotvec space, sign-consistent):
```python
# Sign-fixup: flip q[i] when dot(q[i], q[i-1]) < 0
rotvecs = Rotation.from_quat(rotations_fixed).as_rotvec()   # (N, 3)
rotvecs_smooth = MotionFilterFactory.filter(rotvecs)         # per-component
rotations_smooth = Rotation.from_rotvec(rotvecs_smooth).as_quat()
```

**Step 3 — Lock scale to session median:**
```python
valid = (scales > _SCALE_MIN) & (scales < _SCALE_MAX)
s_stable = np.median(scales[valid])   # raises ValueError if no valid scales
```

**Step 4 — Re-derive translation from anchor constraint:**
```python
t_new[i] = t0[i] - s_stable * (R_smooth[i] @ vertices[i][anchor_idx])
```

**Why this preserves the anchor:** substituting back:
`s_stable × R_smooth @ s0 + t_new = s_stable × R_smooth @ s0 + t0 − s_stable × R_smooth @ s0 = t0 ✓`

**NPZ schema extension:** `anchor_idx = sticker_vertex_indices[0]` must be saved in
the raw NPZ (new key `sticker_vertex_indices`, int array shape `(K,)`) so Step 1 can
run without reading the original CSV. `HandMotionManager.save()` and `load()` are
extended accordingly. Existing NPZs lacking this key raise immediately on load (fail-fast;
re-run `generate_3d_hand_in_motion` once to regenerate).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Two separate pipeline steps (original plans) | Each concern isolated | Translation in Plan 2 smoothed independently — breaks anchor; the plans are not composable without a shared re-derivation pass | Rejected — merged into one step |
| Apply both corrections at generation time | No new pipeline step | Cannot re-smooth without re-generating; raw NPZ has no diagnostic value | Rejected — post-generation step preserves raw |
| Independent translation filtering | Trivial implementation | Violates anchor constraint; blue sticker vertex drifts | Rejected |
| SLERP-based rotation smoothing | Geometrically exact | No existing infrastructure; much harder to implement | Rejected |
| Kalman filter on pose | Handles uncertainty properly | No existing infrastructure; overkill for offline batch processing | Rejected |
| Rigid Procrustes (no scale DOF) | Most principled for scale | Requires per-subject MANO calibration step that does not exist | Deferred |

### Architecture Changes

**New files:**
```
code/src/preprocessing/motion_analysis/hand_tracking/
└── pose_stabilisation.py      — PoseStabilisation class (4-step correction)

code/scripts/_3_preprocessing/_2_hand_tracking/
└── stabilise_hand_motion.py   — pipeline entry-point script
```

**Modified files:**
```
code/src/preprocessing/motion_analysis/hand_tracking/models/hand_motion_manager.py
    — process_frame(): store self._sticker_vertex_indices on first call
    — save(): add sticker_vertex_indices key to NPZ
    — load(): restore self._sticker_vertex_indices; KeyError if key missing

code/src/preprocessing/motion_analysis/hand_tracking/__init__.py
    — export PoseStabilisation

configs/preprocess_workflow_kinect_auto_dag.yaml
    — add stabilise_hand_motion task after generate_3d_hand_in_motion

code/scripts/preprocess_workflow_kinect_auto.py
    — add stabilise_hand_motion_flow() @flow wrapper (~line 297 pattern)
    — add pipeline stage entry (~line 529)

code/src/preprocessing/motion_analysis/hand_tracking/handmesh_overlay_renderer.py
    — prefer *_handmodel_motion_stabilised.npz; fall back to raw; raise if neither

code/scripts/preprocess_workflow_kinect_visualisation.py
    — update hand_motion_path construction (line 143) with same preference logic
```

**Reused infrastructure:**
- `MotionFilterFactory` — `code/src/preprocessing/stickers_analysis/xyz/motion_correction/motion_filter_factory.py`
- `ButterworthFilter` / `SavgolFilter` — same package
- `HandMotionManager.load()` / numpy NPZ I/O — `hand_motion_manager.py`
- Pipeline stage pattern — `generate_3d_hand_in_motion_flow` lines 297–332 and stage entry lines 523–529 in `preprocess_workflow_kinect_auto.py`

---

## Implementation Plan

### Phase 1: NPZ schema extension
**Goal:** Save and restore `sticker_vertex_indices` in `HandMotionManager` so the
post-generation step can locate the blue sticker vertex without reading the CSV.
**Started:** 2026-05-12
**Completed:** 2026-05-12

- [x] In `process_frame()`, store `sticker_vertex_indices` as `self._sticker_vertex_indices`
      on the first call (indices are constant across frames — same MANO topology).
- [x] In `save()`: add `"sticker_vertex_indices": np.array(self._sticker_vertex_indices)`
      to `save_dict`.
- [x] In `load()`: restore `self._sticker_vertex_indices = list(data["sticker_vertex_indices"])`;
      raise `KeyError` with clear message if key is missing.

**Files Modified:**
- `code/src/preprocessing/motion_analysis/hand_tracking/models/hand_motion_manager.py` — save/load/process_frame

**Dependencies:** None

### Phase 2: PoseStabilisation class
**Goal:** Implement the 4-step correction as a reusable class.
**Started:** 2026-05-12
**Completed:** 2026-05-12

- [x] Create `pose_stabilisation.py` with `PoseStabilisation` class:
  - `stabilise(vertices, translations, rotations_xyzw, scales, anchor_idx, fps, filter_method, filter_params) → (translations_new, rotations_new, scales_new)`
  - Step 1: reconstruct `t0[i]` from stored data.
  - Step 2: sign-fixup loop on quaternions; rotvec round-trip; `MotionFilterFactory` per-component filtering.
  - Step 3: median scale with `(scale > _SCALE_MIN) & (scale < _SCALE_MAX)` validity mask; raise `ValueError` if no valid scales remain.
  - Step 4: anchor-constrained translation re-derivation.
  - Raise `ValueError` if `len(translations) < min_frames_required` for the chosen filter.
- [x] Export `PoseStabilisation` from `__init__.py`.

**Files Modified:**
- `code/src/preprocessing/motion_analysis/hand_tracking/pose_stabilisation.py` — new
- `code/src/preprocessing/motion_analysis/hand_tracking/__init__.py` — add export

**Dependencies:** Phase 1

### Phase 3: Pipeline script
**Goal:** Wrap `PoseStabilisation` as a standalone pipeline script with idempotency.
**Started:** 2026-05-12
**Completed:** 2026-05-12

- [x] Create `stabilise_hand_motion.py`:
  - Load `*_handmodel_motion.npz`; raise `FileNotFoundError` immediately if missing.
  - Call `PoseStabilisation.stabilise(...)` with filter options from DAG config.
  - Save corrected `rotations`, `scales`, `translations` plus unchanged `vertices`,
    `faces`, `timestamps`, `fps`, `sticker_vertex_indices` to
    `*_handmodel_motion_stabilised.npz`.
  - Honour `force_processing=False`: skip if output already exists.

**Files Modified:**
- `code/scripts/_3_preprocessing/_2_hand_tracking/stabilise_hand_motion.py` — new

**Dependencies:** Phase 2

### Phase 4: DAG and pipeline wiring
**Goal:** Register `stabilise_hand_motion` in the preprocess auto pipeline.
**Started:** 2026-05-12
**Completed:** 2026-05-12

- [x] Add task to `configs/preprocess_workflow_kinect_auto_dag.yaml` after
      `generate_3d_hand_in_motion`:
  ```yaml
  stabilise_hand_motion:
    enabled: true
    options:
      force_processing: false
      filter_method: butterworth
      filter_params:
        butterworth:
          order: 2
          cutoff_hz: 5.0
    depends_on: [generate_3d_hand_in_motion]
  ```
- [x] Add `stabilise_hand_motion_flow()` `@flow` wrapper in
      `preprocess_workflow_kinect_auto.py` following the exact pattern of
      `generate_3d_hand_in_motion_flow` (lines 297–332).
- [x] Add pipeline stage entry after line 529 with `"name": "stabilise_hand_motion"`,
      `"params"` reading `hand_motion_npz_path` from context and filter options from
      `dag_handler.get_task_options("stabilise_hand_motion")`.

**Files Modified:**
- `configs/preprocess_workflow_kinect_auto_dag.yaml` — new task entry
- `code/scripts/preprocess_workflow_kinect_auto.py` — flow wrapper + stage entry

**Dependencies:** Phase 3

### Phase 5: Renderer and visualisation integration
**Goal:** Have rendering consumers prefer the stabilised NPZ.
**Started:** 2026-05-12
**Completed:** 2026-05-12

- [x] `handmesh_overlay_renderer.py`: at NPZ load time, check for
      `*_handmodel_motion_stabilised.npz` first; fall back to `*_handmodel_motion.npz`;
      raise if neither exists.
- [x] `preprocess_workflow_kinect_visualisation.py` (line 143): apply the same
      preference logic to `hand_motion_path` construction.

**Files Modified:**
- `code/src/preprocessing/motion_analysis/hand_tracking/handmesh_overlay_renderer.py` — NPZ load preference
- `code/scripts/preprocess_workflow_kinect_visualisation.py` — hand_motion_path preference

**Dependencies:** Phase 3

### Phase 6: Tests and plan cleanup
**Goal:** Cover the anchor invariant and all edge cases; retire superseded plans.
**Started:** 2026-05-12
**Completed:** 2026-05-12

- [x] Add `TestPoseStabilisation` class to `code/tests/test_hand_motion_manager.py`:
  - `test_anchor_preserved` — 3-frame synthetic session; verify `‖s_stable × R_smooth @ s0 + t_new − t0‖ < 1e-5` for each frame.
  - `test_scale_locked_to_median` — known scales `[1.0, 1.05, 0.95, 1.02, 0.98]`; all output scales == median.
  - `test_scale_filters_out_of_range` — scales containing values outside `[_SCALE_MIN, _SCALE_MAX]`; median from valid only.
  - `test_raises_on_no_valid_scales` — all scales invalid; expect `ValueError`.
  - `test_raises_on_short_session` — fewer frames than `2 × filter_order + 1`; expect `ValueError`.
  - `test_rotation_smoothed` — synthetic jittery rotvec sequence; output rotvecs are smoother than input.
  - `test_raw_npz_unchanged` — raw NPZ is byte-identical after running stabilisation.
  - `test_sign_fixup` — synthetic trajectory crossing quaternion sign boundary; no artefact in output.
- [x] Mark `docs/development/plans/pending/lock-handmesh-scale-to-median.md` and
      `docs/development/plans/pending/smooth-handmesh-rotation.md` as superseded.

**Files Modified:**
- `code/tests/test_hand_motion_manager.py` — new test class
- `docs/development/plans/pending/lock-handmesh-scale-to-median.md` — superseded note
- `docs/development/plans/pending/smooth-handmesh-rotation.md` — superseded note

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] `test_anchor_preserved` — anchor constraint holds to float precision after stabilisation
- [ ] `test_scale_locked_to_median` — all output scales equal session median
- [ ] `test_scale_filters_out_of_range` — out-of-range scales excluded before median
- [ ] `test_raises_on_no_valid_scales` — `ValueError` when no valid scales remain
- [ ] `test_raises_on_short_session` — `ValueError` with frame count when session too short for filter
- [ ] `test_rotation_smoothed` — output rotvecs are smoother than input
- [ ] `test_raw_npz_unchanged` — raw NPZ byte-identical after stabilisation step
- [ ] `test_sign_fixup` — no artefact when quaternion trajectory crosses sign boundary

### Manual Verification
- [ ] Run `stabilise_hand_motion` on a real session; confirm stabilised NPZ created and raw NPZ unchanged.
- [ ] Render handmesh overlay from stabilised NPZ: no size breathing, no rotational jitter.
- [ ] Verify blue sticker vertex remains on the sticker marker across all frames.
- [ ] Confirm `stabilise_hand_motion` appears in pipeline GUI under Preprocess [Auto].
- [ ] Re-run with `force_processing=False` — step is skipped.
- [ ] Re-run with `force_processing=True` — stabilised NPZ is regenerated.

### Edge Cases
- [ ] Session with degenerate frames (scale = 0 from Procrustes guard): filtered before median; must not raise unless all scales are invalid.
- [ ] Session shorter than `2 × filter_order + 1` frames: `ValueError` with frame count and minimum requirement.
- [ ] NaN quaternion frames: detect before sign-fixup, raise with clear message.
- [ ] Raw NPZ missing `sticker_vertex_indices` key: `KeyError` immediately (fail-fast; re-generate required).

---

## Documentation Plan

- [ ] No CLAUDE.md or README changes required — internal pipeline step.
- [ ] Inline comment in `pose_stabilisation.py` on the sign-fixup rationale (non-obvious
      double-cover constraint) and on why translation is re-derived rather than filtered.

---

## Rollback Plan

The raw NPZ is never modified. To rollback:
1. Delete all `*_handmodel_motion_stabilised.npz` files for affected sessions.
2. Revert the load-preference change in `handmesh_overlay_renderer.py` and
   `preprocess_workflow_kinect_visualisation.py` (4 lines total).
3. Remove the DAG entry and flow wrapper from `preprocess_workflow_kinect_auto.py`
   and `preprocess_workflow_kinect_auto_dag.yaml`.

No data migrations, no schema changes visible to downstream consumers (corrected streams
use identical NPZ keys to the raw file).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Rotation smoothing over-smooths fast hand motion at default 5 Hz cutoff | Med | Med | Exposed as configurable DAG parameter; user can lower cutoff |
| Sign-fixup misses wrap-around edge (e.g. continuous 360° rotation) | Low | High | `test_sign_fixup` covers this; also verify on a session with large rotations |
| Anchor reconstruction float error accumulates across many frames | Very Low | Low | Correction is a single multiply-add per frame; float32 precision is ample for mm-scale coordinates |
| Existing raw NPZs missing `sticker_vertex_indices` key | Certain (first run after Phase 1) | Low | Fail-fast on load with clear message; user must re-run `generate_3d_hand_in_motion` once |

---

## References

- Superseded: `docs/development/plans/pending/lock-handmesh-scale-to-median.md`
- Superseded: `docs/development/plans/pending/smooth-handmesh-rotation.md`
- `HandMotionManager`: `code/src/preprocessing/motion_analysis/hand_tracking/models/hand_motion_manager.py`
- Procrustes scale computation: `_calculate_alignment_procrustes()` lines 310–316
- Anchor translation formula: line 338 (`t = t0 - s * R @ s0`)
- Scale application at render time: `__getitem__()` line 223 (`matrix[:3, :3] *= scale`)
- Filter infrastructure: `code/src/preprocessing/stickers_analysis/xyz/motion_correction/`
- NPZ arrays: `vertices (N,V,3)`, `translations (N,3)`, `rotations (N,4) XYZW`, `scales (N,)`, `timestamps (N,)`, `faces`, `fps`
- Pipeline stage pattern: `generate_3d_hand_in_motion_flow` lines 297–332 and stage entry lines 523–529 in `preprocess_workflow_kinect_auto.py`
- Existing tests: `code/tests/test_hand_motion_manager.py::TestProcrustesScaleGuard`
