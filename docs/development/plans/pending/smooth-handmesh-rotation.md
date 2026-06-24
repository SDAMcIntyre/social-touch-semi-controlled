# Plan: Smooth Hand-Mesh Pose Stream

**Date:** 2026-05-12
**Author:** Basil Duvernoy
**Status:** Draft
**Base Branch:** `fix/contact-detection-false-positives`
**Branch:** `feature/smooth-handmesh-rotation`

---

> **Superseded** by `docs/development/plans/active/stabilise-handmesh-pose.md` (2026-05-12). This plan is no longer active.

## Overview

The MANO hand-mesh shows visible rotational vibration between frames caused by
Procrustes alignment sensitivity to residual noise in the 3 XYZ sticker anchors.
This plan adds a post-alignment `smooth_hand_motion` pipeline step that filters
the saved rotation stream (in rotation-vector space), translation, and scale
of the hand motion NPZ and writes a smoothed copy alongside the original.

---

## Problem Statement

`HandMotionManager` aligns the mesh frame-by-frame independently using only
3 sticker anchor points. Even after the upstream Butterworth 6 Hz sticker filter,
residual noise in those anchors propagates through Procrustes and appears as
rotational jitter across rendered frames. There is currently no temporal
regularisation at the pose level. This degrades the visual quality of handmesh
overlay videos.

---

## Goals

### In Scope
1. Temporal smoothing of the rotation stream via rotation-vector space (correct
   for quaternion double-cover).
2. Temporal smoothing of the translation and scale streams.
3. Smoothed output saved as `*_handmodel_motion_smooth.npz` alongside the
   original NPZ (raw file untouched).
4. `smooth_hand_motion` step wired into the preprocess auto DAG config and
   pipeline runner, with filter parameters configurable from YAML.
5. Handmesh overlay renderer and visualisation script prefer the smooth NPZ
   when it exists.

### Out of Scope
- Changing the Butterworth cutoff on upstream sticker XYZ filtering.
- Smoothing mesh vertices directly (only pose parameters are smoothed).
- Scale stabilisation (handled upstream by `HandMotionManager.lock_scale_to_median()`
  in `lock-handmesh-scale-to-median.md` — scale is constant before this step runs).
- Changes to the contact-detection or somatosensory pipeline.
- Manual DAG (`preprocess_workflow_kinect_manual_dag.yaml`) — it does not
  invoke `generate_3d_hand_in_motion`.

---

## Success Criteria

- [ ] Rotational jitter is visibly gone or clearly reduced in the handmesh overlay.
- [ ] Raw NPZ (`*_handmodel_motion.npz`) is byte-for-byte unchanged after the step runs.
- [ ] `smooth_hand_motion` appears in the DAG launcher GUI under the preprocess pipeline.
- [ ] `force_processing=False` skips smoothing when the smooth NPZ already exists.
- [ ] Filter method, cutoff, and order are all configurable from the DAG YAML.

---

## Technical Design

### Approach

**Rotation smoothing via rotation-vector space**

Quaternions have a double-cover: `q` and `-q` represent the same rotation, but
component-wise filtering across a sign flip produces incorrect results. Correct
approach:

1. **Sign-consistency fixup** — Walk the XYZW quaternion array; flip `q[i]` when
   `q[i] · q[i-1] < 0` to keep all quaternions in the same hemisphere.
2. **XYZW → rotation vector** —
   `scipy.spatial.transform.Rotation.from_quat(quats).as_rotvec()` → `(N, 3)`.
   This space is locally Euclidean for small rotations and has no sign ambiguity.
3. **Per-component filtering** — Apply the chosen filter to each of the 3 components
   independently, reusing `MotionFilterFactory` from the existing sticker motion
   correction infrastructure.
4. **Rotation vector → XYZW** — `Rotation.from_rotvec(smoothed).as_quat()`.

Translation (3 components) is filtered directly with the same interface,
identical to the sticker motion correction pattern. Scale is excluded from
temporal smoothing because it is already locked to a constant median value
by `HandMotionManager.lock_scale_to_median()` (applied during
`generate_3d_hand_in_motion`, see `lock-handmesh-scale-to-median.md`).

**Output naming:** `<stem>_handmodel_motion_smooth.npz` alongside the existing
`<stem>_handmodel_motion.npz`.

**GUI registration:** No change required. The DAG launcher reads `tasks:` from the
YAML dynamically; adding the task entry to the YAML is sufficient for it to appear
in the pipeline GUI.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Rotation-vector smoothing | Correct; reuses existing filter infra | Slightly more code than naive quat filtering | Chosen |
| Component-wise quaternion smoothing | Trivial | Double-cover artefacts at sign flips | Rejected |
| SLERP-based sliding window | Geometrically exact | No existing infra; much harder to implement | Rejected |
| Tighten upstream sticker filter | Addresses root cause | Risks over-smoothing sticker data used by contact detection | Rejected |
| Kalman filter on pose | Handles uncertainty | No existing infra; overkill for offline batch | Rejected |

### Architecture Changes

**New files:**
```
code/src/preprocessing/motion_analysis/hand_tracking/
└── pose_smoothing.py        — PoseSmoothing class

code/scripts/_3_preprocessing/_2_hand_tracking/
└── smooth_hand_motion.py    — pipeline entry-point script
```

**Modified files:**
```
code/src/preprocessing/motion_analysis/hand_tracking/__init__.py
    — export PoseSmoothing

configs/preprocess_workflow_kinect_auto_dag.yaml
    — add smooth_hand_motion task after generate_3d_hand_in_motion

code/scripts/preprocess_workflow_kinect_auto.py
    — add smooth_hand_motion_flow() @flow wrapper and pipeline stage entry
      (after generate_3d_hand_in_motion_flow, ~line 297 / stage ~line 529)

code/scripts/preprocess_workflow_kinect_visualisation.py
    — update hand_motion_path construction (line 143) to prefer smooth NPZ

code/src/preprocessing/motion_analysis/hand_tracking/handmesh_overlay_renderer.py
    — prefer *_handmodel_motion_smooth.npz at load time
```

**Reused infrastructure:**
- `MotionFilterFactory` — `code/src/preprocessing/stickers_analysis/xyz/motion_correction/motion_filter_factory.py`
- `ButterworthFilter` / `SavgolFilter` — same package
- `HandMotionManager.load()` / numpy NPZ I/O — `hand_motion_manager.py`

---

## Implementation Plan

### Phase 1: Core smoothing logic
**Goal:** Implement `PoseSmoothing` in the hand_tracking package.

- [ ] Create `pose_smoothing.py` with `PoseSmoothing` class:
  - `smooth(translations, rotations_xyzw, fps, filter_method, filter_params) → (translations, rotations_xyzw)`
  - Sign-fixup loop on quaternions before rotvec conversion
  - `scipy.spatial.transform.Rotation` round-trip for rotations
  - `MotionFilterFactory` for per-component filtering of both streams
  - Raise `ValueError` if `len(translations) < min_frames_required` for the chosen filter
- [ ] Export `PoseSmoothing` from `__init__.py`

**Files Modified:**
- `code/src/preprocessing/motion_analysis/hand_tracking/pose_smoothing.py` — new
- `code/src/preprocessing/motion_analysis/hand_tracking/__init__.py` — add export

**Dependencies:** None

### Phase 2: Pipeline script
**Goal:** Integrate `PoseSmoothing` as a standalone pipeline script.

- [ ] Create `smooth_hand_motion.py`:
  - Load `*_handmodel_motion.npz` (raise `FileNotFoundError` if missing — fail-fast)
  - Call `PoseSmoothing.smooth(...)` with DAG-configured filter options
  - Save smoothed arrays + unchanged `faces`, `timestamps`, `fps` to
    `*_handmodel_motion_smooth.npz`
  - Honour `force_processing=False` idempotency guard (skip if output exists)

**Files Modified:**
- `code/scripts/_3_preprocessing/_2_hand_tracking/smooth_hand_motion.py` — new

**Dependencies:** Phase 1

### Phase 3: DAG and pipeline wiring
**Goal:** Wire the step into the preprocess auto DAG config and runner.

- [ ] Add `smooth_hand_motion` task to `configs/preprocess_workflow_kinect_auto_dag.yaml`
  after `generate_3d_hand_in_motion`:
  ```yaml
  smooth_hand_motion:
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
- [ ] Add `smooth_hand_motion_flow()` `@flow` wrapper in
  `preprocess_workflow_kinect_auto.py` following the exact pattern of
  `generate_3d_hand_in_motion_flow` (lines 297–332):
  - Receives `hand_motion_npz_path: Path`, `output_dir: Path`, filter options
  - Returns `smoothed_npz_path: Path`
- [ ] Add pipeline stage entry after line 529:
  - `"name": "smooth_hand_motion"`
  - `"params"` lambda reads `context.get("hand_motion_npz_path")` and
    `dag_handler.get_task_options("smooth_hand_motion")` for filter config
  - `"outputs": ["smoothed_hand_motion_npz_path"]`

**Files Modified:**
- `configs/preprocess_workflow_kinect_auto_dag.yaml`
- `code/scripts/preprocess_workflow_kinect_auto.py`

**Dependencies:** Phase 2

### Phase 4: Renderer and visualisation integration
**Goal:** Have rendering consumers prefer the smoothed NPZ.

- [ ] `handmesh_overlay_renderer.py` — update NPZ load logic to check for
  `*_handmodel_motion_smooth.npz` first, fall back to `*_handmodel_motion.npz`,
  raise if neither exists.
- [ ] `preprocess_workflow_kinect_visualisation.py` (line 143) — update
  `hand_motion_path` construction with the same preference logic.

**Files Modified:**
- `code/src/preprocessing/motion_analysis/hand_tracking/handmesh_overlay_renderer.py`
- `code/scripts/preprocess_workflow_kinect_visualisation.py`

**Dependencies:** Phase 2

---

## Testing Plan

### Manual Verification
- [ ] Run `smooth_hand_motion` on a real session's NPZ; confirm smooth NPZ is created
  and raw NPZ is unchanged.
- [ ] Render handmesh overlay from the smooth NPZ and compare to the raw: rotational
  jitter should be visibly absent.
- [ ] Re-run with `force_processing=False` when smooth NPZ exists — confirm step is
  skipped.
- [ ] Set `force_processing=True` and confirm regeneration.
- [ ] Confirm `smooth_hand_motion` task appears in the pipeline GUI task list for
  the Kinect [Auto] workflow.

### Edge Cases
- [ ] Session with NaN frames in sticker positions — verify NaN quaternions don't
  crash sign-fixup (handle or raise with clear message).
- [ ] Session shorter than `2 × filter_order + 1` frames — must raise `ValueError`
  with frame count and minimum requirement.
- [ ] Input NPZ missing — must raise `FileNotFoundError` immediately.

---

## Documentation Plan

- [ ] No CLAUDE.md or README changes required — this is an internal pipeline step.
- [ ] Inline comment in `pose_smoothing.py` on the sign-fixup rationale (non-obvious
  double-cover constraint).

---

## Rollback Plan

The raw NPZ is never modified. Rollback = delete `*_handmodel_motion_smooth.npz`
files and revert the four-line load-order change in the renderer and visualisation
script. No data migrations, no schema changes, no downstream breakage.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Quaternion sign-flip artefact not caught by fixup (wrap-around edge) | Low | High | Test with synthetic trajectory that crosses sign boundary |
| Butterworth `filtfilt` min-length crash on short sessions | Low | Med | Raise `ValueError` with frame count before calling filter |
| Over-smoothing hides real fast hand motion at 5 Hz cutoff | Med | Med | Default is conservative; exposed as configurable DAG parameter |

---

## References

- Related plan: `docs/development/plans/pending/fix-contact-detection-false-positives.md`
- `HandMotionManager`: `code/src/preprocessing/motion_analysis/hand_tracking/models/hand_motion_manager.py`
- Existing filter infra: `code/src/preprocessing/stickers_analysis/xyz/motion_correction/`
- NPZ arrays: `vertices (N,V,3)`, `translations (N,3)`, `rotations (N,4) XYZW`,
  `scales (N,)`, `timestamps (N,)`, `faces`, `fps` — saved in `generate_3d_hand_in_motion.py` lines 134–146
- Rotation convention: XYZW stored; converted via `np.roll(q, 1)` to WXYZ for `trimesh`
- Pipeline stage pattern: `generate_3d_hand_in_motion_flow` lines 297–332 and stage entry lines 523–529 in `preprocess_workflow_kinect_auto.py`
