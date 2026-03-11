# Plan: Transformed Visualization Mode for Neural-Kinect Viewer

**Date:** 2026-03-11
**Author:** Basil Duvernoy
**Status:** Active
**Branch:** `feature/transformed-visualization-mode`

---

## Overview

**What:** Add a `view_neural_kinect_scene_transformed` task to the existing
visualization DAG that launches the NeuralKinectViewer in a registered-frame mode.
**Why:** Multi-forearm sessions produce ICP-registered coordinates and a unified
forearm, but the viewer currently only shows per-block data in raw camera coordinates.
Researchers need to see all geometry in the common registered frame to verify
spatial alignment across blocks.
**How:** Extend the viewer with optional constructor parameters for a 4x4
registration transform and a unified forearm. The flow script loads these from
existing registration artifacts and passes them to the viewer, which applies the
transform at render time to the MKV cloud, hand mesh, and stickers.

## Problem Statement

The NeuralKinectViewer displays per-block forearm snapshots and raw camera-frame
coordinates. For multi-forearm sessions, the ICP registration pipeline already
produces a unified forearm point cloud and per-block 4x4 transforms, and
downstream tasks (`determine_receptive_field`, `process_unified_touches`) already
consume `_transformed` columns. However, there is no way to **visualize** the
registered data in 3D. This forces researchers to trust the registration purely
from fitness scores, with no spatial verification tool.

## Goals

### In Scope

1. New DAG task `view_neural_kinect_scene_transformed` in the existing visualization DAG and flow script
2. Viewer shows unified registered forearm PLY instead of per-block forearm snapshots
3. Viewer reads `contact_points_transformed` column for contact surface rendering
4. MKV point cloud, hand mesh, and sticker positions are transformed at render time using the block's 4x4 registration matrix
5. Graceful skip (with warning) for single-forearm sessions or missing registration artifacts

### Out of Scope

- Modifying the existing `view_neural_kinect_scene` task behavior
- Adding a toggle to switch between raw/transformed at runtime within the viewer
- Transforming the time-series panel data (Nerve_freq, contact_depth, contact_area are scalar — frame-independent)
- GPU-accelerating the 4x4 transform (the per-frame cost is negligible after AABB crop)

## Success Criteria

- [ ] `view_neural_kinect_scene_transformed` task exists in DAG config (disabled by default)
- [ ] Enabling the task on a multi-forearm session launches the viewer with unified forearm and transformed geometry
- [ ] Enabling the task on a single-forearm session logs a clear warning and skips without error
- [ ] In transformed mode, contact points visually sit on the unified forearm surface
- [ ] MKV cloud, hand mesh, and stickers are spatially coherent with the unified forearm
- [ ] Existing `view_neural_kinect_scene` task is unaffected

---

## Technical Design

### Approach

Add three optional constructor parameters to `NeuralKinectViewer` (`registration_transform`,
`unified_forearm`, `use_transformed_columns`). When provided, these switch the viewer
into transformed mode without changing any existing codepaths. The flow script loads
registration artifacts using existing `ForearmCatalog` methods and resolves the per-block
transform key using the existing `_find_applicable_transform_key` function.

The AABB crop centre stays in the original camera frame (raw MKV data is cropped first,
then transformed), keeping the existing GPU crop optimization intact. Camera focal point
uses the transformed centroid.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Optional params on existing viewer class | Minimal code change; zero risk to existing task; no class proliferation | Viewer constructor grows by 3 params | **Chosen** |
| Subclass `TransformedNeuralKinectViewer` | Clean separation | Deep coupling with base class internals (VTK actors, preloader); fragile to base changes | Rejected |
| Separate viewer script/class | Complete isolation | Massive code duplication (~1800 lines); maintenance burden | Rejected |

### Architecture Changes

No new modules or classes. Changes are additive to existing files:

- **`NeuralKinectViewer`** gains 3 optional params and 4 insertion points in `_update_frame()`
- **Flow script** gains one new function (`run_single_session_pipeline_transformed`)
- **DAG YAML** gains one new task definition

### Knowledge Base Constraints

From `note-forearm-icp-registration.md`:
- Use pre-computed transforms from disk; do not recompute ICP
- Handle single-forearm sessions as no-op (no transforms file exists)
- Log fitness scores when loading transforms

From `note-cupy-import-order.md`:
- The flow script already imports CuPy before preprocessing — no change needed
- The viewer file imports from `preprocessing.*` — new imports for `csv_spatial_transformer`
  must come after the existing CuPy guard (already satisfied by file structure)

From `note-somatosensory-units-and-calculations.md`:
- All coordinates are in mm in both raw and transformed modes; no unit conversion needed

---

## Implementation Plan

### Phase 1: Make `_apply_rigid_transform` Public
**Goal:** Expose the rigid transform helper for external consumers

- [ ] Rename `_apply_rigid_transform` to `apply_rigid_transform` in `csv_spatial_transformer.py`
- [ ] Update the one internal caller in the same file (line 146)

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py` — Rename function, update internal call

**Dependencies:** None

### Phase 2: Viewer Constructor Extension
**Goal:** Accept registration transform, unified forearm, and column-selection flag

- [ ] Add 3 optional keyword parameters to `NeuralKinectViewer.__init__`
- [ ] Store `_registration_transform` and `_use_transformed_columns` as instance vars
- [ ] When `unified_forearm` is provided, set `_forearms_dict = {0: unified_forearm}` (skip catalog)
- [ ] When `_use_transformed_columns` is True, read `contact_points_transformed` column using `resolve_column()`
- [ ] Compute `_transformed_centroid` from registration transform applied to `contact_centroid`
- [ ] Use `_transformed_centroid` for camera focal point (lines 1043, 1456)

**Files Modified:**
- `code/src/merging/gui/neural_kinect_scene_viewer.py` — Constructor params, forearm loading, contact column selection, centroid computation, camera focal point

**Dependencies:** Phase 1

### Phase 3: Render Loop Transform
**Goal:** Apply 4x4 registration transform to per-frame geometry in `_update_frame()`

- [ ] Import `apply_rigid_transform` and `resolve_column` at top of viewer file
- [ ] After `_crop_pointcloud_gpu()` returns `(pts, cols)` (line ~1126): apply transform to `pts`
- [ ] After hand mesh vertex extraction (line ~1182): apply transform to `verts`
- [ ] After sticker position read (line ~1208): apply transform to `pos` and `prev_pos` for compass
- [ ] Contact points (step 5): no transform needed — already from `_transformed` column

**Files Modified:**
- `code/src/merging/gui/neural_kinect_scene_viewer.py` — Four insertion points in `_update_frame()`

**Dependencies:** Phase 2

### Phase 4: Flow Script and DAG Config
**Goal:** Add the new task to the DAG and wire it into the batch dispatcher

- [ ] Add `view_neural_kinect_scene_transformed` task to DAG YAML (enabled: false)
- [ ] Add `run_single_session_pipeline_transformed()` function to flow script
- [ ] Load unified forearm via `ForearmCatalog.get_unified_pointcloud(session_id)` — skip if None
- [ ] Load transforms via `ForearmCatalog.load_registration_transforms(session_id)` — skip if None
- [ ] Resolve per-block transform key via `_find_applicable_transform_key()` — skip if None
- [ ] Launch viewer with the 3 extra params
- [ ] Call both pipeline functions per block in `run_batch_sequentially()`

**Files Modified:**
- `configs/merging_pipeline_neuron_to_kinect_visualisation_dag.yaml` — New task definition
- `code/scripts/merging_pipeline_neuron_to_kinect_visualisation.py` — New function + batch wiring

**Dependencies:** Phase 3

---

## Reusable Utilities (no changes needed beyond Phase 1 rename)

| Utility | Location |
|---------|----------|
| `apply_rigid_transform(points, T)` | `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py:76` |
| `resolve_column(df, base, use_transformed)` | `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py:89` |
| `_find_applicable_transform_key(transforms, stem)` | `code/scripts/_3_preprocessing/_3_forearm_extraction/apply_registration_transform.py:12` |
| `ForearmCatalog.get_unified_pointcloud(session_id)` | `code/src/preprocessing/forearm_extraction/models/forearm_catalog.py:164` |
| `ForearmCatalog.load_registration_transforms(session_id)` | `code/src/preprocessing/forearm_extraction/models/forearm_catalog.py:177` |

---

## Testing Plan

### Manual Verification

- [ ] **Single-forearm session:** Enable `view_neural_kinect_scene_transformed` in DAG config, run on a session with one forearm snapshot. Expect: warning log "no registration transforms", task skipped cleanly, batch continues
- [ ] **Multi-forearm session:** Enable the task, run on a session with ICP registration completed. Expect: viewer opens showing unified forearm, contact points aligned to forearm surface, MKV cloud and hand mesh spatially coherent
- [ ] **Both tasks enabled:** Enable both `view_neural_kinect_scene` and `view_neural_kinect_scene_transformed`. Expect: original viewer launches first (per-block forearms, raw coords), then transformed viewer launches second (unified forearm, registered frame)
- [ ] **Missing merged CSV:** Enable transformed task on a session without merged CSV. Expect: viewer opens in pure-3D mode (no neural overlay), transformed geometry still works

### Edge Cases

- [ ] Block with no own forearm snapshot (relies on preceding block's transform via `_find_applicable_transform_key`)
- [ ] First block in session (canonical reference = identity transform) — geometry should appear unchanged from raw mode
- [ ] Session where `_unified_registered.csv` was used for merging but `_unified_registered.ply` was deleted — should fail gracefully at unified forearm load

---

## Documentation Plan

- [ ] No README/CLAUDE.md changes needed (internal visualization feature)
- [ ] No new user guide needed (DAG config is self-documenting with task description)

---

## Rollback Plan

All changes are additive:
1. The new DAG task defaults to `enabled: false` — disabling it restores original behavior
2. The 3 new viewer constructor params default to `None`/`False` — existing callers are unaffected
3. If rollback needed: revert the feature branch commits; no data migration or state cleanup required

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Transform applied to wrong block (key mismatch) | Low | High | Reuse proven `_find_applicable_transform_key()` with its block-order logic |
| Camera auto-centering in wrong location | Medium | Low | Separate `contact_centroid` (crop) from `_transformed_centroid` (camera) |
| `_apply_rigid_transform` renamed breaks existing imports | Low | Medium | Only internal callers exist in same file; grep to confirm |
| Merged CSV missing `_transformed` columns | Medium | Medium | `resolve_column()` raises `KeyError` with clear message; flow script catches and skips |
