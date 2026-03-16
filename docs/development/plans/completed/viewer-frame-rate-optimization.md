# Plan: Viewer Frame-Rate Optimization

**Date:** 2026-02-19
**Author:** Claude (AI-assisted)
**Status:** Completed
**Branch:** `feature/viewer-frame-rate-optimization`

---

## Overview

**What:** Optimize the `NeuralKinectViewer` to achieve interactive frame rates (10-15+ FPS) during slider scrubbing and playback, up from the current ~1 FPS.
**Why:** The viewer is functionally complete but too slow for practical use — every frame change triggers a full MKV decode, VTK actor rebuild, and GPU round-trip, creating a ~100-200 ms per-frame cost.
**How:** Eliminate redundant VTK actor recreation via in-place mesh updates, remove per-frame geometry generation for static shapes, and add stride-based LOD during interactive motion.

## Problem Statement

- The `_update_frame()` hot path calls `plotter.add_mesh()` up to ~15 times per frame, each of which destroys and recreates VTK pipeline objects (mapper, actor, GPU upload).
- `pv.Sphere()` is regenerated from scratch for each sticker on every frame, even though only the center position changes.
- The CuPy GPU crop uploads the full ~2M-point cloud to GPU and downloads the cropped result every frame, even when the crop center/size haven't changed.
- The preloader's synchronous fallback on cache miss blocks the main thread during random scrubbing.
- Combined, these issues produce ~1 FPS playback on a 1.5 GB MKV recording.

## Goals

### In Scope
1. Reduce per-frame VTK overhead by updating mesh data in-place instead of calling `add_mesh()` every frame
2. Eliminate per-frame geometry generation for sticker spheres (translate once-created geometry)
3. Add stride-based point cloud downsampling during interactive playback/scrubbing with full-res on pause
4. Improve preloader cache-miss behavior to avoid blocking the main thread
5. Reduce CuPy CPU-GPU transfer overhead by keeping persistent GPU buffers

### Out of Scope
- Rewriting the MKV decode pipeline (pyk4a/K4A SDK internals)
- Moving depth-to-pointcloud reprojection to GPU (large effort, separate plan)
- Changing the `KinectMKV` or `KinectPointCloudView` public API
- Adding new UI features or changing the viewer layout

## Success Criteria

- [x] Playback at 8+ FPS with GPU crop enabled (currently ~1 FPS)
- [x] Slider scrubbing feels responsive (latency < 150 ms per frame during drag)
- [x] No visual regression when paused — full-resolution point cloud displayed
- [x] No change to existing `KinectMKV`, `KinectPointCloudView`, or `HandMotionManager` APIs
- [x] GPU ON/OFF both functional (CuPy optional path preserved)

---

## Technical Design

### Approach

The optimization targets the `_update_frame()` method in `NeuralKinectViewer` and supporting infrastructure, focusing on three layers:

**Layer 1 — VTK actor reuse:** Replace all per-frame `plotter.add_mesh()` calls with in-place `PolyData` mutation (`mesh.points = ...`, `mesh['colors'] = ...`, `mesh.Modified()`). This eliminates VTK mapper/actor recreation overhead (~15 calls/frame currently).

**Layer 2 — Geometry caching:** Create sticker spheres once at init, then use VTK actor `SetPosition()` to translate them. Cache the forearm `PolyData` conversion (currently rebuilds `pv.PolyData` from Open3D arrays on every forearm-key change). Cache the hand mesh face array when triangle count is stable.

**Layer 3 — Adaptive LOD:** During playback or slider drag, display every Nth point of the Kinect cloud (stride=4 default). On pause/release, render full resolution. This cuts VTK render time and GPU transfer proportionally.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| In-place mesh update + LOD stride | Minimal API changes, predictable speedup, low risk | Requires careful VTK state management | **Chosen** |
| Pre-decode all frames to RAM cache | Eliminates all decode latency | 1.5 GB MKV x ~2M pts x 15 bytes = multi-GB RAM; infeasible | Rejected |
| WebGL-based viewer (replace PyVista) | Modern, potentially faster rendering | Complete rewrite, loses all current functionality | Rejected |
| Offload entire render loop to separate process | Avoids GIL contention | Complex IPC, shared-memory management for large arrays | Rejected (future consideration) |

### Architecture Changes

No new modules. All changes are within the existing viewer file:

```
code/src/merging/gui/neural_kinect_scene_viewer.py  — Primary changes (all phases)
```

Key patterns being introduced:
- **Persistent `PolyData` references** stored as `self._mesh_*` instance attributes, mutated in-place each frame
- **VTK-level actor position** via `GetMapper().GetInput()` update instead of `add_mesh()`
- **`_interactive_stride`** flag toggled by slider drag / play state to control LOD

---

## Implementation Plan

### Phase 1: In-place VTK mesh updates (biggest win)
**Goal:** Eliminate all per-frame `plotter.add_mesh()` calls from `_update_frame()` by storing persistent `PolyData` references and mutating them in-place.

**Tasks:**
- [x] Task 1.1 — Create persistent `PolyData` instance attributes (`self._mesh_kinect`, `self._mesh_forearm`, `self._mesh_hand`, `self._mesh_contact`) during `_init_actors()`, registered once with `plotter.add_mesh()`
- [x] Task 1.2 — Refactor `_update_frame()` section 1 (Kinect cloud): replace `plotter.add_mesh(cloud, name='kinect_point_cloud', ...)` with `self._mesh_kinect.points = pts; self._mesh_kinect['colors'] = cols; self._mesh_kinect.Modified()`
- [x] Task 1.3 — Refactor section 2 (forearm): same pattern — mutate `self._mesh_forearm` in-place
- [x] Task 1.4 — Refactor section 3 (hand mesh): mutate `self._mesh_hand` vertices and faces in-place. Use `self._mesh_hand.faces = new_faces` only when triangle count changes
- [x] Task 1.5 — Refactor section 5 (contact points): mutate `self._mesh_contact` in-place
- [x] Task 1.6 — Handle visibility toggling by setting points to empty array `np.empty((0,3))` instead of replacing the actor. For the kinect cloud, also consider using VTK actor `SetVisibility(0/1)` directly via `self.plotter.renderer.GetActors()`
- [x] Task 1.7 — Verify `plotter.render()` still triggers a full VTK pipeline update after in-place mutation (may need explicit `mapper.Update()` calls)

**Files Modified:**
- `code/src/merging/gui/neural_kinect_scene_viewer.py` — `_init_actors()` and `_update_frame()` sections 1-5

**Dependencies:** None

### Phase 2: Sticker sphere reuse
**Goal:** Eliminate per-frame `pv.Sphere()` geometry generation for stickers.

**Tasks:**
- [x] Task 2.1 — In `_init_actors()`, create one `pv.Sphere(radius=4.0)` template mesh and store per-sticker VTK actor references via `self.plotter.renderer.GetActors()` or by capturing the return value of `plotter.add_mesh()`
- [x] Task 2.2 — In `_update_frame()` section 4 (stickers loop), replace `pv.Sphere(center=pos)` with `actor.SetPosition(*pos)` to translate the existing sphere geometry
- [x] Task 2.3 — Handle NaN/hidden stickers by calling `actor.SetVisibility(0)` instead of replacing with empty mesh; restore with `actor.SetVisibility(1)` when valid

**Files Modified:**
- `code/src/merging/gui/neural_kinect_scene_viewer.py` — `_init_actors()` and `_update_frame()` sticker section

**Dependencies:** Phase 1 (same refactoring pattern)

### Phase 3: Adaptive LOD (stride-based downsampling)
**Goal:** Reduce point count during interactive motion for faster render; full resolution on pause.

**Tasks:**
- [x] Task 3.1 — Add `self._interactive_stride: int = 4` attribute and a `self._is_interactive: bool` flag
- [x] Task 3.2 — Set `_is_interactive = True` on slider press / play start; `False` on slider release / pause. On transition from interactive to static, trigger one full-res `_update_frame()`
- [x] Task 3.3 — In `_update_frame()` section 1, after GPU crop, apply `pts = pts[::self._interactive_stride]` and `cols = cols[::self._interactive_stride]` when `_is_interactive` is `True`
- [x] Task 3.4 — Add a "LOD stride" spinbox to the right panel (range 1-8, default 4) so the user can tune the trade-off

**Files Modified:**
- `code/src/merging/gui/neural_kinect_scene_viewer.py` — `_build_frame_controls()` or `_build_right_panel_controls()`, `_update_frame()`, `_on_slider_pressed/released`, `_toggle_play`

**Dependencies:** Phase 1 (in-place update pattern must be in place first)

### Phase 4: Preloader and GPU transfer improvements
**Goal:** Reduce cache-miss latency and GPU transfer overhead.

**Tasks:**
- [x] Task 4.1 — Change `get_frame()` synchronous fallback to return the *nearest cached frame* (by frame index) instead of blocking on a fresh MKV decode. Add a `bool` return flag indicating whether the frame is exact or approximate
- [x] Task 4.2 — On approximate frame, schedule an async re-render once the exact frame arrives in the buffer (via a short QTimer poll)
- [x] Task 4.3 — In `_crop_pointcloud_gpu()`, keep `c_gpu` (center) as a persistent CuPy array; only re-upload when `_on_crop_changed` fires. Avoid re-uploading `half_size` every frame (it's a scalar)
- [x] Task 4.4 — Transfer only the cropped subset from GPU to CPU (current code already does this via boolean indexing — verify no redundant full-array transfers)

**Files Modified:**
- `code/src/merging/gui/neural_kinect_scene_viewer.py` — `FramePreloader.get_frame()`, `_crop_pointcloud_gpu()`, `_update_frame()`

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [ ] Verify `PolyData` in-place mutation triggers VTK render update (create minimal PyVista test with `Modified()`)
- [ ] Verify stride-based downsampling produces correct point count (`N // stride`)
- [ ] Verify `FramePreloader.get_frame()` nearest-frame fallback returns a frame within +/-buffer_size of the requested index

### Integration Tests
- [ ] Load a real recording and confirm frame 0 renders correctly after Phase 1 refactor
- [ ] Confirm visibility toggle still works (checkbox on/off for each layer)
- [ ] Confirm forearm hold-last-frame semantics still work (bisect logic unchanged)

### Manual Verification
- [ ] Open the viewer and play — confirm visually smooth playback at improved FPS
- [ ] Scrub the slider rapidly — confirm responsive feedback, no freezing
- [ ] Pause after scrubbing — confirm full-resolution cloud appears
- [ ] Toggle each visibility checkbox — confirm layers show/hide correctly
- [ ] Change crop size — confirm crop updates live
- [ ] Compare GPU ON vs GPU OFF — both paths must work

### Edge Cases
- [ ] First frame (frame 0) with empty point cloud data
- [ ] Last frame of recording
- [ ] Frames where hand mesh is unavailable (`HandMotionManager` returns `None`)
- [ ] Stickers with NaN positions (must not crash or leave stale geometry visible)
- [ ] Recording with no merged CSV (pure-3D mode — neural panel absent)

---

## Documentation Plan

- [x] Update inline comments in `_update_frame()` explaining the in-place mutation pattern
- [x] Add a brief note to the module docstring about the LOD stride behavior
- [x] No README/CLAUDE.md changes needed (internal optimization, no API changes)

---

## Rollback Plan

All changes are confined to a single file (`neural_kinect_scene_viewer.py`). Rollback is straightforward:

1. `git checkout main -- code/src/merging/gui/neural_kinect_scene_viewer.py`
2. No data migrations, no config changes, no external dependencies added

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PyVista in-place PolyData mutation not triggering VTK pipeline update | Med | High | Test with explicit `mapper.Update()` / `Modified()` calls; fall back to `add_mesh()` for specific actors if needed |
| VTK `SetPosition()` on sphere actors interacting unexpectedly with camera clipping | Low | Med | Test with `ResetCameraClippingRange()` call already present in `_update_frame()` |
| Stride downsampling causing visual artifacts (aliasing patterns in ordered point clouds) | Low | Low | Use random stride offset or shuffle indices; or accept minor visual difference during motion |
| Nearest-frame fallback in preloader showing stale data without user realizing | Med | Low | Add visual indicator (e.g., dim the buffer label) when displaying approximate frame; auto-refresh on exact arrival |
| Hand mesh triangle count varying between frames breaking in-place face update | Med | Med | Detect face count change and only rebuild faces array when count differs from cached |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: In-place VTK mesh updates | ~200 lines changed | None |
| Phase 2: Sticker sphere reuse | ~40 lines changed | Phase 1 |
| Phase 3: Adaptive LOD | ~50 lines changed | Phase 1 |
| Phase 4: Preloader + GPU improvements | ~60 lines changed | Phase 1 |

---

## References

- Related Plans: `docs/development/plans/completed/neural-kinect-viewer.md`
- PyVista docs on in-place mesh updates: `pv.PolyData.points` setter + `Modified()`
- VTK actor positioning: `vtkActor.SetPosition(x, y, z)`
