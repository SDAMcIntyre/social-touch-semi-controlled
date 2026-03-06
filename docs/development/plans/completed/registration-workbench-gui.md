# Plan: Registration Workbench GUI

**Date:** 2026-03-05
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/multi-snapshot-forearm-registration`

---

## Context

The forearm registration pipeline currently separates processing (`register_session_forearms()`) from inspection (`show_registration_viewer()`). The user must re-run the script to try different parameters (canonical key, method, unification mode). The existing viewer shows a single 3D scene with toggle-based before/after comparison, making it hard to compare original vs. transformed simultaneously.

This plan introduces a `RegistrationWorkbench` class that unifies parameter controls and dual-viewport visualization in a single interactive GUI, enabling rapid parameter exploration without script restarts.

---

## Overview

A new `RegistrationWorkbench` class providing:
- **Top panel (20%):** dropdowns for `unification_mode`, `canonical_key`, `registration_method` + green "Process" button
- **Bottom panel (80%):** snapshot selector dropdown + two synchronized side-by-side 3D viewports (left: unified white + original red, right: unified white + transformed green)

## Problem Statement

Iterating on registration parameters requires editing script arguments and re-running the full pipeline. The existing viewer (`show_registration_viewer()`) uses a single 3D scene with toggled overlays, preventing simultaneous before/after comparison. This slows down parameter tuning and makes quality assessment harder.

## Goals

### In Scope
1. New `RegistrationWorkbench` class in the registration subpackage
2. Dual synchronized 3D viewports with camera persistence across snapshot/reprocessing changes
3. Interactive parameter controls triggering in-GUI re-registration
4. Standalone `__main__` block with fake point cloud data for testing

### Out of Scope
- Replacing `show_registration_viewer()` (it remains for the non-interactive pipeline path)
- Integrating into `register_session_forearms()` (future work)
- Off-thread registration processing (v1 blocks the GUI thread during ICP)
- Saving/exporting results from the GUI

## Success Criteria

- [ ] GUI launches with fake data via `python registration_workbench.py`
- [ ] Dropdowns populate correctly and "Process" triggers registration
- [ ] Left viewport shows unified (white) + original (red), right shows unified (white) + transformed (green)
- [ ] Camera rotation on one viewport is mirrored on the other
- [ ] Changing snapshots or reprocessing does not reset the camera view

---

## Technical Design

### Approach

Build on the existing Open3D native GUI patterns already used in `registration_viewer.py` and `visualize_point_cloud_comparison.py`. The dual-viewport + camera sync pattern is proven in `visualize_point_cloud_comparison.py`. The parameter panel uses `gui.Combobox` (Open3D's dropdown widget).

### Architecture Constraints (from knowledge base)

**`note-open3d-scenewidget-layout.md` -- CRITICAL:**
- `SceneWidget` must be a **direct child of the Window**, not nested in `gui.Vert`/`gui.Horiz`
- `SceneWidget.frame` must be set **explicitly** in `set_on_layout`
- Use `calc_preferred_size()` on text panels; derive scene space from remainder

**`note-forearm-icp-registration.md`:**
- `ForearmRegistrator` is the reusable ICP class -- no need to reimplement registration logic
- Canonical reference = lowest `representative_frame_id` (auto-select default)

### Window Children (all direct)

```
Window
  +-- top_panel         (gui.Vert)     -- parameter controls + process button
  +-- snapshot_panel    (gui.Horiz)    -- snapshot dropdown + fitness label
  +-- scene_left        (SceneWidget)  -- unified white + original red
  +-- scene_right       (SceneWidget)  -- unified white + transformed green
```

### Camera Synchronization

Reuse the polling pattern from `visualize_point_cloud_comparison.py` (lines 132-201):
- A `sync_loop` function is repeatedly posted to main thread via `post_to_main_thread()`
- Each iteration reads both cameras' model matrices, detects which changed, applies the changed pose to the other via `camera.look_at()`
- Projection is also synced via `set_projection()` to handle zoom

### Camera Persistence

Before any geometry rebuild: save `scene.camera.get_model_matrix()` to `self._saved_camera`.
After rebuild: restore via `look_at()` decomposition instead of calling `setup_camera()` (which resets to bounding-box default). Only call `setup_camera()` on first display.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Open3D native GUI | Consistent with existing viewers, no extra deps | Combobox is new (untested in codebase) | **Chosen** |
| PyQt5 + PyVista | Proven scene_viewer.py pattern, richer widgets | Mixes frameworks; registration module is pure Open3D | Rejected |
| Extend `show_registration_viewer()` | Less code | Single-scene design doesn't support side-by-side | Rejected |

---

## Implementation Plan

### Phase 1: Skeleton + Layout
**Goal:** Window with 4 direct children, correct layout, no functionality

- [ ] Create `registration_workbench.py` with imports and constants
- [ ] Define `RegistrationWorkbench.__init__(clouds)` storing input data
- [ ] Implement `_build_gui()` creating all widgets (dropdowns, button, two SceneWidgets)
- [ ] Implement `_on_layout()` with explicit frame computation for all 4 children
- [ ] Implement `show()` initializing app + calling `app.run()`
- [ ] Add `__main__` block with 3 fake Gaussian point clouds

**Files:**
- `code/src/preprocessing/forearm_extraction/registration/registration_workbench.py` -- **new file**

### Phase 2: Processing + Geometry
**Goal:** "Process" button runs ICP and populates viewports

- [ ] Implement `_on_process_clicked()`: read dropdown values, run `ForearmRegistrator`, store results
- [ ] Implement `_rebuild_geometries()`: clear scenes, upload unified (white) + all snapshot variants (hidden)
- [ ] Implement `_on_snapshot_changed()`: show/hide correct base/xf geometry per scene
- [ ] Populate snapshot dropdown after processing with keys + fitness scores

**Key dependency:** `ForearmRegistrator` from `forearm_registrator.py`

### Phase 3: Camera Sync + Persistence
**Goal:** Synchronized viewports that survive snapshot/reprocessing changes

- [ ] Implement `_start_sync_loop()` adapted from `visualize_point_cloud_comparison.py`
- [ ] Add camera save/restore around `_rebuild_geometries()`
- [ ] Verify camera is not reset on snapshot change (geometry show/hide, not add/remove)

### Phase 4: Export + Polish
**Goal:** Module integration and final testing

- [ ] Update `registration/__init__.py` to export `RegistrationWorkbench`
- [ ] Test with fake data: launch, process, switch snapshots, rotate, reprocess, verify camera persists

**Files:**
- `code/src/preprocessing/forearm_extraction/registration/__init__.py` -- add export

---

## Key Code Patterns to Reuse

| Pattern | Source File | Lines |
|---------|-----------|-------|
| Color constants + material setup | `registration_viewer.py` | 28-75 |
| Geometry naming (`_geo_base`/`_geo_xf`) | `registration_viewer.py` | 44-48 |
| Show/hide geometry toggle | `registration_viewer.py` | 86-95 |
| Dual SceneWidget layout | `visualize_point_cloud_comparison.py` | 56-130 |
| Camera sync polling loop | `visualize_point_cloud_comparison.py` | 132-201 |
| `apply_pose_to_camera()` decomposition | `visualize_point_cloud_comparison.py` | 140-150 |
| `ForearmRegistrator` API | `forearm_registrator.py` | full |

---

## Testing Plan

### Manual Verification
- [ ] `python registration_workbench.py` launches without errors
- [ ] Three dropdowns display correct options (mode, canonical, method)
- [ ] Click "Process" -- both viewports populate with geometry
- [ ] Snapshot dropdown shows all keys with fitness scores
- [ ] Switching snapshots: left shows red overlay, right shows green overlay, camera unchanged
- [ ] Rotate in left viewport -- right follows; rotate in right -- left follows
- [ ] Click "Process" again with different method -- camera view preserved, geometries updated
- [ ] Change canonical key and reprocess -- results update correctly

### Edge Cases
- [ ] Single snapshot in clouds dict -- process should handle gracefully (registrator skip)
- [ ] Rapid snapshot switching -- no geometry corruption
- [ ] Window resize -- layout recomputes correctly, scenes fill available space

---

## Documentation Plan

- [ ] Inline docstrings on `RegistrationWorkbench` class and public `show()` method
- [ ] Update knowledge base if new Open3D GUI patterns are discovered (e.g., Combobox usage)

---

## Rollback Plan

1. Delete `registration_workbench.py`
2. Remove export from `registration/__init__.py`
3. No other files are modified -- zero risk to existing pipeline

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `gui.Combobox` API differs from docs | Low | High | Verify in `__main__` block first; fallback to radio checkboxes if needed |
| ICP freeze on GUI thread (large clouds) | Med | Med | Accept for v1; status label warns "Processing..."; future: off-thread |
| `clear_geometry()` resets camera | Med | Med | Save/restore camera matrix around rebuilds |
| Sync loop CPU overhead | Low | Low | Polling is lightweight (matrix comparison only); proven pattern |

---

## References

- Active plan: `docs/development/plans/active/multi-snapshot-forearm-registration.md`
- Knowledge base: `docs/development/knowledge-base/note-open3d-scenewidget-layout.md`
- Knowledge base: `docs/development/knowledge-base/note-forearm-icp-registration.md`
- Camera sync reference: `code/src/utils/gui/visualize_point_cloud_comparison.py`
