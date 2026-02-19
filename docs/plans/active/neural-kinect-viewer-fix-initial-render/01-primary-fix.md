# Fix Initial Render — Phase 1: Primary Fix (Clipping Range Reset)

## Goal

Add `ResetCameraClippingRange()` to every frame render so that VTK's near/far
clipping planes always encompass the current scene geometry.

---

## Why this fixes the bug

VTK computes clipping planes from the scene's aggregate bounding box at
render time — but only when the camera is "modified" (e.g. by user
interaction).  Programmatic `plotter.render()` calls do **not**
automatically trigger a clipping-range recalculation.

When all initial actors are empty meshes (zero-point `PolyData`), VTK
computes a degenerate clipping range.  Any real geometry added later (point
clouds at `contact_centroid ± 400 mm`) falls outside this range and is
clipped to invisibility.

Calling `renderer.ResetCameraClippingRange()` before `render()` forces VTK
to recompute the near/far planes from the current actor bounds, without
touching camera position/focal/up.

---

## Change

**File:** `code/src/merging/gui/neural_kinect_scene_viewer.py`
**Method:** `NeuralKinectViewer._update_frame`
**Location:** step 5 — the single render call (currently line ~1060-1061)

### Before

```python
        # 5. Single render call -----------------------------------------
        self.plotter.render()
        self._refresh_cam_pos_label()
```

### After

```python
        # 5. Single render call -----------------------------------------
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()
        self._refresh_cam_pos_label()
```

That is the **only** change in this phase.

---

## What `ResetCameraClippingRange` does (VTK internals)

1. Iterates all actors in the renderer's actor collection.
2. For each visible actor, unions its bounding box into an aggregate AABB.
3. Computes `near` and `far` clipping distances from the camera position to
   the nearest/farthest corners of that AABB.
4. Applies a small expansion factor (VTK default: 0.01) to avoid Z-fighting.
5. Sets `camera.SetClippingRange(near, far)`.

**Does NOT change:** camera position, focal point, up vector, view angle,
or parallel scale.

**Performance:** O(N) where N = number of actors (~10 in this viewer).
Sub-microsecond.  Negligible compared to the mesh upload and VTK render pass.

---

## Verify

1. Launch the viewer:
   ```bash
   cd /mnt/f/GitHub/social-touch-semi-controlled
   python code/scripts/view_merged_neural_kinect.py
   ```

2. **Test A — Startup:** Window opens → 3D data should be visible
   immediately (or the background should at least not be blank when moving
   to a frame with data via slider).

3. **Test B — Slider scrub:** Move the slider from frame 0 to frame 10+.
   Data should appear on the first frame that has non-empty point-cloud /
   forearm / sticker data, without any camera interaction.

4. **Test C — Camera unchanged:** After data appears, verify the camera
   position matches the expected startup view (contact centroid, elevated
   front view).  It should NOT have jumped to VTK's default isometric view.

### If Test A fails (startup still blank)

The first few frames likely have all-empty data, so
`ResetCameraClippingRange` has no bounds to work with on those frames.
Proceed to **Phase 2** (bounding proxy).

### If Test B fails (slider frames still invisible)

The issue is not clipping.  Add diagnostic prints:
```python
print(f"[DIAG] clipping range = {self.plotter.camera.clipping_range}")
print(f"[DIAG] camera position = {self.plotter.camera.position}")
print(f"[DIAG] scene bounds = {self.plotter.renderer.ComputeVisiblePropBounds()}")
```
and investigate further.

### If Test C fails (camera jumps)

Verify `camera_set = True` is still set in `_init_actors`.
`ResetCameraClippingRange` should never change camera position — if it does,
something else is calling `reset_camera()`.
