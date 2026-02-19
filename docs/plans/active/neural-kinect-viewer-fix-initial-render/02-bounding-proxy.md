# Fix Initial Render — Phase 2: Bounding Proxy (if Phase 1 alone is insufficient)

## When to apply this phase

Only if Phase 1 (clipping reset) passes Test B (slider scrub works) but
fails Test A (startup is still blank).

This means: `ResetCameraClippingRange()` works once there is real geometry,
but the very first frames (0, 1, ...) have all-empty data.  With no actor
bounds, `ResetCameraClippingRange` has nothing to compute from and the
clipping range stays degenerate until the first non-empty frame.

---

## Strategy

Seed the scene with an **invisible bounding-box actor** centred on
`contact_centroid` with the crop half-size as extent.  This gives VTK a
plausible bounding box from the very first render, even when all real data
actors are empty.

The proxy is invisible (`opacity=0.0`) so it never appears on screen.
It is removed on the first frame that contains real data.

---

## Changes

**File:** `code/src/merging/gui/neural_kinect_scene_viewer.py`

### Change 1 — Add proxy in `_init_actors`

**Location:** end of `_init_actors`, after `self.plotter.camera_set = True`
(line ~912), before the `AddObserver` call.

```python
        # Invisible bounding proxy — gives VTK a plausible clipping range
        # even when frames 0-N have all-empty data.  Removed by
        # _update_frame once real geometry is loaded.
        self._bounds_proxy_active = True
        _bounds_proxy = pv.Box(bounds=(
            cx - self._crop_half_size, cx + self._crop_half_size,
            cy - self._crop_half_size, cy + self._crop_half_size,
            cz - self._crop_half_size, cz + self._crop_half_size,
        ))
        self.plotter.add_mesh(
            _bounds_proxy, opacity=0.0, name='_bounds_proxy', pickable=False,
        )
```

Note: `cx, cy, cz` are already defined on line 907 from
`self.contact_centroid.tolist()`.

### Change 2 — Remove proxy once real data appears in `_update_frame`

**Location:** top of `_update_frame`, after `self.current_index = frame_idx`
(line ~933).

```python
        # Remove the invisible bounding proxy once any real actor has data,
        # so it no longer inflates the scene bounds unnecessarily.
        if self._bounds_proxy_active:
            has_real_data = False

            # Check if kinect cloud has points
            if self._visibility.get('kinect_point_cloud', True):
                pc_data = self._preloader.get_frame(frame_idx)
                if (pc_data is not None
                        and pc_data.points is not None
                        and pc_data.points.shape[0] > 0):
                    has_real_data = True

            if has_real_data:
                self._bounds_proxy_active = False
                try:
                    self.plotter.remove_actor('_bounds_proxy')
                except Exception:
                    pass
```

**Important:** This check must **not** consume the preloader frame.  Since
`get_frame` returns a cached object (no side effect), this is safe.  The
same `pc_data` will be fetched again in step 1 of `_update_frame`.

### Change 3 — Init flag in `__init__`

Not needed — `_bounds_proxy_active` is set in `_init_actors`, which is
called from `__init__`.

---

## Verify

1. Launch the viewer.
2. **Frame 0 should not be blank.**  Even if frame 0 has no real data, the
   background should show the axes widget and origin sphere (proving VTK is
   rendering with valid clipping).
3. Move slider to frame 2+ → real data appears instantly.
4. Verify the invisible box is not visible (no wireframe cube in scene).
5. After moving past the first valid frame, verify the proxy has been removed
   (optional: add a one-time `print("[DEBUG] bounds proxy removed")` to
   confirm).

---

## If this is still insufficient

If the viewport remains blank even with the bounding proxy:

1. Check `self.contact_centroid` — if it contains `NaN`, the proxy box
   bounds will be `NaN` and VTK will ignore it.  Fix:
   ```python
   if np.any(np.isnan(self.contact_centroid)):
       self.contact_centroid = np.array([0.0, 0.0, 500.0])
   ```

2. Check whether `opacity=0.0` prevents VTK from including the actor in
   `ComputeVisiblePropBounds()`.  If so, use `opacity=0.001` (effectively
   invisible but technically "visible" to VTK's bound computation).
