# Fix: 3D viewport blank on startup — Implementation Plan

## Bug Reference

`docs/bugs/neural-kinect-viewer-initial-render.md`

## Problem

When `NeuralKinectViewer` opens, the 3D viewport is blank. Slider changes
have no visible effect. Data only appears after the user manually
zooms/pans/rotates the VTK viewport (any mouse interaction that triggers a
VTK camera event).

## Root Cause

**VTK clipping planes become degenerate when all initial actors are empty.**

The failure chain:

1. `_init_actors` registers all dynamic actors as **empty** `PolyData`
   (0 points). The only geometry with real bounds is a tiny `Sphere(radius=2)`
   at the origin.
2. The camera is positioned at `contact_centroid ± 400 mm` and
   `camera_set = True` locks it.
3. VTK internally computes **near/far clipping planes** from the scene's
   aggregate bounding box.  With all-empty actors the bounds are near-zero,
   so the clipping range collapses to an extremely thin slab.
4. `_deferred_start` calls `_update_frame(0)`.  If frame 0 (or 1) has
   None / empty data — which is common at recording start — every actor is
   replaced with another empty mesh.  The clipping range stays degenerate.
5. User moves the slider to frame N where real data exists.
   `_update_frame(N)` replaces actors with real geometry hundreds of
   millimetres from the origin, calls `plotter.render()` — but the geometry
   falls **outside the stale near/far clipping planes** → invisible.
6. User zooms or pans → VTK's interactor style internally calls
   `ResetCameraClippingRange()` → clipping planes expand to encompass the
   real geometry → everything appears.

The earlier hypothesis (render-window size `(0, 0)` / timing) may also
contribute at startup but is **not** the primary cause: even after the
`showEvent` timing fix, frames with valid data remain invisible until camera
interaction.

---

## Fix

### Primary fix — reset clipping range in `_update_frame`

**File:** `code/src/merging/gui/neural_kinect_scene_viewer.py`
**Method:** `_update_frame`, step 5 (the single render call, line ~1060)

Add `renderer.ResetCameraClippingRange()` before the existing `render()`:

```python
# 5. Single render call -----------------------------------------
self.plotter.renderer.ResetCameraClippingRange()
self.plotter.render()
```

This recalculates near/far from the current scene bounds **every frame**,
guaranteeing that newly-added geometry is never clipped.  It does **not**
move the camera position, focal point, or up vector — only the clipping
frustum depth is adjusted.

Performance cost: negligible.  `ResetCameraClippingRange` iterates the
actor list once to find the aggregate bounding box — O(number of actors),
which is ~10 actors here.

### Secondary fix — seed `_init_actors` with a real bounding proxy

**File:** same
**Method:** `_init_actors`, after the camera setup (line ~912)

Add a temporary invisible bounding-box actor at startup so VTK has sensible
initial clipping even before any real frame is loaded:

```python
# Invisible bounding proxy — gives VTK a plausible initial clipping
# range so that any geometry near contact_centroid is visible from the
# first render.  Replaced on the first real _update_frame call.
cx, cy, cz = self.contact_centroid.tolist()
h = self._crop_half_size
_bounds_proxy = pv.Box(bounds=(
    cx - h, cx + h,
    cy - h, cy + h,
    cz - h, cz + h,
))
self.plotter.add_mesh(
    _bounds_proxy, opacity=0.0, name='_bounds_proxy', pickable=False,
)
```

Then in `_update_frame`, remove the proxy once real data has been rendered
(one-time cleanup):

```python
# At the top of _update_frame, after setting self.current_index:
if self.plotter.renderer.HasViewProp(
    self.plotter.renderer.GetActors().GetLastActor()
):
    try:
        self.plotter.remove_actor('_bounds_proxy')
    except Exception:
        pass
```

**Simpler alternative:** skip the proxy entirely if the primary
`ResetCameraClippingRange()` fix alone is sufficient (test first).

### Tertiary fix — keep the `showEvent` timing improvement

The `showEvent`-based deferral from Phase 2 remains a good improvement for
ensuring the VTK render window has valid pixel dimensions.  Keep it as-is.

---

## Phases

### Phase 1 — Apply primary fix + verify
**Detail:** `docs/plans/active/neural-kinect-viewer-fix-initial-render/01-primary-fix.md`
- Add `ResetCameraClippingRange()` before `plotter.render()` in `_update_frame`
- Test if this alone resolves the blank viewport

### Phase 2 — Apply bounding proxy if needed
**Detail:** `docs/plans/active/neural-kinect-viewer-fix-initial-render/02-bounding-proxy.md`
- Only if Phase 1 alone is insufficient for the very first frame (all-empty
  data on frames 0-1 means the first `ResetCameraClippingRange` still has
  no bounds to work with)
- Add invisible `pv.Box` proxy seeded at `contact_centroid ± crop_half_size`

### Phase 3 — Verify & clean up
**Detail:** `docs/plans/active/neural-kinect-viewer-fix-initial-render/03-verify.md`
- Manual test checklist (8 scenarios)
- Remove diagnostic instrumentation
- Remove stale investigation doc

---

## Files Modified

| File | Phase | Change |
|---|---|---|
| `code/src/merging/gui/neural_kinect_scene_viewer.py` | 1, 2 | Add clipping reset + optional bounding proxy |

## Files Removed After Fix

| File | Phase |
|---|---|
| `docs/bugs/neural-kinect-viewer-initial-render.md` | 3 |

---

## Architecture Impact

None. The fix adds one VTK call per frame (`ResetCameraClippingRange`) and
optionally one invisible startup actor. No new classes, no new files, no API
changes.
