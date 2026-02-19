# Bug: 3D viewport is blank until user interacts with the camera

**Component:** `code/src/merging/gui/neural_kinect_scene_viewer.py`
**Class:** `NeuralKinectViewer`
**Reported:** 2026-02-19
**Status:** Fix in progress (`fix/initial-render-clipping`)

---

## Symptom

When the Neural-Kinect Viewer window first opens, the 3D viewport (PyVista /
VTK area) is entirely blank — no point cloud, no forearm, no hand mesh, no
stickers.  Moving the frame slider does not help.  Data only appears after
the user **zooms or rotates** the 3D viewport (i.e. any mouse interaction
that triggers a VTK camera event).

## Additional context

Frames 0 and 1 may have None or empty input data, which is common at
recording start.

---

## Root Cause

**VTK clipping planes become degenerate when all initial actors are empty.**

1. `_init_actors` registers all dynamic actors as **empty** `PolyData`
   (0 points).  The only geometry with real bounds is a tiny
   `Sphere(radius=2)` at the origin.
2. The camera is positioned at `contact_centroid ± 400 mm` and
   `camera_set = True` locks it.
3. VTK computes **near/far clipping planes** from the scene's aggregate
   bounding box — which is near-zero → clipping range collapses.
4. `_deferred_start` calls `_update_frame(0)`.  If frame 0 has None/empty
   data, all actors stay empty.  The clipping range stays degenerate.
5. User moves slider to frame N with real data → `_update_frame(N)` replaces
   actors with real geometry, calls `plotter.render()` — but the geometry
   falls **outside the stale near/far clipping planes** → invisible.
6. User zooms/pans → VTK's interactor style internally calls
   `ResetCameraClippingRange()` → clipping planes expand → data appears.

---

## Fix Plan

See `docs/plans/active/fix-initial-render.md`
