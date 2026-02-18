# NeuralKinectViewer — Implementation Plan

## Context

The existing `view_somatosensory_assessement` visualization task renders a combined 3D scene (Kinect RGB+depth
point cloud, forearm point cloud, wireframe hand mesh, colored sticker spheres) via `SceneViewer` in
`scene_viewer.py`. Its core bottleneck: `_update_plot()` calls `plotter.clear()` on every frame, destroying
and re-uploading all VTK actors to the GPU (~48 MB/frame for the Kinect cloud alone), making the viewer
sluggish and unresponsive.

A new standalone viewer is needed that:
- Eliminates `plotter.clear()` using PyVista's named-actor in-place replacement
- GPU-accelerates spatial cropping with **CuPy** (RTX 4070 Ti Super) to reduce rendered points 8-16x
- Pre-loads MKV frames in a background thread (8-frame ring buffer) to eliminate seek latency
- Adds neural data overlays from the merged CSV: `Nerve_freq`, `contact_depth`, `contact_area`
- Shows per-sticker 2D velocity compass widgets in the right panel
- Centers camera and crop window on the average contact XYZ
- Displays the recording name in the window title and as a 3D text actor

---

## Parallel Work Groups & Dependencies

### 🟢 Group 1: Sticker Velocity Compass Widget
**Dependencies:** None
**Can parallelize:** YES
**Detail:** `docs/plans/active/neural-kinect-viewer/01-foundation.md`

### 🟢 Group 2: FramePreloader (Background MKV Thread)
**Dependencies:** None
**Can parallelize:** YES (with Group 1)
**Detail:** `docs/plans/active/neural-kinect-viewer/01-foundation.md`

### 🟡 Group 3: NeuralDataPanel (Matplotlib Time-Series)
**Dependencies:** None structurally, but logically after Groups 1-2 (all in same file)
**Can parallelize:** YES
**Detail:** `docs/plans/active/neural-kinect-viewer/02-core.md`

### 🟡 Group 4: Extract `define_custom_colors()`
**Dependencies:** None
**Can parallelize:** YES
**Detail:** `docs/plans/active/neural-kinect-viewer/02-core.md`

### 🔴 Group 5: NeuralKinectViewer Main Class
**Dependencies:** Groups 1, 2, 3, 4
**Can parallelize:** NO
**Detail:** `docs/plans/active/neural-kinect-viewer/03-integration.md`

### 🔴 Group 6: Update `common/__init__.py`
**Dependencies:** Group 5
**Can parallelize:** NO
**Detail:** `docs/plans/active/neural-kinect-viewer/03-integration.md`

### 🔴 Group 7: Entry Point Script
**Dependencies:** Group 5
**Can parallelize:** NO
**Detail:** `docs/plans/active/neural-kinect-viewer/04-entry-point.md`

---

## Dependency Graph

```
Group 1 (Compass)  ──────────────────┐
Group 2 (Preloader) ─────────────────┤
Group 3 (NeuralDataPanel) ───────────┼──► Group 5 (NeuralKinectViewer) ──► Group 6 (__init__)
Group 4 (define_custom_colors) ──────┘                                  └──► Group 7 (Entry point)
```

---

## New Files

| File | Group | Purpose |
|---|---|---|
| `code/src/preprocessing/common/gui/sticker_velocity_compass.py` | 1 | QPainter compass widget per sticker |
| `code/src/preprocessing/common/gui/neural_kinect_scene_viewer.py` | 2,3,5 | FramePreloader + NeuralDataPanel + NeuralKinectViewer |
| `code/scripts/view_merged_neural_kinect.py` | 7 | Standalone entry-point script |

## Modified Files

| File | Group | Change |
|---|---|---|
| `code/src/preprocessing/common/__init__.py` | 6 | Export `NeuralKinectViewer` |
| `code/scripts/_3_preprocessing/_4_somatosensory_quantification/view_somatosensory_3d_scene.py` | 4 | Replace `define_custom_colors()` with import |

---

## Architecture Summary

**Key invariant**: Each named actor lives exactly once in the VTK render window.
`_update_frame()` calls `plotter.add_mesh(new_data, name='X')` → PyVista replaces actor X in-place.
`plotter.render()` is called once at the end. No `plotter.clear()` anywhere.

```
NeuralKinectViewer(QMainWindow)
├── QtInteractor (self.plotter)        ← 3D viewport, named actors
├── NeuralDataPanel(QWidget)           ← matplotlib 3-panel time series (bottom)
├── StickerVelocityCompass × N         ← QPainter compass, right panel per sticker
└── FramePreloader(Thread)             ← ring buffer of 8 decoded MKV frames
```

**Layout:**
```
[3D viewport (stretch=4)] | [Right panel: visibility, sliders, compasses]
[Frame slider | ▶ Play | Recenter | Crop ±mm | N/T]
[NeuralDataPanel: Nerve_freq | contact_depth | contact_area + red cursor]
```

---

## Speed Improvement Summary

| Optimization | Mechanism | Estimated Gain |
|---|---|---|
| Named-actor replacement | Full 48MB upload → only changed actors | ~5-10x frame update |
| CuPy AABB crop (±400mm) | 800k → ~75k points reach VTK | ~8-16x Kinect render |
| Background MKV preloader | Eliminates pyk4a seek latency | ~15-25ms/frame |
| Lazy hand mesh | `HandMotionManager[i]` only on demand | Startup 30-60s → <5s |

---

## Graceful Degradation

| Missing | Behavior |
|---|---|
| `cupy` not installed | CPU NumPy fallback + one-time warning |
| `merged_csv_path=None` | No NeuralDataPanel, no compasses; pure 3D viewer |
| Hand motion NPZ missing | Skips hand mesh actor, rest of scene shown |

---

## Verification

See `docs/plans/active/neural-kinect-viewer/04-entry-point.md` for full test checklist.

Quick smoke test:
```bash
cd /mnt/f/GitHub/social-touch-semi-controlled
python code/scripts/view_merged_neural_kinect.py
```
- Title bar shows recording name
- Slider movement updates scene without camera reset
- Compass widgets animate per sticker
- Neural panel cursor follows slider
- Play button runs at ~30fps without lag
