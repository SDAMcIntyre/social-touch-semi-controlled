# NeuralKinectViewer — Generated Files

Feature: interactive 3D viewer for merged neural + Kinect recordings.
Original planning documents: [`docs/plans/active/neural-kinect-viewer/`](../plans/active/neural-kinect-viewer/)

---

## Source files

### `code/src/merging/` — new package

| File | Classes / symbols | Notes |
|---|---|---|
| `__init__.py` | — | Package root for the `merging` source package |
| `gui/__init__.py` | `NeuralKinectViewer`, `StickerVelocityCompass` | Public API re-exports |
| `gui/neural_kinect_scene_viewer.py` | `FramePreloader`, `NeuralDataPanel`, `NeuralKinectViewer`, `define_custom_colors`, `_parse_contact_points_cell` | Main viewer module (see below) |
| `gui/sticker_velocity_compass.py` | `StickerVelocityCompass` | QPainter velocity compass widget |

> **Location note:** The plan documents (`docs/plans/active/neural-kinect-viewer/`) originally
> placed the GUI files under `code/src/preprocessing/common/gui/`.  They were moved to
> `code/src/merging/gui/` before the first commit because the viewer is conceptually part of
> the merging stage, not the preprocessing stage.

#### `gui/neural_kinect_scene_viewer.py` — class breakdown

| Class | Plan group | Responsibility |
|---|---|---|
| `FramePreloader(Thread)` | Group 2 | Background daemon thread; maintains a 32-frame ring buffer of decoded `KinectPointCloudView` frames to eliminate MKV seek latency. Single producer — only this thread calls `KinectPointCloudView[idx]`. |
| `NeuralDataPanel(QWidget)` | Group 3 | Fixed-height Matplotlib panel (3 shared-x axes): `Nerve_freq`, `contact_depth`, `contact_area`. Tracks current frame with a red vertical cursor. Supports zoom / full-view toggle and an adjustable ±N s window. |
| `NeuralKinectViewer(QMainWindow)` | Group 5 | Top-level viewer window. Owns the PyVista `QtInteractor`, the `FramePreloader`, the `NeuralDataPanel`, and one `StickerVelocityCompass` per sticker. Uses named-actor in-place replacement — `plotter.clear()` is never called. |

#### `gui/sticker_velocity_compass.py` — class breakdown

| Class | Plan group | Responsibility |
|---|---|---|
| `StickerVelocityCompass(QWidget)` | Group 1 | 120 × 140 px QPainter widget. Draws a 2D XY velocity arrow (length ∝ XY speed, color ∝ Z speed: blue = approaching / red = receding) inside a ring drawn in the sticker's assigned color. |

---

## Entry-point script

| File | Notes |
|---|---|
| `code/scripts/view_merged_neural_kinect.py` | Prefect flow that iterates block config files, resolves all paths from `KinectConfig`, and launches `NeuralKinectViewer` sequentially (one viewer at a time). Imports CuPy before any preprocessing package to avoid the NumPy dtype-registry conflict (see `docs/bugs/cupy-bool8-import-order.md`). |

---

## DAG configuration

| File | Notes |
|---|---|
| `configs/view_merged_neural_kinect_dag.yaml` | Task: `view_neural_kinect_scene`. Key option: `crop_half_size_mm` (default 400 mm). `kinect_configs_directory` points to the block config folder for the session to view. |

---

## Documentation files

| File | Notes |
|---|---|
| `docs/bugs/cupy-bool8-import-order.md` | Explains the CuPy / NumPy 2.0 `bool8` dtype-registry crash and the early-import fix applied in every entry-point script that uses CuPy alongside preprocessing packages. |
| `docs/gpu_cupy_setup_and_viewer_improvements.md` | Environment setup guide for CuPy (CUDA toolkit, pip install) and notes on planned viewer improvements. |
| `docs/plans/active/neural-kinect-viewer.md` | Feature plan: motivation, parallel work groups, dependency graph, architecture summary, speed improvement table, graceful-degradation table. |
| `docs/plans/active/neural-kinect-viewer/01-foundation.md` | Detailed spec for `StickerVelocityCompass` (Group 1) and `FramePreloader` (Group 2). |
| `docs/plans/active/neural-kinect-viewer/02-core.md` | Detailed spec for `NeuralDataPanel` (Group 3) and extraction of `define_custom_colors` (Group 4). |
| `docs/plans/active/neural-kinect-viewer/03-integration.md` | Detailed spec for `NeuralKinectViewer` main class (Group 5) and `common/__init__.py` update (Group 6). |
| `docs/plans/active/neural-kinect-viewer/04-entry-point.md` | Detailed spec for the entry-point script (Group 7) and full verification / testing checklist. |

---

## Deviations from the original plan

| Plan | Reality | Reason |
|---|---|---|
| GUI files in `preprocessing/common/gui/` | Moved to `merging/gui/` | The viewer visualises *merged* data; placing it in `preprocessing` was semantically wrong. |
| `common/__init__.py` updated to export `NeuralKinectViewer` (Group 6) | Not done | Export lives in `merging/gui/__init__.py` instead; `preprocessing.common` is not affected. |
| `view_somatosensory_3d_scene.py` updated to import `define_custom_colors` (Group 4) | Not done | Left as a follow-up; the function was copied into the new module but the old script was not changed. |
| Entry point with hardcoded paths | Entry point uses `KinectConfig` + DAG YAML | Adopted the config-driven pattern used by all other workflow scripts in the project. |
| `FramePreloader` buffer size: 8 frames | Increased to 32 frames | 8 frames is less than one second at 30 fps; 32 frames gives ~1 s of headroom with minimal RAM cost. |
