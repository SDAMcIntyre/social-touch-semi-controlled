# GPU / CuPy Setup & Neural-Kinect Viewer Improvements

## Context

The `NeuralKinectViewer` (`code/src/preprocessing/common/gui/neural_kinect_scene_viewer.py`)
uses CuPy for GPU-accelerated AABB point-cloud cropping.  This document records the
viewer improvements made and the CuPy installation issues encountered on the Windows
workstation (CUDA 12.9, Anaconda env `py31018`, Python 3.10, NumPy 2.2.6).

---

## 1. Viewer improvements (completed)

### 1.1 GPU status indicator
A text actor is now rendered in the upper-right corner of the 3-D viewport:
- **GPU: ON** (lime green) when CuPy + CUDA is detected at startup.
- **GPU: OFF** (tomato red) when CuPy is unavailable or falls back to NumPy.

### 1.2 Slider responsiveness
The frame slider previously called the full `_update_frame()` (MKV decode + 3-D
render) on every `valueChanged` tick while the user dragged.  Fixed by splitting into:

| Signal | Handler | What it does |
|--------|---------|--------------|
| `sliderPressed` | `_on_slider_pressed` | Sets `_slider_dragging = True` |
| `valueChanged` (during drag) | `_on_slider_change` (fast path) | Updates frame label + sends `seek()` hint to preloader only |
| `sliderReleased` | `_on_slider_released` | Sets `_slider_dragging = False`, calls full `_update_frame()` |
| `valueChanged` (playback / click) | `_on_slider_change` (normal path) | Calls full `_update_frame()` directly |

Result: no MKV decode or VTK render fires during drag; the full render happens once
on release.

### 1.3 Larger preloader ring buffer
`FramePreloader` buffer size increased from **8 → 32 frames**.

### 1.4 Smarter buffer eviction
On a seek signal the old code wiped the entire buffer.  Now only frames more than
`buffer_size` indices away from the new target are evicted, so back-and-forth
scrubbing within the same region reuses cached frames.

### 1.5 Buffer fill indicator
A small `Buf: N/32` label appears next to the frame counter, showing preloader
progress in real time.

---

## 2. CuPy installation issues

### 2.1 Root cause
NumPy 2.0 removed the `bool8` type alias.  CuPy versions older than 13.0 reference
`np.bool8` at module-init time, causing:

```
TypeError: Alias 'bool8' was removed in NumPy 2.0. Use a name without a digit at the end.
```

The environment pins **NumPy 2.2.6** (required by pandas, scipy, scikit-image, etc.),
so downgrading NumPy is not viable.

### 2.2 Steps taken

| Step | Command | Result |
|------|---------|--------|
| Initial install | `pip install cupy-cuda12x` | Installed old version incompatible with NumPy 2 |
| Attempted upgrade | `pip install --upgrade cupy-cuda12x` | Upgraded to 14.0.0, but a second bare `cupy` (old) was also present |
| Identified duplicate | `pip list \| findstr cupy` | Showed both `cupy` and `cupy-cuda12x` |
| Clean reinstall | `pip uninstall cupy cupy-cuda12x -y` then `pip install "cupy-cuda12x>=13.0"` | Leftover temp dirs `~-py` / `~-py_backends` had to be deleted manually first |
| Manual cleanup | `rmdir /S /Q "...\site-packages\~-py"` etc. | Cleared stale compiled extensions |
| Final state | `pip install "cupy-cuda12x>=13.0"` | Installed `cupy-cuda12x 14.0.0` cleanly, no bare `cupy` pulled in |

Terminal verification now passes:
```
python -c "import cupy as cp; cp.array([1.0]); print('OK', cp.__version__)"
OK 14.0.0
```

### 2.3 requirements.txt fix
Pinned minimum version to prevent regressions on fresh installs:
```
cupy-cuda12x>=13.0; platform_system != "Darwin"
```

---

## 3. Outstanding issue — VSCode debug console

**Symptom:** running `import cupy as cp` in the VSCode debug console (while paused
at a breakpoint) still raises the `bool8` TypeError.  Additionally,
`import sys; print(sys.executable)` returns an empty string in that same console.

**Likely cause:** the VSCode debug console evaluates expressions in the paused
frame's context, which can have a partially-initialised or restricted Python
environment.  `sys.executable` being empty is a known behaviour in certain conda +
Windows + debugpy configurations.  It does **not** necessarily mean the actual
Python process is using the wrong interpreter.

**Next verification step:**
Run the viewer from the terminal without the debugger:
```bash
python code/scripts/view_merged_neural_kinect.py
```
Check whether the 3-D viewport shows **GPU: ON** or **GPU: OFF**.
- If **GPU: ON** → CuPy works correctly; the debug console error is a false alarm.
- If **GPU: OFF** + a warning is printed → the actual process also fails to import
  CuPy, and further investigation is needed (see below).

**If the problem persists in the actual process:**
1. Check the VSCode launch configuration:
   - Open `.vscode/launch.json` and ensure `"python": "${command:python.interpreterPath}"` is present in the relevant configuration.
   - The bottom-right interpreter selector controls IntelliSense, not necessarily
     the debugger; they can diverge.
2. Open a fresh integrated terminal, run `conda activate py31018`, then launch
   the debugger with F5 from that terminal.
3. Run `python -c "import sys; print(sys.executable)"` in the integrated terminal
   (not the debug console) to confirm the terminal itself is using the right Python.
