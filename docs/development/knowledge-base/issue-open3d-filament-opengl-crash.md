# Issue: Open3D Filament OpenGL context creation failure on Windows

**Date:** 2026-06-10
**Status:** Resolved — fallback UI implemented; see `arm_segmentation_view.py`
**Affects:** ALL scripts using `ArmSegmentation(interactive=True)` — both the
pipeline (`extract_participant_forearm.py`) and the standalone tool
(`code/scripts/__misc/standalone_mkv_to_forearm_mesh.py`)

---

## Problem

Open3D's Filament-based GUI (`gui.Application.create_window()`) crashes with:

```
FEngine (64 bits) created at ... (threading is enabled)
FEngine resolved backend: OpenGL
wglCreateContextAttribs() failed, whdc=...
Windows error code: 87. (null)
```

No Python traceback — the crash is in the native Filament rendering layer.
The process exits immediately.

## Root cause

**NVIDIA driver 610.47 breaks Filament's `wglCreateContextAttribs()` call.**

A baseline test — no tkinter, no multiprocessing, no project imports, just
`gui.Application.create_window()` — crashes identically. The issue is a
driver-level incompatibility with Filament's OpenGL context creation, not
any Python-level library conflict.

The legacy GLFW-based `Visualizer` (`wglCreateContext`, without the
"Attribs" variant) works fine on the same machine.

### Earlier misdiagnosis: tkinter contamination

The crash was initially attributed to tkinter/OpenCV imports corrupting
process-global Windows state before Filament ran. This hypothesis was
disproved by the baseline test showing the crash occurs even in a clean
process with only Open3D imported. The deferred-import changes to the
standalone script (moving cv2/tkinter into `_gui_worker()`) remain in
place as good practice but are not the fix.

## Environment

| | Primary dev machine (current) | Secondary machine |
|---|---|---|
| GPU | RTX 4070 Ti SUPER | RTX 2070 Super |
| NVIDIA driver | **610.47** (regression) | 560.94 |
| Open3D | 0.19.0 | 0.19.0 |
| Windows | 11 build 26200 | — |
| Python | 3.10.20 (conda) | — |
| Legacy Visualizer | Works | — |
| Filament GUI | **Broken** | **Broken** |

Driver 591.86 was the last known working driver on the primary machine.

## What was tried

| Fix | Result |
|-----|--------|
| `cv2.destroyAllWindows()` + `cv2.waitKey(1)` between steps | No effect |
| Early `gui.Application.instance.initialize()` before OpenCV | No effect |
| `multiprocessing.Process` to isolate Tkinter/OpenCV in child | No effect |
| Move cv2/tkinter imports inside `_gui_worker()` only | No effect (good practice, kept) |
| `SetProcessDpiAwarenessContext(DPI_UNAWARE)` before Open3D | No effect |
| `SetProcessDPIAware()` before Open3D | No effect |
| `OPEN3D_CPU_RENDERING=true` env var | No effect (not supported on Windows) |
| `FILAMENT_BACKEND=vulkan` env var | No effect (pip wheel is OpenGL-only) |
| **Downgrade to Open3D 0.18.0** | Same crash — both versions use same Filament OpenGL path |
| **Mesa3D software OpenGL (mesa-dist-win 26.1.1)** | Mesa DLLs load correctly, NVIDIA ICD bypassed, but Mesa's `wglCreateContextAttribs` also fails |
| Mesa + `GALLIUM_DRIVER=llvmpipe` + `MESA_GL_VERSION_OVERRIDE=4.5` | Same crash |
| NVIDIA driver rollback | Not an option for the user |

### Mesa details

- Downloaded `mesa3d-26.1.1-release-msvc.7z` from
  [pal1000/mesa-dist-win](https://github.com/pal1000/mesa-dist-win).
- Placed `opengl32.dll` + `libgallium_wgl.dll` (x64) in the conda env
  directory alongside `python.exe`.
- Verified via `GetModuleFileNameW` that Mesa's DLLs were loaded and
  `nvoglv64.dll` was NOT loaded.
- `wglCreateContextAttribs` still fails — Mesa's Windows WGL layer does
  not fully implement this extension.

## Diagnostic tools

Two scripts in `code/scripts/_3_preprocessing/_3_forearm_extraction/`:

- **`debug_opengl.py`** — phased diagnostic that tests imports, geometry,
  rendering context, tkinter, pyk4a, and preprocessing barrel imports.
  All phases pass except Filament window creation.

- Several test scripts in `code/scripts/__misc/_test_*.py` were used for
  targeted isolation during this investigation. They can be deleted.

## What works

- `o3d.visualization.Visualizer()` — legacy GLFW path, uses
  `wglCreateContext` (not the "Attribs" variant). Works for both
  `visible=True` and `visible=False`.
- `o3d.visualization.draw_geometries()` — uses legacy Visualizer
  internally. Works.
- `o3d.visualization.draw_geometries_with_key_callbacks()` — legacy
  Visualizer with keyboard hooks. Works.
- All Open3D geometry operations (voxel downsample, DBSCAN, bbox crop,
  PLY I/O, normals estimation). No rendering needed.

## What is broken

- `gui.Application.instance.create_window()` — Filament GUI. Crashes.
- `rendering.OffscreenRenderer` — uses EGL Headless, not supported on
  Windows.
- Any code path that uses `gui.SceneWidget` or `rendering.Open3DScene`,
  including `ArmSegmentation`'s interactive mode.

## Resolution: legacy Visualizer fallback for ArmSegmentation (IMPLEMENTED)

The fix was implemented in
`code/src/preprocessing/forearm_extraction/arm_segmentation_view.py`
as part of the "Split ArmSegmentation into Processing + View Modules" refactor.

### Key functions

- **`_probe_filament()`** — Runs `gui.Application.instance.create_window()`
  in a subprocess (`subprocess.run`, timeout 10 s). Returns `True` if the
  subprocess exits with code 0, `False` otherwise.  The probe is pure: it
  imports `gui` only inside the subprocess, so the parent process is never
  contaminated by a crashing Filament backend.

- **`_is_filament_available()`** — Wraps the probe with module-level caching
  (`_filament_available: bool | None`). The probe runs exactly once per
  process; subsequent interactive calls return immediately.  Emits a
  `RuntimeWarning` if Filament is unavailable.

- **`_display_legacy()`** — Full keyboard-driven interactive mode using
  `o3d.visualization.VisualizerWithKeyCallback` (legacy GLFW path,
  `wglCreateContext` — works on NVIDIA 610.47).  Controls:
  - Up / Down  — select parameter
  - Left / Right — fine adjustment (one step)
  - `-` / `=`   — coarse adjustment (10 steps)
  - `Space` / `Enter` — run processing function with current params
  - `Q` / `Esc` — accept and continue to next step
  - `H`         — print key-binding help to terminal

- **`display_pointcloud_interactive()`** — Entry point called by
  `ArmSegmentation._display_pointcloud()` via lazy import.  Routes to
  `_display_filament()` when Filament is available, otherwise to
  `_display_legacy()`.

### Transparency

On machines where Filament works (e.g. after a driver update or on a
machine without the 610.47 regression), the full slider-based GUI is used
automatically with no code changes required.

### Import isolation

`arm_segmentation.py` imports `arm_segmentation_view` only inside
`_display_pointcloud()` when `self.interactive is True`, so batch-mode
processing (`interactive=False`) never triggers the lazy import and
`arm_segmentation_view` is not loaded.  Note: `open3d.visualization.gui`
and `.rendering` still appear in `sys.modules` after `import open3d as o3d`
because Open3D eagerly loads them as submodules — this is an Open3D
implementation detail unrelated to our code.

## Other issues encountered

- **Non-ASCII MKV paths:** `pyk4a` cannot open paths containing non-ASCII
  characters (e.g. `ö` in `Linköpings universitet`). Solved with
  `_k4a_safe_path()` context manager that tries Windows 8.3 short path,
  falls back to copying MKV to a temp directory.
- **VideoFrameSelector not visible:** `root.withdraw()` hides the Tk root
  AND its transient children. Fixed by using
  `root.geometry("1x1+10000+10000")` to move root off-screen instead.
