# Plan: Replace RF 3D Heatmap Renderer with PyVista Offscreen

**Date:** 2026-05-11
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `dev`
**Branch:** `feature/replace-rf-3d-renderer-with-pyvista`

---

## Overview

The 3D forearm heatmap PNGs produced by `render_forearm_heatmap()` do not match the orientation the researcher sets in the RF Camera Settings Viewer. The root cause is a dual-technology split: camera settings are captured in PyVista (VTK perspective camera), but static PNGs are rendered in matplotlib (pseudo-orthographic 3D projection). This plan replaces the matplotlib 3D rendering path with PyVista offscreen rendering, guaranteeing orientation fidelity since both the viewer and renderer use the same engine and camera model.

## Problem Statement

The RF Camera Settings Viewer (`RFCameraSettingsViewer`) lets the researcher interactively orient the forearm per session using a PyVista/VTK perspective camera. The saved parameters (position, focal_point, up_vector, view_angle) are consumed downstream by `render_forearm_heatmap()`, which:

1. Converts the camera settings to a 3x3 rotation matrix via `camera_settings_to_rotation()`
2. Rotates all geometry (mesh, spike points, hull points) into "camera space"
3. Renders with matplotlib's `mpl_toolkits.mplot3d` using a hardcoded `ax.view_init(elev=90, azim=-90)`

This translation is fundamentally broken:
- matplotlib's 3D projection is pseudo-orthographic, not perspective
- `view_init(elev, azim)` provides only 2 DOF (no roll), while a full camera has 3 DOF
- The hardcoded `(90, -90)` values produce a 90-degree on-screen rotation and inverted depth ordering relative to the intended view
- No combination of `view_init` values can reproduce PyVista's perspective rendering

Meanwhile, all interactive GUI viewers use PyVista and display the correct orientation. The static PNG output is the only place where matplotlib 3D is used.

## Goals

### In Scope
1. Replace the matplotlib 3D rendering path in `render_forearm_heatmap()` with PyVista offscreen rendering
2. Change the function's `rotation_matrix` parameter to accept the raw `camera_settings` dict, setting the PyVista camera directly from saved parameters
3. Preserve the 2D projection path (matplotlib 2D) — derive rotation_matrix from camera_settings internally when the 2D path is taken
4. Update the two pipeline callers (`rf_simple_pipeline.py`, `rf_cluster_pipeline.py`) to pass camera_settings

### Out of Scope
- Changing the 2D projection path (`rf_2d_renderer.py`) — it's matplotlib 2D (not 3D) and works correctly
- Changing `rf_projection.py` or `rf_metrics.py` — pure NumPy/SciPy, technology-agnostic
- Changing `camera_settings_to_rotation()` — still needed by the 2D projection and metrics paths
- Replacing matplotlib in the interactive GUI viewers — they already use PyVista
- The separate `enforce-rf-camera-settings-all-viewers` plan (threads rotation into viewers that currently lack it)

## Success Criteria

- [ ] 3D heatmap PNG orientation matches the RF Camera Settings Viewer for sessions with non-trivial camera angles
- [ ] Perspective rendering matches the viewer (same engine, same camera params)
- [ ] 2D projection PNGs (`_cylindrical_unwrap`, `_tangent_plane`) are unchanged
- [ ] `compute_rf_metrics()` output is unchanged (still receives rotation_matrix from its caller)
- [ ] Cluster pipeline heatmaps (`visualize_receptive_fields_clustered`) render correctly
- [ ] Point cloud fallback works when mesh reconstruction fails

---

## Technical Design

### Approach

Replace the matplotlib 3D rendering code (lines 288-552 of `rf_cluster_visualizer.py`) with PyVista offscreen rendering. The implementation follows the established gallery viewer pattern (`rf_cluster_gallery_viewer.py:920-1115`) and the offscreen plotter pattern (`scene_viewer_video_maker.py:114`).

Key design decisions:
- **Camera settings passed directly** — the function receives the raw `camera_settings` dict and applies it to the PyVista camera. No rotation matrix needed for the 3D path.
- **Rotation derived internally for 2D path** — when `projection_method` is set, the function calls `camera_settings_to_rotation(camera_settings)` to get the rotation matrix for `project_to_2d()`.
- **Reuse existing utilities** — `mesh_to_pyvista()`, `map_scalars_to_mesh()`, `load_or_build_forearm_mesh()` are all reused as-is.
- **LogNorm via pre-computation** — PyVista doesn't support matplotlib's LogNorm natively. Pre-apply `np.log1p()` to scalar values and use linear clim on the log-scaled values.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Replace 3D path with PyVista offscreen | Exact camera match (same engine); existing patterns in codebase; handles all 3 camera DOF | Adds PyVista dependency to static rendering path (already a project dependency) | **Chosen** |
| Fix matplotlib `view_init` translation | Minimal code change (2 lines) | Cannot reproduce perspective projection; only 2 DOF (no roll); depth sorting fundamentally different | Rejected — attempted on `fix/rf-simple-3d-heatmap-camera-orientation` branch, confirmed insufficient |
| Compute `view_init(elev, azim)` from camera params without rotation | No rotation needed | Still only 2 DOF; still pseudo-orthographic; lose roll information | Rejected |
| Keep both engines, add manual camera presets per session | Preserves existing code | Doesn't solve the problem; researcher must set camera twice | Rejected |

### Architecture Changes

No new modules or classes. The change is internal to `render_forearm_heatmap()`.

**Function signature change:**
```
# Before
def render_forearm_heatmap(..., rotation_matrix: np.ndarray = None, ...) -> None

# After
def render_forearm_heatmap(..., camera_settings: dict = None, ...) -> None
```

**New imports in `rf_cluster_visualizer.py`:**
```python
import pyvista as pv
from .rf_surface_utils import mesh_to_pyvista  # already imported: load_or_build_forearm_mesh, map_scalars_to_mesh
from .tangent_plane_alignment import camera_settings_to_rotation  # for 2D path derivation
```

**Removed imports** (3D path no longer needs):
```python
# No longer needed after removal of matplotlib 3D path:
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from .rf_surface_utils import apply_rotation_to_mesh  # camera handles orientation now
from .tangent_plane_alignment import align_points      # camera handles orientation now
```

Note: `matplotlib`, `matplotlib.pyplot`, `matplotlib.cm`, `matplotlib.colors` remain — used by the 2D projection path.

---

## Implementation Plan

### Phase 1: Replace 3D path in `render_forearm_heatmap()`
**Goal:** Replace the matplotlib 3D rendering code with PyVista offscreen rendering
**Started:** 2026-05-11
**Completed:** 2026-05-11

- [x] Task 1.1 — Change `rotation_matrix: np.ndarray = None` parameter to `camera_settings: dict = None`
- [x] Task 1.2 — In the 2D projection path (lines 211-287), derive `rotation_matrix` from `camera_settings` via `camera_settings_to_rotation()` when needed, then pass to `project_to_2d()` as before
- [x] Task 1.3 — Replace the matplotlib 3D path (lines 288-552) with PyVista offscreen rendering:
  - Create `pv.Plotter(off_screen=True, window_size=[2000, 1600])`
  - Set black background
  - Load forearm mesh via `load_or_build_forearm_mesh()`, convert via `mesh_to_pyvista()`
  - Map spike scalars to mesh vertices via `map_scalars_to_mesh()` (reuse existing)
  - Add mesh with `add_mesh(scalars=..., cmap="RdYlBu_r", nan_color="lightgrey", smooth_shading=True)`
  - For LogNorm: pre-apply `np.log1p()` to per-vertex scalars, use linear clim on log-scaled values
  - For spike_ratio metric: use ratio values directly with `clim=[0, 1]`
  - Add scalar bar with white text
- [x] Task 1.4 — Implement point cloud fallback when mesh reconstruction fails: `pv.PolyData(vertices)` with point rendering (follow gallery viewer lines 1004-1056)
- [x] Task 1.5 — Implement convex hull rendering: compute via `scipy.spatial.ConvexHull`, create PyVista PolyData with lines, add as wireframe with `#00aaff` (neuron) and `#ffaa00` (cluster) colors
- [x] Task 1.6 — Add metadata text overlay via `plotter.add_text()` and title
- [x] Task 1.7 — Set camera from saved settings: assign `camera.position`, `camera.focal_point`, `camera.up`, `camera.view_angle` directly from the `camera_settings` dict
- [x] Task 1.8 — Save PNG via `plotter.screenshot(str(output_path))` then `plotter.close()`
- [x] Task 1.9 — Handle `interactive=True`: use `plotter.show()` instead of saving (PyVista's built-in 3D interaction replaces the matplotlib slider)
- [x] Task 1.10 — Remove unused matplotlib 3D imports (`Axes3D`, `Slider`, `apply_rotation_to_mesh`, `align_points`) and the `_draw_hull_3d()` helper function

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_visualizer.py` — replace 3D path (~260 lines rewritten), change parameter, update imports

**Dependencies:** None

### Phase 2: Update pipeline callers
**Goal:** Pass raw camera_settings dict to `render_forearm_heatmap()` instead of rotation_matrix
**Started:** 2026-05-11
**Completed:** 2026-05-11

- [x] Task 2.1 — In `rf_simple_pipeline.py`: load raw camera settings via `load_rf_camera_settings(camera_settings_dir)` and extract `cameras[session_id]`; pass as `camera_settings=session_cam` to `render_forearm_heatmap()`
- [x] Task 2.2 — In `rf_cluster_pipeline.py::run_cluster_rf_visualization()`: load raw camera settings dict for rendering; keep loading `rotation_matrix` separately via `load_rf_camera_rotation()` for `compute_rf_metrics()` calls; pass `camera_settings=session_cam` to `render_forearm_heatmap()`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_simple_pipeline.py` — change `load_rf_camera_rotation()` to `load_rf_camera_settings()` + dict lookup; pass `camera_settings=` to renderer
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — load raw settings for rendering alongside rotation_matrix for metrics; pass `camera_settings=` to renderer

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Open Camera Settings Viewer (`set_rf_camera_settings`), note the forearm orientation for a test session
- [ ] Run `map_receptive_fields_simple` with `force_processing: true` — compare PNG against viewer; orientation should match exactly
- [ ] Repeat with a second session where camera is rotated to a non-trivial (non-axis-aligned) angle
- [ ] Run `visualize_receptive_fields_clustered` with `force_processing: true` — verify cluster heatmap PNGs match viewer orientation
- [ ] Verify 2D projection PNGs (`_cylindrical_unwrap`) are visually unchanged
- [ ] Verify `display_metric="spike_ratio"` heatmaps render correctly with ratio values in [0, 1]
- [ ] Verify colorbar appears with correct label ("Spike count" or "Spike ratio")

### Edge Cases
- [ ] Session with near-axis-aligned camera (looking almost along the forearm long axis) — should render without errors
- [ ] Missing forearm PLY or mesh build failure — should fall back to point cloud rendering
- [ ] Session with `camera_settings=None` (if settings file absent for simple pipeline) — `load_rf_camera_rotation` currently raises `ValueError`; verify this still happens cleanly
- [ ] Very sparse spike data (1-2 spike positions) — colorbar range should not crash on degenerate vmin==vmax

---

## Documentation Plan

- [ ] No CLAUDE.md changes needed (the RF rendering pipeline is described at sufficient level; the technology change is internal)
- [ ] No user-facing docs needed (pipeline output format unchanged — still produces PNGs at the same paths)
- [ ] Update `code/src/analysis/CLAUDE.md` to note that 3D heatmap PNGs now use PyVista offscreen rendering (in the "Receptive field mapping" section)

---

## Rollback Plan

1. Revert modified files in `rf_cluster_visualizer.py`, `rf_simple_pipeline.py`, `rf_cluster_pipeline.py`
2. Re-run `map_receptive_fields_simple` and `visualize_receptive_fields_clustered` with `force_processing: true` to regenerate PNGs with the old renderer
3. No data, config, or external state changes to reverse — the PNGs are regenerated on each pipeline run

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PyVista offscreen rendering requires a display server (X11/Xvfb) on headless systems | Med | Med | This pipeline runs on the researcher's workstation (Windows with display), not headless CI. PyVista supports `off_screen=True` on Windows natively. |
| LogNorm pre-computation via `np.log1p()` produces slightly different visual contrast than matplotlib's exact LogNorm | Low | Low | The visual difference is minimal; `log1p` is the standard approximation. If exact match needed, can use matplotlib's `LogNorm.__call__()` to pre-normalize scalars before assigning to PyVista. |
| Gallery viewer's `_build_scene()` pattern uses Delaunay mesh (not BPA); the renderer uses BPA mesh from `load_or_build_forearm_mesh()` | Low | Low | Both produce trimesh objects; `mesh_to_pyvista()` handles either. The BPA mesh is already tested and cached. |
| Interactive mode behavior changes (PyVista interaction vs matplotlib slider) | Low | Low | Interactive mode is a debugging aid, rarely used. PyVista's built-in orbit/zoom is more capable than the matplotlib slider. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Replace 3D path | ~2 hours | None |
| Phase 2: Update callers | ~30 minutes | Phase 1 |

---

## References

- Gallery viewer scene building (template): `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py` lines 920-1115
- Gallery viewer PNG export: `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py` lines 1200-1245
- Offscreen plotter pattern: `code/src/preprocessing/common/gui/scene_viewer_video_maker.py` line 114
- Mesh conversion utility: `code/src/analysis/receptive_field_mapping/rf_surface_utils.py` lines 157-164 (`mesh_to_pyvista`)
- Scalar-to-vertex mapping: `code/src/analysis/receptive_field_mapping/rf_surface_utils.py` lines 131-154 (`map_scalars_to_mesh`)
- Camera settings viewer (producer): `code/src/analysis/receptive_field_mapping/gui/rf_camera_settings_viewer.py`
- Camera rotation builder: `code/src/analysis/receptive_field_mapping/tangent_plane_alignment.py` lines 8-39
- Related plan: `docs/development/plans/pending/enforce-rf-camera-settings-all-viewers.md` (separate concern — threading rotation into viewers)

---
