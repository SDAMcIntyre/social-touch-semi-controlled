# RF Feature-Space Explorer — Post-Layout-Bugfix Status

**Date:** 2026-05-03
**Branch:** `feature/rf-explorer-layout-bugfix-cache` → `feature/rf-explorer-event-and-scalar-fix`

---

## What Was Implemented

The `feature/rf-explorer-layout-bugfix-cache` branch delivered three changes
to the RF Feature-Space Explorer:

1. **Layout overhaul** — `QSplitter` replaced by `QGridLayout` overlay.
   The PyVista `QtInteractor` and the matplotlib `FigureCanvasQTAgg` now share
   cell (0, 0); the scatter canvas is pinned `Qt.AlignBottom | Qt.AlignLeft`
   at 350×280 px with a semi-transparent dark background.

2. **Point cloud rendering** — Delaunay mesh removed from both the data model
   (`ExplorerSessionData.forearm_vertices` replaces `forearm_mesh` + `vertices`)
   and the GUI (`pv.PolyData(vertices)` with `render_points_as_spheres=True,
   point_size=5`). `_apply_filter_update()` rebuilds the `PolyData` and calls
   `add_mesh(name="forearm")` on each update.

3. **Sidecar `.npz` cache** — `_explorer_cache_path()` / `_save_explorer_cache()`
   / `_load_explorer_cache()` added to `rf_explorer_data.py`. Cache is validated
   by mtime against both the series CSV and the forearm PLY.

**Post-ship hotfixes (same branch, uncommitted):**
- NaN axis limits: `set_xlim`/`set_ylim` guarded with `np.isfinite` + `x_lo < x_hi`
  to handle sessions with all-NaN pressure/velocity data.
- Cache pickle crash: `allow_pickle=True` in `_load_explorer_cache`; labels
  saved as `dtype=str` in `_save_explorer_cache` to avoid object-array pickle.

The `feature/rf-explorer-event-and-scalar-fix` branch resolved both remaining
issues (see below).

---

## Resolved Issues

### Issue 1 — Filter rectangle could not receive mouse events

**Symptom:** The draggable filter rectangle (`DraggableFilterRect`) could not be
drawn or moved by the user. Mouse clicks and drags on the scatter overlay had
no effect.

**Confirmed root cause:** VTK's OpenGL render window captures mouse input at the
native/OS level, bypassing Qt's widget stacking order entirely. No combination
of `raise_()`, `WA_TranslucentBackground`, `WA_NativeWindow`, or focus policies
can route events to a `FigureCanvasQTAgg` overlaid in the same `QGridLayout`
cell as a `QtInteractor`. This is a known unsolved problem:
[pyvistaqt#91](https://github.com/pyvista/pyvistaqt/issues/91).

**Fix (`feature/rf-explorer-event-and-scalar-fix`):** The `QGridLayout` overlay
was replaced with a `QDockWidget` docked at the bottom of the main window.
The `QtInteractor` is now the sole central widget. The dock widget creates a
separate widget hierarchy with its own event handling — no competition with
VTK's native mouse capture. All transparency attributes, `WA_NativeWindow`,
and `canvas.raise_()` calls were removed.

### Issue 2 — Checkbox toggle permanently cleared the RF heatmap

**Symptom:** Unchecking any gesture-type checkbox caused the RF heatmap to
disappear. Re-checking it did not restore the heatmap.

**Confirmed root cause:** The `remove_actor("forearm")` + `add_mesh(name="forearm")`
cycle triggers PyVista's "keep extremum" logic on the scalar bar
([pyvista#1366](https://github.com/pyvista/pyvista/issues/1366),
[pyvista#2899](https://github.com/pyvista/pyvista/issues/2899)). When all
checkboxes were unchecked, `vertex_spikes` became all-zeros → scalar bar range
cached as `[0, 0]`. On re-check, the cached `[0, 0]` range corrupted the lookup
table permanently — the explicit `clim=(0, max_spikes)` was ignored because the
"keep extremum" logic only expands, never contracts.

**Fix (`feature/rf-explorer-event-and-scalar-fix`):** Switched to in-place scalar
updates using `copy_mesh=False` in the initial `add_mesh()`. On filter changes,
`self._cloud["spike_density"]` is updated in-place and
`self._actor.mapper.scalar_range = (0, max_spikes)` is set directly on the
actor, bypassing the scalar bar's clim logic entirely. The remove/add cycle is
gone. Verified working on installed PyVista 0.46.1.

---

## Not Broken

- Session switching works correctly.
- 3D point cloud renders visually.
- Sidecar cache writes and invalidates on source-file mtime change.
- Scatter draws correctly with axis trimming (no more NaN crash).
- Frame counter updates on filter changes.
