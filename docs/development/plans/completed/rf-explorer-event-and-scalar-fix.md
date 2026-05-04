# Plan: RF Explorer — Fix Mouse Events & Scalar Bar Corruption

**Date:** 2026-05-03
**Author:** Basil Duvernoy
**Status:** Completed
**Base Branch:** `feature/rf-explorer-layout-bugfix-cache`
**Branch:** `feature/rf-explorer-event-and-scalar-fix`

---

## Overview

**What:** Fix the two remaining bugs in the RF Feature-Space Explorer: the
filter rectangle that cannot receive mouse events, and the RF heatmap that
disappears permanently after any checkbox toggle.

**Why:** Both bugs render core interactive features completely unusable. The
filter rectangle is the primary data-exploration tool; the checkbox toggle is
the only way to isolate gesture types. Without fixes, the explorer is
view-only.

**How:** Replace the `QGridLayout` overlay with a `QDockWidget` (solving the
VTK event-routing conflict), and switch from remove/add-mesh to in-place
scalar updates via `actor.mapper.scalar_range` (bypassing PyVista's scalar bar
clim corruption).

## Problem Statement

The `feature/rf-explorer-layout-bugfix-cache` branch attempted two fixes that
both failed due to framework-level bugs:

1. **Mouse events:** The `QGridLayout` overlay places a matplotlib
   `FigureCanvasQTAgg` on top of a PyVista `QtInteractor` in the same grid
   cell. VTK's OpenGL render window captures mouse input at the native/OS
   level, bypassing Qt's widget stacking order. This is a known unsolved
   problem ([pyvistaqt#91](https://github.com/pyvista/pyvistaqt/issues/91)).
   No combination of `raise_()`, `WA_TranslucentBackground`,
   `WA_NativeWindow`, or focus policies can route events to the canvas.

2. **Scalar bar corruption:** The `remove_actor("forearm")` +
   `add_mesh(name="forearm")` cycle triggers PyVista's "keep extremum" logic
   on the scalar bar
   ([pyvista#1366](https://github.com/pyvista/pyvista/issues/1366)). When all
   checkboxes are unchecked, `vertex_spikes` is all-zeros → the scalar bar
   range becomes `[0, 0]`. On re-check, the cached `[0, 0]` range corrupts
   the lookup table permanently — the explicit `clim=(0, max_spikes)` is
   ignored because the "keep extremum" logic only expands, never contracts.

## Goals

### In Scope

1. Filter rectangle receives mouse events and supports all 8-zone
   drag/resize interactions (corners, edges, interior move)
2. RF heatmap correctly updates when gesture-type checkboxes are toggled in
   any order — including unchecking all and re-checking
3. Scatter panel is resizable, floatable, and dockable by the user

### Out of Scope

- Scatter overlay on top of the 3D view (abandoned — VTK incompatible)
- Configurable point size, colormap, or scatter position
- Camera persistence across sessions

## Success Criteria

- [ ] Drag filter rectangle on scatter — 3D heatmap updates in real time
- [ ] Resize rectangle from any corner or edge — geometry updates correctly
- [ ] Uncheck all gesture types → RF clears; re-check → RF reappears
- [ ] Toggle checkboxes in any sequence — RF always shows correct data
- [ ] Scatter dock can be resized, floated, and re-docked without crash
- [ ] Session switching still works correctly with the new layout
- [ ] Zero-density vertices render as dark gray (not blue); scalar bar unchanged

---

## Technical Design

### Approach

**Issue 1 — QDockWidget layout:** Move the matplotlib scatter canvas from a
grid overlay into a `QDockWidget` docked at the bottom of the main window.
The 3D `QtInteractor` becomes the sole central widget. The dock widget creates
a separate widget hierarchy with its own event handling — no competition with
VTK's native mouse capture.

**Issue 2 — In-place scalar update:** Use `copy_mesh=False` in the initial
`add_mesh()` call so PyVista holds a reference to the `PolyData`. On filter
updates, modify `self._cloud["spike_density"]` in-place and set
`actor.mapper.scalar_range` directly, bypassing the scalar bar's add_mesh
clim logic entirely. Verified working on installed PyVista 0.46.1:

```python
cloud["spike_density"] = new_values
actor.mapper.scalar_range = (0, max_spikes)
plotter.render()
```

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Layout: QDockWidget** | Own event context, standard Qt pattern, resizable | Not a true overlay — scatter below 3D, not on top | Chosen — overlay is impossible with VTK |
| **Layout: Frameless child window** | True overlay, own native window | Requires manual repositioning on resize, fragile across DPI | Rejected — over-engineered for the benefit |
| **Layout: Event filter forwarding** | Keeps overlay visual | Complex, fragile, must handle all event types, VTK may still intercept | Rejected — fighting the framework |
| **Layout: QSplitter (original)** | Known working | Wastes space, was already replaced for good reason | Rejected — regression |
| **Scalar: In-place update via mapper** | Bypasses scalar bar clim logic, no remove/add cycle | Requires `copy_mesh=False` | Chosen — clean, verified locally |
| **Scalar: `plotter.clear()` + fresh `add_mesh`** | Guaranteed clean state | Loses camera position, flickers, heavy for frequent updates | Rejected — inferior UX |
| **Scalar: Direct VTK LUT manipulation** | Full control | Low-level VTK API, version-fragile | Rejected — PyVista mapper API sufficient |

### Architecture Changes

**Modified class:** `RFFeatureSpaceExplorer` in
`gui/rf_feature_space_explorer.py`

**New instance variable:**
- `self._actor: Optional[pv.plotting.actor.Actor]` — stores the return value
  of `add_mesh()` for direct mapper access during filter updates

**Removed from `_build_ui()`:**
- `QGridLayout` overlay container
- `WA_TranslucentBackground`, `WA_NativeWindow` attributes
- Transparent stylesheet on canvas
- `canvas.raise_()` calls
- `setFixedSize(350, 280)`

**Added to `_build_ui()`:**
- `QDockWidget("Feature Space", self)` containing the matplotlib canvas
- Central widget set to `self._plotter.interactor` directly

**Changed in `_render_3d()`:**
- `add_mesh()` gains `copy_mesh=False`
- Return value stored in `self._actor`

**Changed in `_apply_filter_update()`:**
- `remove_actor` + `pv.PolyData` rebuild removed
- Replaced with in-place `self._cloud["spike_density"] = ...` +
  `self._actor.mapper.scalar_range = ...`

**Reused existing code:**
- `DraggableFilterRect` — unchanged, works in dock widget context
- All data loading, scatter drawing, session switching — unchanged

---

## Implementation Plan

### Phase 1: QDockWidget Layout
**Goal:** Replace the broken grid overlay with a dock widget so mouse events
reach the matplotlib canvas.

- [x] Task 1.1 — Rewrite `_build_ui()`: remove `QGridLayout` overlay
  container, set `self._plotter.interactor` as central widget, create
  `QDockWidget` containing `self._canvas`, dock at `Qt.BottomDockWidgetArea`
- [x] Task 1.2 — Update imports: remove `QGridLayout`, add `QDockWidget`
- [x] Task 1.3 — Remove transparency attributes: `WA_TranslucentBackground`,
  `WA_NativeWindow`, transparent stylesheet, `canvas.raise_()`
- [x] Task 1.4 — Make figure background opaque dark `(0.1, 0.1, 0.1)` without
  alpha channel; replace `setFixedSize` with `setMinimumSize`
- [x] Task 1.5 — Remove `self._canvas.raise_()` from `_deferred_start()`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py`
  — `_build_ui()`, `_deferred_start()`, imports

**Dependencies:** None

### Phase 2: In-Place Scalar Update
**Goal:** Fix the RF heatmap disappearing on checkbox toggle by bypassing
PyVista's scalar bar clim corruption.

- [x] Task 2.1 — Add `self._actor: Optional[pv.plotting.actor.Actor] = None`
  to `__init__`
- [x] Task 2.2 — In `_render_3d()`: add `copy_mesh=False` to `add_mesh()`,
  store return value in `self._actor`
- [x] Task 2.3 — Rewrite `_apply_filter_update()`: remove `remove_actor` and
  `pv.PolyData` rebuild; instead assign `self._cloud["spike_density"]`
  in-place and set `self._actor.mapper.scalar_range`
- [x] Task 2.4 — In `_load_session()`: reset `self._actor = None` alongside
  `self._cloud = None`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py`
  — `__init__`, `_render_3d()`, `_apply_filter_update()`, `_load_session()`

**Dependencies:** None (parallel with Phase 1)

### Phase 3: Knowledge Base Update
**Goal:** Record confirmed root causes and fixes for future reference.

- [x] Task 3.1 — Update
  `note-rf-explorer-post-layout-bugfix-status.md`: mark both issues as
  resolved, replace "unverified" hypotheses with confirmed root causes and
  links to upstream issues

**Files Modified:**
- `docs/development/knowledge-base/note-rf-explorer-post-layout-bugfix-status.md`

**Dependencies:** Phases 1–2

### Phase 4: Zero-Density Coloring (post-handoff addition)
**Goal:** Replace the blue-on-zero appearance (jet colormap minimum) with a
neutral dark-gray silhouette that makes the forearm outline readable when
spike density is zero everywhere.

**Root cause:** The jet colormap maps value 0 to blue. With most vertices
having zero spike density, the entire forearm appeared as a uniform blue
blob, giving no spatial reference. The filter update (`_apply_filter_update`)
similarly produced all-blue output whenever the filtered region had few spikes.

**Approach:** Set zero-density vertices to `np.nan` before assigning to the
cloud scalar array. PyVista's `nan_color` renders NaN values with a flat
opaque color — `[0.3, 0.3, 0.3]` (pre-multiplied dark gray, equivalent to
50% transparent gray on black background). All vertices remain fully opaque,
so VTK uses the fast opaque rendering path. The jet colormap and scalar bar
are unchanged.

**Rejected approaches:**
- Per-vertex RGBA with `alpha=0.5` for zero-density vertices (triggers VTK's
  transparency pipeline — observed ~3–4× slower render; scalar bar lost
  because `rgb=True` mode disables automatic scalar bar)
- Forearm PLY vertex colors as base layer (the session PLY files used by the
  explorer have no vertex colors; adds `load_forearm_vertex_colors()` overhead
  for all 11 sessions at startup)

- [x] Task 4.1 — In `_render_3d()`: convert zero-density vertices to NaN
  before assigning to `self._cloud["spike_density"]`; add `nan_color=[0.3,
  0.3, 0.3]` to `add_mesh()` call
- [x] Task 4.2 — In `_apply_filter_update()`: apply same NaN masking to
  `vertex_spikes_display` before the in-place cloud assignment

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py`
  — `_render_3d()`, `_apply_filter_update()`

**Dependencies:** Phase 2

---

### Phase 5: Checkbox-Filtered Scatter + Settings Persistence
**Goal:** Scatter plot respects checkbox state; velocity width and pressure
height persist across runs via a JSON sidecar.

**Scatter filtering:** `_draw_scatter()` now reads `self._checkboxes` and only
plots gesture types whose checkbox is checked. Axis limits are still computed
from all data (not just visible types) so the coordinate system stays
consistent. After `ax.cla()` clears the axes, the filter rect patch is
re-added before `canvas.draw()`.

**Settings persistence:** `_save_settings()` writes
`{"velocity_width": ..., "pressure_height": ...}` to
`<database>/4_analysed/rf_explorer_settings.json`. `_load_settings()` reads
it back and applies values to the spinboxes (which triggers
`_update_rect_from_controls` via the existing signal connection). Called after
`_init_filter_rect()` in both `_deferred_start()` and `_load_session()`, and
auto-saved in `closeEvent()`. An explicit "Save filter dims" button in the
toolbar lets the user save at any point without closing.

- [x] Task 5.1 — Modify `_draw_scatter()`: filter by checkbox state; re-add
  rect patch after `ax.cla()`
- [x] Task 5.2 — Modify `_on_checkbox_changed()`: call `_draw_scatter()` then
  `_apply_filter_update()`
- [x] Task 5.3 — Add `settings_path: Optional[Path] = None` to `__init__`;
  add `_load_settings()` and `_save_settings()` methods; call them in
  `_deferred_start()`, `_load_session()`, and `closeEvent()`
- [x] Task 5.4 — Add "Save filter dims" button to toolbar
- [x] Task 5.5 — `launch_feature_space_explorer()`: derive settings path from
  `first_database_path / "4_analysed" / "rf_explorer_settings.json"` and pass
  it to the constructor

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py`
  — imports, `__init__`, `_build_toolbar`, `_draw_scatter`, `_on_checkbox_changed`,
  `_load_settings`, `_save_settings`, `_deferred_start`, `_load_session`, `closeEvent`
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py`
  — `launch_feature_space_explorer`

**Dependencies:** None (independent of Phases 1–4)

---

## Testing Plan

### Manual Verification

- [ ] Launch explorer via DAG (`explore_rf_feature_space` task)
- [ ] Click and drag filter rectangle — responds to all 8-zone interactions
- [ ] Drag rectangle to new position — 3D heatmap updates in real time
- [ ] Resize rectangle from corner — geometry updates correctly
- [ ] Uncheck all gesture types → RF clears (forearm visible, no spike color)
- [ ] Re-check all gesture types → RF reappears matching initial state
- [ ] Toggle individual types in random order — RF always correct
- [ ] Switch sessions — scatter, 3D, and filter all reset correctly
- [ ] Resize dock panel, float it, re-dock — no crash
- [ ] Verify scalar bar range stays at `(0, max_spikes)` across all toggles

### Edge Cases

- [ ] All checkboxes unchecked then rectangle dragged — no crash, empty RF
- [ ] Rectangle moved fully outside data range — empty heatmap, no crash
- [ ] Session with very few valid frames (<10) — renders without crash
- [ ] Session with no spikes — forearm silhouette visible as dark gray, no spike color

---

## Documentation Plan

- [ ] Update `docs/development/knowledge-base/note-rf-explorer-post-layout-bugfix-status.md`
  — confirmed root causes, fix descriptions, upstream issue links
- [ ] Update `docs/development/knowledge-base/note-rf-feature-space-explorer-gui-components.md`
  — reflect QDockWidget layout replacing grid overlay

---

## Rollback Plan

All changes confined to one source file plus one knowledge base note:

1. Revert changes to `gui/rf_feature_space_explorer.py` — restores grid
   overlay layout and remove/add-mesh filter update
2. Revert changes to `note-rf-explorer-post-layout-bugfix-status.md`

No migrations, no config changes, no API changes.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `copy_mesh=False` causes stale render on session switch | Low | Medium | `_load_session()` calls `plotter.clear()` + fresh `_render_3d()`, creating new cloud + actor |
| Dock widget bottom position wastes vertical space | Low | Low | User can float or resize the dock; `setMinimumSize` keeps it compact |
| `actor.mapper.scalar_range` API changes in future PyVista | Low | Low | Pinned to 0.46.1; mapper API is stable since 0.43 |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: QDockWidget layout | ~30 min | None |
| Phase 2: In-place scalar update | ~20 min | None |
| Phase 3: Knowledge base update | ~10 min | Phases 1–2 |

---

## References

- Active parent plan: `docs/development/plans/active/rf-explorer-layout-bugfix-cache.md`
- [pyvistaqt#91 — Qt overlay on VTK widget is unsupported](https://github.com/pyvista/pyvistaqt/issues/91)
- [pyvista#1366 — clim not updating due to keep-extremum logic](https://github.com/pyvista/pyvista/issues/1366)
- [pyvista#2899 — clim ignored for certain scalar types](https://github.com/pyvista/pyvista/issues/2899)
- [PyVista add_mesh docs — copy_mesh parameter](https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.add_mesh)
- [PyVista DataSetMapper.scalar_range](https://docs.pyvista.org/api/plotting/_autosummary/pyvista.datasetmapper.scalar_range)
