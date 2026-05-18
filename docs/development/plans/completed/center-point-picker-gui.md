# Plan: Interactive Center-Point Picker GUI

**Date:** 2026-05-15
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-18 20:33
**Base Branch:** `feature/flatten-forearm-sandbox`
**Branch:** `feature/center-point-picker-gui`

---

## Overview

Add an interactive 3D point-picking GUI to `flatten_forearm_sandbox.py` so the
user can visually select the flattening center (e.g., RF hot spot) by clicking
on the forearm point cloud, instead of hard-coding a `CENTER_POINT` tuple.
Uses a blocking PyVista `Plotter` with Ctrl+click vertex picking, consistent
with the existing `PointSelectorWindow` pattern in the codebase.

## Problem Statement

The approved center-weighted flattening plan introduces a `CENTER_POINT`
constant that the user must set manually as a 3D coordinate tuple. This is
impractical: the user would need to know the exact 3D coordinates of the RF
hot spot (from `rf_metrics.py` output or external tools) and copy-paste them
into the script. An interactive picker removes this friction and allows
visual confirmation that the selected point is anatomically correct.

## Goals

### In Scope

1. Interactive point-picking window: load the forearm PLY, display it in a
   PyVista plotter, let the user Ctrl+click to select a vertex.
2. Visual feedback: red sphere at the picked vertex, updated on each new pick.
3. Close-to-confirm: closing the window (Q or X) confirms the last pick and
   proceeds to flattening.
4. Tri-state `CENTER_POINT`: `None` (no center), `"interactive"` (show
   picker), or `(x, y, z)` tuple (hard-coded). No new constants.
5. Print the picked 3D coordinate to stdout for future reference (user can
   paste it back as a tuple for reproducible runs).

### Out of Scope

- Non-blocking / signal-based Qt integration (sandbox is a one-shot script).
- Multi-point selection (only one center point needed).
- Saving/loading picked points to/from files.
- Integration with `rf_metrics.py` to auto-populate the RF center.
- PyQt5 dependency — uses standalone PyVista `Plotter` (no `pyvistaqt`).

## Success Criteria

- [ ] Setting `CENTER_POINT = "interactive"` opens a PyVista window showing
      the forearm point cloud.
- [ ] Ctrl+clicking a point places a red sphere on the nearest vertex.
- [ ] Re-clicking moves the sphere to the new vertex (only one sphere
      visible at a time).
- [ ] Closing the window proceeds to flattening with the picked point as
      center.
- [ ] If the user closes without picking, the script raises an error
      (fail-fast, no silent fallback to no-center mode).
- [ ] Setting `CENTER_POINT = None` skips the picker (original behavior).
- [ ] Setting `CENTER_POINT = (x, y, z)` skips the picker and uses the
      hard-coded coordinate (original behavior from the center-weighted
      flattening plan).
- [ ] The picked 3D coordinate is printed to stdout in copy-pasteable
      tuple format.

---

## Technical Design

### Approach

A **blocking PyVista `Plotter`** with a VTK `LeftButtonPressEvent` observer
(Ctrl+click picking), matching the `PointSelectorWindow` pattern at
`code/src/preprocessing/motion_analysis/hand_tracking/gui/point_selector_window.py`.

The picker runs **before** mesh construction and flattening. Workflow:

```
Load PLY → [if interactive] Show picker → User picks → Close window
         → Build mesh → Clean mesh → Resolve picked point to mesh vertex
         → Run flattening methods → Visualize
```

The picked point is a raw 3D coordinate from the original PLY. After mesh
cleaning (which reindexes vertices), it is matched to the nearest mesh
vertex via `find_nearest_vertex()` (from the center-weighted flattening plan).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| PyVista blocking `Plotter` + VTK observer | Consistent with existing `PointSelectorWindow`; no Qt event loop needed; simple blocking flow | Requires Ctrl+click (not plain click, to avoid interfering with camera rotation) | **Chosen** |
| Open3D `VisualizerWithEditing` | Already imported in the script; built-in `pick_points()` | Limited API; no visual feedback (no sphere); awkward multi-step pick flow; less consistent with codebase patterns | Rejected |
| PyVista `enable_point_picking()` built-in | One-line setup | Less control over visual feedback; tooltip-based, no sphere marker; callback signature less flexible | Rejected |
| Full PyQt5 GUI (`pyvistaqt.QtInteractor`) | Rich controls (buttons, sliders) | Overkill for a sandbox; adds Qt event loop complexity; unnecessary for single-point pick | Rejected |

### Architecture Changes

**Modified:** `code/scripts/flatten_forearm_sandbox.py`
- New function: `pick_center_point(points, colors)` (~60 lines)
- Modified constant: `CENTER_POINT` type annotation updated to
  `tuple[float, float, float] | str | None`
- Modified `__main__` block: dispatch on `CENTER_POINT` value

No new files. No new dependencies (PyVista is already in `requirements.txt`).

---

## Implementation Plan

### Phase 1: Picker function
**Goal:** Implement the interactive point-picking window.
**Started:** 2026-05-15
**Completed:** 2026-05-15

- [x] Add `pick_center_point(points, colors) -> np.ndarray` function:
  - Convert `points` (N×3) to `pv.PolyData`
  - If `colors` is not None, assign as vertex scalars (RGB)
  - Create `pv.Plotter` with window title "Select Center Point — Ctrl+Click"
  - Add the point cloud mesh (render as spheres, appropriate point size)
  - Add instruction text overlay: "Ctrl+Click to pick center. Close window (Q) to confirm."
  - Register `LeftButtonPressEvent` observer (VTK interactor callback)
  - In callback: check `GetControlKey()`, ray-pick via `GetPicker().Pick()`,
    find nearest vertex via `mesh.find_closest_point()`, place/replace red
    sphere at picked location, store picked coordinate in closure variable
  - Call `plotter.show()` (blocking)
  - After window closes: if no point was picked, raise `RuntimeError`
  - Return the picked 3D coordinate as `np.ndarray(3,)`
  - Print coordinate in tuple format for reproducibility

**Files Modified:**
- `code/scripts/flatten_forearm_sandbox.py` — add function after `load_pcd()`

**Dependencies:** None

### Phase 2: Wire up CENTER_POINT dispatch
**Goal:** Integrate the picker into the script's main flow.
**Started:** 2026-05-15
**Completed:** 2026-05-15

- [x] Update `CENTER_POINT` type annotation to `tuple[float, float, float] | str | None`
- [x] Update `CENTER_POINT` docstring to document the three modes
- [x] In `__main__`: after loading PLY and before building the mesh, dispatch:
  - `CENTER_POINT == "interactive"` → call `pick_center_point(orig_points, orig_colors)`
  - `CENTER_POINT` is a tuple → `np.array(CENTER_POINT)`
  - `CENTER_POINT is None` → `center_3d = None`
- [x] Pass `center_3d` downstream (to `find_nearest_vertex` and the flattening
      functions, as defined in the center-weighted flattening plan)

**Files Modified:**
- `code/scripts/flatten_forearm_sandbox.py` — modify constants + `__main__` block

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification

- [ ] Set `CENTER_POINT = "interactive"`, run the script — PyVista window
      opens showing the forearm point cloud with vertex colors.
- [ ] Ctrl+click on the point cloud — red sphere appears at the picked vertex.
- [ ] Ctrl+click again — sphere moves to the new location (old sphere removed).
- [ ] Close the window — script prints the picked coordinate and proceeds to
      flattening.
- [ ] Verify the printed coordinate is in `(x, y, z)` format that can be
      copy-pasted back as `CENTER_POINT = (x, y, z)`.
- [ ] Set `CENTER_POINT = None` — no picker window, original behavior.
- [ ] Set `CENTER_POINT = (x, y, z)` with a previously printed coordinate —
      no picker window, uses the hard-coded value.
- [ ] Close the picker window without clicking — script raises `RuntimeError`
      with a clear message.

### Edge Cases

- [ ] PLY with no vertex colors — picker should still work (single-color point
      cloud).
- [ ] Very dense point cloud (50k+ vertices) — verify PyVista renders without
      significant lag.
- [ ] Click on empty space (not on point cloud) — no sphere placed, no crash.

---

## Documentation Plan

- [ ] Update the script's module docstring to document `CENTER_POINT = "interactive"`
      mode.
- [ ] No CLAUDE.md or README changes needed (dev-only sandbox script).

---

## Rollback Plan

1. Revert the `pick_center_point()` function and `__main__` dispatch changes.
2. Restore `CENTER_POINT: tuple[float, float, float] | None = None` type
   annotation.
3. Single-file change, trivially reversible.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| PyVista `Plotter` conflicts with matplotlib `Qt5Agg` backend (both use Qt) | Medium | Medium | PyVista plotter runs and closes before matplotlib is used; no concurrent windows. If conflict occurs, use `matplotlib.use("Agg")` for the save-only path. |
| Ctrl+click not discoverable for users unfamiliar with the pattern | Low | Low | Instruction text overlay on the window; matches existing `PointSelectorWindow` convention. |
| VTK picker ray misses point cloud (sparse regions) | Low | Low | `find_closest_point()` always returns the nearest vertex regardless of ray hit precision. |
| GLFW cleanup warnings at exit | High | None | Expected and benign for single interactive window; documented in knowledge base (`bug-glfw-cleanup-forearm-batch-pipeline.md`). |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Picker function | ~60 lines | None |
| Phase 2: Wire up dispatch | ~15 lines | Phase 1 |

---

## References

- Related Plan: `docs/development/plans/active/flatten-forearm-sandbox.md`
- Existing pattern: `code/src/preprocessing/motion_analysis/hand_tracking/gui/point_selector_window.py`
- Knowledge base: `docs/development/knowledge-base/bug-glfw-cleanup-forearm-batch-pipeline.md`
