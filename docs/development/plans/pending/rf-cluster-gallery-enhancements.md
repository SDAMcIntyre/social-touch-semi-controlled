# Plan: RF Cluster Gallery Viewer — Enhancements

**Created:** 2026-04-28 14:00
**Approved:** 2026-04-28 14:30
**Completed:** —
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/rf-cluster-extraction-visualization-split`

---

## Overview

**What:** Three enhancements to the RF Cluster Gallery Viewer: (1) display clustering parameter ranges as text overlay on the 3D view, (2) synchronize camera angles per-session across clusters, (3) centralize all rendering settings into a far-right panel mirroring the camera angle picker's layout.

**Why:** The gallery viewer currently lacks parameter context on the 3D view, loses camera orientation when switching between clusters for the same session, and has limited rendering controls compared to the camera angle picker.

**How:** Extend `_GallerySettings`, restructure the UI layout (bottom panel → right panel), add per-session camera state management, and add a text overlay using PyVista's `add_text()`.

## Problem Statement

1. **No parameter context:** When viewing a cluster's 3D heatmap, the user cannot see which feature ranges define that cluster without hovering tooltips or checking JSON files.
2. **Camera resets per cell:** Switching from session A / cluster X to session A / cluster Y resets the camera, forcing the user to re-orient for every cluster — tedious when comparing clusters for the same forearm.
3. **Limited rendering controls:** The gallery viewer only offers colormap, metric, opacity, and hull wireframe toggles in a bottom strip. The camera angle picker has a full right-side settings panel with color pickers, surface/point-cloud toggle, point size, sphere/cube glyphs, and axes — all absent from the gallery viewer.

## Goals

### In Scope

1. Text overlay (upper-left of 3D view) showing clustering parameter ranges from `cluster_description.json`
2. Per-session camera state: adjusting camera on session A persists across all clusters for that session
3. Right-side settings panel (260px scrollable) with all rendering controls from the camera angle picker, plus gallery-specific controls (metric, hull wireframes)

### Out of Scope

- Persisting camera angles to disk (only in-memory for the viewer session)
- Persisting rendering settings between viewer launches
- Sharing camera angles between the gallery viewer and the camera angle picker
- Extracting shared settings widgets into a common module (both viewers are self-contained)

## Success Criteria

- [ ] Cluster parameter ranges visible as text overlay on upper-left of 3D view when a cell is loaded
- [ ] Camera angle for session A is preserved when switching from cluster X to cluster Y (and vice versa)
- [ ] Right-side settings panel contains: background color, forearm color/surface-toggle/point-size/opacity/spheres, contact colormap/metric/point-size/spheres, scalar bar/hull wireframes/axes
- [ ] Settings changes preserve the current camera (no camera reset on toggle)
- [ ] Thumbnail generation still works correctly with the new settings
- [ ] Default colormap is `"jet"` (extended list: YlOrRd, viridis, plasma, inferno, magma, coolwarm, RdBu_r, jet)

---

## Technical Design

### Approach

All three requirements modify a single primary file (`rf_cluster_gallery_viewer.py`) with one minor change to `rf_cluster_pipeline.py`. The implementation order is R3 (settings panel) → R1 (text overlay) → R2 (camera sync), because R3 restructures the layout that R1 and R2 plug into.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Shared settings widget module | DRY between picker and gallery | Over-abstraction for 2 consumers; couples two independent GUIs | Rejected |
| Bottom panel with more controls | Minimal layout change | Horizontal layout doesn't scale for grouped controls; user explicitly asked for right panel | Rejected |
| Per-cell camera (not per-session) | Finer-grained control | Defeats the purpose: user wants session A to look the same across clusters | Rejected |

### Architecture Changes

**Layout change** — before vs after:

```
BEFORE:                              AFTER:
┌─────────────────────────┐          ┌──────────────────────────────┐
│ toolbar                 │          │ toolbar                      │
├────────┬────────────────┤          ├────────┬───────────┬─────────┤
│sidebar │ 3D plotter     │          │sidebar │ 3D plotter│ settings│
│ 160px  │                │          │ 160px  │  (stretch)│  260px  │
│        │                │          │        │           │ (scroll)│
├────────┴────────────────┤          │        │           │         │
│ bottom settings strip   │          │        │           │         │
└─────────────────────────┘          └────────┴───────────┴─────────┘
```

---

## Implementation Plan

### Phase 1: Centralized Right-Side Settings Panel (R3)
**Goal:** Replace the bottom settings strip with a 260px scrollable right-side panel matching the camera angle picker's structure.
**Started:** 2026-04-28 14:30
**Completed:** 2026-04-28 14:30

- [x] 1.1 — Extend `_GallerySettings` with new fields
- [x] 1.2 — Add `QColorDialog` and `QColor` imports
- [x] 1.3 — Expand `_COLORMAPS` to extended list with jet default
- [x] 1.4 — Add static helpers `_color_button()`, `_slider()`, and `_pick_color()`
- [x] 1.5 — Add `_on_surface_toggle()` method
- [x] 1.6 — Replace `_build_settings_panel()` with `_build_right_settings_panel()`
- [x] 1.7 — Update `_build_ui()` layout (three-panel splitter)
- [x] 1.8 — Update `_build_scene()` for new settings
- [x] 1.9 — Add contact point rendering as separate layer in point cloud mode

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py`

### Phase 2: Cluster Parameter Ranges Text Overlay (R1)
**Goal:** Display clustering parameter ranges as text in the upper-left corner of the 3D view.
**Started:** 2026-04-28 14:30
**Completed:** 2026-04-28 14:30

- [x] 2.1 — Rename `_description_summary_line()` → `description_summary_line()`
- [x] 2.2 — Import `description_summary_line` in the gallery viewer
- [x] 2.3 — Add `plotter.add_text()` overlay in `_build_scene()`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py`
- `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py`

### Phase 3: Per-Session Camera Synchronization (R2)
**Goal:** Camera angles are stored per-session across clusters.
**Started:** 2026-04-28 14:30
**Completed:** 2026-04-28 14:30

- [x] 3.1 — Add `_session_cameras` dict in `__init__`
- [x] 3.2 — Modify `_load_cell()` for per-session camera capture/restore

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py`

---

## Testing Plan

### Manual Verification

- [ ] Launch gallery viewer → right-side settings panel appears with all groups
- [ ] Toggle "Render as surface" off → point cloud mode with contact overlay
- [ ] Change colormap → 3D view updates; default is "jet"
- [ ] Change background color → 3D view background updates
- [ ] Cluster parameter ranges appear as text in upper-left
- [ ] Switch between clusters for same session → camera preserved
- [ ] Settings changes preserve camera orientation
- [ ] Thumbnails generate correctly during startup

---

## Rollback Plan

All changes are in a single feature branch. Rollback = `git revert` the commit(s).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Text overlay in thumbnails | Medium | Low | Small text at thumbnail scale is acceptable |
| Point cloud mode without contact layer | Low | Medium | Added contact overlay in point cloud branch |
