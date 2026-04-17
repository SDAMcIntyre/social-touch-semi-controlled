# Plan: Preprocessing 3D Viewer — Merging Layout Refactor

**Date:** 2026-04-14
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/preprocessing-viewer-merging-layout`

---

## Overview

Refactor the preprocessing `SceneViewer` / `SceneViewerVideoMaker` to adopt the same window structure as the merging `NeuralKinectViewer`: a single window with 3D plotter + scrollable controls on top, frame controls in the middle, and an optional time series panel at the bottom. Introduce a generic `TimeSeriesPanel` widget and wire it into the two preprocessing 3D visualisations to display sticker XYZ coordinates and inter-sticker edge lengths, replacing the current standalone rigidity matplotlib popup.

## Problem Statement

The kinect preprocessing 3D viewers (`view_xyz_stickers` and `view_somatosensory_assessment`) use a flat horizontal layout where controls sit beside the plotter and the rigidity analysis pops up as a separate blocking matplotlib window. The merging viewer already demonstrates a cleaner, more consistent pattern — one integrated window with an optional time series panel below the 3D scene. Unifying the preprocessing viewers to that pattern:

- Reduces cognitive load when switching between preprocessing and merging workflows (one visual language).
- Integrates the rigidity/edge-length plot into the main viewer so it scrolls with the frame cursor instead of popping up disconnected.
- Adds sticker XYZ time series, which is useful for debugging tracking but is currently not visible anywhere in the viewer.
- Cleans up a long-standing duplication of `define_custom_colors` across four files.

## Goals

### In Scope

1. Restructure `SceneViewer.__init__` to match `NeuralKinectViewer._build_ui`'s vertical layout (top row with plotter + scrollable right panel, middle frame controls, optional bottom panel).
2. Create a generic `TimeSeriesPanel(QWidget)` in `preprocessing.common.gui` that accepts configurable subplot specs and replicates `NeuralDataPanel`'s cursor/zoom/dark-theme mechanics.
3. Update `SceneViewerVideoMaker._setup_frame_controls` to match the new `(self) -> QWidget` signature while preserving the Export Video button.
4. Wire the `TimeSeriesPanel` into `view_xyz_stickers_on_depth_data` and `view_somatosensory_3d_scene` showing:
   - 3 subplots for sticker X/Y/Z coordinates (one line per sticker, coloured by sticker colour)
   - 1 subplot for inter-sticker edge lengths (P0-P1, P1-P2, P2-P0)
5. Remove the separate `plt.show(block=False)` rigidity popup from `view_xyz_stickers_on_depth_data` (data now lives in the integrated panel).
6. Consolidate the duplicated `define_custom_colors` function into `preprocessing.common` and update all importers.

### Out of Scope

- Refactoring the 2D Tkinter viewers (`EllipseFitViewGUI`, `ConsolidatedTracksReviewGUI`) — these are video playback windows with a fundamentally different concern; converting them would be a full PyQt5 rewrite.
- Retrofitting `NeuralDataPanel` in the merging viewer to use the new `TimeSeriesPanel`. The new panel is designed to be forward-compatible with that refactor, but the merging viewer is untouched in this work.
- GPU cropping / `FramePreloader` / sticker velocity compasses — these are merging-specific enhancements.
- Playback controls (play/pause button, LOD spinbox, crop spinbox) — not needed for preprocessing use cases.

## Success Criteria

- [ ] `view_xyz_stickers_on_depth_data` opens a single window with the merging-style vertical layout (3D plotter + scrollable right panel on top, frame controls middle, time series bottom).
- [ ] The time series panel shows 4 subplots (X, Y, Z, edge lengths) with a red vertical cursor that moves when the frame slider is dragged.
- [ ] Mouse-wheel over the time series zooms in/out, and the `±` spinbox stays synchronised.
- [ ] The standalone rigidity matplotlib popup no longer appears.
- [ ] `view_somatosensory_3d_scene` shows the same layout and time series behaviour.
- [ ] `SceneViewerVideoMaker` still exports video correctly; the Export Video button is visible in the frame controls row.
- [ ] `define_custom_colors` is defined in exactly one place (`preprocessing.common.gui.scene_viewer`); all four other copies are removed and their call sites import from the canonical location.
- [ ] Existing preprocessing 2D Tkinter viewers continue to work unchanged.

---

## Technical Design

### Approach

Mirror the proven layout pattern from `NeuralKinectViewer._build_ui` (line 779-814 of `code/src/merging/gui/neural_kinect_scene_viewer.py`) in the preprocessing `SceneViewer`. Extract the cursor/zoom/matplotlib mechanics from `NeuralDataPanel` (line 322-461) into a generic `TimeSeriesPanel` that takes subplot specifications as data, allowing both preprocessing (sticker XYZ + edges) and — in a future, out-of-scope refactor — the merging viewer to use the same widget.

The `SceneViewerVideoMaker._setup_frame_controls` override pattern stays the same (duplicate the base implementation and add the Export button), but its signature changes from `(self, parent_layout) -> None` to `(self) -> QWidget` so it composes cleanly into the new outer `QVBoxLayout`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Extract `TimeSeriesPanel` as new generic widget, reuse in preprocessing | Clean separation, future-proofs merging refactor, callers pass arbitrary `SubplotSpec` lists | Slightly more code than a one-off panel | **Chosen** |
| Import and reuse `NeuralDataPanel` directly from `merging.gui` | Zero new code | Hardcoded to `Nerve_freq`/`contact_depth`/`contact_area` columns; inverts dependency direction (preprocessing depending on merging); fragile | Rejected |
| Add playback/crop/LOD controls to SceneViewer too | Full feature parity with merging | Out of scope; these features rely on merging-specific machinery (FramePreloader, GPU AABB cropping) | Rejected |
| Keep rigidity popup separate from time series panel | Minimal change to existing analysis code | Defeats the "one window" goal; the cursor doesn't sync | Rejected |
| Merge all 5 preprocessing viz tasks into one window | Most literal interpretation of "one window" | The 2D Tkinter viewers show different modalities (raw video) and would require a complete rewrite; would break the existing task isolation | Rejected |

### Architecture Changes

**New module**: `code/src/preprocessing/common/gui/time_series_panel.py`
- `LineSpec` dataclass: `label`, `data` (1D ndarray), `color`, `linewidth`
- `SubplotSpec` dataclass: `ylabel`, `lines: List[LineSpec]`
- `TimeSeriesPanel(QWidget)`: `__init__(subplot_specs, total_frames, *, fps=30.0, default_zoom_seconds=15.0)` and `update_cursor(frame_idx: int)`

**Modified module**: `code/src/preprocessing/common/gui/scene_viewer.py`
- `SceneViewer.__init__` rebuilt with `QVBoxLayout` outer and `QScrollArea`-wrapped right panel
- `_setup_frame_controls(self) -> QWidget` — new signature, returns the row widget
- `_setup_object_controls_panel` removed (inlined)
- New method `set_time_series_panel(panel)` appends panel to outer layout and stores reference
- `_on_slider_change` also calls `time_series_panel.update_cursor(value)` when panel set
- `define_custom_colors` function relocated here (canonical home)

**Modified module**: `code/src/preprocessing/common/gui/scene_viewer_video_maker.py`
- `_setup_frame_controls` override updated to `(self) -> QWidget` signature

**Modified module**: `code/src/preprocessing/common/__init__.py`
- Exports `TimeSeriesPanel`, `SubplotSpec`, `LineSpec`, `define_custom_colors`

**Updated call sites**:
- `code/scripts/_3_preprocessing/_1_sticker_tracking/view_xyz_stickers_with_depth_data.py` — wire TimeSeriesPanel, drop local `define_custom_colors`, change `plot_variations=True` to `False`
- `code/scripts/_3_preprocessing/_4_somatosensory_quantification/view_somatosensory_3d_scene.py` — wire TimeSeriesPanel, switch `define_custom_colors` import
- `code/scripts/_3_preprocessing/_1_sticker_tracking/view_summary_stickers_on_rgb_data.py` — drop local `define_custom_colors`
- `code/scripts/_3_preprocessing/_6_metadata_matching/define_trial_chunks.py` — drop local `define_custom_colors`
- `code/src/merging/gui/neural_kinect_scene_viewer.py` — import `define_custom_colors` from `preprocessing.common`

**Target layout** (matches `NeuralKinectViewer._build_ui`):

```
QVBoxLayout (outer)
├── top_widget (QHBoxLayout, stretch=4)
│   ├── plotter_widget (stretch=4)
│   │   └── QtInteractor
│   └── QScrollArea (fixed 220 px, stretch=1)
│       └── _right_panel (QVBoxLayout: object groupboxes)
├── frame_controls_widget (returned by _setup_frame_controls)
└── time_series_panel (optional, added via set_time_series_panel)
```

---

## Implementation Plan

### Phase 1: Foundation

**Goal:** Land the new reusable widget and the `define_custom_colors` consolidation. No behavioural change to existing viewers yet.

- [x] Task 1.1 — Create `TimeSeriesPanel`, `SubplotSpec`, `LineSpec` in new module. Model the matplotlib/zoom mechanics on `NeuralDataPanel` but parameterise the subplots. Compute height as `max(180, 55 * n_subplots + 40)` clamped to 400 px.
- [x] Task 1.2 — Relocate `define_custom_colors` into `scene_viewer.py` (canonical home).
- [x] Task 1.3 — Remove local duplicates in the four other files and switch them to `from preprocessing.common import define_custom_colors`.
- [x] Task 1.4 — Update `code/src/preprocessing/common/__init__.py` to export `TimeSeriesPanel`, `SubplotSpec`, `LineSpec`, `define_custom_colors`.

**Files Modified:**
- `code/src/preprocessing/common/gui/time_series_panel.py` — **new file** with dataclasses and widget
- `code/src/preprocessing/common/gui/scene_viewer.py` — add `define_custom_colors`
- `code/src/preprocessing/common/__init__.py` — export new names
- `code/src/merging/gui/neural_kinect_scene_viewer.py` — remove local `define_custom_colors` (line 105), import from `preprocessing.common`
- `code/scripts/_3_preprocessing/_1_sticker_tracking/view_xyz_stickers_with_depth_data.py` — remove local `define_custom_colors` (line 174-201), import from `preprocessing.common`
- `code/scripts/_3_preprocessing/_1_sticker_tracking/view_summary_stickers_on_rgb_data.py` — remove local (line 15), import from `preprocessing.common`
- `code/scripts/_3_preprocessing/_6_metadata_matching/define_trial_chunks.py` — remove local (line 26), import from `preprocessing.common`

**Dependencies:** None

### Phase 2: SceneViewer Layout Refactor

**Goal:** Restructure `SceneViewer` to the merging vertical layout. All existing viewers continue to work (no time series attached yet).

- [x] Task 2.1 — Rebuild `SceneViewer.__init__` with outer `QVBoxLayout`: top `QHBoxLayout` (plotter + `QScrollArea` wrapping `_right_panel`), middle frame controls, placeholder reference for optional bottom panel. Import `QScrollArea`.
- [x] Task 2.2 — Change `_setup_frame_controls` signature to `(self) -> QWidget` and return a constructed row widget.
- [x] Task 2.3 — Remove `_setup_object_controls_panel` (inline its content into `__init__`).
- [x] Task 2.4 — Add `set_time_series_panel(panel: QWidget)` public method; store reference and append to outer layout.
- [x] Task 2.5 — Update `_on_slider_change` to call `self.time_series_panel.update_cursor(value)` when panel is attached.
- [x] Task 2.6 — Update `SceneViewerVideoMaker._setup_frame_controls` to the new `(self) -> QWidget` signature, preserving the Export Video button.
- [ ] Task 2.7 — Verify `SceneViewer.main()` and `SceneViewerVideoMaker.main()` demo functions still run (they use no time series panel — should still work).

**Files Modified:**
- `code/src/preprocessing/common/gui/scene_viewer.py` — layout refactor, new `set_time_series_panel`
- `code/src/preprocessing/common/gui/scene_viewer_video_maker.py` — signature update on `_setup_frame_controls`

**Dependencies:** Phase 1 (needs `TimeSeriesPanel` imported for type hints; can use forward reference string)

### Phase 3: Integration

**Goal:** Wire the time series panel into the two preprocessing 3D viewers and remove the standalone rigidity popup.

- [x] Task 3.1 — In `view_xyz_stickers_on_depth_data`, after adding scene objects and before `viewer.show()`, build:
  - 3 `SubplotSpec` for X/Y/Z (one `LineSpec` per sticker, `color=custom_colors[sticker_name]`)
  - 1 `SubplotSpec` for edge lengths (3 `LineSpec`: P0-P1 red, P1-P2 green, P2-P0 blue) computed from `coordinates_over_time`
  - Create `TimeSeriesPanel(all_subplots, total_frames=max_len, fps=30.0)` and call `viewer.set_time_series_panel(panel)`.
- [x] Task 3.2 — Flip `plot_variations=True` to `False` in the `validate_triangle_consistency` call; drop the now-unused `import matplotlib.pyplot as plt`.
- [x] Task 3.3 — In `view_somatosensory_3d_scene`, build the same XYZ + edge-length subplots and attach via `viewer.set_time_series_panel(panel)`. Switch `define_custom_colors` import.

**Files Modified:**
- `code/scripts/_3_preprocessing/_1_sticker_tracking/view_xyz_stickers_with_depth_data.py`
- `code/scripts/_3_preprocessing/_4_somatosensory_quantification/view_somatosensory_3d_scene.py`

**Dependencies:** Phase 1 + Phase 2

---

## Testing Plan

### Unit Tests

This feature is GUI-heavy; automated unit tests are limited to non-GUI helpers.

- [ ] `SubplotSpec` / `LineSpec` dataclass construction with valid and invalid inputs.
- [ ] `define_custom_colors` returns expected mapping for sticker-name inputs (moved from duplicates — existing behaviour should be unchanged).

### Integration Tests

- [ ] Instantiate `SceneViewer()` in a test harness, confirm `central_widget.layout()` is a `QVBoxLayout` with three expected children (top widget, frame controls widget, initially no time series).
- [ ] Call `viewer.set_time_series_panel(panel)` and confirm the layout now has a fourth child and `viewer.time_series_panel` references it.
- [ ] Instantiate `SceneViewerVideoMaker()` and confirm the frame controls row contains the Export button.

### Manual Verification

- [ ] Run `python -m view_xyz_stickers_with_depth_data` (or via the `__main__` block) on a known-good session:
   - Window opens with vertical layout; right panel is scrollable at 220 px width.
   - Time series shows 4 stacked subplots with coloured lines.
   - Dragging the slider moves the red cursor in the time series.
   - Mouse wheel over the time series zooms, `±` spinbox syncs.
   - No separate rigidity matplotlib window pops up.
- [ ] Run `view_somatosensory_3d_scene` on the same session — verify same layout and behaviour; hand mesh still renders.
- [ ] Run `SceneViewerVideoMaker` via `view_somatosensory_3d_scene`, click Export Video on a short frame range; confirm a playable MP4 is produced.
- [ ] Run the full `preprocess_workflow_kinect_visualisation.py` DAG with `view_xyz_stickers` and `view_somatosensory_assessement` enabled — verify both launch successfully.

### Edge Cases

- [ ] Session where one or more stickers have many NaN frames — matplotlib should break lines at NaN, not crash.
- [ ] Only 2 stickers present (edge-length computation assumes 3 for triangle edges) — gracefully skip the edge subplot.
- [ ] `define_custom_colors` with sticker names containing no known colour keyword — default colour (`magenta`) still returned by callers.
- [ ] Very long recording (>3600 s) — zoom spinbox cap at 3600 s still prevents pathological xlim values.

---

## Documentation Plan

- [ ] Update `code/src/preprocessing/common/__init__.py` docstring listing the new exported classes.
- [ ] Inline docstrings on `TimeSeriesPanel`, `SubplotSpec`, `LineSpec`, and `SceneViewer.set_time_series_panel`.
- [ ] No CLAUDE.md change required (no new architectural invariants introduced).
- [ ] No new knowledge-base note required unless a surprising issue surfaces during implementation.

---

## Rollback Plan

All changes are local to the `feature/preprocessing-viewer-merging-layout` branch. Rollback steps:

1. **Before merge**: simply delete the branch. Nothing on `main` or `dev` has been touched.
2. **After merge to dev but before publish**: `git revert` the merge commit on `dev`.
3. **After publish to main**: `git revert` the publish merge commit. No data migrations or stored state affected — this is a pure GUI refactor, so revert is safe at any point.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `QScrollArea` sizing quirks hide right-panel groupboxes in some resolutions | Low | Low | `setWidgetResizable(True)` + fixed width 220 px matches what merging viewer uses in production |
| Subclass override breakage (`_setup_frame_controls` signature change) affects other undiscovered subclasses | Low | Medium | `grep` before implementation; only `SceneViewerVideoMaker` overrides it today |
| Edge-length subplot crashes when sticker count ≠ 3 | Medium | Low | Guard the edge computation; build the subplot only when `len(stickers_xyz_dict) == 3` |
| `define_custom_colors` relocation breaks a caller that was relying on subtle local behaviour | Low | Low | All four local copies are textually identical; `grep` confirms no divergent logic |
| Matplotlib figure size inside fixed-height Qt widget renders oddly on Hi-DPI screens | Low | Low | `tight_layout=True` already used in `NeuralDataPanel`; copy that setting |
| Removing the blocking `plt.show(block=False)` popup changes perceived workflow for users who relied on it | Low | Low | Console print of rigidity metrics is retained; edges are now always visible in the cursor-synced panel |

---

## References

- Reference pattern: `code/src/merging/gui/neural_kinect_scene_viewer.py` — `_build_ui` (line 779-814) and `NeuralDataPanel` (line 322-461)
- Reusable base: `code/src/preprocessing/common/gui/scene_viewer.py` — `SceneViewer` (line 331-497)
- Reusable base: `code/src/preprocessing/common/gui/scene_viewer_video_maker.py` — `SceneViewerVideoMaker` (line 24)
- Affected view functions:
  - `code/scripts/_3_preprocessing/_1_sticker_tracking/view_xyz_stickers_with_depth_data.py` (line 203)
  - `code/scripts/_3_preprocessing/_4_somatosensory_quantification/view_somatosensory_3d_scene.py` (line 38)
- Historical plan noting `define_custom_colors` consolidation as follow-up: `docs/development/plans/completed/neural-kinect-viewer-files.md` (line 76)
