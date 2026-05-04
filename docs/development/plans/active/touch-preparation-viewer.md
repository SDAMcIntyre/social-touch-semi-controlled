# Plan: Touch Preparation Viewer

**Date:** 2026-05-04
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `dev`
**Branch:** `feature/touch-preparation-viewer`

---

## Overview

**What:** A new PyQt5 viewer that displays the output of the touch preparation
pipeline (Stage 1 of touch_analytics) — per-touch time-series signals on a
matplotlib panel alongside a 3D forearm view with the contact centroid rendered
per frame.

**Why:** The preparation pipeline produces cleaned, interpolated CSVs
(`<session>_prepared.csv`) but there is no way to visually inspect the output.
Users need to verify that interpolation is reasonable, gesture classifications
are correct, and per-touch signal profiles make sense before downstream stages
consume the data.

**How:** Create a `TouchPreparationViewer` QMainWindow with a 3D PyVista
forearm view (contact centroid on forearm), a matplotlib time-series panel
(contact_depth, contact_area, contact_location, Nerve_freq, Nerve_spike), and
session/block/trial/touch navigation. Integrate as a new `explore_preparation`
task in the analysis workflow DAG.

## Problem Statement

The preparation pipeline transforms raw merged session CSVs into prepared CSVs:
NaN gaps are filled via interpolation, gesture types are classified, and
non-touch rows are filtered. This is the foundation for all downstream analysis
(series transforms, feature extraction, clustering, RF mapping).

Currently there is no way to visually inspect the preparation output:
- Did cubic interpolation produce reasonable curves for contact_depth and
  contact_area?
- Are gesture classifications (tap / stroke_proximal / stroke_distal) correct?
- Do the interpolated contact trajectories look plausible on the forearm
  surface?

Errors at this stage propagate silently through the entire analysis pipeline.

## Goals

### In Scope

1. New `TouchPreparationViewer` QMainWindow displaying prepared CSV data
2. 3D PyVista forearm view with contact centroid rendered per frame
3. Matplotlib time-series panel showing key interpolated signals + neural data
4. Session / block / trial / touch navigation via cascading combo boxes
5. Frame slider with play/pause and speed control (handling 1 kHz frame rate)
6. Gesture type display with color-coded badge
7. New `explore_preparation` DAG task integrated into analysis workflow

### Out of Scope

- Before/after interpolation comparison (raw vs prepared overlay)
- Spike heatmap accumulation (available in TouchPlaybackExplorer)
- Dual 3D views (single forearm view is sufficient)
- Sticker position visualization (sticker_blue/green/yellow columns)
- Preparation pipeline modifications (viewer is read-only)
- .npz sidecar caching (prepared CSVs are small enough to load directly)

## Success Criteria

- [ ] Viewer opens and loads `<session>_prepared.csv` + forearm PLY
- [ ] 3D view shows forearm point cloud with contact centroid as red sphere
- [ ] Time-series panel shows contact_depth, contact_area, contact_location_x, Nerve_freq, Nerve_spike
- [ ] Combo cascade navigates session / block / trial / touch correctly
- [ ] Frame slider scrubs through 1 kHz frames; cursor syncs between 3D and time-series views
- [ ] Play/pause animates frames with configurable speed
- [ ] Gesture type badge shows correct type with color coding
- [ ] DAG task `explore_preparation` runs after `touch_preparation` in analysis workflow

---

## Technical Design

### Approach

Create a standalone `TouchPreparationViewer(QMainWindow)` with a horizontal
splitter: 3D PyVista `QtInteractor` on the left, matplotlib `FigureCanvas` on
the right. A toolbar provides cascading combo boxes for navigation, and a
bottom bar provides frame controls. The data model is a separate module with
dataclasses and loading functions.

The viewer lives in a new `touch_analytics/gui/` subpackage because it
validates touch_analytics output — not RF mapping data. This establishes a
clean pattern: viewers go in the `gui/` subpackage of the relevant analysis
sub-system.

The prepared CSV has `contact_location_x/y/z` as scalar columns (one contact
centroid per frame at 1 kHz). This differs from `TouchPlaybackExplorer` which
parses multi-point `contact_points` string arrays at 30 Hz. The simpler data
model means a lighter data loader.

Since the prepared CSV is at 1 kHz (~1000 frames/sec vs ~30 in
TouchPlaybackExplorer), playback uses frame-skipping controlled by a speed
spinbox. The slider still addresses all frames for precise scrubbing.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| New standalone viewer in `touch_analytics/gui/` | Clean separation, no RF mapping dependency, establishes gui subpackage pattern | New directory | **Chosen** |
| Extend TouchPlaybackExplorer with a "preparation mode" | Code reuse | Tightly coupled to series-augmented CSV format and spike heatmap logic; would complicate an already 663-line class | Rejected |
| Add to `receptive_field_mapping/gui/` alongside existing viewers | Consistent location | Preparation has no RF mapping concern; would blur package boundaries | Rejected |
| Time-series only (no 3D view) | Simpler, no PyVista dependency | Cannot validate contact trajectory on forearm surface; inconsistent with other analysis viewers | Rejected |

### Architecture Changes

```
code/src/analysis/touch_analytics/gui/
    __init__.py                           NEW  ~10 lines — exports
    preparation_viewer_data.py            NEW  ~200-250 lines — data model + loading
    touch_preparation_viewer.py           NEW  ~500-600 lines — QMainWindow viewer
```

**Data model:**

```python
@dataclass
class PreparationTouchData:
    block_order_id: str
    trial_id: int
    single_touch_id: int
    gesture_type: str
    time: np.ndarray                        # (n_frames,) seconds
    signals: dict[str, np.ndarray]          # column_name -> (n_frames,) values
    contact_location: np.ndarray            # (n_frames, 3) — x/y/z RF-centered

@dataclass
class PreparationViewerData:
    session_id: str
    forearm_vertices: np.ndarray            # (N, 3) from PLY
    block_order_ids: list[str]
    trial_ids_by_block: dict[str, list[int]]
    touches_by_block_trial: dict[tuple, list[PreparationTouchData]]
```

**Signals displayed in time-series panel:**

| Signal | Column | Default visible |
|--------|--------|----------------|
| Contact depth | `contact_depth` | Yes |
| Contact area | `contact_area` | Yes |
| Contact location X | `contact_location_x` | Yes |
| Contact location Y | `contact_location_y` | Toggle-able |
| Contact location Z | `contact_location_z` | Toggle-able |
| Neural firing rate | `Nerve_freq` | Yes |
| Neural spikes | `Nerve_spike` | Yes (raster ticks) |

**UI layout:**

```
+-----------------------------------------------------------------------+
| Session: [v]  Block: [v]  Trial: [v]  Touch: [v]  | tap (green)      |
+-----------------------------------------------------------------------+
|                              |  contact_depth  ~~~~~~~~~~~~~~~~~~~~   |
|   [3D PyVista forearm view]  |  contact_area   ~~~~~~~~~~~~~~~~~~~~   |
|   - grey forearm point cloud |  location_x     ~~~~~~~~~~~~~~~~~~~~   |
|   - red contact centroid     |  Nerve_freq     ~~~~~~~~~~~~~~~~~~~~   |
|                              |  Nerve_spike    | | || ||| | || |     |
|                              |                 ^ cursor               |
+------------------------------+----------------------------------------+
| Frame: [========o=========]  142 / 1350   [Play]   [Speed: 1.0x]     |
+-----------------------------------------------------------------------+
```

**Reusable code from existing codebase:**

| Component | Source | How reused |
|-----------|--------|-----------|
| `resolve_forearm_ply()` | `rf_data_loader.py:98` | Forearm PLY path resolution |
| `load_forearm_vertices()` | `rf_data_loader.py` | Load PLY as (N, 3) array |
| `session_id_from_path()` | `pipeline_shared.py` | Extract session ID from CSV path |
| `group_touches()` | `preparation/grouping.py` | Group prepared CSV by touch identity |
| Combo cascade pattern | `TouchPlaybackExplorer:127-238` | Session/block/trial/touch navigation |
| Frame slider + play/pause | `PostprocessingStageViewer:322-349` | Frame control bar |

**Knowledge base constraints applied:**

- `note-qt-itemchanged-signal-recursion.md` — use `blockSignals()` when
  programmatically updating combo boxes in signal handlers to prevent infinite
  re-entry

---

## Implementation Plan

### Phase 1: Data model
**Goal:** Create the data loading module that reads prepared CSVs and forearm PLYs into structured dataclasses
**Started:** 2026-05-04
**Completed:** 2026-05-04

- [x] 1.1 — Create `code/src/analysis/touch_analytics/gui/__init__.py` with exports
- [x] 1.2 — Define `PreparationTouchData` and `PreparationViewerData` dataclasses
- [x] 1.3 — Implement `load_preparation_viewer_data(prepared_csv, forearm_ply)`: read CSV, group by `(block_order_id, trial_id, single_touch_id)` using `group_touches()`, extract per-touch time/signals/contact_location arrays, load forearm via `load_forearm_vertices()`, build cascade structures
- [x] 1.4 — Implement `resolve_preparation_paths(input_items)`: for each `(csv_path, database_path)`, resolve prepared CSV at `database_path / '4_analysed' / 'preparation' / f'{session_id}_prepared.csv'` and forearm PLY via `resolve_forearm_ply(csv_path.parent, session_id)`. Raise if either is missing.
- [x] 1.5 — Implement `launch_preparation_viewer(input_items)`: resolve paths, load data per session (threaded if multiple), create QApplication, instantiate viewer, `app.exec_()`

**Files Modified:**
- `code/src/analysis/touch_analytics/gui/__init__.py` — NEW, ~10 lines
- `code/src/analysis/touch_analytics/gui/preparation_viewer_data.py` — NEW, ~200-250 lines

**Dependencies:** None

### Phase 2: GUI viewer
**Goal:** Create the PyQt5 QMainWindow with 3D forearm view, time-series panel, and frame controls
**Started:** 2026-05-04
**Completed:** 2026-05-04

- [x] 2.1 — Implement `TouchPreparationViewer.__init__()`: accept `PreparationViewerData` + optional sessions list, initialize state (current touch, frame index, timer), call `_build_ui()`
- [x] 2.2 — Implement `_build_toolbar()`: session / block / trial / touch QComboBoxes in a QToolBar, gesture type QLabel badge with color coding (tap=green, stroke_proximal=blue, stroke_distal=red, stroke_unknown=grey). Cascade pattern: changing session repopulates blocks, changing block repopulates trials, etc. Use `blockSignals()` guards.
- [x] 2.3 — Implement 3D view panel: PyVista `QtInteractor` showing forearm point cloud (grey) + contact centroid as red sphere. `_update_3d_frame(idx)` moves the sphere to `contact_location[idx]`. Black background, interactive camera.
- [x] 2.4 — Implement time-series panel: matplotlib `FigureCanvas` with stacked subplots (shared X = time axis). Default: contact_depth, contact_area, contact_location_x, Nerve_freq, Nerve_spike raster. Red vertical cursor line on all axes synced with frame slider.
- [x] 2.5 — Implement frame controls bar: QSlider (range 0 to n_frames-1), frame label ("X / Total"), play/pause QPushButton with QTimer, speed QDoubleSpinBox (0.1x–10x). Timer interval = `33ms / speed` with frame-skipping to maintain real-time rate at 1 kHz.
- [x] 2.6 — Implement `_load_touch(touch)`: update all 3D actors and matplotlib plots for the selected touch. Reset frame slider range, cursor, and playback state.
- [x] 2.7 — Implement `_update_frame(idx)`: move 3D contact sphere, update cursor `xdata` on all axes, update frame label. Use `canvas.draw_idle()` for efficient redraws.
- [x] 2.8 — Implement Qt lifecycle: `showEvent` / `_deferred_start` for delayed VTK init, `closeEvent` for cleanup, slider drag throttling (80ms QTimer).

**Files Modified:**
- `code/src/analysis/touch_analytics/gui/touch_preparation_viewer.py` — NEW, ~500-600 lines
- `code/src/analysis/touch_analytics/gui/__init__.py` — Add viewer export

**Dependencies:** Phase 1

### Phase 3: Integration
**Goal:** Wire the viewer into the analysis workflow DAG
**Started:** 2026-05-04
**Completed:** 2026-05-04

- [x] 3.1 — Add `explore_preparation_flow` Prefect flow function to `analysis_workflow.py`: accept `input_items` and `force_processing`, call `launch_preparation_viewer(input_items)`. Follow `explore_touch_playback_flow` pattern (line 481).
- [x] 3.2 — Add `("explore_preparation", explore_preparation_flow)` to the `available_tasks` list (line 563-578). Place after `touch_preparation` and before `touch_series_transforms`.
- [x] 3.3 — Add import of `launch_preparation_viewer` from `analysis.touch_analytics.gui` at top of script.
- [x] 3.4 — Add `explore_preparation` task entry to `analyse_workflow_dag.yaml`: `enabled: false`, `depends_on: [touch_preparation]`, with description.

**Files Modified:**
- `code/scripts/analysis_workflow.py` — Add import, flow function, available_tasks entry (~25 lines)
- `configs/analyse_workflow_dag.yaml` — Add task entry (~8 lines)

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests

No unit tests for this feature — GUI viewers in this project are not
unit-tested (no existing test pattern for PyQt5+PyVista viewers). All existing
viewers (`TouchPlaybackExplorer`, `RFFeatureSpaceExplorer`,
`RFClusterGalleryViewer`, `PostprocessingStageViewer`) rely on manual
verification.

### Manual Verification

- [ ] Enable `explore_preparation: enabled: true` in DAG config and run the analysis workflow
- [ ] Viewer opens with forearm point cloud visible in 3D view
- [ ] Contact centroid renders as red sphere at correct position on forearm
- [ ] Time-series panel shows contact_depth, contact_area, contact_location_x, Nerve_freq, Nerve_spike
- [ ] Session combo lists all loaded sessions (when multiple)
- [ ] Block combo repopulates when session changes
- [ ] Trial combo repopulates when block changes
- [ ] Touch combo repopulates when trial changes
- [ ] Gesture type badge updates with correct color on touch change
- [ ] Frame slider scrubs through all 1 kHz frames; 3D sphere and cursor move in sync
- [ ] Play/pause animation works at default speed
- [ ] Speed spinbox changes playback rate (e.g., 2x = twice as fast)
- [ ] Closing the viewer returns control to the workflow

### Edge Cases

- [ ] Session with only tap touches — no stroke trajectory, centroid should be roughly stationary
- [ ] Session with stroke_unknown gesture type — badge shows grey, viewer does not crash
- [ ] Touch with very few frames (<10) — slider and playback still work
- [ ] Forearm PLY missing — viewer raises with clear error message (fail-fast convention)
- [ ] Prepared CSV missing — viewer raises with clear error message

---

## Documentation Plan

- [ ] No README changes needed (internal visualization tool)
- [ ] No CLAUDE.md changes needed (no architectural pattern changes; analysis CLAUDE.md already documents the gui infrastructure if needed)
- [ ] No user guide needed (internal tool, discoverable via DAG config)

---

## Rollback Plan

All changes are additive — no existing functionality is modified or removed.

1. Delete `code/src/analysis/touch_analytics/gui/` directory
2. Revert additions to `analysis_workflow.py` (import, flow function, available_tasks entry)
3. Revert additions to `analyse_workflow_dag.yaml` (explore_preparation task)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 1 kHz frame rate causes sluggish matplotlib redraws during playback | Med | Med | Use `set_xdata()` on cursor axvline + `draw_idle()` instead of full redraws; skip frames at higher speeds |
| PyVista `QtInteractor` conflicts with matplotlib `FigureCanvas` in same window | Low | High | Both are standard Qt widgets; same pattern used successfully in `RFFeatureSpaceExplorer` (PyVista + matplotlib scatter in one window) |
| Forearm PLY coordinate space mismatch with contact_location | Low | High | Both are in RF-centered space — `resolve_forearm_ply()` documentation confirms this. Validate visually during manual testing. |
| Large sessions (>100 touches, >500k rows) cause slow load times | Low | Low | Prepared CSVs are typically 50-200k rows; `pd.read_csv` handles this in <1s. Can add progress indicator if needed. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Data model | ~200-250 lines new code | None |
| Phase 2: GUI viewer | ~500-600 lines new code | Phase 1 |
| Phase 3: Integration | ~35 lines modified | Phase 2 |

---

## References

- Completed plan (model): `docs/development/plans/completed/postprocessing-stage-viewer.md`
- Existing viewer pattern: `code/src/analysis/receptive_field_mapping/gui/touch_playback_explorer.py`
- Playback data model: `code/src/analysis/receptive_field_mapping/touch_playback_data.py`
- Forearm PLY resolver: `code/src/analysis/receptive_field_mapping/rf_data_loader.py:98`
- Preparation pipeline: `code/src/analysis/touch_analytics/preparation_pipeline.py`
- Column constants: `code/src/analysis/touch_analytics/preparation/interpolation.py:16-41`
- Knowledge base: `docs/development/knowledge-base/note-qt-itemchanged-signal-recursion.md`

---
