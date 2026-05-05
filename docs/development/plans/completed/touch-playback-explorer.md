# Plan: Touch Playback Explorer

**Date:** 2026-05-04
**Author:** Basil Duvernoy
**Status:** Completed
**Base Branch:** `dev`
**Branch:** `feature/touch-playback-explorer`

---

## Overview

**What:** A new analysis GUI tool that animates individual single-touch events frame-by-frame on a 3D forearm model, showing contact points and spike heatmaps evolving over time.

**Why:** There is no way to visualize how a single touch unfolds spatially on the forearm — the existing RF Feature-Space Explorer shows aggregated heatmaps across all frames, not temporal progression. Playing touches individually reveals the spatio-temporal relationship between contact movement and neural spiking.

**How:** A PyQt5 QMainWindow with dual side-by-side PyVista 3D views (contact points + spike heatmap), toolbar dropdowns for session/trial/touch selection, and QTimer-based frame-stepping animation at ~30Hz (deduplicated Kinect rate).

## Problem Statement

- The RF Feature-Space Explorer aggregates all contact frames into a single heatmap — no temporal dimension
- Researchers need to observe how contact points move across the forearm and when/where spikes occur during individual touch events
- No existing tool in the pipeline provides per-frame playback of touch data on the 3D forearm model

## Goals

### In Scope

1. Dual 3D view: contact points (red) + accumulating spike heatmap (jet colormap)
2. Session / trial_id / single_touch_id dropdown cascade for selecting a specific touch
3. Play button (single touch) and Play All button (all touches in selected trial)
4. Speed control for playback rate
5. Pipeline integration as a new DAG task

### Out of Scope

- Trail/history visualization on the contact point view (heatmap already shows history)
- 2D scatter plot (the feature-space explorer already covers this)
- Saving/exporting animations to video
- Frame scrubbing slider (can be added later)
- Per-touch feature overlays (pressure waveform, velocity graph)

## Success Criteria

- [ ] GUI launches from the DAG pipeline with session data loaded
- [ ] Session dropdown switches between loaded sessions
- [ ] Trial and touch dropdowns populate correctly from session data
- [ ] Play animates the selected touch at configurable speed with visible contact point movement
- [ ] Play All sequences through all touches in the selected trial
- [ ] Left view shows current-frame contact points as red spheres on grey forearm
- [ ] Right view shows spike heatmap accumulating from touch start (jet colormap, NaN=grey)
- [ ] Both 3D views have linked cameras (rotate one, both follow)

---

## Technical Design

### Approach

Create a new data model (`PlaybackData`) that preserves trial/touch grouping info, and a new GUI class (`TouchPlaybackExplorer`) with dual `QtInteractor` panes. Reuse existing building blocks: `load_forearm_vertices()` from `rf_data_loader.py`, `compute_tangent_plane_rotation()` from `tangent_plane_alignment.py`, `cKDTree` vertex snapping, and the bracket-regex contact-point parser — all proven in `rf_explorer_data.py`.

The existing `ExplorerData` is intentionally session-level with no grouping — extending it would require invasive changes that could regress the feature-space explorer. A separate `PlaybackData` dataclass cleanly separates concerns.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| New `PlaybackData` dataclass + dedicated loader | Clean separation, no risk to existing explorer, preserves grouping info | Some code duplication for CSV parsing | **Chosen** |
| Extend `ExplorerData` with optional grouping | Reuses existing cache, single loader | Invasive changes to dedup logic, complicates existing GUI, trial/touch info not in current cache format | Rejected |
| Load raw series CSV on-the-fly per touch selection | No new dataclass needed | Too slow (~3-5s per CSV read), bad UX when switching touches | Rejected |

### Knowledge Base Notes

- **Qt `itemChanged` Signal Recursion** (`note-qt-itemchanged-signal-recursion.md`) — Relevant for camera sync between the two PyVista views. The VTK camera `ModifiedEvent` observer can re-fire when the synced camera is updated, causing infinite recursion. Apply the re-entrancy guard pattern (`_syncing` flag).
- **RF Feature-Space Explorer GUI Components** (`note-rf-feature-space-explorer-gui-components.md`) — Direct reference for deferred VTK init pattern, PyVista actor replacement via `name=` parameter, and `showEvent()` → `_deferred_start()` architecture.
- No other notes are applicable.

### Architecture Changes

```
code/src/analysis/receptive_field_mapping/
    touch_playback_data.py              # NEW — PlaybackData dataclass + loader
    gui/
        __init__.py                     # MODIFY — add TouchPlaybackExplorer export
        touch_playback_explorer.py      # NEW — the GUI (~500-600 lines)
    rf_cluster_pipeline.py              # MODIFY — add launch_touch_playback_explorer()
    __init__.py                         # MODIFY — add launch_touch_playback_explorer export

code/scripts/analysis_workflow.py       # MODIFY — add flow + wire into available_tasks
configs/analyse_workflow_dag.yaml       # MODIFY — add explore_touch_playback task
```

### Data Model

```python
@dataclass
class PlaybackSessionData:
    forearm_vertices: np.ndarray      # (N, 3) rotated to tangent plane
    tangent_rotation: np.ndarray      # (3, 3) rotation matrix

@dataclass
class TouchEvent:
    trial_id: int
    single_touch_id: int
    gesture_type: str                         # 'tap', 'stroke_proximal', etc.
    # Per deduplicated 30Hz frame:
    frame_contact_pts: list[np.ndarray]       # list of (K_i, 3) rotated contact coords
    frame_vertex_indices: list[np.ndarray]    # list of (K_i,) nearest vertex per contact pt
    frame_spikes: np.ndarray                  # (n_frames,) bool — any spike in the 1kHz run

@dataclass
class PlaybackData:
    session_data: PlaybackSessionData
    trial_ids: list[int]                              # sorted unique, excluding 0
    touches_by_trial: dict[int, list[TouchEvent]]     # trial_id -> sorted TouchEvents
```

### GUI Layout

```
QMainWindow
+-- QToolBar
|   +-- QLabel("Session:") + QComboBox
|   +-- QLabel("Trial:") + QComboBox
|   +-- QLabel("Touch:") + QComboBox
|   +-- QPushButton("Play")
|   +-- QPushButton("Play All")
|   +-- QPushButton("Stop")
|   +-- QLabel("Speed:") + QDoubleSpinBox (0.1x - 4.0x, default 1.0)
|   +-- QLabel("Frame: 0/0")
+-- QWidget (central)
    +-- QSplitter(Qt.Horizontal, 1:1)
        +-- QtInteractor (left)  -- contact points view
        +-- QtInteractor (right) -- spike heatmap view
```

### Animation System

- **Base rate:** ~30 FPS (33ms interval), matching Kinect sample rate
- **Speed control:** Timer interval = `33ms / speed_multiplier`
- **QTimer** drives `_advance_frame()`: increments frame index, updates both 3D views
- **Play All:** queues all `TouchEvent`s for the selected trial; when one finishes, loads the next

### Camera Linking

Both `QtInteractor` instances observe each other's VTK camera `ModifiedEvent`. A `_syncing` boolean flag prevents infinite re-entrancy (same pattern as the `blockSignals` guard from the Qt `itemChanged` knowledge base note).

### 3D Rendering Per Frame

**Left view (contact points):**
- Persistent actor: forearm vertices as grey point cloud (`color=[0.3, 0.3, 0.3]`, `point_size=3`, `name="forearm"`)
- Updated actor: current frame's contact points as red spheres (`color="red"`, `point_size=8`, `render_points_as_spheres=True`, `name="contacts"`)
- PyVista's `name=` parameter replaces the actor in-place without clearing the scene

**Right view (spike heatmap):**
- Persistent actor: forearm vertices with `spike_density` scalar field (`cmap="jet"`, `clim=(0, 1)`, `nan_color=[0.3, 0.3, 0.3]`, `name="forearm"`)
- Per frame: accumulate `bincount(vertex_indices[:frame+1], weights=spikes[:frame+1])` then normalize, update scalar array, `cloud.Modified()` + `plotter.render()`
- Incremental computation: maintain running `spike_sum` and `contact_count` arrays, add current frame's contribution each step (O(K) per frame, not O(frames*K))

---

## Implementation Plan

### Phase 1: Data Model
**Goal:** `PlaybackData` dataclass and loader that reads series-augmented CSV, groups by trial/touch, deduplicates to 30Hz, and maps contact points to forearm vertices.
**Started:** 2026-05-04
**Completed:** 2026-05-04

- [x] Define `PlaybackSessionData`, `TouchEvent`, `PlaybackData` dataclasses
- [x] Implement `load_playback_data(series_csv_path, forearm_ply_path) -> PlaybackData`
  - Read CSV, filter `trial_id > 0` and `single_touch_id > 0`
  - Reuse `load_forearm_vertices()` + `compute_tangent_plane_rotation()`
  - Group by `(trial_id, single_touch_id)`, deduplicate consecutive identical `contact_points` strings within each group
  - Parse contact points via bracket regex, rotate, snap to nearest vertex via `cKDTree`
  - Apply 15mm distance filter (same as `rf_explorer_data.py`)
  - Aggregate spike per deduplicated frame: `any(spikes)` across the 1kHz run
- [x] Implement `.npz` sidecar cache (save/load, mtime freshness check)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/touch_playback_data.py` — NEW (~200 lines)

**Dependencies:** None

### Phase 2: GUI Shell
**Goal:** Working `TouchPlaybackExplorer` window with dual 3D views, toolbar, and static frame rendering (no animation yet).
**Started:** 2026-05-04
**Completed:** 2026-05-04

- [x] Implement `TouchPlaybackExplorer.__init__()` and `_build_ui()` — toolbar with combos/buttons, QSplitter with two QtInteractors
- [x] Implement toolbar cascade: session combo → repopulate trial combo → repopulate touch combo
- [x] Implement `_render_forearm()` — grey point cloud on both plotters
- [x] Implement `_render_frame(frame_idx)` — update contact points (left) and heatmap (right)
- [x] Implement deferred VTK init via `showEvent()` → `_deferred_start()`
- [x] Implement camera linking between the two plotters with re-entrancy guard
- [x] Implement `closeEvent()` — stop timer, close both plotters

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/touch_playback_explorer.py` — NEW (~500 lines)
- `code/src/analysis/receptive_field_mapping/gui/__init__.py` — add export

**Dependencies:** Phase 1

### Phase 3: Animation
**Goal:** QTimer-based playback with Play, Play All, Stop, and speed control.
**Started:** 2026-05-04
**Completed:** 2026-05-04

- [x] Implement `_play()` — start QTimer for current touch from frame 0
- [x] Implement `_advance_frame()` — increment frame, update views, handle touch end
- [x] Implement `_play_all()` — queue all touches in selected trial, play sequentially
- [x] Implement `_stop()` — stop timer, clear queue
- [x] Wire speed spinbox to timer interval: `interval_ms = int(33 / speed_value)`
- [x] Update frame label on each advance: `"Frame: X / Y"`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/touch_playback_explorer.py` — extend

**Dependencies:** Phase 2

### Phase 4: Pipeline Integration
**Goal:** Wire the new GUI into the DAG pipeline so it can be launched from the pipeline GUI.
**Started:** 2026-05-04
**Completed:** 2026-05-04

- [x] Add `launch_touch_playback_explorer()` to `rf_cluster_pipeline.py` — follow `launch_feature_space_explorer()` pattern
- [x] Add export to `receptive_field_mapping/__init__.py`
- [x] Add `explore_touch_playback_flow()` Prefect flow to `analysis_workflow.py`
- [x] Wire into `available_tasks` list in `run_batch_analysis()`
- [x] Add `explore_touch_playback` task to `analyse_workflow_dag.yaml` with `depends_on: [touch_series_transforms]`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — add launcher function
- `code/src/analysis/receptive_field_mapping/__init__.py` — add export
- `code/scripts/analysis_workflow.py` — add flow function and available_tasks entry
- `configs/analyse_workflow_dag.yaml` — add task definition

**Dependencies:** Phase 3

---

## Testing Plan

### Manual Verification
- [ ] Launch pipeline GUI, enable `explore_touch_playback` task, run — GUI opens with session data
- [ ] Switch session → trial and touch combos repopulate
- [ ] Select a specific touch → both 3D views render frame 0 (grey forearm + red contact points on left, grey forearm on right)
- [ ] Click Play → animation runs, contact points move, heatmap accumulates
- [ ] Click Stop mid-animation → animation halts at current frame
- [ ] Click Play All → all touches in the trial play sequentially
- [ ] Change speed → animation rate changes
- [ ] Rotate left view → right view follows (and vice versa)
- [ ] Test with a zero-spike touch → heatmap stays grey throughout
- [ ] Test with a tap (short touch, ~5 frames) → animation completes quickly
- [ ] Test with a long stroke → animation shows contact points traversing the forearm

### Edge Cases
- [ ] Touch with no valid contact points after filtering → skip gracefully
- [ ] Session with only one trial → combo has single entry
- [ ] Trial with only one touch → Play All equivalent to Play

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — add Touch Playback Explorer to GUI section
- [ ] Add knowledge-base note if novel patterns emerge during implementation

---

## Rollback Plan

1. All new files (`touch_playback_data.py`, `touch_playback_explorer.py`) can be deleted
2. Modifications to existing files are additive only (new exports, new function, new DAG entry) — revert by removing the additions
3. No data migrations, no breaking changes to existing code

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Camera sync jank between two QtInteractors | Medium | Low | Debounce sync callback or use `EndInteractionEvent` instead of `ModifiedEvent` |
| Memory usage with many sessions loaded | Low | Medium | Same as existing explorer — ThreadPoolExecutor with max_workers=4 limits peak memory |
| First frame of some touches has NaN contact_points | Medium | Low | Skip NaN/empty frames at start of each touch during data loading |
| Two VTK render windows double GPU memory | Low | Low | Standard for side-by-side views, same pattern as scientific visualization tools |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Data Model | ~200 lines, 1 new file | None |
| Phase 2: GUI Shell | ~500 lines, 1 new file + 1 edit | Phase 1 |
| Phase 3: Animation | ~100 lines extending Phase 2 file | Phase 2 |
| Phase 4: Integration | ~50 lines across 4 existing files | Phase 3 |

---

## References

- Related Plans: `docs/development/plans/active/rf-explorer-event-and-scalar-fix.md`
- Related Plans: `docs/development/plans/active/rf-explorer-handoff.md`
- Reference implementation: `code/src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py`
- Data loader reference: `code/src/analysis/receptive_field_mapping/rf_explorer_data.py`
- Pipeline integration reference: `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py`
