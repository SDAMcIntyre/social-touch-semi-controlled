# Plan: Interactive Depth-Aggregation Diagnostics Viewer

**Date:** 2026-04-14
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-04-17 15:34
**Branch:** `feature/depth-weighted-xyz-aggregation` (reuse active branch) or `feature/depth-aggregation-diagnostics-viewer`

---

## Overview

Add a Tkinter-based interactive diagnostics viewer that lets Basil scrub through frames with a timeline slider and inspect every intermediate variable of the depth-weighted XYZ aggregation pipeline (mask, sampled pixels, depth distribution, guard, weighting, weighted-vs-plain medians). The viewer is launched via a new `view_xyz_depth_aggregation` task in the existing visualisation DAG, mirroring the pattern of `view_summary_stickers_on_rgb_data` / `view_ellipse_tracking_adjusted`.

## Problem Statement

The depth-weighted XYZ aggregation (`EllipseDepthExtractor`) was recently implemented on `feature/depth-weighted-xyz-aggregation`. The extracted XYZ positions are not matching expectations, and there is currently no way to inspect what the pipeline is doing per frame. The only existing debug hooks print text or block on a single matplotlib figure — neither lets Basil scrub a session to find the exact frame where the output diverges from intuition and to see which intermediate (mask, sample, guard, weight, median) caused it.

## Goals

### In Scope
1. Interactive Tkinter GUI with a `ttk.Scale` timeline slider over the session's frames, identical navigation UX to `ConsolidatedTracksReviewGUI` / `EllipseFitViewGUI` (play/pause, keyboard shortcuts, speed control).
2. Per-frame recomputation of the full `EllipseDepthExtractor` pipeline using the existing static helpers (`_build_ellipse_mask`, `_sample_depth_within_mask`, `_weighted_median`), guaranteeing the viewer's numbers match the extractor's output.
3. 6-panel embedded matplotlib diagnostic (mask overlay, sampled pixels, depth histogram, weight profile, weighted-vs-plain scatter, summary text).
4. Sticker-selection dropdown (the tracking CSV carries multiple stickers; the viewer shows one at a time).
5. New DAG task `view_xyz_depth_aggregation` in `preprocess_workflow_kinect_visualisation_dag.yaml`, wired into the visualisation pipeline script.

### Out of Scope
- Modifying `EllipseDepthExtractor` or its public API.
- Persisting diagnostic data to disk (per-frame intermediates are recomputed on demand).
- A 3D point-cloud view of the sample pixels (the existing `XYZExtractionVisualizer` already covers 3D inspection for the centroid extractor; this feature focuses on the aggregation-specific diagnostics).
- Editing the tracking / aggregation output from the viewer (read-only).
- Exporting frames or videos from the viewer.

## Success Criteria

- [ ] Enabling `view_xyz_depth_aggregation: true` in `preprocess_workflow_kinect_visualisation_dag.yaml` launches the viewer for the configured sessions.
- [ ] The slider can reach every frame in the session; scrubbing is smooth enough for exploratory debugging (sub-second response for cached frames).
- [ ] Numeric values in the "Summary" panel exactly match the corresponding row in the session's `*_xyz.csv` produced by the extractor.
- [ ] The sticker dropdown is populated from the tracking CSV's object list; switching sticker redraws all panels.
- [ ] Keyboard shortcuts match existing viewers: `<Left>`/`<Right>` (±1), `<Control-Left>`/`<Control-Right>` (±10), `<space>` (play/pause).
- [ ] Fallback / guard-fired / uniform-z frames render without crashing, showing appropriate status text.
- [ ] No change to the extractor's public API or factory signatures.

---

## Technical Design

### Approach

Create a standalone viewer mirroring `ConsolidatedTracksReviewGUI`:
- Tkinter root with `ttk.Scale` timeline, play/pause button, speed control, sticker dropdown.
- Matplotlib figure (3×2 panels) embedded via `FigureCanvasTkAgg`.
- Data sources: `VideoMP4Manager` (RGB frames + FPS), `KinectMKV` (point clouds, accessed via a small LRU cache), `ConsolidatedTracksManager` (ellipse tracking).
- Per-frame flow: slider change → load point cloud (cached) → rebuild mask + samples via the extractor's static helpers → recompute intermediates (guard, weights, medians) in a pure helper → redraw the 6 panels → `canvas.draw_idle()`.

A separate pure helper (`depth_aggregation_recomputer.py`) centralises the recomputation so the GUI file stays focused on UI and the helper can be unit-tested against the extractor.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Tkinter GUI + embedded matplotlib (slider-based) | Matches existing viewer pattern (`ConsolidatedTracksReviewGUI`, `EllipseFitViewGUI`); free scrubbing UX; integrates into visualisation DAG | Requires embedding matplotlib in Tk | **Chosen** |
| Blocking per-frame `plt.show()` in extractor `debug=True` | Minimal code; reuses existing `debug` flag on `EllipseDepthExtractor` (see `XYZExtractionVisualizer` pattern in `xyz_extractor_centroid.py`) | No scrubbing; single-direction stepping; blocks the whole pipeline; doesn't match the "pipeline visualiser" UX requested | Rejected |
| Pre-extract diagnostics to `.npz` + static plotter | Fast playback; reusable data | Large on-disk footprint; stale when extractor params change; extra DAG task for extraction | Rejected |
| PyQt + PyQtGraph viewer | Faster rendering for many pixels | Adds another GUI framework; codebase's frame-viewer convention is Tkinter; higher onboarding cost | Rejected |

### Architecture Changes

**New files:**
```
code/src/preprocessing/stickers_analysis/xyz/gui/__init__.py
code/src/preprocessing/stickers_analysis/xyz/gui/depth_aggregation_recomputer.py
code/src/preprocessing/stickers_analysis/xyz/gui/depth_aggregation_diagnostics_gui.py
code/scripts/_3_preprocessing/_1_sticker_tracking/view_xyz_depth_aggregation.py
```

**Modified files:**
- `configs/preprocess_workflow_kinect_visualisation_dag.yaml` — add the new task entry.
- `code/scripts/preprocess_workflow_kinect_visualisation.py` — add Prefect flow function and DAG wiring.
- `code/src/preprocessing/stickers_analysis/__init__.py` — re-export `DepthAggregationDiagnosticsGUI` (follows existing convention for `ConsolidatedTracksReviewGUI`).

**Integration points:**
- Imports `EllipseDepthExtractor._build_ellipse_mask`, `_sample_depth_within_mask`, `_weighted_median` (static methods, safe to reuse).
- Uses `VideoMP4Manager`, `KinectMKV` from `preprocessing.common`.
- Uses `ConsolidatedTracksFileHandler.load`, `ConsolidatedTracksManager` from `preprocessing.stickers_analysis.common`.

### Knowledge-base relevance check

- [`note-cupy-import-order.md`](../../knowledge-base/note-cupy-import-order.md) — Applies to the launch script. The script imports from `preprocessing.*`; if CuPy is anywhere in the transitive import (via `KinectMKV`), CuPy must be imported first. The sibling `view_*.py` scripts either don't trigger the issue or already handle it — follow their import order.
- [`note-somatosensory-units-and-calculations.md`](../../knowledge-base/note-somatosensory-units-and-calculations.md) — Informational: point-cloud values are in mm (Kinect SDK). The Summary panel must clearly label units.
- Other notes (`note-open3d-scenewidget-layout`, `note-qt-itemchanged-signal-recursion`, `note-forearm-icp-registration`) — Not applicable (different frameworks / features).

**Related completed plan:** [`sticker-depth-edge-gradient-bias-correction.md`](../completed/sticker-depth-edge-gradient-bias-correction.md) — the predecessor bias-correction design whose behaviour this viewer visualises.

---

## Implementation Plan

### Phase 1: Recomputation helper
**Goal:** A pure, UI-free helper that rebuilds every intermediate variable for one frame, reusing the extractor's static methods.

- [x] Task 1.1 — Create `code/src/preprocessing/stickers_analysis/xyz/gui/__init__.py`.
- [x] Task 1.2 — Create `depth_aggregation_recomputer.py` with `recompute_frame_diagnostics(point_cloud, tracked_obj_row, sticker_diameter_mm, depth_weight_sigma) -> dict` that returns: `mask`, `xs/ys/zs`, `fallback_triggered`, `z_std`, `z_min/z_max/z_range`, `z_range_clipped`, `candidate_xs/ys/zs`, `clipped_xs/ys/zs`, `z_norm`, `weights`, `uniform_weights`, `cz_min/cz_max`, `weighted_x/y/z`, `plain_x/y/z`.
- [x] Task 1.3 — Reuse `EllipseDepthExtractor._build_ellipse_mask`, `_sample_depth_within_mask`, `_weighted_median` (imported from the extractor module).

**Files Modified:**
- `code/src/preprocessing/stickers_analysis/xyz/gui/__init__.py` — new package init.
- `code/src/preprocessing/stickers_analysis/xyz/gui/depth_aggregation_recomputer.py` — new helper.

**Dependencies:** None.

### Phase 2: Diagnostics GUI
**Goal:** The Tkinter window with slider navigation and embedded 6-panel matplotlib figure.

- [x] Task 2.1 — Create `DepthAggregationDiagnosticsGUI` class with constructor accepting `video_manager`, `mkv_path`, `tracks_manager`, `sticker_diameter_mm=10.0`, `depth_weight_sigma=0.3`, `title`, `windowState='maximized'`.
- [x] Task 2.2 — UI layout: top bar (sticker dropdown), centre (`FigureCanvasTkAgg` with a 3×2 matplotlib figure, figsize ≈ 16×10), bottom bar (frame label, play/pause, speed selector, `ttk.Scale` timeline).
- [x] Task 2.3 — Playback loop using `root.after(self._playback_delay_ms, ...)` with `video_manager.fps`; speed multiplier 0.25×–8×.
- [x] Task 2.4 — Keyboard bindings: `<Left>`/`<Right>` (±1), `<Control-Left>`/`<Control-Right>` (±10), `<space>` (play/pause).
- [x] Task 2.5 — LRU cache wrapper around `KinectMKV` for point-cloud access (size ≈ 64 frames; sequential seek from nearest cached frame on cache miss).
- [x] Task 2.6 — `_update_ui_for_frame(frame_num)`: fetch `tracked_obj_row` from `tracks_manager`, fetch cached point cloud, call `recompute_frame_diagnostics(...)`, call `_render_panels(diag)` and `canvas.draw_idle()`.
- [x] Task 2.7 — Implement the 6 panel renderers (see **Panels** below).
- [x] Task 2.8 — Status-message rendering when the row is skipped (`should_process_row` returns False) or the MKV read fails.
- [x] Task 2.9 — Proper cleanup on window close: `root.after_cancel`, `plt.close(fig)`.

**Files Modified:**
- `code/src/preprocessing/stickers_analysis/xyz/gui/depth_aggregation_diagnostics_gui.py` — new GUI class.
- `code/src/preprocessing/stickers_analysis/__init__.py` — re-export `DepthAggregationDiagnosticsGUI`.

**Dependencies:** Phase 1.

#### Panels

| # | Title | Content |
|---|-------|---------|
| 1 | Mask overlay | Depth crop around ellipse; red ellipse outline (`matplotlib.patches.Ellipse`); green crosshair at tracking centroid. |
| 2 | Sampled pixels | Same crop; scatter of valid pixels coloured by z (`coolwarm`); invalid pixels as black dots; "FALLBACK" watermark when `len(zs) < 3`. |
| 3 | Depth distribution | Histogram of `zs`; red dashed threshold at `z_min + sticker_diameter_mm` if guard fires; clipped bars shaded red; `z_min`/`z_max` vertical lines. |
| 4 | Weight profile | Curve `w = exp(-z_norm/sigma)`; scatter of candidate pixels at `(z_norm_i, w_i)`; vertical lines at weighted-median and plain-median z_norm; "Uniform weights" label when all-equal z. |
| 5 | Weighted vs plain | XZ scatter of candidate pixels sized/coloured by weight; red star (weighted median) and green diamond (plain median); arrow annotated with shift in mm; clipped pixels as faded red `x`. |
| 6 | Summary text | Inputs (centre_px, axes), depth stats (z_min/max/range/std), guard (threshold, fired, clipped count), weighting (sigma, weight range), weighted vs plain medians + deltas. Monospace, axes off. |

### Phase 3: DAG integration
**Goal:** A new enabled-toggleable task in the visualisation pipeline.

- [x] Task 3.1 — Create `view_xyz_depth_aggregation.py` launch script modelled on `view_summary_stickers_on_rgb_data.py`; wire up `VideoMP4Manager`, `ConsolidatedTracksFileHandler.load`, and the new GUI.
- [x] Task 3.2 — Add the `view_xyz_depth_aggregation` task (enabled: false, no dependencies) to `configs/preprocess_workflow_kinect_visualisation_dag.yaml`.
- [x] Task 3.3 — Add `view_xyz_depth_aggregation_flow(...)` to `code/scripts/preprocess_workflow_kinect_visualisation.py`; gate with `dag_handler.can_run('view_xyz_depth_aggregation')`.

**Files Modified:**
- `code/scripts/_3_preprocessing/_1_sticker_tracking/view_xyz_depth_aggregation.py` — new launcher.
- `configs/preprocess_workflow_kinect_visualisation_dag.yaml` — new task entry.
- `code/scripts/preprocess_workflow_kinect_visualisation.py` — new flow function + wiring.

**Dependencies:** Phase 2.

---

## Testing Plan

### Unit Tests
- [ ] `recompute_frame_diagnostics` on a synthetic flat surface — weighted and plain medians coincide; `uniform_weights` True only when `z_max == z_min`.
- [ ] `recompute_frame_diagnostics` on a synthetic tilted surface — weighted z is strictly less than plain z (favours near-camera).
- [ ] `recompute_frame_diagnostics` with `z_range > sticker_diameter_mm` — `z_range_clipped` True; `clipped_*` arrays non-empty; candidate z_max ≤ z_min + diameter.
- [ ] `recompute_frame_diagnostics` with <3 valid pixels — `fallback_triggered` True; weighted/plain medians are NaN.
- [ ] Cross-check: `recompute_frame_diagnostics(..).weighted_z` equals `EllipseDepthExtractor.extract(...).coords_3d["z_mm"]` on the same row/point-cloud input.

### Integration Tests
- [ ] Launch the viewer on a known session; verify all panels render without exceptions for at least 100 distinct frames including fallback and guard-fired ones.
- [ ] Summary panel numbers match the corresponding row of the session's extractor-produced `*_xyz.csv`.

### Manual Verification
- [ ] Enable the DAG task, run the visualisation pipeline, scrub through a session.
- [ ] Switch sticker via dropdown; all panels update.
- [ ] Keyboard shortcuts behave identically to existing viewers.
- [ ] Play/pause loops at FPS; speed multiplier scales the delay.
- [ ] Close the window cleanly; no lingering matplotlib figures or timer callbacks.

### Edge Cases
- [ ] Row status is "Failed" / "Black Frame" / "Ignored" — panels replaced with a centred status message; navigation still works.
- [ ] Ellipse partially off-frame — panels crop to the in-bounds mask without error.
- [ ] Frame index beyond available MKV frames (trailing rows) — status message; no crash.
- [ ] Sticker missing for a subset of frames — dropdown item disabled or panels show "No data for this sticker at frame X".
- [ ] All-uniform z — Panel 4 flat line at 1.0 with "Uniform weights" label; Panel 5 zero shift.

---

## Documentation Plan

- [ ] Update `docs/development/plans/active/depth-weighted-xyz-aggregation.md` with a link to this viewer plan in its References section (the viewer is the diagnostic counterpart to that feature).
- [ ] Add a short section to the session-README / pipeline walkthrough describing how to enable `view_xyz_depth_aggregation` in the visualisation DAG (if such a README exists; otherwise skip).
- [ ] Inline docstrings on `DepthAggregationDiagnosticsGUI` and `recompute_frame_diagnostics` describing inputs, return structure, and the recomputation guarantee (must mirror `_aggregate_depth`).
- [ ] No CLAUDE.md update required (no new project-wide convention).

---

## Rollback Plan

1. **Before release:**
   - [ ] Single-branch revert of the three new files and the three modified files; no data migration.
2. **Data considerations:** None — no persisted data is produced.
3. **Rollback procedure:**
   - Revert the commits introducing the viewer.
   - Delete `code/src/preprocessing/stickers_analysis/xyz/gui/`.
   - Revert the DAG config and pipeline script edits.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Random-access MKV reads are too slow for smooth scrubbing | Med | Med | LRU cache of point clouds (~64 frames); fall back to sequential seek from nearest cached frame; document expected latency for cold frames. |
| Matplotlib + Tkinter event-loop conflicts (freezes on rapid slider drag) | Med | Med | Use `canvas.draw_idle()` (not `draw`); debounce slider input with `after_idle`; follow the exact pattern from `threshold_selector_tool_gui.py`. |
| Recomputation helper drifts from `_aggregate_depth` semantics | Low | High | Unit cross-check against `EllipseDepthExtractor.extract` on the same inputs; reuse the extractor's static helpers rather than re-implementing them. |
| Multiple stickers make the UI cluttered | Low | Low | Single-sticker view via dropdown; keyboard-shortcut future-work if needed. |
| CuPy / NumPy `bool8` import-order crash in the launcher | Low | Med | Mirror sibling `view_*.py` import order; reference [`note-cupy-import-order.md`](../../knowledge-base/note-cupy-import-order.md) during implementation. |

---

## Timeline

| Phase | Description | Dependencies |
|-------|-------------|--------------|
| Phase 1 | Recomputation helper + unit tests | None |
| Phase 2 | Tkinter GUI + embedded matplotlib panels | Phase 1 |
| Phase 3 | DAG/launcher wiring | Phase 2 |

---

## References

- Active plan this viewer supports: `docs/development/plans/active/depth-weighted-xyz-aggregation.md`
- Predecessor design: `docs/development/plans/completed/sticker-depth-edge-gradient-bias-correction.md`
- Viewer patterns to mirror:
  - `code/src/preprocessing/stickers_analysis/common/gui/consolidated_tracks_gui.py` (slider + playback)
  - `code/src/preprocessing/stickers_analysis/ellipse/gui/ellipse_fit_view_gui.py` (keyboard shortcuts, object toggles)
  - `code/src/preprocessing/stickers_analysis/ellipse/gui/threshold_selector_tool_gui.py` (embedded `FigureCanvasTkAgg`)
- Extractor being diagnosed: `code/src/preprocessing/stickers_analysis/xyz/core/xyz_extractor_ellipse_depth.py`
- Knowledge-base notes considered: `note-cupy-import-order.md`, `note-somatosensory-units-and-calculations.md`
