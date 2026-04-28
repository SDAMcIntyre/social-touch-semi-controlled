# Plan: Fix RF Visualization GUI Responsiveness

**Created:** 2026-04-28 09:30
**Approved:** ---
**Completed:** ---
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/rf-cluster-extraction-visualization-split`

---

## Overview

**What:** Fix three issues preventing the RF visualization GUIs (gallery viewer, camera angle picker) from appearing promptly when running `visualize_receptive_fields_clustered`.

**Why:** With `gallery_viewer: true` and manual camera angle mode, the user expects both interactive windows within seconds. Currently the gallery viewer never appears (bug), and the camera angle picker takes 20+ minutes to load due to expensive per-row CSV parsing.

**How:** (1) Fix the `continue` that skips gallery viewer launch on up-to-date visualization, (2) make thumbnail generation non-blocking, (3) cache the selectivity overlay computation in the camera angle task.

## Problem Statement

Running the visualization flow with visualization already up-to-date produces this sequence:

1. `[RF Visualization] pressure_velocity_mean/binning... Visualization up-to-date, skipping.`
2. `[RF Visualization] pressure_velocity_mean/type_stratified... Visualization up-to-date, skipping.`
3. `[Camera Picker] Manual mode -- checking 11 session(s)...` (prints all sessions)
4. DtypeWarning on `rf_camera_angle_task.py:79` --- then silence for 20+ minutes.

**Issue 1 --- Gallery viewer never launches.** In `rf_cluster_pipeline.py::run_cluster_rf_visualization()`, the up-to-date check at line 779--785 does `continue`, which skips the entire loop body including the `if gallery_viewer:` block at line 985--987. The gallery viewer loads from disk artifacts --- it should launch regardless of whether heatmaps were just rendered or already existed.

**Issue 2 --- Gallery viewer thumbnail generation blocks the UI.** `rf_cluster_gallery_viewer.py::_generate_thumbnails()` (line 556--574) iterates ALL cells synchronously on the Qt main thread: builds a 3D scene, GPU screenshot, pixmap conversion --- per cell. With 11 sessions x N clusters, this freezes the GUI for minutes.

**Issue 3 --- Camera angle picker takes 20+ minutes before GUI appears.** `rf_camera_angle_task.py::collect_session_scene_data()` calls `_compute_selectivity_overlay()` which does row-by-row Python iteration over all `blocks_rf_centered/*.csv` files for all 11 sessions:

```python
for idx in range(len(df)):                       # Pure Python loop
    pts = parse_contact_points(raw_points.iat[idx])  # String parsing per row
```

This dominates the 20-minute wait. The DtypeWarning confirms the code is in this function.

## Goals

### In Scope

1. Fix gallery viewer launch so it appears even when visualization is already up-to-date
2. Make gallery viewer thumbnail generation non-blocking (UI remains responsive during generation)
3. Cache selectivity overlay results to disk so the camera angle picker loads instantly on repeated runs

### Out of Scope

- Vectorizing `_compute_selectivity_overlay()` internals (would require changes to `parse_contact_points` in the preprocessing package)
- Lazy loading of forearm meshes in the camera angle picker
- Background threading for 3D rendering (VTK is not thread-safe)

## Success Criteria

- [ ] Gallery viewer appears within seconds when `gallery_viewer: true` and visualization is already up-to-date
- [ ] Gallery viewer UI remains interactive (can click, scroll, resize) while thumbnails generate progressively
- [ ] Camera angle picker appears within seconds on repeated runs (selectivity cache hit)
- [ ] First run of camera angle picker still works (cache miss, computes and saves)
- [ ] Cache invalidated when block CSVs are newer than cache files

---

## Technical Design

### Approach

Three targeted fixes, each independent and testable in isolation.

**Issue 1** is a one-line structural fix. **Issue 2** replaces a synchronous loop with incremental single-cell processing using chained `QTimer.singleShot` calls. **Issue 3** adds mtime-gated NPY caching around the expensive computation.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Chained QTimer for thumbnails** (chosen) | Simple, no threading, PyVista-safe | Slightly slower total time (event loop overhead) | **Chosen** --- VTK is not thread-safe |
| Background QThread for thumbnails | True parallelism | VTK rendering must happen on main thread; offscreen rendering requires separate context | Rejected --- too complex, fragile |
| Skip thumbnails entirely | Instant startup | No visual overview in sidebar | Rejected --- thumbnails are a core feature |
| Vectorize `parse_contact_points` | Fastest possible overlay | Requires changing preprocessing package (out of scope) | Rejected --- out of scope |
| **NPY cache for selectivity** (chosen) | Instant on cache hit, simple invalidation | Disk space (~100KB per session) | **Chosen** |
| Lazy-load selectivity after GUI shows | GUI appears fast | Complex state management, user sees incomplete data | Rejected --- caching is simpler |

### Architecture Changes

No new modules or classes. Changes are localized to three existing files.

**Reused patterns:**
- NPY caching pattern: `rf_data_loader.py::load_forearm_vertices()` already implements mtime-gated `.npy` caching for PLY files --- the selectivity cache follows the same pattern.
- `QTimer.singleShot(0, ...)` for deferred work: already used in `rf_cluster_gallery_viewer.py:602`.

---

## Implementation Plan

### Phase 1: Fix Gallery Viewer Launch on Up-to-Date Visualization
**Goal:** Gallery viewer launches even when visualization heatmaps are already up-to-date.
**Started:** 2026-04-28
**Completed:** 2026-04-28

**Tasks:**
- [x] 1.1 --- In `rf_cluster_pipeline.py::run_cluster_rf_visualization()`, add `if gallery_viewer:` launch call inside the up-to-date `continue` block (before the `continue` statement), so the viewer launches regardless of rendering status. Keep the existing launch at line 985--987 for the non-skipped path.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` (lines 779--787) --- add gallery viewer launch before `continue`

**Dependencies:** None

### Phase 2: Non-Blocking Thumbnail Generation
**Goal:** Gallery viewer UI remains responsive while thumbnails populate progressively.
**Started:** 2026-04-28
**Completed:** 2026-04-28

**Tasks:**
- [x] 2.1 --- Add `_thumbnail_keys` (list) and `_thumbnail_index` (int) instance attributes to `RFClusterGalleryViewer`, initialized in `_generate_thumbnails()`.
- [x] 2.2 --- Replace the synchronous `for` loop in `_generate_thumbnails()` with initialization of the iterator state, then a call to `_generate_next_thumbnail()`.
- [x] 2.3 --- Implement `_generate_next_thumbnail()`: process one cell (build scene, screenshot, store pixmap, update visible widget if present), increment index, schedule next call via `QTimer.singleShot(0, self._generate_next_thumbnail)`. When all cells are processed, rebuild the current cell's scene.
- [x] 2.4 --- Update `_repopulate_sidebar()` to set thumbnails from `cell.thumbnail` if already generated (progressive display as user navigates). (Already implemented in existing code; no change needed.)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py` --- refactor `_generate_thumbnails()` into incremental `_generate_next_thumbnail()`

**Dependencies:** None (independent of Phase 1)

### Phase 3: Cache Selectivity Overlay in Camera Angle Task
**Goal:** `collect_session_scene_data()` loads selectivity from NPY cache on repeated runs, making the camera angle picker appear in seconds.
**Started:** 2026-04-28
**Completed:** 2026-04-28

**Tasks:**
- [x] 3.1 --- Add `_selectivity_cache_path(output_dir)` helper returning `(points_path, scores_path)` tuple: `{output_dir}/selectivity_points.npy` and `{output_dir}/selectivity_scores.npy`.
- [x] 3.2 --- Add `_selectivity_cache_is_valid(output_dir)` helper: returns True if both `.npy` files exist and their mtime is newer than the newest `blocks_rf_centered/*.csv` file.
- [x] 3.3 --- Add `_load_selectivity_cache(output_dir)` helper: loads and returns `(points, scores)` from NPY.
- [x] 3.4 --- Add `_save_selectivity_cache(output_dir, points, scores)` helper: saves both arrays as NPY.
- [x] 3.5 --- Modify `collect_session_scene_data()`: for each session, check cache before calling `_compute_selectivity_overlay()`. On cache miss, compute and save. On cache hit, load from NPY.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_camera_angle_task.py` --- add cache helpers, modify `collect_session_scene_data()`

**Dependencies:** None (independent of Phases 1--2)

---

## Testing Plan

### Unit Tests
- [ ] Selectivity cache round-trip: save points/scores, load back, assert equal
- [ ] Selectivity cache invalidation: create cache, touch a CSV to be newer, assert cache is stale

### Manual Verification
- [ ] Run `visualize_receptive_fields_clustered` with vis already up-to-date and `gallery_viewer: true`:
  - Gallery viewer appears within seconds
  - Thumbnails populate progressively in sidebar while UI is responsive
  - Clicking a thumbnail during generation loads the 3D view immediately
- [ ] Close gallery viewer, camera angle picker appears within seconds (second run, cache hit)
- [ ] Delete selectivity cache files, rerun --- picker takes longer (cache miss) then saves cache
- [ ] Settings changes (colormap, metric) still work correctly in the gallery viewer

### Edge Cases
- [ ] No `blocks_rf_centered/` directory for a session --- selectivity is (None, None), no cache written, camera picker still works
- [ ] Corrupt NPY cache file --- `np.load` raises, fall through to recompute
- [ ] Gallery viewer with zero cells (empty extraction) --- no thumbnails to generate, no crash

---

## Documentation Plan

- [ ] No external documentation changes needed --- these are bug fixes and performance improvements to existing features
- [ ] Update the gallery viewer plan's edge case note: "Very large dataset (50+ cells) --- thumbnail generation doesn't freeze UI" can be marked addressed

---

## Rollback Plan

1. **Before deployment:**
   - All three fixes are independent. Any can be reverted without affecting the others.
   - No schema changes, no new artifacts that other code depends on.

2. **Rollback procedure:**
   - Revert the specific commits on the feature branch.
   - Delete any `selectivity_points.npy` / `selectivity_scores.npy` files in session output dirs (cache files, not consumed by anything else).

3. **Data considerations:**
   - No migrations. Selectivity cache files are purely additive and safe to delete.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Incremental thumbnail generation leaves visible flickering in the 3D view as scenes swap | Med | Low | Acceptable for a background task; the user is typically looking at the sidebar, not the plotter during generation |
| Selectivity cache files become stale if block CSVs are regenerated outside the pipeline | Low | Med | Cache validation compares mtime against newest CSV; stale cache triggers recompute |
| `QTimer.singleShot(0, ...)` scheduling overhead makes total thumbnail time slower | Low | Low | Overhead is ~1ms per cell; total time increase negligible vs. seconds-per-cell rendering |

---

## References

- Gallery viewer plan: `docs/development/plans/active/rf-cluster-gallery-viewer.md`
- Extraction/visualization split plan: `docs/development/plans/active/rf-cluster-extraction-visualization-split.md`
- NPY caching pattern: `code/src/analysis/receptive_field_mapping/rf_data_loader.py::load_forearm_vertices()`
- Gallery viewer: `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py`
- Camera angle task: `code/src/analysis/receptive_field_mapping/rf_camera_angle_task.py`
- Pipeline function: `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py::run_cluster_rf_visualization()` (lines 779--787, 985--987)
