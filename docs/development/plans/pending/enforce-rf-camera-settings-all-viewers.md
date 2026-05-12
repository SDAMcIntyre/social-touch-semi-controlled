# Plan: Enforce RF Camera Settings Across All 3D-Dependent RF Viewers

**Date:** 2026-05-11
**Author:** Basil Duvernoy
**Status:** Draft
**Base Branch:** `dev`
**Branch:** `feature/enforce-rf-camera-settings-all-viewers`

---

## Overview

Three RF mapping GUI viewers (Touch Population Explorer, Touch Playback Explorer, Single-Touch RF Explorer) display 3D forearm geometry in raw Kinect coordinates instead of the researcher's manually-set camera-aligned orientation. This plan threads RF camera settings rotation through their data loaders, following the established pattern in `rf_explorer_data.py`, and deletes the dead `rf_visualizer.py` module.

## Problem Statement

The RF camera settings system (`rf_camera_settings.json`) provides per-session forearm orientation chosen by the researcher. All pipeline outputs (heatmap PNGs, RF metrics, gallery viewer, feature-space explorer) correctly apply this rotation. However, three interactive RF viewers bypass it entirely, showing raw Kinect coordinates with a generic `view_xy()` camera.

This was a deliberate decision in prior refactors (commits `57c2b7a` and `bbf1641`) to match the Preparation Viewer's raw coordinate display. However, these are RF mapping tools, not preparation tools, and must use the RF camera settings for consistency.

Additionally, `rf_visualizer.py` is dead code (zero imports anywhere) that renders without camera settings.

## Goals

### In Scope
1. Add RF camera rotation to `touch_population_data.py::load_population_data()`
2. Add RF camera rotation to `touch_playback_data.py::load_playback_data()`
3. Add RF camera rotation to `single_touch_rf_explorer.py::load_single_touch_rf_data()`
4. Bump cache schema versions and include `rf_camera_settings.json` mtime in freshness checks
5. Delete the dead `rf_visualizer.py` module
6. Update `note-rf-camera-settings-connections.md` to list the three new consumers

### Out of Scope
- Changes to the Preparation Viewer (`touch_analytics/gui/`) -- it is not an RF mapping tool
- Changes to the RF Camera Settings Viewer itself -- the `view_xy()` fallback there is correct (the researcher needs an initial view to orient from)
- Changes to `fit_cylinder_axis()` PCA-based axis fitting -- this is geometric computation, not a camera bypass
- Changes to the Gallery Viewer's `_set_auto_face_on_camera()` -- data IS already rotated there, this only affects the PyVista display angle

## Success Criteria

- [ ] Touch Population Explorer displays forearm in the same orientation as the RF Feature-Space Explorer and Gallery Viewer
- [ ] Touch Playback Explorer displays forearm in the same orientation as the RF Feature-Space Explorer and Gallery Viewer
- [ ] Single-Touch RF Explorer displays forearm in the same orientation as the RF Feature-Space Explorer and Gallery Viewer
- [ ] All three viewers raise `ValueError` when launched without saved camera settings
- [ ] Old `.npz` caches auto-invalidate (schema version bump)
- [ ] Re-saving camera settings invalidates caches (mtime check)
- [ ] `rf_visualizer.py` is deleted

---

## Technical Design

### Approach

Follow the established `rf_explorer_data.py` pattern (lines 337-344):

1. Load raw forearm vertices via `load_forearm_vertices()`
2. Load rotation matrix via `load_rf_camera_rotation(camera_settings_dir, session_id)` -- raises `ValueError` if settings are missing (fail-fast)
3. Rotate vertices: `vertices = (rotation @ vertices.T).T`
4. Rotate contact points: `contacts = (rotation @ contacts.T).T`
5. Build KDTree on rotated vertices, snap rotated contacts
6. Store `tangent_rotation` in the data model and cache

The KDTree snapping produces identical vertex indices whether done in raw or rotated space (rotation preserves distances). The practical difference is that the GUI renders camera-aligned geometry, and the cache stores rotated coordinates.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Pre-rotate data in loader (match `rf_explorer_data.py`) | Consistent with existing pattern; data is camera-aligned for all consumers; `view_xy()` shows correct orientation | Cache invalidation needed; breaks Preparation Viewer alignment | Chosen |
| Set PyVista camera from saved settings (keep raw data) | No cache invalidation; raw data preserved | Inconsistent with pipeline pattern; exported images still use arbitrary `view_xy()` unless camera is manually set; data-level consumers get raw coords | Rejected |
| Leave viewers as-is, only fix pipeline outputs | No code changes | Violates the constraint that all 3D RF outputs use camera settings | Rejected |

### Architecture Changes

**Data loader signature changes** (new required parameters):

| Function | New Parameters |
|----------|---------------|
| `load_population_data()` | `camera_settings_dir: Path`, `session_id: str` |
| `load_playback_data()` | `camera_settings_dir: Path`, `session_id: str` |
| `load_single_touch_rf_data()` | `camera_settings_dir: Path` |

**Dataclass field additions:**

| Dataclass | New Field |
|-----------|-----------|
| `PopulationData` | `tangent_rotation: np.ndarray  # (3, 3)` |
| `PlaybackSessionData` | `tangent_rotation: np.ndarray  # (3, 3)` |

**Cache schema version bumps:**

| File | Old Version | New Version |
|------|-------------|-------------|
| `touch_population_data.py` | 5 | 6 |
| `touch_playback_data.py` | 3 | 4 |

**File deletion:** `rf_visualizer.py` (dead code, zero imports)

---

## Implementation Plan

### Phase 1: Touch Population Data
**Goal:** Add RF camera rotation to `load_population_data()` and its cache

- [ ] Add `from .rf_extraction_io import load_rf_camera_rotation` import
- [ ] Add `camera_settings_dir: Path` and `session_id: str` parameters to `load_population_data()`
- [ ] After loading raw vertices (line 504): load rotation and apply `vertices = (rotation @ vertices.T).T`
- [ ] After concatenating contact points into `all_pts` (line 635): apply `all_pts = (rotation @ all_pts.T).T`
- [ ] Add `tangent_rotation: np.ndarray` field to `PopulationData` dataclass; update comment from `# (V, 3) raw mesh` to `# (V, 3) camera-aligned`
- [ ] Bump `_CACHE_SCHEMA_VERSION` from 5 to 6
- [ ] Save `tangent_rotation` in `_save_population_cache()`
- [ ] Load and validate `tangent_rotation` shape `(3, 3)` in `_load_population_cache()`
- [ ] Add `rf_camera_settings.json` mtime to cache freshness check
- [ ] Fix stale docstring at line 438-439 (claims rotation but code didn't apply it)
- [ ] Update `launch_touch_population_explorer()` in `rf_cluster_pipeline.py` to pass `camera_settings_dir` and `session_id`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/touch_population_data.py` -- rotation logic, cache, dataclass
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` -- launcher call site

**Dependencies:** None

### Phase 2: Touch Playback Data
**Goal:** Add RF camera rotation to `load_playback_data()` and its cache

- [ ] Add `from .rf_extraction_io import load_rf_camera_rotation` import
- [ ] Add `camera_settings_dir: Path` and `session_id: str` parameters to `load_playback_data()`
- [ ] After loading vertices (line 507): load rotation and apply
- [ ] Rotate parsed contact point arrays before KDTree snapping
- [ ] Add `tangent_rotation: np.ndarray` to `PlaybackSessionData` dataclass; update comment from `# (N, 3) raw coordinates` to `# (N, 3) camera-aligned`
- [ ] Bump `_CACHE_SCHEMA_VERSION` from 3 to 4
- [ ] Reverse cache rejection at line 258-266: old code rejected caches with `tangent_rotation`; new code should require it
- [ ] Save `tangent_rotation` in cache; load and validate shape `(3, 3)` on read
- [ ] Add `rf_camera_settings.json` mtime to cache freshness check
- [ ] Update `launch_touch_playback_explorer()` in `rf_cluster_pipeline.py` to pass `camera_settings_dir` and `session_id`
- [ ] Update `load_playback_data()` call in `rf_single_touch_pipeline.py` to pass new params

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/touch_playback_data.py` -- rotation logic, cache, dataclass
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` -- launcher call site
- `code/src/analysis/receptive_field_mapping/rf_single_touch_pipeline.py` -- pipeline consumer call site

**Dependencies:** None (independent of Phase 1)

### Phase 3: Single-Touch RF Explorer
**Goal:** Add RF camera rotation to `load_single_touch_rf_data()`

- [ ] Add `from analysis.receptive_field_mapping.rf_extraction_io import load_rf_camera_rotation` import
- [ ] Add `camera_settings_dir: Path` parameter to `load_single_touch_rf_data()` (session_id already exists)
- [ ] After loading forearm vertices: load rotation and apply to vertices
- [ ] The `.npz` RF data contains vertex indices (topology-based) -- these remain valid after rotation
- [ ] Update `launch_single_touch_rf_explorer()` in `rf_cluster_pipeline.py` to pass `camera_settings_dir`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/single_touch_rf_explorer.py` -- rotation in loader
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` -- launcher call site

**Dependencies:** None (independent of Phases 1-2)

### Phase 4: Delete Dead Code + Documentation
**Goal:** Remove `rf_visualizer.py` and update connection map

- [ ] Delete `code/src/analysis/receptive_field_mapping/rf_visualizer.py` (confirmed zero imports)
- [ ] Update `docs/development/knowledge-base/note-rf-camera-settings-connections.md` to add three new consumers to the connection table

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_visualizer.py` -- delete
- `docs/development/knowledge-base/note-rf-camera-settings-connections.md` -- add consumers

**Dependencies:** After Phases 1-3

---

## Testing Plan

### Manual Verification
- [ ] Delete existing `.npz` sidecar caches for a test session
- [ ] Launch Touch Population Explorer -- verify forearm orientation matches the Gallery Viewer / Feature-Space Explorer
- [ ] Launch Touch Playback Explorer -- verify same orientation match
- [ ] Launch Single-Touch RF Explorer -- verify same orientation match
- [ ] In each viewer, verify heatmaps/contact points render at correct positions on the forearm
- [ ] Verify that launching any viewer without saved camera settings raises a clear `ValueError` with the message to run `set_rf_camera_settings` first
- [ ] Verify that old `.npz` caches auto-invalidate (stale cache present -> regeneration)
- [ ] Re-save camera settings for a session, verify the population/playback caches regenerate on next launch

### Edge Cases
- [ ] Session with camera settings but no forearm PLY -- should raise `ValueError` from `load_forearm_vertices()`
- [ ] Session with very sparse forearm mesh (few vertices) -- rotation should not affect KDTree accuracy
- [ ] `rf_single_touch_pipeline.py` re-run after changes -- verify produced `.npz` files contain valid vertex indices (heatmaps render correctly in the explorer)

---

## Documentation Plan

- [ ] Update `docs/development/knowledge-base/note-rf-camera-settings-connections.md` with three new consumers
- [ ] No CLAUDE.md changes needed (the RF camera settings system is already documented)
- [ ] No user-facing docs needed (internal viewer behavior change)

---

## Rollback Plan

1. Revert modified files to their previous state
2. Delete any `.npz` cache files generated with new schema versions (old-format caches will auto-regenerate)
3. No database, configuration, or external state changes to reverse

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Cache invalidation forces re-computation for all sessions | Certain | Low | Schema version bumps are the established pattern; caches auto-regenerate on next launch; computation takes seconds per session |
| Touch Playback Explorer loses raw-coordinate parity with Preparation Viewer | Certain | Low | The playback explorer is an RF mapping tool; Preparation Viewer alignment was a convenience, not a requirement. If raw-coordinate parity is needed in the future, a toggle can be added |
| `rf_single_touch_pipeline.py` produces different vertex indices after rotation | Near-zero | Low | Rotation is an isometry (preserves distances exactly); KDTree nearest-neighbor results are identical. Re-run via `should_process_task` auto-triggers since playback cache dependency changes |
| Missing camera settings blocks all three viewers | Intended | None | This IS the fail-fast convention; the `ValueError` message directs user to run `set_rf_camera_settings` first |

---

## References

- Reference implementation: `code/src/analysis/receptive_field_mapping/rf_explorer_data.py` lines 337-344
- Knowledge base: `docs/development/knowledge-base/note-rf-camera-settings-connections.md`
- Prior refactors that removed rotation: commits `57c2b7a` (population) and `bbf1641` (playback)
- Completed plan for playback raw-coordinates refactor: `docs/development/plans/completed/touch-playback-raw-coordinates.md`

---
