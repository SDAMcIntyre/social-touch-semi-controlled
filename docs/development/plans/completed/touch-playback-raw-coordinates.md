# Plan: Touch Playback Raw Coordinates

**Date:** 2026-05-05
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-05 16:54
**Base Branch:** `dev`
**Branch:** `feature/touch-playback-raw-coordinates`

---

## Overview

Refactor the Touch Playback Explorer to load and display raw contact point coordinates — matching the Preparation Viewer's data logic exactly. Currently, the playback explorer applies tangent plane rotation, frame deduplication, distance filtering, and empty-frame skipping that distort the expected motion of single touches.

## Problem Statement

The Touch Playback Explorer (`explore_touch_playback`) shows incorrect motion for single touch events. The same touch displayed correctly in the Preparation Viewer (`explore_preparation`) appears wrong in the playback explorer due to coordinate transformations applied during data loading.

The preparation viewer shows raw XYZ sensor coordinates at 1kHz frame rate. The playback explorer applies:
1. Tangent plane rotation (rotates all coords to a tangent plane at the contact centroid)
2. Frame deduplication (collapses ~33 identical 1kHz rows into 1 unique 30Hz frame)
3. 15mm distance filter (drops contact points far from forearm mesh)
4. Empty-frame skipping (excludes frames with no parseable contact points)

These transformations served the original heatmap-focused purpose but prevent correct motion inspection.

## Goals

### In Scope
1. Remove tangent plane rotation from the data loading pipeline
2. Remove frame deduplication — keep all 1kHz rows as individual frames
3. Remove 15mm distance filter
4. Show raw forearm vertices (no rotation)
5. Keep vertex snapping for heatmap accumulation (nearest unrotated vertex)
6. Update cache format to reflect new data structure

### Out of Scope
- Changes to the Preparation Viewer (already correct)
- Changes to other RF mapping viewers (gallery, feature-space explorer)
- Removing the heatmap right-panel entirely
- Adding new UI features to the playback explorer

## Success Criteria

- [ ] Same touch in both viewers shows identical contact point positions and motion path
- [ ] Frame count matches between viewers for the same touch (both show all 1kHz frames)
- [ ] Right-panel heatmap still accumulates spike density at contacted vertices
- [ ] Old `.npz` cache files auto-invalidate and regenerate correctly

---

## Technical Design

### Approach

Match the preparation viewer's data loading logic (`preparation_viewer_data.py` lines 121-135): parse `contact_points` per row, forward-fill within each touch, keep empty frames as empty arrays, use raw Nerve_spike values per row. Add vertex snapping (cKDTree on raw forearm vertices) for the heatmap. Optimize parsing by detecting consecutive identical strings and reusing results.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Remove all transforms, match prep viewer | Simplest, correct by definition, raw mm coords | ~33x more frames per touch (1kHz vs 30Hz), larger cache | Chosen |
| Keep dedup but remove rotation + filter | Fewer frames, smaller cache, smoother playback | Still differs from prep viewer frame count; user requested exact match | Rejected |
| Add toggle for raw vs transformed mode | Flexible, preserves original behavior | Over-engineered, adds UI complexity for no clear benefit | Rejected |

### Architecture Changes

**`touch_playback_data.py`** — Major simplification:
- Remove `compute_tangent_plane_rotation` import and usage
- Remove `_DISTANCE_THRESHOLD_MM` constant
- Remove `tangent_rotation` from `PlaybackSessionData`
- Replace per-group dedup+filter logic with row-by-row parsing (matching preparation viewer)
- Keep cKDTree vertex snapping on raw (unrotated) vertices

**`touch_playback_explorer.py`** — Minor:
- Forearm rendered in original coordinates (unchanged API, just different data)
- Left/right views receive raw coordinates (no code change needed beyond data)

---

## Implementation Plan

### Phase 1: Data Loader Refactor
**Goal:** Make `load_playback_data()` produce the same per-frame contact points as `load_preparation_viewer_data()`

- [x] Remove `tangent_rotation` field from `PlaybackSessionData`
- [x] Remove tangent plane rotation computation (centroid, rotation matrix, vertex rotation)
- [x] Load raw forearm vertices (no rotation applied)
- [x] Build cKDTree on raw vertices
- [x] Replace dedup+filter per-group logic with row-by-row parsing:
  - Forward-fill `contact_points` within group
  - Parse each row's string (optimize: reuse for consecutive identical strings)
  - Keep empty frames as `np.empty((0, 3))`
  - Snap non-empty contact points to nearest vertex (no distance filter)
  - Use raw `Nerve_spike` per row (no OR-aggregation)
- [x] Update cache save/load to remove `tangent_rotation` key (triggers auto-invalidation of old caches)
- [x] Update cache reconstruction to handle 1kHz frame counts

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/touch_playback_data.py` — Main refactor

**Dependencies:** None

### Phase 2: GUI Cleanup
**Goal:** Ensure the explorer GUI works with raw coordinate data

- [x] Verify left-view contact point rendering works with raw coords (no code change expected)
- [x] Verify right-view heatmap accumulation works with unrotated vertex indices
- [x] Remove any tangent_rotation references in the GUI if present

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/touch_playback_explorer.py` — Minor cleanup if needed

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Run `explore_preparation` — select a stroke touch in block 1, note contact point motion direction and frame count
- [ ] Run `explore_touch_playback` — select the same block/trial/touch, verify contact points trace the same motion path
- [ ] Verify frame counts match between both viewers for the same touch
- [ ] Verify heatmap right panel accumulates colors at contacted vertices during playback
- [ ] Delete existing `.npz` cache files and verify regeneration works
- [ ] Run with a stale cache present — verify it auto-invalidates (mtime or missing-key check)

### Edge Cases
- [ ] Touch with no valid contact points at the start (leading NaN rows) — should show empty frames then contact
- [ ] Very short tap (< 100ms) — verify all frames present
- [ ] Session with large forearm mesh — verify vertex snapping performance is acceptable

---

## Documentation Plan

- [ ] No external docs needed — internal viewer change
- [ ] Update `code/src/analysis/CLAUDE.md` if the Touch Playback Explorer description mentions tangent plane rotation

---

## Rollback Plan

1. Revert the two modified files to their previous state
2. Delete any new `.npz` cache files generated with the new format (old format caches will regenerate)
3. No database or configuration changes to reverse

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 1kHz frame rate makes playback too slow at 1x speed | Medium | Low | Speed spinbox already supports up to 4x; user can scrub with slider. Default timer interval (33ms) means ~30 frames/sec display regardless of data rate |
| Large cache files (~33x bigger) | Medium | Low | Compressed .npz format; disk space is not a constraint for this research pipeline |
| Vertex snapping on raw coords less accurate than rotated | Low | Low | cKDTree nearest-neighbor works in any coordinate frame; accuracy depends on mesh resolution, not orientation |

---

## References

- Reference implementation: `code/src/analysis/touch_analytics/gui/preparation_viewer_data.py`
- Knowledge base: `docs/development/knowledge-base/note-3d-to-2d-surface-projection-algorithms.md`
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-metric-units.md`

---
