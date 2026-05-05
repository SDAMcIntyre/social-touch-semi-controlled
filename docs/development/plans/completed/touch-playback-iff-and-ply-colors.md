# Plan: Touch Playback — IFF Heatmap Toggle & PLY Vertex Colors

**Date:** 2026-05-05
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/touch-playback-raw-coordinates`
**Branch:** `feature/touch-playback-raw-coordinates` (continue on current branch)

---

## Overview

Add two visual enhancements to the Touch Playback Explorer: (1) a toolbar
toggle to switch the right-panel heatmap between spike density and mean IFF
(instantaneous firing frequency), and (2) render the forearm point cloud on
the left panel using its PLY vertex colors instead of flat grey.

## Problem Statement

The viewer currently only visualises binary spike events (0/1 density) on the
right panel. `Nerve_freq` (IFF in Hz) is already present in the series CSV but
is never loaded or displayed — preventing inspection of firing-rate spatial
patterns. Additionally, the left panel renders the forearm as uniform grey,
discarding the RGB colour information in the PLY file that helps orient the
viewer anatomically.

## Goals

### In Scope
1. Load `Nerve_freq` per frame into the data model alongside `Nerve_spike`
2. Add a QComboBox ("Spike density" / "IFF (Hz)") to the toolbar
3. Render mean IFF per vertex on the right panel when IFF mode is selected
4. Display PLY vertex colours on the left-panel forearm point cloud

### Out of Scope
- Blending spike/IFF heatmap onto vertex colours (gallery-viewer style composite)
- Additional neural signal columns (e.g., amplitude)
- Colour legend or settings panel for the left panel

## Success Criteria

- [ ] Right panel defaults to "Spike density" and behaves identically to current behaviour
- [ ] Switching to "IFF (Hz)" shows mean firing frequency per vertex with Hz-scaled colorbar
- [ ] Toggling mid-playback recomputes the heatmap correctly up to the current frame
- [ ] Left panel shows forearm in natural PLY colours (falls back to grey if colours unavailable)
- [ ] Old `.npz` caches auto-invalidate (schema version bump)

---

## Technical Design

### Approach

Extend the existing `TouchEvent` dataclass with a `frame_iff` array (same
shape as `frame_spikes`). Always accumulate both spike and IFF values per
frame; render whichever the active mode selects. For PLY colours, reuse the
existing `load_forearm_vertex_colors()` from `rf_data_loader.py` and pass the
RGB array to PyVista via `scalars="rgb", rgb=True`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Accumulate only the active mode's values | Slightly less work per frame | Must replay from scratch on every toggle | Rejected |
| Always accumulate both, render active | Toggle is instant (no replay for forward frames) | Tiny extra per-frame cost | **Chosen** |
| Dynamic clim (auto-scale per frame) | Always fills colour range | Colourmap jumps during animation | Rejected |
| Fixed clim per touch (max IFF from data) | Stable animation | May under-use colour range for low-IFF touches | **Chosen** |

### Architecture Changes

No new modules. Two existing files modified:

```
code/src/analysis/receptive_field_mapping/
├── touch_playback_data.py        # +frame_iff field, +vertex_colors, cache bump
└── gui/
    └── touch_playback_explorer.py  # +heatmap combo, +IFF accumulator, +PLY colours
```

Reused utility: `rf_data_loader.py::load_forearm_vertex_colors()` (lines 66–96).

---

## Implementation Plan

### Phase 1: Data Model
**Goal:** Make IFF and vertex-colour data available to the GUI
**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] Add `frame_iff: np.ndarray` (float64, n_frames) to `TouchEvent` dataclass
- [x] Add `forearm_vertex_colors: Optional[np.ndarray]` (uint8, N×3) to `PlaybackSessionData`
- [x] Extract `Nerve_freq` column in the per-group loop of `load_playback_data()`
- [x] Call `load_forearm_vertex_colors(forearm_ply_path)` during session loading
- [x] Serialize `frame_iff_data` in `_save_playback_cache()`; add to `required_keys` in `_load_playback_cache()`
- [x] Store `forearm_vertex_colors` in cache (optional key — missing = None)
- [x] Bump `_CACHE_SCHEMA_VERSION` to force old-cache regeneration

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/touch_playback_data.py` — all changes above

**Dependencies:** None

### Phase 2: GUI Enhancements
**Goal:** Expose IFF mode toggle and PLY colours in the viewer
**Started:** 2026-05-05
**Completed:** 2026-05-05

- [x] Add `_heatmap_mode: str`, `_iff_sum: np.ndarray`, `_touch_max_iff: float` state in `__init__`
- [x] Add QComboBox ("Spike density", "IFF (Hz)") to `_build_toolbar()` after Speed spinbox
- [x] Connect `currentIndexChanged` → `_on_heatmap_mode_changed()`
- [x] Implement `_on_heatmap_mode_changed()`: set mode, re-add right-panel mesh with correct clim, call `_recompute_heatmap_from_scratch()`
- [x] Implement `_recompute_heatmap_from_scratch()`: reset accumulators, replay accumulation 0..current_frame, update scalars
- [x] Extract `_update_heatmap_scalars()` helper (spike: sum/count clim 0–1; IFF: sum/count clim 0–max)
- [x] Modify `_render_frame()`: accumulate `_iff_sum` alongside `_spike_sum`, call `_update_heatmap_scalars()`
- [x] Reset `_iff_sum` in all existing reset locations (init, `_load_touch`, slider drag)
- [x] Compute `_touch_max_iff` in `_load_touch()` for stable IFF clim
- [x] In `_render_forearm()`: if vertex colours available, set `cloud["rgb"]` and render with `rgb=True`; else fall back to grey

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/touch_playback_explorer.py` — all changes above

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Launch explorer with a session that has both `Nerve_spike` and `Nerve_freq` data
- [ ] Confirm left panel shows forearm in PLY colours (skin tones, not grey)
- [ ] Play a touch in "Spike density" mode — right panel matches previous behaviour
- [ ] Switch to "IFF (Hz)" — heatmap rescales, colourbar shows Hz units
- [ ] Toggle mid-animation — heatmap recomputes correctly for current frame
- [ ] Drag slider back → accumulators reset and rebuild correctly in both modes
- [ ] Test with a session whose PLY has no colours → should fall back to grey gracefully

### Edge Cases
- [ ] Touch with zero spikes: IFF heatmap shows 0 Hz everywhere contacted
- [ ] Touch where `Nerve_freq` is 0 on all frames: heatmap range defaults to (0, 1)
- [ ] PLY colour array length mismatch with vertices → grey fallback

---

## Documentation Plan

- [ ] No external docs needed (internal viewer, no public API change)
- [ ] Update `CLAUDE.md` analysis section if architecture note needed (unlikely)

---

## Rollback Plan

Both enhancements are additive and confined to two files on an existing feature
branch. Rollback = revert the commits on the branch. No data migrations, no
config schema changes.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `Nerve_freq` missing in older CSVs | Low | Med | Fail-fast: raise if column absent (pipeline convention) |
| IFF replay slow for long touches (>5k frames) | Low | Low | Vectorised numpy accumulation (no Python loop per frame) |
| PyVista `clim` not updating on re-add | Low | Med | Remove + re-add mesh actor (proven pattern in gallery viewer) |
