# Plan: Neural-Kinect Viewer — Persistent Panel State + Skip Decode When Kinect Hidden

**Date:** 2026-05-12
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/population-rf-grid-group-dialog`
**Branch:** `feature/neural-kinect-viewer-persistent-panel-state`

---

## Overview

Make the Neural-Kinect viewer's right-panel controls (visibility checkboxes,
per-layer point sizes, LOD stride, crop half-size) survive session/block
switches via the toolbar dropdowns, and stop the background Kinect MKV
decoder whenever the Kinect Cloud layer is hidden so frame scrubbing is no
longer dominated by `pyk4a` decode contention.

## Problem Statement

The `NeuralKinectViewer`
(`code/src/merging/gui/neural_kinect_scene_viewer.py`) is a hot-swap
PyQt5 viewer: the user picks a session and block from toolbar dropdowns
and the right panel + 3D scene rebuild without closing the window. Two
pain points motivate this change:

1. **All panel controls reset on every block switch.**
   `_build_right_panel_data` (lines 1014–1091) re-initialises
   `self._visibility` and `self._point_sizes` to defaults, and creates fresh
   checkboxes with `cb.setChecked(True)` and sliders at default point sizes.
   `_load_block` (line 870) also rewrites `self._interactive_stride = 4`
   and pushes it back into `lod_spinbox`. Toggling *Kinect Cloud* off,
   tweaking a point-size slider, or changing the LOD spinbox is lost the
   moment the user picks another block from the dropdown.

2. **Hidden Kinect cloud still triggers MKV decode every frame.**
   `_update_frame` skips the cloud crop/convert path at line 1319 when the
   visibility flag is False, but:
   - The background `FramePreloader` thread keeps decoding MKV frames
     indefinitely.
   - The look-ahead `_preloader.seek(frame_idx + 1)` at line 1478 fires on
     every slider move, telling the preloader to evict and refill the buffer
     around the new index — even when the cloud is invisible.
   - The bounds-proxy block (lines 1294–1306) calls
     `_preloader.get_frame()` even when the cloud is hidden.

   `pyk4a` decoding is heavy; this contention is the dominant cost during
   scrubbing and is what makes the slider feel sluggish even with the cloud
   unchecked.

   `FramePreloader` already exposes `pause()` / `resume()` (lines 340–346)
   with a docstring noting "Suspend MKV decoding (e.g. when the Kinect
   cloud is hidden)" — they exist but are never called.

## Goals

### In Scope
1. Persist visibility state of every right-panel checkbox (Kinect Cloud,
   Forearms, Hand Mesh, Contact Points, every sticker) across session and
   block switches in the same window session.
2. Persist per-layer point-size sliders, the LOD spinbox value, and the
   Crop ± spinbox value across session/block switches.
3. Pause `FramePreloader` when *Kinect Cloud* is unchecked; resume + reseek
   when re-enabled. Gate the remaining `_update_frame` call sites that
   touch the preloader so scrubbing with the cloud hidden becomes free.
4. Honour persisted Kinect-Cloud visibility when a new block is loaded
   (start the freshly-spawned preloader paused if the user had it hidden).

### Out of Scope
- Persistence across process restarts (no settings file written to disk).
- Pausing decoding when *other* layers are hidden (hand mesh, forearm,
  stickers, contact points). These paths are already cheap — main-thread
  guards already exist for each — and the user explicitly called out
  Kinect Cloud as the slow case.
- Any change to camera state, registration transforms, or rendering logic
  outside of visibility/state plumbing.

## Success Criteria

- [ ] Uncheck *Kinect Cloud* → switch block via toolbar → new block opens
      with Kinect Cloud still unchecked and no point cloud rendered.
- [ ] Same persistence applies to *Forearms*, *Hand Mesh*, *Contact Points*,
      and every sticker checkbox across both block and session switches.
- [ ] Non-default point-size, LOD, and Crop ± values persist across block
      switches.
- [ ] With *Kinect Cloud* unchecked, dragging the frame slider produces a
      visibly smoother experience than current `dev`; Task Manager confirms
      `python.exe`/`pyk4a` worker CPU drops while paused.
- [ ] Re-checking *Kinect Cloud* restores rendering; the first frame after
      re-enable is approximate, then the existing exact-frame polling
      upgrades it to full resolution within ~1 s.
- [ ] `pytest code/tests/` passes with no test modifications.

---

## Technical Design

### Approach

Lift the four pieces of mutable UI state (`_visibility`, `_point_sizes`,
`_interactive_stride`, and the in-memory `_crop_half_size`) out of the
per-block rebuild paths and into one-time initialisation on the
`NeuralKinectViewer` instance. The right-panel rebuild (`_build_right_panel_data`)
then consults the persisted dicts instead of overwriting them, and a
`setdefault` pass registers any newly-introduced sticker keys with sensible
defaults. The checkbox widgets read their initial state from
`self._visibility.get(key, True)` rather than hardcoding `True`.

For the decode-when-hidden problem, wire `_on_visibility_changed` to call
`FramePreloader.pause()` / `resume()` for the `'kinect_point_cloud'` key
(both methods already exist and were designed for this case but never
hooked up). On resume, call `_preloader.seek(self.current_index)` first
so the buffer refills around the current position. Gate the two remaining
preloader call sites inside `_update_frame` (bounds-proxy check, look-ahead
seek) on the same visibility flag so they no-op when the cloud is hidden.
Finally, when `_load_block` spawns a fresh preloader thread, immediately
`pause()` it if the persisted visibility says the cloud is hidden.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Persist state in instance dicts + wire `pause()`/`resume()` (this plan) | Minimal surface area; reuses existing `FramePreloader.pause()`/`resume()` methods; no new threads or classes | Touches ~5 call sites in one file | Chosen |
| Persist to a YAML file under `configs/` so state survives restarts | State carries across runs | Out of scope for this request; introduces new config file and load path; risk of stale state from old code versions | Rejected — user asked only for in-session persistence |
| Destroy and recreate the `FramePreloader` thread on toggle instead of `pause()`/`resume()` | Frees pyk4a resources fully | More fragile (extra teardown/start lifecycle); slower re-enable; throws away buffer | Rejected — `pause()`/`resume()` already designed for exactly this case |
| Skip the kinect crop on hidden but leave the background decoder running | Smallest code change | Doesn't solve the actual perf problem — decoder contention is what slows scrubbing | Rejected — leaves the dominant cost in place |
| Centralise *all* layer-skip decisions inside `FramePreloader` | Cleaner separation | The other layers don't have a worker thread; symmetry would require restructuring all four data loaders | Rejected — over-engineering for the user's stated need |

### Architecture Changes

No new modules, no new classes, no API changes. The change is internal to
`NeuralKinectViewer` and consists of:

1. State dicts (`_visibility`, `_point_sizes`, `_interactive_stride`)
   are seeded once in `__init__` instead of in `_build_right_panel_data`
   and `_load_block`.
2. `_build_right_panel_data` calls `self._visibility.setdefault(key, True)`
   rather than re-assigning the dict; checkbox initial state reads from
   `self._visibility[key]`; slider initial state reads from
   `self._point_sizes[key]` (already does so — verify and keep).
3. `_on_visibility_changed` gains a `'kinect_point_cloud'` branch that
   calls `_preloader.pause()` / `_preloader.seek + _preloader.resume()`.
4. `_update_frame` gates the bounds-proxy `_preloader.get_frame()` call
   and the look-ahead `_preloader.seek(frame_idx + 1)` call on
   `self._visibility.get('kinect_point_cloud', True)`.
5. `_load_block` pauses the freshly-started preloader if Kinect Cloud is
   currently unchecked.

---

## Implementation Plan

### Phase 1: Persistent state initialisation
**Goal:** Move state dicts out of per-block rebuild paths so they survive
session/block switches.

**Started:** 2026-05-12
**Completed:** 2026-05-12

- [x] Seed `self._visibility`, `self._point_sizes`, `self._interactive_stride`,
      and `self._is_interactive` in `NeuralKinectViewer.__init__` before
      `_build_ui()` (around line 668).
- [x] In `_load_block`, delete the per-block reassignments of
      `self._interactive_stride` and `self._is_interactive` (around line 870–871).
      Keep `current_index`, `_slider_dragging`, `_exact_frame_pending`,
      `_recording_name`.
- [x] In `_load_block`, when syncing `lod_spinbox` and `crop_spinbox` (lines
      882–883), wrap the `setValue` calls in `blockSignals(True/False)` so
      the assignment does not feed back through `_on_lod_changed` /
      `_on_crop_changed`.
- [x] In `_build_right_panel_data` (lines 1027–1040), delete the
      `self._visibility = {...}` and `self._point_sizes = {...}` reassignments.
      Replace with a `setdefault` pass over the canonical key list plus
      `list(self._stickers_xyz_dict.keys())`.
- [x] In the nested `_add_object_group` (line 1042), change
      `cb.setChecked(True)` to `cb.setChecked(self._visibility.get(key, True))`.
      Verify the slider already reads `self._point_sizes.get(key, point_size)`.

**Files Modified:**
- `code/src/merging/gui/neural_kinect_scene_viewer.py` — `__init__`,
  `_load_block`, `_build_right_panel_data`, `_add_object_group`.

**Dependencies:** None

### Phase 2: Pause decoder when Kinect Cloud is hidden
**Goal:** Stop the `FramePreloader` thread from doing pyk4a work when the
cloud is not being rendered.

**Started:** 2026-05-12
**Completed:** 2026-05-12

- [x] Extend `_on_visibility_changed` (lines 1636–1638): when
      `key == 'kinect_point_cloud'` and `self._preloader is not None`,
      call `self._preloader.seek(self.current_index)` then `.resume()` on
      check; `.pause()` on uncheck. Then call `_update_frame` as before.
- [x] Gate the bounds-proxy block in `_update_frame` (lines 1294–1306) on
      `self._visibility.get('kinect_point_cloud', True)`. The bounds proxy
      is already invisible (`opacity=0.001`) and gets removed on first
      cloud render — leaving it in place when the cloud is hidden is
      harmless.
- [x] Gate the look-ahead `self._preloader.seek(frame_idx + 1)` call
      (line 1478) on the same visibility flag.
- [x] Verify the exact-frame schedule branch (lines 1480–1482) is already
      correct: when the cloud is hidden, `_got_exact` keeps its `True`
      default from line 1318 and the branch is skipped. Add a brief
      inline note.
- [x] In `_load_block`, after `self._preloader.start()` (line 863), add
      `if not self._visibility.get('kinect_point_cloud', True):
      self._preloader.pause()`.

**Files Modified:**
- `code/src/merging/gui/neural_kinect_scene_viewer.py` —
  `_on_visibility_changed`, `_update_frame`, `_load_block`.

**Dependencies:** Phase 1 (uses the persisted `self._visibility`).

---

## Testing Plan

### Unit Tests
This change is GUI-only and touches a Qt main window class that the
project's test stub strategy explicitly avoids loading. No new unit tests
are added; the existing test suite must continue to pass unchanged.

- [ ] `pytest code/tests/` passes on the feature branch with no
      test modifications.

### Integration Tests
Not applicable — `NeuralKinectViewer` is not covered by an integration
test harness in this repository.

### Manual Verification

Setup:
- [ ] `conda activate social-touch-env && python code/scripts/launch_pipeline_gui.py`.
- [ ] Launch the merging visualisation workflow on a session with at
      least two blocks and (ideally) a second session for cross-session
      checks.

Tests:

- [ ] **Visibility persistence — block switch.** Uncheck *Kinect Cloud* →
      switch block via toolbar dropdown → checkbox stays unchecked, no
      point cloud rendered. Repeat for *Forearms*, *Hand Mesh*,
      *Contact Points*, and a sticker.
- [ ] **Visibility persistence — session switch.** Uncheck a layer →
      switch session → checkbox state carries over.
- [ ] **Slider persistence.** Move the *Kinect Cloud* point-size slider,
      the *LOD* spinbox, and the *Crop ±* spinbox to non-default values.
      Switch block → all three values persist; rendered scene reflects them.
- [ ] **Scrubbing performance with cloud hidden.** Uncheck *Kinect Cloud*,
      drag the frame slider across the full range. Slider should be
      visibly smoother than current `dev`. Task Manager confirms
      `python.exe`/`pyk4a` worker CPU drops while paused.
- [ ] **Re-enable correctness.** Re-check *Kinect Cloud* → cloud reappears
      within ~1 s. Jump to several distant frame indices; first render at
      each is approximate, then resolves to exact via the existing
      `_schedule_exact_frame` polling.

### Edge Cases
- [ ] **Block with new sticker names.** Switch to a block whose stickers
      are not in `_visibility` yet → new sticker keys appear as checked
      (default True) without affecting previously-set sticker visibility.
- [ ] **Toggle Kinect Cloud during playback.** Pause/resume of the
      preloader while the play timer is active does not crash and resumes
      cleanly.
- [ ] **Initial open with no merged CSV.** The `Contact Points` checkbox
      is not created (existing behaviour); persistence dict gets a default
      `True` entry but no UI feedback — still correct.

---

## Documentation Plan

- [ ] No README.md / CLAUDE.md changes required — this is a behavioural
      change inside one viewer, not an architectural or API change.
- [ ] No knowledge-base note required; no new framework constraint or
      cross-cutting pattern is introduced.
- [ ] No changelog entry required by current project convention.
- [ ] No new inline comments beyond a one-line note next to the gated
      look-ahead seek explaining the pause-when-hidden behaviour.

---

## Rollback Plan

The change is GUI-only and confined to one file. Rollback is
straightforward.

1. **Before merge:** abandon the branch — no shared state is mutated, no
   data files are written.
2. **After merge:** revert the merge commit on `dev`:
   ```
   git revert -m 1 <merge-commit-sha>
   ```
3. **Data considerations:** none — the change writes no files, alters no
   configs, runs no migrations.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `FramePreloader.pause()` deadlocks on a paused pyk4a decode call | Low | Med | `pause()` only clears an `Event`; the worker loop checks `_active.is_set()` between frame decodes — confirmed by reading lines 340–346 and 352–357. No mid-decode interruption. |
| Re-enable after long pause shows stale buffer | Low | Low | `_on_visibility_changed` calls `seek(self.current_index)` *before* `resume()`, evicting far frames and refilling around the current position. Existing `_schedule_exact_frame` upgrades approximate frames to exact. |
| Persisted state diverges from spinbox/slider widgets after block switch | Low | Med | Spinbox sync in `_load_block` wraps `setValue` in `blockSignals(True/False)` so persisted state stays authoritative. |
| New sticker keys in a different session break the rebuilt panel | Low | Low | `_build_right_panel_data` uses `setdefault(key, True)` so new keys default to visible without overwriting existing user choices. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — Persistent state initialisation | ~1 h | None |
| Phase 2 — Pause decoder when Kinect Cloud hidden | ~1 h | Phase 1 |
| Manual verification | ~30 min | Phase 2 |

Total: roughly half a day.

---

## References

- Approved plan draft (plan-mode scratch):
  `~/.claude/plans/analyse-neural-kinect-viewer-i-fluttering-sparrow.md`
- Source file under change:
  `code/src/merging/gui/neural_kinect_scene_viewer.py`
- Related: `FramePreloader.pause()` / `.resume()` were authored for this
  use case (see docstring at line 340–346) but were never wired into a
  visibility handler.
