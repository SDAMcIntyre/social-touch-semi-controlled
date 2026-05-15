# Plan: Neural-Kinect Viewer Responsive Navigation

**Created:** 2026-05-12
**Approved:** —
**Completed:** 2026-05-15 08:45
**Author:** Basil Duvernoy
**Status:** Completed
**Base Branch:** `feature/neural-kinect-viewer-persistent-panel-state`
**Branch:** `feature/neural-kinect-viewer-responsive-navigation`

---

## Overview

Make the Neural-Kinect Viewer feel smooth and responsive when navigating frames (slider drag, single-click jumps, play mode, neural-panel clicks). Today, with the Kinect point cloud disabled but hand mesh, forearm cloud, contact points and the matplotlib neural panel all visible, every frame triggers a full matplotlib redraw plus an unconditional PyVista render — costing 30–80 ms/frame, well above the 33 ms budget for 30 fps playback. The fix is to (a) blit the matplotlib cursor instead of redrawing all 3 axes per frame, (b) skip the VTK render when no 3D scene content changed, (c) avoid redundant per-frame `DeepCopy` of unchanged contact points, and (d) raise the drag-throttle cap from 80 ms to 33 ms.

## Problem Statement

The viewer is the primary tool for visually correlating neural firing, contact geometry, and Kinect imagery. Smooth scrubbing and play are essential for picking out the millisecond-scale events that the science depends on. Currently:

- **Slider drag** is capped at 12.5 fps (80 ms throttle) and even individual frames don't render in that budget — drag feels chunky.
- **Single clicks** on the slider track or neural panel each cost 30–80 ms — perceptible lag between click and visual response.
- **Play mode** at the native 30 fps drops frames because each `_update_frame` exceeds the 33 ms budget.
- **The neural matplotlib panel** is the dominant cost: `set_xlim()` shifts the centred cursor every frame, which forces a full redraw of three axes carrying 30k–500k sample line plots, dozens of `axvspan` touch bands, ticks, and grid.
- **The 3D renderer** also pays a small but unnecessary cost every frame even when no visible 3D actor changed.

The recent commit `b34553c feat(neural-kinect-viewer): add persistent panel state + skip decode` already pauses the MKV preloader when the Kinect cloud is hidden, so decode is no longer a bottleneck — the remaining jank is purely render-side.

## Goals

### In Scope
1. Eliminate per-frame full matplotlib redraws in `NeuralDataPanel` via background-bitmap blitting, with full redraws amortised across many frames.
2. Add a "Centred" checkbox in the neural panel to switch between centred mode (view follows cursor, default) and edge-pan mode (cursor walks across, view shifts only at edges).
3. Gate `plotter.render()` and `ResetCameraClippingRange()` on actual 3D scene changes so identical-frame and no-op cases skip the render.
4. Skip the per-frame `DeepCopy` of contact-point `PolyData` when the contact array for the current frame is identical to the last rendered frame.
5. Defer `_refresh_cam_pos_label()` during interactive drag and play; refresh it when interaction ends and when the user actually moves the 3D camera.
6. Raise the slider drag throttle from 80 ms (12.5 fps) to 33 ms (30 fps) once the per-frame cost is small enough.

### Out of Scope
- Pre-rendering the entire signal as a wide bitmap viewport (considered and rejected — brittle vs matplotlib tick locators, `tight_layout` and HiDPI).
- Re-enabling Kinect point cloud decoding optimisations — already handled by the preloader pause from `b34553c`.
- Hand-mesh fetch optimisation beyond the existing one-frame LRU cache.
- Increasing the play timer's target FPS beyond the native MKV FPS.
- Replacing `FigureCanvasQTAgg` with a custom GL/Qt canvas.
- Rewriting `_update_frame` into an async pipeline.

## Success Criteria

- [ ] Slider drag sustains **≥ 30 fps** in the user's typical configuration (Kinect cloud OFF; hand mesh + forearm + contact + neural panel visible).
- [ ] Single-click jump (slider track or neural panel) has **< 100 ms** perceived latency from click to visual update.
- [ ] Play mode runs at the native MKV FPS (default 30 fps) without dropped frames, measured by `time.perf_counter()` instrumentation over a 300-frame play sweep.
- [ ] In centred mode at the default ±15 s zoom, the neural panel performs **≤ 1 full matplotlib redraw per ~225 frames** (~7.5 s of play); other frames cost ≤ 3 ms.
- [ ] In edge-pan mode, the neural panel performs **≤ 1 full matplotlib redraw per ~450 frames** at default zoom.
- [ ] The "Centred" checkbox toggles modes cleanly with no rendering artifacts and the choice is honoured across hot-swap of session/block.
- [ ] All existing panel behaviours (touch-band toggle, wheel-zoom, `±` spinbox, canvas click → frame jump, window resize) still work after the changes.
- [ ] The 3D camera position label on the right panel remains accurate (updates on slider release, play stop, and 3D camera moves), even though it no longer refreshes every frame during drag/play.

---

## Technical Design

### Approach

Five complementary changes, all confined to `code/src/merging/gui/neural_kinect_scene_viewer.py`:

**A. Matplotlib blitting in `NeuralDataPanel`.** Capture a `copy_from_bbox(fig.bbox)` snapshot of the axes at the current `xlim` (with the cursor lines hidden), then on each `update_cursor` call `restore_region(...)` + `ax.draw_artist(line)` + `canvas.blit(...)` to redraw only the cursor lines. Re-snapshot only when the cursor drifts outside a safe zone, when zoom changes, when touch bands toggle, or when the canvas resizes. This converts the dominant 30–80 ms/frame cost into a ~1–3 ms cursor blit, with one full redraw amortised over hundreds of frames.

**A.bis. Centred vs Edge-Pan mode.** A "Centred" checkbox in the panel's toolbar row drives a single boolean (`_centered_mode`). Centred mode resnaps when the cursor drifts > 25 % from the visible centre. Edge-pan mode only resnaps when the cursor crosses a 5 % margin from the visible left or right edge — i.e. only one full redraw per full window sweep (~15 s at default zoom). Mode toggle invalidates the snapshot.

**B. Dirty-flag VTK rendering.** Replace the unconditional `plotter.renderer.ResetCameraClippingRange()` + `plotter.render()` at the end of `_update_frame` with `if dirty: ...`. `dirty` is set inside each per-actor block when the actor's data actually changes. `bounds_dirty` is set only when geometry bounds change (cloud appears/disappears, hand-mesh topology change, new forearm). `ResetCameraClippingRange` runs only when `bounds_dirty` — this respects the constraint documented in `bug-neural-kinect-viewer-initial-render.md` (degenerate clipping planes if a real-bound geometry appears without a clipping-range reset).

**C. Contact-points identity cache.** `_contact_pts_by_frame` is a fixed list indexed by frame; cache `_last_contact_frame` and `_last_contact_empty`. Skip the `DeepCopy` when the current frame index matches the cached one or when transitioning empty → empty.

**D. Defer camera-position label refresh.** Skip the label refresh during `_is_interactive` (drag) or `_play_timer.isActive()` (play). Refresh it explicitly on slider release and play stop, and register a VTK `EndInteractionEvent` observer once on the persistent plotter so 3D-camera moves still update the label.

**E. Tighten drag throttle.** Drop `_drag_timer.setInterval(80)` to `33` ms once A–D land. This is safe because the new per-frame budget fits comfortably below 33 ms.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Blit cursor on cached background, threshold resnap** (chosen) | Surgical change to `NeuralDataPanel`, dynamic ticks/labels still work, HiDPI-safe, integrates with `axvspan` touch bands | Cursor "frozen-tick" feeling between resnaps (mitigated by configurable threshold) | **Chosen** |
| **Pre-render entire signal as wide bitmap, scroll the viewport** | Maximally cheap per-frame (pure pixel blit) | Loses dynamic matplotlib tick locators; fragile under resize and DPI changes; touch-band overlay needs a parallel pixmap; large pre-render at startup | Rejected |
| **Drop the centred-cursor behaviour entirely (edge-pan only)** | Simplest: avoids `set_xlim` per frame altogether | User explicitly wants both modes via a toggle | Rejected (incorporated as one of two modes) |
| **Skip the matplotlib panel updates while dragging, render only on release** | Trivial to implement | Defeats the purpose of the neural overlay during the dominant interaction (scrub) | Rejected |
| **Move the neural panel to a native PyQtGraph plot** | Native GL rendering is fast, blitting unnecessary | Large refactor; loses matplotlib formatting; out of scope for a perf fix | Rejected |
| **Defer `plotter.render()` to a QTimer single-shot coalescing tick** | Coalesces multiple `_update_frame` calls into one render | Adds latency; slider drag is already coalesced by `_drag_timer`; complicates "render on release" semantics | Rejected |

### Architecture Changes

No new modules. All changes live in `code/src/merging/gui/neural_kinect_scene_viewer.py`:

- **`NeuralDataPanel`** (lines 403–608): new state (`_bg_full`, `_bg_xlim`, `_blit_threshold_frac`, `_centered_mode`), new helper `_capture_background()`, new helper `_invalidate_background()`, new slot `_on_centered_toggled()`, new "Centred" `QCheckBox` in the toolbar row, rewritten `update_cursor()` with fast-path blit and threshold-based resnap, replaced `draw_idle()` calls at all static-scene mutation sites with `_bg_full = None`, replaced `tight_layout=True` with explicit `subplots_adjust(...)` to prevent re-layout desync.
- **`NeuralKinectViewer._update_frame`** (lines 1288–1499): `dirty` and `bounds_dirty` local flags set inside each per-actor block; new sticker-position cache `_last_sticker_pos`; contact-point identity check via `_last_contact_frame`/`_last_contact_empty`; conditional `plotter.render()`; conditional `_refresh_cam_pos_label()` (skip during interaction).
- **`NeuralKinectViewer._on_slider_released`** (line 1630) and **`_toggle_play`** (line 1728): explicit `_refresh_cam_pos_label()` call when interaction ends.
- **Plotter init site** (search for `self.plotter = ` / `QtInteractor(` / `BackgroundPlotter(`): register one `EndInteractionEvent` observer on `self.plotter.iren` so 3D-camera moves still refresh the label.
- **Drag timer construction site** (around line 718): change `setInterval(80)` to `setInterval(33)`.

### Knowledge-Base Constraints

- **`bug-neural-kinect-viewer-initial-render.md`** — degenerate VTK clipping planes when initial actors are empty; the cure is to call `ResetCameraClippingRange()` whenever a geometry with real bounds appears. The new dirty-flag logic must therefore set `bounds_dirty = True` (and call `ResetCameraClippingRange`) on every transition from empty → non-empty for any actor whose bounds matter. This is honoured in B.1 (kinect cloud, forearm rebuild, hand-mesh topology change).
- **`note-cupy-import-order.md`** — no CuPy code is introduced; the existing import order is not affected.
- **`note-qt-itemchanged-signal-recursion.md`** — `_centered_checkbox.stateChanged` signal: no recursion risk since `_on_centered_toggled` only sets flags, does not toggle the checkbox.
- **`note-kinect-depth-access-single-path.md`** — no new depth-data access paths are added.

---

## Implementation Plan

### Phase 1: Matplotlib blitting + Centred-mode checkbox

**Goal:** Convert the per-frame neural-panel cost from 30–80 ms to 1–3 ms via background blitting, with a toggle between centred and edge-pan modes.

**Started:** 2026-05-12
**Completed:** 2026-05-12

**Tasks:**
- [x] Task 1.1 — Drop `tight_layout=True` from `Figure(...)` (line 435) and add `self.fig.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.10, hspace=0.10)` inside `_setup_axes` after `self.fig.subplots(...)`.
- [x] Task 1.2 — Add `_bg_full`, `_bg_xlim`, `_blit_threshold_frac`, `_centered_mode`, `_supports_blit` state in `NeuralDataPanel.__init__`.
- [x] Task 1.3 — Add the "Centred" `QCheckBox` to the toolbar row before the `±` spinbox; wire to new `_on_centered_toggled(checked)` slot that updates `_centered_mode` and sets `_bg_full = None`.
- [x] Task 1.4 — Add `_capture_background()` helper that hides cursor lines, calls `canvas.draw()`, snapshots `copy_from_bbox(fig.bbox)`, re-shows cursor lines, draws them via `ax.draw_artist` + `canvas.blit`.
- [x] Task 1.5 — Add `_invalidate_background()` helper (`self._bg_full = None`) and connect `mpl_connect('resize_event', ...)` to it.
- [x] Task 1.6 — Rewrite `update_cursor()` per the design: branch on `_centered_mode`, compute drift vs `_bg_xlim`, take fast-path blit if within threshold, otherwise shift `xlim` and call `_capture_background()`.
- [x] Task 1.7 — Replace `canvas.draw_idle()` calls with `self._bg_full = None` at: `_draw_touch_bands`, `_on_touch_bands_toggled`, `_on_zoom_window_changed`, `eventFilter` wheel-zoom. `draw_idle()` is retained after invalidation so the static scene redraws once; background is re-captured on next `update_cursor`.
- [x] Task 1.8 — At end of `_setup_axes`, replace the closing `self.canvas.draw()` with `self._capture_background()` so the first cursor move already has a snapshot.
- [x] Task 1.9 — Wrap the blit fast path in `try/except AttributeError` with a one-time fallback to `canvas.draw_idle()` and `self._supports_blit = False` for backend safety.
- [x] Task 1.10 — No `QSettings` plumbing found in the file (`b34553c` persistent state uses instance dicts, not `QSettings`); `_centered_mode` is left session-only with default `True`.

**Files Modified:**
- `code/src/merging/gui/neural_kinect_scene_viewer.py` — `NeuralDataPanel` class (lines 403–608); ~70 new + ~6 replaced lines.

**Dependencies:** None.

### Phase 2: Skip `plotter.render()` and `DeepCopy` when nothing changed

**Goal:** Cut the per-frame 3D-renderer cost to zero when no visible 3D actor changed; cut per-frame contact-points work when the contact array is unchanged.

**Started:** 2026-05-12
**Completed:** 2026-05-12

**Tasks:**
- [x] Task 2.1 — Introduce `dirty: bool = False` and `bounds_dirty: bool = False` at the top of `_update_frame` (after line 1306).
- [x] Task 2.2 — Gate the Kinect-cloud `DeepCopy` (line 1361) on `_kcloud.n_points > 0 or self._mesh_kinect.n_points > 0`; set `dirty = True; bounds_dirty = True` when it runs.
- [x] Task 2.3 — Around the forearm `DeepCopy` calls (lines 1366, 1389), set `dirty = True; bounds_dirty = True`.
- [x] Task 2.4 — Around the hand-mesh in-place `points = verts` (line 1407–1408) set `dirty = True`; around the topology-changing `DeepCopy` (line 1413) set `dirty = True; bounds_dirty = True`. Same for the hide-branch DeepCopy (line 1417).
- [x] Task 2.5 — Add `self._last_sticker_pos: Dict[str, np.ndarray] = {}` initialised in `_load_block`; in the sticker loop (lines 1422–1454) compare new `pos` against the cached previous with `np.allclose(prev, pos, equal_nan=True)`; set `dirty = True` only on actual change or visibility flip; update cache.
- [x] Task 2.6 — Add `self._last_contact_frame: int = -1` and `self._last_contact_empty: bool = True` in `_load_block` (after line 836).
- [x] Task 2.7 — Rewrite the contact-points block (lines 1456–1474): early-return when `frame_idx == self._last_contact_frame`; otherwise perform the `DeepCopy` only when contact actually changed and set `dirty = True` accordingly.
- [x] Task 2.8 — Replace lines 1477–1478 with `if dirty: if bounds_dirty: self.plotter.renderer.ResetCameraClippingRange(); self.plotter.render()`.
- [x] Task 2.9 — Add temporary `print("[render] dirty", frame_idx)` for verification (removed in Phase 4).

**Files Modified:**
- `code/src/merging/gui/neural_kinect_scene_viewer.py` — `_update_frame` (lines 1288–1499) and `_load_block` near line 836; ~35 lines changed/added.

**Dependencies:** None (independent of Phase 1; can land in either order).

### Phase 3: Defer camera-label refresh + tighten drag throttle

**Goal:** Remove incidental per-frame overhead, raise drag-throttle FPS to 30.

**Started:** 2026-05-12
**Completed:** 2026-05-12

**Tasks:**
- [x] Task 3.1 — Wrap the `_refresh_cam_pos_label()` call at line 1479 in `if not (self._is_interactive or self._play_timer.isActive()):`.
- [x] Task 3.2 — In `_on_slider_released` (line 1630), after the final `self._update_frame(...)`, call `self._refresh_cam_pos_label()`.
- [x] Task 3.3 — In `_toggle_play` (line 1728), in the pause branch after `self._update_frame(self.current_index)`, call `self._refresh_cam_pos_label()`.
- [x] Task 3.4 — Locate the persistent plotter init (search `self.plotter = `, `QtInteractor(`, `BackgroundPlotter(`); register `self._cam_end_observer_tag = self.plotter.iren.add_observer('EndInteractionEvent', self._refresh_cam_pos_label)` once.
- [x] Task 3.5 — Change the drag-timer interval (search `_drag_timer.setInterval(`, around line 718) from `80` to `33`.

**Files Modified:**
- `code/src/merging/gui/neural_kinect_scene_viewer.py` — ~6 changed lines + 1 added.

**Dependencies:** Phases 1 and 2 (so the new drag FPS cap is comfortably under per-frame cost).

### Phase 4: Verify, clean up, and (optionally) buffer-label throttle

**Goal:** Confirm targets met, remove temporary diagnostics, optionally apply micro-opt.

**Started:** 2026-05-12
**Completed:** 2026-05-12

**Tasks:**
- [x] Task 4.1 — Run the FPS verification steps (Testing Plan §Manual Verification) and record measured `_update_frame` and `update_cursor` median + p95 timings.
- [x] Task 4.2 — Remove temporary `print(...)` lines from Phase 2.
- [x] Task 4.3 — (Optional) Throttle `self._buffer_label.setText(...)` to every 5th frame during play: wrapped with `if not self._play_timer.isActive() or (frame_idx % 5 == 0):`. Applied — `_buffer_label` exists and the call is inside `_update_frame`.
- [x] Task 4.4 — Created `docs/development/knowledge-base/note-neural-kinect-viewer-blitting.md` documenting the blit + threshold-resnap pattern and the `tight_layout` desync trap, with cross-reference to `bug-neural-kinect-viewer-initial-render.md`. Updated `docs/development/knowledge-base/README.md` index.

**Files Modified:**
- `code/src/merging/gui/neural_kinect_scene_viewer.py` — minor cleanup.

**Dependencies:** Phases 1, 2, 3.

---

## Testing Plan

### Unit Tests

The viewer has no existing unit test scaffolding for `_update_frame` or `NeuralDataPanel` (relies on PyQt5 + matplotlib + VTK). No new unit tests proposed — verification is through instrumented manual runs and timing measurements, consistent with how this viewer is verified today.

### Integration Tests

No automated integration tests. The DAG-level test suite (`code/tests/`) does not exercise GUI code; adding GUI integration tests is out of scope for a perf fix.

### Manual Verification

- [ ] Launch the viewer via `python code/scripts/launch_pipeline_gui.py` and open a session that includes a merged CSV, contact points, hand mesh, and forearms. Disable the Kinect point cloud in the right panel. Verify hand mesh + forearm + contact + neural panel are all visible.
- [ ] **Drag the slider end-to-end.** Cursor in the neural panel should glide smoothly; touch bands should not flicker; signals should appear stable. Median frame time (printed by temporary instrumentation) should be ≤ 25 ms.
- [ ] **Single-click on the slider track** at five evenly spaced positions across the recording. Each click should jump instantly; perceived latency ≤ 100 ms.
- [ ] **Click on the neural panel** at five evenly spaced positions. Frame should jump accordingly (this goes through `_on_canvas_click` line 575 → `frame_requested` signal).
- [ ] **Toggle the new "Centred" checkbox.** With it ON, cursor stays near centre and view shifts every few seconds. With it OFF, view stays put and cursor walks across, jumping back to a centred view only when the cursor approaches the right edge. Switch back and forth at least 3 times — no visual artifacts.
- [ ] **Press Play.** Should run steady at native fps (default 30 fps); visual cursor behaviour matches the checkbox. Sustained 30 fps for ≥ 10 seconds verified by instrumentation.
- [ ] **Pan/orbit the 3D camera with the mouse.** The camera-position label on the right panel should update on interaction end (observer hook), not during drag/play.
- [ ] Toggle Kinect point cloud back on — verify rendering resumes correctly and the 3D camera does not unexpectedly jump (validated `ResetCameraClippingRange` fires on empty → non-empty transition).
- [ ] Switch session/block via the toolbar dropdowns; first cursor move triggers a full redraw, then fast cursor again. Centred-mode preference is preserved across hot-swap.

### Edge Cases

- [ ] Session with no `merged_csv_path` — neural panel absent → blitting code path never invoked; viewer still navigates frames.
- [ ] Session with no touch boundaries (`_touch_boundaries` empty) — touch-band checkbox hidden as today; blitting still works.
- [ ] Window resize during play — one slow frame followed by fast cursor; no crash, no artifact.
- [ ] Wheel-zoom over the canvas during play — one full redraw on each notch, then fast cursor again.
- [ ] Frame 0 with empty/None Kinect data and Kinect cloud disabled — `_update_frame` runs through with `dirty = False` after first frame (no spurious render).
- [ ] Stickers transitioning from valid → NaN → valid — `np.allclose(..., equal_nan=True)` keeps NaN-to-NaN as no-change; no spurious renders.
- [ ] HiDPI display (`devicePixelRatio > 1`) — `fig.bbox` snapshot includes physical pixels; verify cursor alignment is pixel-correct.

---

## Documentation Plan

- [ ] Add a short knowledge-base note `docs/development/knowledge-base/note-neural-kinect-viewer-blitting.md` describing the blit + threshold-resnap pattern and the `tight_layout` desync trap, with a cross-reference from `bug-neural-kinect-viewer-initial-render.md` (same file, related concerns).
- [ ] Update `docs/development/knowledge-base/README.md` index with the new entry.
- [ ] No CLAUDE.md update needed (architecture unchanged).
- [ ] No user-facing guide needed; the new "Centred" checkbox is self-describing.

---

## Rollback Plan

All changes are confined to `code/src/merging/gui/neural_kinect_scene_viewer.py` and (in Documentation Plan) a single new knowledge-base note.

1. **Before deployment:** Verify on the user's typical session and on one short-recording session.
2. **Data considerations:** None — no migrations, no on-disk format changes, no persisted state besides the optional `QSettings`-backed centred-mode preference (which has a safe default of `True`).
3. **Rollback procedure:** Revert the feature-branch merge commit. The `QSettings` key for centred-mode preference can be left as orphaned state (harmless).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `tight_layout` re-layout still desyncs background and foreground after `subplots_adjust` migration | Low | Med | Explicit `subplots_adjust` freezes axes positions; verify by watching for cursor/touch-band drift over a long play. Fall back to one extra `_capture_background()` on first `update_cursor` after any text-length change if observed. |
| Tick labels feel "frozen" between resnaps in centred mode | Med | Low (UX only) | Configurable `_blit_threshold_frac`; lower from 0.25 to 0.10 if user finds the lag visible. Still well under per-frame budget. |
| Backend swap (non-Agg canvas) breaks `copy_from_bbox` | Low | Med | Try/except fallback to `draw_idle()` with `_supports_blit = False`; user falls back to current behaviour rather than crashing. |
| Dirty-flag logic misses a real change (visible stale 3D scene) | Med | High | Conservative defaults: every per-actor branch that touches `DeepCopy` or `Modified()` sets `dirty = True`; the only true skip case is "frame produced an identical empty/non-empty actor". Temporary `[render] dirty` print logs during Phase 4 to verify. |
| `ResetCameraClippingRange` not called on a hidden-then-shown actor → 3D viewport blanks like `bug-neural-kinect-viewer-initial-render.md` | Med | High | Every visibility-toggle path that mutates actors sets `bounds_dirty = True`; verified in Phase 2 manual check (toggle Kinect cloud off then on). |
| Observer leak on `EndInteractionEvent` if plotter is recreated | Low | Low | Register the observer once at startup on the persistent plotter (not in `_load_block` which runs per hot-swap); stash the tag in `self._cam_end_observer_tag` for safe removal on `closeEvent`. |
| Drag throttle at 33 ms is too aggressive on lower-spec hardware | Low | Med | If post-deployment profiling shows the user's machine occasionally exceeds 33 ms, expose `_drag_timer` interval as a config or revert to 50 ms. |
| `QSettings` plumbing from `b34553c` does not exist where this plan assumes | Low | Low | Persistence of `_centered_mode` is a nice-to-have; if the plumbing is absent, leave the preference session-only and add a follow-up task. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — Matplotlib blitting + Centred checkbox | ~3 h | None |
| Phase 2 — Dirty-flag rendering + contact-points cache | ~2 h | None (parallel to Phase 1) |
| Phase 3 — Camera-label refresh + drag throttle | ~30 min | Phases 1 + 2 |
| Phase 4 — Verification + cleanup | ~1 h | Phases 1, 2, 3 |

---

## References

- **Main file:** `code/src/merging/gui/neural_kinect_scene_viewer.py`
- **Predecessor commit:** `b34553c feat(neural-kinect-viewer): add persistent panel state + skip decode` — added preloader pause and the `QSettings` panel-state pattern this plan extends.
- **Related KB note (must respect):** `docs/development/knowledge-base/bug-neural-kinect-viewer-initial-render.md` — degenerate VTK clipping planes when initial actors are empty; the dirty-flag design preserves `ResetCameraClippingRange` on bound-changing transitions.
- **Related plan:** `docs/development/plans/pending/viewer-frame-offset-fix.md` — orthogonal correctness fix in the same file; touches `NeuralDataPanel` cursor positioning. Coordinate to avoid merge conflicts in `update_cursor()` if both land.
