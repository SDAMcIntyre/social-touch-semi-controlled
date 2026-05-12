# Matplotlib Blitting in NeuralDataPanel — Blit + Threshold-Resnap Pattern

**Component:** `code/src/merging/gui/neural_kinect_scene_viewer.py`
**Class:** `NeuralDataPanel`
**Added:** 2026-05-12

---

## 1. Symptom

Every frame navigation (slider drag, play, neural-panel click) triggered a full
matplotlib redraw of three axes carrying 30k–500k sample line plots, dozens of
`axvspan` touch bands, ticks, and grid lines.  Cost was 30–80 ms per frame,
well above the 33 ms budget for 30 fps playback.  The viewer felt chunky during
scrubbing and could not sustain play at the native MKV frame rate.

---

## 2. Investigation

The per-frame cost was traced to `update_cursor()`, which called `set_xlim()`
to keep the cursor centred and triggered a full axes redraw.  The axes contain
dense signal data that cannot be culled; tick locators run on every redraw.

Options evaluated:

| Approach | Decision |
|----------|----------|
| Pre-render signal as a wide static bitmap, scroll a viewport | Rejected — loses dynamic matplotlib tick locators; fragile under resize and HiDPI scaling |
| Drop centred mode entirely (cursor walks the view) | Incorporated as one of two modes; user can toggle |
| Skip neural panel updates while dragging | Rejected — defeats the purpose of visual correlation during scrub |
| PyQtGraph replacement | Rejected — large refactor, out of scope for a perf fix |
| **Blit cursor on cached background (chosen)** | Converts 30–80 ms full redraw to ~1–3 ms per frame |

---

## 3. Root Cause

`set_xlim()` invalidates the axes layout, which forces a complete re-render
of every artist in the figure.  With dense time-series data and span artists,
the re-render is proportional to the number of samples visible, not the number
of pixels changed.

---

## 4. Architecture Constraints

### `tight_layout=True` desync trap

`Figure(tight_layout=True)` (or `fig.tight_layout()`) continuously adjusts
axes margins based on tick-label text extents.  If the figure uses `tight_layout`
and a background bitmap is captured at one layout state, then the layout shifts
between full redraws (e.g. because the time axis tick labels changed width),
the restored bitmap is offset from the current axes boundaries.  This manifests
as cursor lines that drift away from their true position and touch bands that
appear at the wrong x-coordinate after a resnap.

**Fix:** Replace `tight_layout=True` with an explicit `fig.subplots_adjust(...)`
call using fixed margin values.  This freezes the axes geometry so all blits
align with the stored background.

### When to trigger a full redraw (resnap)

Too-frequent resnaps restore the blitting benefit to zero; too-infrequent resnaps
make the tick labels feel frozen.  The chosen thresholds balance label freshness
against render cost:

- **Centred mode** (view follows cursor, default): resnap when the cursor drifts
  more than 25 % of the visible half-window width from the centre.  At 30 fps
  and ±15 s zoom, this gives one full redraw every ~225 frames (~7.5 s).
- **Edge-pan mode** (cursor walks across, view holds): resnap only when the
  cursor approaches within 5 % of the visible left or right edge, i.e. one full
  redraw per full window sweep (~450 frames at default zoom).

The threshold is stored in `_blit_threshold_frac` and can be tuned at runtime.

---

## 5. Fix Applied

### Pattern: hide → draw → snapshot → show → draw_artist + blit

```python
def _capture_background(self) -> None:
    # 1. Hide dynamic artists (cursor lines).
    for ln in self._cursor_lines:
        ln.set_visible(False)
    # 2. Full draw of the static scene (axes, signal, bands, ticks).
    self.canvas.draw()
    # 3. Snapshot the full figure bbox.
    self._bg_full = self.canvas.copy_from_bbox(self.fig.bbox)
    self._bg_xlim = self.axes[0].get_xlim()
    # 4. Restore and blit cursor lines so the display isn't blank.
    for ln in self._cursor_lines:
        ln.set_visible(True)
    for ax, ln in zip(self.axes, self._cursor_lines):
        ax.draw_artist(ln)
    self.canvas.blit(self.fig.bbox)
```

On each `update_cursor()` call:
1. Compute cursor drift relative to `_bg_xlim`.
2. **Fast path (within threshold):** `restore_region(_bg_full)` + `draw_artist` +
   `canvas.blit` — no axes layout, no signal re-render.
3. **Slow path (beyond threshold or first call):** shift `xlim`, call
   `_capture_background()` to record the new static scene.

### Invalidation sites

The background snapshot must be discarded (set to `None`) whenever the static
scene changes:

- `_draw_touch_bands` — span artists added or removed
- `_on_touch_bands_toggled` — visibility flip
- `_on_zoom_window_changed` — xlim scale change
- `eventFilter` wheel-zoom — xlim scale change
- `resizeEvent` (via `mpl_connect('resize_event', ...)`) — pixel geometry changed
- Session/block hot-swap — new data loaded

After invalidating, call `canvas.draw_idle()` so the static scene redraws once
asynchronously; `_capture_background()` runs on the next `update_cursor()` call.

### Backend safety

`copy_from_bbox` and `restore_region` require the Agg renderer.  Wrap the fast
path in `try/except AttributeError` with a one-time fallback:

```python
except AttributeError:
    self._supports_blit = False
    self.canvas.draw_idle()
```

When `_supports_blit` is `False`, `update_cursor()` falls back to `draw_idle()`
(the pre-optimisation behaviour) rather than crashing.

---

## 6. Reusable Pattern

For any matplotlib canvas embedded in a PyQt5 widget that needs per-frame cursor
or marker updates over a static background:

1. Create the figure with **explicit margins** (`subplots_adjust`) — never
   `tight_layout=True`.
2. Implement `_capture_background()` using the hide → draw → `copy_from_bbox` →
   show → `draw_artist` + `blit` sequence.
3. Implement `_invalidate_background()` as `self._bg_full = None` and call
   `canvas.draw_idle()`.
4. In `update_cursor()`, check drift against a threshold; take the fast blit path
   unless the threshold is exceeded or `_bg_full is None`.
5. Connect `mpl_connect('resize_event', ...)` to `_invalidate_background()`.
6. Guard the fast path with `try/except AttributeError` and a `_supports_blit`
   flag for backend safety.

---

## 7. References

- **Main file:** `code/src/merging/gui/neural_kinect_scene_viewer.py`
  — `NeuralDataPanel` class (`_capture_background`, `_invalidate_background`,
  `update_cursor`, `_on_centered_toggled`)
- **Related bug:** [`bug-neural-kinect-viewer-initial-render.md`](bug-neural-kinect-viewer-initial-render.md)
  — degenerate VTK clipping planes in the same viewer (different subsystem).
  The dirty-flag rendering introduced alongside the blitting work preserves
  `ResetCameraClippingRange()` calls on every empty → non-empty bounds transition
  to avoid re-triggering that bug.
- **Plan:** `docs/development/plans/active/neural-kinect-viewer-responsive-navigation.md`
  — full design, alternatives considered, and phase-by-phase implementation record.
