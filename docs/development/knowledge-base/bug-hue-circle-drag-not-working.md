# BUG: Hue-circle drag interaction not working in Open3D ImageWidget

| Field      | Value |
|------------|-------|
| Status     | **Resolved** — split-panel layout; `SceneWidget` added as direct window child |
| Affected   | `ArmSegmentation` interactive GUI, `_make_hue_range_circle` |
| File       | `code/src/preprocessing/forearm_extraction/arm_segmentation.py` |
| Open3D     | 0.19.0 |
| Platform   | Windows / WSL2 (Linux 6.6, backend: `qtagg`) |
| Dev note   | [note-open3d-scenewidget-layout.md](note-open3d-scenewidget-layout.md) |

---

## Symptom

The circular hue-range selector (`_make_hue_range_circle`) displays correctly —
the colour wheel renders, the arc highlight and two drag handles are visible.
The companion `NumberEdit` boxes (H start / H end) work correctly and update
the circle image in real time.

**Dragging the handles does not work.** Clicking and dragging anywhere on the
circle produces no change in handle position or hue range.

---

## Background: what was already fixed

### Fix 1 — `layout.set_on_key` crash
```
AttributeError: 'open3d.cpu.pybind.visualization.gui.Vert' object
                has no attribute 'set_on_key'
```
`gui.Vert` does not expose `set_on_key`.  The offending line was removed.
The window now opens without error.

### Fix 2 — `BUTTON_DOWN` was returning `IGNORED` for non-handle clicks
The original code returned `IGNORED` whenever the click landed within the
image widget but missed both handles.  In Open3D, a widget that returns
`IGNORED` from `BUTTON_DOWN` does not "own" the subsequent drag sequence —
the parent container gets subsequent events instead.

**Fix applied:** The handler now returns `HANDLED` for **any** click within
the image widget bounds (not just handle hits), so the `ImageWidget` captures
the drag regardless of where on the circle the user started clicking.

```python
# also catch clicks anywhere on the widget so DRAG events stay routed here
if not (0 <= ix <= size_px and 0 <= iy <= size_px):
    return gui.Widget.EventCallbackResult.IGNORED
...
# (hit-test handles here, set state['dragging'] if near one)
...
return gui.Widget.EventCallbackResult.HANDLED   # ← was IGNORED before
```

### Fix 3 — Also handle `MOVE` events alongside `DRAG`
Some Open3D builds / platforms send `MouseEvent.MOVE` with a button held
instead of `MouseEvent.DRAG`.  Both are now matched:

```python
if event.type in (gui.MouseEvent.DRAG, gui.MouseEvent.MOVE) \
        and state['dragging'] is not None:
```

Despite all three fixes, drag still does not work.

---

## Diagnostic session — 2026-02-21

A series of isolated test scripts was written and run to systematically confirm
or clear each hypothesis.  All scripts are in `code/scripts/__misc/`.

### Scripts created

| Script | Purpose |
|--------|---------|
| `hue_circle_v1_probe.py` | Bare `set_on_mouse` probe on `ImageWidget` |
| `hue_circle_v2_event_capture.py` | Counts DRAG events after BUTTON_DOWN returns HANDLED |
| `hue_circle_v2b_vert_handler.py` | Tests `set_on_mouse` on `gui.Vert` and `gui.Window` |
| `hue_circle_v2c_scene_handler.py` | Tests `set_on_mouse` on `gui.SceneWidget` |
| `hue_circle_v3_coord_debug.py` | Draws crosshair at computed click position to verify coordinate mapping |
| `hue_circle_v4_hit_test.py` | Full handle hit-test debug with green highlight on hit |
| `hue_circle_v5_full_current.py` | Production implementation as standalone window (same bug) |
| `hue_circle_v6_scene_circle.py` | First SceneWidget attempt (broken — see below) |
| `hue_circle_v6b_scene_circle.py` | Second SceneWidget attempt (partial — see below) |

### Terminal output collected

**v1 — MOVE (hover) events fire on ImageWidget:**
```
[v1] #0001  MOVE          x=297  y=122
[v1] #0002  MOVE          x=297  y=123
...
```

**v2 — zero DRAG events despite BUTTON_DOWN returning HANDLED:**
```
[v2] BUTTON_DOWN  x=106  y=115  → HANDLED
[v2] BUTTON_UP    x=193  y=207  drag_events=0
[v2] BUTTON_DOWN  x=210  y=217  → HANDLED
[v2] BUTTON_UP    x=127  y=125  drag_events=0
```

**v2b — neither Vert nor Window expose set_on_mouse:**
```
[v2b] set_on_mouse on gui.Vert: FAILED — '…Vert' object has no attribute 'set_on_mouse'
[v2b] set_on_mouse_event on gui.Window: FAILED — '…Window' object has no attribute 'set_on_mouse_event'
```

**v2c — SceneWidget delivers DRAG events correctly:**
```
[v2c] BUTTON_DOWN  x=230  y=113  → HANDLED
[v2c] DRAG   x=231  y=113  (drag #1)
[v2c] DRAG   x=232  y=114  (drag #3)
...
[v2c] BUTTON_UP  x=241  y=120  drag_events=18
```

**v3 — coordinate mapping is correct:**
```
[v3] BUTTON_DOWN  win=(154,79)  widget_rel=(154.0,27.0)  image_px=(81,27)  frame=[0,52  380x200]
```
The crosshair appeared at the clicked position.  Frame was 380×200 (widget is
wider than the image, but the transform `ix = px / fw * SIZE` compensates).

### Hypothesis outcomes

| Hypothesis | Outcome |
|-----------|---------|
| **H1** — `ImageWidget` drops `DRAG` events | ✅ **Confirmed root cause.** `drag_events=0` every time in v2. |
| **H2** — DPI / coordinate mismatch | ✅ **Cleared.** v3 crosshair lands at the correct pixel. |
| **H3** — `ImageWidget` gets no mouse events at all | ✅ **Cleared.** v1 shows `MOVE` / `BUTTON_DOWN` / `BUTTON_UP` all arrive. |
| **H4** — Hit radius too small | ⏸ **Not yet testable** — blocked until DRAG delivery is fixed. |

### Confirmed root cause

`gui.ImageWidget.set_on_mouse` receives `BUTTON_DOWN`, `BUTTON_UP`, and
`MOVE` (hover) events, but **never receives `DRAG` events** (button held +
move).  Returning `HANDLED` from `BUTTON_DOWN` does not give the widget mouse
capture in Open3D 0.19.  The `DRAG` events go to whatever widget owns the
capture — likely the parent window or 3D scene — and are never forwarded back.

The only widget confirmed to receive `DRAG` events is `gui.SceneWidget`
(verified by v2c: 18 drag events on a single drag).

---

## Current approach: replace ImageWidget with SceneWidget

The `ImageWidget` is replaced by a `SceneWidget` for both display and
interaction.  The hue wheel is rendered as a vertex-coloured `TriangleMesh`
added to the scene.  Coordinate mapping is unchanged (`px / fw * SIZE`).

### v6 — broken (do not use)

`_on_layout` accessed `container.children[1]` which does not exist in
Open3D's Python bindings → `AttributeError` → window closed on first render.
`set_background(color, image)` was used for display but renders nothing.

### v6b — partial

Fixed the layout crash and switched to a vertex-coloured mesh.  The hue wheel
renders correctly.  `DRAG` events arrive.

**Remaining problem:** The `SceneWidget`'s built-in 3D camera orbit is still
active during drag, so dragging moves the camera (rotating the disc in 3D)
instead of — or in addition to — updating the hue angle.

Two suppression attempts were made, neither fully effective:
1. `scene_widget.set_view_controls(gui.SceneWidget.Controls.PICK_POINTS)`
2. Returning `HANDLED` unconditionally for all `DRAG`/`MOVE` events

---

## Resolution

All three issues were resolved.  Full details and the reusable pattern are in
**[note-open3d-scenewidget-layout.md](note-open3d-scenewidget-layout.md)**.

### Fix summary

**Camera orbit (v6b open item):** Resolved by returning
`gui.Widget.EventCallbackResult.HANDLED` unconditionally for **every** mouse
event type in `_on_mouse` (including the catch-all at the bottom of the
handler).  `set_view_controls(PICK_POINTS)` was not needed; event interception
alone is sufficient.

**Disappear-on-click (layout collapse):** The `SceneWidget` was being nested
inside a `gui.Vert` panel.  `gui.Vert` cannot size a `SceneWidget` because
`SceneWidget` reports 0×0 as its preferred size — it is designed to fill
explicitly assigned space.  On the first mouse interaction a relayout was
triggered, `gui.Vert` assigned 0 px height, and the background image was
cleared to grey.

**Fix:** Split the right-side controls into three **direct window children**
(`panel_top`, `hue_scene_widget`, `panel_bottom`) and size all three explicitly
in the window's `on_layout` callback.  `_make_hue_range_circle` was updated to
return a dict of fragments (`"top"`, `"scene"`, `"bottom"`) so the caller
controls placement.

**Background cleared on click:** `set_background` is invalidated whenever
Open3D issues an internal scene reset on a mouse event.  Fixed by calling
`_refresh()` (which re-applies `set_background`) inside every `BUTTON_DOWN`
and `BUTTON_UP` branch of `_on_mouse`.

---

## Code locations

| Symbol | Line (approx.) |
|--------|----------------|
| `_make_hue_range_circle` | ~512 |
| `_on_mouse` (the handler) | ~604 |
| `_render_hue_arc_overlay` | ~457 |
| `_render_hue_wheel` | ~408 |
| `handle_r` definition | ~538 |
| `grab_r2` definition | ~630 |
