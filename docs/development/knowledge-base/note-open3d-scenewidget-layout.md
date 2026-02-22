# Dev Note: Embedding a SceneWidget inside an Open3D panel

**Problem class:** `gui.SceneWidget` nested inside `gui.Vert` collapses on the
first mouse interaction, showing only a grey background.

| Field | Value |
|-------|-------|
| Resolved in | `arm_segmentation.py` — `_make_hue_range_circle` + `_display_pointcloud` |
| Open3D version | 0.19.0 |
| Platform | Windows / WSL2 |
| Related bug | [bug-hue-circle-drag-not-working.md](bug-hue-circle-drag-not-working.md) |
| Related plan | [hue-circle-integration-sandbox.md](../plans/active/hue-circle-integration-sandbox.md) |

---

## 1. Symptom

A hue-wheel selector (`SceneWidget` used as a 2-D canvas via `set_background`)
renders correctly on first paint but **disappears on the first mouse click**,
leaving only a grey rectangle where the wheel was.  The `NumberEdit` companion
boxes continue to work; only the visual widget is affected.

---

## 2. Investigation path

### 2.1 Why `ImageWidget` was abandoned first

The initial implementation used `gui.ImageWidget` for the hue wheel.
`ImageWidget` receives `BUTTON_DOWN`, `BUTTON_UP`, and `MOVE` (hover) events,
but **never receives `DRAG` events** in Open3D 0.19 — even when `BUTTON_DOWN`
returns `HANDLED`.

Returning `HANDLED` from `BUTTON_DOWN` is supposed to give the widget mouse
capture so subsequent drag events are routed to it.  On `ImageWidget` this does
not work; drag events are swallowed by the parent window or the 3-D scene and
never forwarded back.

Diagnostic evidence (`hue_circle_v2_event_capture.py`):
```
[v2] BUTTON_DOWN  x=106  y=115  → HANDLED
[v2] BUTTON_UP    x=193  y=207  drag_events=0   ← zero drags delivered
```

`gui.SceneWidget` does deliver `DRAG` events reliably (`hue_circle_v2c_scene_handler.py`):
```
[v2c] BUTTON_DOWN  x=230  y=113  → HANDLED
[v2c] DRAG   x=231  y=113  (drag #1)
...
[v2c] BUTTON_UP  x=241  y=120  drag_events=18  ← 18 drags on one gesture
```

**Decision:** Replace `ImageWidget` with `SceneWidget`.  Render the hue wheel
as a flat RGBA background image via `scene.set_background(color, image)`.  The
3-D camera is suppressed by returning `HANDLED` unconditionally for every mouse
event in `_on_mouse`.

### 2.2 The disappear-on-click bug in the integrated layout

After switching to `SceneWidget`, the standalone test script
(`hue_circle_v8_bg_image.py`) worked perfectly.  But when the same widget was
embedded inside `_display_pointcloud`'s right-side `gui.Vert` panel, the wheel
rendered on startup then vanished on the first click.

**Standalone (working) vs. integrated (broken) comparison:**

| Aspect | Standalone v8 | Integrated (`arm_segmentation`) |
|--------|--------------|--------------------------------|
| Widget parent | Direct child of `Window` | Nested: `Window → gui.Vert → SceneWidget` |
| Frame sizing | Explicit: `scene_widget.frame = gui.Rect(x, y, sq, sq)` set in `set_on_layout` | Implicit: relies on `gui.Vert` auto-layout |
| `set_on_layout` | Explicitly sizes the `SceneWidget` | Only sized the main 3-D scene and the right panel — **never touched the inner `SceneWidget`** |

---

## 3. Root Cause

`gui.SceneWidget` has **no intrinsic 2-D preferred size**.  Unlike `gui.Label`
or `gui.NumberEdit`, it cannot report a natural height to its `gui.Vert`
parent.  Open3D's `gui.Vert` auto-layout assigns it zero (or minimal) height.

The sequence of events:
1. Window opens → `on_layout` fires → `gui.Vert` gives `SceneWidget` 0 px
   height → `set_background` renders into a zero-height frame (the image is
   clipped but the initial OS paint happens to show something).
2. User clicks → Open3D issues a relayout → `gui.Vert` again assigns 0 px → the
   background image is cleared to the scene's default grey → widget appears
   empty.

The same zero-height collapse plagued earlier `SceneWidget` attempts (v6, v6b).

---

## 4. Architecture constraints

### 4.1 `gui.Vert` cannot size a child `SceneWidget`

`gui.Vert.calc_preferred_size` works by summing the preferred sizes of its
children.  `SceneWidget` returns 0×0 as its preferred size because it is
designed to fill whatever space is explicitly given to it by a layout callback.
There is no `set_preferred_height` or `Constraints` workaround available in
Open3D 0.19.

### 4.2 Nested `set_on_layout` callbacks are not reliably supported

Wrapping the `SceneWidget` in a custom container with its own `_on_layout`
(Strategy C from the plan) is unreliable because Open3D's Python bindings do
not guarantee that nested layout callbacks fire in the correct order or at all.

### 4.3 `set_background` is invalidated by scene resets

Every time a `SceneWidget` receives a mouse event, Open3D may issue an internal
scene reset that clears the `set_background` image.  The `_on_mouse` handler
must call `_refresh()` (which re-applies `set_background`) on every
`BUTTON_DOWN` and `BUTTON_UP` event to counteract this.

---

## 5. Fix Applied — Split Panel + Direct Window Child

### 5.1 Strategy

The right-side panel is split into **three siblings**, all direct children of
the window:

```
Window children (direct):
  scene           — 3D SceneWidget (left 4/5)
  panel_top       — gui.Vert above the hue circle (label)
  hue_scene_widget — SceneWidget for hue wheel (explicit square frame)
  panel_bottom    — gui.Vert below the hue circle (S/V sliders, buttons)
```

None of the panel widgets are nested inside each other.  The `on_layout`
callback explicitly computes and assigns the frame of every widget.

### 5.2 `_make_hue_range_circle` — returns fragments, not a container

The method returns a **dict of independent UI fragments** rather than a single
assembled container, so the caller controls placement:

```python
return {
    "top":    title_lbl,    # gui.Label  — placed in panel_top
    "scene":  scene_widget, # SceneWidget — added directly to window
    "bottom": edit_row,     # gui.Horiz  — placed in panel_bottom
}, get_hue_range
```

### 5.3 `_display_pointcloud` — layout wiring

```python
# Three sibling widgets, all direct window children:
panel_top    = gui.Vert(...)
panel_bottom = gui.Vert(...)
hue_scene_widget = None

w.add_child(panel_top)
w.add_child(panel_bottom)

# ... widget-building loop ...
# When hsv_h_range with is_hue_circle=True is encountered:
hue_elements, get_range = self._make_hue_range_circle(...)
panel_top.add_child(hue_elements["top"])          # label goes into panel_top
hue_scene_widget = hue_elements["scene"]
w.add_child(hue_scene_widget)                     # SceneWidget → direct window child
current_panel = panel_bottom
current_panel.add_child(hue_elements["bottom"])   # edits go into panel_bottom
```

### 5.4 `on_layout` — explicit frame computation

```python
def on_layout(layout_context):
    r = w.content_rect
    panel_w = max(1, r.width // 5)
    scene_w = r.width - panel_w
    px = r.get_right() - panel_w

    # 3D scene (left 4/5)
    scene.frame = gui.Rect(r.x, r.y, scene_w, r.height)

    if hue_scene_widget is not None:
        # 1. Top panel — query its natural height
        try:
            top_h = panel_top.calc_preferred_size(
                layout_context, gui.Widget.Constraints()
            ).height
        except Exception:
            top_h = int(3.5 * em)
        panel_top.frame = gui.Rect(px, r.y, panel_w, top_h)

        # 2. Hue circle — explicit square, centred in panel width
        avail_h      = max(10, r.height - top_h)
        bottom_min_h = int(10 * em)          # reserve space for sliders+buttons
        max_sq       = max(10, avail_h - bottom_min_h)
        sq           = min(panel_w, max_sq)
        x_offset     = px + (panel_w - sq) // 2
        hue_scene_widget.frame = gui.Rect(x_offset, r.y + top_h, sq, sq)

        # 3. Bottom panel — remainder of the right column
        bottom_y = r.y + top_h + sq
        panel_bottom.frame = gui.Rect(
            px, bottom_y, panel_w, max(1, r.height - bottom_y)
        )
    else:
        # No hue circle for this processing step — standard single-panel layout
        panel_top.frame    = gui.Rect(px, r.y, panel_w, r.height)
        panel_bottom.frame = gui.Rect(px, r.y, 0, 0)
```

### 5.5 `_on_mouse` — camera suppression and background refresh

```python
def _on_mouse(event):
    fr = scene_widget.frame
    px = event.x - fr.x
    py = event.y - fr.y
    fw, fh = fr.width, fr.height

    if event.type == gui.MouseEvent.BUTTON_DOWN:
        if fw > 0 and fh > 0:
            ix = px / fw * size_px
            iy = py / fh * size_px
            if 0 <= ix <= size_px and 0 <= iy <= size_px:
                # hit-test handles, set state['dragging']
                ...
        # Re-apply background: clicking triggers an internal scene reset
        # that clears set_background — repainting here counteracts it.
        _refresh()
        return gui.Widget.EventCallbackResult.HANDLED   # camera never sees this

    if event.type in (gui.MouseEvent.DRAG, gui.MouseEvent.MOVE):
        if state['dragging'] is not None and fw > 0 and fh > 0:
            ix = px / fw * size_px
            iy = py / fh * size_px
            # ... update hue angle, call _refresh() ...
        return gui.Widget.EventCallbackResult.HANDLED

    if event.type == gui.MouseEvent.BUTTON_UP:
        state['dragging'] = None
        _refresh()
        return gui.Widget.EventCallbackResult.HANDLED

    # Catch scroll, right-drag, middle-drag — camera must not respond to any.
    return gui.Widget.EventCallbackResult.HANDLED
```

---

## 6. Reusable Pattern

Use this checklist whenever you need to embed a `gui.SceneWidget` as a 2-D
canvas inside a multi-widget Open3D window.

### Checklist

- [ ] **Do NOT nest the `SceneWidget` inside a `gui.Vert` or `gui.Horiz`.**
  Add it as a **direct child of the `Window`**.

- [ ] **Always set `SceneWidget.frame` explicitly** in the window's
  `set_on_layout` callback.  Never rely on auto-layout to size it.

- [ ] **Split surrounding widgets** into separate `gui.Vert` panels (above and
  below, or left and right) that are also direct window children.  Size each
  panel in `on_layout`.

- [ ] **Use `calc_preferred_size`** on text/slider panels to get their natural
  height, then derive the `SceneWidget`'s allocated space from the remainder.

- [ ] **Return UI fragments, not a container**, from any factory method that
  builds a `SceneWidget`-based widget.  The caller must place fragments in the
  layout individually.

- [ ] **Re-apply `set_background`** in every `BUTTON_DOWN` and `BUTTON_UP`
  handler to counteract Open3D's internal scene reset on mouse events.

- [ ] **Return `HANDLED` for every mouse event** in the `SceneWidget`'s
  `set_on_mouse` handler to prevent the 3-D camera orbit from interfering.

- [ ] **Enable `enable_scene_caching(False)`** on the hue-circle `SceneWidget`
  so that `set_background` repaints are not suppressed by the scene cache.

### Template fragment

```python
# In your factory function:
scene_widget = gui.SceneWidget()
scene_widget.scene = rendering.Open3DScene(renderer)
scene_widget.enable_scene_caching(False)
scene_widget.scene.set_background([0, 0, 0, 1], initial_bg_image)
scene_widget.set_on_mouse(_on_mouse)

return {
    "above": label_widget,      # goes into panel_top  (Vert, direct window child)
    "scene": scene_widget,      # added directly to window; frame set in on_layout
    "below": controls_widget,   # goes into panel_bottom (Vert, direct window child)
}

# In _display_pointcloud / on_layout:
def on_layout(ctx):
    r = w.content_rect
    # ... compute panel geometry ...

    above_h = panel_top.calc_preferred_size(ctx, gui.Widget.Constraints()).height
    panel_top.frame = gui.Rect(px, r.y, panel_w, above_h)

    sq = min(panel_w, max(10, r.height - above_h - below_min_h))
    scene_widget.frame = gui.Rect(px, r.y + above_h, sq, sq)

    below_y = r.y + above_h + sq
    panel_bottom.frame = gui.Rect(px, below_y, panel_w, max(1, r.height - below_y))
```

---

## 7. Alternatives considered and rejected

| Strategy | Why rejected |
|----------|-------------|
| **`SceneWidget` inside `gui.Vert`** | `gui.Vert` assigns 0 px height; widget collapses on relayout. |
| **`preferred_height` on `SceneWidget`** | Not exposed in Open3D 0.19 Python bindings. |
| **Nested `set_on_layout` callbacks** | Unreliable in O3D 0.19; order of invocation not guaranteed. |
| **`gui.ImageWidget` for display** | Does not deliver `DRAG` events in O3D 0.19 — confirmed with zero-drag diagnostic. |
| **PyQt5 colour wheel popup** | Separate event loop, separate window; overkill since `SceneWidget` rendering works. |

---

## 8. References

| Document | Location |
|----------|----------|
| Bug report (drag + disappear) | [bug-hue-circle-drag-not-working.md](bug-hue-circle-drag-not-working.md) |
| Integration sandbox plan | [hue-circle-integration-sandbox.md](../plans/active/hue-circle-integration-sandbox.md) |
| HSV range slider feature plan | [hsv-range-slider-ui.md](../plans/completed/hsv-range-slider-ui.md) |
| Implementation | `code/src/preprocessing/forearm_extraction/arm_segmentation.py` |
| Key methods | `_make_hue_range_circle` (~line 512), `_display_pointcloud` `on_layout` (~line 1026) |
| Diagnostic scripts | `code/scripts/__misc/hue_circle_v1_probe.py` through `v8_bg_image.py` |
