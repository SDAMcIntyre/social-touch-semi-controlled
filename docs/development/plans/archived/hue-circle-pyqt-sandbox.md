# Plan: Hue Circle — PyQt5 Sandbox Replacement

**Date:** 2026-02-21
**Author:** Claude (AI-assisted)
**Status:** Draft
**Branch:** `feature/hue-circle-pyqt-sandbox`

---

## Overview

Replace the broken Open3D-based hue circle drag widget with a PyQt5 `QWidget`
implementation. Open3D's `ImageWidget` never delivers DRAG events (confirmed),
and `SceneWidget` conflates drag with 3D camera orbit. PyQt5 is already a
project dependency and provides native, reliable 2D mouse interaction. This plan
covers a standalone sandbox script to validate the approach before integration.

## Problem Statement

The `ArmSegmentation` GUI needs a hue-range selector: a colour wheel with two
draggable handles (start/stop) that define an angular arc. The current
implementation renders correctly but **drag interaction is completely broken**
because `open3d.visualization.gui.ImageWidget` does not forward `DRAG` events
to its `set_on_mouse` callback (see `docs/development/knowledge-base/bug-hue-circle-drag-not-working.md`).

Eight diagnostic scripts (v1–v6b) confirmed this is an Open3D framework
limitation, not a code bug. The `SceneWidget` workaround partially works but
introduces camera orbit interference that cannot be reliably suppressed.

## Goals

### In Scope
1. Build a standalone PyQt5 `HueCircleWidget` with two draggable arc handles
2. Validate drag interaction, coordinate mapping, and wrap-around hue ranges
3. Expose a callback API (`on_range_changed(h_start, h_end)`) for integration
4. Support companion `QSpinBox` / `QDoubleSpinBox` text inputs for precise entry
5. Produce a single self-contained sandbox script that can be run independently

### Out of Scope
- Full integration into `arm_segmentation.py` (separate follow-up plan)
- Replacing the entire Open3D GUI with PyQt5 (too large; only the hue widget)
- GPU-accelerated rendering (unnecessary for a 200px circle)
- Touch/multi-touch support

## Success Criteria

- [ ] Sandbox script runs standalone: `python hue_circle_pyqt_sandbox.py`
- [ ] Hue wheel renders with correct HSV colours (0° red at top, clockwise)
- [ ] Two handles (start/stop) are visible and visually distinguishable
- [ ] Dragging either handle smoothly updates its angular position
- [ ] The selected arc is visually highlighted on the wheel
- [ ] Wrap-around ranges work (e.g. start=330°, end=30° highlights through 0°)
- [ ] Companion spin boxes sync bidirectionally with handle positions
- [ ] `on_range_changed` callback fires on every drag update
- [ ] No external dependencies beyond PyQt5 + NumPy (both already in project)

---

## Technical Design

### Approach

A single `QWidget` subclass (`HueCircleWidget`) that:

1. **Paints** the hue wheel, arc highlight, and handles in `paintEvent()` using
   `QPainter` — no raster image conversion needed; QPainter can draw arcs,
   gradients, and circles natively with anti-aliasing.
2. **Handles mouse** via `mousePressEvent`, `mouseMoveEvent`, `mouseReleaseEvent`
   — these are guaranteed to work in Qt; no event-routing surprises.
3. **Emits signals** (`PyQt5.QtCore.pyqtSignal`) when the range changes, which
   the parent can connect to any update logic.

The existing NumPy raster approach (`_render_hue_wheel`) can be kept as an
alternative rendering backend, but QPainter's `QConicalGradient` produces
a smoother, resolution-independent wheel with one line of code.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|---------|
| **PyQt5 QWidget + QPainter** | Reliable mouse events; anti-aliased rendering; already a dependency; native 2D toolkit | Separate window from Open3D (minor) | **Chosen** |
| **Open3D SceneWidget** (current) | Same window as 3D view | Camera orbit hijacks drag; TriangleMesh for 2D is over-engineered; fragile across O3D versions | Rejected |
| **Matplotlib interactive figure** | Already a dependency; has event callbacks | Heavy for a tiny widget; limited styling; not a natural UI control | Rejected |
| **Tkinter Canvas** | Zero extra deps (stdlib) | Third GUI toolkit alongside PyQt5 + Open3D; inconsistent styling | Rejected |
| **Open3D ImageWidget + timer polling** | No DRAG needed | Hacky; jittery UX; fragile | Rejected |

### Architecture

```
HueCircleWidget(QWidget)
├── paintEvent(QPaintEvent)        # Renders wheel, arc, handles
│   ├── _draw_wheel(painter)       # QConicalGradient hue ring
│   ├── _draw_arc(painter)         # Highlighted arc between handles
│   └── _draw_handles(painter)     # Two draggable circles on ring
├── mousePressEvent(QMouseEvent)   # Hit-test handles, begin drag
├── mouseMoveEvent(QMouseEvent)    # Update handle angle during drag
├── mouseReleaseEvent(QMouseEvent) # End drag
├── _pos_to_hue(QPoint) → float   # Pixel coords → hue degrees
├── _hue_to_pos(float) → QPointF  # Hue degrees → pixel coords on ring
├── set_range(h_start, h_end)      # Programmatic update (from spinboxes)
└── range_changed = pyqtSignal(float, float)  # Emitted on change

HueCircleSandbox(QWidget / QDialog)
├── HueCircleWidget               # The circle
├── QDoubleSpinBox (h_start)      # Text input for start
├── QDoubleSpinBox (h_end)        # Text input for end
└── QLabel (current range)        # Debug readout
```

### Key Design Decisions

1. **QPainter rendering over raster NumPy** — `QConicalGradient` gives a
   perfect hue wheel in ~5 lines. No need to rasterize, convert to QImage,
   and blit. Falls back gracefully on any platform Qt supports.

2. **Hue convention: 0° = red at 12 o'clock, increasing clockwise** — matches
   the existing `_render_hue_wheel` convention. Internally QPainter angles run
   counter-clockwise from 3 o'clock, so a transform is applied.

3. **Handle hit-test with generous radius** — same 2.5× handle radius grab zone
   as the current code. When handles overlap (start ≈ end), the most recently
   dragged handle wins.

4. **Bidirectional sync with spinboxes** — a `_syncing` guard prevents recursive
   signal loops, same pattern as the existing `syncing[0]` flag.

---

## Implementation Plan

### Phase 1: Core Widget
**Goal:** Rendering + drag interaction working in isolation

**Tasks:**
- [ ] Task 1.1 — Create `HueCircleWidget(QWidget)` with `paintEvent` rendering
      the hue ring using `QConicalGradient` clipped to an annulus
- [ ] Task 1.2 — Implement `_pos_to_hue(QPoint)` and `_hue_to_pos(float)`
      coordinate transforms (atan2-based, matching existing convention)
- [ ] Task 1.3 — Implement `mousePressEvent` with handle hit-testing
- [ ] Task 1.4 — Implement `mouseMoveEvent` for drag (update angle, repaint)
- [ ] Task 1.5 — Implement `mouseReleaseEvent` to end drag
- [ ] Task 1.6 — Draw arc highlight between start/stop handles (support wrap-around)
- [ ] Task 1.7 — Draw two visually distinct handles (white vs grey, or labelled)

**Files Created:**
- `code/scripts/__misc/hue_circle_pyqt_sandbox.py` — standalone sandbox

**Dependencies:** None

### Phase 2: Text Input + Signal API
**Goal:** Companion spinboxes and a clean callback API for integration

**Tasks:**
- [ ] Task 2.1 — Add `range_changed = pyqtSignal(float, float)` emission on drag
- [ ] Task 2.2 — Add `set_range(h_start, h_end)` method for programmatic updates
- [ ] Task 2.3 — Wire two `QDoubleSpinBox` widgets (0–360, 0.1 step) with
      bidirectional sync (guarded against signal loops)
- [ ] Task 2.4 — Layout: circle + spinboxes + debug label in a `QVBoxLayout`

**Files Modified:**
- `code/scripts/__misc/hue_circle_pyqt_sandbox.py` — extend with UI wrapper

**Dependencies:** Phase 1

### Phase 3: Validation + Edge Cases
**Goal:** Confirm all success criteria; stress-test edge cases

**Tasks:**
- [ ] Task 3.1 — Test wrap-around range: start=330°, end=30° (arc through 0°)
- [ ] Task 3.2 — Test full-circle range: start=0°, end=359.9° (nearly full arc)
- [ ] Task 3.3 — Test degenerate range: start=end (zero-width arc)
- [ ] Task 3.4 — Test rapid drag (no stutter, no coordinate jumps)
- [ ] Task 3.5 — Test spinbox → handle sync and handle → spinbox sync
- [ ] Task 3.6 — Verify visual match: hue colours match OpenCV HSV convention
      (H in [0,180] maps to [0,360] degrees)

**Files Modified:**
- `code/scripts/__misc/hue_circle_pyqt_sandbox.py` — bug fixes from testing

**Dependencies:** Phase 2

---

## Testing Plan

### Manual Verification
- [ ] Run `python hue_circle_pyqt_sandbox.py` — window opens, wheel renders
- [ ] Click and drag start handle — angle updates smoothly, arc follows
- [ ] Click and drag end handle — same behaviour
- [ ] Drag across 0°/360° boundary — no jump or glitch
- [ ] Type values in spinboxes — handles move to match
- [ ] Drag handles — spinbox values update to match
- [ ] Console prints range on every change (debug callback)

### Edge Cases
- [ ] Click dead centre of circle (inside ring) — no crash, no drag
- [ ] Click outside the widget — no response
- [ ] Drag off widget edge while holding button — handle tracks to nearest
      valid angle (clamp to ring), no crash on release outside
- [ ] Resize window — widget scales or stays fixed size (document which)

---

## Documentation Plan

- [ ] Inline docstrings on `HueCircleWidget` public API
- [ ] Update `docs/development/knowledge-base/bug-hue-circle-drag-not-working.md` with resolution status
- [ ] No README/CLAUDE.md changes needed (sandbox only; integration is a follow-up)

---

## Rollback Plan

This is a sandbox — no production code is modified.

1. Delete `code/scripts/__misc/hue_circle_pyqt_sandbox.py`
2. Delete branch `feature/hue-circle-pyqt-sandbox`
3. Production code in `arm_segmentation.py` is untouched

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| QConicalGradient colour banding on small widget | Low | Low | Fall back to NumPy raster rendering if needed |
| PyQt5 event loop conflicts with Open3D at integration time | Medium | Medium | Sandbox is standalone; integration plan will address event loop bridging |
| Handle overlap when start ≈ end makes one unreachable | Low | Low | Most-recently-dragged handle takes priority; or switch on which side of midpoint click lands |
| HiDPI scaling mismatch | Low | Medium | Use `QWidget.devicePixelRatio()` in coordinate transforms |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Core Widget | Small | None |
| Phase 2: Text Input + Signals | Small | Phase 1 |
| Phase 3: Validation | Small | Phase 2 |

---

## References

- Bug report: `docs/development/knowledge-base/bug-hue-circle-drag-not-working.md`
- Diagnostic scripts: `code/scripts/__misc/hue_circle_v1–v6b*.py`
- Current implementation: `code/src/preprocessing/forearm_extraction/arm_segmentation.py:408–682`
- Qt docs: `QConicalGradient`, `QWidget` mouse events
