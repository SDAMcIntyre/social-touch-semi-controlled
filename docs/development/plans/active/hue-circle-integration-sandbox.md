# Plan: Hue Circle Integration Sandbox

**Date:** 2026-02-21
**Author:** Claude (AI-assisted)
**Status:** Draft
**Branch:** `feature/hue-circle-integration-sandbox`

---

## Overview

Build a standalone sandbox that reproduces the exact layout of
`ArmSegmentation._display_pointcloud` — a 3D `SceneWidget` on the left (4/5)
and a controls `Vert` panel on the right (1/5) — with the v8 hue circle
embedded in the right panel.  This isolates and fixes the integration bug where
the hue circle disappears on first click, leaving only a grey background.

## Problem Statement

The v8 hue-circle approach (`SceneWidget` + `set_background` with an RGBA
image) works perfectly as a standalone script (`hue_circle_v8_bg_image.py`):
drag is responsive, the wheel stays visible, handles move correctly.

However, when the same code was integrated into `arm_segmentation.py`
(`_make_hue_range_circle`, lines 512–670), the user reports: **"I could see the
hue circle, but when clicking, it disappeared leaving behind only a gray
background."**

### Root Cause Analysis

The v8 standalone and the arm_segmentation integration differ in one critical
way: **how the hue circle's `SceneWidget` is sized**.

| Aspect | v8 standalone (works) | arm_segmentation (broken) |
|--------|----------------------|--------------------------|
| SceneWidget parent | Direct child of `Window` | Nested: Window → `layout` (Vert, right panel) → `container` (Vert) → `scene_widget` |
| Frame sizing | **Explicit** via `win.set_on_layout` callback: `scene_widget.frame = gui.Rect(x0, y, sq, sq)` | **Implicit** — relies on `gui.Vert` auto-layout to allocate height |
| `set_on_layout` | Sets frames for `title_lbl`, `scene_widget`, `edit_row` explicitly | Only sets frames for main `scene` (3D view, left 4/5) and `layout` (right panel, 1/5) — **never sizes the hue circle's inner SceneWidget** |

A `SceneWidget` has **no intrinsic 2D size** — unlike a `Label` or
`NumberEdit`, it cannot tell its parent Vert how tall it wants to be.  Open3D's
`gui.Vert` auto-layout likely assigns it zero or minimal height.  The initial
`set_background` call during construction renders at whatever initial size the
widget gets, but on first mouse interaction, a relayout is triggered and the
widget collapses → grey background, wheel gone.

This is the same class of issue that plagued v6/v6b: Open3D's layout system
does not handle nested `SceneWidget` sizing without explicit frame assignment.

### Why a Sandbox

Debugging layout issues inside the full `arm_segmentation.py` pipeline is
painful — it requires point cloud data, the full processing chain, and the
multi-step interactive flow.  A sandbox that reproduces only the **layout
structure** (3D scene left, controls panel right, hue circle in the panel)
allows rapid iteration on the fix in isolation.

## Goals

### In Scope
1. Reproduce the arm_segmentation `_display_pointcloud` layout: left 4/5 = 3D
   SceneWidget with a dummy point cloud, right 1/5 = Vert panel with controls
2. Embed the v8 hue circle (SceneWidget + background image) inside the right
   panel, exactly as `_make_hue_range_circle` does
3. Reproduce the "disappears on click" bug
4. Fix the layout so the hue circle maintains its size across interactions
5. Include companion S/V range sliders and Process/Continue buttons to match
   the real panel density
6. Validate drag, NumberEdit sync, and wrap-around ranges in the integrated
   layout

### Out of Scope
- Actual point cloud processing (use a random dummy cloud)
- Porting the fix back to `arm_segmentation.py` (separate follow-up commit)
- HSV hover readout on the 3D scene (not related to the bug)
- Replacing Open3D with PyQt5 (v8 approach works; just needs correct sizing)

## Success Criteria

- [ ] Sandbox runs standalone: `python hue_circle_v9_integration_sandbox.py`
- [ ] Window opens with 3D scene (left 4/5) and controls panel (right 1/5)
- [ ] Hue circle renders in the right panel on startup
- [ ] Clicking anywhere on the hue circle does **not** make it disappear
- [ ] Dragging handles updates hue angles smoothly (verified by console output)
- [ ] NumberEdit boxes sync bidirectionally with handles
- [ ] Wrap-around range (e.g. 330°–30°) displays correctly
- [ ] Resizing the window re-lays out correctly (circle stays square, visible)
- [ ] Process button runs a dummy re-processing (confirms panel survives relayout)
- [ ] The fix is a minimal, isolated change that can be ported back to
      `_make_hue_range_circle` with confidence

---

## Technical Design

### Approach

The sandbox script replicates the `_display_pointcloud` layout skeleton and
embeds the v8 hue circle inside the controls panel.  The fix will use one of
two strategies (to be determined during Phase 1):

**Strategy A — Explicit frame in the window's `on_layout` callback:**
The window's `set_on_layout` callback already positions the main scene and the
right panel.  Extend it to also walk into the panel and explicitly set
`scene_widget.frame` for the hue circle to a fixed square size.  This matches
what v8 standalone does.

**Strategy B — SceneWidget preferred size hack:**
Before adding the hue circle's SceneWidget to the Vert container, attempt to
set a preferred/fixed height so auto-layout allocates enough space.  Open3D's
`gui.Widget` has `preferred_height` / `preferred_width` in some versions; if
not available, use a `gui.Widget.Constraints` workaround.

**Strategy C — Wrapper widget with `_on_layout`:**
Wrap the hue circle's SceneWidget in a custom container that has its own
layout callback, explicitly sizing the SceneWidget to a square.

Strategy A is the most reliable (it's what v8 does), but it couples the hue
circle's layout to the window.  Strategy C is cleaner for encapsulation.
Phase 1 will test both.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|---------|
| **A: Explicit frame in window on_layout** | Proven in v8; simple | Couples hue circle sizing to window layout | Test first |
| **B: preferred_height on SceneWidget** | Clean, no layout coupling | May not exist in O3D 0.19; undocumented | Test second |
| **C: Wrapper with own _on_layout** | Encapsulated; portable | Extra nesting; unclear if nested layout callbacks work in O3D | Fallback |
| **Replace with PyQt5** | Reliable 2D toolkit | Separate window; event loop bridging; overkill since v8 rendering works | Archived (see `docs/development/plans/archived/hue-circle-pyqt-sandbox.md`) |

### Architecture

```
Sandbox window layout (mimics _display_pointcloud):
┌─────────────────────────────────────────────┬──────────────┐
│                                             │ H range:     │
│                                             │ ┌──────────┐ │
│                                             │ │  hue     │ │
│              3D SceneWidget                 │ │  circle  │ │
│              (dummy point cloud)            │ │(SceneW.) │ │
│              (left 4/5)                     │ └──────────┘ │
│                                             │ H start: [__]│
│                                             │ H end:   [__]│
│                                             │              │
│                                             │ S range:     │
│                                             │ [slider] [__]│
│                                             │ [slider] [__]│
│                                             │              │
│                                             │ V range:     │
│                                             │ [slider] [__]│
│                                             │ [slider] [__]│
│                                             │              │
│                                             │ [Process]    │
│                                             │ [Continue]   │
│                                             │  (right 1/5) │
└─────────────────────────────────────────────┴──────────────┘
```

The key change vs current `_make_hue_range_circle`: the hue circle's
`SceneWidget` must receive an **explicit square frame** of known pixel
dimensions, not rely on `gui.Vert` auto-layout.

---

## Implementation Plan

### Phase 1: Reproduce the Bug
**Goal:** Confirm the disappear-on-click bug in a minimal layout that mirrors
arm_segmentation.

**Tasks:**
- [ ] Task 1.1 — Create `hue_circle_v9_integration_sandbox.py` with the
      window layout: left 4/5 = 3D SceneWidget + dummy point cloud, right 1/5
      = `gui.Vert` panel
- [ ] Task 1.2 — Copy `_make_hue_range_circle` (as-is from arm_segmentation,
      lines 512–670) into the sandbox and embed its returned container into the
      right panel
- [ ] Task 1.3 — Add Process and Continue buttons to the panel (mimicking
      `_display_pointcloud`)
- [ ] Task 1.4 — Add the window's `set_on_layout` callback (only sizing main
      scene + panel, same as arm_segmentation lines 1006–1010)
- [ ] Task 1.5 — Run and confirm the bug: circle renders on startup, disappears
      on first click

**Files Created:**
- `code/scripts/__misc/hue_circle_v9_integration_sandbox.py`

**Dependencies:** None

### Phase 2: Fix the Layout
**Goal:** The hue circle's SceneWidget survives clicks and relayouts.

**Tasks:**
- [ ] Task 2.1 — **Test Strategy A:** Extend the window's `on_layout` to
      explicitly set `hue_scene_widget.frame` to a fixed square (e.g. panel
      width × panel width)
- [ ] Task 2.2 — If Strategy A works, verify drag + NumberEdit sync function
      correctly with the explicit frame
- [ ] Task 2.3 — If Strategy A doesn't work or is too coupled, **test Strategy
      C:** wrap the hue circle SceneWidget in a container that overrides
      layout to force a square frame
- [ ] Task 2.4 — Verify the 3D scene's mouse hover (for HSV readout) still
      works alongside the hue circle's mouse handler (no event conflicts)
- [ ] Task 2.5 — Add diagnostic prints: `[v9] on_layout: hue_sw.frame = ...`
      to confirm the frame is set on every relayout

**Files Modified:**
- `code/scripts/__misc/hue_circle_v9_integration_sandbox.py`

**Dependencies:** Phase 1

### Phase 3: Full Interaction Validation
**Goal:** Confirm all interactions work in the integrated layout.

**Tasks:**
- [ ] Task 3.1 — Drag both handles, verify console output shows correct hue
      values
- [ ] Task 3.2 — Type values in NumberEdit boxes, verify handles move
- [ ] Task 3.3 — Set wrap-around range (330°–30°), verify arc highlights
      through 0°
- [ ] Task 3.4 — Click "Process" button, verify the panel and hue circle
      survive the relayout triggered by re-processing
- [ ] Task 3.5 — Resize the window, verify the hue circle stays square and
      visible
- [ ] Task 3.6 — Interact with the 3D scene (rotate, pan), verify it does
      **not** interfere with the hue circle and vice versa
- [ ] Task 3.7 — Document the minimal fix (which lines/strategy) as a comment
      block at the top of the sandbox, ready for porting back

**Files Modified:**
- `code/scripts/__misc/hue_circle_v9_integration_sandbox.py` — final polish

**Dependencies:** Phase 2

---

## Testing Plan

### Manual Verification
- [ ] `python hue_circle_v9_integration_sandbox.py` opens without errors
- [ ] Hue circle is visible on startup in the right panel
- [ ] Single click on the hue circle — does NOT disappear
- [ ] Drag start handle clockwise past 0° — smooth, no jump
- [ ] Drag end handle to overlap start handle — both handles reachable
- [ ] Type 330 in H start, 30 in H end — arc wraps through red
- [ ] Click Process — 3D cloud re-renders, hue circle unaffected
- [ ] Resize window smaller — circle shrinks but stays square, visible
- [ ] Resize window larger — circle grows, handles stay on ring track

### Edge Cases
- [ ] Minimize and restore window — hue circle repaints correctly
- [ ] Rapidly alternate: drag handle → click Process → drag handle — no crash
- [ ] Panel narrower than circle min size — graceful degradation (clipped, not
      crashed)

---

## Documentation Plan

- [ ] Comment block at top of sandbox explaining the fix and why it's needed
- [ ] Update `docs/development/knowledge-base/bug-hue-circle-drag-not-working.md` with the layout root
      cause and fix strategy once validated
- [ ] No README/CLAUDE.md changes (sandbox only)

---

## Rollback Plan

Sandbox only — no production code modified.

1. Delete `code/scripts/__misc/hue_circle_v9_integration_sandbox.py`
2. Delete branch `feature/hue-circle-integration-sandbox`
3. `arm_segmentation.py` is untouched until the fix is ported in a follow-up

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `gui.Vert` auto-layout overrides explicit frame on relayout | Medium | High | Strategy C (wrapper with own layout callback) as fallback; test with diagnostic prints |
| Nested `set_on_layout` callbacks not supported in O3D 0.19 | Medium | Medium | Fall back to Strategy A (window-level sizing) |
| Mouse events routed to wrong SceneWidget (3D scene vs hue circle) | Low | High | Events are routed by frame bounds; verify with diagnostic prints |
| `set_background` image not scaling to non-square frame | Low | Medium | Force square frame in layout; pad with black if needed |
| `enable_scene_caching(False)` causes performance issues with two SceneWidgets | Low | Low | Only the hue circle disables caching; the 3D scene keeps default |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Reproduce the bug | Small | None |
| Phase 2: Fix the layout | Small–Medium | Phase 1 |
| Phase 3: Validation | Small | Phase 2 |

---

## References

- Bug report: `docs/development/knowledge-base/bug-hue-circle-drag-not-working.md`
- Working standalone: `code/scripts/__misc/hue_circle_v8_bg_image.py`
- Broken integration: `code/src/preprocessing/forearm_extraction/arm_segmentation.py:512–670` (`_make_hue_range_circle`)
- Integration layout: `code/src/preprocessing/forearm_extraction/arm_segmentation.py:1006–1010` (`on_layout`)
- Archived alternative: `docs/development/plans/archived/hue-circle-pyqt-sandbox.md`
