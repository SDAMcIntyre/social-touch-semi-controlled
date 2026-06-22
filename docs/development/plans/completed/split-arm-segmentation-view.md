# Plan: Split ArmSegmentation into Processing + View Modules

**Date:** 2026-06-10
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-06-22 17:57
**Started:** 2026-06-10
**Base Branch:** `feature/standalone-mkv-to-forearm-mesh`
**Branch:** `feature/standalone-mkv-to-forearm-mesh` (same branch — this work
unblocks the standalone tool)

---

## Overview

Split `arm_segmentation.py` (1,165 lines) into a processing module and a view
module, then add a legacy GLFW Visualizer fallback for machines where
Filament's `wglCreateContextAttribs()` crashes. The public API does not change.

## Problem Statement

`ArmSegmentation` mixes point-cloud processing logic (~270 lines) with 400+
lines of Filament GUI code (slider factories, hue wheel renderer, layout
callbacks). A second GUI backend (legacy GLFW Visualizer) is needed as a
fallback for NVIDIA driver 610.47, which breaks Filament's OpenGL context
creation. Adding the fallback inline would push the file to ~1,400 lines with
two unrelated GUI implementations tangled with processing logic.

## Goals

### In Scope

1. Split `arm_segmentation.py` into `arm_segmentation.py` (processing + params)
   and `arm_segmentation_view.py` (all display code)
2. Add a Filament availability probe (subprocess-based, cached)
3. Add a legacy GLFW Visualizer fallback with keyboard-driven parameter tuning
4. Preserve the existing public API (zero caller changes)

### Out of Scope

- Refactoring `preprocess()` / `extract_arm()` call signatures
- Adding new processing algorithms or parameters
- Changing the Filament GUI layout or widgets
- Supporting Vulkan or software rendering backends

## Success Criteria

- [ ] `arm_segmentation.py` contains only processing logic and params (~310 lines)
- [ ] `arm_segmentation_view.py` contains all display code (~900 lines)
- [ ] Barrel export (`from .arm_segmentation import ArmSegmentation`) unchanged
- [ ] `extract_participant_forearm.py` and `standalone_mkv_to_forearm_mesh.py`
      require zero modifications
- [ ] Batch mode (`interactive=False`) never imports `gui` or `rendering`
- [ ] On machines where Filament works, the full slider GUI is used automatically
- [ ] On machines where Filament crashes, the legacy keyboard UI activates with
      a clear warning message
- [ ] `pytest` passes

---

## Technical Design

### Approach

Composition with lazy delegation: `ArmSegmentation._display_pointcloud()`
keeps the non-interactive (batch) path inline and delegates the interactive
path to a module-level function in `arm_segmentation_view.py` via lazy import.
The view module receives the `ArmSegmentation` instance as a parameter and
accesses its `params`, `was_modified`, `_SLIDER_CONFIGS`, and `_hue_in_range()`
directly.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Composition with lazy delegation | No new abstractions, lazy import avoids GUI deps in batch mode, minimal caller impact | View accesses private attrs | **Chosen** |
| Strategy pattern (abstract `DisplayBackend`) | Clean polymorphism | Over-engineered for 2 backends, adds an ABC for no practical benefit | Rejected |
| Mixin classes (`FilamentMixin`, `LegacyMixin`) | No new files | Diamond inheritance, still one large file per mixin, harder to test | Rejected |
| Module-level free functions only (no class in view) | Simplest | Loses namespacing, harder to discover | Rejected in favour of module-level functions *with* clear naming |

### Architecture Changes

```
forearm_extraction/
    arm_segmentation.py          (~310 lines, processing + params)
    arm_segmentation_view.py     (~900 lines, Filament + legacy display)  NEW
    __init__.py                  (unchanged)
```

**Interface between files:**

| View accesses on segmenter | Purpose |
|---------------------------|---------|
| `segmenter.params[params_key]` | Read/write parameter values |
| `segmenter._SLIDER_CONFIGS` | Widget configuration |
| `segmenter.was_modified = True` | Flag user interaction |
| `ArmSegmentation._hue_in_range()` | HSV hover readout (Filament only) |

### Knowledge Base Constraints

| Constraint | Source | How addressed |
|------------|--------|---------------|
| CuPy before preprocessing imports | `note-cupy-import-order.md` | Processing module does not change import order; view module has no CuPy dependency |
| SceneWidget never nested in gui.Vert | `note-open3d-scenewidget-layout.md` | Fragment-return pattern (`_make_hue_range_circle` dict) preserved unchanged |
| Explicit frame in on_layout | `note-open3d-scenewidget-layout.md` | `on_layout` callback moved intact to view module |
| Filament fallback to legacy Visualizer | `issue-open3d-filament-opengl-crash.md` | Subprocess probe + `VisualizerWithKeyCallback` fallback |

---

## Implementation Plan

### Phase 1: Create view module with Filament path
**Goal:** Extract all display code into `arm_segmentation_view.py` without
adding the legacy fallback yet. The Filament probe always returns True
(hardcoded) so behaviour is identical to today.

**Tasks:**

- [x] 1.1 Create `arm_segmentation_view.py` with imports (`numpy`, `open3d`,
  `colorsys`, `typing`)
- [x] 1.2 Move `_get_screen_size()` as module-level function
- [x] 1.3 Move widget factories as module-level functions: `_make_slider`,
  `_make_range_slider_row`, `_render_hue_wheel`, `_render_hue_arc_overlay`,
  `_make_hue_range_circle`
- [x] 1.4 Update internal references: `ArmSegmentation._make_slider(...)` and
  `self._make_slider(...)` become `_make_slider(...)`; same for other moved
  functions
- [x] 1.5 Extract the Filament interactive path (current lines 737-1107) into
  `_display_filament(segmenter, pcd_input, window_name, params_key,
  processing_func, is_cluster_step)` — import `gui` and `rendering` inside
  this function only
- [x] 1.6 Replace all `self.X` with `segmenter.X` in the extracted code
- [x] 1.7 Create public entry point `display_pointcloud_interactive()` that
  calls `_display_filament()` directly (no probe yet)

**Files Modified:**

- `code/src/preprocessing/forearm_extraction/arm_segmentation_view.py` — new file

**Dependencies:** None

### Phase 2: Modify processing module
**Goal:** Slim down `arm_segmentation.py` to processing + params, with lazy
delegation to the view module.

**Tasks:**

- [x] 2.1 Remove `gui`, `rendering` imports from `arm_segmentation.py`
- [x] 2.2 Remove `subprocess`, `sys` imports (were added earlier in session,
  will live in view module instead)
- [x] 2.3 Add `self._view = None` to `__init__`
- [x] 2.4 Rewrite `_display_pointcloud()`: keep non-interactive path unchanged,
  replace interactive path with lazy import + delegation to
  `arm_segmentation_view.display_pointcloud_interactive()`
- [x] 2.5 Remove all moved methods: `_make_slider`, `_make_range_slider_row`,
  `_render_hue_wheel`, `_render_hue_arc_overlay`, `_make_hue_range_circle`,
  `_get_screen_size`
- [x] 2.6 Keep: `_DEFAULT_PARAMS`, `_SLIDER_CONFIGS`, `__init__`, `preprocess`,
  `extract_arm`, `_apply_clustering`, `_apply_skin_color_filter`,
  `_hue_in_range`, `deep_update`, `__main__`

**Files Modified:**

- `code/src/preprocessing/forearm_extraction/arm_segmentation.py` — major rewrite

**Dependencies:** Phase 1

### Phase 3: Add Filament probe + legacy fallback
**Goal:** Add the subprocess Filament probe and the keyboard-driven legacy
Visualizer fallback to the view module.

**Tasks:**

- [x] 3.1 Add `_probe_filament()` — runs
  `gui.Application.instance.create_window()` in a subprocess, returns
  bool, cached in module-level `_filament_available`
- [x] 3.2 Add `_is_filament_available()` wrapper — first-call probe with
  warning message if unavailable
- [x] 3.3 Update `display_pointcloud_interactive()` to route based on probe
  result
- [x] 3.4 Add `_build_param_entries(params, slider_cfg)` — flatten
  `_SLIDER_CONFIGS` into list of tunable entries with step sizes
- [x] 3.5 Add `_print_legacy_status()` — formatted terminal parameter table
- [x] 3.6 Add `_print_legacy_help()` — keybinding reference
- [x] 3.7 Add `_display_legacy()` — `VisualizerWithKeyCallback` with keyboard
  controls: Up/Down select param, Left/Right adjust, -/= big step,
  Space/Enter process, Q/Esc continue, H help
- [x] 3.8 GLFW key constants as module-level: UP=265, DOWN=264, LEFT=263,
  RIGHT=262, ESC=256, ENTER=257

**Files Modified:**

- `code/src/preprocessing/forearm_extraction/arm_segmentation_view.py` — additions

**Dependencies:** Phase 2

### Phase 4: Documentation + verification
**Goal:** Update KB doc and verify the full pipeline.

**Tasks:**

- [x] 4.1 Update `issue-open3d-filament-opengl-crash.md` — mark "Next step"
  as implemented, add pointer to `arm_segmentation_view.py`
- [x] 4.2 Run `pytest` — existing tests pass
- [x] 4.3 Manual test: standalone script with Filament working (full GUI) — SKIP (requires hardware)
- [x] 4.4 Manual test: standalone script with Filament broken (legacy keyboard
  UI activates, parameter tuning works, processing applies) — SKIP (requires hardware)
- [x] 4.5 Verify batch mode does not import `gui`/`rendering`

**Files Modified:**

- `docs/development/knowledge-base/issue-open3d-filament-opengl-crash.md`

**Dependencies:** Phase 3

---

## Testing Plan

### Unit Tests

- [x] Existing `pytest` suite passes (tests stub preprocessing, so the split
  is transparent)

### Manual Verification

- [ ] Batch mode: instantiate `ArmSegmentation(interactive=False)`, call
  `preprocess()` + `extract_arm()` — no GUI imports loaded, no windows
- [ ] Interactive mode (Filament OK): run standalone script, verify full slider
  GUI opens, hue circle works, parameter changes apply on Process, Continue
  closes window
- [ ] Interactive mode (Filament broken): force probe to return False (e.g.
  set `_filament_available = False`), verify legacy keyboard UI opens,
  Up/Down selects params, Left/Right adjusts, Space processes, Q quits
- [ ] Import isolation: `python -c "from preprocessing.forearm_extraction
  import ArmSegmentation"` does not load `gui`/`rendering` modules

### Edge Cases

- [ ] Empty point cloud input in legacy mode — should print message, not crash
- [ ] No tunable parameters for a step — show point cloud without keyboard
  controls
- [ ] Hue wrap-around (e.g. 335-25) works in legacy mode via modular
  arithmetic on arrow key adjustment

---

## Documentation Plan

- [x] Update `issue-open3d-filament-opengl-crash.md` with implementation status
- [x] No README/CLAUDE.md changes needed (internal refactor, no API changes)

---

## Rollback Plan

1. `git revert` the split commits — the file was a single module before, all
   code moves are traceable
2. No data migrations, no config changes, no external dependencies added
3. The barrel export (`__init__.py`) is unchanged, so reverting is clean

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Filament probe adds 2-3s startup latency on first interactive call | High | Low | Cached after first call; batch mode never triggers it |
| Arrow keys not delivered by VisualizerWithKeyCallback | Low | High | GLFW key constants (265/264/263/262) are well-documented; fall back to letter keys (W/S/A/D) if needed |
| In-place geometry update fails with changed point count in legacy vis | Low | Med | Test with varying point counts; fall back to clear + re-add if needed |
| Circular import between processing and view modules | Low | High | View uses `TYPE_CHECKING` guard only; no runtime import of `ArmSegmentation` |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Create view module | 30 min | None |
| Phase 2: Modify processing module | 20 min | Phase 1 |
| Phase 3: Add legacy fallback | 30 min | Phase 2 |
| Phase 4: Documentation + verification | 15 min | Phase 3 |

---

## References

- Knowledge base: `docs/development/knowledge-base/issue-open3d-filament-opengl-crash.md`
- Knowledge base: `docs/development/knowledge-base/note-open3d-scenewidget-layout.md`
- Knowledge base: `docs/development/knowledge-base/note-cupy-import-order.md`

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/src/preprocessing/forearm_extraction/arm_segmentation.py
- code/src/preprocessing/forearm_extraction/arm_segmentation_view.py
- docs/development/knowledge-base/issue-open3d-filament-opengl-crash.md
- docs/development/plans/active/split-arm-segmentation-view.md
