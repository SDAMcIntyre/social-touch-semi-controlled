# Plan: Swap Kinect Directories and Tasks columns in DAG Launcher

**Date:** 2026-03-04
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/dag-launcher-swap-columns`

---

## Overview

Swap the center and right columns of the DAG config launcher GUI so that the Tasks panel is in the center and the Kinect Config Directories tree is on the right. The workflow selector remains on the left.

## Problem Statement

The current column order (Workflows | Kinect Directories | Tasks) does not match the desired layout. Tasks are the primary editing surface and belong in the center; Kinect directories are a selection panel better suited to the right.

## Goals

### In Scope
1. Move the Tasks panel from the right column to the center column
2. Move the Kinect Config Directories panel from the center column to the right column
3. Preserve existing stretch factors (tasks get 2/4 width, kinect dirs get 1/4)

### Out of Scope
- Changing widget internals or behavior
- Adding new features to any panel
- Modifying signal/slot wiring (references widgets by instance, not position)

## Success Criteria

- [ ] GUI launches with columns ordered: Workflows (left) | Tasks (center) | Kinect Directories (right)
- [ ] Tasks panel occupies ~50% width, other two panels ~25% each
- [ ] All existing functionality (workflow selection, kinect toggling, task editing, save, run) works unchanged

---

## Technical Design

### Approach

Swap the `addWidget` call order and corresponding stretch factors in `launcher_window.py`. The signal/slot connections reference widget instances (not splitter indices), so no other wiring changes are needed.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Swap addWidget order | Minimal change, no side effects | None | Chosen |
| Rebuild with different layout manager | More flexible | Unnecessary complexity | Rejected |

### Architecture Changes

None — same widgets, same signals, just reordered in the splitter.

---

## Implementation Plan

### Phase 1: Swap columns
**Goal:** Reorder the splitter children

- [ ] Swap `addWidget` calls so `TaskPanel` is added before `KinectDirectorySelector`
- [ ] Swap stretch factors so index 1 gets `2` and index 2 gets `1`
- [ ] Swap initial sizes in `showEvent` so center gets `w // 2` and right gets `w // 4`

**Files Modified:**
- `code/src/utils/gui/dag_launcher/launcher_window.py` — Reorder `addWidget` calls (~3 lines), update `setStretchFactor` indices (~2 lines), update `setSizes` list (~1 line)

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Run `python code/scripts/launch_dag_config_gui.py`
- [ ] Confirm column order: Workflows | Tasks | Kinect Directories
- [ ] Select a workflow and verify tasks populate in the center
- [ ] Toggle kinect directories on the right and verify model updates
- [ ] Save, resize, and run a workflow to confirm no regressions

---

## Documentation Plan

- [ ] No documentation changes needed (internal layout only)

---

## Rollback Plan

1. Revert the single commit on `launcher_window.py`

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Signal/slot breaks from reorder | Low | Med | Connections use widget instances, not indices — verified in code |

---

## References

- Main file: `code/src/utils/gui/dag_launcher/launcher_window.py`
