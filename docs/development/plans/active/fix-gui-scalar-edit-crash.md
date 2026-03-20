# Plan: Fix GUI Crash When Editing Scalar Task Options

**Date:** 2026-03-20
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/fix-gui-scalar-edit-crash`

---

## Overview

The pipeline launcher GUI crashes immediately when a user edits an inline scalar
text value (e.g. changing `mode` from `"compare"` to `"correct"` in the
`correct_xyz_stickers_motion` task). The root cause is infinite signal recursion
in `TaskPanel._on_item_changed()`. The fix is a one-line `blockSignals` guard.

## Problem Statement

When a user double-clicks an editable string or numeric cell in the task table,
types a new value, and confirms the edit, the GUI terminates with a
`RecursionError`. This affects every inline-editable scalar cell — currently
`mode` and `filter_method` in the preprocessing configs — making those options
impossible to change from the GUI.

## Goals

### In Scope
1. Eliminate the infinite-recursion crash when editing inline scalar cells
2. Add a knowledge-base note documenting the Qt signal-recursion pattern

### Out of Scope
- Adding dropdown/combobox widgets for constrained-choice options (future enhancement)
- Refactoring other signal handling in the task panel

## Success Criteria

- [ ] Editing `mode` from `"compare"` to `"correct"` (and back) does not crash
- [ ] Editing `filter_method` from `"butterworth"` to `"savgol"` (and back) does not crash
- [ ] Saving after edits produces correct YAML output
- [ ] Existing checkbox and complex-cell editing still works

---

## Technical Design

### Approach

The crash is caused by a missing `blockSignals` guard in the success path of
`_on_item_changed()`. When the handler calls `item.setData(Qt.UserRole, new_val)`
(line 363), Qt emits `itemChanged` again, causing infinite re-entry:

```
itemChanged → _on_item_changed → item.setData() → itemChanged → _on_item_changed → ...
```

The revert path (lines 355–357) already uses `blockSignals(True/False)` correctly.
The fix applies the same pattern to the success path.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `blockSignals` around `setData` | Minimal change, mirrors existing revert pattern | None | **Chosen** |
| Disconnect/reconnect `itemChanged` | Also prevents recursion | More code, risk of lost signals | Rejected |
| Guard flag (`self._updating`) | Framework-agnostic | Extra state to manage, easy to forget | Rejected |

### Architecture Changes

None — single method modified in one file.

### Knowledge Base

No existing notes apply. A new note (`note-qt-itemchanged-signal-recursion.md`)
will be added documenting the pattern for future reference.

---

## Implementation Plan

### Phase 1: Fix the crash
**Goal:** Eliminate the infinite recursion

- [x] Wrap the success-path writes in `_on_item_changed()` with `blockSignals`

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_panel.py` — add `blockSignals(True/False)` around lines 362–363

**Dependencies:** None

### Phase 2: Knowledge base note
**Goal:** Document the pattern for future Qt signal work

- [x] Create `docs/development/knowledge-base/note-qt-itemchanged-signal-recursion.md`
- [x] Add entry to knowledge-base `README.md` index

**Files Modified:**
- `docs/development/knowledge-base/note-qt-itemchanged-signal-recursion.md` (new)
- `docs/development/knowledge-base/README.md` — add index row

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Launch GUI (`python code/scripts/launch_pipeline_gui.py`)
- [ ] Select `preprocess_workflow_kinect_auto` workflow
- [ ] Double-click `mode` cell for `correct_xyz_stickers_motion`, change to `"compare"`, press Enter — no crash
- [ ] Change it back to `"correct"` — no crash
- [ ] Double-click `filter_method`, change to `"savgol"`, press Enter — no crash
- [ ] Verify title shows dirty marker (`*`)
- [ ] Save (Ctrl+S), reopen — verify YAML has updated values
- [ ] Toggle `enabled` checkboxes — still works
- [ ] Click a complex cell (e.g. `filter_params`) — YAML editor still opens

### Edge Cases
- [ ] Type an invalid value for a numeric cell, confirm revert still works (no crash)
- [ ] Edit a cell then immediately switch workflows — no crash

---

## Documentation Plan

- [ ] Knowledge-base note (Phase 2)
- [ ] No other documentation changes needed

---

## Rollback Plan

Revert the single commit on the feature branch. No data migrations or breaking changes.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `blockSignals` suppresses a needed signal | Low | Low | Only blocks during the two-line setData+setOption call; `task_changed.emit()` is called after unblocking |

---

## References

- **Bug location:** `code/src/utils/gui/dag_launcher/task_panel.py:337–364`
- **Existing correct pattern:** same file, lines 355–357 (revert path)
