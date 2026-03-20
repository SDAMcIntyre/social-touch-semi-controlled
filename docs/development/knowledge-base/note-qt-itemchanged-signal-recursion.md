# Qt `itemChanged` Signal Recursion

## Symptom

`RecursionError` (Python maximum recursion depth exceeded) when a user edits an
inline scalar cell (e.g. a string or numeric value) in a `QTableWidget` and
confirms the edit. The crash occurs immediately on pressing Enter or clicking
away — the GUI terminates without displaying an error dialog.

## Investigation

The handler `_on_item_changed()` is connected to `QTableWidget.itemChanged`.
Logging revealed it was called more than once per user edit. Adding a stack
trace showed the call stack growing without bound — confirming infinite
re-entry, not a bug in the model or YAML layer.

## Root Cause

`QTableWidgetItem.setData()` emits `itemChanged` unconditionally, even when
called from within an `itemChanged` handler. The success path of
`_on_item_changed()` called `item.setData(Qt.UserRole, new_val)` to persist the
new value on the item — this re-fired `itemChanged`, re-entered the handler, hit
`setData` again, and so on indefinitely.

The revert path (type-coercion failure) already used `blockSignals(True/False)`
correctly; the success path did not.

## Architecture Constraints

- `QTableWidget.blockSignals(True)` suppresses **all** signals on the widget,
  not just `itemChanged`. This is safe here because the only signal of interest
  during the two-line write (`setData` + implicit repaint) is `itemChanged`
  itself, and `task_changed.emit()` is called explicitly **after** unblocking.
- Blocking signals on the **table** (not the item) is the correct level —
  `QTableWidgetItem` has no `blockSignals` method.

## Fix Applied

Wrap the success-path `setData` call with `blockSignals`, mirroring the
existing revert pattern:

```python
# Before (crashes)
self._model.set_task_option(task_name, opt_key, new_val)
item.setData(Qt.UserRole, new_val)
self.task_changed.emit()

# After (correct)
self._model.set_task_option(task_name, opt_key, new_val)
self._table.blockSignals(True)
item.setData(Qt.UserRole, new_val)
self._table.blockSignals(False)
self.task_changed.emit()
```

File: `code/src/utils/gui/dag_launcher/task_panel.py`, method
`_on_item_changed()`.

## Reusable Pattern

Any time a Qt signal handler mutates item/widget state that would re-emit the
same signal, guard the mutation with `blockSignals`:

```python
widget.blockSignals(True)
item.setData(role, value)   # or setText(), setValue(), etc.
widget.blockSignals(False)
```

Checklist:
- [ ] Is the handler connected to a signal that `setData`/`setText`/`setValue`
  also emits? If yes, guard the write.
- [ ] Are there other signals on the same widget that must NOT be suppressed
  during the guard? If yes, consider a narrower guard (disconnect/reconnect the
  specific signal instead of `blockSignals`).
- [ ] Is any downstream emit needed after the guarded write? Call it explicitly
  after `blockSignals(False)`.

## References

- **Plan:** `docs/development/plans/active/fix-gui-scalar-edit-crash.md`
- **Fix location:** `code/src/utils/gui/dag_launcher/task_panel.py:337–365`
- **Existing revert pattern (pre-fix reference):** same file, lines 355–357
