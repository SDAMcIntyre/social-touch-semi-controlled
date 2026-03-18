# Plan: GUI Embedded Console Output

**Date:** 2026-03-17
**Author:** Basil Duvernoy
**Status:** Active
**Branch:** `feature/unified-touch-analysis-pipeline`

---

## Overview

Add an embedded console panel to the DAG Launcher GUI that captures and displays real-time stdout/stderr output from pipeline subprocesses. Currently all pipeline output is lost unless the user monitors a separate terminal window.

## Problem Statement

When a pipeline is launched from the GUI via `_on_run()`, the subprocess is created with no stdout/stderr capture (`subprocess.Popen` at `launcher_window.py:321`). All print statements, logging output, and Prefect task logs go to the parent console — which is invisible if the GUI was launched without a terminal. Users have no way to monitor pipeline progress, diagnose errors, or see task-level status from within the GUI.

## Goals

### In Scope
1. Capture subprocess stdout and stderr in real-time
2. Display captured output in a scrollable, read-only console panel within the GUI
3. Non-blocking output reading (GUI remains responsive during long runs)
4. Console clears on each new run; previous output is not retained across runs

### Out of Scope
- Log persistence / export to file (future enhancement)
- ANSI color code rendering (output displayed as plain text)
- Filtering or searching within console output
- Multiple concurrent pipeline runs

## Success Criteria

- [ ] Pipeline stdout/stderr appears line-by-line in the GUI console panel during execution
- [ ] GUI remains responsive (no freezing) during output-heavy pipeline runs
- [ ] Abort still works cleanly — output stops, no orphan threads
- [ ] Closing the window during a run terminates both subprocess and reader thread
- [ ] Console auto-scrolls to latest output

---

## Technical Design

### Approach

Use `subprocess.PIPE` to capture the child process output, a `QThread` worker to read the pipe without blocking the GUI event loop, and a `QPlainTextEdit` widget to display lines as they arrive via Qt signal/slot.

This is the standard PyQt5 pattern for subprocess output capture. The alternative (`QProcess`) was considered but rejected — see below.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `subprocess.Popen` + PIPE + QThread reader | Full control over env/cwd (matches existing code), simple signal-based line delivery | Must manage thread lifecycle manually | **Chosen** |
| `QProcess` (Qt's built-in process API) | Built-in `readyRead` signal, no manual threading | Different API from current `subprocess.Popen` usage; would require rewriting `_on_run`, `_poll_process`, `_on_abort`, and `closeEvent`; less flexible env control | Rejected |
| Redirect to temp file + tail with QTimer | No threading needed | Latency (poll-based), temp file cleanup, platform differences | Rejected |

### Architecture Changes

Two new modules created; one existing module modified:

```
code/src/utils/gui/dag_launcher/
    console_widget.py          # NEW — QPlainTextEdit wrapper
    process_output_reader.py   # NEW — QThread pipe reader
    launcher_window.py         # MODIFIED — layout + wiring
```

No new dependencies — uses only `PyQt5` (already required) and `subprocess` (already imported).

---

## Implementation Plan

### Phase 1: Output Reader Thread
**Goal:** Create a reusable QThread that reads subprocess stdout line-by-line and emits a Qt signal.

- [ ] Create `process_output_reader.py` with `ProcessOutputReader(QThread)` class
- [ ] Constructor accepts `subprocess.Popen` instance (must have `stdout=PIPE`)
- [ ] `run()` reads `process.stdout` line-by-line, emits `line_received(str)` signal
- [ ] Loop terminates naturally when pipe closes (process exit or abort)
- [ ] Decode with `utf-8`, `errors="replace"` for robustness

**Files Modified:**
- `code/src/utils/gui/dag_launcher/process_output_reader.py` — **New file**

**Dependencies:** None

### Phase 2: Console Widget
**Goal:** Create a read-only text display panel.

- [ ] Create `console_widget.py` with `ConsoleWidget(QWidget)` class
- [ ] Contains a `QPlainTextEdit` set to read-only with monospace font
- [ ] `append_line(text: str)` method — appends and auto-scrolls to bottom
- [ ] `clear()` method — clears all content
- [ ] `setMaximumBlockCount(10_000)` to cap memory usage on long runs

**Files Modified:**
- `code/src/utils/gui/dag_launcher/console_widget.py` — **New file**

**Dependencies:** None

### Phase 3: Integration into LauncherWindow
**Goal:** Wire the console widget and reader thread into the existing launcher.

- [ ] Import `ConsoleWidget` and `ProcessOutputReader`
- [ ] Wrap existing horizontal splitter + new console widget in a vertical `QSplitter` (top: workflow panels ~70%, bottom: console ~30%)
- [ ] Run bar remains below the vertical splitter (not inside it)
- [ ] In `_on_run()`: add `stdout=subprocess.PIPE, stderr=subprocess.STDOUT` to `Popen`
- [ ] In `_on_run()`: call `console.clear()`, create `ProcessOutputReader`, connect `line_received` to `console.append_line`, start thread
- [ ] In `_poll_process()`: after detecting process exit, call `reader.wait()` to flush remaining lines
- [ ] In `_on_abort()`: existing `process.terminate()` closes the pipe, which ends the reader thread naturally
- [ ] In `closeEvent()`: if reader thread exists, `wait()` on it before proceeding
- [ ] Store `_reader: ProcessOutputReader | None` alongside existing `_process` attribute

**Files Modified:**
- `code/src/utils/gui/dag_launcher/launcher_window.py` — Layout restructuring + subprocess wiring

**Dependencies:** Phase 1, Phase 2

---

## Testing Plan

### Manual Verification
- [ ] Launch GUI (`python code/scripts/launch_pipeline_gui.py`), select a workflow, click Run — output appears line-by-line in console
- [ ] During a long run, interact with the GUI (resize, click tabs) — no freezing
- [ ] Click Abort during a run — output stops, no error dialogs, no hanging threads
- [ ] Close the window during a run — clean shutdown, no orphan processes or threads
- [ ] Run a second pipeline after the first completes — console clears and shows new output
- [ ] Run a pipeline that produces stderr output (e.g., warnings) — stderr lines appear in console

### Edge Cases
- [ ] Pipeline that produces no output — console stays empty, no errors
- [ ] Pipeline that produces very large output (>10,000 lines) — oldest lines are discarded, GUI stays responsive
- [ ] Pipeline that exits immediately (e.g., config error) — error output appears before "Failed" status

---

## Documentation Plan

- [ ] No external documentation needed — this is a GUI-internal enhancement
- [ ] Inline docstrings in new modules

---

## Rollback Plan

1. Revert the three file changes (two new files, one modified file)
2. No data migrations or external state changes involved
3. The existing subprocess execution path is preserved in structure — rollback restores the original `Popen` call without pipes

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Pipe buffer fills up if GUI can't consume fast enough (deadlock) | Low | High | `stderr=STDOUT` merges into single pipe; reader thread consumes continuously; `QPlainTextEdit.setMaximumBlockCount` caps widget memory |
| Reader thread doesn't terminate on abort | Low | Med | Pipe close from `process.terminate()` causes `readline()` to return empty, ending the loop; `wait()` in cleanup as safety net |
| Encoding errors from non-UTF-8 subprocess output | Low | Low | `errors="replace"` in decode |

---

## References

- Current execution code: `code/src/utils/gui/dag_launcher/launcher_window.py` lines 303-345
- Knowledge base: `docs/development/knowledge-base/note-open3d-scenewidget-layout.md` (layout patterns — tangentially relevant)
