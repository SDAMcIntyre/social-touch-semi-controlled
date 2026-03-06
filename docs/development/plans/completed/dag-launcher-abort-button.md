# Plan: Add Abort Button to DAG Launcher GUI

**Date:** 2026-03-04
**Status:** Completed
**Branch:** dev

---

## Overview

**What:** Add a red Abort button to the DAG launcher run bar that terminates the running subprocess.
**Why:** Once a workflow is started there is no way to stop it from within the GUI — the user must kill the process externally.
**How:** Add an `_aborting` flag and an Abort button that calls `process.terminate()`; the existing poll timer handles all cleanup.

---

## Problem Statement

- Once a workflow subprocess is launched via the Run button, there is no way to cancel it from the GUI.
- The user must use an external tool (e.g. Task Manager, `kill`) to stop a running workflow.
- This blocks starting a different workflow without waiting for the current one to finish.

---

## Goals

### In Scope

1. Red Abort button visible only while a process is running.
2. Clicking Abort terminates the subprocess and shows "Aborted" in the status bar.
3. Closing the window while a process is running terminates it cleanly (no zombie processes).

### Out of Scope

- SIGKILL fallback if terminate() doesn't stop the process within a timeout.
- Progress indicators or log streaming for the running process.

---

## Success Criteria

- [ ] Run button disabled and Abort button visible while process runs.
- [ ] Clicking Abort hides the button, shows "Aborting …" then "Aborted"; Run re-enables.
- [ ] Normal completion shows "Finished (exit code 0)" with no Abort button visible.
- [ ] Closing the window with a running process terminates it cleanly.
- [ ] Double-clicking Abort is a no-op (button hidden after first click, `_process is None` guard).

---

## Technical Design

### Approach

- Add `self._aborting: bool = False` flag to `__init__`.
- Add `self._abort_button` (hidden by default) to `_build_run_bar`.
- Show/hide Abort button in `_on_run` / `_poll_process`.
- `_on_abort` sets the flag, hides the button, and calls `process.terminate()`.
- `_poll_process` branches on `_aborting` for the status message.
- `closeEvent` terminates and synchronously waits before proceeding.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `terminate()` + poll timer cleanup | Reuses existing cleanup path, minimal code | None | Chosen |
| `kill()` immediately | Guaranteed stop | Skips graceful shutdown; overkill | Rejected |
| Dedicated cleanup method | Slightly cleaner | Duplicate code paths | Rejected |

### Architecture Changes

Single file modified: `code/src/utils/gui/dag_launcher/launcher_window.py`

**State machine:**

| State | `_process` | `_aborting` | Run btn | Abort btn |
|-------|-----------|------------|---------|-----------|
| Idle | None | False | enabled | hidden |
| Running | Popen | False | disabled | visible |
| Aborting | Popen (terminating) | True | disabled | hidden |
| After complete | None | — | enabled | hidden |

---

## Implementation Notes

- `_on_abort` does NOT do cleanup — it only sets the flag and calls `terminate()`. The poll timer fires within 500 ms and handles all state reset.
- `closeEvent` uses `process.wait()` (synchronous) because the event loop is stopping; polling cannot continue.
- No `_update_run_bar` changes needed — the Abort button is not affected by workflow selection state.

---

## Knowledge Base Check

No knowledge base notes apply (subprocess termination, not Open3D/CuPy/GLFW).
