# Plan: Persistent Prefect Server at GUI Launch

**Date:** 2026-03-14
**Author:** Basil Duvernoy
**Status:** Active
**Branch:** `feature/persistent-prefect-server-at-gui-launch`

---

## Overview

Start a persistent Prefect server process when the DAG Config Launcher GUI opens, so that workflow subprocesses connect to it via `PREFECT_API_URL` instead of each initializing their own ephemeral in-process ASGI server and SQLite database. This eliminates the per-run startup delay experienced across all workflows. The server is stopped when the GUI closes.

## Problem Statement

Every time a workflow is launched from the GUI, the spawned subprocess initializes Prefect 3.x in ephemeral mode: it boots an in-process ASGI server, creates a fresh SQLite database (the previous one is wiped by `_clear_prefect_db()`), and sets up schema from scratch. This causes a noticeable multi-second delay before any actual work begins. The delay is consistent across all workflows and is independent of the workflow's own import overhead.

Additionally, the current `_clear_prefect_db()` workaround (deleting `~/.prefect/prefect.db*` before each run) exists to avoid "database is locked" errors from stale ephemeral runs, but it makes the problem worse by forcing full DB re-creation every time.

## Goals

### In Scope
1. Start a `prefect server start` background process when the GUI opens
2. Pass `PREFECT_API_URL` to workflow subprocesses so they connect to the persistent server
3. Stop the server cleanly when the GUI closes
4. Remove the `_clear_prefect_db()` workaround
5. Handle edge cases: server already running, startup failure, crash recovery, ephemeral fallback

### Out of Scope
- Optimizing Python import time in workflow scripts (separate concern)
- Deploying Prefect to a remote server or using Prefect Cloud
- Changing how workflow scripts use `@flow`/`@task` decorators
- Adding a Prefect dashboard/UI to the GUI

## Success Criteria

- [ ] Prefect server starts automatically when GUI opens, visible in Task Manager
- [ ] Status bar shows server status ("Starting...", "Ready", or failure warning)
- [ ] Workflows complete successfully using the persistent server (no ephemeral init delay)
- [ ] Server stops when GUI is closed
- [ ] If server is unavailable, workflows fall back to ephemeral mode gracefully
- [ ] No "database is locked" errors

---

## Technical Design

### Approach

Extract server lifecycle management into a dedicated `PrefectServerManager` class. The GUI's `LauncherWindow` instantiates it at init, starts the server, and stops it on close. Workflow subprocesses receive `PREFECT_API_URL` via their environment, which Prefect 3.x auto-detects — no changes to workflow scripts needed.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Persistent server at GUI launch | Eliminates per-run init; clean lifecycle; no script changes | Adds process management complexity | **Chosen** |
| Remove Prefect entirely | Zero overhead; simplest | Loses `.submit()` parallelism, flow tracking, run logging | Rejected |
| Keep ephemeral, just stop clearing DB | Minimal change | Still boots ephemeral server each run; doesn't fix root delay | Rejected |
| Lazy imports in workflow scripts | Reduces import overhead | Doesn't address Prefect init overhead; high code churn | Rejected (orthogonal) |

### Architecture Changes

New module:
```
code/src/utils/gui/dag_launcher/
├── prefect_server_manager.py   ← NEW
├── launcher_window.py          ← MODIFIED
├── ...
```

No changes to workflow scripts (`code/scripts/*.py`) — Prefect auto-detects `PREFECT_API_URL` from environment.

---

## Implementation Plan

### Phase 1: Create `PrefectServerManager`
**Goal:** Encapsulate server lifecycle in a testable, reusable class.

- [ ] Create `code/src/utils/gui/dag_launcher/prefect_server_manager.py`
- [ ] Implement `PrefectServerManager` class with:
  - `__init__(port=4200)` — store port, derive `api_url`
  - `start()` — check if server already running on port (probe health endpoint), if not spawn `subprocess.Popen([sys.executable, "-m", "prefect", "server", "start", "--host", "127.0.0.1", "--port", str(port)])`; use `CREATE_NO_WINDOW` flag on Windows
  - `wait_until_ready(timeout_seconds=30.0)` — poll `http://127.0.0.1:{port}/api/health` every 0.5s; call `QApplication.processEvents()` between polls for GUI responsiveness; return bool
  - `stop()` — `terminate()` then `wait(timeout=5)` then `kill()` if needed; no-op if server was pre-existing (not owned)
  - `is_running()` — check owned process poll or probe health endpoint
  - `api_url` property — `"http://127.0.0.1:{port}/api"`
  - `get_env()` — return `{**os.environ, "PREFECT_API_URL": self.api_url}`

**Files Created:**
- `code/src/utils/gui/dag_launcher/prefect_server_manager.py`

**Dependencies:** None

### Phase 2: Integrate into `LauncherWindow`
**Goal:** Wire the server manager into the GUI lifecycle.

- [ ] Import `PrefectServerManager` in `launcher_window.py`
- [ ] In `__init__()` (after line ~59): instantiate `self._server_manager = PrefectServerManager()`, call `self._server_manager.start()`, use `QTimer.singleShot(0, self._wait_for_prefect_server)` to defer blocking wait
- [ ] Add `_wait_for_prefect_server()` method: show status bar "Starting Prefect server...", call `wait_until_ready()`, update status bar to "Prefect server ready" or show warning dialog on failure
- [ ] Modify `_on_run()`:
  - Remove call to `self._clear_prefect_db()` (line 301)
  - Add crash recovery: if `not self._server_manager.is_running()`, attempt restart
  - Pass `env=self._server_manager.get_env()` to `subprocess.Popen()` when server is running, `env=None` otherwise (ephemeral fallback)
- [ ] Add `closeEvent()` override: call `self._server_manager.stop()` then `super().closeEvent(event)`
- [ ] Delete the `_clear_prefect_db()` static method entirely

**Files Modified:**
- `code/src/utils/gui/dag_launcher/launcher_window.py`

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Launch GUI — confirm `prefect server start` process appears in Task Manager and status bar shows "Prefect server ready"
- [ ] Run a workflow — confirm it completes successfully without the usual startup delay
- [ ] Close GUI — confirm server process disappears from Task Manager
- [ ] Pre-start server manually in terminal — launch GUI — confirm it detects and reuses existing server — close GUI — confirm manually-started server is still running
- [ ] Kill server process mid-session — click Run — confirm GUI restarts the server or falls back to ephemeral mode

### Edge Cases
- [ ] Port 4200 occupied by non-Prefect process — GUI warns and workflows use ephemeral fallback
- [ ] GUI force-killed — next GUI launch detects orphaned server via health check and reuses it
- [ ] Multiple consecutive workflow runs — all succeed without DB lock errors

---

## Documentation Plan

- [ ] No README/CLAUDE.md changes needed (internal GUI improvement)
- [ ] Add inline docstrings to `PrefectServerManager` class and methods

---

## Rollback Plan

1. Revert the two files (`launcher_window.py` changes + delete `prefect_server_manager.py`)
2. `_clear_prefect_db()` and ephemeral mode are fully restored
3. No data migrations or breaking changes — purely additive process management

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Server fails to start (port conflict, missing deps) | Low | Low | Ephemeral fallback — workflows still work, just slower |
| Orphaned server after GUI crash | Medium | Low | Next GUI launch detects via health check and reuses |
| Server crash mid-workflow | Low | Medium | Workflow fails with connection error; user re-runs; `_on_run()` attempts restart |
| `CREATE_NO_WINDOW` flag behavior differences across Windows versions | Low | Low | Flag is well-established Win32 API; guard with `sys.platform` check |

---

## References

- Prefect 3.x server docs: `prefect server start` CLI
- Current launcher code: `code/src/utils/gui/dag_launcher/launcher_window.py` (lines 284-314)
- Prefect version: 3.4.22 (pinned in `requirements.txt`)
