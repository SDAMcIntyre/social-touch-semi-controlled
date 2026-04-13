# Plan: Postprocess Monitor Lifecycle Fix

**Date:** 2026-03-23
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/postprocess-monitor-lifecycle`

---

## Overview

The postprocessing pipeline (`postprocess_workflow_kinect_auto.py`) hangs after completing all sessions because its `main()` creates a raw `multiprocessing.Queue` with no consumer. This fix aligns the postprocessing monitor lifecycle with the working preprocessing pipeline pattern.

## Problem Statement

When the postprocessing DAG workflow is launched from the GUI, the Python subprocess never exits after all sessions complete. The GUI stays stuck on "Running..." indefinitely.

**Root cause:** `main()` creates a bare `multiprocessing.Queue()` and passes it to worker monitors in client mode. No server-mode `PipelineMonitor` is created to drain the queue. On Windows, the undrained queue's internal feeder thread blocks interpreter shutdown.

The preprocessing pipeline (`preprocess_workflow_kinect_auto.py`) solves this correctly with a server-mode `PipelineMonitor` that owns the queue, runs a coordinator thread, shows a live dashboard, and cleans up in a `finally` block.

## Goals

### In Scope
1. Create a server-mode `PipelineMonitor` in the postprocessing `main()` to own and drain the queue
2. Show the live dashboard during postprocessing execution
3. Properly shut down the monitor in a `finally` block so the process exits cleanly
4. Read `parallel_execution` from the DAG config instead of hardcoding `False`

### Out of Scope
- Changes to `PipelineMonitor` internals
- Changes to `TaskExecutor` or `DagConfigHandler`
- Changes to the preprocessing pipeline
- Adding new monitoring features

## Success Criteria

- [ ] Postprocessing subprocess exits cleanly after all sessions complete
- [ ] GUI status bar shows "Finished (exit code 0)" instead of hanging on "Running..."
- [ ] Live dashboard window appears during postprocessing
- [ ] Timestamped `.xlsx` report is saved in `reports/`
- [ ] `parallel_execution` parameter is read from DAG config

---

## Technical Design

### Approach

Mirror the preprocessing pipeline's `main()` pattern exactly. The infrastructure (`PipelineMonitor`, `LivePlotter`, coordinator thread) already exists and works — the postprocessing script simply doesn't use it correctly.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Mirror preprocessing `main()` pattern | Reuses proven code, consistent across pipelines | None significant | **Chosen** |
| Just drain the queue at exit | Minimal change | No dashboard, no report, diverges from preprocessing | Rejected |
| Remove monitoring entirely | Simplest fix | Loses all monitoring capability | Rejected |

### Architecture Changes

No new modules or classes. Single file edit to wire existing infrastructure correctly.

---

## Implementation Plan

### Phase 1: Rewrite `main()` monitor lifecycle
**Goal:** Make postprocessing monitor lifecycle match preprocessing

**Started:** 2026-03-23
**Completed:** 2026-03-23

**Tasks:**
- [x] Task 1.1 — Add `setup_environment()` helper (mirror `preprocess_workflow_kinect_auto.py:661-673`)
- [x] Task 1.2 — Create server-mode `PipelineMonitor` with `live_plotting=True` and call `show_dashboard()`
- [x] Task 1.3 — Pass `main_monitor.queue` to `run_batch_postprocessing` instead of a raw `Queue()`
- [x] Task 1.4 — Wrap `run_batch_postprocessing` in `try/finally` with `main_monitor.close_dashboard(block=True)`
- [x] Task 1.5 — Read `parallel_execution` from DAG handler and pass to `run_batch_postprocessing`
- [x] Task 1.6 — Add 10-second sleep before dashboard close (let user see final state)

**Files Modified:**
- `code/scripts/postprocess_workflow_kinect_auto.py` — Rewrite `main()`, add `setup_environment()`

**Dependencies:** None

**Reference code (preprocessing pattern to mirror):**
- `code/scripts/preprocess_workflow_kinect_auto.py:661-711` — `setup_environment()` and `main()`
- `code/src/utils/pipeline/monitoring/pipeline_monitor.py:29-44` — server vs client mode init

---

## Testing Plan

### Manual Verification
- [ ] Launch postprocessing workflow from DAG GUI with a small config subset
- [ ] Confirm live dashboard window appears
- [ ] Confirm terminal output shows monitor status messages being consumed
- [ ] Let all sessions complete — confirm process exits cleanly
- [ ] Confirm GUI status bar transitions to "Finished (exit code 0)"
- [ ] Confirm timestamped `.xlsx` report exists in `reports/`

### Edge Cases
- [ ] Abort mid-run via GUI abort button — confirm process terminates and dashboard closes
- [ ] Run with `parallel_execution: true` in DAG config — confirm flag is respected

---

## Documentation Plan

- [ ] No documentation changes needed — this is a bug fix aligning with existing patterns

---

## Rollback Plan

1. Revert the single commit on `postprocess_workflow_kinect_auto.py`
2. No data or config changes to reverse

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Live plotter process fails to spawn | Low | Low | `PipelineMonitor` already handles this gracefully with a fallback message |
| `parallel_execution: true` path untested | Med | Low | Parallel path already exists in `run_batch_postprocessing` as a stub — reading the flag doesn't change behavior |

---

## References

- Reference implementation: `code/scripts/preprocess_workflow_kinect_auto.py:661-711`
- Monitor infrastructure: `code/src/utils/pipeline/monitoring/pipeline_monitor.py`
- Live plotter: `code/src/utils/pipeline/monitoring/pipeline_monitor_live_plotter.py`

---
