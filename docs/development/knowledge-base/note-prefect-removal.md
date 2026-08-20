# Dev Note: Prefect removal — a shared `PREFECT_HOME`, a swallowed error, and a silenced root logger

**Problem class:** A workflow orchestrator that contributed no runtime feature
this repo used, but did contribute a server, a shared SQLite database, a
cross-repository version coupling, and a root log handler that silently
disabled every entry script's `logging.basicConfig`.

| Field | Value |
|-------|-------|
| Resolved in | `code/scripts/` (10 entry scripts), `code/src/utils/gui/dag_launcher/` |
| Affected versions | `prefect` 3.x (pinned `3.4.22`, installed `3.6.22`) alongside the sibling analysis repo's `3.7.5` |
| Platform | Windows 11 |
| Plan | [remove-prefect-orchestration.md](../plans/active/remove-prefect-orchestration.md) |
| Superseded design | [persistent-prefect-server-at-gui-launch.md](../plans/completed/persistent-prefect-server-at-gui-launch.md) |

---

## 1. Symptom

Three symptoms, one dependency.

**A — the GUI froze for 30 s at launch and did nothing useful afterwards.**
`launch_pipeline_gui.py` took 32.40 s to become interactive. The status bar
ended up reading *"Prefect server unavailable — workflows will use ephemeral
mode"*, with no indication anywhere of why. Measured cause: a 30 s readiness
wait on a server that had already died.

**B — the same 30 s wait on every Run click**, because the Run handler
restarted the server if it was not healthy.

**C — `logger.info(...)` produced no output** in two entry scripts
(`preprocess_pipeline_nerve_auto.py`,
`merging_pipeline_neuron_to_kinect_auto.py`), even though both called
`logging.basicConfig(level=logging.INFO, ...)`. No error, exit code 0, just
silence.

---

## 2. Investigation

**A/B.** The server *was* being spawned — `Started Prefect server (PID 20812)
on port 4200` appeared in the log — and then never became ready. Running the
same `prefect server start` command by hand, with stdout attached, produced a
one-line answer immediately:

```
Can't locate revision identified by '79e7a60e43d8'
```

`PREFECT_HOME` defaults to `~/.prefect`. That directory is **per user, not per
project**, so this repo and the sibling `social-touch-semi-controlled-analysis`
repo shared one `~/.prefect/prefect.db`. The analysis repo runs Prefect 3.7.5;
opening the database there ran an alembic upgrade to revision `79e7a60e43d8`.
This repo's 3.6.22 has no such revision in its migration tree, so its server
aborted at startup. Whichever repo ran last won.

The reason this cost most of a day rather than a minute is
`prefect_server_manager.py:77-78`:

```python
stdout=subprocess.DEVNULL,
stderr=subprocess.DEVNULL,
```

The subprocess that knew the answer was told to discard it. The failure
therefore presented as a freeze, not a message.

**C.** Traced by inspecting the root logger's state at import time.
`from prefect import flow` runs `prefect/main.py:50 → setup_logging() →
logging.config.dictConfig(...)` with `root: {level: WARNING, handlers:
[console]}`. In all six entry scripts the `prefect` import sits **above** the
`basicConfig` call, so by the time `basicConfig` runs, `root.handlers` is
non-empty — and `basicConfig` is documented to be a **no-op** when the root
logger already has handlers. Verified directly: root stayed at `WARNING` with a
`PrefectConsoleHandler` attached. Every module-level `logger.info` in those
scripts had been dead the whole time; only `get_run_logger()` narration was
reaching the console.

---

## 3. Root Cause

Three independent defects with one common origin — a dependency that owned
global process state it did not need to own:

1. **Global, user-scoped configuration.** `PREFECT_HOME` is a per-user default,
   so two projects on one machine share a schema-versioned database. Any
   version skew between sibling repositories bricks the older one, with no
   isolation and no warning.
2. **Error output routed to `DEVNULL`.** A managed child process whose stdout
   and stderr are discarded converts every one-line diagnosis into a timeout.
3. **Import-time mutation of the root logger.** A library that calls
   `dictConfig` at import time silently overrides the application's logging
   policy, and does so *invisibly* because `basicConfig` fails quietly rather
   than raising.

What made all three intolerable rather than merely annoying: **the repo used
almost none of Prefect.** Of 55 decorators, exactly four carried any argument
beyond `name`, all `log_prints=True`. No retries, timeouts, task runners,
result persistence, cache keys or deployments existed anywhere in the tree. The
only genuine runtime feature in use — parallel execution — was dead code: two
of the three `.submit()` call sites targeted functions that were not decorated
at all and would have raised `AttributeError`, and the third branch was a bare
`pass`. `parallel_execution: false` in all seven config files, with no GUI
control to change it.

The actual orchestration — `DagConfigHandler`, `TaskExecutor`,
`PipelineMonitor`, `should_process_task` — was already the repo's own and did
not touch Prefect.

---

## 4. Architecture Constraints

- **A per-project `PREFECT_HOME` fixes only symptom A/B.** It was tried first
  and verified working
  (`conda env config vars set PREFECT_HOME=C:\Users\basil\.prefect-social-touch -n social-touch`).
  It leaves the server lifecycle, the readiness waits, the version constraint
  and the root-logger override untouched.
- **Pinning both repos to one Prefect version is not durable.** It makes an
  unrelated repository's upgrade cadence a hard constraint on this one, across
  a file neither repo declares as an interface. The pin was already drifting:
  `requirements.txt` said `3.4.22` while the environment had `3.6.22`.
- **`basicConfig(..., force=True)` fixes symptom C but changes behaviour.**
  Measured: it un-suppresses `httpx` INFO, adding ~10
  `HTTP Request: POST .../api/flow_runs/` lines per flow run to the console.
- **Losing `log_prints=True` re-exposes stdout block buffering.** Prefect was
  incidentally keeping the GUI console live; a child Python process writing to
  a pipe block-buffers by default.

---

## 5. Fix Applied

Prefect was **removed, not pinned**. The reasoning is a cost/benefit with no
close call: the benefit was four non-default decorator arguments; the cost was
a server process, a shared schema-versioned database, a cross-repository
version coupling, two 30 s waits per session, and a silent logging override.
Pinning keeps the entire cost to preserve none of the benefit.

Removal in six phases (see the plan for detail):

| Phase | Change |
|-------|--------|
| 1 | Add `basicConfig` + module `logger` to the two scripts lacking them; add `-u` to the GUI's workflow subprocess |
| 2 | Delete the dead parallel branches; `parallel_execution: true` now raises `NotImplementedError` |
| 3 | Remove 55 decorators, 10 `from prefect import` lines, 10 `get_run_logger()` calls |
| 4 | Delete `prefect_server_manager.py` (143 lines) and its 6 wiring sites in `launcher_window.py` |
| 5 | Drop the dependency from `requirements.txt` / `environment.yml`; update docs |
| 6 | Verification against a real run |

Two ordering constraints were load-bearing:

- **Phase 1 comes first, while Prefect is still present.** The `basicConfig`
  and `-u` fixes are improvements on their own merits, and having them in place
  means later phases are judged against a console that already behaves.
- **Within Phase 3, the `prefect` import and the `get_run_logger()` calls must
  be deleted in the *same* edit per file.** Deleting the calls first leaves a
  window where `basicConfig` is still inert (Prefect still imported) *and*
  `get_run_logger` is gone — narration goes silent with no error.

The `basicConfig` fix landed for free: once the `prefect` import is gone,
`root.handlers` is empty and `basicConfig` sets `INFO` + a stderr
`StreamHandler`. Measured before/after on both affected scripts — before,
`logger.info(...)` produced nothing; after, the expected `INFO` line appeared on
stderr.

**Measured result:** GUI launch-to-interactive **32.40 s → 1.28 s (−96%)**.
Imports account for 0.25–0.37 s in both runs; the entire 31 s delta was the
server wait.

---

## 6. Reusable Pattern

**Before adopting or keeping a framework that owns process-global state, price
it against what you actually use.**

- [ ] **Count the features you use, not the features it has.** Grep for every
  symbol imported from it and every non-default argument passed. If the answer
  is "decorators with a `name=`", it is documentation, not orchestration.
- [ ] **Find its global state.** Does it default to a per-user directory
  (`~/.<tool>`)? A shared database? An environment variable? If two projects on
  one machine can collide through it, they eventually will — and the failure
  will be attributed to the innocent project, since the guilty one still works.
- [ ] **Check whether it configures logging at import time.** `import x; ...;
  logging.basicConfig(...)` is a no-op if `x` attached a root handler. Test:
  `import logging, x; print(logging.root.level, logging.root.handlers)`.
  Symptom is silence, not error.
- [ ] **Never route a managed subprocess's stderr to `DEVNULL`.** Capture it and
  surface it on failure. `DEVNULL` turns one-line diagnoses into timeouts.
- [ ] **When a version-skew bug appears, ask whether the coupling should exist
  at all** before pinning. A pin makes another team's release cadence your
  constraint.
- [ ] **When removing a logging-adjacent dependency, harden logging first, in a
  separate commit**, so a narration regression cannot hide inside the removal
  diff.

---

## 7. References

| Document | Location |
|----------|----------|
| Removal plan (all six phases, with measurements) | `docs/development/plans/active/remove-prefect-orchestration.md` |
| Changelog, incl. surviving prose mentions | `docs/changelogs/remove-prefect-orchestration.md` |
| Superseded design record for what was deleted | `docs/development/plans/completed/persistent-prefect-server-at-gui-launch.md` |
| Repo orchestration that replaces Prefect's role | `code/src/utils/pipeline/pipeline_config_manager.py`, `code/src/utils/pipeline/task_executor.py`, `code/src/utils/pipeline/monitoring/pipeline_monitor.py`, `code/src/utils/should_process_task.py` |
| Regression test for the dead parallel branches | `code/tests/test_parallel_execution_removed.py` |
