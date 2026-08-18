# Prefect was removed from this repository

**Date:** 2026-08-18
**Branch:** `feature/remove-prefect-orchestration`
**Plan:** `docs/development/plans/active/remove-prefect-orchestration.md`
**Knowledge-base note:** `docs/development/knowledge-base/note-prefect-removal.md`

This repository no longer depends on Prefect. Workflows in `code/scripts/` are
plain Python entry-point functions. Task ordering is resolved by
`DagConfigHandler` and execution is wrapped by `TaskExecutor` — as it already
was; that layer never used Prefect.

## What changed

| Phase | Change |
|-------|--------|
| 1 | `logging.basicConfig` + a module-level `logger` added to `preprocess_pipeline_nerve_auto.py` and `merging_pipeline_neuron_to_kinect_auto.py`; `-u` added to the GUI's workflow subprocess |
| 2 | Dead parallel-execution branches deleted in the four batch runners; `parallel_execution: true` now raises `NotImplementedError` |
| 3 | 55 `@flow`/`@task` decorators, 10 `from prefect import` lines and 10 `get_run_logger()` calls removed from the 10 entry scripts |
| 4 | `prefect_server_manager.py` deleted (143 lines) and all six of its wiring sites removed from `launcher_window.py` |
| 5 | `prefect` dropped from `requirements.txt` and `environment.yml`; docs updated |

**Measured:** GUI launch-to-interactive **32.40 s → 1.28 s (−96%)**. The whole
delta was a 30 s wait on a Prefect server that could not start.

## What did *not* change

- `DagConfigHandler`, `TaskExecutor`, `PipelineMonitor`, `should_process_task` —
  untouched.
- Every DAG YAML, including the `parallel_execution` key, which is still read so
  existing configs stay valid.
- Task ordering, skip logic, failure reporting, the monitor dashboard, and the
  Excel status report.
- `pydantic-settings` (`requirements.txt:40`, `environment.yml:52`). It was a
  Prefect transitive dependency and **nothing in the repository imports it**, so
  it is now orphaned. Removing it was explicitly out of scope for this work —
  see "Follow-ups" below.

## Parallel execution

It is gone, and it was never alive. Two of the three `.submit()` call sites
targeted functions that carried no decorator at all and would have raised
`AttributeError`; the third branch was a bare `pass` that silently processed
zero sessions. All seven config files shipped `parallel_execution: false`, and
the GUI offered no way to change it.

The key is still read. Setting it to `true` now raises `NotImplementedError`
as the first statement of the batch runner, before any config is loaded.
Covered by `code/tests/test_parallel_execution_removed.py` (8 tests).

## Surviving "prefect" mentions — all deliberate

A grep for `prefect` still returns hits. **None of them are leftovers.** This
is the complete list as of this change, from:

```bash
grep -rn "prefect" . -i --include=*.py --include=*.txt --include=*.yml \
     --include=*.yaml --include=*.toml --exclude-dir=.git
```

| Location | Why it stays |
|----------|--------------|
| `code/scripts/merging_pipeline_neuron_to_kinect_auto.py:286,289` | Text of the `NotImplementedError` raised on `parallel_execution: true`. Explains *why* the feature is absent and points at the plan. Prose in a string literal |
| `code/scripts/postprocess_workflow_kinect_auto.py:429,432` | Same message, same reason |
| `code/scripts/preprocess_workflow_kinect_auto.py:668,671` | Same message, same reason |
| `code/scripts/primary_workflow_kinect_auto.py:132,135` | Same message, same reason |
| `code/tests/test_parallel_execution_removed.py:3,34` | Docstring and `PLAN_REFERENCE` constant naming the plan this test enforces |
| `code/tests/test_parallel_execution_removed.py:54` | `getattr(fn, "fn", fn)  # unwrap a prefect @flow if one is still present`. A no-op on plain functions; kept so the test survives being run against an older checkout |
| `code/src/semi_controlled.egg-info/SOURCES.txt:335` | **Generated build artifact, untracked and gitignored** (`.gitignore:23`). Lists the deleted `prefect_server_manager.py` because the egg-info predates the deletion. Disappears on the next `pip install -e .`. Not a source file |

Nothing above is an import, a decorator, a call, or a dependency declaration.
Under `code/src/` there is **no** occurrence of the string at all any more.

Documentation outside this table also mentions Prefect — the plan, the
knowledge-base note, this changelog, and roughly seventy historical plan
documents under `docs/development/plans/completed/`. Those are dated records of
what the repository was at the time they were written and are deliberately not
rewritten. `docs/development/plans/completed/persistent-prefect-server-at-gui-launch.md`
carries a superseded-by header pointing here.

## Follow-ups

- **`pydantic-settings` is orphaned.** Verified: no `import pydantic_settings`,
  no `from pydantic_settings ...`, and no `BaseSettings` subclass anywhere in
  the tree. It survives only as an explicit pin in both dependency files. Out of
  scope here; remove it in a dependency-hygiene pass.
- **The four `NotImplementedError` messages and the test's `PLAN_REFERENCE`
  point at `docs/development/plans/active/...`.** When the plan moves to
  `completed/`, those five strings go stale. `test_parallel_execution_removed.py`
  asserts the constant appears in each message, so the test will keep them
  consistent with each other but not with the filesystem.
- **`~/.prefect` and `~/.prefect-social-touch` are now inert.** They were not
  deleted — they are user-level directories shared with the sibling analysis
  repo, which still uses Prefect.
