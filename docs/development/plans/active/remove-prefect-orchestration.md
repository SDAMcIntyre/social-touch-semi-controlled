# Plan: Remove Prefect Orchestration

**Date:** 2026-08-18
**Created:** 2026-08-18 11:00
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/filter-contact-depth-field-by-neural-quality`
**Branch:** `feature/remove-prefect-orchestration`

---

## Overview

Delete Prefect from the repository. It contributes 55 decorators, one logger and a server that must
be alive before anything runs; the orchestration that does the actual work — `DagConfigHandler`,
`TaskExecutor`, `PipelineMonitor`, `should_process_task` — is already the repo's own. The removal is
mostly deletion, confined to `code/scripts/` and one GUI module, with **zero changes under
`code/src/`** apart from deleting `prefect_server_manager.py`.

## Problem Statement

Prefect is a hard runtime dependency that provides almost nothing this pipeline uses, and it has
become an active obstacle.

**What it actually provides.** Ten entry-point files import it. Three symbols are used: `flow` (52
decorators), `task` (3), `get_run_logger` (10 call sites, plus one dead import at
`preprocess_workflow_kinect_auto.py:10`). Of 55 decorators, exactly **four** carry any argument
beyond `name`, all `log_prints=True`. There are no retries, timeouts, task runners, result
persistence, cache keys or deployments anywhere in the tree. Every flow return is a plain Python
value; no `State` is inspected outside dead code.

**What it costs.** A local server must start before workflows run. The GUI blocks up to 30 s at
launch waiting for it (`launcher_window.py:70,297`) and up to 30 s again on every Run click
(`:330-336`). `PREFECT_HOME` defaults to `~/.prefect`, shared with the sibling analysis repo, whose
newer Prefect migrated the shared SQLite database to alembic revision `79e7a60e43d8`. This repo's
Prefect 3.6.22 cannot read that revision, so the server dies at startup — and
`prefect_server_manager.py:77-78` sends its stdout and stderr to `DEVNULL`, so a one-line
`Can't locate revision identified by '79e7a60e43d8'` presented to the user as a silent freeze.
Working around it cost most of a day on 2026-08-18.

**How little is load-bearing.** The parallel-execution path — the only genuine Prefect runtime
feature in use — is dead. Two of the three `.submit()` sites target functions that are **not
decorated at all** (`preprocess_workflow_kinect_auto.py:465` has its `@flow` commented out;
`primary_workflow_kinect_auto.py:66` never had one), so they would raise `AttributeError`
immediately. The third workflow's parallel branch is a literal `pass`
(`postprocess_workflow_kinect_auto.py:465-467`). `parallel_execution: false` in **all seven** config
files including the template, and the GUI offers no way to set it.

That someone already commented out `@flow` on `run_single_session_pipeline`
(`preprocess_workflow_kinect_auto.py:465`) without regression is the strongest evidence that the
decorators are decoration.

## Goals

### In Scope

1. Remove `prefect` from every import, decorator and call site in `code/scripts/`.
2. Delete `code/src/utils/gui/dag_launcher/prefect_server_manager.py` and its wiring in
   `launcher_window.py`, so the GUI starts and runs with no server lifecycle at all.
3. Delete the dead parallel-execution branches and the `PrefectFuture` import.
4. Replace the 10 `get_run_logger()` call sites with the repo's standard `logging` pattern —
   **including the `basicConfig` calls the two affected files are currently missing** (see Risks).
5. Fix child-process stdout buffering in the GUI so console output stays live once `log_prints=True`
   is gone.
6. Drop `prefect` from `requirements.txt` and `environment.yml`.
7. Keep the pipeline's observable behaviour equivalent: same task ordering, same skip logic, same
   failure reporting, same monitor dashboard, same Excel status report.

### Out of Scope

- **Reimplementing parallel execution.** It is dead code today (see Problem Statement). The
  `parallel_execution` YAML key is *retained and still read* so existing configs stay valid, but the
  branch it guards raises `NotImplementedError` instead of pretending to work. Bringing it back with
  `ProcessPoolExecutor` is a separate plan — and a hard one, see the note below.
- **Replacing Prefect with another orchestrator.** The repo already has its own; adding a second
  would repeat the mistake.
- **Changing `DagConfigHandler`, `TaskExecutor`, `PipelineMonitor`, `should_process_task`,** or any
  DAG YAML schema beyond removing the dependency.
- **Any change under `code/src/` other than deleting `prefect_server_manager.py`.** Nothing else
  there imports Prefect — confirmed by grep across the whole package tree.
- **Removing `pydantic-settings`** (`environment.yml:55`). It is a Prefect transitive dependency but
  is declared independently; check whether anything still needs it, but do not remove it here.
- **Deleting `~/.prefect` or `~/.prefect-social-touch`.** Not ours to delete; they become inert.
- **Retro-fixing the GUI's `DEVNULL` error swallowing** — the code that does it is being deleted.

### Success Criteria

- [ ] `grep -rn "prefect" code/ --include=*.py -i` returns only prose comments (no imports, no
      decorators, no calls). Enumerate the survivors in the changelog.
- [ ] `pip uninstall prefect` (or an env without it) leaves every entry script importable and every
      workflow runnable.
- [ ] The full test suite passes at no fewer than its pre-change count (currently **320 passed,
      7 skipped**).
- [ ] The GUI opens with **no** Prefect server, and Run starts a workflow with no readiness wait.
      Measure launch-to-interactive before and after.
- [ ] Console output in the GUI appears **line by line** during a long-running workflow, not in
      bursts — verified on a workflow that prints steadily.
- [ ] `preprocess_pipeline_nerve_auto.py` and `merging_pipeline_neuron_to_kinect_auto.py` still emit
      their `INFO` narration. This is the headline regression risk; assert it explicitly, do not
      assume it.
- [ ] One real workflow runs end to end and produces byte-identical outputs to a pre-change run of
      the same inputs (hash-compared).
- [ ] `parallel_execution: true` raises `NotImplementedError` loudly rather than silently doing
      nothing or crashing with `AttributeError`.
- [ ] `prefect` appears in no dependency file.

## Definitions

- **Removed**: no `import prefect`, no `from prefect ...`, no `@flow`/`@task`, no `get_run_logger`,
  no `PrefectFuture`, no `PREFECT_*` environment variable set by repo code. Prose comments that
  merely *mention* Prefect may remain if still accurate.
- **Behaviourally equivalent**: for the same inputs and the same DAG config, the sequence of tasks
  executed, the artifacts written, and their bytes are unchanged. Log *text* may differ; log
  *content* (which tasks ran, which failed) may not.
- **Narration preserved**: every `logger.info(...)` that reaches the console today still reaches it
  afterwards. Testable by capturing stdout+stderr of a script run and asserting expected lines.
- **Dead parallel branch**: any `if parallel:` block whose body calls `.submit()` on an undecorated
  function, or is a bare `pass`. Three of the four qualify today.

---

## Technical Design

### Approach

Fix the two silent-degradation hazards **first**, while Prefect is still in place and its behaviour
is available as a reference. Only then delete. That ordering means every later phase can be judged
against a console that already behaves correctly, instead of debugging buffering and logging changes
tangled together with the removal.

```
Phase 1  logging + buffering hardening      (Prefect still present; no behaviour lost)
Phase 2  delete dead parallel branches      (removes the only real runtime feature — already broken)
Phase 3  strip decorators + get_run_logger  (10 entry scripts; mechanical)
Phase 4  delete the GUI server integration  (8 edit sites + 1 file)
Phase 5  drop the dependency + docs
Phase 6  verification against a real run
```

Phase 1 is deliberately *additive and independently valuable*: adding `basicConfig` to two scripts
that lack it, and `-u` to the GUI's child process, are improvements whether or not Prefect is
removed. If the removal were abandoned after Phase 1, the repo would be strictly better off.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Remove Prefect entirely; keep the repo's own orchestration** | Deletes a server, a database, a version constraint and a whole failure class; ~55 decorators become plain functions; no `code/src/` changes | One-off migration risk, concentrated in logging | **Chosen** |
| Keep Prefect, fix the shared-DB collision with a per-project `PREFECT_HOME` | Already done on 2026-08-18 and verified working; zero code change | Keeps the server lifecycle, the 30 s waits, the version constraint, and a dependency providing 4 non-default decorator arguments in total. Treats the symptom | Rejected as the end state — but it is the fallback if this plan is abandoned mid-way |
| Disable the persistent server only; let flows use Prefect's ephemeral mode | Small change; `launcher_window.py:341` already degrades to `env=None` | Still Prefect, still the same database, still the same alembic failure — ephemeral mode is where it was first seen. Fewer benefits, same exposure | Rejected |
| Replace Prefect with another orchestrator (Luigi, Dagster, plain Airflow) | Feature parity on paper | The repo does not use the features it already has. Adding a second orchestrator alongside `DagConfigHandler` repeats the original mistake | Rejected |
| Remove Prefect **and** port parallel execution to `ProcessPoolExecutor` | Restores a capability | The capability has never worked. Porting is genuinely hard: `PipelineMonitor.queue` is a `multiprocessing.Queue` that cannot be passed as a pool argument (verified: `RuntimeError: Queue objects should only be shared between processes through inheritance`); `KinectConfig.__getattr__` makes picklability version-dependent; `TaskExecutor` can raise Qt popups whose main-thread guard stops working in a subprocess; N workers means N CUDA contexts | Rejected — separate plan if ever wanted |

### Architecture & Module Contracts

No new modules. The change is subtractive, and the contracts below are what must hold **after**
deletion.

| Module / layer | Responsibility | Inputs → Outputs | Must NOT know about |
|----------------|----------------|------------------|---------------------|
| `code/scripts/*_workflow_*.py`, `*_pipeline_*.py` (10 files) | Entry points: resolve configs, order stages, call stage functions | CLI `--dag-config` → artifacts on disk | Prefect, any orchestrator, the GUI |
| `DagConfigHandler` (unchanged) | Task enablement + dependency resolution | DAG YAML → `can_run` / `mark_completed` | Unchanged |
| `TaskExecutor` (unchanged) | Per-task state, timing, exception capture | task name + monitor queue → state transitions | Unchanged |
| `PipelineMonitor` (unchanged) | Live dashboard + Excel status report | `multiprocessing.Queue` → plot process + xlsx | Unchanged; already Prefect-free (one docstring mention) |
| `launcher_window.py` (modified) | Launch a workflow subprocess, stream its output | script + dag-config → `Popen`, console lines, exit code | Prefect, server lifecycle, API URLs |
| `prefect_server_manager.py` | **DELETED** | — | — |

```
code/scripts/                       # 10 files: decorators + imports + parallel branches removed
code/src/utils/gui/dag_launcher/
├── prefect_server_manager.py       # DELETED (144 lines)
└── launcher_window.py              # MODIFIED — 8 edit sites
requirements.txt                    # MODIFIED — drop prefect==3.4.22 and its section header
environment.yml                     # MODIFIED — drop prefect>=3.4 and its section header
```

### Constraints and findings from the investigation

- **`get_run_logger` is used in exactly two files**, and *neither* configures logging:
  `preprocess_pipeline_nerve_auto.py` (5 sites) and `merging_pipeline_neuron_to_kinect_auto.py`
  (5 sites). Four sibling entry scripts already call
  `logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')`.
  Copy that. See Risks — this is the single highest-risk item in the plan.
- **The GUI console reader parses nothing Prefect-specific.** `process_output_reader.py` strips ANSI
  escapes and handles tqdm `\r`; there is no regex or `startswith` keyed on `Flow run`, `INFO |`, or
  any Prefect string. Console *content* changes; console *parsing* cannot break.
- **Run identity is already carried by the messages themselves.** Nearly every line is prefixed with
  `[{block_name}]` or `[{output_dir.name}]`, so losing Prefect's adjective-animal run names costs
  nothing operationally.
- **Nested `@flow` calls are direct calls.** In Prefect 3 a directly-called subflow propagates the
  original exception to the parent, exactly as a plain call does — so every existing
  `try/except Exception` block keeps working verbatim. No code depends on subflow semantics.
- **Prefect currently coerces arguments against type annotations.** Plain functions will not. Shipped
  configs are correctly typed today, and two defensive re-coercions already exist
  (`primary_workflow_kinect_auto.py:53`, `preprocess_workflow_kinect_auto.py:460`), suggesting the
  author never fully relied on it. Low probability, silent when it bites — noted in Risks.
- **No `prefect.yaml`, `.prefectignore`, or `prefect.toml` exists**; `pyproject.toml` does not list
  prefect; no test references it; `PREFECT_API_URL` is the only `PREFECT_*` variable and it is set in
  exactly one place.

---

## Implementation Plan

### Phase 1: Harden logging and output buffering
**Goal:** The two silent-degradation hazards are fixed while Prefect is still present, so later
phases cannot hide behind them.
**Started:** 2026-08-18 10:45  **Completed:** 2026-08-18 11:05

- [x] 1.1 — Add the repo-standard
      `logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')`
      and a module-level `logger = logging.getLogger(__name__)` to
      `code/scripts/preprocess_pipeline_nerve_auto.py` and
      `code/scripts/merging_pipeline_neuron_to_kinect_auto.py`, matching the four sibling scripts.
- [x] 1.2 — Verify with Prefect **still in place** that narration is unchanged: capture a run's
      output before and after 1.1 and diff the set of INFO lines.
      *Done as the targeted import-and-call substitute (a full workflow run needs session data):
      both modules imported, root-logger state inspected, and `logging.getLogger(<module>).info(...)`
      emitted before and after. Prefect narration via `get_run_logger()` is byte-for-byte unchanged.
      See the finding below — the `basicConfig` is currently inert and only takes effect in Phase 3.*
- [x] 1.3 — Add `-u` to the workflow subprocess command at
      `code/src/utils/gui/dag_launcher/launcher_window.py:338` (or set `PYTHONUNBUFFERED=1` in the
      child env). Prints outside flows already block-buffer today, so this is a fix on its own merits.
      *Chose `-u`: it keeps the decision visible at the call site and survives the `env=None` branch
      two lines below, which `PYTHONUNBUFFERED` in a constructed env would not.*
- [x] 1.4 — Verify console output arrives line by line in the GUI on a workflow that prints steadily.
      *Done with a `subprocess.Popen` harness mimicking `_on_run` + `process_output_reader`'s
      `for raw_line in process.stdout`, not by launching the GUI. A child printing 8 lines at 0.4 s
      intervals: without `-u` all 8 arrive together at +3.30 s (process exit); with `-u` they arrive
      at +0.05 s … +2.89 s, one per interval.*
- [x] 1.5 — Remove the dead `get_run_logger` import at `preprocess_workflow_kinect_auto.py:10`.

**Finding — `basicConfig` in every entry script is inert while Prefect is imported.**
`from prefect import ...` triggers `prefect/main.py:50 → setup_logging() → logging.config.dictConfig`
with `root: {level: WARNING, handlers: [console]}`. Because that import sits *above* the
`basicConfig` line in all six entry scripts, `basicConfig` then sees a non-empty `root.handlers` and
returns as a no-op (verified: root stays `WARNING` with a `PrefectConsoleHandler`). This is
pre-existing and affects the four sibling scripts equally — it is **not** introduced by 1.1.

Consequences:
- Nothing regresses now. `get_run_logger()` narration is untouched, and module-level `logger.info`
  was already silent before 1.1.
- The fix lands automatically in **Phase 3/5**: with the `prefect` import gone, `root.handlers` is
  empty and `basicConfig` sets `INFO` + a stderr `StreamHandler` (verified directly).
- **Phase 3 ordering constraint:** the `logger = get_run_logger()` lines in a file must not be
  deleted while that file — or anything it imports — still pulls in `prefect`, or its narration
  goes silent in the window between. Delete the `prefect` import in the same edit.
- `basicConfig(..., force=True)` would make it effective immediately, but it was **not** applied:
  it deviates from the sibling pattern and changes behaviour today — measured, it un-suppresses
  `httpx` INFO, adding ~10 `HTTP Request: POST .../api/flow_runs/` lines per flow run to the console.
  That noise disappears with Prefect, so waiting for Phase 3 is the cheaper path.

**Files Modified:**
- `code/scripts/preprocess_pipeline_nerve_auto.py` — add logging config
- `code/scripts/merging_pipeline_neuron_to_kinect_auto.py` — add logging config
- `code/scripts/preprocess_workflow_kinect_auto.py` — drop the dead import
- `code/src/utils/gui/dag_launcher/launcher_window.py` — unbuffered child process

**Dependencies:** None

### Phase 2: Delete the dead parallel branches
**Goal:** The only genuine Prefect runtime feature is gone, and enabling it fails loudly.
**Started:** 2026-08-18 10:44  **Completed:** 2026-08-18 10:58

- [x] 2.1 — `merging_pipeline_neuron_to_kinect_auto.py`: delete the `if parallel:` submit/collect
      logic (`:301-316`, `:321-330`) and the `PrefectFuture` import (`:9`).
      *The `futures_or_states` accumulator went with it — it was read only by the collect loop.
      The sequential call and its `try/except` around config loading are unchanged.*
- [x] 2.2 — `preprocess_workflow_kinect_auto.py`: delete the parallel branch (`:678-698`,
      `:700-704`). It raises `AttributeError` today — the target is undecorated.
- [x] 2.3 — `primary_workflow_kinect_auto.py`: delete the parallel branch (`:137-151`, `:153-155`).
      Same defect.
- [x] 2.4 — `postprocess_workflow_kinect_auto.py`: replace the `pass` stub (`:465-467`).
- [x] 2.5 — In all four, **keep reading `parallel_execution`** so existing configs remain valid, but
      `raise NotImplementedError` with a message pointing at this plan when it is `true`. Silently
      processing zero sessions — today's postprocess behaviour — is the failure mode to eliminate.
      *The guard is the first statement of each batch runner, before any config is loaded, so
      nothing is processed before it fires. Identical message in all four files.*
- [x] 2.6 — Leave the `parallel_execution` key in all seven YAML files untouched.
      *Verified: all seven still read `parallel_execution: false`; `git diff -- configs/` shows no
      change to any of them.*

**Test added:** `code/tests/test_parallel_execution_removed.py` (8 tests). Two layers:
an `ast` pass asserting the guard is the first statement of each batch runner with the plan
reference in its message (no third-party imports, so it runs under the stubbed unit-test
environment), and a behavioural pass that actually calls all four runners with `parallel=True`.
The behavioural pass runs in a **clean child interpreter** because `conftest.py` stubs the real
`utils` package that the four entry scripts import; it unwraps a `@flow` via `.fn` so it does not
drive Prefect, and skips with an explicit reason if a module cannot be imported.

**Files Modified:**
- `code/scripts/merging_pipeline_neuron_to_kinect_auto.py`
- `code/scripts/preprocess_workflow_kinect_auto.py`
- `code/scripts/primary_workflow_kinect_auto.py`
- `code/scripts/postprocess_workflow_kinect_auto.py`
- `code/tests/test_parallel_execution_removed.py` — new

**Dependencies:** Phase 1

### Phase 3: Strip decorators and `get_run_logger`
**Goal:** No `code/scripts/` file imports Prefect.
**Started:** 2026-08-18 10:55  **Completed:** 2026-08-18 11:06

Work file by file, committing per file is acceptable within the phase if the diff is large. The
`@flow` names carry the stage numbering ("8. Generate Somatosensory Characteristics") — preserve that
information in a comment or docstring where it is not already obvious from the function name.

- [x] 3.1 — `preprocess_workflow_kinect_auto.py` (18 flows).
- [x] 3.2 — `preprocess_workflow_kinect_manual.py` (11 flows, one with `log_prints=True`).
- [x] 3.3 — `postprocess_workflow_kinect_auto.py` (7 flows).
- [x] 3.4 — `preprocess_pipeline_nerve_auto.py` (5 flows, 5 `get_run_logger`). Delete each
      `logger = get_run_logger()` line — the Phase 1 module-level `logger` shadows it, so every
      `logger.info(...)` body stays byte-identical.
- [x] 3.5 — `merging_pipeline_neuron_to_kinect_auto.py` (2 flows, 3 tasks, 5 `get_run_logger`). Same
      treatment.
- [x] 3.6 — `primary_workflow_kinect_auto.py` (3 flows).
- [x] 3.7 — `prepare_configs_workflow.py` (2 flows).
- [x] 3.8 — `preprocess_workflow_kinect_visualisation.py` (2 flows, one `log_prints=True`).
- [x] 3.9 — `merging_pipeline_neuron_to_kinect_visualisation.py` (1 flow, `log_prints=True`).
- [x] 3.10 — `postprocess_visualization.py` (1 flow, `log_prints=True`).
- [x] 3.11 — After each file, confirm it still imports and `--help` still works.

**Outcome — 55 decorators, 10 `from prefect import` lines, 10 `get_run_logger()` calls removed**
(18 + 11 + 7 + 5 + 5 + 3 + 2 + 2 + 1 + 1 = 55; counts matched the plan exactly). The `prefect`
import and the `get_run_logger()` calls were removed in the **same edit** per file, so narration
never went through a silent window.

Stage numbering was preserved as a leading `# Stage N: ...` comment for the 26 decorators whose
`name=` carried information the function name does not — all 18 in
`preprocess_workflow_kinect_auto.py`, the 3 in `primary_workflow_kinect_auto.py`, the 3 `@task`
stages in `merging_pipeline_neuron_to_kinect_auto.py`, `3. Track LED Blinking` and
`Manual: Define correlation videos thresholding` in `preprocess_workflow_kinect_manual.py`. The
remaining 29 names were redundant with the function name (e.g. `@flow(name="create_kinect_configs")`
on `create_kinect_configs_flow`) and were dropped outright.

Two already-commented-out decorators were also deleted — `# @flow(name="Run Single Session Pipeline")`
(`preprocess_workflow_kinect_auto.py`) and `# @flow(name="Run Single Session Postprocessing")`
(`postprocess_workflow_kinect_auto.py`) — both redundant with their function names and dead
references to a framework being removed.

**Surviving `prefect` mentions under `code/scripts/`** (all prose, none executable):
- the four `NotImplementedError` messages added in Phase 2 (2 lines each in
  `merging_pipeline_neuron_to_kinect_auto.py`, `postprocess_workflow_kinect_auto.py`,
  `preprocess_workflow_kinect_auto.py`, `primary_workflow_kinect_auto.py`)
- `_3_preprocessing/_4_somatosensory_quantification/poc_contact_depth_field.py:32` and
  `__misc/standalone_mkv_to_forearm_mesh.py:4` — docstrings noting "no Prefect flow". These become
  stale once the dependency is gone; reword in Phase 5.6.
- `code/tests/test_parallel_execution_removed.py:54` — `getattr(fn, "fn", fn)` with an explanatory
  comment. Still correct on plain functions; left as-is.

**Narration check (Success Criterion, measured).** Baseline = the HEAD (pre-Phase-3) copy of each
file loaded via `importlib`; after = the current file imported as the entry script does, then a real
module function called.

| | `preprocess_pipeline_nerve_auto` | `merging_pipeline_neuron_to_kinect_auto` |
|---|---|---|
| before: root logger | `WARNING`, `['PrefectConsoleHandler']` | `WARNING`, `['PrefectConsoleHandler']` |
| before: module logger effective | `WARNING` | `WARNING` |
| before: `logger.info(...)` on stderr | **nothing** | **nothing** |
| after: `'prefect' in sys.modules` | `False` | `False` |
| after: root logger | `INFO`, `StreamHandler -> <stderr>` | `INFO`, `StreamHandler -> <stderr>` |
| after: real call | `run_batch_processing([], ...)` | `filter_by_neural_quality_flow(...)` |
| after: line on stderr | `... - INFO - Starting batch processing for 0 block files.` | `... - INFO - [does-not-exist-merged.csv] Filtering by neural quality xlsx: ...` |

All 10 entry scripts import cleanly and `--help` exits 0. Suite: **328 passed, 7 skipped** (unchanged).

**Files Modified:** the 10 entry scripts listed above.

**Dependencies:** Phase 2

### Phase 4: Delete the GUI server integration
**Goal:** The GUI launches with no server lifecycle.
**Started:** —  **Completed:** —

- [ ] 4.1 — `launcher_window.py`: delete the import (`:31`), construction and start (`:68-70`),
      `_wait_for_prefect_server` (`:295-304`), the restart-on-Run block (`:330-336`), the `env=`
      plumbing (`:341`, and `env=env` at `:346`), and `stop()` in `closeEvent` (`:292`). Preserve
      `event.accept()` at `:293`.
- [ ] 4.2 — Delete `code/src/utils/gui/dag_launcher/prefect_server_manager.py` (144 lines).
- [ ] 4.3 — Check `code/src/utils/gui/dag_launcher/__init__.py` for a re-export before deleting.
- [ ] 4.4 — Remove the four Prefect-specific status-bar strings; keep every other message.
- [ ] 4.5 — Confirm the abort button, console reader, status bar and exit-code handling are untouched
      — the investigation found none of them depend on Prefect.
- [ ] 4.6 — Measure GUI launch-to-interactive before and after. The 30 s readiness wait is gone.

**Files Modified:**
- `code/src/utils/gui/dag_launcher/launcher_window.py`
- `code/src/utils/gui/dag_launcher/prefect_server_manager.py` — deleted
- `code/src/utils/gui/dag_launcher/__init__.py` — only if it re-exports

**Dependencies:** Phase 3

### Phase 5: Drop the dependency and update docs
**Goal:** Prefect is not installed, not declared, and not documented as required.
**Started:** —  **Completed:** —

- [ ] 5.1 — Remove `prefect==3.4.22` and its `# --- Workflow Orchestration ---` header from
      `requirements.txt:53-54`.
- [ ] 5.2 — Remove `- prefect>=3.4` and its header from `environment.yml:61-62`.
- [ ] 5.3 — Check whether anything still imports `pydantic_settings`; if not, note it as orphaned but
      do not remove it here.
- [ ] 5.4 — Update `CLAUDE.md`: the Architecture Overview says workflows are "orchestrated by
      Prefect" and describes `@flow` functions. Correct it.
- [ ] 5.5 — Update `docs/architecture/gui-dag-launcher-reference.md` (`:35`, `:838`, `:1209`,
      `:1433`).
- [ ] 5.6 — Changelog `docs/changelogs/remove-prefect-orchestration.md`, listing the surviving prose
      mentions so a future grep hit is not mistaken for a leftover.
- [ ] 5.7 — Knowledge-base note recording the shared-`PREFECT_HOME` collision and why the dependency
      was dropped, so the reasoning survives the code.

**Files Modified:**
- `requirements.txt`, `environment.yml`, `CLAUDE.md`
- `docs/architecture/gui-dag-launcher-reference.md`
- `docs/changelogs/remove-prefect-orchestration.md` — new
- `docs/development/knowledge-base/note-prefect-removal.md` — new

**Dependencies:** Phase 4

### Phase 6: Verification against a real run
**Goal:** Measured equivalence, not argument.
**Started:** —  **Completed:** —

- [ ] 6.1 — In an environment **without** prefect installed, confirm all 10 entry scripts import and
      `--help` works.
- [ ] 6.2 — Full test suite: no fewer than 320 passed / 7 skipped.
- [ ] 6.3 — Run one real workflow end to end before and after (a cheap one — the merging DAG on a
      single session) and hash-compare every output artifact.
- [ ] 6.4 — Assert the narration: capture stdout+stderr of `preprocess_pipeline_nerve_auto.py` and
      `merging_pipeline_neuron_to_kinect_auto.py` and confirm the expected INFO lines are present.
- [ ] 6.5 — GUI: open, press Run, confirm the workflow starts with no readiness delay and the console
      streams line by line.
- [ ] 6.6 — Set `parallel_execution: true` in one config and confirm a loud `NotImplementedError`.
- [ ] 6.7 — Record before/after GUI launch time and workflow start latency.

**Files Modified:** none (verification only)

**Dependencies:** Phase 5

---

## Testing Plan

### Unit Tests
- [ ] No test currently references Prefect — confirm the suite is unaffected by construction.
- [x] Add a test asserting `parallel_execution: true` raises `NotImplementedError` for each of the
      four batch runners. — `code/tests/test_parallel_execution_removed.py`
- [ ] Add a guard test that imports all 10 entry-point modules, so a stray Prefect import fails CI
      rather than at runtime.

### Integration Tests
- [ ] One real workflow, before/after, hash-compared outputs (6.3).
- [ ] Narration assertion for the two `get_run_logger` files (6.4).
- [ ] Entry scripts import in a prefect-free environment (6.1).

### Manual Verification
- [ ] GUI opens with no server, no 30 s wait.
- [ ] Run streams console output live during a long workflow.
- [ ] Abort still terminates the workflow subprocess.
- [ ] `PipelineMonitor` dashboard and the Excel status report still appear.

### Edge Cases
- [ ] A workflow that raises mid-task still reports FAILURE via `TaskExecutor` and a non-zero exit
      code reaches the GUI.
- [ ] A DAG config with an unknown task name behaves as before.
- [ ] Closing the GUI mid-run still terminates cleanly with no orphaned child.

---

## Documentation Plan

- [ ] `CLAUDE.md` — Architecture Overview currently states workflows are "orchestrated by Prefect"
      and defined as `@flow` functions. Correct both.
- [ ] `docs/architecture/gui-dag-launcher-reference.md` — four references.
- [ ] Changelog `docs/changelogs/remove-prefect-orchestration.md`, including the list of surviving
      prose mentions.
- [ ] Knowledge-base note: the shared-`PREFECT_HOME` collision, the `DEVNULL` swallowing that hid it,
      and why the dependency was dropped rather than pinned.
- [ ] Mark `docs/development/plans/completed/persistent-prefect-server-at-gui-launch.md` as superseded
      by this plan — it is the design record for what is being deleted.

---

## Rollback Plan

1. **Phase-per-commit** makes this cheap: revert from the last phase backwards. Phases 2-5 are
   independent of each other in the revert direction.
2. Phase 1 should be **kept regardless** — the `basicConfig` and `-u` fixes are improvements
   independent of Prefect.
3. If the removal is abandoned mid-way, the fallback end state is the already-verified per-project
   `PREFECT_HOME` (`conda env config vars set PREFECT_HOME=C:\Users\basil\.prefect-social-touch -n social-touch`),
   which was confirmed working on 2026-08-18.
4. **Data considerations:** none. No artifact format, path, or schema changes. No migration. Prefect's
   own databases (`~/.prefect`, `~/.prefect-social-touch`) become inert but are not deleted.
5. Re-adding the dependency is `pip install prefect==3.4.22` plus reverting the commits.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **`get_run_logger()` → `getLogger(__name__)` silences two entry points.** Neither file calls `basicConfig`, so the root logger stays at WARNING with no handler: every `.info()` vanishes, no error, exit code 0 | **High if missed** | **High** | Phase 1 adds `basicConfig` *before* anything is removed; 6.4 asserts the narration explicitly. This is the plan's single most dangerous item |
| Loss of `log_prints=True` block-buffers the GUI console — output arrives in bursts, looks frozen | High if missed | Med | Phase 1.3 adds `-u`; 6.5 verifies live streaming. Worth doing regardless |
| Loss of Prefect's pydantic argument coercion on 55 functions | Low | Med | Shipped configs are correctly typed; two defensive re-coercions already exist. Silent when it bites — noted, not otherwise mitigated |
| A `@flow` name carried information the function name does not | Low | Low | Preserve stage numbering in a comment where not obvious (3.x) |
| Merge conflict with the open depth-field branch, which modifies `merging_pipeline_neuron_to_kinect_auto.py` | **High** | Med | Base branch is that feature branch, so this stacks on top rather than diverging. See Sequencing below |
| A hidden Prefect dependency surfaces at runtime rather than import time | Low | Med | 6.1 runs every entry script in a prefect-free environment; the import-guard test makes it a CI failure |
| Someone later sets `parallel_execution: true` expecting it to work | Med | Low | 2.5 raises `NotImplementedError` naming this plan. Today it silently processes zero sessions in postprocess |
| Scope creep into rewriting the orchestration layer | Med | High | Out of Scope is explicit: `DagConfigHandler`, `TaskExecutor`, `PipelineMonitor` and `should_process_task` are untouched |

### Sequencing

`**Base Branch:**` is `feature/filter-contact-depth-field-by-neural-quality`, not `dev`, deliberately.
That branch is complete but unmerged, and it modifies
`code/scripts/merging_pipeline_neuron_to_kinect_auto.py` — one of the ten files this plan rewrites.
Branching from `dev` would guarantee a conflict in exactly the file with the most Prefect surface
(3 tasks, 2 flows, 5 `get_run_logger` sites).

If the depth-field branch is merged to `dev` before this work starts, re-base this plan on `dev` and
update the header.

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — logging + buffering | ~15 LOC + verification | None |
| Phase 2 — dead parallel branches | ~80 LOC deleted | Phase 1 |
| Phase 3 — decorators + logger | 10 files, ~120 lines touched, mostly deletions | Phase 2 |
| Phase 4 — GUI server integration | ~40 LOC deleted + 1 file deleted (144 lines) | Phase 3 |
| Phase 5 — dependency + docs | ~10 LOC + docs | Phase 4 |
| Phase 6 — verification | measurement only | Phase 5 |

---

## References

- Design record for what is being deleted:
  `docs/development/plans/completed/persistent-prefect-server-at-gui-launch.md`
- The blocking incident (2026-08-18): shared `~/.prefect/prefect.db` at alembic revision
  `79e7a60e43d8`, unreadable by this environment's Prefect 3.6.22, with the error swallowed by
  `prefect_server_manager.py:77-78`
- Open branch this stacks on:
  `docs/development/plans/active/filter-contact-depth-field-by-neural-quality.md`
- Repo orchestration that replaces Prefect's role: `code/src/utils/pipeline/pipeline_config_manager.py`
  (`DagConfigHandler`), `code/src/utils/pipeline/task_executor.py`,
  `code/src/utils/pipeline/monitoring/pipeline_monitor.py`, `code/src/utils/should_process_task.py`
- Guides: `~/.claude/knowledge/data-pipeline-engineering/` 02 (§1 pipe-and-filter, §5 idempotency,
  §8 error boundaries), 05 (§YAGNI, §technical debt)
