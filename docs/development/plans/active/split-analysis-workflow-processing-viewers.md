# Plan: True separation of analysis workflow (processing vs viewers)

**Created:** 2026-05-20
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/rf-surface-shape-viewer`
**Branch:** `feature/split-analysis-workflow-processing-viewers`

---

## Overview

**What:** Split `code/scripts/analysis_workflow.py` (1488 lines, 28 Prefect
flows) into two structurally independent entry scripts —
`analysis_workflow_processing.py` and `analysis_workflow_viewers.py` —
each holding only the flows, imports, and dispatch logic for its own
slice. Move shared bootstrap helpers into a new
`code/src/analysis/pipeline/` sub-package and delete the old monolith.

**Why:** Today the split is purely a run-time category filter (`--mode
processing` vs `--mode viewers`) on a shared engine. Both entry scripts
import everything: a `--mode processing` run still loads every viewer
flow, every PyQt5 launcher, and the full `analysis.receptive_field_mapping`
GUI surface. That defeats the purpose of having two pipelines and ties
processing-side maintenance to viewer-side regressions.

**How:** Each entry script becomes self-contained — its own flow
definitions, its own minimal import set, and its own
`pipeline_stages: list[dict]` registry (dict-of-stages with lambda
kwargs, mirroring the cleanest pattern in the repo:
`preprocess_workflow_kinect_auto.py:481-626`). A new
`code/src/analysis/pipeline/` package exports a single generic
`run_pipeline_stages(...)` dispatcher and the bootstrap helpers
(`collect_unique_session_dirs`, `discover_input_items`). The old
`analysis_workflow.py` is deleted; the launcher and the two DAG YAMLs
need no changes.

## Problem Statement

The current architecture has three concrete problems:

1. **Coupling at import time.** Both wrappers do
   `from analysis_workflow import run_batch_analysis`. Python then
   executes the full module body, which imports every viewer launcher and
   every touch-analytics pipeline regardless of which slice the user is
   running. Any GUI-side import error breaks the processing pipeline and
   vice versa.

2. **A 150-line if/elif kwargs chain.** `analysis_workflow.py:1311-1441`
   builds per-task kwargs through a sequence of `if task_name == "X":`
   branches that mix concerns from every task in both slices. Adding a
   new task means editing the dispatcher; renaming an option means
   hunting through this chain.

3. **No structural enforcement of the split.** The category metadata
   in the DAG YAMLs (`category: processing` / `category: viewer`) and
   the `_PROCESSING_CATEGORIES` / `_VIEWER_CATEGORIES` sets in the
   workflow are a soft contract. A viewer flow accidentally added to
   the processing DAG would silently run.

The launcher (`configs/launcher.yaml:72-77`) and the DAG configs already
treat processing and viewers as two products. The Python layer is the
only place where they are still one.

## Goals

### In Scope

1. New sub-package `code/src/analysis/pipeline/` with three modules:
   `__init__.py`, `session_discovery.py`, `stage_runner.py`.
2. Full rewrite of `code/scripts/analysis_workflow_processing.py` —
   self-contained with the 20 processing-slice flows inline.
3. Full rewrite of `code/scripts/analysis_workflow_viewers.py` —
   self-contained with the 8 viewer-slice flows inline.
4. Delete `code/scripts/analysis_workflow.py`.
5. Adopt the **dict-of-stages with lambda kwargs** dispatch pattern (per
   `preprocess_workflow_kinect_auto.py`) in both entry files —
   eliminating the if/elif kwargs chain entirely.

### Out of Scope

- DAG YAML changes — `configs/analyse_workflow_processing_dag.yaml`
  and `configs/analyse_workflow_viewers_dag.yaml` stay byte-identical.
- Launcher changes — `configs/launcher.yaml:72-77` already points at the
  two entry scripts.
- Flow body changes — each flow's internal logic is copied verbatim. No
  behavioural change to any task.
- Touch-analytics or receptive-field-mapping package refactors — only
  import sites move.
- Removing the legacy `map_receptive_fields_clustered_flow` (still
  referenced by the processing DAG with `enabled: false`; keep it
  bundled with the processing slice for now).
- Re-introducing a category filter or `--mode` flag in the new scripts.

## Success Criteria

- [ ] `code/scripts/analysis_workflow.py` no longer exists.
- [ ] `code/scripts/analysis_workflow_processing.py` and
  `code/scripts/analysis_workflow_viewers.py` do not import from each
  other and do not import any common analysis module that the other
  doesn't need.
- [ ] `grep "from analysis_workflow"` and `grep "import analysis_workflow"`
  return zero results across `code/`, `configs/`, and `docs/` (excluding
  this plan and historical git/log content).
- [ ] Running each entry script with its default DAG config and every
  task disabled completes cleanly with `"Task '<x>' is disabled in DAG.
  Skipping."` for every task.
- [ ] Launching `python code/scripts/launch_pipeline_gui.py` shows both
  "Analysis [Processing]" and "Analysis [Viewers]" entries and each
  dispatches to the new file.
- [ ] At least one cheap processing task (e.g. `summarize_session_blocks`)
  and one cheap viewer task (e.g. `explore_rf_surface`) run end-to-end
  against a known-good session.
- [ ] `pytest code/tests/` passes (no test currently imports
  `analysis_workflow`; this should still hold).

---

## Technical Design

### Approach

Three concrete moves:

1. **Extract shared bootstrap into a new sub-package.** The only code
   that is identical between the two slices is session discovery and
   the dispatch loop skeleton. A new `code/src/analysis/pipeline/`
   package holds:
   - `session_discovery.collect_unique_session_dirs(block_files,
     project_data_root) -> Dict[Path, Path]` — moved verbatim from
     `analysis_workflow.py:1177-1195`.
   - `session_discovery.discover_input_items(session_map) ->
     List[Tuple[Path, Path]]` — extracted from
     `analysis_workflow.py:1270-1283` (the
     `glob("*_semicontrolled_aggregated_session.csv")` scan).
   - `stage_runner.run_pipeline_stages(pipeline_stages, dag_handler,
     monitor, items_to_process, block_id_prefix)` — generic dispatcher
     modelled on `preprocess_workflow_kinect_auto.py:611-626`. Walks
     `pipeline_stages`, opens a `TaskExecutor` per stage, invokes the
     stage's `params()` lambda lazily, injects common kwargs
     (`input_items`, `force_processing` from task options), then calls
     `flow_func(**kwargs)`.

2. **Each entry script declares its own task registry as a list of
   dicts.** Following Pattern 1 from `preprocess_workflow_kinect_auto.py`:

   ```python
   pipeline_stages = [
       {"name": "touch_clustering",
        "func": touch_clustering_flow,
        "params": lambda: {
            "cluster_groups": dag_handler.get_task_options("touch_clustering").get("cluster_groups"),
            "clustering_profiles": dag_handler.get_task_options("touch_clustering").get("clustering_profiles"),
            "reduction": dag_handler.get_task_options("touch_clustering").get("reduction"),
            "evaluation": dag_handler.get_task_options("touch_clustering").get("evaluation"),
        }},
       # ... one entry per task in the slice
   ]
   ```

   Lambdas are evaluated lazily inside the dispatcher, so DAG-config
   lookups only happen for tasks that actually run. Tasks that read
   from another task's options (e.g. `extract_receptive_fields_clustered`
   needs `cluster_group_defs` from the `touch_clustering` task) re-read
   them inside their lambda — no shared mutable state between stages.

3. **Drop the category/mode filter.** With each file scoped to its own
   `pipeline_stages`, a DAG entry referencing an unknown task is simply
   logged and skipped by the runner. No `--mode` arg, no
   `_category_allowed`, no shared `available_tasks` registry.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **True split + shared `analysis.pipeline` sub-package** (chosen) | Eliminates cross-slice imports; matches preprocess Pattern 1; zero launcher/DAG changes | Two files now hold copies of flow definitions (no logic duplication, just code-movement) | **Chosen** |
| Keep `analysis_workflow.py` as the engine and just add a stricter category filter | Smallest diff | Doesn't fix import coupling; doesn't fix the if/elif chain; soft contract only | Rejected |
| Inline-duplicate the bootstrap helpers into both files (no shared package) | Maximum independence | Two copies of `collect_unique_session_dirs` etc. drift over time | Rejected |
| Move shared helpers into `code/src/utils/pipeline/` instead of `analysis/pipeline/` | Reuses existing utils package | Mixes analysis-specific scanning (`*_semicontrolled_aggregated_session.csv` glob) into a generic location | Rejected |
| Per-task `kwargs_builder` callable registered alongside each flow (third element of tuple) | Keeps task-specific config code next to its flow | Bigger refactor (every flow gets a sibling builder); preprocess Pattern 1 already pushes kwargs into the lambda for free | Rejected (YAGNI) |
| Keep `analysis_workflow.py` as a deprecation stub | Defensive against unknown external callers | No external caller exists (verified via grep); pure deadweight | Rejected |

### Architecture Constraints (from knowledge base)

Relevance check against `docs/development/knowledge-base/` notes: **no
applicable notes.** Existing notes cover GUI/PyVista layout, CuPy import
order, Kinect parallax, Qt signal recursion, RF projection algorithms,
matplotlib blitting, and coordinate spaces — none touch Prefect
orchestration or DAG dispatch. The split is a pure code-movement
refactor; the flow bodies and their dependencies are not modified.

- **CuPy import order** — N/A (no new CuPy or `preprocessing` imports
  introduced; existing imports stay inside their flows).
- **Git merge** — `--no-ff` per project convention.

### Architecture Changes

```
code/src/analysis/
├── pipeline/                          ← NEW sub-package
│   ├── __init__.py                   ← re-exports
│   ├── session_discovery.py          ← collect_unique_session_dirs,
│   │                                    discover_input_items
│   └── stage_runner.py               ← run_pipeline_stages

code/scripts/
├── analysis_workflow.py              ← DELETED
├── analysis_workflow_processing.py   ← REWRITTEN (self-contained)
│   ├── imports from analysis.touch_analytics.*
│   ├── imports from analysis.receptive_field_mapping.* (processing symbols only)
│   ├── 20 @flow definitions (verbatim from old file)
│   └── pipeline_stages list + main()
└── analysis_workflow_viewers.py      ← REWRITTEN (self-contained)
    ├── imports for launch_*, precompute_explorer_caches, launch_preparation_viewer
    ├── 8 @flow definitions (verbatim from old file)
    └── pipeline_stages list + main()
```

No changes to:
- `configs/analyse_workflow_processing_dag.yaml`
- `configs/analyse_workflow_viewers_dag.yaml`
- `configs/launcher.yaml`
- Any `code/src/analysis/touch_analytics/**` or
  `code/src/analysis/receptive_field_mapping/**` module.

### Flow inventory — final placement

**`analysis_workflow_processing.py` — 20 flows** (categories `processing`
+ the one `viewer_required` task that emits camera-settings JSON
consumed by processing):

| Flow | Source (current line in `analysis_workflow.py`) |
|---|---|
| `summarize_session_blocks_flow` | 69 |
| `map_receptive_fields_simple_flow` | 105 |
| `precompute_forearm_slim_uv_flow` | 132 |
| `map_single_touch_rf_flow` | 202 |
| `visualize_population_rf_maps_flow` | 229 |
| `map_population_rf_grid_flow` | 259 |
| `reduce_population_rf_grid_flow` | 366 |
| `visualize_population_rf_grid_metrics_flow` | 445 |
| `visualize_session_comparison_flow` | 495 |
| `touch_preparation_flow` | 571 |
| `touch_series_transforms_flow` | 593 |
| `touch_feature_extraction_flow` | 617 |
| `touch_clustering_flow` | 646 |
| `touch_comparing_flow` | 680 |
| `analyse_ap_efficacy_flow` | 716 |
| `map_receptive_fields_clustered_flow` *(legacy, kept for back-compat)* | 755 |
| `extract_receptive_fields_clustered_flow` | 796 |
| `compute_receptive_field_metrics_flow` | 830 |
| `visualize_receptive_fields_clustered_flow` | 870 |
| `set_rf_camera_settings_flow` *(viewer_required)* | 1105 |

**`analysis_workflow_viewers.py` — 8 flows** (categories `viewer` +
`viewer_support`):

| Flow | Source |
|---|---|
| `precompute_explorer_caches_flow` *(viewer_support)* | 915 |
| `explore_rf_feature_space_flow` | 936 |
| `explore_touch_playback_flow` | 955 |
| `explore_touch_population_flow` | 974 |
| `explore_single_touch_rf_flow` | 997 |
| `explore_rf_gallery_flow` | 1017 |
| `explore_preparation_flow` | 1068 |
| `explore_rf_surface_flow` | 1087 |

---

## Implementation Plan

### Phase 1: New `analysis.pipeline` sub-package

**Goal:** Land the shared bootstrap helpers so both entry scripts can
import them without depending on each other.
**Started:** 2026-05-20
**Completed:** 2026-05-20

- [x] Task 1.1 — Create `code/src/analysis/pipeline/__init__.py`
  re-exporting `collect_unique_session_dirs`, `discover_input_items`,
  `run_pipeline_stages`.
- [x] Task 1.2 — Create `code/src/analysis/pipeline/session_discovery.py`.
  Move `collect_unique_session_dirs` verbatim from
  `analysis_workflow.py:1177-1195`. Add `discover_input_items`
  extracted from the items-to-process scan at
  `analysis_workflow.py:1270-1283`.
- [x] Task 1.3 — Create `code/src/analysis/pipeline/stage_runner.py`
  defining `run_pipeline_stages(pipeline_stages, dag_handler, monitor,
  items_to_process, block_id_prefix)`. Loop body mirrors
  `preprocess_workflow_kinect_auto.py:611-626`: open `TaskExecutor`,
  check `executor.can_run`, call the stage's `params()` lambda, inject
  `input_items` and `force_processing` as common kwargs, call
  `flow_func(**kwargs)`, swallow exceptions onto `executor.error_msg`.
  Skip and log any task absent from the registry.

**Files Modified:**
- `code/src/analysis/pipeline/__init__.py` — new file (~10 lines).
- `code/src/analysis/pipeline/session_discovery.py` — new file
  (~45 lines).
- `code/src/analysis/pipeline/stage_runner.py` — new file (~50 lines).

**Dependencies:** None.

### Phase 2: Self-contained `analysis_workflow_processing.py`

**Goal:** Rewrite the processing entry script to hold its own flows and
dispatch, no longer importing `analysis_workflow`.
**Started:** 2026-05-20
**Completed:** 2026-05-20

- [x] Task 2.1 — Replace the current 66-line wrapper. Add the imports
  needed by the 20 processing flows (the existing
  `analysis_workflow.py:30-50, 62-65` import block, minus the viewer
  launchers).
- [x] Task 2.2 — Inline the 20 flow definitions verbatim from the
  source-line table above. No edits to flow bodies.
- [x] Task 2.3 — Build the `pipeline_stages` list in the same task
  order as the current `available_tasks` registry
  (`analysis_workflow.py:1234-1254`). Each entry has `name`, `func`,
  and a `params` lambda that reads its options via
  `dag_handler.get_task_options("<task>")`. Cross-task option reads
  (e.g. `cluster_group_defs` consumed by extract/compute/visualize RF
  flows; `grid_group_defs` consumed by reduce/visualize grid flows) go
  inside the consuming lambda — no precomputed shared variables.
- [x] Task 2.4 — Rewrite `main()` to call
  `run_pipeline_stages(pipeline_stages, dag_handler, monitor,
  items_to_process, "batch_run_processing")` after computing
  `session_map = collect_unique_session_dirs(...)` and
  `items_to_process = discover_input_items(session_map)`. Keep the
  `--dag-config` arg, default
  `configs/analyse_workflow_processing_dag.yaml`, and report file
  `reports/analysis_processing_status.xlsx`.

**Files Modified:**
- `code/scripts/analysis_workflow_processing.py` — full rewrite
  (~1050 lines once flows are inlined).

**Dependencies:** Phase 1.

### Phase 3: Self-contained `analysis_workflow_viewers.py`

**Goal:** Rewrite the viewer entry script the same way.
**Started:** 2026-05-20
**Completed:** 2026-05-20

- [x] Task 3.1 — Replace the wrapper. Imports needed: the viewer-side
  symbols from `analysis.receptive_field_mapping`
  (`precompute_explorer_caches`, `launch_feature_space_explorer`,
  `launch_touch_playback_explorer`, `launch_touch_population_explorer`,
  `launch_single_touch_rf_explorer`, `launch_gallery_viewer`,
  `launch_rf_surface_viewer`), plus `launch_preparation_viewer` from
  `analysis.touch_analytics.gui`. No touch-analytics pipeline imports.
- [x] Task 3.2 — Inline the 8 viewer flow definitions verbatim.
- [x] Task 3.3 — Build `pipeline_stages` in viewer-DAG order. Most
  lambdas return only the bare common kwargs (the runner already
  injects `input_items` and `force_processing`); three add
  `neuron_mode` (`explore_touch_population`, `explore_single_touch_rf`,
  `explore_rf_surface`); `explore_rf_gallery` reads `cluster_groups`
  from its own options and `cluster_group_defs` from
  `dag_handler.get_task_options("touch_clustering")`.
- [x] Task 3.4 — Rewrite `main()` to call
  `run_pipeline_stages(..., "batch_run_viewers")` and write to
  `reports/analysis_viewers_status.xlsx`.

**Files Modified:**
- `code/scripts/analysis_workflow_viewers.py` — full rewrite
  (~420 lines once flows are inlined).

**Dependencies:** Phase 1.

### Phase 4: Delete the monolith and audit references

**Goal:** Remove `analysis_workflow.py` and confirm nothing imports it.
**Started:** 2026-05-20
**Completed:** 2026-05-20

- [x] Task 4.1 — `git rm code/scripts/analysis_workflow.py`.
- [x] Task 4.2 — Grep audit:
  `rg "analysis_workflow(?!_processing|_viewers)" code/ configs/`.
  Expect zero hits (the launcher already points at the two new files;
  the DAG YAMLs reference no Python paths).
- [x] Task 4.3 — Touch up `code/src/analysis/CLAUDE.md` if it names the
  old combined file (only update if it appears; skip otherwise).

**Files Modified:**
- `code/scripts/analysis_workflow.py` — DELETED.
- `code/src/analysis/CLAUDE.md` — text-only adjustment if needed.

**Dependencies:** Phase 2, Phase 3.

---

## Testing Plan

### Unit Tests

The current test suite does not import `analysis_workflow` and the new
flow bodies are copies (no logic changes), so no new unit tests are
strictly required. Optional additions:

- [ ] `test_run_pipeline_stages_skips_disabled` — pass a
  `pipeline_stages` with one stage whose DAG entry has
  `enabled: false`. Assert the stage's `func` is never called.
- [ ] `test_run_pipeline_stages_unknown_task_ignored` — pass a DAG
  config referencing a task not in `pipeline_stages`. Assert the
  runner logs a warning and completes without error.

### Integration Tests

- [ ] `pytest code/tests/` — must pass unchanged.
- [ ] Bare-import smoke test (both new scripts):
  ```bash
  python -c "import importlib.util as u, pathlib; \
    [u.spec_from_file_location(n, p).loader.exec_module(u.module_from_spec(u.spec_from_file_location(n, p))) \
     for n, p in [('proc', 'code/scripts/analysis_workflow_processing.py'), \
                  ('view', 'code/scripts/analysis_workflow_viewers.py')]]"
  ```
  Expected: no `ImportError`, no `NameError`.

### Manual Verification

- [ ] **Processing dry-run.** Disable every task in
  `configs/analyse_workflow_processing_dag.yaml`, then
  `python code/scripts/analysis_workflow_processing.py
  --dag-config configs/analyse_workflow_processing_dag.yaml`.
  Confirm each task name is logged as skipped, exit code 0.
- [ ] **Viewers dry-run.** Same with the viewer DAG.
- [ ] **GUI smoke.** `python code/scripts/launch_pipeline_gui.py`.
  Under the **Analysis** category, both "Processing" and "Viewers"
  entries appear. Open each, enable one cheap task, and run on a
  known-good session. Confirm completion.
- [ ] **Import isolation check.** From a fresh Python interpreter,
  `import sys` then run the processing script's import block; confirm
  `sys.modules` does not contain any of the viewer-side
  `launch_*_viewer` names. Mirror check for viewers (no
  `analysis.touch_analytics.preparation_pipeline` etc.).

### Edge Cases

- [ ] DAG file with all tasks disabled — runner exits cleanly with no
  TaskExecutor errors.
- [ ] DAG file with a typo in a task name (e.g. `touch_prepartion`) —
  runner logs an unknown-task warning and continues with the rest.
- [ ] DAG file with one task whose option lambda raises (e.g. missing
  required key surfaces from a `.get(...)`-less access) — exception
  surfaces as `executor.error_msg`, other tasks still run.

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` if and only if it currently
  names `analysis_workflow.py`; point at the two entry scripts and the
  new `analysis.pipeline` sub-package.
- [ ] No top-level README changes — `launch_pipeline_gui.py` remains
  the documented entry point.
- [ ] No changelog file convention found in this repo
  (`docs/changelogs/` does not exist); skip.
- [ ] Docstrings on the three new `analysis.pipeline` modules — one
  module-level docstring each, describing the responsibility.

---

## Rollback Plan

1. The pre-split state is preserved in git: the merge commit on
   `dev` can be reverted with `git revert -m 1 <merge-sha>`.
2. No data migration — no on-disk artifacts change shape, no output
   paths change, no DAG schema changes.
3. If only a partial revert is needed (e.g. flows behave correctly but
   one entry script has an import bug), the previous
   `analysis_workflow.py` and the thin wrappers can be restored
   verbatim from the pre-split commit while the new
   `analysis.pipeline` sub-package stays in place — it is purely
   additive.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| A flow body silently relies on a module-level helper from `analysis_workflow.py` (`_collect_unified_files`, etc.) that isn't carried over | Medium | High | Before deletion: grep every helper defined in `analysis_workflow.py` for cross-references inside the 28 flow bodies; carry over any genuinely-used helper into the appropriate entry script |
| Lambda late-binding bug — all lambdas close over the same loop variable in a comprehension | Low | High | Use explicit `def` or capture-by-default-arg pattern in the registry; preprocess Pattern 1 already avoids this by using literal dict entries, not a loop |
| `TaskExecutor` semantics differ between the preprocess use-case and what `analysis_workflow.py:1326-1444` relies on (e.g. `monitor` argument, batch-id format) | Low | Medium | Audit `TaskExecutor.__init__` signature once; keep the same `batch_id = f"batch_run_{task_name}"` convention as the current dispatcher |
| Cross-task DAG option reads (`cluster_group_defs`, `grid_group_defs`) silently return `None` when the upstream task is disabled but the downstream task is enabled | Medium | Medium | Each downstream lambda re-reads via `dag_handler.get_task_options(...)`; existing flow bodies already raise on missing required args — keep that behaviour |
| Deleting `analysis_workflow.py` breaks an external caller outside the repo (a notebook, an ad-hoc script) | Low | Low | Project-internal use only per CLAUDE.md; grep confirms no callers. If a downstream complaint arrives, restore as a deprecation stub in a follow-up |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — `analysis.pipeline` sub-package | ~0.25 day | None |
| Phase 2 — Rewrite processing entry script | ~0.5 day | Phase 1 |
| Phase 3 — Rewrite viewers entry script | ~0.25 day | Phase 1 |
| Phase 4 — Delete monolith + audit | ~0.1 day | Phases 2, 3 |

---

## References

- **Current monolith:** `code/scripts/analysis_workflow.py` (1488 lines)
- **Current thin wrappers:**
  `code/scripts/analysis_workflow_processing.py`,
  `code/scripts/analysis_workflow_viewers.py`
- **Pattern source — dict-of-stages with lambda kwargs:**
  `code/scripts/preprocess_workflow_kinect_auto.py:481-626`
- **Existing utilities reused:**
  `code/src/utils/` (`DagConfigHandler`, `PipelineMonitor`,
  `TaskExecutor`, `path_tools`,
  `session_config_resolver.resolve_session_configs`).
- **Launcher (no change):** `configs/launcher.yaml:70-77`
- **DAG configs (no change):**
  `configs/analyse_workflow_processing_dag.yaml`,
  `configs/analyse_workflow_viewers_dag.yaml`
- **Internal scratch plan that preceded this one:**
  `~/.claude/plans/analyse-the-analysis-workflow-py-glimmering-starfish.md`
