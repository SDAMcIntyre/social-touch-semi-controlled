# Bug: --tasks filter silently skips stages whose dependencies weren't run in the same execution

**Date:** 2026-05-05  
**Status:** Known, not fixed  
**Observed:** Running `--tasks touch_series_transforms` alone produced no output and no error.

---

## Symptom

Running a single stage with `--tasks` completes silently with exit code 0 but writes no output:

```bash
python code/scripts/analysis_workflow.py \
  --dag-config configs/analyse_workflow_dag.yaml \
  --tasks touch_series_transforms
```

Output stops after startup messages. No task progress, no warnings, no error.

---

## Root cause

`DagConfigHandler.can_run()` in
`code/src/utils/pipeline/pipeline_config_manager.py` (line 58) gates each task
on its `depends_on` entries being present in `self.completed_tasks`:

```python
dependencies: List[str] = task_config.get('depends_on', [])
if not set(dependencies).issubset(self.completed_tasks):
    return False
```

`completed_tasks` is populated **only at runtime** — a task is added to it by
`mark_completed()` (line 62) when it finishes successfully in the current process.
There is no check against disk: whether a dependency's output file already exists
is irrelevant to `can_run`.

When `--tasks touch_series_transforms` is passed, the task loop skips
`touch_preparation` (not in the filter), so it is never marked completed, so
`touch_series_transforms.can_run()` returns False, and the task is silently skipped.
The `TaskExecutor` context manager prints nothing when `can_run` is False (line 29: 
`if not self.can_run: return`), so there is no visible indication of the skip.

---

## Workaround

Include all ancestor tasks in the `--tasks` filter. Completed stages are
idempotent — they check output mtimes and skip immediately if already up to date:

```bash
# Stage 1 (touch_preparation already done — will skip in ~0.1s):
python code/scripts/analysis_workflow.py \
  --dag-config configs/analyse_workflow_dag.yaml \
  --tasks touch_preparation touch_series_transforms

# Stage 2a (both predecessors already done):
python code/scripts/analysis_workflow.py \
  --dag-config configs/analyse_workflow_dag.yaml \
  --tasks touch_preparation touch_series_transforms touch_feature_extraction
```

Dependency chains for each stage:

| Stage | `--tasks` value needed |
|---|---|
| `touch_preparation` | `touch_preparation` |
| `touch_series_transforms` | `touch_preparation touch_series_transforms` |
| `touch_feature_extraction` | `touch_preparation touch_series_transforms touch_feature_extraction` |
| `touch_clustering` | `touch_preparation touch_series_transforms touch_feature_extraction touch_clustering` |
| `extract_receptive_fields_clustered` | ...`touch_clustering extract_receptive_fields_clustered` |
| `visualize_receptive_fields_clustered` | ...`extract_receptive_fields_clustered visualize_receptive_fields_clustered` |

Stages with no `depends_on` (`touch_preparation`, `map_receptive_fields_simple`,
`summarize_session_blocks`) can always be run alone.

---

## Potential fix

`can_run` should distinguish between "this dependency will run later in this
execution" and "this dependency has already been satisfied on disk." The simplest
approach would be to pre-populate `completed_tasks` before the task loop with any
dependency whose output artifact already exists, rather than relying purely on
runtime ordering.

---

## Affected file

`code/src/utils/pipeline/pipeline_config_manager.py` — `DagConfigHandler.can_run()`
