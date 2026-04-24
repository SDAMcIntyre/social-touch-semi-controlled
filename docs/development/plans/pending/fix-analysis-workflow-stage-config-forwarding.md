# Plan: Fix analysis workflow — forward reduction/evaluation configs

**Created:** 2026-04-23
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/align-clustering-pipeline-with-guidelines` (extended)

---

## Overview

**What:** Fix the config-forwarding gap that makes the `reduction` and
`evaluation` blocks in `analyse_workflow_dag.yaml` dead config — defined but
never delivered to the clustering pipeline.

**Why:** Commit `67c5996` added `reduction/` and `evaluation/` modules to
`touch_analytics` and wired them into `clustering_pipeline.py`, but the
workflow script (`analysis_workflow.py`) and the `run_clustering` public API
were never updated to carry these configs from the DAG YAML into the pipeline.
The pipeline silently falls back to hardcoded defaults (standard scaler, 20
bootstrap rounds), so user-configured values (e.g. `scaler: robust`,
`stability: null`) have no effect.

**How:** Thread `reduction` and `evaluation` configs from the DAG options
through `touch_clustering_flow` into `run_clustering`, where they are injected
into each combination dict before the per-combination loop.

## Problem Statement

The DAG YAML places `reduction` and `evaluation` as task-level options —
siblings of `feature_combinations`:

```yaml
touch_clustering:
  options:
    feature_combinations: { ... }
    clustering_profiles: { ... }
    reduction:                      # ← task-level
      scaler: standard
      variance_filter: null
      decomposition: null
    evaluation:                     # ← task-level
      internal_metrics: [silhouette, davies_bouldin, calinski_harabasz]
      stability:
        method: bootstrap
        n_rounds: 20
        ...
```

But the pipeline reads them from **inside each combination dict**:

- `ReductionPipeline.fit_transform(feature_df, combination_config)` reads
  `combination_config.get("reduction", {})` at
  `reduction/pipeline.py:63`.
- `_cluster_combination` reads `combination_config.get('evaluation', {})`
  at `clustering_pipeline.py:409`.

Nothing bridges these two levels. The forwarding chain has three breaks:

1. **`run_batch_analysis`** (`analysis_workflow.py:396–427`) — builds kwargs
   for `touch_clustering_flow` but has no branch for `"reduction"` or
   `"evaluation"` keys.
2. **`touch_clustering_flow`** (`analysis_workflow.py:127–154`) — signature
   accepts only `feature_combinations` and `clustering_profiles`; no
   parameters for `reduction` or `evaluation`.
3. **`run_clustering`** (`clustering_pipeline.py:175–181`) — signature accepts
   only `feature_combinations`, `clustering_profiles`, `force`,
   `extraction_dir`; no parameters for `reduction` or `evaluation`.

Result: the `reduction` and `evaluation` YAML blocks are parsed, stored in
the `DagConfigHandler`, displayed in the GUI dialogs, but silently discarded
at runtime.

## Goals

### In Scope

1. Add `reduction` and `evaluation` as optional parameters to
   `run_clustering()`.
2. Inside `run_clustering`, inject task-level `reduction`/`evaluation` into
   each combination dict (as defaults — a per-combination override wins if
   present).
3. Add `reduction` and `evaluation` parameters to `touch_clustering_flow()`
   and forward them.
4. Add `"reduction"` and `"evaluation"` forwarding branches to
   `run_batch_analysis()`.

### Out of Scope

- Changes to `ReductionPipeline` or `evaluation/` modules — they already
  work correctly once they receive config.
- Per-combination `reduction`/`evaluation` overrides in the DAG YAML (the
  injection logic supports it, but no YAML schema change is proposed).
- GUI changes — the `ReductionConfigDialog` and `EvaluationConfigDialog`
  already read/write the task-level keys correctly.
- Changes to `extraction_pipeline.py` (preparation module wiring is a
  separate concern).
- Changes to `touch_comparing_flow` or `run_comparing` — they don't consume
  reduction/evaluation config.

## Success Criteria

- [ ] Setting `reduction.scaler: robust` in the DAG YAML produces
  `cluster_metadata.json` with `"reduction": {"scaler": "robust", ...}`.
- [ ] Setting `evaluation.stability: null` in the DAG YAML produces
  `cluster_metadata.json` with no `"stability"` key (stability is skipped).
- [ ] Setting `evaluation.stability.n_rounds: 5` produces metadata with
  `"stability": {"n_rounds": 5, ...}`.
- [ ] A per-combination `reduction` override in `feature_combinations` wins
  over the task-level default.
- [ ] Omitting `reduction` and `evaluation` from the DAG YAML entirely still
  works (hardcoded defaults in `ReductionPipeline` and `_cluster_combination`
  apply).
- [ ] No regression: existing clustering outputs, RF mapping, and comparing
  pipelines run unchanged.

---

## Technical Design

### Approach

Thread config through the existing call chain with minimal signature changes.
The injection point is `run_clustering`, which already iterates over
combinations — it simply merges the shared config into each combination dict
before passing it to `_cluster_combination`.

**`run_clustering` signature change:**

```python
def run_clustering(
    output_dir: Path,
    feature_combinations: dict,
    clustering_profiles: dict,
    force: bool = False,
    extraction_dir: Path = None,
    reduction: dict = None,       # NEW
    evaluation: dict = None,      # NEW
) -> dict[str, List[Path]]:
```

**Injection logic** (inside the `for combination_name, combination_config`
loop, before `_cluster_combination`):

```python
if reduction is not None and "reduction" not in combination_config:
    combination_config["reduction"] = reduction
if evaluation is not None and "evaluation" not in combination_config:
    combination_config["evaluation"] = evaluation
```

This is a shallow default: if the combination already defines its own
`reduction` or `evaluation`, the task-level value is skipped. Deep-merging
is not proposed (YAGNI — no per-combination overrides exist today).

**`touch_clustering_flow` signature change:**

```python
def touch_clustering_flow(
    input_items: List[Tuple[Path, Path]],
    force_processing: bool = False,
    feature_combinations: dict = None,
    clustering_profiles: dict = None,
    reduction: dict = None,       # NEW
    evaluation: dict = None,      # NEW
) -> List[Path]:
```

Forwards `reduction` and `evaluation` to `run_clustering`.

**`run_batch_analysis` kwargs additions** (two new branches in the options
dispatch block):

```python
if "reduction" in options:
    kwargs["reduction"] = options["reduction"]
if "evaluation" in options:
    kwargs["evaluation"] = options["evaluation"]
```

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Thread as explicit params** (chosen) | Minimal change; clear data flow; backward-compatible (defaults to None) | Two new params on `run_clustering` | **Chosen** |
| Inject in `touch_clustering_flow` before calling `run_clustering` | No change to `run_clustering` signature | Workflow script owns pipeline-internal config merging — wrong layer | Rejected |
| Nest `reduction`/`evaluation` inside each `feature_combinations` entry in YAML | No code change needed | Redundant config; GUI already writes them at task level; breaks existing YAML | Rejected |
| Deep-merge per-combination overrides with task-level defaults | More flexible | No use-case today; complicates reasoning about which config wins | Rejected (YAGNI) |

### Architecture Constraints (from knowledge base)

- **CuPy import order** — N/A (no new imports of preprocessing).
- **Qt signal recursion** — N/A (no GUI changes).
- **Git merge** — `--no-ff` per project convention.

### Architecture Changes

No new modules or classes. Signature changes only:

```
code/scripts/analysis_workflow.py
├── touch_clustering_flow()      — +2 params: reduction, evaluation
└── run_batch_analysis()         — +2 kwargs forwarding branches

code/src/analysis/touch_analytics/clustering_pipeline.py
└── run_clustering()             — +2 params: reduction, evaluation
                                   +injection logic in combination loop
```

---

## Implementation Plan

### Phase 1: Add parameters and forwarding
**Goal:** Thread `reduction` and `evaluation` from DAG config to the
clustering pipeline so user-configured values take effect.
**Started:** —
**Completed:** —

- [ ] Task 1.1 — Add `reduction: dict = None` and `evaluation: dict = None`
  parameters to `run_clustering()` in `clustering_pipeline.py:175`.
- [ ] Task 1.2 — Inside `run_clustering`, after line 240
  (`for combination_name, combination_config`), inject task-level
  `reduction`/`evaluation` into `combination_config` when the combination
  doesn't already define its own.
- [ ] Task 1.3 — Add `reduction: dict = None` and `evaluation: dict = None`
  parameters to `touch_clustering_flow()` in `analysis_workflow.py:128`.
  Forward them to `run_clustering()`.
- [ ] Task 1.4 — Add `"reduction"` and `"evaluation"` forwarding branches
  to `run_batch_analysis()` in `analysis_workflow.py` (after the existing
  `clustering_profiles` branch, around line 409).

**Files Modified:**
- `code/src/analysis/touch_analytics/clustering_pipeline.py` — `run_clustering`
  signature + injection logic (~6 lines).
- `code/scripts/analysis_workflow.py` — `touch_clustering_flow` signature
  + `run_batch_analysis` kwargs (~8 lines).

**Dependencies:** None.

---

## Testing Plan

### Unit Tests

- [ ] `test_run_clustering_forwards_reduction` — call `run_clustering` with
  `reduction={"scaler": "robust"}` and a single combination that has no
  `reduction` key. Assert `cluster_metadata.json` contains
  `"reduction": {"scaler": "robust", ...}`.
- [ ] `test_run_clustering_combination_override_wins` — call `run_clustering`
  with `reduction={"scaler": "robust"}` but the combination dict already has
  `reduction: {"scaler": "none"}`. Assert metadata shows `"scaler": "none"`.
- [ ] `test_run_clustering_no_reduction_param` — call `run_clustering` without
  `reduction` param. Assert pipeline uses default (`"standard"`).
- [ ] `test_run_clustering_evaluation_null_stability` — call with
  `evaluation={"stability": None}`. Assert metadata has no `"stability"` key.

### Integration Tests

- [ ] Run `touch_clustering_flow` with the full DAG YAML config (including
  `reduction.scaler: standard` and `evaluation.stability.n_rounds: 20`).
  Assert `cluster_metadata.json` reflects those values.

### Manual Verification

- [ ] Edit `analyse_workflow_dag.yaml`: set `scaler: robust`. Run the
  workflow on a single session. Inspect `cluster_metadata.json` — confirm
  `"reduction": {"scaler": "robust", ...}`.
- [ ] Set `stability: null` in YAML. Run. Confirm no `"stability"` key in
  metadata.
- [ ] Restore defaults. Run. Confirm output matches pre-fix behaviour
  (standard scaler, 20 bootstrap rounds).
- [ ] Open the GUI, select `touch_clustering`, open Reduction dialog, change
  scaler to `robust`, save YAML, run from GUI. Confirm metadata reflects the
  change.

### Edge Cases

- [ ] DAG YAML missing both `reduction` and `evaluation` keys entirely —
  pipeline uses hardcoded defaults (no crash).
- [ ] `reduction: {}` (empty dict) — pipeline uses default scaler
  (`"standard"`).
- [ ] `evaluation: {}` (empty dict) — stability defaults to 20 rounds.

---

## Documentation Plan

- [ ] No user-facing doc changes — this is a bug fix for existing config
  that was already documented in the YAML comments.
- [ ] Docstring update on `run_clustering` to document the new parameters.

---

## Rollback Plan

1. Revert the commit(s) from this plan. The three signature changes are
   backward-compatible (all new params default to `None`), so reverting does
   not break any existing callers.
2. No data migration — output schema is unchanged.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Injecting `reduction`/`evaluation` into `combination_config` mutates the caller's dict | Medium | Low | Use `combination_config = {**combination_config}` (shallow copy) before injection so the original dict is not modified across iterations |
| A future per-combination `reduction` override behaves unexpectedly because it fully replaces (not merges with) the task-level config | Low | Low | Documented as shallow default; deep-merge can be added later if needed |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — Add parameters and forwarding | ~0.5 day | None |

---

## References

- **Alignment plan:**
  `docs/development/plans/active/align-clustering-pipeline-with-guidelines.md`
- **Pipeline guidelines:**
  `docs/design/timeseries_clustering_pipeline_guidelines.md`
- **Reduction pipeline:** `code/src/analysis/touch_analytics/reduction/pipeline.py:63`
  — reads `config.get("reduction", {})`
- **Evaluation config read:** `code/src/analysis/touch_analytics/clustering_pipeline.py:409`
  — reads `combination_config.get('evaluation', {})`
- **DAG config:** `configs/analyse_workflow_dag.yaml:135–145` — `reduction` and
  `evaluation` blocks
