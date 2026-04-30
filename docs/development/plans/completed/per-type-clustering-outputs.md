# Plan: Per-Type Clustering Outputs

**Date:** 2026-04-29
**Author:** Basil Duvernoy
**Status:** Completed
**Base Branch:** `feature/hand-velocity-scalars-and-cartesian-binning`
**Branch:** `feature/per-type-clustering-outputs`

---

## Overview

The clustering pipeline currently pools all touch types (`tap`, `stroke_proximal`,
`stroke_distal`) into one feature matrix, producing a single `feature_space.png` and
`pooled_touch_summary_clustered.csv` per `(group, clusterer)` pair. This plan adds a
`per_type_clustering: true` flag to cluster groups that runs reduction, clustering, evaluation,
and visualization independently for each touch type, writing separate subdirectories with their
own `feature_space.png`, CSV, and metadata.

## Problem Statement

Clustering across mixed touch types blends kinematically distinct regimes (e.g. tap vs. stroke).
The resulting clusters are hard to interpret: a tap cluster and a stroke cluster that share similar
pressure/velocity values will be merged, obscuring type-specific structure. The existing
`TypeStratifiedClusterer` wrapper clusters per type but then merges all results into one output
file and one `feature_space.png` — giving no per-type visual or CSV. The requirement is that
clusters must be defined and exported independently per touch type, with individual
`feature_space.png` plots that show only same-type data.

## Goals

### In Scope
1. Add `per_type_clustering: true/false` group-level config flag to `clustering_pipeline.py`
2. When enabled: split pooled data by `gesture_type`, fit reduction independently per type,
   run each clusterer per type, write outputs to `<group>/<clusterer>/<type>/` subdirectories
3. Each type-specific subdirectory contains: `pooled_touch_summary_clustered.csv`,
   `cluster_metadata.json`, `feature_space.png`, `heatmaps/`
4. Enable the flag for the active `pressure_velocity_mean_cartesian_binning` cluster group
   in the DAG config
5. Preserve existing flat-output behavior when the flag is absent or `false`

### Out of Scope
- Updating `rf_cluster_pipeline.py` to consume per-type subdirectory paths (follow-on plan)
- Deprecating `TypeStratifiedClusterer` (different use case: single merged CSV for RF mapping)
- Changing feature extraction or series transforms

## Success Criteria

- [ ] Running `touch_clustering` with `per_type_clustering: true` produces three subdirectories —
      `tap/`, `stroke_proximal/`, `stroke_distal/` — under `<group>/<clusterer>/`
- [ ] Each subdirectory contains its own `feature_space.png` rendered on that type's data only
- [ ] Each `pooled_touch_summary_clustered.csv` contains only rows for its gesture type
- [ ] `cluster_metadata.json` sample counts across the three types sum to approximately
      the flat total
- [ ] Existing cluster groups without the flag produce unchanged flat output
- [ ] Re-running without `force_processing: true` skips up-to-date type subdirectories

---

## Technical Design

### Approach

Refactor `_cluster_combination()` into two layers:

1. **Outer layer** (`_cluster_combination()`): reads `per_type_clustering` from config and
   decides whether to loop over types or run flat. Feature column selection is done once on the
   full pooled DataFrame (ensures consistent column set across types).

2. **Inner layer** (extracted `_run_reduction_and_clusterers()` helper): given a
   (possibly type-filtered) DataFrame + an output base path, runs:
   reduction → per-clusterer loop → save CSV/JSON → render PNG → heatmaps.
   This contains all logic currently inside the `for clusterer_name in clustering_profiles:`
   loop in `_cluster_combination()`.

When `per_type_clustering: true`:
```
for gesture_type in ['tap', 'stroke_proximal', 'stroke_distal']:
    type_pooled = pooled[pooled['gesture_type'] == gesture_type]
    # skip if empty — log warning
    for clusterer_name, config in clustering_profiles.items():
        out_dir = output_dir / combination_name / clusterer_name / gesture_type
        # reduction fitted on type_pooled only
        # clusterer run on type_pooled
```

When `per_type_clustering: false` / absent (unchanged behavior):
```
for clusterer_name, config in clustering_profiles.items():
    out_dir = output_dir / combination_name / clusterer_name
    # reduction fitted on full pooled
    # clusterer run on full pooled
```

The reduction is fitted once per type (outside the per-clusterer loop) so all clusterers
within a type share the same scaled space — matching the existing behavior for the flat case.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `per_type_clustering` group-level flag | Opt-in, backward compat, no RF pipeline impact until ready | Requires DAG config update | **Chosen** |
| Always split per type (no flag) | No config needed | Breaks backward compat; RF pipeline breaks immediately | Rejected |
| Reuse `TypeStratifiedClusterer` | Already exists | Merges to one output; no per-type PNGs or CSVs | Rejected |
| Post-process flat clusters by splitting | No core change needed | Reduction fitted globally contaminates per-type scaling | Rejected |

### Architecture Changes

**New private function** `_run_reduction_and_clusterers()` in `clustering_pipeline.py`:

```python
def _run_reduction_and_clusterers(
    combination_name: str,
    combination_config: dict,
    pooled: pd.DataFrame,       # type-filtered or full
    out_base: Path,             # output_dir / combination_name (type subdir appended inside)
    clustering_profiles: dict,
    force: bool,
    feature_cols: list[str],
    gesture_type: str | None,   # None → flat output; str → type subdir appended
) -> dict[str, List[Path]]:
```

Contains: reduction pipeline, per-clusterer loop, GMM/cartesian `feature_space.png` rendering,
heatmap generation. Currently this is the body of `_cluster_combination()` lines ~512–802.

**Updated `_cluster_combination()`**:

```python
def _cluster_combination(...):
    feature_cols = _select_feature_columns(pooled, ...)   # unchanged
    per_type = combination_config.get('per_type_clustering', False)
    if per_type:
        for gesture_type in _GESTURE_TYPES:
            type_pooled = pooled[pooled['gesture_type'] == gesture_type]
            if type_pooled.empty:
                logging.warning(f"[{combination_name}] No touches for type '{gesture_type}' — skipping.")
                continue
            _run_reduction_and_clusterers(..., pooled=type_pooled, gesture_type=gesture_type)
    else:
        _run_reduction_and_clusterers(..., pooled=pooled, gesture_type=None)
```

**Output directory structure comparison:**

Without flag (unchanged):
```
touch_clusters/
  pressure_velocity_mean_cartesian_binning/
    cartesian_binning/
      pooled_touch_summary_clustered.csv
      cluster_metadata.json
      feature_space.png
      heatmaps/
```

With `per_type_clustering: true`:
```
touch_clusters/
  pressure_velocity_mean_cartesian_binning/
    cartesian_binning/
      tap/
        pooled_touch_summary_clustered.csv
        cluster_metadata.json
        feature_space.png
        heatmaps/
      stroke_proximal/
        pooled_touch_summary_clustered.csv
        cluster_metadata.json
        feature_space.png
        heatmaps/
      stroke_distal/
        pooled_touch_summary_clustered.csv
        cluster_metadata.json
        feature_space.png
        heatmaps/
```

**`_GESTURE_TYPES` constant**: define locally in `clustering_pipeline.py` as
`_GESTURE_TYPES = ['tap', 'stroke_proximal', 'stroke_distal']` to avoid coupling the
pipeline to the `TypeStratifiedClusterer` wrapper.

**`ClusteringContext`**: when `gesture_type is not None`, all rows in `type_pooled` have the
same gesture type, so `context.gesture_type_labels` will be a uniform array. This is fine —
no clusterer currently uses `gesture_type_labels` except `TypeStratifiedClusterer`, which
won't be invoked for per-type runs.

**`_generate_session_heatmaps()`**: receives a single-type `result_df`. The existing loop over
`('tap', 'stroke_proximal', 'stroke_distal')` produces one populated column and two empty —
acceptable without change.

**Return value of `run_clustering()`**: when `per_type_clustering`, keys change from
`"<group>/<clusterer>"` to `"<group>/<clusterer>/<type>"`. The RF pipeline reads disk
directly and is unaffected by this change.

---

## Implementation Plan

### Phase 1: Refactor and add per-type branching

**Started:** 2026-04-30
**Completed:** 2026-04-30

**Goal:** Extract inner clusterer logic into `_run_reduction_and_clusterers()` and add the
per-type outer loop in `_cluster_combination()`.

- [x] Define `_GESTURE_TYPES = ['tap', 'stroke_proximal', 'stroke_distal']` near top of
      `clustering_pipeline.py`
- [x] Extract the body of the `for clusterer_name, clusterer_config in clustering_profiles.items():`
      loop in `_cluster_combination()` (lines ~540–800) into
      `_run_reduction_and_clusterers(combination_name, combination_config, pooled,
      out_base, clustering_profiles, force, feature_cols, gesture_type)`.
      - `out_dir` inside becomes `out_base / clusterer_name` if `gesture_type is None`,
        else `out_base / clusterer_name / gesture_type`
      - Reduction (`ReductionPipeline.fit_transform`) moves inside this helper, fitted on the
        passed `pooled` — this preserves per-type independence
- [x] Rewrite `_cluster_combination()` to:
      - Select feature columns once from the full `pooled`
      - Check `combination_config.get('per_type_clustering', False)`
      - Branch: type loop → `_run_reduction_and_clusterers()` per type vs. single call

**Files Modified:**
- `code/src/analysis/touch_analytics/clustering_pipeline.py` — refactor + per-type loop

**Dependencies:** None

### Phase 2: DAG config update

**Started:** 2026-04-30
**Completed:** 2026-04-30

**Goal:** Enable per-type clustering for the active cluster group.

- [x] Add `per_type_clustering: true` under
      `touch_clustering.options.cluster_groups.pressure_velocity_mean_cartesian_binning`
      in `configs/analyse_workflow_dag.yaml`

**Files Modified:**
- `configs/analyse_workflow_dag.yaml`

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests

No new unit tests required — existing clusterer unit tests are unaffected. The refactoring
is structural (extract + branch), not algorithmic.

### Manual Verification

- [ ] Run `touch_clustering` with `force_processing: true` on one session
- [ ] Confirm `pressure_velocity_mean_cartesian_binning/cartesian_binning/` contains
      `tap/`, `stroke_proximal/`, `stroke_distal/` subdirectories
- [ ] Open each `feature_space.png` — verify it shows only data for its own type
- [ ] Check `cluster_metadata.json` sample counts sum to approximately the flat total
- [ ] Confirm a cluster group without `per_type_clustering` produces unchanged flat output
- [ ] Re-run without `force_processing: true` — confirm per-type outputs are skipped as
      up-to-date

### Edge Cases

- [ ] A gesture type absent from the pooled data: guard `if type_pooled.empty` logs and skips
- [ ] A gesture type with fewer touches than `min_touches_per_component` (GMM):
      existing `max_k = max(1, ...)` clamp handles this

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — note `per_type_clustering` flag in clustering
      stage description
- [ ] No user guide needed (config-driven, no new user-facing commands)

---

## Rollback Plan

1. Remove `per_type_clustering: true` from the DAG config — pipeline reverts to flat output
   on next `force` run
2. Per-type subdirectories already on disk do not interfere with flat-output mode (different
   paths); delete manually if disk space is a concern

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `rf_cluster_pipeline.py` fails to find flat CSV (path now `<type>/pooled_...csv`) | High | Med | Out of scope — disable `extract_receptive_fields_clustered` in DAG until RF follow-on plan is implemented |
| Very few touches for one type causes degenerate clustering | Med | Low | Empty-type guard; GMM `max_k` clamp; existing stability/metric warnings unchanged |
| Reduction fitted per-type changes cluster semantics vs. existing results | Low | Low | Expected and desired; re-run with `force_processing: true` to regenerate |

---

## References

- Related memory: `analysis_pipeline_issue_mixed_types_in_clusters.md`
- Related plan: (RF follow-on to consume per-type subdirectories — not yet created)
- Key file: `code/src/analysis/touch_analytics/clustering_pipeline.py` (lines 493–802 — `_cluster_combination`)
- Key file: `configs/analyse_workflow_dag.yaml` (lines 231–272 — `pressure_velocity_mean_cartesian_binning`)
- `TypeStratifiedClusterer`: `code/src/analysis/touch_analytics/clustering/type_stratified_clusterer.py` — left unchanged
