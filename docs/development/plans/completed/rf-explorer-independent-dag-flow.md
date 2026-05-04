# Plan: RF Explorer — Independent DAG Flow

**Date:** 2026-05-01
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-04 14:39
**Base Branch:** `feature/rf-feature-space-explorer`
**Branch:** `feature/rf-explorer-independent-dag-flow`

---

## Overview

**What:** Promote the RF Feature-Space Explorer launch from an embedded option
flag on `visualize_receptive_fields_clustered` into its own first-class DAG
task with a direct dependency on `touch_series_transforms`.

**Why:** The explorer consumes series-augmented CSVs, not RF clustering
artifacts. Tying its launch to the clustering visualization task creates a
false dependency and hides the explorer behind a flag that conceptually belongs
to a different pipeline stage.

**How:** Add `explore_rf_feature_space_flow` as a new `@flow` in
`analysis_workflow.py`, register it in `available_tasks`, add the task to
`analyse_workflow_dag.yaml` with `depends_on: [touch_series_transforms]`, and
strip `feature_space_explorer` from `visualize_receptive_fields_clustered` and
`run_cluster_rf_visualization`.

---

## Problem Statement

`launch_feature_space_explorer()` reads series-augmented CSVs produced by
`touch_series_transforms`. It has no logical dependency on RF clustering or
visualization. Yet it is currently triggered via `feature_space_explorer: true`
inside the `visualize_receptive_fields_clustered` task, meaning:

1. A user must enable `visualize_receptive_fields_clustered` to launch the
   explorer, even if they have no interest in the cluster-based RF pipeline.
2. The option is invisible in the DAG overview — it looks like a rendering flag,
   not a separate GUI tool.
3. The explorer cannot be enabled or disabled independently of RF visualization.

---

## Goals

### In Scope

1. New `explore_rf_feature_space_flow` Prefect flow in `analysis_workflow.py`
2. New `explore_rf_feature_space` DAG task in `analyse_workflow_dag.yaml` with
   `depends_on: [touch_series_transforms]`
3. Remove `feature_space_explorer` parameter and call from
   `visualize_receptive_fields_clustered_flow` in `analysis_workflow.py`
4. Remove `feature_space_explorer` parameter and `launch_feature_space_explorer`
   call from `run_cluster_rf_visualization()` in `rf_cluster_pipeline.py`
5. Update the active plan `rf-feature-space-explorer.md` to reflect the revised
   Phase 4 architecture

### Out of Scope

- Changes to `launch_feature_space_explorer()` itself
- Changes to `RFFeatureSpaceExplorer` GUI
- Any new options or parameters on the explorer task
- DAG launcher GUI changes (the task appears automatically via the YAML)

---

## Success Criteria

- [ ] `explore_rf_feature_space` appears as an independent task in
  `analyse_workflow_dag.yaml` with `depends_on: [touch_series_transforms]`
- [ ] Running the analysis workflow with only `touch_preparation`,
  `touch_series_transforms`, and `explore_rf_feature_space` enabled launches
  the explorer without error
- [ ] `visualize_receptive_fields_clustered` no longer accepts or forwards a
  `feature_space_explorer` flag
- [ ] `run_cluster_rf_visualization()` no longer accepts or calls
  `launch_feature_space_explorer`
- [ ] Active plan `rf-feature-space-explorer.md` Phase 4 is updated to match

---

## Technical Design

### Approach

The new flow is minimal: it receives `input_items` (the standard
`List[Tuple[Path, Path]]` of aggregated CSV + database path) and calls
`launch_feature_space_explorer(input_items)` directly. Path resolution for
series-augmented CSVs is already handled inside `launch_feature_space_explorer`
— no `series_dir` injection is needed at the flow level.

`force_processing` is accepted by the flow signature (consistent with all other
tasks) but is a no-op — the GUI is stateless and always launches fresh.

No special kwargs are needed in `run_batch_analysis` beyond the standard
`input_items` + `force_processing` pair, so the kwargs block requires no new
branch for this task.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Keep as option on `visualize_receptive_fields_clustered` | No code change needed beyond YAML | Wrong dependency, invisible in DAG, cannot be enabled independently | Rejected |
| Pass `series_dir` explicitly to the flow | Consistent with `touch_feature_extraction_flow` | Unnecessary — `launch_feature_space_explorer` already resolves paths internally | Rejected |
| Separate launcher script entry in `launcher.yaml` | Fully independent GUI launch | Duplicates session config resolution; the DAG task approach is already the established pattern for optional GUI steps | Rejected |

### Architecture Changes

Three files modified, one file updated (active plan):

```
code/scripts/analysis_workflow.py
    - Add @flow explore_rf_feature_space_flow()
    - Add ("explore_rf_feature_space", explore_rf_feature_space_flow) to available_tasks
    - Remove feature_space_explorer param from visualize_receptive_fields_clustered_flow
    - Remove feature_space_explorer kwarg forwarding in run_batch_analysis

code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py
    - Remove feature_space_explorer param from run_cluster_rf_visualization()
    - Remove launch_feature_space_explorer() call inside that function

configs/analyse_workflow_dag.yaml
    - Add explore_rf_feature_space task block
    - Remove feature_space_explorer line from visualize_receptive_fields_clustered

docs/development/plans/active/rf-feature-space-explorer.md
    - Update Phase 4 description and tasks to reflect new architecture
```

---

## Implementation Plan

### Phase 1: Workflow Script

**Goal:** Add the new flow and remove the old option from
`analysis_workflow.py`.

**Started:** 2026-05-01

- [x] Task 1.1 — Add `explore_rf_feature_space_flow` `@flow` after
  `visualize_receptive_fields_clustered_flow`; body calls
  `launch_feature_space_explorer(input_items)` with a guarded import
- [x] Task 1.2 — Add `("explore_rf_feature_space", explore_rf_feature_space_flow)`
  to `available_tasks` in `run_batch_analysis`, after
  `visualize_receptive_fields_clustered`
- [x] Task 1.3 — Remove `feature_space_explorer: bool = False` param from
  `visualize_receptive_fields_clustered_flow` signature and the corresponding
  kwarg forwarding to `run_cluster_rf_visualization`
- [x] Task 1.4 — Remove the `if "feature_space_explorer" in options:` kwargs
  branch in `run_batch_analysis`

**Files Modified:**
- `code/scripts/analysis_workflow.py`

**Dependencies:** None

### Phase 2: Pipeline Module Cleanup

**Goal:** Remove `feature_space_explorer` from `run_cluster_rf_visualization`.

- [x] Task 2.1 — Remove `feature_space_explorer: bool = False` param from
  `run_cluster_rf_visualization()` in `rf_cluster_pipeline.py`
- [x] Task 2.2 — Remove the `if feature_space_explorer:` block (call to
  `launch_feature_space_explorer`) at the end of that function

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py`

**Dependencies:** Phase 1

### Phase 3: DAG Config + Plan Doc Update

**Goal:** Surface the new task in the DAG and update the active plan.

- [x] Task 3.1 — Add `explore_rf_feature_space` task block to
  `analyse_workflow_dag.yaml` after `touch_series_transforms`, with
  `enabled: true` and `depends_on: [touch_series_transforms]`
- [x] Task 3.2 — Remove the `feature_space_explorer: true` line from the
  `visualize_receptive_fields_clustered` task block in the same YAML
  (line was never present in the YAML — the flag existed only in the code layer)
- [x] Task 3.3 — Update Phase 4 of `docs/development/plans/active/rf-feature-space-explorer.md`
  to document the revised integration architecture

**Files Modified:**
- `configs/analyse_workflow_dag.yaml`
- `docs/development/plans/active/rf-feature-space-explorer.md`

**Dependencies:** Phase 2

---

## Testing Plan

### Manual Verification

- [ ] Run analysis workflow with only `touch_preparation`,
  `touch_series_transforms`, `explore_rf_feature_space` enabled — explorer
  launches correctly
- [ ] Run analysis workflow with `explore_rf_feature_space` disabled and
  `visualize_receptive_fields_clustered` enabled — no error, no explorer
  window
- [ ] Run full analysis workflow with both tasks enabled — both gallery viewer
  and explorer launch independently

### Edge Cases

- [ ] `explore_rf_feature_space` enabled but `touch_series_transforms`
  disabled — `launch_feature_space_explorer` raises `ValueError` on missing
  series CSV (fail-fast, expected behavior)
- [ ] Confirm `run_cluster_rf_visualization` still runs correctly without
  `feature_space_explorer` param (no regression)

---

## Documentation Plan

- [ ] Update `docs/development/plans/active/rf-feature-space-explorer.md`
  Phase 4 (covered in Task 3.3 above)
- [ ] No CLAUDE.md update needed — no architectural pattern is introduced

---

## Rollback Plan

All changes are in four files. To revert:

1. Restore `feature_space_explorer: bool = False` param in
   `visualize_receptive_fields_clustered_flow` and re-add forwarding
2. Restore `feature_space_explorer` param and `launch_feature_space_explorer`
   call in `run_cluster_rf_visualization`
3. Remove the `explore_rf_feature_space` task block from the DAG YAML and
   restore the `feature_space_explorer: true` line
4. Remove the new flow and `available_tasks` entry from `analysis_workflow.py`

No data migrations, no breaking changes to external interfaces.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `launch_feature_space_explorer` import path breaks when called from new flow | Low | Medium | Import is guarded inside the flow body (same pattern as `gallery_viewer` launch) |
| `run_cluster_rf_visualization` callers outside `analysis_workflow.py` still pass `feature_space_explorer` | Low | Low | Grep for all call sites before removing the param |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Workflow script | ~30 min | None |
| Phase 2: Pipeline cleanup | ~15 min | Phase 1 |
| Phase 3: DAG config + plan doc | ~15 min | Phase 2 |

---

## References

- Active plan: `docs/development/plans/active/rf-feature-space-explorer.md`
- Pipeline module: `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py`
- Workflow script: `code/scripts/analysis_workflow.py`
- DAG config: `configs/analyse_workflow_dag.yaml`
