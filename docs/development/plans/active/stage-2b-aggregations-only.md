# Plan: Reduce Stage 2b feature extraction to aggregations only

**Date:** 2026-04-23
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/stage-2b-aggregations-only`

---

## Overview

Rewrite the `touch_feature_extraction` node so its only options are *aggregation techniques* (mean, median, min, max, std, range, skewness, …) applied to every quantitative column produced by Stage 2a. Delete the complex extractors (`temporal`, `mechanics_of_solids`, `pressure_velocity_*`) because their outputs are already reproducible as aggregations of Stage 2a-augmented columns. Promote `touch_category` out of the feature list into its own top-level segmentation toggle.

## Problem Statement

`docs/design/timeseries_clustering_pipeline_guidelines.md` defines Stage 2b as "collapse each entity into a fixed-length vector of scalar descriptors" — pure aggregations of per-frame series. The current node violates that contract:

- `StatisticalExtractor` hardcodes four kinematic variables (depth, area, velocity, acceleration) instead of sweeping whatever quantitative channels Stage 2a leaves in the augmented CSV.
- `TemporalExtractor`, `MechanicsOfSolidsExtractor`, and `PressureExtractor` compute domain-specific derived scalars that are either Stage 2a concerns or can be re-expressed as `aggregation(Stage-2a-column)`.
- `TouchCategoryExtractor` emits one-hot metadata (is_tap, is_stroke, dir_proximal, dir_distal) — not an aggregation at all. It is conceptually a segmentation layer *above* the aggregation sweep, not a peer of it.

This mixing forces every future channel/feature to grow a bespoke extractor, duplicates aggregation logic, and obscures what the clustering stage is actually consuming. Now that Stage 2a (`series_pipeline.run_series_transforms`) persists `velocity_magnitude`, `acceleration_magnitude`, `geo_pressure`, and `mos_*` columns into `<session>_series_augmented.csv`, all of those "complex" features become trivial aggregations — there is no reason to keep the bespoke code paths.

## Goals

### In Scope
1. Rewrite `StatisticalExtractor` to auto-discover quantitative columns from the input DataFrame and apply the configured aggregation(s) to each.
2. Delete `TemporalExtractor`, `MechanicsOfSolidsExtractor`, `PressureExtractor` and all their registry/DAG wiring.
3. Lift `touch_category` to a top-level, independent option under `touch_feature_extraction.options` (sibling of `features:`), with its own `enabled` checkbox in the GUI.
4. Rewrite `configs/analyse_workflow_dag.yaml` `feature_combinations` entries that reference deleted features so the clustering stage still runs.
5. Update downstream consumers that hardcode the old short column names (`depth_mean`, `velocity_mean`, `area_mean`) — specifically `touch_config.KINEMATIC_SIGNALS` and `matrix_generation.py` heatmap axis selection — to use the raw column names (`contact_depth_mean`, `velocity_magnitude_mean`, `contact_area_mean`).
6. GUI: in `task_detail_panel.py`, render `touch_category` as a separate group box above the aggregation grid for the `touch_feature_extraction` task.

### Out of Scope
- New aggregation functions beyond the existing seven. `skewness` stays; `kurtosis`/`mad`/`quantile` are future additions.
- Stage 2a changes. `series_pipeline.py` is already complete (per the active `split-feature-extraction-into-2a-2b-flows` plan); this plan assumes its augmented CSV contract is final.
- Touching `touch_comparing`, `analyse_ap_efficacy`, or the RF heatmap rendering code beyond the column-name rename.
- Backward-compatible translation of the old `extraction_profiles` / removed-feature configs — the DAG YAML is updated in lockstep with the code.
- Fixing the `reduction` / `evaluation` forwarding gap (tracked separately in `fix-analysis-workflow-stage-config-forwarding.md`).

## Success Criteria

- [ ] `touch_feature_extraction.options.features` in the DAG YAML contains only aggregation names (mean, median, min, max, std, range, skewness).
- [ ] `touch_feature_extraction.options.touch_category.enabled` toggles category-one-hot CSV output independently of the aggregation list.
- [ ] With Stage 2a fully enabled (kinematics + pressure + mos) and `mean` enabled in Stage 2b, `4_analysed/touch_features/mean/<session>_touch_summary.csv` contains a `{col}_mean` column for every quantitative column in the augmented CSV (including `contact_depth_mean`, `contact_area_mean`, `contact_location_{x,y,z}_mean`, `velocity_magnitude_mean`, `acceleration_magnitude_mean`, `geo_pressure_mean`, and each `mos_*_mean`) and *no* aggregation of `block_order_id`, `trial_id`, `Nerve_spike`, or `sticker_*_position_*`.
- [ ] Enabling any removed feature name (`temporal`, `mechanics_of_solids`, `pressure_velocity_mean`, `pressure_velocity_max`) under `features:` raises a clear error that points the user to the new aggregation-only contract.
- [ ] The `mean_and_type` clustering combination (rewritten from `pressure_velocity_mean_and_type`) runs end-to-end with the `type_stratified` clusterer.
- [ ] `map_receptive_fields_clustered` renders heatmaps for the `only_mean` combination using the new column names (no `KeyError` on `velocity_mean`, `depth_mean`, `area_mean`).
- [ ] GUI: opening the DAG launcher on `analyse_workflow_dag.yaml` shows `touch_feature_extraction` with a `touch_category` checkbox in its own section above an aggregation-only grid.
- [ ] No references remain in `code/src/analysis/touch_analytics/` to `TemporalExtractor`, `MechanicsOfSolidsExtractor`, or `PressureExtractor`.

---

## Technical Design

### Approach

**Single-path aggregation sweep.** `StatisticalExtractor.extract(group, config)` iterates over every numeric column in `group` that is not on an explicit exclude list, producing `{column}_{agg}` for each configured aggregation. All other extractors are deleted; their outputs were already a strict subset of what this sweep produces once Stage 2a is enabled.

`touch_category` is moved to its own top-level config key and runs as a separate output branch in `extraction_pipeline.run_feature_extraction`, writing `<output_dir>/touch_category/<session>_touch_summary.csv` independently of the per-aggregation folders.

Column naming uses the raw Stage 2a column name verbatim (`contact_depth_mean`, `velocity_magnitude_mean`) — no alias mapping — because aliasing is fragile and the only consumer of the old short names (`matrix_generation.py`) is updated in the same change.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Auto-discover quantitative columns + exclude list (this plan) | One code path; adding a Stage 2a channel automatically propagates; matches the "aggregations only" intent. | Requires an exclude set that must be audited; breaking column-name change for downstream code. | **Chosen** |
| Keep all current extractors, disable them via config | Minimal code churn. | Does not fix the design violation; extractors remain fossils; doesn't generalise to new channels. | Rejected |
| Explicit `channels: [...]` whitelist in the Stage 2b config | Exact control; no exclude-list guesswork. | Adding a Stage 2a channel needs *two* config edits; re-introduces the coupling between 2a and 2b configs. | Rejected (user selected auto-discover) |
| Aggregate every numeric column with no exclusions | Simplest rule. | Produces meaningless outputs (`Nerve_spike_mean`, `block_order_id_mean`, `sticker_*_mean`). | Rejected |
| Alias short names (`depth_mean`, `velocity_mean`) via rename map | Preserves downstream column names, no ripple. | Extra indirection; rename map is a long-term maintenance hazard and the old names (`depth` rather than `contact_depth`) are less informative. | Rejected |
| Keep `touch_category` as a `features:` entry | No shape change. | Contradicts "only aggregation techniques"; misrepresents a metadata one-hot as a feature peer of aggregations. | Rejected (user explicitly requested independent checkbox) |

### Architecture Constraints

- Stage 2a is the sole source of quantitative channels. Anything Stage 2b should aggregate must already be a column in `<session>_series_augmented.csv`. If a new channel is wanted (e.g. jerk), it is added in `series_pipeline.py`; Stage 2b picks it up automatically.
- `should_process_task` / `clean_task_outputs` idempotency utilities already used in `extraction_pipeline.py` must also guard the new `touch_category/` output path.
- The `spike_elicited` binary column is already derived from `Nerve_spike` inside `_extract_all_touches` and written into shared columns; aggregating `Nerve_spike` itself is both redundant and nonsensical — hence its place on the exclude list.
- **Knowledge-base relevance check:** no applicable notes. The closest candidate (`note-somatosensory-units-and-calculations.md`) documents input unit conventions that are unchanged by this plan.

### Architecture Changes

Existing files modified (no new modules, no new abstractions):

```
code/src/analysis/touch_analytics/
├── representation/
│   └── feature_characterization/
│       ├── statistical.py                    [rewrite extract() + add exclude set]
│       ├── __init__.py                       [trim registry + __all__]
│       ├── temporal.py                       [DELETE]
│       ├── mos.py                            [DELETE]
│       ├── pressure.py                       [DELETE]
│       └── touch_category.py                 [unchanged]
├── extraction_pipeline.py                    [add touch_category branch; reject legacy names]
└── touch_config.py                           [update KINEMATIC_SIGNALS]
└── matrix_generation.py                      [update heatmap axis column names]

code/scripts/analysis_workflow.py              [forward touch_category kwarg]
code/src/utils/gui/dag_launcher/task_detail_panel.py  [render touch_category separately]
configs/analyse_workflow_dag.yaml              [rewrite features block + feature_combinations]
```

### Exclude set (canonical list, to live in `statistical.py`)

```python
EXCLUDED_QUANTITATIVE_COLUMNS: frozenset[str] = frozenset({
    # Identifiers / indices
    'block_order_id', 'trial_id', 'single_touch_id', 'frame_id',
    # Time base
    'time', 't', 'timestamp',
    # Event channel — binary; spike_elicited is the aggregate
    'Nerve_spike',
    # Raw sticker positions — subsumed by velocity_magnitude / acceleration_magnitude
    'sticker_blue_position_x', 'sticker_blue_position_y', 'sticker_blue_position_z',
    'sticker_green_position_x', 'sticker_green_position_y', 'sticker_green_position_z',
    'sticker_red_position_x',  'sticker_red_position_y',  'sticker_red_position_z',
})
```

### Output column naming migration

| Before | After |
|--------|-------|
| `depth_{agg}` | `contact_depth_{agg}` |
| `area_{agg}` | `contact_area_{agg}` |
| `velocity_{agg}` | `velocity_magnitude_{agg}` |
| `acceleration_{agg}` | `acceleration_magnitude_{agg}` |
| `geo_pressure_mean`, `geo_pressure_max` (from PressureExtractor) | `geo_pressure_{agg}` (from StatisticalExtractor) |
| `geo_velocity_mean`, `geo_velocity_max` | `velocity_magnitude_{agg}` (same column as above) |
| `strain_max`, `strain_mean` (from MoSExtractor) | `mos_strain_{agg}` |
| `stress_max_kpa`, `stress_mean_kpa` | `mos_stress_kpa_{agg}` |
| `strain_rate_max` | `mos_strain_rate_{agg}` |
| `elastic_energy_max_mj`, `elastic_energy_total_mj` | `mos_elastic_energy_mj_{agg}` (note: `_total` is no longer produced — sum is not in the aggregation set) |
| `impulse_total_mns` | `mos_impulse_mns_{agg}` (sum not produced; see note below) |

**Note on `_total` / sum aggregations.** The current MoS extractor produces `elastic_energy_total_mj` (sum) and `impulse_total_mns` (sum). The new aggregation set does not include `sum`. Two options if sums are still needed downstream:
- (a) Add `sum` to `AGGREGATION_NAMES` in a follow-up.
- (b) Accept that `_mean × duration_frames` recovers the sum with a multiplication at analysis time.

This plan leaves `sum` out; if downstream analysis breaks on the missing totals, that's a follow-up.

---

## Implementation Plan

### Phase 1: Core extractor rewrite
**Goal:** `StatisticalExtractor` sweeps all quantitative columns; registry trimmed.
**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Task 1.1 — Rewrite `StatisticalExtractor.extract`: add `EXCLUDED_QUANTITATIVE_COLUMNS`, `_discover_quantitative_columns(df)`, replace hardcoded variable dict with the discovered list. Drop `get_kinematics` import and the `fps` config read.
- [x] Task 1.2 — Trim `feature_characterization/__init__.py`: remove `TemporalExtractor`, `MechanicsOfSolidsExtractor`, `PressureExtractor` imports; shrink `EXTRACTOR_REGISTRY` to `{'statistical': StatisticalExtractor, 'touch_category': TouchCategoryExtractor}`; update `__all__`.
- [x] Task 1.3 — Delete `representation/feature_characterization/temporal.py`, `mos.py`, `pressure.py`.
- [x] Task 1.4 — Grep the repo for any lingering imports of the deleted classes (`TemporalExtractor`, `MechanicsOfSolidsExtractor`, `PressureExtractor`). Expected hits: `feature_extraction/__init__.py` shim — update it too.

**Files Modified:**
- `code/src/analysis/touch_analytics/representation/feature_characterization/statistical.py` — new auto-discover loop + exclude list.
- `code/src/analysis/touch_analytics/representation/feature_characterization/__init__.py` — registry trim.
- `code/src/analysis/touch_analytics/feature_extraction/__init__.py` — re-export list trim.
- (Deleted) `representation/feature_characterization/temporal.py`, `mos.py`, `pressure.py`.

**Dependencies:** None.

### Phase 2: Pipeline + workflow wiring
**Goal:** `touch_category` flows as an independent top-level option; legacy feature names are rejected.

- [x] Task 2.1 — `extraction_pipeline.run_feature_extraction`: add `touch_category: Optional[dict] = None` parameter. When truthy and `touch_category.get('enabled')`, run `TouchCategoryExtractor` once per session and write `<output_dir>/touch_category/<session>_touch_summary.csv`.
- [x] Task 2.2 — `extraction_pipeline`: on entry, validate `features` keys against `AGGREGATION_NAMES ∪ {'statistical'}`; if a removed legacy key is present (`temporal`, `mechanics_of_solids`, `pressure_velocity_mean`, `pressure_velocity_max`), raise with a message pointing to the Stage 2a + aggregation replacement.
- [x] Task 2.3 — Remove `_translate_extraction_profiles` and the old-format detection branch in `run_feature_extraction`.
- [x] Task 2.4 — `code/scripts/analysis_workflow.py`: add `touch_category: dict = None` to `touch_feature_extraction_flow` signature and forward it to `run_feature_extraction`.
- [x] Task 2.5 — `run_batch_analysis` kwargs dispatch: add `"touch_category"` branch next to the existing `"features"` branch so the DAG option reaches the flow.

**Files Modified:**
- `code/src/analysis/touch_analytics/extraction_pipeline.py` — `touch_category` branch, legacy-name rejection, translator removed.
- `code/scripts/analysis_workflow.py` — signature + dispatch.

**Dependencies:** Phase 1.

### Phase 3: DAG config + downstream column rename
**Goal:** DAG YAML reflects the new shape; `matrix_generation` uses raw column names.

- [x] Task 3.1 — Rewrite `touch_feature_extraction.options` in `configs/analyse_workflow_dag.yaml`: `features` contains only aggregation entries; `touch_category` is promoted to a sibling block.
- [x] Task 3.2 — Rewrite `touch_clustering.feature_combinations`: delete `pressure_velocity_mean`, `pressure_velocity_max`, `pressure_and_category`, `all_features`; rename `pressure_velocity_mean_and_type` → `mean_and_type` with `features: [mean, touch_category]`.
- [x] Task 3.3 — Mirror the same rewrite in `map_receptive_fields_clustered.feature_combinations`.
- [x] Task 3.4 — Update `code/src/analysis/touch_analytics/touch_config.py`: change `KINEMATIC_SIGNALS` to `('contact_depth', 'contact_area', 'velocity_magnitude', 'acceleration_magnitude', 'geo_pressure')`.
- [x] Task 3.5 — Update `code/src/analysis/touch_analytics/matrix_generation.py`: hierarchy_order list at line 149 and `heat_x`/`heat_y1`/`heat_y2` assignments at lines 218–220 use the new column names.

**Files Modified:**
- `configs/analyse_workflow_dag.yaml` — `features`, `feature_combinations` (×2 tasks).
- `code/src/analysis/touch_analytics/touch_config.py` — `KINEMATIC_SIGNALS`.
- `code/src/analysis/touch_analytics/matrix_generation.py` — heatmap axis columns.

**Dependencies:** Phase 2.

### Phase 4: GUI adjustment
**Goal:** `touch_category` renders as its own checkbox section above the aggregation grid.

- [x] Task 4.1 — In `task_detail_panel.py`, identify the dispatch for `touch_feature_extraction` (around the `_is_feature_dict` / `_make_feature_section` wiring, ~lines 209–226). Split `options` so `touch_category` is handled first as a single-checkbox group, then pass the remaining `features` dict through the existing aggregation grid builder.
- [x] Task 4.2 — In `_make_feature_section`, suppress the per-feature "…" parameter button when the feature dict contains only `enabled` (aggregation entries will have no sub-params). (Already implemented via `has_params` guard; also fixed `_is_feature_dict` to accept `{}` sub-dicts so the aggregation grid renders correctly.)
- [ ] Task 4.3 — Smoke-test via the GUI: open `analyse_workflow_dag.yaml`, confirm layout. (Manual verification required.)

**Files Modified:**
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — split `touch_category` out of the feature grid; hide `...` button when no sub-params.

**Dependencies:** Phase 3.

---

## Testing Plan

### Unit Tests

- [ ] `StatisticalExtractor` — given a DataFrame with known quantitative columns and a known exclude-list column, `extract(group, {'aggregations': ['mean']})` returns one key per non-excluded column and none for excluded columns.
- [ ] `StatisticalExtractor` — skewness on a constant series returns NaN without raising.
- [ ] `extraction_pipeline.run_feature_extraction` — legacy key `temporal` in the `features` dict raises an error mentioning "removed" and the Stage 2a replacement.
- [ ] `extraction_pipeline.run_feature_extraction` — `touch_category={'enabled': True}` produces a CSV at `<output>/touch_category/…` with `is_tap`, `is_stroke`, `dir_proximal`, `dir_distal`.

### Integration Tests

- [ ] End-to-end on one session (e.g. `valid_configs_ST13-01`) with Stage 2a enabled (kinematics + pressure + mos). Stage 2b `mean`-only output contains every expected `{col}_mean` column and none of the excluded ones.
- [ ] `touch_clustering` flow on the rewritten `mean_and_type` combination with `type_stratified` clusterer produces a non-empty labelled CSV.
- [ ] `map_receptive_fields_clustered` renders heatmaps for `only_mean` without `KeyError` on the old column names.

### Manual Verification

- [ ] Run `python code/scripts/analysis_workflow.py --dag configs/analyse_workflow_dag.yaml` on one session. Verify Stage 2a → 2b → 4 completes without error.
- [ ] Open the DAG launcher GUI on `analyse_workflow_dag.yaml`. Confirm `touch_feature_extraction` displays a `touch_category` checkbox *above* the aggregation grid, and the grid only lists the seven aggregation names.
- [ ] Disable `touch_category`, re-run: confirm no `touch_features/touch_category/` folder is produced.

### Edge Cases

- [ ] Session CSV missing a Stage 2a column (e.g. `mos_*` absent when MoS was disabled in 2a): extractor does not raise; output simply lacks the corresponding `mos_*_mean` columns.
- [ ] Session CSV with a single-frame "touch" group: aggregations that require >1 sample (std with `ddof=1`, skewness) should produce NaN, not raise.
- [ ] `features: {}` (empty) with `touch_category.enabled: true`: only the `touch_category/` output is written; no aggregation folders are created. A warning is logged.
- [ ] `features: {mean: {enabled: true}}` with `touch_category.enabled: false`: only the `mean/` folder is written.

---

## Documentation Plan

- [ ] Update `docs/design/timeseries_clustering_pipeline_guidelines.md` only if it currently references any of the removed extractors (scan first — likely does not).
- [ ] Update `CLAUDE.md` — Memory section (project state) will be refreshed after implementation to reflect new Stage 2b contract.
- [ ] No user-facing guide changes required; the DAG YAML is the contract.
- [ ] Inline comments in `statistical.py` explain the exclude set rationale (IDs, event channels, raw sensor positions).

---

## Rollback Plan

**Before merge:** the branch is self-contained. Rollback = close the PR without merging.

**After merge:** revert the merge commit. No data migration: Stage 2b outputs are regenerated from Stage 2a augmented CSVs on demand, and Stage 2a output is untouched by this change.

1. `git revert -m 1 <merge-commit>` on the dev branch.
2. Re-enable the old DAG YAML entries if needed by restoring the pre-merge `analyse_workflow_dag.yaml`.
3. Because the column-name rename is output-only, regenerating old outputs is a matter of rerunning the workflow; there is no on-disk state that ties us to the new column names.

**Data considerations:** existing `4_analysed/touch_features/*/` CSVs produced before the rename will have short column names; after the rename they'll have the raw names. If both are on disk at the same time, clustering may pick up stale CSVs. Mitigation: `force_processing: true` for one run after merge, or delete `4_analysed/touch_features/` before re-running.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| A downstream consumer not identified here (e.g. a notebook, a one-off script) references the old short column names (`depth_mean`, etc.) and silently `KeyError`s. | Medium | Medium | Grep for `depth_mean|area_mean|velocity_mean|acceleration_mean` across the repo before merge; list uses found in non-code (docs, plans) and assess. |
| `sum`-style aggregates previously produced by MoSExtractor (`elastic_energy_total_mj`, `impulse_total_mns`) are required by an analysis not in this codebase. | Low | Low | Documented in "Output column naming migration" table; follow-up plan can add `sum` to `AGGREGATION_NAMES` in ~15 lines. |
| Auto-discover includes a new Stage 2a column that should have been excluded (e.g. a future `frame_time` addition). | Low | Low | Exclude set is a single module-level constant; easy to amend. Integration test asserts the *negative* (Nerve_spike_mean etc. are absent). |
| GUI refactor in Phase 4 regresses unrelated panels. | Low | Medium | Change is scoped to the `touch_feature_extraction` branch in `task_detail_panel.py`; smoke-test other tasks (series_transforms, clustering) in the launcher after the change. |
| The `_total` aggregates disappearing silently breaks a clustering combination that depended on them. | Low | Low | No current combination references them; `all_features` is the only combination that touched MoS and is deleted in Phase 3. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — Core extractor rewrite | 0.5 day | None |
| Phase 2 — Pipeline + workflow wiring | 0.5 day | Phase 1 |
| Phase 3 — DAG config + column rename | 0.5 day | Phase 2 |
| Phase 4 — GUI adjustment | 0.5 day | Phase 3 |

---

## References

- Design reference: `docs/design/timeseries_clustering_pipeline_guidelines.md` (Stage 2b definition, lines 71–86).
- Upstream plan: `docs/development/plans/active/split-feature-extraction-into-2a-2b-flows.md` — establishes the augmented-CSV contract this plan depends on.
- Upstream plan: `docs/development/plans/active/align-clustering-pipeline-with-guidelines.md` — parent alignment effort on the same branch theme.
- Related: `docs/development/plans/pending/fix-analysis-workflow-stage-config-forwarding.md` — sibling fix for `reduction` / `evaluation` forwarding (independent of this plan).
- Knowledge-base relevance check: no applicable notes.
