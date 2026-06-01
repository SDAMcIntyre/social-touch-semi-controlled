# Plan: Add spike-count metric to stimulus tuning curves

**Date:** 2026-05-31
**Author:** Basil
**Status:** Completed
**Completed:** 2026-06-01 14:04
**Started:** 2026-05-31
**Base Branch:** `feature/stimulus-iff-tuning-enrichment`
**Branch:** `feature/stimulus-tuning-spike-count-metric`

---

## Overview

Generalize the `stimulus_iff_tuning_curves` analysis task from an IFF-only Y metric
(mean/max) to a generic **response metric** selector that also plots **per-touch spike
count**. Spike count gets two bin-aggregation flavours that parallel IFF mean/max:
`spike_count_mean` (mean spikes-per-touch within a bin, with STD bars) and
`spike_count_total` (sum of spikes across the bin's touches). A new dedicated
`SpikeCountExtractor` produces the per-touch `Nerve_spike_count` column upstream.

## Problem Statement

The tuning task can only show instantaneous firing frequency (`Nerve_freq_{mean|max}`)
on the Y axis. The researcher needs to inspect how raw neural output (number of spikes
per touch) tunes to touch features (contact area, pressure, velocity, depth), not just
firing-rate intensity. Spike count is currently unavailable anywhere downstream: the
per-frame `Nerve_spike` marker is explicitly excluded from feature aggregation
(`statistical.py::_EXCLUDE_FROM_AGGREGATION`), and only the binary `spike_elicited`
survives. So the task cannot plot it today.

## Goals

### In Scope
1. Produce a per-touch spike-count column (`Nerve_spike_count`) via a dedicated extractor
   registered in the feature-extraction registry, written to
   `4_analysed/stimulus_extract_features/spike_count/`.
2. Generalize the tuning pipeline from `iff_metric` to a `response_metric` registry
   supporting `iff_mean`, `iff_max`, `spike_count_mean`, `spike_count_total`, and `all`.
3. Add a per-bin aggregation mode (`mean` vs `sum`) to the renderer's binning so
   `spike_count_total` sums the response within each bin.
4. Wire the new option through the flow, DAG config, and GUI dropdown — scoped to this
   task only.

### Out of Scope
- Renaming `iff_metric` anywhere other than `stimulus_iff_tuning_curves` (it is used by
  ~9 unrelated RF tasks for NPZ aggregation selection — left untouched).
- Changing the shared `IFF_METRICS` constant in `shared_constants.py`.
- Adding spike count to any other analysis task (RF maps, boundaries, cross-domain).
- A generic `sum` aggregation across all numeric columns (rejected — see Alternatives).

## Success Criteria

- [ ] `stimulus_extract_features` produces `spike_count/<session>_touch_summary.csv` with
      `TOUCH_ID_COLS` + `Nerve_spike_count`.
- [ ] `SpikeCountExtractor` raises `ValueError` when `Nerve_spike` is absent (fail-fast).
- [ ] Running the tuning task with `response_metric: all` produces four output trees
      (`iff_mean`, `iff_max`, `spike_count_mean`, `spike_count_total`) each with
      per-session + overlay PNGs and CSVs.
- [ ] `spike_count_mean` shows ±1 STD error bars; `spike_count_total` shows summed bars
      with no error bars and a Y axis scaled to per-bin sums.
- [ ] The DAG launcher shows a `response_metric` dropdown with the new choices; existing
      `iff_metric` tasks are unaffected.
- [ ] `pytest code/tests/` passes (new extractor test + existing tests).

---

## Technical Design

### Approach

Two coordinated changes plus wiring:

1. **Upstream extractor** — a single-purpose `SpikeCountExtractor` summing the per-frame
   `Nerve_spike` (which may be ≥1 per frame; the existing `spike_elicited` code's
   `.clip(0,1)` confirms counts can exceed 1) into one `Nerve_spike_count` value per
   touch. This follows the canonical extractor-registry pattern
   (`note-feature-extractor-registry-pattern.md`), modelled on `MeanDuringIffExtractor`.

2. **Response-metric registry** in the tuning pipeline mapping each token to
   `(source_col, agg_folder, bin_agg, ylabel, subdir)`:

   ```
   iff_mean          -> Nerve_freq_mean,    "mean",        "mean", "Mean IFF (Hz)",       "iff_mean"
   iff_max           -> Nerve_freq_max,     "max",         "mean", "Max IFF (Hz)",        "iff_max"
   spike_count_mean  -> Nerve_spike_count,  "spike_count", "mean", "Mean spikes / touch", "spike_count_mean"
   spike_count_total -> Nerve_spike_count,  "spike_count", "sum",  "Total spikes",        "spike_count_total"
   "all" -> the four above
   ```

   The existing `iff_metric == "max"` path already loads a separate aggregation CSV and
   merges the response column on `TOUCH_ID_COLS` (the mean CSV always supplies the
   X-axis features). We generalize that single special case to: *if the response column
   is not already in the mean CSV, load its folder CSV and inner-merge it* — which then
   serves `iff_max`, `spike_count_mean`, and `spike_count_total` identically. The
   resolver validates the token against the registry and raises on unknown values
   (fail-fast, per `note-fail-fast-no-fallback.md`).

3. The renderer's `_bin_data`, ylim, and label code are already column-generic. The only
   new behaviour is a `bin_agg` mode: `"sum"` reports the per-bin sum (STD left NaN → no
   error bars, which the renderer already handles).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Dedicated `SpikeCountExtractor` (registry) | Single-purpose, one clean column, no effect on other folders, matches `mean_during_iff` pattern | One new file + registry entry | **Chosen** |
| Generic `sum` aggregation + un-exclude `Nerve_spike` | Maximally general | Emits `_sum` for every numeric column; changes all aggregation folders; pollutes outputs | Rejected |
| Keep `iff_metric`, just add `spike_count*` values | Lowest churn | Key name becomes inaccurate ("IFF" metric naming a spike count) | Rejected |
| Rename `iff_metric` → `response_metric` (this task only) | Accurate naming; other tasks untouched | Touches flow signature, DAG config, GUI dropdown | **Chosen** |
| Per-bin Y = mean only / total only | Simpler | Researcher asked for both flavours | Rejected (provide both) |

### Architecture Changes

- **New extractor** in the `feature_characterization` registry (one new file + one
  registry entry). No change to `StatisticalExtractor` or the exclusion set.
- **New local `RESPONSE_METRICS` registry** in `rf_iff_tuning_pipeline.py`; the existing
  `max`-only merge special case becomes a general "load-and-merge from agg folder" step.
- **`_bin_data` gains a `bin_agg` parameter**; ylim helper gains a sum-aware branch.
- **Option rename `iff_metric` → `response_metric`** confined to the tuning flow, its
  options read site, the DAG config entry, and the GUI dropdown map.

```
stimulus_extract_features/
  mean/ max/ ... + spike_count/   <- NEW folder (Nerve_spike_count)
stimulus_iff_tuning_curves/
  iff_mean/ iff_max/               <- existing
  spike_count_mean/ spike_count_total/   <- NEW trees
```

---

## Implementation Plan

### Phase 1: Upstream spike-count column
**Goal:** Per-touch `Nerve_spike_count` produced by `stimulus_extract_features`.

- [x] 1.1 — Add `SpikeCountExtractor(FeatureExtractor)` summing `Nerve_spike` per group,
      returning `{'Nerve_spike_count': int(group['Nerve_spike'].sum())}`; raise
      `ValueError` if `Nerve_spike` absent.
- [x] 1.2 — Register `'spike_count': SpikeCountExtractor` in `EXTRACTOR_REGISTRY` and
      `__all__`.
- [x] 1.3 — Add `spike_count: { enabled: true }` under
      `stimulus_extract_features.options.features` in the DAG config (ruamel.yaml RT).

**Files Modified:**
- `code/src/analysis/touch_analytics/representation/feature_characterization/spike_count.py` — new extractor
- `code/src/analysis/touch_analytics/representation/feature_characterization/__init__.py` — register
- `configs/analyse_workflow_processing_dag.yaml` — enable `spike_count` feature

**Dependencies:** None

### Phase 2: Tuning pipeline generalization
**Goal:** Response-metric registry + generic load/merge + per-bin aggregation.

- [x] 2.1 — Add `RESPONSE_METRICS` registry to `rf_iff_tuning_pipeline.py`; resolve the
      `response_metric` option to `(response_col, agg_folder, bin_agg, ylabel)`; validate
      and raise on unknown token.
- [x] 2.2 — Generalize Pass-1 load: read mean CSV for features; if `response_col` not
      present, load `_find_feature_csv(database_path, agg_folder, session_id)` and
      inner-merge on `TOUCH_ID_COLS` with the existing row-count guard.
- [x] 2.3 — Add `bin_agg` to `_bin_data` (`"sum"` → per-bin sum, STD NaN); thread it from
      the pipeline.
- [x] 2.4 — Generalize `_compute_global_iff_ylim` to `bin_agg`: keep raw-percentile for
      `"mean"`; for `"sum"` compute the upper bound from per-bin sums across sessions ×
      gesture subsets.
- [x] 2.5 — Rename internal `iff_*` locals to `response_*`; update `_write_bin_csv` to
      write `response_value`/`response_std` + a `metric` column (verify no downstream
      reader depends on `iff_mean`/`iff_std`).

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_iff_tuning_pipeline.py`
- `code/src/analysis/receptive_field_mapping/rendering/rf_iff_tuning_renderer.py` — `_bin_data` bin_agg; optional param renames

**Dependencies:** Phase 1

### Phase 3: Flow / config / GUI wiring
**Goal:** Expose `response_metric` end-to-end (scoped to this task).

- [x] 3.1 — Rename `stimulus_iff_tuning_curves_flow` param `iff_metric` →
      `response_metric` (default `iff_mean`); expand via registry tokens incl. `all`;
      build per-metric `output_dir` from the spec `subdir`.
- [x] 3.2 — Update the task-options read site to read `response_metric`.
- [x] 3.3 — Add a `response_metric` dropdown entry in `task_detail_panel.py`; leave
      `iff_metric` entry unchanged.
- [x] 3.4 — Set `response_metric: all` in the tuning task's DAG options.

**Files Modified:**
- `code/scripts/analysis_workflow_processing.py` — tuning flow + its options read site only
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — new dropdown
- `configs/analyse_workflow_processing_dag.yaml` — `response_metric` option

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] `SpikeCountExtractor` sums `Nerve_spike` correctly and returns only
      `Nerve_spike_count` (mirror `test_iff_windowed_mean.py`).
- [ ] `SpikeCountExtractor` raises `ValueError` when `Nerve_spike` is missing.
- [ ] `_bin_data(bin_agg="sum")` returns per-bin sums with STD = NaN; `bin_agg="mean"`
      unchanged.
- [ ] Response-metric registry resolution: token → expected col/folder/bin_agg/label/subdir;
      unknown token raises.

### Integration Tests
- [ ] Full `pytest code/tests/` passes with no regression to existing `iff_metric` tasks.

### Manual Verification
- [ ] Re-run `stimulus_extract_features` (`force_processing: true`); confirm
      `spike_count/<session>_touch_summary.csv` has `Nerve_spike_count`.
- [ ] Run `stimulus_iff_tuning_curves` with `response_metric: all`; confirm four output
      trees with PNGs + CSVs; spot-check STD bars on `spike_count_mean` and summed,
      error-bar-free `spike_count_total` with a sum-scaled Y axis.
- [ ] DAG launcher shows the `response_metric` dropdown; `iff_metric` tasks unaffected.

### Edge Cases
- [ ] Touch with zero spikes → `Nerve_spike_count == 0` (bin contributes 0, not NaN).
- [ ] Bin with a single touch → `spike_count_mean` value present, STD NaN (no bar).
- [ ] Missing `spike_count` folder (extraction not re-run) → tuning task fails fast with
      a clear message (no silent skip).

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` (stage map / RF mapping notes) to mention the
      `spike_count` extractor and the `response_metric` selector.
- [ ] Note the new output trees in any tuning-task docstrings/headers.
- [ ] On completion, move this plan to `completed/` and reference it from the tuning
      pipeline module docstring.

---

## Rollback Plan

1. Revert the feature branch commits (pure additive change; no migrations).
2. Data: the new `spike_count/` folder and `spike_count_*` output trees can be deleted;
   no existing artifacts are overwritten (IFF trees keep their names).
3. The `iff_metric` → `response_metric` rename is confined to one task; reverting the
   flow/config/GUI edits restores the prior `iff_metric: both` behaviour.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Re-run of `stimulus_extract_features` forgotten → missing column | Med | Low | Fail-fast error names the missing `Nerve_spike_count`; documented in verification |
| `iff_metric` rename leaks into unrelated tasks | Low | High | Scope strictly to the tuning flow; grep confirms ~9 other usages stay on `iff_metric`/`IFF_METRICS` |
| `_write_bin_csv` column rename breaks a downstream reader | Low | Med | Only reader is the overlay concat (name-agnostic); verify before renaming |
| `Nerve_spike` semantics (count vs binary) misread | Low | Med | Sum of raw per-frame values; existing `.clip(0,1)` confirms counts can exceed 1 |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 | Small | None |
| Phase 2 | Medium | Phase 1 |
| Phase 3 | Small | Phase 2 |

---

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/scripts/analysis_workflow_processing.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_iff_tuning_pipeline.py
- code/src/analysis/receptive_field_mapping/rendering/rf_iff_tuning_renderer.py
- code/src/analysis/touch_analytics/representation/feature_characterization/__init__.py
- code/src/analysis/touch_analytics/representation/feature_characterization/spike_count.py
- code/src/utils/gui/dag_launcher/task_detail_panel.py
- configs/analyse_workflow_processing_dag.yaml
- docs/development/plans/active/stimulus-tuning-spike-count-metric.md

---

## References

- Knowledge base: `note-feature-extractor-registry-pattern.md`, `note-fail-fast-no-fallback.md`
- Related plans: `docs/development/plans/completed/iff-metric-mean-max-selection.md`,
  `docs/development/plans/active/stimulus-iff-tuning-enrichment.md`
- Internal scratch plan: `C:\Users\basil\.claude\plans\analyse-the-stimulus-iff-nested-wand.md`

---
