# Plan: IFF Stimulus Tuning Curves — STD, Designed-Level Composition Bars, Overlapping Bins, CSV Export

**Date:** 2026-05-29
**Author:** Basil
**Status:** In Progress
**Base Branch:** `dev`
**Branch:** `feature/stimulus-iff-tuning-enrichment`

---

## Overview

Enrich the `stimulus_iff_tuning_curves` analysis task so its output is more informative
and more legible. The task bins single touches along a stimulus feature (depth, pressure,
velocity, contact area) and plots per-bin mean IFF (left axis) against a touch-count bar
(right axis). This plan adds per-bin IFF standard deviation, composition-coloured count
bars driven by the **designed stimulus levels**, optional overlapping bins, and CSV export
of the binned data.

## Problem Statement

The current task is information-poor and emits no machine-readable data:

- IFF is shown as a bare mean line — no indication of spread/variability.
- Count bars are a single flat colour — they hide how each bin's touches distribute across
  the experiment's designed stimulus levels (force, speed, contact area).
- Bins are strictly disjoint — there is no way to smooth the count/response profile with
  overlapping windows.
- Output is PNG-only — the binned values cannot be re-analysed or audited downstream.

This limits interpretation of stimulus→neuron tuning and forces re-derivation of binned
quantities for any further analysis.

## Goals

### In Scope
1. Add per-bin IFF standard deviation; render IFF as dots with ±1 STD error bars.
2. Split each count bar into stacked, proportionally-sized colour segments keyed to the
   **designed stimulus levels** of a mapped metadata column.
3. Add an `overlap_ratio` task option (0.0–0.5) producing sliding-window overlapping bins,
   surfaced in the GUI launcher.
4. Export per-bin data (geometry, count, IFF mean/STD, per-level composition) as CSV next
   to each PNG, plus a pooled cross-session CSV.

### Out of Scope
- Re-running / modifying feature extraction to carry the designed-level metadata columns
  (decision: read the metadata from a configurable upstream-CSV source instead).
- Statistical tercile/quantile categorisation of continuous features (decision: colour by
  designed `*_metadata` levels only; depth & pressure map to `force_metadata`).
- Changes to the overlay plot's count axis (overlay remains IFF-only).
- New GUI widgets (the new scalar option auto-renders; no panel code change).

## Success Criteria

- [ ] Per-session PNGs show IFF as dots with ±1 STD error bars (no whisker where a bin has
      < 2 touches).
- [ ] Count bars are stacked and coloured by designed level, with a legend mapping
      colour → level (handles both 3-level and 7-level dimensions).
- [ ] `overlap_ratio` appears in the GUI launcher; values in `[0, 0.5]` work, values outside
      raise a `ValueError`; `overlap_ratio = 0` reproduces today's disjoint counts/means.
- [ ] A `*.csv` is written next to every per-session PNG and a pooled `overlay_tuning.csv`
      next to every overlay PNG, with the documented schema.
- [ ] Spot-check: CSV per-level proportions match plotted segment heights; `iff_std` matches
      whisker lengths.
- [ ] `pytest code/tests/ -k tuning` passes.

---

## Technical Design

### Approach

Keep the existing two-pass pipeline / pure-renderer split. The renderer's `_bin_data`
becomes window-based (parallel `low`/`high` edge arrays instead of one shared edge array,
so windows may overlap) and returns a richer `BinResult` NamedTuple carrying STD and
per-level composition. The pipeline resolves the designed-level metadata by loading a
configurable upstream CSV (default the prepared-session CSV) and merging the `*_metadata`
columns per touch on `TOUCH_ID_COLS`. Each x-axis feature is coloured by a designed-metadata
column via a configurable mapping. CSV export is a small pipeline-side IO helper.

The designed stimulus levels already exist as four `*_metadata` columns produced in
preparation (`code/src/analysis/touch_analytics/preparation/interpolation.py:55-58`):
`type_metadata`, `speed_metadata` (~7 levels), `contact_area_metadata` (3 levels: hand vs
fingertip body-part), `force_metadata` (3 levels). They are per-frame/per-touch in the
prepared & series CSVs but are dropped at feature extraction (only `type_metadata` survives
via `pipeline_shared.py:19` `SHARED_COLUMNS` + `extraction_pipeline.py:428`).

Default feature → category mapping:

| Tuning feature (x-axis) | Designed-level column | Levels |
|-------------------------|-----------------------|--------|
| `contact_area_mean` | `contact_area_metadata` | 3 |
| `hand_velocity_amplitude_mean` | `speed_metadata` | ~7 |
| `pressure_mean` | `force_metadata` | 3 |
| `contact_depth_mean` | `force_metadata` | 3 |

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Colour by designed `*_metadata` levels | Matches the real experiment design; exact, interpretable levels | Levels not in feature CSV → must be sourced | **Chosen** |
| Colour by statistical terciles of the feature | No metadata sourcing needed | Invents levels the experiment didn't use; misleading | Rejected |
| Propagate metadata via feature extraction | Single source of truth, benefits all analyses | Requires re-running `stimulus_extract_features`; broader blast radius | Rejected (researcher chose configurable input source) |
| Read metadata from a configurable upstream CSV | No extraction re-run; localised to this task; explicit, fail-fast | Couples task to a CSV path/schema (mitigated by configurable option) | **Chosen** |
| Disjoint `np.linspace` edges + `np.digitize` (status quo) | Simple | Cannot overlap windows | Replaced by window edges |

### Architecture Changes

- Renderer gains a `BinResult` NamedTuple and a category-palette/legend helper; `_bin_data`
  reworked to window-based aggregation with STD + per-level composition; render functions
  draw error-bar dots and stacked coloured bars.
- Pipeline gains `_compute_bin_windows`, a metadata-source loader/merger, global level
  resolution, `count_category_by` validation, and a `_write_bin_csv` IO helper.
- Flow + DAG registry + DAG YAML gain the new options. The renderer keeps
  `matplotlib.use('Agg')` + `savefig` (consistent with the debugpy-matplotlib note; never
  add `plt.show`).

Output tree (unchanged folders, now with CSVs):

```
4_analysed/stimulus_iff_tuning_curves/iff_{metric}/{feature}/{gesture_subset}/
    {session_id}_tuning.png
    {session_id}_tuning.csv
    overlay_tuning.png
    overlay_tuning.csv
```

---

## Implementation Plan

### Phase 1: Window-based binning + STD + composition (renderer + pipeline core)
**Goal:** Replace disjoint binning with overlap-capable windows and produce STD + per-level
composition data.
**Started:** 2026-05-30
**Completed:** 2026-05-30

- [x] 1.1 — Add `BinResult` NamedTuple (`bin_centers, bin_low, bin_high, mean_iff, std_iff,
      counts, level_counts, level_props`) to the renderer module.
- [x] 1.2 — Rewrite `_bin_data(df, feature_col, iff_col, bin_low, bin_high, category_col=None,
      category_levels=None)`: mask `(f >= low) & (f < high)` (inclusive `high` on the last
      window only); `mean_iff` (NaN if empty); `std_iff = std(ddof=1)` (NaN if count < 2);
      per-level counts via `category == level`; NaN/unknown → `unassigned` bucket; guarded
      `level_props`.
- [x] 1.3 — Add `_compute_bin_windows(pooled_series, n_bins, clip_percentile, overlap_ratio)
      -> (low, high)`: validate `overlap_ratio ∈ [0, 0.5]` (raise otherwise); `w=(hi-lo)/n_bins`,
      `stride=w*(1-overlap_ratio)`; clamp final `high` to `hi`. Replace `_compute_bin_edges`
      call sites.
- [x] 1.4 — Update `_compute_global_count_max` to take `(bin_low, bin_high)` and read
      `BinResult.counts`; docstring notes overlapping-window counts are not mutually exclusive.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_iff_tuning_renderer.py` — `BinResult`, `_bin_data` rewrite.
- `code/src/analysis/receptive_field_mapping/pipelines/rf_iff_tuning_pipeline.py` — `_compute_bin_windows`, `_compute_global_count_max`.

**Dependencies:** None

### Phase 2: Designed-level resolution + CSV export (pipeline)
**Goal:** Source the designed-level metadata and emit CSV data.
**Started:** 2026-05-30
**Completed:** 2026-05-30

- [x] 2.1 — Resolve metadata CSV per session: `database_path/'4_analysed'/metadata_dir/
      metadata_filename.format(session_id=...)`; raise if missing.
- [x] 2.2 — Load it, `groupby(TOUCH_ID_COLS).first()` of the needed metadata columns
      (union of `count_category_by.values()`), inner-merge into the feature df on
      `TOUCH_ID_COLS`; raise on row-count drop (mirror the existing max-merge guard).
- [x] 2.3 — Validate every `tuning_feature` is present in `count_category_by`; raise listing
      unmapped features.
- [x] 2.4 — Resolve a global ordered level list per metadata column from pooled merged data
      (numeric sort if numeric else lexical); thread `category_col` + `category_levels` into
      `_bin_data` and `_compute_global_count_max`.
- [x] 2.5 — Add `_write_bin_csv(bin_result, meta, out_path)`; write per-session CSV next to
      each PNG and a pooled `overlay_tuning.csv` per feature × gesture_subset.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_iff_tuning_pipeline.py` — metadata load/merge, level resolution, validation, `_write_bin_csv`, render loop.

**Dependencies:** Phase 1

### Phase 3: Renderer visuals + option plumbing (renderer + flow + config)
**Goal:** Draw the new visuals and expose the new options end-to-end.
**Started:** 2026-05-30
**Completed:** 2026-05-30

- [x] 3.1 — `render_session_tuning_curve`: IFF dots + error bars
      (`ax_iff.errorbar(..., fmt='o', capsize=2)`, NaN std → no whisker); keep optional
      smoothing line over the dots.
- [x] 3.2 — Stacked coloured count bars: accumulate `bottom` across levels; palette sampled
      from a sequential colormap sized to `len(category_levels)`; compact dark legend keyed by
      level, titled with the category display name; title gains `| by {cat_display}`.
- [x] 3.3 — `render_overlay_tuning_curve`: accept `(centers, mean_iff, std_iff)` triples; faint
      (`alpha≈0.2`) error bars under each session line.
- [x] 3.4 — Add flow params (`overlap_ratio`, `metadata_dir`, `metadata_filename`,
      `count_category_by`) to `stimulus_iff_tuning_curves_flow` (~L976) and thread into the
      `options` dict; wire matching reads in the DAG registry (~L1492).
- [x] 3.5 — Add the options block to `configs/analyse_workflow_processing_dag.yaml`
      (ruamel round-trip / comment-preserving manual edit).

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_iff_tuning_renderer.py` — error-bar dots, stacked bars, palette/legend helper, overlay std.
- `code/scripts/analysis_workflow_processing.py` — flow params + registry wiring.
- `configs/analyse_workflow_processing_dag.yaml` — new options.

**Dependencies:** Phase 2

New YAML options:

```yaml
overlap_ratio: 0.0                       # 0.0–0.5; sliding-window bin overlap
metadata_dir: touch_prepare_sessions     # under <database>/4_analysed/
metadata_filename: "{session_id}_prepared.csv"
count_category_by:                       # tuning_feature -> designed-metadata column
  contact_area_mean: contact_area_metadata
  hand_velocity_amplitude_mean: speed_metadata
  pressure_mean: force_metadata
  contact_depth_mean: force_metadata
```

---

## Testing Plan

### Unit Tests
- [ ] `overlap_ratio == 0` window binning matches the old `np.digitize` counts/means on a fixture.
- [ ] `overlap_ratio` outside `[0, 0.5]` raises `ValueError` (pipeline and flow).
- [ ] `_bin_data` STD: NaN for empty and single-touch bins; correct sample std otherwise.
- [ ] Category split: per-level counts + `unassigned` sum to the bin total; proportions sum to
      1 on non-empty bins; covered for both a 3-level and a 7-level dimension.
- [ ] `_write_bin_csv` produces the documented schema incl. per-level `count_*`/`prop_*` columns.

### Integration Tests
- [ ] Unmapped `tuning_feature`, missing metadata CSV, missing metadata column, and merge
      row-count drop each raise an actionable error.

### Manual Verification
- [ ] `python code/scripts/launch_pipeline_gui.py` → Analysis → `stimulus_iff_tuning_curves`;
      confirm `overlap_ratio` and the metadata-source fields appear; set `overlap_ratio = 0.5`,
      run.
- [ ] Per-session PNGs show IFF dots with error bars and stacked level-coloured count bars with
      a legend; `*.csv` files exist under
      `4_analysed/stimulus_iff_tuning_curves/iff_*/{feature}/{gesture_subset}/`.
- [ ] Open one CSV and confirm per-level proportions match the PNG segment heights and `iff_std`
      matches whisker lengths.

### Edge Cases
- [ ] Empty bin → NaN mean/std, zero-height segments; single-touch bin → dot, no whisker.
- [ ] NaN/unknown designed level → counted in total, bucketed as `unassigned`, excluded from
      segments/proportions.
- [ ] `overlap_ratio == 0` faithfully reproduces disjoint behaviour (half-open windows,
      inclusive last edge).

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` if the `stimulus_iff_tuning_curves` output/option
      surface description needs refreshing.
- [ ] No README/user-guide change anticipated (task is GUI-driven and self-describing); add a
      changelog note if the project keeps one for analysis tasks.
- [ ] Inline docstrings on `_compute_bin_windows`, `_bin_data`, and `_write_bin_csv` covering
      overlap semantics, the `unassigned` bucket, and the CSV schema.

---

## Rollback Plan

1. The change is additive and isolated to the IFF tuning task; to revert, restore the four
   modified files from `dev` and delete the new `*.csv` outputs.
2. Data considerations: no migrations; outputs are regenerated artifacts under `4_analysed/`.
   The task is idempotent via `iff_tuning_sentinel.json` — deleting the sentinel forces a clean
   re-render.
3. Procedure: `git revert` the feature merge (or restore the two pipeline/renderer modules,
   the flow script, and the DAG YAML block), then re-run the task with `force_processing: true`.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Prepared CSV path/filename differs per deployment | Med | Med | Source is a configurable task option (`metadata_dir`/`metadata_filename`); missing file raises clearly |
| Designed-level values are dtype-inconsistent (e.g. "1.0" vs 1) | Med | Med | Resolve a single global ordered level list per column; sort numeric-if-numeric; legend shows raw values |
| 7-level speed dimension is visually cluttered | Med | Low | Sequential ordered palette + compact legend; STD/series still readable on left axis |
| `overlap_ratio > 0` inflates counts (non-exclusive windows) misread as more data | Low | Med | Document in docstring + plan; count axis remains relative; CSV records `overlap_ratio` |
| Behaviour drift vs current disjoint output | Low | Med | Unit test asserts `overlap_ratio == 0` parity with `np.digitize` |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 | ~0.5 day | None |
| Phase 2 | ~0.5 day | Phase 1 |
| Phase 3 | ~0.5 day | Phase 2 |

---

## References

- Scratch design: `C:\Users\basil\.claude\plans\analyse-the-analysis-process-warm-deer.md`
- Prior task plan: `docs/development/plans/completed/stimulus-iff-tuning-curves.md`
- Designed-level columns: `code/src/analysis/touch_analytics/preparation/interpolation.py:55-58`
- Tercile convention (not used here, for context): `code/src/analysis/touch_analytics/touch_config.py`

---

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/scripts/analysis_workflow_processing.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_iff_tuning_pipeline.py
- code/src/analysis/receptive_field_mapping/rendering/rf_iff_tuning_renderer.py
- configs/analyse_workflow_processing_dag.yaml
- docs/development/plans/active/stimulus-iff-tuning-enrichment.md
