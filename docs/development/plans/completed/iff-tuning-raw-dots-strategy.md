# Plan: IFF Tuning Curves — Raw Dots Binning Strategy

**Date:** 2026-06-01
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-06-01 14:04
**Base Branch:** `feature/iff-tuning-neuron-type-coloring`
**Branch:** `feature/iff-tuning-raw-dots`

---

## Overview

Add a `raw_dots` alternative to the existing `sliding_window` binning strategy in the `stimulus_iff_tuning_curves` pipeline. Instead of aggregating touches into equal-width bins and showing mean ± STD error bars, every individual touch is plotted as a semi-transparent dot at its exact `(feature_value, response_value)` coordinates, and a polynomial regression fit line is drawn through them. This is the modern standard in neurophysiology for showing stimulus–response relationships, as it reveals per-touch scatter, outliers, and density in a way that binned means cannot.

## Problem Statement

The current pipeline only supports binned summary curves. Bins hide the raw data distribution: two sessions with the same bin means but wildly different scatter are visually indistinguishable. Reviewers and collaborators increasingly expect to see individual observations alongside trend lines for statistical transparency. There is currently no mechanism to switch to raw-data mode without rewriting the renderer.

## Goals

### In Scope
1. Add `binning_strategy: "raw_dots"` option to `stimulus_iff_tuning_curves` task config
2. Implement `render_session_raw_dots()` — per-session scatter + polynomial fit line + R² annotation
3. Implement `render_overlay_raw_dots()` — multi-session scatter + fit lines, consistent with existing overlay style
4. Expose `fit_degree`, `dot_alpha`, and `show_fit_ci` config knobs
5. CSV export of raw per-touch data (one row per touch, analogous to the existing per-bin CSV)

### Out of Scope
- Modifying the existing `sliding_window` path — it must remain 100% unchanged
- Applying this to `stimulus_iff_instruction_tuning` (that pipeline uses discrete categorical levels, not continuous feature axes; a fit line does not apply)
- Interactive / live rendering (all outputs are static exported PNGs)
- Adding new response metrics or features

## Success Criteria

- [ ] `binning_strategy: "raw_dots"` in the DAG config produces per-session PNGs with visible scatter dots and a fit line
- [ ] `binning_strategy: "sliding_window"` (or absent) produces identical output to the current pipeline
- [ ] Per-session title includes `N=xxx, R²=x.xx`
- [ ] Overlay PNGs show faint per-session dots + bold per-session fit lines
- [ ] Per-session CSV contains one row per touch with columns `session_id`, `feature_value`, `response_value`, `gesture_subset`, `fit_degree`, `metric`
- [ ] Output is written to `raw_dots_d{fit_degree}/` subdirectory (separate from `b{n}_ov{r}/`)
- [ ] `show_fit_ci=True` with `fit_degree > 1` issues a warning and disables the CI band gracefully

---

## Technical Design

### Approach

Add two parallel renderer functions and a thin pipeline branch. The existing code paths are wrapped in an `if binning_strategy == "sliding_window":` block; the `raw_dots` path computes nothing beyond a global ylim from raw response values, then passes `(feature_vals, response_vals)` arrays directly to the new renderer functions. No existing functions are modified.

Visual parameters are grounded in published neurophysiology best practices: small markers (`s=20`, ~5 px diameter), semi-transparent (`alpha=0.35` per-session, `alpha=0.15` in overlay), no edge colours, bold fit line (`linewidth=2.5`). The right-axis count bars are removed (no bins exist); total N is shown in the title instead.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Add `raw_dots` as new `bin_agg` mode inside existing `_bin_data` | Minimal new code | `_bin_data` returns a `BinResult` built around bin centers; shoehorning raw points in would corrupt its contract and require callers to distinguish | Rejected |
| Replace binning entirely | Simpler codebase | Breaks existing output; users rely on binned curves | Rejected |
| Separate script/task | Clean isolation | Duplicates all session-loading, feature-validation, neuron-color logic | Rejected |
| Branch inside existing pipeline (chosen) | Zero change to existing code path; full reuse of loading + color-scheme infrastructure | Slightly longer `run_iff_tuning_curves` function | **Chosen** |

### Architecture Changes

**New functions in `rf_iff_tuning_renderer.py`** (appended after `render_overlay_tuning_curve`):
- `render_session_raw_dots(feature_vals, response_vals, ..., fit_degree, dot_alpha, show_fit_ci)` — single-axis dark-theme scatter + fit line
- `render_overlay_raw_dots(session_data: dict[str, tuple[ndarray, ndarray]], ..., fit_degree, dot_alpha, show_fit_ci)` — multi-session scatter + fit lines

**New helpers in `rf_iff_tuning_pipeline.py`** (inserted after existing helpers):
- `_compute_raw_response_ylim(session_dfs, response_col, clip_percentile)` — pools raw response values to compute `(0.0, upper_percentile)`; parallel to the `bin_agg="mean"` path in `_compute_global_response_ylim` but without requiring `bin_windows`
- `_write_raw_dots_csv(feature_vals, response_vals, session_id, feature, gesture_subset, fit_degree, metric, out_path)` — writes per-touch CSV

**Modified in `rf_iff_tuning_pipeline.py`**:
- `run_iff_tuning_curves()` — option parsing, `overlap_dir` selection, global-computation branch, per-session render branch, overlay branch

**Config**:
- `configs/analyse_workflow_processing_dag.yaml` — four new keys under `stimulus_iff_tuning_curves.options`

---

## Implementation Plan

### Phase 1: Renderer functions
**Goal:** Implement and validate the two new renderer functions in isolation.

- [x] Task 1.1 — Implement `render_session_raw_dots` in `rf_iff_tuning_renderer.py`
  - Drop NaN pairs, compute fit via `np.polyfit`, compute R², scatter + fit line (+ optional CI band)
  - Title: `"{session_id} | {gesture_subset} | N={n}, R²={r2:.2f}"`
  - CI band: degree-1 only; warn + disable if `fit_degree > 1` and `show_fit_ci=True`
  - R² formula: `r2 = 1 - ss_res/ss_tot`; return `0.0` when `ss_tot == 0`
- [x] Task 1.2 — Implement `render_overlay_raw_dots` in `rf_iff_tuning_renderer.py`
  - Per session: faint scatter (`alpha=dot_alpha`, `s=20`) + bold fit line (`linewidth=2.5`)
  - Legend block mirrors `render_overlay_tuning_curve` (by_type / by_session)
- [x] Task 1.3 — Add `render_session_raw_dots, render_overlay_raw_dots` to the import block in `rf_iff_tuning_pipeline.py`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_iff_tuning_renderer.py` — append two functions
- `code/src/analysis/receptive_field_mapping/pipelines/rf_iff_tuning_pipeline.py` — import additions only

**Dependencies:** None

### Phase 2: Pipeline helpers
**Goal:** Add the two new pipeline helper functions.

- [x] Task 2.1 — Implement `_compute_raw_response_ylim` after `_compute_global_count_max` (~line 408)
- [x] Task 2.2 — Implement `_write_raw_dots_csv` after `_write_bin_csv` (~line 514)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_iff_tuning_pipeline.py`

**Dependencies:** Phase 1

### Phase 3: Pipeline branch
**Goal:** Wire the `raw_dots` branch into `run_iff_tuning_curves`.

- [x] Task 3.1 — Parse new options at the top of `run_iff_tuning_curves`: `binning_strategy`, `fit_degree`, `dot_alpha`, `show_fit_ci`; raise `ValueError` on unknown `binning_strategy`
- [x] Task 3.2 — Select `overlap_dir`: `f"raw_dots_d{fit_degree}"` vs `f"b{n_bins}_ov{overlap_ratio:.2f}"`
- [x] Task 3.3 — Wrap global bin-window + count-max computation in `if binning_strategy == "sliding_window": ... else: bin_windows = {}; count_maxima = {}; response_ylim = _compute_raw_response_ylim(...)`
- [x] Task 3.4 — Branch per-session render loop: `sliding_window` path unchanged; `raw_dots` path calls `render_session_raw_dots`, `_write_raw_dots_csv`, appends `(feat_arr, resp_arr)` to `overlay_session_data`
- [x] Task 3.5 — Branch overlay render: add `else` block calling `render_overlay_raw_dots` twice (by_type + by_session) with `dot_alpha=0.15` hardcoded at call site

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_iff_tuning_pipeline.py`

**Dependencies:** Phase 2

### Phase 4: Config
**Goal:** Expose the new options in the DAG config.

- [x] Task 4.1 — Add `binning_strategy: "sliding_window"`, `fit_degree: 1`, `dot_alpha: 0.35`, `show_fit_ci: false` under `stimulus_iff_tuning_curves.options` in `configs/analyse_workflow_processing_dag.yaml`

**Files Modified:**
- `configs/analyse_workflow_processing_dag.yaml`

**Dependencies:** Phase 3

---

## Testing Plan

### Manual Verification
- [ ] Set `binning_strategy: "raw_dots"` and `force_processing: true`; run `stimulus_iff_tuning_curves` on one session; confirm per-session PNG shows scatter dots and a fit line with `N=xxx, R²=x.xx` in the title
- [ ] Confirm no count bars appear in the per-session plot
- [ ] Inspect `overlay_tuning_by_type.png` and `overlay_tuning_by_session.png` — each session has faint dots + bold fit line in its neuron-type/session color
- [ ] Confirm output is written to `raw_dots_d1/` subdirectory
- [ ] Confirm per-session CSV has columns `session_id`, `feature_value`, `response_value`, `gesture_subset`, `fit_degree`, `metric`
- [ ] Set `binning_strategy: "sliding_window"` (or remove the key); confirm output is identical to pre-change behavior
- [ ] Set `fit_degree: 2`; confirm a curved fit line and output in `raw_dots_d2/`
- [ ] Set `show_fit_ci: true, fit_degree: 1`; confirm CI shading appears around the fit line
- [ ] Set `show_fit_ci: true, fit_degree: 2`; confirm a warning is printed and no CI shading is drawn

### Edge Cases
- [ ] Session with all-zero response values → `ss_tot == 0` → `R²=0.00`, flat fit line, no crash
- [ ] Session with fewer than `fit_degree + 1` valid rows → fit skipped, `R²=n/a`, dots only, no crash
- [ ] Feature column with all-NaN values → session skipped by existing `_MIN_ROWS` guard, no crash
- [ ] Unknown `binning_strategy` value in config → `ValueError` raised immediately

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — note that `stimulus_iff_tuning_curves` now supports `binning_strategy: "raw_dots"` and document the new config keys

---

## Rollback Plan

1. All changes are additive — `sliding_window` path is untouched. To revert `raw_dots` mode: set `binning_strategy: "sliding_window"` in the config (or remove the key).
2. To revert the code entirely: `git revert` the feature branch commits. Existing `b{n}_ov{r}/` output directories are unaffected.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `np.polyfit` numerically unstable for poorly-conditioned data (very large x range) | Low | Low | Degenerate fits just produce a poorly-shaped curve — not a crash. R² will reveal poor fits. |
| CI band formula incorrect for high-leverage points | Low | Low | CI is optional and degree-1 only; standard error formula is closed-form and well-validated |
| `overlay_session_data` type inconsistency (tuples of 3 vs 2 elements) | Low | Medium | The two strategy branches never mix within one run; pipeline branch structure guarantees consistency |
| Config keys ignored silently on old pipeline runs | Low | Low | Default `binning_strategy: "sliding_window"` preserves existing behavior; no action needed |

---

## Post-Implementation Fixes

Issues found during manual verification, after the four planned phases:

1. **Prefect flow layer dropped the new options (critical).** The four new
   keys reached the YAML and `run_iff_tuning_curves`'s own `options.get(...)`
   parsing, but `code/scripts/analysis_workflow_processing.py` did not forward
   them. That file has a hand-maintained option whitelist in two places —
   the `stimulus_iff_tuning_curves_flow` signature + its inline `options={...}`
   dict, and the dispatch `params` lambda. Both only pass keys they explicitly
   list, so `binning_strategy` (etc.) were silently dropped and the pipeline
   always defaulted to `sliding_window`. Fixed by adding `binning_strategy`,
   `fit_degree`, `dot_alpha`, `show_fit_ci` to all three spots. **The original
   plan never listed this file** — Phase 3 assumed `run_iff_tuning_curves` read
   options straight from YAML, but the flow wrapper sits in between.
2. **GUI dropdown.** Added `binning_strategy` to `_OPTION_ENUMS` in
   `task_detail_panel.py` so the strategy renders as a Sliding Window / Raw Dots
   combobox instead of a free-text field.

**Lesson for the Testing Plan:** the "switch strategies" checks below must be
run through the actual GUI/flow dispatch (a full `stimulus_iff_tuning_curves`
run), **not** by calling `run_iff_tuning_curves` directly — only an end-to-end
run exercises the flow-layer option forwarding where this bug lived.

---

## Modified Files

<!-- auto-generated by /plan-implement; manually appended post-implementation fixes (flow wiring + GUI dropdown) -->
- code/scripts/analysis_workflow_processing.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_iff_tuning_pipeline.py
- code/src/analysis/receptive_field_mapping/rendering/rf_iff_tuning_renderer.py
- code/src/utils/gui/dag_launcher/task_detail_panel.py
- configs/analyse_workflow_processing_dag.yaml
- docs/development/plans/active/iff-tuning-raw-dots-strategy.md
