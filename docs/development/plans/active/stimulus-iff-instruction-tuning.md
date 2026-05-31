# Plan: IFF Instruction Tuning — Categorical Bar Charts by Designed Stimulus Levels

**Date:** 2026-05-31
**Author:** Basil
**Status:** In Progress
**Started:** 2026-05-31
**Base Branch:** `feature/stimulus-tuning-spike-count-metric`
**Branch:** `feature/stimulus-iff-instruction-tuning`

---

## Overview

Add a new analysis task `stimulus_iff_instruction_tuning` that groups single
touches by their **designed metadata instruction levels** (contact area, speed,
force) and renders mean IFF bar charts per instruction category. This is a
sibling task to `stimulus_iff_tuning_curves`, which bins touches along
continuous calculated metrics — the new task instead uses the discrete
categorical labels from the experiment design.

## Problem Statement

The existing IFF tuning curves show how IFF varies with *measured* stimulus
features (depth, pressure, velocity, contact area). But the experiment is
designed around discrete *instructed* levels — "1 finger" vs "whole hand", or
"slow" vs "fast". There is currently no direct way to compare IFF response
across the designed instruction levels, which is the natural framing for
answering "does the neuron respond differently to the instructed conditions?"

## Goals

### In Scope
1. Per-session bar charts showing mean IFF per instruction level, with ±1 STD
   error bars and `n=XX` count annotations.
2. Cross-session overlay plots with jittered dots per session, color-coded,
   enabling visual comparison of instruction-level IFF across sessions.
3. CSV export of per-category data (category label, count, IFF mean, IFF std)
   alongside each PNG, plus pooled cross-session CSVs.
4. Full gesture-subset iteration: `all`, `tap`, `stroke`, `stroke_proximal`,
   `stroke_distal`.
5. Support for both `iff_metric: mean` and `iff_metric: max` (and `both`).

### Out of Scope
- Modifying feature extraction to carry metadata columns through — metadata is
  re-sourced from the prepared CSV (same approach as `stimulus_iff_tuning_curves`).
- Statistical tests between instruction levels (p-values, ANOVA) — future work.
- Box plots or violin plots — bar + error bar + count annotation was chosen.
- Changes to the existing `stimulus_iff_tuning_curves` task.

## Success Criteria

- [ ] Per-session PNGs show one IFF bar per instruction level with ±1 STD error
      bars (no whisker where count < 2) and `n=XX` count annotations above bars.
- [ ] Overlay PNGs show jittered dots per session with per-session colors and a
      legend; shared IFF Y-axis limits across all sessions.
- [ ] CSV files written next to every PNG with columns: `category_label`, `count`,
      `iff_mean`, `iff_std`, `session_id`, `category_column`, `gesture_subset`.
- [ ] Task runs for all three metadata columns: `contact_area_metadata`,
      `speed_metadata`, `force_metadata`.
- [ ] Task runs for both IFF metrics (`mean` and `max`) when `iff_metric: both`.
- [ ] Existing `pytest code/tests/ -k tuning` passes unchanged.
- [ ] Spot-check: CSV `iff_mean` values match bar heights in PNG.

---

## Technical Design

### Approach

Create a new pipeline + renderer pair that follows the established two-pass
pattern from `stimulus_iff_tuning_curves`. Instead of continuous binning
(`_compute_bin_windows` + `_bin_data`), the new task uses a simple groupby on
the metadata column's categorical values. The renderer produces single-axis
bar charts (not dual-axis) with count annotations rather than count bars.

Data flow:
1. Load feature CSV (for IFF column + touch ID columns)
2. Load metadata CSV (for instruction-level columns)
3. Merge on `TOUCH_ID_COLS`
4. Group by metadata column → compute mean/std IFF per category
5. Render bar chart (per-session) and dot-plot overlay (cross-session)

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| New pipeline + renderer files | Clean separation; follows existing pattern; no risk of breaking current task | Two new files | **Chosen** |
| Add to existing IFF tuning pipeline | Less file proliferation | Conflates two fundamentally different X-axis types (continuous vs categorical); `_bin_data` / `_group_by_category` share nothing meaningful | Rejected |
| Box/violin plots | Richer statistical view | Inconsistent with existing style; adds complexity | Rejected |
| Dual-axis bars (IFF + count) | Mirrors existing tuning curve layout | Cluttered with few categories; count annotations are cleaner | Rejected |

### Architecture Changes

Two new files, four modified files:

**New files:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_iff_instruction_tuning_renderer.py`
  — `CategoryResult` NamedTuple, `_group_by_category()`, `render_session_instruction_tuning()`,
  `render_overlay_instruction_tuning()`
- `code/src/analysis/receptive_field_mapping/pipelines/rf_iff_instruction_tuning_pipeline.py`
  — `run_iff_instruction_tuning()`, `_write_category_csv()`,
  `_compute_global_category_count_max()`

**Modified files:**
- `code/src/analysis/pipeline/output_dirs.py` — add constant
- `code/src/analysis/receptive_field_mapping/__init__.py` — add export
- `code/scripts/analysis_workflow_processing.py` — add flow + stage descriptor
- `configs/analyse_workflow_processing_dag.yaml` — add task entry

Output layout:
```
4_analysed/stimulus_iff_instruction_tuning/iff_{metric}/
    iff_instruction_tuning_sentinel.json
    {category_col}/
        {gesture_subset}/
            {session_id}_instruction_tuning.png
            {session_id}_instruction_tuning.csv
            overlay_instruction_tuning.png
            overlay_instruction_tuning.csv
```

### Reused Functions

These functions are imported directly — no duplication:

| Function | Source | Purpose |
|----------|--------|---------|
| `_filter_gesture()` | `rf_iff_tuning_pipeline` | Gesture-subset filtering |
| `_compute_global_iff_ylim()` | `rf_iff_tuning_pipeline` | Shared Y-axis limits |
| `_write_sentinel()` | `rf_iff_tuning_pipeline` | Idempotency sentinel |
| `_load_session_metadata_df()` | `rf_iff_tuning_pipeline` | Metadata CSV loader |
| `_find_feature_csv()` | `rf_touch_feature_radar_pipeline` | Feature CSV discovery |
| `assign_session_colors()` | `rf_stimulus_session_comparison_renderer` | Session color assignment |
| `_style_dark_ax()`, `_BG`, `_AX_BG`, etc. | `rf_iff_tuning_renderer` | Dark theme styling |
| `_cat_display()` | `rf_iff_tuning_renderer` | Category display names |

---

## Implementation Plan

### Phase 1: Renderer — `CategoryResult` + rendering functions
**Goal:** Create the new renderer with categorical grouping and bar chart rendering.

- [x] 1.1 — Define `CategoryResult` NamedTuple with fields: `category_labels`
      (list[str]), `mean_iff` (ndarray), `std_iff` (ndarray), `counts` (ndarray).
- [x] 1.2 — Implement `_group_by_category(df, category_col, iff_col, global_levels)`
      that groups df by metadata column, computes mean/std IFF and count per level,
      returning a `CategoryResult` with entries for every global level (NaN/0 for
      empty categories).
- [x] 1.3 — Implement `render_session_instruction_tuning()`: single-axis bar chart
      with IFF bars, ±1 STD error bars (no whisker if count < 2), `n=XX` count
      annotations above each bar, dark theme, rotated X-tick labels for >4
      categories.
- [x] 1.4 — Implement `render_overlay_instruction_tuning()`: cross-session jittered
      dot plot with per-session colors and error bars, legend showing session IDs.

**Files Created:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_iff_instruction_tuning_renderer.py`

**Dependencies:** None

### Phase 2: Pipeline — two-pass orchestrator + CSV export
**Goal:** Create the pipeline function with data loading, merging, and render dispatch.

- [x] 2.1 — Implement `run_iff_instruction_tuning()` two-pass pipeline:
      Pass 1 loads feature + metadata CSVs, merges on `TOUCH_ID_COLS`, resolves
      global category levels (numeric sort first, lexical fallback).
      Pass 2 iterates category_col × gesture_subset × session, calling the
      renderer for per-session and overlay plots.
- [x] 2.2 — Implement `_write_category_csv()` helper writing per-category data
      (category_label, count, iff_mean, iff_std, session_id, category_column,
      gesture_subset) to CSV.
- [x] 2.3 — Implement `_compute_global_category_count_max()` to find the maximum
      per-category count across all sessions and gesture subsets (for annotation
      positioning / Y-axis sizing).

**Files Created:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_iff_instruction_tuning_pipeline.py`

**Dependencies:** Phase 1

### Phase 3: Registration — output dir + exports + flow + DAG
**Goal:** Wire the new task into the pipeline infrastructure.

- [x] 3.1 — Add `STIMULUS_IFF_INSTRUCTION_TUNING = "stimulus_iff_instruction_tuning"`
      to `output_dirs.py`.
- [x] 3.2 — Add import and `__all__` entry for `run_iff_instruction_tuning` in
      `receptive_field_mapping/__init__.py`.
- [x] 3.3 — Add `stimulus_iff_instruction_tuning_flow()` Prefect flow function in
      `analysis_workflow_processing.py` with params: `tuning_categories`,
      `iff_metric`, `clip_percentile`, `metadata_dir`, `metadata_filename`.
- [x] 3.4 — Add stage descriptor in `_build_pipeline_stages()` after the existing
      `stimulus_iff_tuning_curves` entry.
- [x] 3.5 — Add task entry to `analyse_workflow_processing_dag.yaml` under
      `stimulus_sensitivity` category.

**Files Modified:**
- `code/src/analysis/pipeline/output_dirs.py`
- `code/src/analysis/receptive_field_mapping/__init__.py`
- `code/scripts/analysis_workflow_processing.py`
- `configs/analyse_workflow_processing_dag.yaml`

**Dependencies:** Phase 2

DAG YAML entry:
```yaml
stimulus_iff_instruction_tuning:
  category: stimulus_sensitivity
  enabled: true
  options:
    force_processing: false
    iff_metric: both
    clip_percentile: 1.0
    tuning_categories:
    - contact_area_metadata
    - speed_metadata
    - force_metadata
    metadata_dir: touch_prepare_sessions
    metadata_filename: '{session_id}_prepared.csv'
  depends_on: [stimulus_extract_features]
```

---

## Testing Plan

### Unit Tests
- [ ] `_group_by_category` returns correct mean/std/count for a known fixture
      DataFrame with 3 categories.
- [ ] `_group_by_category` returns NaN mean/std and zero count for categories
      absent in the session but present in `global_levels`.
- [ ] `_group_by_category` returns NaN std when a category has exactly 1 touch.
- [ ] `CategoryResult` fields have correct shape matching `len(global_levels)`.

### Integration Tests
- [ ] Missing metadata CSV raises `ValueError` with actionable message.
- [ ] Missing metadata column in CSV raises `ValueError`.
- [ ] Merge row-count drop raises `ValueError`.
- [ ] Empty `tuning_categories` raises `ValueError`.

### Manual Verification
- [ ] Launch GUI → Analysis → enable `stimulus_iff_instruction_tuning` → run.
- [ ] Per-session PNGs show bars with error bars and `n=XX` annotations under
      `4_analysed/stimulus_iff_instruction_tuning/iff_mean/{category_col}/{gesture}/`.
- [ ] Overlay PNGs show jittered dots with per-session colors.
- [ ] CSVs exist next to every PNG with correct columns and values.
- [ ] Spot-check: CSV `iff_mean` matches bar height; CSV `iff_std` matches
      error bar length.

### Edge Cases
- [ ] Empty category (no touches in a level for one session) → NaN bar, no
      annotation, no error.
- [ ] Single-touch category → bar shown, no whisker, `n=1` annotation.
- [ ] Session with fewer than 5 rows for a gesture subset → skipped with warning.
- [ ] Overlay with <2 qualifying sessions → skipped with warning.

---

## Documentation Plan

- [x] Update `code/src/analysis/CLAUDE.md` to add the new task to the
      `stimulus_sensitivity` category listing.

---

## Rollback Plan

1. The change is additive and isolated — the new task shares no mutable state
   with existing tasks.
2. To revert: delete the two new files, restore the four modified files from
   `dev`, and delete any `4_analysed/stimulus_iff_instruction_tuning/` output.
3. Data considerations: no migrations; outputs are regenerated artifacts under
   `4_analysed/`. The task is idempotent via sentinel JSON.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Metadata column values vary across sessions (dtype, naming) | Med | Med | Resolve global level list across all sessions with numeric-first sorting; consistent ordering |
| Some sessions may lack certain instruction levels entirely | Med | Low | `_group_by_category` uses `global_levels` and fills missing categories with NaN/0 |
| Speed metadata has ~7 levels — X-axis labels may overlap | Med | Low | Rotate tick labels for >4 categories; compact font size |
| Overlay with many sessions (12) may be visually dense | Low | Low | Jitter spread and alpha tuning; session colors from existing palette |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — Renderer | ~0.5 day | None |
| Phase 2 — Pipeline | ~0.5 day | Phase 1 |
| Phase 3 — Registration | ~0.25 day | Phase 2 |

---

## References

- Sibling task plan: `docs/development/plans/active/stimulus-iff-tuning-enrichment.md`
- Existing pipeline: `code/src/analysis/receptive_field_mapping/pipelines/rf_iff_tuning_pipeline.py`
- Existing renderer: `code/src/analysis/receptive_field_mapping/rendering/rf_iff_tuning_renderer.py`
- Metadata column definitions: `code/src/analysis/touch_analytics/preparation/interpolation.py:52-58`

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/scripts/analysis_workflow_processing.py
- code/src/analysis/CLAUDE.md
- code/src/analysis/pipeline/output_dirs.py
- code/src/analysis/receptive_field_mapping/__init__.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_iff_instruction_tuning_pipeline.py
- code/src/analysis/receptive_field_mapping/rendering/rf_iff_instruction_tuning_renderer.py
- configs/analyse_workflow_processing_dag.yaml
- docs/development/plans/active/stimulus-iff-instruction-tuning.md
