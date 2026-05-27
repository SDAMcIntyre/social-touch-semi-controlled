# Plan: IFF Tuning Curves Pipeline

**Date:** 2026-05-27
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/rf-circular-population-plot`
**Branch:** `feature/stimulus-iff-tuning-curves`

---

## Overview

**What:** A new analysis pipeline task that generates binned tuning curves showing how each selected touch feature relates to mean neural firing rate (IFF), both per-session and as cross-session overlays.

**Why:** No analysis currently plots a touch feature against IFF directly. IFF is only used as a windowing criterion (`mean_during_iff`). Tuning curves are a standard neuroscience visualisation for characterising stimulus-response relationships.

**How:** Pool per-touch feature summary CSVs across sessions, bin each feature into equal-width ranges, compute mean IFF and touch count per bin, and render matplotlib PNGs with dark theme. Feature selection via DAG GUI checkboxes.

## Problem Statement

The codebase has rich infrastructure for extracting per-touch features and IFF-windowed statistics, but no pipeline that directly visualises the relationship between a stimulus property (touch feature) and the neural response (IFF). Researchers currently cannot see tuning curves showing, for example, how mean firing rate varies with contact pressure or velocity across sessions.

## Goals

### In Scope
1. Per-session dual Y-axis tuning curve plots (IFF line + touch count bars)
2. All-sessions overlay tuning curve plots (one IFF curve per session)
3. Plots faceted by gesture subset (all, tap, stroke, stroke_proximal, stroke_distal)
4. Global Y-axis limits across sessions for consistent visual comparison
5. DAG GUI checkbox-based feature selection
6. Sentinel-based idempotency

### Out of Scope
- Neuron type grouping (SA/FA/CT metadata does not exist in the data)
- Interactive GUI viewer (static PNGs only)
- Statistical significance testing or confidence intervals on tuning curves
- Logarithmic binning or adaptive bin widths (fixed equal-width bins only)

## Success Criteria

- [ ] Task `stimulus_iff_tuning_curves` appears in the DAG GUI under `stimulus_sensitivity`
- [ ] Feature checkboxes render correctly and persist selections to YAML
- [ ] Per-session PNGs show dual Y-axes (IFF left, count right) with globally consistent limits
- [ ] All-sessions overlay PNGs show distinct coloured curves per session with legend
- [ ] Plots are generated for each gesture subset (all, tap, stroke, stroke_proximal, stroke_distal)
- [ ] Existing tests pass (`pytest code/tests/`)

---

## Technical Design

### Approach

Follow the two-pass cross-session pipeline pattern established by `rf_stimulus_session_comparison_pipeline.py`: load all sessions into a pooled DataFrame, compute global scales, then render per-session and cross-session plots. Use the simple checklist GUI pattern (`_CHECKLIST_UNIVERSES` + `_make_checklist_section`) for feature selection.

Data comes from the `mean` aggregation folder under `stimulus_extract_features/`. Each per-touch row contains `Nerve_freq_mean` (Y-axis) and feature columns like `contact_area_mean`, `pressure_mean` (X-axis candidates). The `mean` aggregation is the most interpretable for tuning curves and requires loading only one folder.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Single `mean` aggregation folder | Simple loading, interpretable | Doesn't show IFF-windowed features | **Chosen** — tuning curves need raw mean per touch |
| `radar_groups`-style feature spec (data_type × aggregation) | Flexible, consistent with radar task | Over-complex for this use case — aggregation is always `mean` | Rejected |
| Scatter + regression instead of binned curves | Shows individual data points | Noisy with many touches, less readable | Rejected — user requested binned curves |

### Architecture Changes

New modules:
```
code/src/analysis/receptive_field_mapping/
    rendering/rf_iff_tuning_renderer.py     (new — pure rendering functions)
    pipelines/rf_iff_tuning_pipeline.py     (new — pipeline orchestrator)
```

Modified files:
- `code/src/analysis/pipeline/output_dirs.py` — add constant
- `code/src/analysis/receptive_field_mapping/__init__.py` — re-export
- `configs/analyse_workflow_processing_dag.yaml` — add task
- `code/scripts/analysis_workflow_processing.py` — add flow + stage
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — add checklist + generalise routing

Reused functions (no modifications needed):
- `_find_feature_csv` from `rf_touch_feature_radar_pipeline.py`
- `_DISPLAY_NAMES` from `rf_touch_feature_radar_pipeline.py`
- `DATA_TYPE_TO_COLUMNS` from `clustering_pipeline.py`
- `assign_session_colors` from `rf_stimulus_session_comparison_renderer.py`
- `session_id_from_path` from `shared_constants`

---

## Implementation Plan

### Phase 1: Infrastructure
**Goal:** Register the new task in the pipeline without any logic yet.
**Started:** 2026-05-27
**Completed:** 2026-05-27

- [x] Task 1.1 — Add `STIMULUS_IFF_TUNING_CURVES` constant to `output_dirs.py`
- [x] Task 1.2 — Add task config to `analyse_workflow_processing_dag.yaml` with `tuning_features` checklist, `n_bins`, `clip_percentile` options
- [x] Task 1.3 — Add `"tuning_features"` entry to `_CHECKLIST_UNIVERSES` in `task_detail_panel.py`
- [x] Task 1.4 — Generalise checklist routing: change `key == "extracted_features"` to `key in _CHECKLIST_UNIVERSES` (line 297)

**Files Modified:**
- `code/src/analysis/pipeline/output_dirs.py` — add one constant after line 22
- `configs/analyse_workflow_processing_dag.yaml` — add task block after `stimulus_compare_sessions`
- `code/src/utils/gui/dag_launcher/task_detail_panel.py` — add checklist universe entry, generalise routing condition

**Dependencies:** None

### Phase 2: Renderer
**Goal:** Implement pure rendering functions that take binned data and produce PNGs.
**Started:** 2026-05-27
**Completed:** 2026-05-27

- [x] Task 2.1 — Create `rf_iff_tuning_renderer.py` with `_bin_data()` helper
- [x] Task 2.2 — Implement `render_session_tuning_curve()` — dual Y-axis (IFF line left, count bars right)
- [x] Task 2.3 — Implement `render_overlay_tuning_curve()` — multi-session single Y-axis overlay
- [x] Task 2.4 — Implement `_style_dark_ax()` and `_display_id()` helpers

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_iff_tuning_renderer.py` — new file

**Dependencies:** None (pure functions, no pipeline dependency)

### Phase 3: Pipeline + Integration
**Goal:** Wire everything together — pipeline orchestrator, Prefect flow, `__init__` re-export.
**Started:** 2026-05-27
**Completed:** 2026-05-27

- [x] Task 3.1 — Create `rf_iff_tuning_pipeline.py` with `run_iff_tuning_curves()` entry point
- [x] Task 3.2 — Implement session loading (reuse `_find_feature_csv`), column resolution, pooling
- [x] Task 3.3 — Implement global bin edge computation with percentile clipping
- [x] Task 3.4 — Implement global IFF ylim and count max computation
- [x] Task 3.5 — Implement gesture filtering (`_filter_gesture`) with `'stroke'` combined subset
- [x] Task 3.6 — Implement render loop (per gesture × per feature × per session + overlay)
- [x] Task 3.7 — Add sentinel JSON write for idempotency
- [x] Task 3.8 — Add re-export in `receptive_field_mapping/__init__.py`
- [x] Task 3.9 — Add `stimulus_iff_tuning_curves_flow` and stage entry in `analysis_workflow_processing.py`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_iff_tuning_pipeline.py` — new file
- `code/src/analysis/receptive_field_mapping/__init__.py` — add import + `__all__` entry
- `code/scripts/analysis_workflow_processing.py` — add imports, flow function, stage entry

**Dependencies:** Phase 1, Phase 2

---

## Testing Plan

### Unit Tests
- [ ] `_bin_data` with known values: correct bin centers, means, counts
- [ ] `_bin_data` with NaN features and NaN IFF: excluded correctly
- [ ] `_bin_data` with empty bins: NaN mean, zero count
- [ ] `_filter_gesture` for each subset: correct filtering logic
- [ ] `_resolve_tuning_columns` with valid and invalid data types

### Integration Tests
- [ ] `run_iff_tuning_curves` with synthetic multi-session DataFrames produces expected output files
- [ ] Sentinel skip logic: second run with `force_processing=False` is a no-op

### Manual Verification
- [ ] Launch pipeline GUI → navigate to `stimulus_iff_tuning_curves` → checkboxes render, persist to YAML
- [ ] Run task on real data → PNGs generated in expected directory structure
- [ ] Per-session plots: Y-axes are consistent across sessions for the same feature
- [ ] All-sessions overlay: each session has a distinct colour, legend is readable
- [ ] Gesture subsets: `stroke` combines proximal + distal correctly

### Edge Cases
- [ ] Session with all zero `Nerve_freq_mean` (no neural response) — line at y=0
- [ ] Session with very few touches (<5) — renders without error, count bars are small
- [ ] Feature with near-constant values across sessions — global binning still produces valid edges
- [ ] Single session in pool — per-session and overlay plots are equivalent

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — add task to stimulus_sensitivity category list
- [ ] Inline docstrings in new pipeline and renderer modules

---

## Rollback Plan

All changes are additive (new files + small insertions into existing files). Rollback:

1. Delete new files: `rf_iff_tuning_pipeline.py`, `rf_iff_tuning_renderer.py`
2. Revert additions to `output_dirs.py`, `__init__.py`, `task_detail_panel.py`, `analysis_workflow_processing.py`, DAG YAML
3. The checklist routing generalisation (line 297) is backward-compatible; reverting it is safe but optional

No data migrations. No breaking changes to existing tasks.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `Nerve_freq_mean` column missing from `mean` aggregation CSVs | Low | High | `Nerve_freq` is not in exclusion list so it should be aggregated. Verify with a real CSV before implementing. Fail-fast with clear error if missing. |
| Multi-column data types (e.g. `hand_velocity` → 3 columns) produce unexpected number of plots | Medium | Low | `tuning_features` universe lists `hand_velocity_amplitude` (1 column) as preferred. Multi-column types still work but produce one plot per component. |
| Empty gesture subsets for some sessions | Medium | Low | Skip rendering for (session, gesture) pairs with zero touches. Log a warning. |
| Percentile clipping removes meaningful extreme values | Low | Low | `clip_percentile` is configurable (default 1.0). Researcher can set to 0 for no clipping. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Infrastructure | Small (config + GUI) | None |
| Phase 2: Renderer | Medium (2 plot types + helpers) | None |
| Phase 3: Pipeline + Integration | Medium (orchestrator + flow wiring) | Phase 1, Phase 2 |

---

## References

- Template pipeline: `code/src/analysis/receptive_field_mapping/pipelines/rf_stimulus_session_comparison_pipeline.py`
- Template renderer: `code/src/analysis/receptive_field_mapping/rendering/rf_stimulus_session_comparison_renderer.py`
- Feature column resolution: `code/src/analysis/receptive_field_mapping/pipelines/rf_touch_feature_radar_pipeline.py`
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md` (unit conventions for axis labels)
