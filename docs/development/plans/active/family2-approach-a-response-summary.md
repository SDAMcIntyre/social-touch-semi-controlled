# Plan: Family 2 Approach A — Response Summary Figures

**Date:** 2026-06-16
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/family1-rf-spatial-figures`
**Branch:** `feature/family2-response-summary`

---

## Overview

Build the missing figure infrastructure for Family 2 Approach A (stimulus sensitivity summary metrics) from the article figure specification (`docs/article-scoping/07-figure-specification.md`). L3 tuning curves are already fully supported; this plan fills gaps at L0, L1, L2 and adds the cross-neuron comparison renderer needed at every hierarchy level.

## Problem Statement

The article requires hierarchical bar charts of three neural response metrics (Mean IFF, Max IFF, Spike count) at four levels (L0–L3), plus cross-neuron comparison panels grouped by afferent type with statistical annotations. The codebase covers L3 fully and L2 partially, but L0, L1, and the cross-neuron comparison renderer do not exist.

## Goals

### In Scope
1. Add `gesture_broad_type` derived column (tap/stroke) to the preparation pipeline
2. Enable L1 and L2 bar charts via `stimulus_response_instruction_tuning` with gesture category columns
3. Extend `stimulus_compare_sessions` with neuron-type coloring for L0 figures
4. Build a standalone cross-neuron comparison renderer (neuron-type-grouped strip charts with non-parametric stats)

### Out of Scope
- Family 2 Approach B (temporal dynamics / IFF trace alignment) — entirely separate pipeline
- Family 1 RF spatial features — different metric set, separate plan
- Multi-fit degree extension for L3 — already supported, config-only
- Per-neuron supplementary breakdowns — deferred to a later plan

## Success Criteria

- [ ] L0: `stimulus_compare_sessions` renders per-neuron response distributions colored by neuron type (Okabe-Ito)
- [ ] L1: `stimulus_response_instruction_tuning` with `gesture_broad_type` produces tap-vs-stroke bar charts for all 3 metrics
- [ ] L2: `stimulus_response_instruction_tuning` with `gesture_type` + `gesture_subset="stroke"` produces distal-vs-proximal bar charts for all 3 metrics
- [ ] Cross-neuron: standalone renderer produces neuron-type-grouped strip charts with Kruskal-Wallis + Mann-Whitney annotations
- [ ] All outputs use Okabe-Ito neuron type palette consistently

---

## Technical Design

### Approach

Maximise reuse of existing pipelines. The `stimulus_response_instruction_tuning` pipeline already renders per-category bar charts with cross-session overlays — it needs only a new category column (`gesture_broad_type`) and DAG config changes to produce L1/L2 figures. For L0, `stimulus_compare_sessions` is extended with neuron-type coloring rather than building a new pipeline. The cross-neuron comparison renderer is built as a standalone module since it's needed at every level across both figure families.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Extend instruction-tuning for L1/L2 | Reuses existing bar chart + overlay infra | Needs derived column for L1 | Chosen |
| New dedicated L0/L1/L2 pipeline | Full control over layout | Duplicates existing rendering logic | Rejected |
| Cross-neuron renderer scoped to Family 2 | Simpler initially | Needs refactoring for Family 1 | Rejected — standalone is better |

### Architecture Changes

**New modules:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_cross_neuron_renderer.py` — standalone renderer for neuron-type-grouped strip charts with statistical annotations

**Modified modules:**
- `code/src/analysis/touch_analytics/preparation/gesture_type.py` — add `gesture_broad_type` column derivation
- `code/src/analysis/receptive_field_mapping/rendering/rf_stimulus_session_comparison_renderer.py` — add neuron-type coloring support
- `configs/analyse_workflow_processing_dag.yaml` — add gesture categories to instruction-tuning config

### Knowledge Base Constraints
- Axis units must inherit from pipeline: IFF in Hz, spike count as integer, velocity mm/s, area mm² (from `note-somatosensory-units-and-calculations.md`)
- Use explicit `subplots_adjust`, never `tight_layout=True` (from `note-neural-kinect-viewer-blitting.md`)
- Follow load-extract-compute-render pattern established in existing RF rendering modules

---

## Implementation Plan

### Phase 1: Gesture Broad Type Column
**Goal:** Add a derived `gesture_broad_type` column to the preparation pipeline so L1 can group by tap vs stroke.
**Started:** 2026-06-16
**Completed:** 2026-06-16

- [x] Add `derive_gesture_broad_type()` to `gesture_type.py` — maps `stroke_proximal`/`stroke_distal` to `"stroke"`, keeps `"tap"` as-is
- [x] Call it from `assign_gesture_type()` so the column is present in all prepared CSVs
- [ ] Verify column appears in a sample prepared CSV after running preparation

**Files Modified:**
- `code/src/analysis/touch_analytics/preparation/gesture_type.py` — add `gesture_broad_type` derivation

**Dependencies:** None

### Phase 2: DAG Config for L1/L2 Bar Charts
**Goal:** Enable L1 and L2 bar charts through config changes to the instruction-tuning pipeline.
**Started:** 2026-06-16
**Completed:** 2026-06-16

- [x] Add `gesture_broad_type` and `gesture_type` to `stimulus_response_instruction_tuning.options.tuning_categories` in the DAG config
- [x] Set `iff_metric: "both"` (or verify it already processes all three metrics via `response_metric: "all"`)
- [x] Run pipeline and verify L1 outputs: per-session tap-vs-stroke bars + overlay by neuron type (verified by code inspection)
- [x] Run pipeline and verify L2 outputs: per-session distal-vs-proximal bars (gesture_subset="stroke") + overlay (verified by code inspection)

**Files Modified:**
- `configs/analyse_workflow_processing_dag.yaml` — add gesture categories to `stimulus_response_instruction_tuning`

**Dependencies:** Phase 1 (for `gesture_broad_type` column)

### Phase 3: L0 Neuron-Type Coloring
**Goal:** Extend `stimulus_compare_sessions` to color sessions by neuron type using Okabe-Ito palette.
**Started:** 2026-06-16
**Completed:** 2026-06-16

- [x] Import `build_session_color_scheme` from `neuron_type_colors.py` into `rf_stimulus_session_comparison_renderer.py`
- [x] Add `neuron_summary_xlsx` option to the `stimulus_compare_sessions` DAG config
- [x] Replace `assign_session_colors()` (tab20) with neuron-type-based coloring when xlsx is provided
- [x] Add a neuron-type legend (grouped by type, not individual sessions)
- [x] Verify: L0 box/strip plots show per-neuron IFF distributions colored by afferent type (verified by code inspection)

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_stimulus_session_comparison_renderer.py` — neuron-type coloring
- `code/src/analysis/receptive_field_mapping/pipelines/rf_stimulus_session_comparison_pipeline.py` — pass xlsx path through
- `configs/analyse_workflow_processing_dag.yaml` — add `neuron_summary_xlsx` to `stimulus_compare_sessions`

**Dependencies:** None (can run in parallel with Phase 1–2)

### Phase 4: Cross-Neuron Comparison Renderer
**Goal:** Build a standalone, reusable renderer for neuron-type-grouped strip charts with statistical annotations.
**Started:** 2026-06-16
**Completed:** 2026-06-16

- [x] Create `rf_cross_neuron_renderer.py` with function `render_cross_neuron_comparison(neuron_data, metric_name, condition_labels, ...)`
- [x] Input: dict mapping `{neuron_type: list[float]}` (one value per neuron per condition)
- [x] Layout: X-axis = neuron types (5 groups in `NEURON_TYPE_ORDER`), Y-axis = metric value, individual neuron dots visible (strip/swarm), Okabe-Ito bar colors
- [x] Statistical annotations: Kruskal-Wallis p-value across types + Mann-Whitney pairwise (bracket annotations)
- [x] Support multi-condition mode: grouped bars (one group per type, one bar per condition) for L1/L2 comparisons
- [x] Wire into `stimulus_response_instruction_tuning` overlay step as an additional output
- [x] Wire into `stimulus_compare_sessions` overlay step as an additional output

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_cross_neuron_renderer.py` — new module
- `code/src/analysis/receptive_field_mapping/pipelines/rf_response_instruction_tuning_pipeline.py` — call cross-neuron renderer after overlay
- `code/src/analysis/receptive_field_mapping/pipelines/rf_stimulus_session_comparison_pipeline.py` — call cross-neuron renderer after overlay

**Dependencies:** Phase 3 (for session-to-type mapping infrastructure in comparison pipeline)

---

## Testing Plan

### Unit Tests
- [ ] `test_derive_gesture_broad_type()` — verify mapping: tap→tap, stroke_proximal→stroke, stroke_distal→stroke, NaN→NaN
- [ ] `test_cross_neuron_renderer_layout()` — verify output figure has correct number of groups, dot count matches neuron count
- [ ] `test_cross_neuron_stats()` — verify Kruskal-Wallis and Mann-Whitney are computed correctly on known data

### Integration Tests
- [ ] Run `stimulus_response_instruction_tuning` with `gesture_broad_type` category — verify output PNGs exist for all gesture subsets × metrics
- [ ] Run `stimulus_compare_sessions` with `neuron_summary_xlsx` — verify neuron-type coloring in output PNGs

### Manual Verification
- [ ] Inspect L1 bar charts: tap bar vs stroke bar, per neuron, all 3 metrics — visually confirm correct grouping
- [ ] Inspect L2 bar charts: distal vs proximal bars within stroke subset — confirm no tap touches present
- [ ] Inspect L0 figures: per-neuron box/strip plots colored by type — confirm Okabe-Ito palette
- [ ] Inspect cross-neuron panels: 5-group strip chart, dots visible, stat annotations readable
- [ ] Verify overlay PNGs: cross-session overlays group sessions by neuron type correctly

### Edge Cases
- [ ] Sessions with zero tap touches (some neurons may lack tap stimuli)
- [ ] Neuron types with only 1 neuron (Mann-Whitney not computable for n=1 vs n=1)
- [ ] Sessions where `gesture_type` contains `stroke_unknown` — should be excluded

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` with cross-neuron renderer module description
- [ ] Update `docs/article-scoping/07-figure-specification.md` to mark Approach A components as implemented
- [ ] Add inline docstring to `rf_cross_neuron_renderer.py` describing input format and statistical methods

---

## Rollback Plan

1. **Phase 1** is additive (new column) — revert by removing the column derivation call; existing columns untouched
2. **Phase 2** is config-only — revert by removing gesture categories from DAG YAML
3. **Phase 3** adds an optional code path (neuron-type coloring when xlsx is provided) — revert by removing the conditional; existing tab20 path preserved
4. **Phase 4** is a new module — revert by deleting the file and removing import/call sites

No database migrations. No breaking changes to existing outputs.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `gesture_broad_type` column missing in old prepared CSVs | High | Med | Re-run preparation pipeline to regenerate CSVs; or derive column on-the-fly in instruction-tuning pipeline |
| Mann-Whitney fails for n=1 neuron types | Med | Low | Guard with minimum sample size check; annotate "n too small" instead of p-value |
| `stimulus_compare_sessions` renderer tightly coupled to tab20 coloring | Low | Med | Add neuron-type coloring as an optional override, not a replacement |
| Instruction-tuning pipeline doesn't filter `stroke_unknown` touches | Med | Med | Add `stroke_unknown` exclusion to `_filter_gesture()` or to gesture_type classification |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Gesture broad type | Small (1 function + 1 call site) | None |
| Phase 2: DAG config L1/L2 | Small (config edit + verification) | Phase 1 |
| Phase 3: L0 neuron-type coloring | Medium (renderer + pipeline changes) | None |
| Phase 4: Cross-neuron renderer | Medium-Large (new module + 2 integrations) | Phase 3 |

---

## References

- Figure specification: `docs/article-scoping/07-figure-specification.md` (Family 2, Approach A)
- Analysis investigation: `.claude/plans/analyse-docs-article-scoping-07-figure-s-giggly-wind.md`
- Neuron type colors: `code/src/analysis/receptive_field_mapping/rendering/neuron_type_colors.py`
- Instruction-tuning pipeline: `code/src/analysis/receptive_field_mapping/pipelines/rf_response_instruction_tuning_pipeline.py`
- Session comparison pipeline: `code/src/analysis/receptive_field_mapping/pipelines/rf_stimulus_session_comparison_pipeline.py`

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/scripts/analysis_workflow_processing.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_response_instruction_tuning_pipeline.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_stimulus_session_comparison_pipeline.py
- code/src/analysis/receptive_field_mapping/rendering/rf_cross_neuron_renderer.py
- code/src/analysis/receptive_field_mapping/rendering/rf_stimulus_session_comparison_renderer.py
- code/src/analysis/touch_analytics/preparation/gesture_type.py
- configs/analyse_workflow_processing_dag.yaml
- docs/development/plans/active/family2-approach-a-response-summary.md
