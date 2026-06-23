# Plan: Per-Center-Point Comparison & Statistics in Proximal-Distal Pipeline

**Created:** 2026-06-23
**Approved:** 2026-06-23
**Completed:** 2026-06-23
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/rf-per-center-point-comparison`

## 1. Overview

**What:** Apply the proximal-distal comparison and statistics uniformly to all three RF center point types: centroid, contour_center, and peak.
**Why:** Currently, shift decomposition and statistical tests are only computed for centroid and contour_center; the peak has no shift metrics, and quiver/aggregate plots are incomplete across center types.
**How:** Add peak shift decomposition to the pipeline, generalize the quiver renderer to accept any center type, add a contour_center aggregate scatter plot, and extend the strip chart to include peak shift metrics.

## 2. Problem Statement

- The `spatial_compare_proximal_distal` pipeline computes shift decomposition (along-arm / across-arm) and statistical tests for centroid and contour_center, but not for peak.
- The quiver/arrow plot exists only for centroid — contour_center and peak have no equivalent.
- The aggregate scatter plot exists for centroid and peak (hotspot), but not for contour_center.
- This asymmetry means results cannot be compared across all three center types on equal footing.

## 3. Goals

### In Scope
1. Add peak shift decomposition (along/across arm) to the pipeline
2. Add peak shift metrics to the strip chart with Wilcoxon + sign test annotations
3. Generate a quiver/arrow plot for each of the three center types
4. Add contour_center aggregate scatter plot (completing symmetry)

### Out of Scope
- Changing the boundary delta metrics (area, perimeter, circularity, etc.) — these are center-independent
- Changing per-session rendering (heatmaps, contour overlays, triptychs, circular crops)
- Refactoring the three aggregate scatter renderers into one parameterized function (follow-up cleanup)

## 4. Success Criteria
- [ ] Summary CSV contains `peak_shift_along_arm_mm` and `peak_shift_across_arm_mm` columns
- [ ] Summary CSV contains `cc_offset_u/v_proximal/distal_mm` columns
- [ ] Strip chart shows 13 metrics (was 11) with Wilcoxon/sign test annotations for peak shift metrics
- [ ] Three quiver/arrow plot PNGs: centroid, contour_center, peak
- [ ] Contour_center aggregate scatter plot PNG generated
- [ ] Pipeline runs end-to-end with `force_processing: true`

## 5. Technical Design

### Approach

Extend the existing proximal-distal comparison pipeline to compute shift metrics for the peak center type (currently missing), generalize the quiver plot renderer to accept column name parameters, and add the missing contour_center aggregate scatter renderer.

The centroid shift pattern (pipeline lines 250-253) is replicated for peak:
```python
_peak_shift_uv = hotspot_proximal - hotspot_distal
_peak_shift_along_arm_mm = float(_peak_shift_uv[0]) * uv_to_mm
_peak_shift_across_arm_mm = float(_peak_shift_uv[1]) * uv_to_mm
```

The quiver plot renderer (`render_centroid_shift_decomposition`) is generalized by adding `along_col`, `across_col`, and `title` parameters, then called three times.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| One strip chart, add 2 columns | Side-by-side comparison of all center types | Wider figure (13 vs 11 cols) | Chosen |
| Separate strip chart per center type | Clean separation | Loses at-a-glance comparison; 3 figures | Rejected |
| Single 1x3 quiver subplot figure | Compact | Compressed arrows, unreadable labels | Rejected |
| Separate quiver files per center type | Clear, independent inspection | More output files | Chosen |

### Architecture Changes

No new modules. Changes to two existing files:
- `rf_proximal_distal_comparison_renderer.py` — generalize quiver, add contour_center aggregate, extend strip metrics
- `rf_proximal_distal_comparison_pipeline.py` — compute peak shift, contour_center offsets, wire up renders

## 6. Implementation Plan

### Phase 1: Renderer — Generalize quiver plot + add contour_center aggregate
**Goal:** Make the shift decomposition renderer parameterizable and add the contour_center aggregate scatter renderer.
**Started:** 2026-06-23
**Completed:** 2026-06-23

**Tasks:**
- [x] Task 1.1 — Rename `render_centroid_shift_decomposition` to `render_shift_decomposition` with new parameters: `along_col`, `across_col`, `title`. Replace hardcoded column names with the parameter values. Replace hardcoded title with the `title` parameter.
- [x] Task 1.2 — Add `render_proximal_distal_contour_center_aggregate` as a near-copy of `render_proximal_distal_hotspot_aggregate` (lines 308-354), changing: parameter name to `session_contour_centers`, axis labels to reference all-gesture contour center, title to `'Proximal vs Distal RF Contour Center Offsets'`.
- [x] Task 1.3 — Add `peak_shift_along_arm_mm` and `peak_shift_across_arm_mm` to the `_STRIP_METRICS` list.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_proximal_distal_comparison_renderer.py`

**Dependencies:** None

### Phase 2: Pipeline — Compute peak shift + contour_center aggregate data
**Goal:** Add the missing metric computations to the per-session data loop.
**Started:** 2026-06-23
**Completed:** 2026-06-23

**Tasks:**
- [x] Task 2.1 — Compute peak shift decomposition after the contour_center shift block (~line 326). Guard with `hotspot_available`. Decompose `hotspot_proximal - hotspot_distal` into along/across arm components via `* uv_to_mm`. NaN when hotspot unavailable.
- [x] Task 2.2 — Compute contour_center for 'all' gesture type via `compute_highest_contour_peak`. Compute offsets: `cc_offset_proximal = (cc_proximal - cc_all) * uv_to_mm`, same for distal. None when any cc is None.
- [x] Task 2.3 — Add peak shift + contour_center offset fields to `valid_data` dict and `summary_rows` dict.
- [x] Task 2.4 — Add corresponding dtype entries to the DataFrame `.astype()` call.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py`

**Dependencies:** None

### Phase 3: Pipeline — Wire up new cross-session renders
**Goal:** Call the new/generalized renderers in the Pass 3 section.
**Started:** 2026-06-23
**Completed:** 2026-06-23

**Tasks:**
- [x] Task 3.1 — Update imports: replace `render_centroid_shift_decomposition` with `render_shift_decomposition`, add `render_proximal_distal_contour_center_aggregate`.
- [x] Task 3.2 — Replace the single `render_centroid_shift_decomposition(...)` call with three `render_shift_decomposition(...)` calls, one per center type (centroid, contour_center, peak) with appropriate column names, titles, and filenames.
- [x] Task 3.3 — Add contour_center aggregate scatter call after the hotspot aggregate block, following the same guard pattern.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py`

**Dependencies:** Phase 1, Phase 2

### Phase 4: Verification
**Goal:** Confirm the pipeline runs and produces all expected outputs.
**Started:** 2026-06-23
**Completed:** 2026-06-23

**Tasks:**
- [x] Task 4.1 — Run pipeline with `force_processing: true` on the full dataset (code-reviewed; no data environment available — deferred to manual run)
- [x] Task 4.2 — Verify summary CSV has the new columns with sensible values (code-reviewed; deferred to manual run)
- [x] Task 4.3 — Verify all three quiver PNGs, the strip chart (13 columns), and the contour_center aggregate scatter PNG are generated (code-reviewed; deferred to manual run)
- [x] Task 4.4 — Update `code/src/analysis/CLAUDE.md` proximal-distal bullet to mention three center types

**Dependencies:** Phase 3

## 7. Testing Plan

### Manual Verification
- [ ] Run `spatial_compare_proximal_distal` with `force_processing: true`
- [ ] Open the summary CSV and confirm `peak_shift_along_arm_mm`, `peak_shift_across_arm_mm`, and `cc_offset_*` columns are present with numeric values
- [ ] Open the strip chart PNG and count 13 metric columns with annotation text
- [ ] Confirm three quiver PNGs exist: `rf_centroid_shift_decomposition.png`, `rf_contour_center_shift_decomposition.png`, `rf_peak_shift_decomposition.png`
- [ ] Confirm `rf_contour_center_proximal_distal_aggregate.png` exists

## Output Files (After Implementation)

| File | Status |
|------|--------|
| `rf_proximal_distal_comparison_summary.csv` | Modified (6 new columns) |
| `rf_proximal_distal_population_strips.png` | Modified (13 metrics, was 11) |
| `rf_centroid_shift_decomposition.png` | Unchanged |
| `rf_contour_center_shift_decomposition.png` | **NEW** |
| `rf_peak_shift_decomposition.png` | **NEW** |
| `rf_contour_center_proximal_distal_aggregate.png` | **NEW** |
| All other existing outputs | Unchanged |

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/src/analysis/CLAUDE.md
- code/src/analysis/receptive_field_mapping/pipelines/rf_proximal_distal_comparison_pipeline.py
- code/src/analysis/receptive_field_mapping/rendering/rf_proximal_distal_comparison_renderer.py
- docs/development/plans/active/rf-per-center-point-comparison.md
