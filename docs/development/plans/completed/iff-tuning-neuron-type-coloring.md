# Plan: Color IFF tuning curves by neuron type

**Date:** 2026-06-01
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-06-01 14:04
**Base Branch:** `feature/stimulus-iff-instruction-tuning`
**Branch:** `feature/iff-tuning-neuron-type-coloring`

---

## Overview

Color the curves produced by the `stimulus_iff_tuning_curves` and
`stimulus_iff_instruction_tuning` analysis tasks by **neuron type** instead of by
arbitrary session position. Neuron type per unit is read from the experimenter
metadata file `MNG-DataSummary.xlsx` (columns "neuron type" and "unit name").
Both tasks share a single color-scheme method so identical sessions are colored
identically across tasks, and the neuron-type→color mapping is hardcoded in one
easily-editable module.

## Problem Statement

Today both IFF tasks color cross-session overlay curves via
`assign_session_colors()` (`rf_stimulus_session_comparison_renderer.py:28-34`),
which maps each session to a `tab20` color **by its index in the list**. This is
arbitrary and fragile: re-ordering the session list reshuffles every color, the
two tasks can assign different colors to the same session, and the color carries
no scientific meaning. Researchers cannot read neuron type off the plots, and
cannot compare the two tasks' overlays side by side because their colors differ.

## Goals

### In Scope
1. Add a global DAG-config parameter pointing to `MNG-DataSummary.xlsx`,
   mirroring how the preprocess/merging pipelines reference the sibling
   `03_metadata` file `semicontrolled_data-collection_quality-check.xlsx`.
2. A single shared method that maps each session → its neuron type → a color,
   used by **both** tasks so coloring is identical between them.
3. A hardcoded, trivially-editable neuron-type→color map for the 5 known types
   (`SAI`, `SAII`, `Field`, `HFA`, `CT`).
4. Apply neuron-type coloring to both the per-session plots and the
   cross-session overlay plots in both tasks.
5. Emit two overlay output files per plot: one with a neuron-type legend, one
   with per-session entries grouped/colored by neuron type.

### Out of Scope
- Changing `assign_session_colors()` itself (the `stimulus_compare_sessions`
  task keeps using it).
- Making the neuron-type→color map config-driven (hardcoding is acceptable at
  this stage; revisit only if requested).
- Any change to the binning/grouping math, CSV exports, or output directory
  structure beyond the overlay PNG filenames.

## Success Criteria

- [ ] `neuron_summary_xlsx` is a resolvable global parameter in
      `configs/analyse_workflow_processing_dag.yaml`, consumed by both tasks.
- [ ] Both tasks call one shared `build_session_color_scheme(...)`; the same
      session gets an identical color in both tasks' outputs.
- [ ] Per-session and overlay plots in both tasks are colored by neuron type.
- [ ] Each overlay produces `*_by_type.png` (neuron-type legend) and
      `*_by_session.png` (per-session entries, grouped/colored by type).
- [ ] Editing one entry in `NEURON_TYPE_COLORS` changes both tasks identically.
- [ ] Fail-fast: missing param/file, unknown unit, or unknown neuron type raises.

---

## Technical Design

### Approach

Introduce one shared module,
`code/src/analysis/receptive_field_mapping/rendering/neuron_type_colors.py`,
that owns: the hardcoded `NEURON_TYPE_COLORS` map, the xlsx parser, the
session-id→unit-name extractor, and a single `build_session_color_scheme()`
entry point returning a `SessionColorScheme` dataclass. Both IFF pipelines
replace their `assign_session_colors(session_ids)` call with this one method —
this is the single seam that guarantees identical coloring. Renderers gain a
color argument (per-session) and a `legend_mode` + neuron-type metadata
(overlay) so the pipeline can render the two legend variants.

The config parameter mirrors the existing `03_metadata` reference pattern
(absolute path, fail-fast resolution) already used by the merging pipeline's
`neural_quality_xlsx`. The xlsx parser reuses the established convention from
`parse_neural_quality_xlsx`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Shared module with `build_session_color_scheme()` called by both pipelines | One code path → guaranteed identical coloring; editable map in one place; matches reuse convention | Small new module + signature changes to renderers | **Chosen** |
| Modify `assign_session_colors()` in-place to accept a neuron-type map | Fewer new files | Couples the unrelated `stimulus_compare_sessions` task to neuron-type logic; muddies a generic util | Rejected |
| Compute colors independently inside each pipeline | Localized | Two code paths drift → tasks can disagree on a session's color (the core bug) | Rejected |
| Config-driven color map (YAML) | No code edit to recolor | Over-engineered for 5 stable values; user explicitly accepted hardcoding | Rejected (future) |

### Architecture Constraints (knowledge base)

Knowledge-base relevance check performed against
`docs/development/knowledge-base/README.md`: **no applicable notes.** The only
adjacent constraint — matplotlib must render headless via `Agg` + `savefig`
(not `plt.show`) — is already honored by the IFF renderers (they call
`matplotlib.use('Agg')`). New rendering code must keep that discipline.

### Architecture Changes

- **New module:** `rendering/neuron_type_colors.py`
  - `NEURON_TYPE_COLORS: dict[str,str]` + `NEURON_TYPE_ORDER: list[str]`
    (user-editable block at top of file).
  - `parse_neuron_summary_xlsx(xlsx_path) -> dict[str,str]` — `pd.read_excel`;
    case-insensitive/stripped lookup of the "neuron type" / "unit name" columns;
    validate they exist (raise `ValueError` listing found columns); `dropna`;
    build `{unit_name -> neuron_type}`. Mirrors
    `parse_neural_quality_xlsx` (`filter_merged_by_neural_quality.py`).
  - `unit_name_from_session_id(session_id) -> str` — `re.search(r'(ST\d+-\d+)')`;
    raise `ValueError` if no match. (Consistent with the date-prefix handling in
    `rf_stimulus_session_comparison_renderer.py::_display_id`.)
  - `@dataclass SessionColorScheme`: `session_color: dict[str,str]`,
    `session_neuron_type: dict[str,str]`, `type_color: dict[str,str]`
    (only the types actually present, in `NEURON_TYPE_ORDER`).
  - `build_session_color_scheme(session_ids, xlsx_path) -> SessionColorScheme` —
    the single shared method; fail-fast on unknown unit / unknown neuron type.
- **Modified pipelines:** `rf_iff_tuning_pipeline.py`,
  `rf_iff_instruction_tuning_pipeline.py` — resolve the xlsx path (absolute, or
  relative to database root `output_base_dir.parents[1]`, fail-fast), swap the
  color source, pass per-session color to the per-session renderer, call the
  overlay renderer twice.
- **Modified renderers:** `rf_iff_tuning_renderer.py`,
  `rf_iff_instruction_tuning_renderer.py` — add color/`legend_mode` params
  (defaults preserve current visuals).
- **Modified config + orchestrator:** new global parameter + flow-signature /
  param-lambda wiring.

---

## Implementation Plan

### Phase 1: Shared color module
**Goal:** One source of truth for neuron-type coloring.
**Started:** 2026-06-01

- [x] 1.1 — Create `rendering/neuron_type_colors.py` with `NEURON_TYPE_COLORS`,
      `NEURON_TYPE_ORDER`, column-name constants, and the editable header block.
- [x] 1.2 — Implement `parse_neuron_summary_xlsx` (reuse `parse_neural_quality_xlsx`
      conventions: read_excel → validate columns → dropna → iterrows → dict).
- [x] 1.3 — Implement `unit_name_from_session_id` and `SessionColorScheme`.
- [x] 1.4 — Implement `build_session_color_scheme` with fail-fast errors.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rendering/neuron_type_colors.py` — new.

**Dependencies:** None

### Phase 2: Config + orchestration wiring
**Goal:** Make the xlsx path reach both task flows.

- [x] 2.1 — Add `neuron_summary_xlsx` (absolute path) under `parameters:` in
      `configs/analyse_workflow_processing_dag.yaml`, with a clarifying comment.
- [x] 2.2 — Add `neuron_summary_xlsx` to both `params` lambdas in
      `_build_pipeline_stages` (`analysis_workflow_processing.py` ~1537-1561).
- [x] 2.3 — Add `neuron_summary_xlsx: Optional[str] = None` to both flow
      signatures (~981, ~1024) and forward it into each `options={...}` dict.

**Files Modified:**
- `configs/analyse_workflow_processing_dag.yaml` — new parameter.
- `code/scripts/analysis_workflow_processing.py` — flow signatures + param lambdas.

**Dependencies:** Phase 1

### Phase 3: Pipelines + renderers
**Goal:** Apply neuron-type colors to per-session and overlay plots.

- [x] 3.1 — In `rf_iff_tuning_pipeline.py::run_iff_tuning_curves` (~774): resolve
      xlsx path (fail-fast), replace `assign_session_colors` with
      `build_session_color_scheme`; pass `scheme.session_color[sid]` to the
      per-session call; call the overlay renderer twice (`by_type`, `by_session`).
- [x] 3.2 — Same changes in
      `rf_iff_instruction_tuning_pipeline.py::run_iff_instruction_tuning` (~382).
- [x] 3.3 — `rf_iff_tuning_renderer.py`: add `line_color: str = _IFF_COLOR` to
      `render_session_tuning_curve`; add `legend_mode`, `session_neuron_types`,
      `type_colors` to `render_overlay_tuning_curve`.
- [x] 3.4 — `rf_iff_instruction_tuning_renderer.py`: add `bar_color: str =
      _IFF_COLOR` to `render_session_instruction_tuning`; add the same
      `legend_mode`/metadata params to `render_overlay_instruction_tuning`.
- [x] 3.5 — Update overlay output filenames:
      `overlay_tuning.png` → `overlay_tuning_by_type.png` +
      `overlay_tuning_by_session.png`; `overlay_instruction_tuning.png` →
      `..._by_type.png` + `..._by_session.png` (CSVs unchanged).

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_iff_tuning_pipeline.py`
- `code/src/analysis/receptive_field_mapping/pipelines/rf_iff_instruction_tuning_pipeline.py`
- `code/src/analysis/receptive_field_mapping/rendering/rf_iff_tuning_renderer.py`
- `code/src/analysis/receptive_field_mapping/rendering/rf_iff_instruction_tuning_renderer.py`

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] `parse_neuron_summary_xlsx` — builds correct `{unit -> type}`; raises
      `ValueError` (listing found columns) when a required column is absent;
      tolerates case/whitespace in headers; drops blank rows.
- [ ] `unit_name_from_session_id` — `2022-06-14_ST13-01` → `ST13-01`; raises on
      a session id with no `ST##-##` token.
- [ ] `build_session_color_scheme` — assigns expected colors; raises when a
      session's unit is missing from the xlsx; raises when the xlsx names a
      neuron type absent from `NEURON_TYPE_COLORS`.

### Integration Tests
- [ ] Run both flows on a small set of sessions spanning ≥2 neuron types;
      assert the same session id maps to the same color in both tasks' outputs.

### Manual Verification
- [ ] Confirm `MNG-DataSummary.xlsx` is reachable at the configured path and its
      headers contain "neuron type" and "unit name"; spot-check unit values look
      like `ST13-01`.
- [ ] `python code/scripts/analysis_workflow_processing.py` with both tasks
      enabled (tuning curves already `force_processing: true`; set it for
      instruction tuning to regenerate).
- [ ] Inspect `4_analysed/stimulus_iff_tuning_curves/...` and
      `4_analysed/stimulus_iff_instruction_tuning/...`: per-session PNGs tinted
      by type; two overlay PNGs present; same-type sessions share a color.
- [ ] Edit one `NEURON_TYPE_COLORS` value, re-run, confirm both tasks change
      identically.

### Edge Cases
- [ ] `neuron_summary_xlsx` unset / file missing → clear `ValueError` /
      `FileNotFoundError`.
- [ ] A session whose unit is not in the xlsx → `ValueError` (no silent skip).
- [ ] Neuron type in xlsx not in the hardcoded map → `ValueError`.
- [ ] Only one neuron type present across the run → `by_type` legend has a
      single entry; rendering still succeeds.

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` (stimulus_sensitivity task notes) to
      mention neuron-type coloring + the `neuron_summary_xlsx` parameter and the
      two overlay outputs.
- [ ] Add a header comment in `neuron_type_colors.py` documenting how to edit the
      color map (serves as the inline user guide).
- [ ] No README/changelog change required (internal analysis feature).

---

## Rollback Plan

1. **Before deployment:** changes are additive and isolated to the two IFF tasks;
   no data migration. Revert by restoring the prior color source.
2. **Data considerations:** outputs are regenerated PNGs/CSVs; the only stale
   artifact is the old `overlay_*.png` filename, which can be deleted or left
   alongside the new `*_by_type.png` / `*_by_session.png`.
3. **Rollback procedure:** revert the feature-branch commits (or the merge
   commit); delete the new module; re-run the two tasks to restore the previous
   `tab20`-by-position overlays.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Actual xlsx column headers differ in case/spacing from "neuron type"/"unit name" | Med | Med | Case-insensitive/stripped column lookup; `ValueError` lists found columns for quick diagnosis |
| Neuron-type strings in xlsx differ from the 5 hardcoded keys | Med | Med | Fail-fast with the offending value named; map is a one-line edit |
| Session id format doesn't contain `ST##-##` for some session | Low | Med | `unit_name_from_session_id` raises clearly; regex matches the documented convention |
| `MNG-DataSummary.xlsx` not present in some environments | Med | Low | Parameter resolution fail-fast; task only runs when enabled |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 | ~0.5 day | None |
| Phase 2 | ~0.25 day | Phase 1 |
| Phase 3 | ~1 day | Phase 2 |

---

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/scripts/analysis_workflow_processing.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_iff_instruction_tuning_pipeline.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_iff_tuning_pipeline.py
- code/src/analysis/receptive_field_mapping/rendering/neuron_type_colors.py
- code/src/analysis/receptive_field_mapping/rendering/rf_iff_instruction_tuning_renderer.py
- code/src/analysis/receptive_field_mapping/rendering/rf_iff_tuning_renderer.py
- configs/analyse_workflow_processing_dag.yaml
- docs/development/plans/active/iff-tuning-neuron-type-coloring.md

---

## References

- Approved design scratch: `C:\Users\basil\.claude\plans\analyse-the-codebase-i-deep-rose.md`
- Reuse — xlsx parsing: `code/scripts/_4_merging/filter_merged_by_neural_quality.py::parse_neural_quality_xlsx`
- Reuse — path resolution/fail-fast: `code/scripts/merging_pipeline_neuron_to_kinect_auto.py:167-175`
- Color seam being replaced: `code/src/analysis/receptive_field_mapping/rendering/rf_stimulus_session_comparison_renderer.py:28-34`
- Related task docs: `code/src/analysis/CLAUDE.md` (stimulus_sensitivity)
