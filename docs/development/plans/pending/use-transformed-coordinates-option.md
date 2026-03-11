# Plan: Use-Transformed Coordinates Option

**Date:** 2026-03-11
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/use-transformed-coordinates-option`

---

## Overview

Add a `use_transformed` boolean option (default `true`) to postprocessing and
analysis tasks so they operate on ICP-registered spatial columns
(`*_transformed`) instead of raw camera-frame columns. This ensures downstream
analytics consume spatially aligned data in multi-forearm sessions, while
keeping the raw-column path available for debugging or single-forearm runs.

## Problem Statement

The preprocessing pipeline (step 8b) already computes ICP-registered coordinates
and writes `_transformed` columns (`contact_location_x/y/z_transformed`,
`contact_points_transformed`) into the CSVs. However, **no downstream consumer
reads them**:

| Workflow | Function | Reads | Should Read |
|----------|----------|-------|-------------|
| postprocess | `set_xyz_reference_from_gestures` | `sticker_{color}_position_x/y/z` | same (no `_transformed` variant exists for stickers) |
| postprocess | `determine_receptive_field` | `contact_points` | `contact_points_transformed` |
| analysis | `_process_touch_analysis` | `sticker_blue_position_x/y/z` | same (no `_transformed` variant exists for stickers) |

**Note:** Only `contact_location_x/y/z` and `contact_points` have `_transformed`
counterparts (produced by `csv_spatial_transformer.py`). Sticker position columns
do **not** have a `_transformed` variant — they are unaffected by this feature.

In multi-forearm sessions the non-transformed `contact_points` are in different
coordinate frames per block, making cross-block receptive-field aggregation
spatially inconsistent.

## Goals

### In Scope

1. Add `use_transformed: true` option to relevant tasks in both DAG YAML configs
2. Propagate the option through the existing `options` injection mechanism to
   each flow function and down into the underlying processing function
3. In each function, swap column references when `use_transformed=True`:
   - `contact_points` → `contact_points_transformed`
   - `contact_location_x/y/z` → `contact_location_x/y/z_transformed`
4. Graceful fallback: if a `_transformed` column is missing from the CSV
   (single-forearm session, no registration), log a warning and fall back to
   the original column

### Out of Scope

- Adding `_transformed` variants for sticker position columns (they don't exist
  in the data pipeline)
- Changing the preprocessing pipeline or `csv_spatial_transformer.py`
- GUI changes (the `TaskPanel` auto-discovers boolean options as checkboxes —
  no code change needed)

## Success Criteria

- [ ] `determine_receptive_field` reads `contact_points_transformed` when
      `use_transformed=True` and the column exists
- [ ] `_process_touch_analysis` reads `contact_location_x/y/z_transformed`
      when `use_transformed=True` and the columns exist
- [ ] Both functions fall back to original columns with a logged warning when
      `_transformed` columns are absent
- [ ] DAG YAML files include `use_transformed: true` under affected tasks
- [ ] Option appears as a toggleable checkbox in the launcher GUI (automatic)
- [ ] Existing behaviour is preserved when `use_transformed: false`

---

## Technical Design

### Approach

Leverage the existing per-task `options` dict mechanism already proven with
`force_processing`. Each affected function gains a `use_transformed: bool = True`
parameter. A small helper resolves the actual column name at runtime.

### Column Resolution Helper

A single utility function avoids duplicating fallback logic:

```python
def resolve_column(df: pd.DataFrame, base: str, use_transformed: bool) -> str:
    """Return `base + '_transformed'` if the flag is set AND the column exists,
    otherwise return `base` with a warning when fallback occurs."""
    if use_transformed:
        candidate = base + "_transformed"
        if candidate in df.columns:
            return candidate
        logging.warning("Column '%s' not found — falling back to '%s'", candidate, base)
    return base
```

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Per-task `use_transformed` option (chosen) | Minimal change, follows existing pattern, per-task control | Slightly repetitive injection code | **Chosen** |
| Global DAG-level parameter | Single toggle for all tasks | Less granular; some tasks don't use spatial columns | Rejected |
| Rename columns in-place before passing to functions | Functions stay unchanged | Mutates shared DataFrame; hard to debug | Rejected |

### Architecture Changes

No new modules. Changes are localised to existing files:

```
code/
├── src/
│   └── analysis/touch_analytics/
│       └── touch_analysis.py          # parameterise column names
├── scripts/
│   ├── _5_postprocessing/
│   │   └── determine_receptive_field.py  # use config for column resolution
│   ├── postprocess_workflow_kinect_auto.py  # inject option
│   └── analysis_workflow.py               # inject option
configs/
├── analyse_workflow_dag.yaml              # add option
└── postprocess_workflow_kinect_auto_dag.yaml  # add option
```

### Knowledge Base Constraints

- **note-forearm-icp-registration.md**: Transforms are optional artefacts —
  only exist after manual extraction + registration. The graceful fallback
  handles the no-transform case.
- **note-somatosensory-units-and-calculations.md**: Units (mm) are preserved
  by the rigid ICP transform — no unit conversion needed.

---

## Implementation Plan

### Phase 1: Column Resolution Helper

**Goal:** Provide a reusable function for safe column swapping with fallback.

**Tasks:**
- [ ] 1.1 — Add `resolve_column(df, base, use_transformed)` to an appropriate
      utility location (e.g. a small helper in
      `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py`
      alongside the existing `_LOCATION_COLS` / `_POINTS_COL` constants,
      or a new lightweight module if that file's scope is too narrow)

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py`
  — add `resolve_column()` function and export it

**Dependencies:** None

### Phase 2: Postprocessing Workflow

**Goal:** Wire `use_transformed` into `determine_receptive_field`.

**Tasks:**
- [ ] 2.1 — Add `use_transformed: bool = True` parameter to
      `determine_receptive_field()` in
      `code/scripts/_5_postprocessing/determine_receptive_field.py`;
      use `resolve_column()` when reading `contact_points`
- [ ] 2.2 — Add `use_transformed: bool = True` parameter to
      `determine_receptive_field_flow()` in
      `code/scripts/postprocess_workflow_kinect_auto.py`; pass it through
- [ ] 2.3 — Inject `use_transformed` from `options` dict in the pipeline
      executor loop (same pattern as `force_processing`)
- [ ] 2.4 — Add `use_transformed: true` under `determine_receptive_field.options`
      in `configs/postprocess_workflow_kinect_auto_dag.yaml`

**Files Modified:**
- `code/scripts/_5_postprocessing/determine_receptive_field.py` — add parameter,
  call `resolve_column` for `contact_points`
- `code/scripts/postprocess_workflow_kinect_auto.py` — flow signature + injection
- `configs/postprocess_workflow_kinect_auto_dag.yaml` — add option

**Dependencies:** Phase 1

### Phase 3: Analysis Workflow

**Goal:** Wire `use_transformed` into `_process_touch_analysis` /
`generate_unified_summary`.

**Tasks:**
- [ ] 3.1 — Add `use_transformed: bool = True` parameter to
      `_process_touch_analysis()` and `generate_unified_summary()` in
      `code/src/analysis/touch_analytics/touch_analysis.py`;
      use `resolve_column()` when reading `contact_location_x/y/z`
      (note: `sticker_blue_position_*` has no `_transformed` variant — leave
      those references unchanged)
- [ ] 3.2 — Add `use_transformed: bool = True` parameter to
      `process_unified_touches_flow()` in
      `code/scripts/analysis_workflow.py`; pass it through
- [ ] 3.3 — Inject `use_transformed` from `options` dict in the analysis
      executor loop
- [ ] 3.4 — Add `use_transformed: true` under
      `process_unified_touches.options` in `configs/analyse_workflow_dag.yaml`

**Files Modified:**
- `code/src/analysis/touch_analytics/touch_analysis.py` — parameterise columns
- `code/scripts/analysis_workflow.py` — flow signature + injection
- `configs/analyse_workflow_dag.yaml` — add option

**Dependencies:** Phase 1

---

## Testing Plan

### Unit Tests
- [ ] `resolve_column()` returns `base_transformed` when flag is `True` and
      column exists
- [ ] `resolve_column()` returns `base` with warning when flag is `True` but
      column is missing
- [ ] `resolve_column()` returns `base` when flag is `False`

### Integration Tests
- [ ] Run `determine_receptive_field` on a CSV with `_transformed` columns and
      `use_transformed=True` — verify it reads from `contact_points_transformed`
- [ ] Run `determine_receptive_field` on a CSV without `_transformed` columns
      and `use_transformed=True` — verify graceful fallback
- [ ] Run `_process_touch_analysis` with `use_transformed=True` on a CSV that
      has `contact_location_x/y/z_transformed` — verify those columns are used

### Manual Verification
- [ ] Launch the GUI → open both DAG configs → confirm `use_transformed`
      checkbox appears and is checked by default
- [ ] Toggle `use_transformed` off in GUI → save → re-open → confirm value
      persists as `false`
- [ ] Run a full postprocess + analysis pipeline end-to-end with
      `use_transformed: true`

### Edge Cases
- [ ] Single-forearm session (no registration, no `_transformed` columns) —
      pipeline completes without error
- [ ] CSV where only some `_transformed` columns exist (partial registration) —
      each column resolved independently

---

## Documentation Plan

- [ ] Add knowledge-base note
      `docs/development/knowledge-base/note-use-transformed-option.md`
      documenting the option, affected columns, and fallback behaviour
- [ ] Update inline comments in modified functions

---

## Rollback Plan

1. Revert the feature branch commits
2. Remove `use_transformed` from both DAG YAML files
3. No data migration needed — the option only controls which existing columns
   are read; no data is written differently

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `_transformed` columns missing in older datasets | Medium | Low | Graceful fallback with warning log |
| Column name typo causes silent wrong-column read | Low | High | `resolve_column()` centralises the logic; unit-tested |
| `set_xyz_reference_from_gestures` users expect sticker columns to also switch | Low | Low | Document clearly that sticker columns have no `_transformed` variant |

---

## References

- Knowledge base: `docs/development/knowledge-base/note-forearm-icp-registration.md`
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
- Transform producer: `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py`
- DAG options mechanism: `code/src/utils/pipeline/pipeline_config_manager.py` → `get_task_options()`
- GUI auto-discovery: `code/src/utils/gui/dag_launcher/task_panel.py` (lines 105-110, 155-177)
