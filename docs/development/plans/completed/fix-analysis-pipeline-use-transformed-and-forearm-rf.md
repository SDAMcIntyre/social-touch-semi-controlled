# Plan: Fix Analysis Pipeline — Deprecate `use_transformed` and Connect Forearm PCA to RF Mapping

**Date:** 2026-03-13
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `fix/analysis-pipeline-use-transformed-and-forearm-rf`

---

## Overview

The new postprocessing pipeline applies ICP registration, PCA calibration, and surface projection **in-place** to base spatial columns — `_transformed` columns no longer exist. The analysis pipeline must stop requesting them via `use_transformed=True` (which now crashes), and must load the PCA-calibrated forearm PLY for RF mapping visualization instead of the broken `ForearmCatalog` call.

## Problem Statement

1. **`use_transformed` crash.** The `map_receptive_fields` DAG config sets `use_transformed: true`. This causes `resolve_column()` to look for `contact_points_transformed`, which no longer exists in aggregated CSVs produced by the new postprocessing pipeline. Result: `KeyError` at runtime.

2. **Broken forearm visualization in RF mapping.** `_load_forearm_pcd()` in `analysis_workflow.py` calls `ForearmCatalog(database_path)` — wrong constructor signature (`TypeError`) — and `catalog.get_forearm_pcd_path()` — method doesn't exist (`AttributeError`). Even if it worked, it would load the raw-space forearm, not the PCA-calibrated one that matches the coordinate frame of contact points in the aggregated CSV. The correct PLY is produced by postprocessing at `session_merged_output_dir/forearm_pca_calibrated/{session_id}_forearm_pca_calibrated.ply`.

## Goals

### In Scope
1. Remove `use_transformed` option from analysis DAG config
2. Remove `use_transformed` parameter from analysis workflow flow signatures and stop forwarding it from DAG options
3. Hardcode `use_transformed=False` at the two internal call sites that still accept the parameter (preserving downstream function signatures for a separate cleanup)
4. Rewrite `_load_forearm_pcd()` to load the PCA-calibrated forearm PLY from the postprocessing output
5. Update the RF mapping call site to pass the correct path information

### Out of Scope
- Removing `use_transformed` from downstream function signatures (`resolve_column()`, `load_grouped_spatial_data()`, `generate_unified_summary()`) — these harmlessly accept `False` and are covered by the active `postprocessing-column-consolidation` plan's separate cleanup scope
- Modifying `rf_data_loader.py`, `touch_analysis.py`, or `csv_spatial_transformer.py`
- Modifying the postprocessing pipeline itself
- Adding new forearm loading utilities to `ForearmCatalog`

## Success Criteria

- [ ] `map_receptive_fields` runs on a session processed through the new postprocessing pipeline without `KeyError` on `contact_points_transformed`
- [ ] `process_unified_touches` runs without `KeyError` on `contact_location_x_transformed`
- [ ] RF mapping visualization with `monitor: true` loads and displays the PCA-calibrated forearm PLY in the correct coordinate frame
- [ ] Sessions without a forearm PLY degrade gracefully (empty left panel, no crash)
- [ ] No `use_transformed` key remains in `analyse_workflow_dag.yaml`

---

## Technical Design

### Approach

Minimal, targeted changes to the analysis workflow script and its DAG config. The `use_transformed` flag is removed from flow signatures and the DAG config; hardcoded `False` is passed at the two call sites that still accept the parameter. The broken `_load_forearm_pcd()` is rewritten to directly load the PLY from the postprocessing output directory, using the aggregated CSV path to derive `session_merged_output_dir`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Remove `use_transformed` from flow sigs + hardcode `False` at call sites | Minimal change; downstream sigs untouched; aligns with consolidation plan scope | Leaves dead parameters in downstream functions | **Chosen** |
| Full removal of `use_transformed` from all signatures | Cleaner | Overlaps with `postprocessing-column-consolidation` plan scope; larger diff | Rejected — separate cleanup |
| Add `get_forearm_pcd_path()` to `ForearmCatalog` | Centralised forearm access | `ForearmCatalog` manages raw-space forearms, not PCA-calibrated ones; wrong abstraction | Rejected |
| Pass forearm PLY path through the flow as a parameter | Explicit data flow | Requires changing flow signatures and DAG config plumbing for a visualisation-only concern | Rejected — overkill |

### Architecture Changes

No new modules or files. Two files modified:

```
configs/analyse_workflow_dag.yaml               — remove use_transformed keys
code/scripts/analysis_workflow.py               — remove use_transformed from flows;
                                                  rewrite _load_forearm_pcd()
```

### Knowledge Base Constraints

- **note-forearm-icp-registration.md**: Single-forearm sessions have no `forearm_pca_calibrated/` PLY — `_load_forearm_pcd()` must return `None` gracefully. `RFVisualizer` already handles `forearm_pcd=None` with an empty placeholder.
- **note-somatosensory-units-and-calculations.md**: All coordinates in mm. After postprocessing, base columns are in PCA-calibrated mm space. The forearm PLY is also in PCA-calibrated space. No unit conversion needed.

---

## Implementation Plan

### Phase 1: DAG Config Cleanup
**Goal:** Remove `use_transformed` from analysis DAG config so no task receives it.

- [ ] Task 1.1 — Remove `use_transformed: false` from `process_unified_touches.options` (line 34)
- [ ] Task 1.2 — Remove `use_transformed: true` from `map_receptive_fields.options` (line 56)

**Files Modified:**
- `configs/analyse_workflow_dag.yaml` — remove 2 lines

**Dependencies:** None

### Phase 2: Remove `use_transformed` from Analysis Workflow Flows
**Goal:** Flow functions no longer accept or forward the flag; internal call sites hardcode `False`.

- [ ] Task 2.1 — Remove `use_transformed: bool = True` from `process_unified_touches_flow` signature (line 83)
- [ ] Task 2.2 — Hardcode `use_transformed=False` in the call to `generate_unified_summary()` inside `process_unified_touches_flow` (line ~114)
- [ ] Task 2.3 — Remove `use_transformed: bool = True` from `map_receptive_fields_flow` signature (line 209)
- [ ] Task 2.4 — Hardcode `use_transformed=False` in the call to `load_grouped_spatial_data()` inside `map_receptive_fields_flow` (line ~273)
- [ ] Task 2.5 — Remove `"use_transformed": use_transformed` from the RF mapping summary JSON dump (line ~323)
- [ ] Task 2.6 — In `run_batch_analysis()`, remove the two lines that extract `use_transformed` from options and inject it into kwargs (lines 486-487)

**Files Modified:**
- `code/scripts/analysis_workflow.py` — remove parameter from 2 flow signatures, hardcode `False` at 2 call sites, remove 2 lines of options forwarding, update JSON dump

**Dependencies:** Phase 1

### Phase 3: Fix Forearm PLY Loading for RF Visualization
**Goal:** RF mapping visualization loads the PCA-calibrated forearm PLY from the postprocessing output.

- [ ] Task 3.1 — Rewrite `_load_forearm_pcd()` (lines 371-380): new signature `_load_forearm_pcd(session_merged_output_dir: Path, session_id: str)`, load from `forearm_pca_calibrated/{session_id}_forearm_pca_calibrated.ply`, remove `ForearmCatalog` import, log warning if PLY not found
- [ ] Task 3.2 — Update call site in `map_receptive_fields_flow` (line ~301): derive `session_merged_dir = input_file.parent` and `session_id = input_file.name.split("_semicontrolled_")[0]`, pass both to `_load_forearm_pcd()`

**Reuse:**
- `input_file.name.split("_semicontrolled_")[0]` — same session ID extraction pattern used at lines 100 and 233 of the same file
- `RFVisualizer.visualize_rf_map(rf_result, forearm_pcd)` — already handles `forearm_pcd=None`

**Files Modified:**
- `code/scripts/analysis_workflow.py` — rewrite `_load_forearm_pcd()` function body + update call site

**Dependencies:** None (postprocessing already produces the PLY)

---

## Testing Plan

### Manual Verification
- [ ] Run `map_receptive_fields` on a session processed through the full new postprocessing pipeline — no `KeyError`
- [ ] Run `process_unified_touches` — no `KeyError`
- [ ] Run `map_receptive_fields` with `monitor: true` in DAG config — forearm PLY loads in Open3D viewer, contact points and forearm are in the same coordinate frame
- [ ] Verify the forearm PLY path `forearm_pca_calibrated/{session_id}_forearm_pca_calibrated.ply` exists before running

### Edge Cases
- [ ] Session without forearm PLY (single-forearm session or postprocessing not run) — `_load_forearm_pcd` returns `None`, visualization proceeds with empty left panel
- [ ] Old aggregated CSV (pre-postprocessing) still has `_transformed` columns — `use_transformed=False` reads base columns correctly, ignores `_transformed`
- [ ] Session ID with unusual characters — `split("_semicontrolled_")[0]` still works (same pattern used elsewhere)

---

## Documentation Plan

- [ ] No README or CLAUDE.md changes needed — this is a bugfix
- [ ] Update the `receptive-field-mapping` active plan to note that `ForearmCatalog` integration point was replaced with direct PLY loading (optional, cosmetic)

---

## Rollback Plan

1. Restore `use_transformed` keys to `analyse_workflow_dag.yaml`
2. Restore `use_transformed` parameters to flow signatures in `analysis_workflow.py`
3. Restore old `_load_forearm_pcd()` function body
4. No data or external state changes — purely code/config

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Old aggregated CSVs on disk still have `_transformed` columns | High | None | `use_transformed=False` causes `resolve_column()` to ignore them entirely |
| Aggregated CSVs not yet reprocessed through new postprocessing | Medium | Medium | Base column names exist in both old and new CSVs — old ones just have raw values. Users must re-run postprocessing for correct spatial data |
| PCA-calibrated forearm PLY missing for a session | Medium | Low | `_load_forearm_pcd()` returns `None`; `RFVisualizer` handles gracefully |
| Session ID extraction from filename fails | Low | Medium | `split("_semicontrolled_")[0]` is the established convention used in 2 other places in the same file |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: DAG Config Cleanup | Trivial (2 lines removed) | None |
| Phase 2: Remove `use_transformed` from Flows | Small (~15 lines changed) | Phase 1 |
| Phase 3: Fix Forearm PLY Loading | Small (~15 lines changed) | None |

---

## References

- Active plan (scope boundary): `docs/development/plans/active/postprocessing-column-consolidation.md`
- Active plan (RF mapping): `docs/development/plans/active/receptive-field-mapping.md`
- `resolve_column()`: `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py:99`
- Forearm PLY output: `code/scripts/_5_postprocessing/export_forearm_pca_calibrated.py`
- Analysis workflow: `code/scripts/analysis_workflow.py`
- DAG config: `configs/analyse_workflow_dag.yaml`
