# Plan: Postprocessing Spatial Transform Pipeline

**Date:** 2026-03-12
**Author:** Basil Duvernoy
**Status:** Draft
**Branch:** `feature/postprocessing-column-consolidation`

---

## Overview

A two-step postprocessing pipeline that (1) applies ICP registration transforms to original spatial columns and (2) applies PCA xyz reference calibration to ALL spatial columns. Instead of relying on `_transformed` columns generated during preprocessing, the transforms are applied fresh in postprocessing from the stored 4x4 matrices, overwriting originals in-place. This eliminates column duplication and the `resolve_column()` / `use_transformed` pattern entirely.

## Problem Statement

- Preprocessing step 8b creates dual `_transformed` columns alongside originals (`contact_location_x/y/z`, `contact_points`), forcing every downstream consumer to choose via `resolve_column()` and a `use_transformed` flag — scattered, error-prone, and adds cognitive overhead.
- The PCA-based xyz reference currently only calibrates sticker positions; contact locations remain in the ICP-registered frame, creating coordinate frame inconsistency between sticker and contact data.
- The dual-column approach makes it impossible to simply read a CSV and get the "right" spatial values — consumers must always be aware of the transformation state.

## Goals

### In Scope
1. Disable preprocessing step 8b (`transform_to_registered_frame`) — stop generating `_transformed` columns
2. New Prefect task: apply ICP registration transform to merged CSVs in postprocessing (overwriting originals)
3. Extend xyz reference task to PCA-calibrate ALL spatial columns (stickers + contact locations + contact points)
4. Subfolder-per-step output architecture (`sessions_registered/`, `sessions_pca_calibrated/`)
5. Updated postprocessing DAG config with new task and dependency chain
6. Remove dead `resolve_column` import from NeuralKinectViewer
7. Disable `use_transformed` flags in analysis/postprocessing DAG configs
8. Archive the `migrate-xyz-reference-to-preprocessing` idea file

### Out of Scope
- Full removal of `resolve_column()` function and `use_transformed` parameters from code (separate cleanup task — they harmlessly resolve to base columns when `use_transformed=False`)
- Modifying the aggregation pipeline (`merging_pipeline_neuron_to_kinect_auto.py`) — it continues reading from `sessions/`
- Updating the analysis workflow to read from postprocessed output (separate future task)
- Removing deprecated `determine_receptive_field` / `filter_by_receptive_field` tasks

## Success Criteria

- [ ] Preprocessing produces `_unified.csv` with no `_transformed` columns (step 8b disabled)
- [ ] Step 1 output CSVs in `sessions_registered/` contain ICP-transformed values in original columns; no `_transformed` columns present
- [ ] Step 1 output spatial values match the old `_transformed` values from previously processed data (backward-compatible ICP computation)
- [ ] Step 2 output CSVs in `sessions_pca_calibrated/` have PCA-calibrated values in ALL spatial columns (stickers, contact_location_x/y/z, contact_points)
- [ ] Step 2 sticker position outputs are identical to the old xyz reference output
- [ ] Single-forearm sessions (no transforms JSON) pass through step 1 unchanged
- [ ] Both steps are idempotent via `should_process_task`
- [ ] Pipeline runs end-to-end via DAG launcher on at least one session

---

## Technical Design

### Approach

Two scripts in `code/scripts/_5_postprocessing/`, each wrapped as a Prefect flow in the postprocessing workflow. Step 1 loads pre-computed ICP 4x4 matrices from `registration_transforms.json` and applies them directly to the original spatial columns in merged CSVs — replacing the dual-column approach with a single set of transformed coordinates. Step 2 extends the existing `set_xyz_reference_from_gestures` to also transform contact spatial columns, reusing `parse_contact_points` / `serialize_contact_points` from `csv_spatial_transformer.py`.

The key shift: the ICP registration transform moves from preprocessing (where it created `_transformed` alongside originals) to postprocessing (where it overwrites originals). This requires disabling preprocessing step 8b and updating the unification step's context mapping.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Apply ICP fresh in postprocessing (overwrite originals) | No dual columns, clean pipeline, transforms applied from source matrices | Requires preprocessing DAG change | **Chosen** |
| Consolidate existing `_transformed` into originals (original plan) | No preprocessing change needed | Still depends on step 8b, just moves columns around | Rejected — doesn't address root cause |
| Single combined step (ICP + PCA in one pass) | Fewer intermediate files | Loses inspectability of intermediate results | Rejected |

### Architecture Changes

New file:
```
code/scripts/_5_postprocessing/
  apply_icp_registration.py              # NEW
```

Modified files:
```
code/src/preprocessing/forearm_extraction/registration/
  csv_spatial_transformer.py             # Add find_applicable_transform_key(), transform_spatial_columns_in_place(); make parse/serialize public
  __init__.py                            # Export new functions

code/scripts/_3_preprocessing/_3_forearm_extraction/
  apply_registration_transform.py        # Import find_applicable_transform_key from shared location

code/scripts/_5_postprocessing/
  set_xyz_reference_from_gestures.py     # Extend to PCA-calibrate contact columns
  __init__.py                            # Add new import

code/scripts/
  preprocess_workflow_kinect_auto.py     # Update unify context key
  postprocess_workflow_kinect_auto.py    # Add new Prefect flow + update pipeline stages

configs/
  preprocess_workflow_kinect_auto_dag.yaml   # Disable step 8b, update unify deps
  postprocess_workflow_kinect_auto_dag.yaml  # Add new task, update dependencies
  analyse_workflow_dag.yaml                  # Disable use_transformed flags

code/src/merging/gui/
  neural_kinect_scene_viewer.py          # Remove dead resolve_column import
```

Output folder structure:
```
{session_merged_output_dir}/
├── sessions/                          # Existing: per-block merged CSVs (no _transformed columns)
│   └── {session_id}_semicontrolled_{block}_merged_data.csv
│
├── sessions_registered/               # NEW Step 1: ICP-registered spatial columns (originals overwritten)
│   └── {session_id}_semicontrolled_{block}_merged_data.csv
│
└── sessions_pca_calibrated/           # NEW Step 2: all spatial cols PCA-calibrated
    ├── {session_id}_semicontrolled_{block}_merged_data_pca-xyz.csv
    └── pca-xyz_transformation-matrices.json
```

### Knowledge Base Constraints

- **note-forearm-icp-registration.md**: Single-forearm sessions have no `registration_transforms.json` — step 1 must handle this gracefully (pass through unchanged). Transforms are pre-computed and persisted to disk; the postprocessing step only loads and applies them.
- **note-somatosensory-units-and-calculations.md**: All coordinates in mm (Azure Kinect SDK). Both ICP rigid transforms and PCA calibration preserve units (rotation + translation, no scaling). No unit conversion needed at any step.
- **note-cupy-import-order.md**: Verified safe — `csv_spatial_transformer.py` only imports numpy/pandas/re, no CuPy dependency.

---

## Implementation Plan

### Phase 1: Preprocessing Disentanglement
**Goal:** Remove step 8b from the preprocessing critical path so merged CSVs no longer contain `_transformed` columns.

- [ ] Disable `transform_to_registered_frame` in preprocessing DAG config (`enabled: false`)
- [ ] Update `unify_processed_data.depends_on`: replace `transform_to_registered_frame` with `compute_somatosensory_characteristics`
- [ ] Update unification stage context key: change `context.get("registered_somatosensory_path")` to `context.get("somatosensory_chars_path")` in the `unify_processed_data` params lambda

**Files Modified:**
- `configs/preprocess_workflow_kinect_auto_dag.yaml` — disable step 8b, update unify dependency
- `code/scripts/preprocess_workflow_kinect_auto.py` — update unify context key (line ~538)

**Dependencies:** None

### Phase 2: Shared Transform Utilities
**Goal:** Move transform key resolution to a shared location and add an in-place spatial column transform function.

- [ ] Move `_find_applicable_transform_key()` from `apply_registration_transform.py:12-75` to `csv_spatial_transformer.py` as public `find_applicable_transform_key()`
- [ ] Make `_parse_contact_points` and `_serialize_contact_points` public (remove underscore prefix)
- [ ] Add `transform_spatial_columns_in_place(df, transform_4x4)` function that applies rigid transform to `contact_location_x/y/z` and `contact_points`, overwriting originals and dropping any `*_transformed` columns
- [ ] Export new/renamed functions through `registration/__init__.py` and `forearm_extraction/__init__.py`
- [ ] Update `apply_registration_transform.py` to import `find_applicable_transform_key` from the shared location

**Reuse:**
- `apply_rigid_transform()` from `csv_spatial_transformer.py:76`
- `_parse_contact_points()` / `_serialize_contact_points()` from `csv_spatial_transformer.py:29,55` (to be made public)

**Files Created:** None

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py` — add `find_applicable_transform_key`, `transform_spatial_columns_in_place`; rename private functions public
- `code/src/preprocessing/forearm_extraction/registration/__init__.py` — export new functions
- `code/src/preprocessing/forearm_extraction/__init__.py` — export new functions
- `code/scripts/_3_preprocessing/_3_forearm_extraction/apply_registration_transform.py` — import from shared location

**Dependencies:** None

### Phase 3: Postprocessing ICP Registration Script
**Goal:** Create the postprocessing step that applies ICP registration to merged CSVs.

- [ ] Create `apply_icp_registration(input_files, session_configs, output_dir, *, force_processing)` function
- [ ] Load `registration_transforms.json` from `session_configs[0].session_processed_output_dir / "forearm_pointclouds/"` via `ForearmRegistrator.load_transforms()`
- [ ] Handle single-forearm sessions: if no transforms file, copy files to output (dropping any legacy `_transformed` columns)
- [ ] For each input file + config pair: derive video stem from `config.session_id` + `config.block_id`, resolve transform key, apply transform via `transform_spatial_columns_in_place`
- [ ] Idempotency via `should_process_task`

**Reuse:**
- `find_applicable_transform_key()` from `csv_spatial_transformer.py` (Phase 2)
- `transform_spatial_columns_in_place()` from `csv_spatial_transformer.py` (Phase 2)
- `ForearmRegistrator.load_transforms()` from `forearm_registrator.py`
- `should_process_task()` from `utils/should_process_task.py`

**Files Created:**
- `code/scripts/_5_postprocessing/apply_icp_registration.py`

**Files Modified:**
- `code/scripts/_5_postprocessing/__init__.py` — add import

**Dependencies:** Phase 2

### Phase 4: Extend PCA Calibration to Contact Columns
**Goal:** Extend `set_xyz_reference_from_gestures` to PCA-calibrate contact spatial columns alongside sticker positions.

- [ ] Add contact location and contact points column names to `CalibrationConfig`
- [ ] In the "Apply Transformation" section (line 242+), after the sticker color loop, add contact location transformation: extract `contact_location_x/y/z` as (N,3) `.values.copy()` array, apply `PCACalibrationEngine.apply_full_transform`, write back to `df_out` with NaN masking
- [ ] Add contact points transformation: per-row parse via `parse_contact_points`, apply `apply_full_transform` to point array copy, re-serialize via `serialize_contact_points`
- [ ] Import `parse_contact_points` and `serialize_contact_points` from `preprocessing.forearm_extraction.registration.csv_spatial_transformer`

**Reuse:**
- `parse_contact_points()` / `serialize_contact_points()` from `csv_spatial_transformer.py` (made public in Phase 2)
- `PCACalibrationEngine.apply_full_transform()` from `calibration_pca_engine.py:76` (already used for stickers)

**Files Modified:**
- `code/scripts/_5_postprocessing/set_xyz_reference_from_gestures.py` — extend CalibrationConfig, add contact column transformation

**Dependencies:** Phase 2 (for public parse/serialize functions)

### Phase 5: Workflow and DAG Integration
**Goal:** Wire both steps into the Prefect workflow with correct dependency chain and fix existing double-nested directory bug.

- [ ] Add `apply_icp_registration_flow` Prefect flow to `postprocess_workflow_kinect_auto.py`
- [ ] Update `pipeline_stages` in `run_single_session_postprocessing`:
  - Stage 1: `apply_icp_registration` — reads from `source_files` (sessions/), outputs to `sessions_registered/`, stores result as `registered_files`
  - Stage 2: `set_xyz_reference_from_gestures` — reads from `registered_files` (Step 1 output), outputs to `sessions_pca_calibrated/`
- [ ] Fix double-nested directory bug: the PCA flow appends subfolder name (line 51) AND the stage params also include it (line 170) — ensure single-level paths
- [ ] Update DAG config: add `apply_icp_registration` task, set `set_xyz_reference_from_gestures` to depend on it, enable both

**Files Modified:**
- `code/scripts/postprocess_workflow_kinect_auto.py` — new flow, updated pipeline stages
- `configs/postprocess_workflow_kinect_auto_dag.yaml` — new task entry, updated dependencies

**Dependencies:** Phase 3, Phase 4

### Phase 6: Downstream Cleanup
**Goal:** Remove dead code paths and disable flags that depend on `_transformed` columns.

- [ ] Remove unused `resolve_column` import from `neural_kinect_scene_viewer.py` (line 98 — imported but never called; viewer applies transforms at render time from JSON)
- [ ] Set `use_transformed: false` in `configs/analyse_workflow_dag.yaml` under affected tasks (or remove the option)
- [ ] Remove `use_transformed: true` from the disabled `determine_receptive_field` options in `configs/postprocess_workflow_kinect_auto_dag.yaml`
- [ ] Archive `docs/development/plans/ideas/migrate-xyz-reference-to-preprocessing.md` with superseded note
- [ ] Move this plan to `docs/development/plans/active/`

**Files Modified:**
- `code/src/merging/gui/neural_kinect_scene_viewer.py` — remove dead import
- `configs/analyse_workflow_dag.yaml` — disable use_transformed
- `configs/postprocess_workflow_kinect_auto_dag.yaml` — cleanup deprecated task options

**Dependencies:** None (independent of other phases)

---

## Testing Plan

### Unit Tests
- [ ] `transform_spatial_columns_in_place` with a DataFrame that has all 4 spatial columns + a 4x4 transform: verify originals are overwritten correctly
- [ ] `transform_spatial_columns_in_place` with identity matrix: verify values unchanged, any `_transformed` columns dropped
- [ ] `transform_spatial_columns_in_place` with legacy `_transformed` columns present: verify they are dropped
- [ ] `find_applicable_transform_key` with multi-block transforms: verify correct key selection (same block earliest, preceding propagation, no-preceding returns None)
- [ ] PCA transform on contact_location columns: verify output matches manual `apply_full_transform` computation
- [ ] PCA transform on contact_points: verify per-row point arrays are correctly transformed and re-serialized

### Integration Tests
- [ ] Full two-step pipeline on a multi-forearm session: verify `sessions_registered/` and `sessions_pca_calibrated/` outputs
- [ ] Full two-step pipeline on a single-forearm session (no transforms JSON): verify data passes through step 1 unchanged, step 2 applies PCA

### Manual Verification
- [ ] Compare Step 1 output spatial values with old `_transformed` column values from existing merged CSVs — should match
- [ ] Compare Step 2 sticker outputs with old xyz reference output on same data — should be identical
- [ ] Inspect `sessions_registered/` CSVs: confirm no `_transformed` columns, spatial values are ICP-transformed
- [ ] Inspect `sessions_pca_calibrated/` CSVs: confirm all spatial columns are PCA-calibrated
- [ ] Run full pipeline end-to-end via DAG launcher

### Edge Cases
- [ ] Empty `contact_points` cells (`"[]"` or NaN): preserved through both steps
- [ ] All-NaN rows in contact location columns: NaN preserved via masking
- [ ] Idempotency: re-running with unchanged inputs skips processing
- [ ] Block with no applicable transform key (before any snapshot): pass through unchanged with warning

---

## Documentation Plan

- [ ] Archive idea file with superseded note
- [ ] Update plan status through lifecycle (pending -> active -> completed)
- [ ] Add inline comments in new/modified scripts for complex logic

---

## Rollback Plan

1. `git revert` the feature branch commits — all changes are in script files and YAML config
2. Re-enable `transform_to_registered_frame` in preprocessing DAG config, restore `unify_processed_data` dependency and context key
3. No data migration: input CSVs in `sessions/` are never modified; new output folders (`sessions_registered/`, `sessions_pca_calibrated/`) can be deleted
4. Aggregation and analysis pipelines are untouched, so they continue working regardless

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Existing merged CSVs on disk have `_transformed` columns from old preprocessing runs | High | Low | `transform_spatial_columns_in_place` drops them before applying fresh transform |
| `apply_full_transform` mutates input array in-place | Certain | Medium | Extract `.values.copy()` before passing to PCA engine for contact locations |
| Per-row `contact_points` parsing slow for large CSVs | Medium | Low | Same approach used in production `csv_spatial_transformer.py`; optimize only if profiling shows bottleneck |
| Preprocessing re-run needed to produce CSVs without `_transformed` columns | Expected | Low | Postprocessing handles both old (with `_transformed`) and new (without) CSVs gracefully |
| CuPy import order violation from importing `csv_spatial_transformer` | None | N/A | Verified: `csv_spatial_transformer.py` only imports numpy/pandas/re — no CuPy dependency |
| Double-nested output dir bug in existing workflow | Medium | Medium | Explicitly fix subfolder path construction in Phase 5 |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Preprocessing Disentanglement | Trivial (~5 lines changed) | None |
| Phase 2: Shared Transform Utilities | Small (~120 lines: move + new function) | None |
| Phase 3: ICP Registration Script | Small (~100 lines) | Phase 2 |
| Phase 4: Extend PCA Calibration | Small (~50 lines added) | Phase 2 |
| Phase 5: Workflow Integration | Medium (~80 lines modified) | Phase 3, 4 |
| Phase 6: Downstream Cleanup | Trivial (config + dead import) | None |

---

## References

- Superseded idea: `docs/development/plans/ideas/migrate-xyz-reference-to-preprocessing.md`
- PCA calibration engine: `code/src/postprocessing/xyz_reference_from_gestures/calibration_pca_engine.py`
- Spatial transformer utilities: `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py`
- ICP registration transform application (preprocessing): `code/scripts/_3_preprocessing/_3_forearm_extraction/apply_registration_transform.py`
- Knowledge base — ICP registration: `docs/development/knowledge-base/note-forearm-icp-registration.md`
- Knowledge base — somatosensory units: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
