# Plan: Reimplement Nerve Data Extraction Pipeline (mat → block-order)

**Date:** 2026-05-15
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `dev`
**Branch:** `feature/nerve-data-extraction-pipeline`

---

## Overview

**What:** Reimplement the deleted MATLAB nerve preprocessing scripts as a Python
pipeline that converts raw `.mat` nerve recordings into block-order-standardised
CSVs, using `scipy.io.loadmat` to replace MATLAB dependencies entirely.

**Why:** Three scripts that produced the intermediate nerve data folders
(`0_matlab_files` → `1_csv_files` → `2_block-order`) were deleted in commit
`c047112` (2025-08-26) during the `source/` → `code/` refactoring. The
`2_block-order/` outputs are the input to the conduction velocity adjustment
pipeline (planned separately). Without these scripts, all intermediate nerve
artefacts are static and unreproducible.

**How:** Merge the date correction (MATLAB `2.0.4`) and mat-to-csv conversion
(MATLAB `2.0.5`) into a single Python step using `scipy.io.loadmat`, then port
the block-order renaming (Python `2.0.6`) — wired as two upstream tasks in the
unified "Nerve [Auto]" pipeline alongside the conduction velocity adjustment
(planned in `nerve-conduction-velocity-reimplementation.md`).

## Problem Statement

The merging pipeline depends on `2_processed/nerve/3_cond-velocity-adj/` CSVs,
which in turn depend on `2_processed/nerve/2_block-order/` CSVs. No current code
path produces either intermediate folder. The NCV reimplementation plan (pending)
addresses the `2_block-order → 3_cond-velocity-adj` step, but the upstream
`0_matlab_files → 2_block-order` chain has no reimplementation plan.

If the `2_block-order/` artefacts are lost, or new sessions are added, the
original MATLAB scripts cannot be run without MATLAB, and the Python renaming
script was also deleted.

## Goals

### In Scope

1. Reimplement `.mat` → CSV conversion in Python (`scipy.io.loadmat`), merging
   date correction (2.0.4) and table extraction (2.0.5) into a single step
2. Port the block-order CSV renaming (2.0.6) to follow current pipeline patterns
3. Wire both as upstream tasks in the unified "Nerve [Auto]" DAG pipeline
   (single launcher entry with all three nerve preprocessing tasks)
4. Write unit tests for both core functions
5. Update the knowledge-base note with the full deleted pipeline chain

### Out of Scope

- Reimplementing the conduction velocity adjustment (already planned in
  `nerve-conduction-velocity-reimplementation.md`)
- Saving intermediate corrected `.mat` files (date correction happens in-memory)
- Processing non-`Semi_contr` stimuli from the `.mat` files
- Supporting `.mat` files from experiments other than the June 2022 semi-controlled
  sessions (ST13–ST18)

## Success Criteria

- [x] `convert_nerve_mat_to_csv()` correctly loads `.mat` files with
      `scipy.io.loadmat`, applies date corrections, and writes per-block CSVs
      matching the original MATLAB output format
- [x] `rename_nerve_to_block_order()` correctly maps `(zoom_id, block_id)` →
      `block_order` using the quality-check xlsx
- [x] Pipeline skips processing when outputs already exist (`should_process_task`)
- [x] Pipeline regenerates outputs when `force_processing: true`
- [x] "Nerve [Auto]" in the GUI launcher shows all three tasks
      (`convert_mat_to_csv` → `rename_to_block_order` → `adjust_conduction_velocity`)
- [ ] Regression check: CSV output for ST14-01 matches existing `1_csv_files/`
      and `2_block-order/` artefacts
- [ ] All unit tests pass
- [x] Knowledge-base note updated with full pipeline chain

---

## Technical Design

### Approach

Add two upstream tasks to the unified "Nerve [Auto]" pipeline (shared with the
conduction velocity adjustment plan). The date correction and mat-to-csv
conversion are merged into task 1 (no intermediate `.mat` output), the
block-order renaming is task 2, and the existing conduction velocity adjustment
becomes task 3. All three tasks share a single DAG config, pipeline script, and
launcher entry.

**MATLAB → Python translation strategy:**

The key MATLAB constructs and their `scipy.io.loadmat` equivalents:

| MATLAB | Python (`scipy.io.loadmat`) |
|--------|---------------------------|
| `S = data.S` | `mat = loadmat(path); S = mat['S']` |
| `S.Exp` | `S['Exp'][0, 0].item()` (scalar in nested array) |
| `S.UnitName` | `S['UnitName'][0, 0].item()` (string) |
| `S.Stimulus` | `S['Stimulus'][0, 0].item()` |
| `S.FullPeriod_D.ContD(b).D` | `S['FullPeriod_D'][0,0]['ContD'][0,0][0, b]['D'][0, 0]` (structured array → DataFrame) |
| `data(b).D.YYYYMMDD` | Access column from the struct-to-DataFrame conversion |
| `writetable(tableData, csvFileName)` | `pd.DataFrame(...).to_csv(...)` |

The `.mat` struct navigation with `scipy.io.loadmat` requires careful handling of
numpy structured arrays. The exact field access pattern will be validated against
a real `.mat` file during Phase 1.

**Input path:** The `.mat` files directory is passed as a DAG parameter
(`nerve_mat_dir`), not derived from per-block kinect configs, because `.mat` files
are per-session and don't map to the per-block config model.

**Date correction in-memory:** Instead of writing corrected `.mat` files, the date
correction is applied to the in-memory table before CSV serialisation. The
correction rules are hardcoded as a lookup (4 known sessions), matching the
original MATLAB script.

**Session deduplication:** Kinect configs are per-block but `.mat` processing is
per-session. The batch processor deduplicates by `config.session_id`.

### GUI Launcher Integration

A single "Nerve [Auto]" entry in `configs/launcher.yaml` under **Preprocess**,
shared with the conduction velocity adjustment plan:

```yaml
  - name: Preprocess
    workflows:
      ...
      - name: Nerve [Auto]
        script: code/scripts/preprocess_pipeline_nerve_auto.py
        dag_config: configs/preprocess_pipeline_nerve_auto_dag.yaml
```

**What the user sees when clicking "Nerve [Auto]":**

Center panel — three-task DAG table:

```
┌──────────────────────────────┬─────────┬──────────────────────────┐
│ Task Name                    │ Enabled │ Depends On               │
├──────────────────────────────┼─────────┼──────────────────────────┤
│ convert_mat_to_csv           │ [✓] [F] │ —                        │
│ rename_to_block_order        │ [✓] [F] │ convert_mat_to_csv       │
│ adjust_conduction_velocity   │ [✓] [F] │ rename_to_block_order    │
└──────────────────────────────┴─────────┴──────────────────────────┘
```

Right panel — session config selector with the kinect configs from the DAG:

```
☑ valid_configs_ST14-01/
☑ valid_configs_ST14-02/
☑ valid_configs_ST14-04/
☑ valid_configs_ST15-01/
☑ valid_configs_ST16-02/
  ...
```

Each task has an **Enabled** checkbox and **Force** checkbox (`[F]` =
`force_processing: true`). The pipeline deduplicates by `session_id` at runtime.
Users can disable upstream tasks if intermediate artefacts already exist (e.g.
disable `convert_mat_to_csv` and `rename_to_block_order` to run only the
conduction velocity adjustment).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Single merged Python step (date + mat2csv) | No intermediate files; simpler pipeline; no MATLAB dependency | Loses traceability of intermediate `.mat` files | **Chosen** |
| Separate tasks for date correction and mat2csv | Preserves original stage boundaries | Requires writing/reading intermediate `.mat` files; more code; `scipy.io.savemat` adds complexity | Rejected |
| Keep MATLAB scripts, call from Python via subprocess | Exact reproduction of original logic | Requires MATLAB license; fragile subprocess calls; defeats purpose | Rejected |

### Architecture Changes

```
code/scripts/
  preprocess_pipeline_nerve_auto.py                        # MOD — add upstream tasks
  _3_preprocessing/
    _8_nerve_velocity_adjustment/                           # (from conduction velocity plan)
    _9_nerve_data_extraction/
      __init__.py                                          # NEW — re-export
      convert_nerve_mat_to_csv.py                          # NEW — mat→csv core
      rename_nerve_to_block_order.py                       # NEW — block-order rename

configs/
  preprocess_pipeline_nerve_auto_dag.yaml                  # MOD — add upstream tasks
  launcher.yaml                                            # MOD — unchanged if NCV plan
                                                           #        already added the entry

code/tests/
  test_nerve_data_extraction.py                            # NEW — unit tests

docs/development/knowledge-base/
  note-nerve-conduction-velocity-adjustment.md             # MOD — expand to full chain
```

No changes to `KinectConfig`, no changes to the 130 kinect config YAML files,
no changes to the merging pipeline. The pipeline script and DAG config are shared
with the conduction velocity adjustment plan — whichever plan is implemented
first creates the files, the other extends them.

---

## Implementation Plan

### Phase 1: Mat-to-CSV Core Function

**Goal:** Implement and test the Python replacement for MATLAB scripts 2.0.4 +
2.0.5.

- [x] Create `code/scripts/_3_preprocessing/_9_nerve_data_extraction/__init__.py`
- [x] Create `convert_nerve_mat_to_csv.py` with core logic
- [x] Validate `scipy.io.loadmat` field access against a real `.mat` file
- [x] Write unit tests for mat-to-csv conversion

**Files Created:**
- `code/scripts/_3_preprocessing/_9_nerve_data_extraction/__init__.py`
- `code/scripts/_3_preprocessing/_9_nerve_data_extraction/convert_nerve_mat_to_csv.py`

**Core function logic** (merged from deleted scripts):

```python
def convert_nerve_mat_to_csv(
    mat_path: Path,
    output_dir: Path,
    force_processing: bool = False,
) -> list[dict] | None:
```

1. `should_process_task(output_paths=output_dir, input_paths=mat_path, force=force_processing)` — return `None` if skip
2. `clean_task_outputs(output_dir)`
3. Load `.mat` with `scipy.io.loadmat(mat_path, squeeze_me=False)`
4. Extract metadata: `Exp`, `Unit`, `Zoom`, `UnitName`, `UnitNumber`,
   `IdxInDataInfo`, `UnitType`, `Stimulus`
5. Skip if `Stimulus != "Semi_contr"`
6. **Date correction (in-memory):** If filename matches known sessions
   (ST14-01, ST14-02: −1 day; ST16-02: +2 days), adjust `YYYYMMDD` column
   in each block's table
7. Create session folder: `{YYYY-MM-DD}_{UnitName}/`
8. For each block `b` in `S.FullPeriod_D.ContD`:
   - Extract table `D` → `pd.DataFrame`
   - Write CSV: `{matname}_block{b}_table.csv`
9. Write metadata text file: `{matname}_metadata.txt`
10. Return list of `{"mat_file", "session_folder", "n_blocks", "date_corrected"}` dicts

**Date correction lookup:**
```python
DATE_CORRECTIONS = {
    "ST14-01": timedelta(days=-1),
    "ST14-02": timedelta(days=-1),
    "ST16-01": timedelta(days=-1),  # present in original but not in data
    "ST16-02": timedelta(days=+2),
}
```

**Reused utilities:**
- `code/src/utils/should_process_task.py` — `should_process_task()`,
  `clean_task_outputs()`

**Dependencies:** None

### Phase 2: Block-Order Rename Core Function

**Goal:** Port the deleted Python script `2.0.6` to follow current pipeline
patterns.

- [x] Create `rename_nerve_to_block_order.py` with core logic
- [x] Write unit tests for block-order renaming

**Files Created:**
- `code/scripts/_3_preprocessing/_9_nerve_data_extraction/rename_nerve_to_block_order.py`

**Core function logic** (ported from deleted `2.0.6`):

```python
def rename_nerve_to_block_order(
    csv_dir: Path,
    output_dir: Path,
    quality_check_xlsx: Path,
    session_id: str,
    force_processing: bool = False,
) -> list[dict] | None:
```

1. `should_process_task(output_paths=output_dir, input_paths=csv_dir, force=force_processing)` — return `None` if skip
2. `clean_task_outputs(output_dir)`
3. Load quality-check xlsx (`semicontrolled_data-collection_quality-check.xlsx`)
4. For each `*_table.csv` in `csv_dir/{session_id}/`:
   - Extract `zoom_id` and `block_id` from filename
   - Look up `block_order` from xlsx via `(zoom_id, block_id)` mapping
   - Copy CSV to `output_dir/{session_id}/{session_id}_semicontrolled_block-order{NN}_nerve.csv`
5. Return list of `{"input_file", "output_file", "block_order"}` dicts

**Dependencies:** Phase 1

### Phase 3: Pipeline Script and DAG Config

**Goal:** Wire core functions into the unified "Nerve [Auto]" pipeline alongside
the conduction velocity adjustment task.

- [x] Add `convert_mat_to_csv` and `rename_to_block_order` tasks to
      `code/scripts/preprocess_pipeline_nerve_auto.py`
- [x] Add upstream tasks and parameters to
      `configs/preprocess_pipeline_nerve_auto_dag.yaml`
- [x] Add "Nerve [Auto]" entry to `configs/launcher.yaml` (if not already added
      by the conduction velocity plan)

**Files Modified:**
- `code/scripts/preprocess_pipeline_nerve_auto.py` — Add upstream task functions
  and integrate into session pipeline flow
- `configs/preprocess_pipeline_nerve_auto_dag.yaml` — Add upstream tasks and
  parameters (`nerve_mat_dir`, `nerve_csv_output_dir`, `quality_check_xlsx`)
- `configs/launcher.yaml` — Add entry if not already present

**Pipeline structure:**
- `convert_mat_to_csv_task` — Prefect `@task`: discovers `.mat` files in input
  directory, calls `convert_nerve_mat_to_csv()` for each
- `rename_to_block_order_task` — Prefect `@task`: calls
  `rename_nerve_to_block_order()` for each session
- `adjust_conduction_velocity_task` — Prefect `@task` (from conduction velocity
  plan): calls `adjust_nerve_conduction_velocity()` for each session
- `run_single_session_pipeline` — Resolves paths, runs all three tasks in sequence
- `run_batch_processing` — Iterates configs with session deduplication
- `main()` — Entry point (argparse + DagConfigHandler + resolve_session_configs)

**DAG config (unified):**
```yaml
parameters:
  parallel_execution: false
  kinect_configs: [valid_configs_ST14-01, valid_configs_ST14-02,
      valid_configs_ST14-04, valid_configs_ST15-01, valid_configs_ST16-02,
      valid_configs_ST16-03, valid_configs_ST16-05, valid_configs_ST18-01,
      valid_configs_ST18-04]
  nerve_mat_dir: "2_processed/nerve/0_matlab_files"
  nerve_csv_output_dir: "2_processed/nerve/1_csv_files"
  nerve_block_order_output_dir: "2_processed/nerve/2_block-order"
  quality_check_xlsx: "semicontrolled_data-collection_quality-check.xlsx"
  nerve_metadata_csv: "1_primary/nerve/semicontrol_unit-name_to_unit-type.csv"

tasks:
  convert_mat_to_csv:
    enabled: true
    description: "Convert .mat nerve files to CSV with in-memory date correction"
    options:
      force_processing: false
    depends_on: []

  rename_to_block_order:
    enabled: true
    description: "Rename nerve CSVs to kinect block-order naming convention"
    options:
      force_processing: false
    depends_on: [convert_mat_to_csv]

  adjust_conduction_velocity:
    enabled: true
    description: "Shift nerve spike timing to compensate for neural conduction delay"
    options:
      force_processing: false
    depends_on: [rename_to_block_order]
```

**Dependencies:** Phase 2

### Phase 4: Documentation

**Goal:** Expand the knowledge-base note and close documentation gaps.

- [x] Update `note-nerve-conduction-velocity-adjustment.md` to document the full
      deleted pipeline chain (all 6 folders, all 5 scripts, all recovery commands)
- [x] Document `0_files_no-TTL` as baseline prior data
- [x] Add pipeline diagram showing the complete chain
- [x] Update sections 4–5 to reflect both reimplementation plans

**Files Modified:**
- `docs/development/knowledge-base/note-nerve-conduction-velocity-adjustment.md`

**Dependencies:** Phase 3

---

## Testing Plan

### Unit Tests

- [x] `test_loadmat_field_access` — Load a real or synthetic `.mat` file, verify
      all expected fields (`Exp`, `UnitName`, `Stimulus`, `FullPeriod_D.ContD.D`)
      are accessible via `scipy.io.loadmat`
- [x] `test_date_correction_applied` — Verify ST14-01 session applies −1 day
      correction to `YYYYMMDD` column
- [x] `test_date_correction_skipped` — Verify sessions not in the lookup are
      unchanged
- [x] `test_non_semicontr_skipped` — `.mat` file with `Stimulus != "Semi_contr"`
      produces no output
- [x] `test_csv_output_format` — Verify output CSV columns and values match
      expected format from MATLAB `writetable`
- [x] `test_metadata_file_written` — Verify metadata text file contains expected
      fields
- [x] `test_block_order_mapping` — Verify `(zoom_id=3, block_id=2)` → correct
      `block_order` from xlsx
- [x] `test_block_order_filename_format` — Verify output filename matches
      `{session}_semicontrolled_block-order{NN}_nerve.csv`
- [x] `test_missing_zoom_block_raises` — Unknown `(zoom_id, block_id)` combination
      raises `ValueError`
- [x] `test_idempotency_skip` — Output exists, `force=False` → skip
- [x] `test_force_reprocessing` — Output exists, `force=True` → regenerate

### Manual Verification

- [ ] Run pipeline on ST14-01 with `force_processing: true` — diff output against
      existing `1_csv_files/2022-06-15_ST14-01/` and
      `2_block-order/2022-06-15_ST14-01/` artefacts
- [ ] Launch GUI — verify "Nerve [Auto]" under Preprocess shows all three tasks
      in the task panel
- [ ] Run full batch — verify all sessions produce output matching existing
      artefacts

### Edge Cases

- [ ] `.mat` file with only one block — verify single CSV output
- [ ] Session with no `.mat` files in input directory — `FileNotFoundError` raised
- [ ] `.mat` file with corrupted or missing `FullPeriod_D` field — `ValueError`
      raised with clear message

---

## Documentation Plan

- [x] Update knowledge-base note (`note-nerve-conduction-velocity-adjustment.md`)
      with full pipeline chain
- [x] No CLAUDE.md changes needed (no new architectural pattern introduced)

---

## Rollback Plan

1. Delete new files: `_9_nerve_data_extraction/`, `test_nerve_data_extraction.py`
2. Remove the two upstream tasks from `preprocess_pipeline_nerve_auto.py` and
   `preprocess_pipeline_nerve_auto_dag.yaml`
3. Revert knowledge-base changes
4. Existing `1_csv_files/` and `2_block-order/` artefacts are never modified
   (read-only unless `force_processing: true`)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `scipy.io.loadmat` struct navigation differs from expected | Medium | High | Validate against a real `.mat` file in Phase 1 before writing conversion logic; add detailed field-access assertions in tests |
| `.mat` file format version incompatibility (v5 vs v7.3/HDF5) | Low | High | Check `.mat` version; if v7.3, use `h5py` instead of `scipy.io.loadmat`; raise immediately if unsupported |
| Quality-check xlsx column names changed since 2.0.6 was deleted | Low | Medium | Validate column names at startup; raise if unexpected |
| Date correction lookup has stale session list | Low | Low | The 4 sessions are from June 2022 data collection and won't change; document the hardcoded list |
| CSV output column order differs between MATLAB `writetable` and `pd.DataFrame.to_csv` | Medium | Medium | Regression test against existing artefacts; match column order explicitly if needed |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Mat-to-CSV core + tests | ~1.5 hours | None |
| Phase 2: Block-order rename + tests | ~45 min | Phase 1 |
| Phase 3: Pipeline + DAG + launcher | ~45 min | Phase 2 |
| Phase 4: Documentation | ~30 min | Phase 3 |

---

## References

- Knowledge-base note: `docs/development/knowledge-base/note-nerve-conduction-velocity-adjustment.md`
- Deleted MATLAB date correction: `git show c047112~1:source/2_primary_processing/2_standard_naming_and_formatting/2.0.4_processed_nerve_mat_correct_date.m`
- Deleted MATLAB mat2csv: `git show c047112~1:source/2_primary_processing/2_standard_naming_and_formatting/2.0.5_processed_nerve_mat2csv.m`
- Deleted Python rename: `git show c047112~1:source/2_primary_processing/2_standard_naming_and_formatting/2.0.6_processed_nerve_standardize_filename_with_TTLdata.py`
- NCV reimplementation plan: `docs/development/plans/pending/nerve-conduction-velocity-reimplementation.md`
- Merging pipeline (structural template): `code/scripts/merging_pipeline_neuron_to_kinect_auto.py`
- Idempotency utility: `code/src/utils/should_process_task.py`
