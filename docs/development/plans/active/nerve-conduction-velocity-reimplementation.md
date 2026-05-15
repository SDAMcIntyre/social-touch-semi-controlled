# Plan: Re-implement Nerve Conduction Velocity Adjustment Pipeline

**Date:** 2026-05-15
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `dev`
**Branch:** `feature/nerve-conduction-velocity-pipeline`

---

## Overview

**What:** Re-implement the deleted nerve conduction velocity adjustment as a
standalone Prefect pipeline, following the same DAG architecture used by the
existing merging pipeline.

**Why:** The script that produced conduction-velocity-adjusted nerve CSVs was
deleted in commit `c047112` (2025-08-26) during the `source/` to `code/`
refactoring and was never re-implemented. The merging pipeline depends on these
outputs but cannot regenerate them — they are static artefacts.

**How:** Create a lightweight standalone pipeline script with a single-task DAG,
a core processing function under `_3_preprocessing/_8_nerve_velocity_adjustment/`,
and register it in the GUI launcher.

## Problem Statement

The merging pipeline (`merging_pipeline_neuron_to_kinect_auto.py`) reads
pre-adjusted nerve CSVs from `2_processed/nerve/3_cond-velocity-adj/{session_id}/`
via `config.nerve_processed_dir`. No current code path can produce these files.

If new sessions are added, if the adjustment logic needs to change, or if the
existing artefacts are lost, there is no way to regenerate them without manually
recovering the deleted script from git history.

## Goals

### In Scope

1. Re-implement the conduction velocity adjustment logic as an idempotent
   pipeline task
2. Wire it as a standalone Prefect pipeline launchable from the GUI
3. Generate the lag report CSV alongside adjusted files
4. Write unit tests for the core processing function
5. Update the knowledge-base note to reflect the re-implementation

### Out of Scope

- Re-implementing the upstream MATLAB `.mat`-to-CSV conversion scripts (these
  produced the `2_block-order/` inputs and are not needed for current sessions)
- Interactive visualization of original vs shifted signals (the deleted script
  had `plt.show(block=True)` which blocks the pipeline; a separate viewer
  workflow can be added later if needed)
- Adding the adjustment as a task in the merging pipeline (architectural
  mismatch — see Alternatives Considered)

## Success Criteria

- [ ] `adjust_nerve_conduction_velocity()` correctly shifts `Nervespike1` and
      `Freq` columns by the computed lag
- [ ] Pipeline skips processing when outputs already exist (`should_process_task`)
- [ ] Pipeline regenerates outputs when `force_processing: true`
- [ ] "Nerve [Auto]" appears in the GUI launcher under the Preprocess category
- [ ] Regression check: output on ST14-01 matches existing artefacts within
      floating-point tolerance
- [ ] All unit tests pass
- [ ] Knowledge-base note updated

---

## Technical Design

### Approach

Create a standalone pipeline following the merging pipeline's structural pattern:
a single script with Prefect `@flow`/`@task` decorators, driven by a DAG config
YAML, using `KinectConfig` for path resolution.

The core processing function lives under
`code/scripts/_3_preprocessing/_8_nerve_velocity_adjustment/` (next available
number after `_7_unification`). It processes one nerve CSV at a time; the
pipeline flow orchestrates calling it for all files in a session.

**Input path derivation:** The input directory (`2_block-order/`) is not in any
config field. It is derived from the existing `nerve_processed_dir`:

```python
block_order_dir = config.nerve_processed_dir.parent.parent / "2_block-order" / config.session_id
```

**Metadata CSV path:** Stored as a DAG parameter (`nerve_metadata_csv`), relative
to `project_data_root`. This follows the `neural_quality_xlsx` pattern in the
merging DAG — a session-global file that doesn't belong in per-block configs.

**Session deduplication:** Kinect configs are per-block, but the adjustment is
per-session. The batch processor deduplicates by `config.session_id` to avoid
redundant processing.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Standalone pipeline | Clean separation of concerns; natural per-session granularity; independent GUI entry | One more pipeline script | **Chosen** |
| Add to merging pipeline | No new script; adjustment runs automatically before merge | Per-block granularity mismatch — runs N times per session, N-1 skip via idempotency; conflates preprocessing with merging | Rejected |
| Add to kinect preprocessing pipeline | Reuses existing infrastructure | Mixes data modalities (nerve vs kinect/video); kinect pipeline has no nerve path resolution | Rejected |

### Architecture Changes

```
code/scripts/
  preprocess_pipeline_nerve_auto.py                  # NEW — pipeline script
  _3_preprocessing/
    _8_nerve_velocity_adjustment/
      __init__.py                                    # NEW — re-export
      adjust_nerve_conduction_velocity.py            # NEW — core function

configs/
  preprocess_pipeline_nerve_auto_dag.yaml            # NEW — DAG config
  launcher.yaml                                      # MOD — add entry

code/tests/
  test_nerve_conduction_velocity.py                  # NEW — unit tests

docs/development/knowledge-base/
  note-nerve-conduction-velocity-adjustment.md       # MOD — update sections 4-5
```

No changes to `KinectConfig`, no changes to the 130 kinect config YAML files,
no changes to the merging pipeline.

---

## Implementation Plan

### Phase 1: Core Processing Function

**Goal:** Implement and test the single-file adjustment function.

- [x] Create `code/scripts/_3_preprocessing/_8_nerve_velocity_adjustment/__init__.py`
- [x] Create `adjust_nerve_conduction_velocity.py` with core logic
- [x] Write unit tests in `code/tests/test_nerve_conduction_velocity.py`
- [x] Run tests and verify all pass

**Files Created:**
- `code/scripts/_3_preprocessing/_8_nerve_velocity_adjustment/__init__.py` — Re-export `adjust_nerve_conduction_velocity`
- `code/scripts/_3_preprocessing/_8_nerve_velocity_adjustment/adjust_nerve_conduction_velocity.py` — Core processing function (~60 lines)
- `code/tests/test_nerve_conduction_velocity.py` — Unit tests (~120 lines)

**Core function logic** (recovered from `git show c047112~1:source/3_preprocessing/3.1_preprocess_nerve_conduction_velocity.py`):

1. `should_process_task(output_paths=output_csv_path, input_paths=input_csv_path, force=force_processing)` — return `None` if skip
2. `clean_task_outputs(output_csv_path)`
3. Load nerve CSV, load metadata CSV
4. Look up per-unit `conduction_velocity (m/s)` and `electrode_endorgan_distance (cm)`
5. `lag_sec = (distance_cm / 100) / velocity_m_s`
6. `lag_nsample = int(sampling_frequency * lag_sec)` where `sampling_frequency = 1 / np.mean(np.diff(nerve['Sec_FromStart'].values))`
7. Shift `Nervespike1` and `Freq` by `-lag_nsample` with `fill_value=0`
8. Write output CSV
9. Return `{"filename", "lag_sec", "lag_nsample"}` for report

**Reused utilities:**
- `code/src/utils/should_process_task.py` — `should_process_task()`, `clean_task_outputs()`

**Dependencies:** None

### Phase 2: Pipeline Script and DAG Config

**Goal:** Wire the core function into a Prefect pipeline launchable from the GUI.

- [x] Create `code/scripts/preprocess_pipeline_nerve_auto.py`
- [x] Create `configs/preprocess_pipeline_nerve_auto_dag.yaml`
- [x] Add "Nerve [Auto]" entry to `configs/launcher.yaml`

**Files Created:**
- `code/scripts/preprocess_pipeline_nerve_auto.py` — Pipeline script (~120 lines)
- `configs/preprocess_pipeline_nerve_auto_dag.yaml` — DAG config

**Files Modified:**
- `configs/launcher.yaml` — Add entry under Preprocess category

**Pipeline structure:**
- `adjust_conduction_velocity_flow` — Prefect `@flow` that discovers `*_nerve.csv` files in the input directory, calls `adjust_nerve_conduction_velocity()` for each, writes `conduction_velocity_lag_report.csv`
- `run_single_session_pipeline` — Resolves paths from `KinectConfig`, delegates to flow
- `run_batch_processing` — Iterates block files with session deduplication
- `main()` — Entry point (argparse + DagConfigHandler + resolve_session_configs)

**DAG config:**
```yaml
parameters:
  parallel_execution: false
  kinect_configs: [valid_configs_ST14-01, valid_configs_ST14-02,
      valid_configs_ST14-04, valid_configs_ST15-01, valid_configs_ST16-02,
      valid_configs_ST16-03, valid_configs_ST16-05, valid_configs_ST18-01,
      valid_configs_ST18-04]
  nerve_metadata_csv: "1_primary/nerve/semicontrol_unit-name_to_unit-type.csv"

tasks:
  adjust_conduction_velocity:
    enabled: true
    description: "Shift nerve spike timing to compensate for neural conduction delay"
    options:
      force_processing: false
    depends_on: []
```

**Dependencies:** Phase 1

### Phase 3: Documentation

**Goal:** Update the knowledge-base note and close the documentation gap.

- [x] Update `note-nerve-conduction-velocity-adjustment.md` sections 4 and 5
- [x] Add metadata CSV known path to the document

**Files Modified:**
- `docs/development/knowledge-base/note-nerve-conduction-velocity-adjustment.md` — Update "Current state" and "Implications" sections

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests

- [ ] `test_basic_shift` — Create synthetic nerve CSV (known `Sec_FromStart`, `Nervespike1`, `Freq`) and metadata CSV. Verify output columns shifted by exact expected `lag_nsample`
- [ ] `test_idempotency_skip` — Output already exists, `force=False` — function returns `None`, output unchanged
- [ ] `test_force_reprocessing` — Output exists, `force=True` — output regenerated with new content
- [ ] `test_missing_unit_raises` — Unit name not in metadata CSV — `ValueError` raised
- [ ] `test_missing_columns_raises` — `Nervespike1` or `Freq` absent from nerve CSV — `ValueError` raised
- [ ] `test_fill_value_zero` — Shifted tail positions filled with `0`, not `NaN`

### Manual Verification

- [ ] Run pipeline on ST14-01 with `force_processing: true` — diff output against existing `3_cond-velocity-adj/2022-06-15_ST14-01/` artefacts (should match within floating-point tolerance)
- [ ] Launch GUI — verify "Nerve [Auto]" appears under Preprocess category and task panel loads correctly
- [ ] Run full batch with `force_processing: false` — all sessions should skip (existing outputs up-to-date)

### Edge Cases

- [ ] Session with no nerve CSV files in block-order directory — `FileNotFoundError` raised
- [ ] Metadata CSV with zero conduction velocity for a unit — guard against division by zero

---

## Documentation Plan

- [ ] Update knowledge-base note (`note-nerve-conduction-velocity-adjustment.md`)
- [ ] No CLAUDE.md changes needed (no new architectural pattern introduced)

---

## Rollback Plan

1. Delete the new files: `preprocess_pipeline_nerve_auto.py`, `_8_nerve_velocity_adjustment/`, `preprocess_pipeline_nerve_auto_dag.yaml`, `test_nerve_conduction_velocity.py`
2. Revert the launcher.yaml and knowledge-base changes
3. Existing `3_cond-velocity-adj/` artefacts are never modified (read-only unless `force_processing: true`), so no data recovery needed

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Metadata CSV column names differ from deleted script | Low | High | Validate column names at startup; raise immediately if unexpected |
| `2_block-order/` directory missing for some sessions | Medium | Medium | Fail-fast with clear error; document in DAG config comments |
| Unit name extraction from filename doesn't match metadata | Low | High | Verify against actual data during regression check |
| Session deduplication misses edge cases | Low | Low | Use `session_id` from config (authoritative), not derived from paths |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Core function + tests | ~1 hour | None |
| Phase 2: Pipeline + DAG + launcher | ~45 min | Phase 1 |
| Phase 3: Documentation | ~15 min | Phase 2 |

---

## References

- Knowledge-base note: `docs/development/knowledge-base/note-nerve-conduction-velocity-adjustment.md`
- Deleted script recovery: `git show c047112~1:source/3_preprocessing/3.1_preprocess_nerve_conduction_velocity.py`
- Merging pipeline (structural template): `code/scripts/merging_pipeline_neuron_to_kinect_auto.py`
- Idempotency utility: `code/src/utils/should_process_task.py`
