# Plan: Split touch_feature_extraction into Stage 2a and 2b flows

**Created:** 2026-04-23
**Approved:** —
**Completed:** 2026-04-24 17:07
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/align-clustering-pipeline-with-guidelines` (extended)

---

## Overview

**What:** Split the monolithic `touch_feature_extraction` Prefect flow into two
separate flows that map to the Stage 2 sub-stages defined in
`docs/design/timeseries_clustering_pipeline_guidelines.md`:

- **Stage 2a** — `touch_series_transforms_flow`: compute and persist
  per-frame kinematics (velocity, acceleration) as new columns in an
  augmented session CSV.
- **Stage 2b** — `touch_feature_extraction_flow`: read the augmented CSV and
  run scalar feature extractors (unchanged output format).

**Why:** The guidelines explicitly separate series-level transformations
(series → series) from feature characterization (series → scalar vector) as
Stage 2a and 2b.  Currently both are fused inside `extraction_pipeline.py`:
three extractors (`StatisticalExtractor`, `MechanicsOfSolidsExtractor`,
`PressureExtractor`) independently and redundantly call
`compute_velocity_magnitudes()` per touch.  Splitting creates a clean stage
boundary, eliminates redundant computation, and makes the kinematics
artifact inspectable on disk.

**How:** Introduce `series_pipeline.py` with `run_series_transforms()` that
persists an augmented session CSV.  Add a `get_kinematics()` helper to
`kinematics.py` that reads pre-computed columns when available.  Update
extractors to use the helper.  Wire the new flow into the workflow script
and DAG config.

## Problem Statement

1. **Stage 2a is invisible.** Kinematics (velocity, acceleration) are
   computed in-memory inside each extractor and immediately consumed.  No
   intermediate artifact exists — the series-level representation is never
   persisted or inspectable.

2. **Redundant computation.** `compute_velocity_magnitudes()` is called
   independently by `StatisticalExtractor` (`statistical.py:20`),
   `MechanicsOfSolidsExtractor` (`mos.py:52`), and `PressureExtractor`
   (`pressure.py:64`).  When multiple features are enabled, the same
   kinematics are recomputed per touch per extractor.

3. **Single flow hides the pipeline structure.** The workflow script exposes
   one `touch_feature_extraction` flow that covers both 2a and 2b.  Users
   cannot run or inspect series-level transforms independently.

4. **Preparation modules are unused.** The alignment refactor created
   `preparation/loader.py`, `preparation/block_id.py`, and
   `preparation/grouping.py`, but `extraction_pipeline.py` still has inline
   copies of that logic (CSV loading at line 176, block-id synthesis at
   lines 184–191, grouping at line 284).  Only `preparation/direction.py` is
   wired up (line 28).

## Goals

### In Scope

1. Create a new Prefect flow `touch_series_transforms_flow` (Stage 2a) that
   computes and persists kinematics as additional columns in an augmented
   session CSV.
2. Create `series_pipeline.py` with `run_series_transforms()` as the
   pipeline entry point.
3. Add a `get_kinematics()` helper to `kinematics.py` that reads
   pre-computed `velocity_magnitude` / `acceleration_magnitude` columns when
   present, falling back to recomputation for backward compatibility.
4. Update all three kinematic-consuming extractors to use `get_kinematics()`.
5. Update `extraction_pipeline.py` to optionally read augmented CSVs from
   the Stage 2a output directory when available.
6. Wire the new flow into `analysis_workflow.py` and
   `analyse_workflow_dag.yaml`.
7. Wire up the existing `preparation/` modules (`loader`, `block_id`,
   `grouping`) in the new `series_pipeline.py` — the 2a flow is the natural
   place to use them since it processes entire sessions.

### Out of Scope

- Adding new series-level transforms beyond kinematics (e.g. wavelet,
  Fourier) — the `transforms` config structure supports future additions but
  this plan only implements kinematics.
- Refactoring `extraction_pipeline.py` to use `preparation/` modules — the
  existing inline code works; the 2a pipeline is the new canonical user of
  these modules.
- Changes to the clustering pipeline, comparing pipeline, or GUI.
- Changes to the on-disk output format of Stage 2b (per-feature CSVs).

## Success Criteria

- [ ] A `touch_series_transforms` task appears in the DAG config and runs
  before `touch_feature_extraction`.
- [ ] Running `touch_series_transforms` produces augmented CSVs at
  `4_analysed/series_transforms/<session>_series_augmented.csv` with
  `velocity_magnitude` and `acceleration_magnitude` columns.
- [ ] Running `touch_feature_extraction` after `touch_series_transforms`
  reads the augmented CSVs and extractors use pre-computed kinematics
  (no calls to `compute_velocity_magnitudes` from position data).
- [ ] Running `touch_feature_extraction` without `touch_series_transforms`
  (2a disabled) still works — extractors fall back to computing kinematics
  from position data.
- [ ] Per-feature CSV output is byte-identical whether or not Stage 2a ran
  first (same numeric values, same columns, same row order).
- [ ] No regression in downstream tasks (`touch_clustering`,
  `touch_comparing`, `map_receptive_fields_clustered`).

---

## Technical Design

### Approach

**Augmented CSV as the Stage 2a artifact.**  The aggregated session CSV
already has one row per frame per touch, with columns like
`sticker_blue_position_x/y/z`, `contact_depth`, `contact_area`.  The 2a
flow adds `velocity_magnitude` and `acceleration_magnitude` columns (one
value per frame) and saves the result.  This is the natural representation:
same format, just more columns.

**Output location:** `4_analysed/series_transforms/<session_id>_series_augmented.csv`

**`get_kinematics()` helper.**  A single function in `kinematics.py` that
returns `(velocity_series, acceleration_series)` for a touch group.  It
checks for pre-computed columns first:

```python
def get_kinematics(
    group: pd.DataFrame, fps: float = 30.0,
) -> Tuple[pd.Series, pd.Series]:
    if 'velocity_magnitude' in group.columns:
        vel = group['velocity_magnitude']
    else:
        vel = compute_velocity_magnitudes(group, fps=fps)

    if 'acceleration_magnitude' in group.columns:
        accel = group['acceleration_magnitude']
    else:
        accel = compute_acceleration_magnitudes(vel, fps=fps)

    return vel, accel
```

**Extractor changes.**  Each of the three kinematic-consuming extractors
replaces its direct `compute_velocity_magnitudes()` call with
`get_kinematics()`:

| Extractor | Current call (line) | Replacement |
|-----------|-------------------|-------------|
| `StatisticalExtractor` | `compute_velocity_magnitudes(group)` (L20) + `compute_acceleration_magnitudes(vel)` (L21) | `vel, accel = get_kinematics(group, fps)` |
| `MechanicsOfSolidsExtractor` | `compute_velocity_magnitudes(group, fps)` (L52) | `vel, _ = get_kinematics(group, fps)` |
| `PressureExtractor` | `compute_velocity_magnitudes(group)` (L64) | `vel, _ = get_kinematics(group, fps)` |

**Extraction pipeline change.**  `run_feature_extraction` accepts an
optional `series_dir: Path = None` parameter.  When provided,
`_extract_session` looks for
`series_dir / <session_id>_series_augmented.csv` and loads it instead of the
raw session CSV.  If the augmented CSV is not found, it falls back to the
raw CSV with a warning.

**DAG dependency chain:**

```
touch_series_transforms → touch_feature_extraction → touch_clustering → ...
```

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Augmented CSV with pre-computed columns** (chosen) | Same format as input; minimal extractor changes; backward-compatible fallback; kinematics inspectable on disk | Augmented CSV duplicates the session data on disk (~2x storage per session) | **Chosen** — disk is cheap; inspectability and clean stage boundary are worth it |
| Store kinematics in a separate CSV (touch-id + velocity + acceleration only) | No data duplication | Extractors need a merge step; session metadata not available in 2b without also loading the raw CSV | Rejected |
| Persist kinematics as Parquet per session | Smaller on disk; typed columns | Adds a format dependency; less inspectable than CSV | Rejected (YAGNI) |
| Keep the fused flow, add internal caching | No new flow; minimal change | Doesn't create a stage boundary; 2a not independently runnable | Rejected — defeats the purpose |

### Architecture Constraints (from knowledge base)

- **CuPy import order** — `series_pipeline.py` will import from
  `preparation/` and `representation/series_level/` which do not use CuPy.
  No guard needed unless preprocessing imports are added later.

### Architecture Changes

```
code/src/analysis/touch_analytics/
├── series_pipeline.py                              # NEW — Stage 2a orchestrator
├── extraction_pipeline.py                          # MODIFIED — accepts series_dir
├── representation/
│   └── series_level/
│       └── kinematics.py                           # MODIFIED — add get_kinematics()
├── representation/
│   └── feature_characterization/
│       ├── statistical.py                          # MODIFIED — use get_kinematics()
│       ├── mos.py                                  # MODIFIED — use get_kinematics()
│       └── pressure.py                             # MODIFIED — use get_kinematics()
├── __init__.py                                     # MODIFIED — export run_series_transforms

code/scripts/
└── analysis_workflow.py                            # MODIFIED — add flow + forwarding

configs/
└── analyse_workflow_dag.yaml                       # MODIFIED — add task + dependency
```

---

## Implementation Plan

### Phase 1: Stage 2a pipeline and kinematics helper
**Goal:** Create the series-level transform pipeline and the
`get_kinematics()` helper.  After this phase, `run_series_transforms()` can
be called standalone and produces augmented CSVs.
**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Task 1.1 — Add `get_kinematics(group, fps)` to
  `representation/series_level/kinematics.py`.  Returns
  `(velocity_series, acceleration_series)`, using pre-computed columns when
  present, falling back to `compute_velocity_magnitudes` /
  `compute_acceleration_magnitudes`.
- [x] Task 1.2 — Create `series_pipeline.py` with `run_series_transforms()`:
  - Signature: `run_series_transforms(input_items, transforms, output_dir,
    force) -> list[Path]`
  - For each session: load via `preparation/loader.load_session_csv()`,
    ensure block-id via `preparation/block_id.ensure_block_id_column()`,
    group via `preparation/grouping.group_touches()`, compute
    `velocity_magnitude` and `acceleration_magnitude` per group, assign
    columns back, save augmented CSV.
  - Idempotency via `should_process_task` (same pattern as
    `extraction_pipeline`).
  - Config: reads `transforms.kinematics.enabled` and
    `transforms.kinematics.fps` (default 30.0).
- [x] Task 1.3 — Export `run_series_transforms` from
  `touch_analytics/__init__.py`.

**Files Modified:**
- `code/src/analysis/touch_analytics/representation/series_level/kinematics.py`
  — add `get_kinematics()`.
- `code/src/analysis/touch_analytics/series_pipeline.py` — new.
- `code/src/analysis/touch_analytics/__init__.py` — add export.

**Dependencies:** None.

### Phase 2: Update extractors to use pre-computed kinematics
**Goal:** Wire the three kinematic-consuming extractors through
`get_kinematics()` so they benefit from pre-computed columns.
**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Task 2.1 — Update `StatisticalExtractor.extract()` in
  `statistical.py`: replace lines 20–21 with
  `vel, accel = get_kinematics(group, fps=fps)`.
- [x] Task 2.2 — Update `MechanicsOfSolidsExtractor.extract()` in `mos.py`:
  replace line 52 with `vel, _ = get_kinematics(group, fps=fps)`.  Apply
  the `* 1e-3` unit conversion to the result.
- [x] Task 2.3 — Update `PressureExtractor.extract()` in `pressure.py`:
  replace line 64 with `vel, _ = get_kinematics(group, fps=fps)`.

**Files Modified:**
- `code/src/analysis/touch_analytics/representation/feature_characterization/statistical.py`
- `code/src/analysis/touch_analytics/representation/feature_characterization/mos.py`
- `code/src/analysis/touch_analytics/representation/feature_characterization/pressure.py`

**Dependencies:** Phase 1 (needs `get_kinematics`).

### Phase 3: Wire extraction pipeline to read augmented CSVs
**Goal:** `run_feature_extraction` optionally reads from the Stage 2a output
directory, so extractors receive DataFrames with pre-computed kinematics
columns.
**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Task 3.1 — Add `series_dir: Path = None` parameter to
  `run_feature_extraction()` in `extraction_pipeline.py`.
- [x] Task 3.2 — In `_extract_session`, when `series_dir` is provided, look
  for `series_dir / <session_id>_series_augmented.csv`.  If found, load it
  instead of the raw session CSV.  If not found, log a warning and fall
  back to the raw CSV.
- [x] Task 3.3 — Pass `series_dir` through from `run_feature_extraction` to
  `_extract_session` (add parameter).

**Files Modified:**
- `code/src/analysis/touch_analytics/extraction_pipeline.py` —
  `run_feature_extraction` and `_extract_session` signature + loading logic.

**Dependencies:** Phase 1 (needs augmented CSV format), Phase 2 (extractors
ready to use pre-computed columns).

### Phase 4: Workflow and DAG config integration
**Goal:** Add the new Prefect flow to the workflow script and DAG config.
Wire up forwarding so Stage 2a runs before Stage 2b.
**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Task 4.1 — Add `touch_series_transforms_flow` to
  `analysis_workflow.py`:
  ```python
  @flow(name="touch_series_transforms")
  def touch_series_transforms_flow(
      input_items: List[Tuple[Path, Path]],
      force_processing: bool = False,
      transforms: dict = None,
  ) -> List[Path]:
  ```
  Calls `run_series_transforms()` with output dir
  `database / '4_analysed' / 'series_transforms'`.
- [x] Task 4.2 — Add `series_dir` parameter to
  `touch_feature_extraction_flow`.  When the `series_transforms` output
  directory exists, pass it to `run_feature_extraction(series_dir=...)`.
- [x] Task 4.3 — Add `touch_series_transforms` to `available_tasks` in
  `run_batch_analysis`, positioned before `touch_feature_extraction`.
- [x] Task 4.4 — Add `"transforms"` and `"series_dir"` forwarding branches
  to the kwargs dispatch in `run_batch_analysis`.
- [x] Task 4.5 — Add `touch_series_transforms` task to
  `analyse_workflow_dag.yaml`:
  ```yaml
  touch_series_transforms:
    enabled: true
    options:
      force_processing: false
      transforms:
        kinematics:
          enabled: true
          fps: 30.0
    depends_on: []
  ```
- [x] Task 4.6 — Update `touch_feature_extraction.depends_on` in the DAG
  YAML to `[touch_series_transforms]`.
- [x] Task 4.7 — Add import for `run_series_transforms` in
  `analysis_workflow.py`.

**Files Modified:**
- `code/scripts/analysis_workflow.py` — new flow, updated flow, updated
  batch dispatcher, new import.
- `configs/analyse_workflow_dag.yaml` — new task, updated dependency.

**Dependencies:** Phases 1–3.

---

## Testing Plan

### Unit Tests

- [ ] `test_get_kinematics_precomputed` — create a group DataFrame with
  `velocity_magnitude` and `acceleration_magnitude` columns.  Call
  `get_kinematics()`.  Assert it returns the pre-computed values (does not
  touch `sticker_blue_position_*`).
- [ ] `test_get_kinematics_fallback` — create a group DataFrame without
  pre-computed columns.  Call `get_kinematics()`.  Assert it computes from
  position data (same as `compute_velocity_magnitudes`).
- [ ] `test_run_series_transforms_creates_augmented_csv` — call
  `run_series_transforms` on a synthetic session CSV.  Assert the augmented
  CSV exists and contains `velocity_magnitude` and `acceleration_magnitude`
  columns.
- [ ] `test_run_series_transforms_idempotent` — run twice without force.
  Assert the second run skips processing.
- [ ] `test_statistical_extractor_uses_precomputed` — pass a group with
  pre-computed kinematics to `StatisticalExtractor.extract()`.  Assert the
  result matches a group without pre-computed columns (numeric equivalence).
- [ ] `test_mos_extractor_uses_precomputed` — same for
  `MechanicsOfSolidsExtractor`.
- [ ] `test_pressure_extractor_uses_precomputed` — same for
  `PressureExtractor`.

### Integration Tests

- [ ] `test_2b_output_identical_with_and_without_2a` — run
  `run_feature_extraction` on a test session twice: once with `series_dir`
  pointing to a 2a output, once without.  Assert the per-feature CSVs are
  byte-identical.
- [ ] `test_full_pipeline_2a_then_2b_then_clustering` — run all three stages
  in sequence on a test session.  Assert clustering output contains expected
  metadata keys.

### Manual Verification

- [ ] Run `touch_series_transforms` from the GUI on a single session.
  Inspect the augmented CSV — confirm `velocity_magnitude` and
  `acceleration_magnitude` columns are present and contain plausible values.
- [ ] Run `touch_feature_extraction` after.  Confirm per-feature CSVs match
  previous output.
- [ ] Disable `touch_series_transforms` in the DAG YAML.  Run
  `touch_feature_extraction` alone.  Confirm it still works (fallback).
- [ ] Run the full pipeline through `touch_clustering`.  Confirm no
  regression in cluster outputs.

### Edge Cases

- [ ] Session CSV with missing `sticker_blue_position_*` columns — 2a
  should log a warning and skip kinematics (augmented CSV written without
  kinematics columns).  2b extractors fall back gracefully.
- [ ] `transforms.kinematics.enabled: false` — 2a flow runs but produces
  no augmented CSVs.  2b falls back to raw CSVs.
- [ ] Augmented CSV exists but is older than raw session CSV — idempotency
  check triggers recomputation.

---

## Documentation Plan

- [ ] Docstrings on `run_series_transforms`, `get_kinematics`, and
  `touch_series_transforms_flow`.
- [ ] No changes to `CLAUDE.md` or `README.md` (internal pipeline change).

---

## Rollback Plan

1. Revert commits from this plan.  All new parameters default to `None`, so
   reverting does not break callers.
2. Delete the `4_analysed/series_transforms/` directory if any augmented
   CSVs were produced.  Downstream stages do not depend on them when 2a is
   disabled.
3. No data migration — Stage 2b output format is unchanged.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Augmented CSVs double per-session disk usage | High | Low | Disk is cheap; augmented CSVs are the same size as raw session CSVs + 2 float columns (~1% overhead on wide DataFrames); can add a `cleanup` option later |
| `get_kinematics()` silently returns stale pre-computed values if 2a ran with a different `fps` than the extractor expects | Low | Medium | The `fps` parameter defaults to 30.0 everywhere; the 2a config and extractor configs share the same default.  Document that non-default `fps` must be consistent across both stages |
| MechanicsOfSolidsExtractor applies `* 1e-3` unit conversion to velocity — pre-computed column is in mm/s, extractor needs m/s | High | High | The `get_kinematics()` helper returns raw mm/s values (same unit as `compute_velocity_magnitudes`).  The `* 1e-3` conversion stays inside the extractor after the `get_kinematics()` call — no change in unit semantics |
| Loading augmented CSV is slower than raw CSV for sessions where only non-kinematic features are needed (temporal, touch_category) | Low | Low | The extra two float columns add negligible load time; the benefit of a clean stage boundary outweighs this |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — Series pipeline + helper | 0.5 day | None |
| Phase 2 — Update extractors | 0.25 day | Phase 1 |
| Phase 3 — Wire extraction pipeline | 0.25 day | Phase 1, 2 |
| Phase 4 — Workflow + DAG integration | 0.5 day | Phase 1–3 |

---

## References

- **Pipeline guidelines:**
  `docs/design/timeseries_clustering_pipeline_guidelines.md` — Stage 2a/2b
  definition (lines 55–87)
- **Alignment plan:**
  `docs/development/plans/active/align-clustering-pipeline-with-guidelines.md`
  — Phase 2 created the module layout; this plan wires up the runtime split
- **Kinematics module:**
  `code/src/analysis/touch_analytics/representation/series_level/kinematics.py`
  — `compute_velocity_magnitudes`, `compute_acceleration_magnitudes`
- **Extractors consuming kinematics:**
  - `representation/feature_characterization/statistical.py:20–21`
  - `representation/feature_characterization/mos.py:52`
  - `representation/feature_characterization/pressure.py:64`
- **Preparation modules (to wire up in 2a):**
  - `preparation/loader.py` — `load_session_csv()`
  - `preparation/block_id.py` — `ensure_block_id_column()`
  - `preparation/grouping.py` — `group_touches()`
- **Companion plan (config forwarding fix):**
  `docs/development/plans/pending/fix-analysis-workflow-stage-config-forwarding.md`
