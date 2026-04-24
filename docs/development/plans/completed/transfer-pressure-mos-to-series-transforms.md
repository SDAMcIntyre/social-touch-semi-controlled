# Plan: Transfer pressure and MoS time series to Stage 2a series transforms

**Date:** 2026-04-23
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-04-24 17:07
**Branch:** `feature/align-clustering-pipeline-with-guidelines`

---

## Overview

**What:** Move per-frame time series creation for geometric pressure and
mechanics-of-solids (MoS) quantities from Stage 2b feature extractors into
Stage 2a series transforms, so that feature extraction only performs
aggregation over pre-computed columns.

**Why:** The `split-feature-extraction-into-2a-2b-flows` plan established
the principle that series-level work (series -> series) belongs in Stage 2a
and scalar characterisation (series -> scalar) belongs in Stage 2b.
Currently `PressureExtractor` and `MechanicsOfSolidsExtractor` still create
per-frame arrays in-memory during extraction and immediately discard them
after aggregation.  This violates the 2a/2b boundary, prevents inspection
of the intermediate signals on disk, and duplicates computation when
multiple feature profiles consume the same underlying series.

**How:** Create new series-level compute modules (`pressure.py`,
`mechanics.py`) under `representation/series_level/`, add `get_*` helper
functions that read pre-computed columns with fallback, update
`series_pipeline.py` to persist the new columns in the augmented CSV, and
refactor the two extractors into pure aggregators.

## Problem Statement

1. **Pressure series computed and discarded.**  `PressureExtractor`
   (pressure.py:58-62) computes `depth / area` per frame, aggregates to a
   scalar, and discards the per-frame array.  When both
   `pressure_velocity_mean` and `pressure_velocity_max` features are enabled
   (as in the current DAG config), the same per-frame array is computed
   twice.

2. **MoS series computed and discarded.**  `MechanicsOfSolidsExtractor`
   (mos.py:49-64) creates 5 per-frame arrays (strain, stress, strain_rate,
   elastic_energy, impulse), aggregates them, and discards.  These arrays
   are never inspectable on disk and cannot be reused across profiles.

3. **2a/2b boundary violation.**  The guidelines define Stage 2a as
   "series -> series" transforms and Stage 2b as "series -> scalar vector"
   characterisation.  Creating per-frame signals inside an extractor
   conflates both stages.

4. **No intermediate artifact.**  Unlike kinematics (velocity,
   acceleration), pressure and MoS per-frame signals are invisible — there
   is no way to inspect or validate the intermediate representation without
   running a debugger.

## Goals

### In Scope

1. Create `representation/series_level/pressure.py` with per-frame
   geometric pressure computation and a `get_geo_pressure()` helper.
2. Create `representation/series_level/mechanics.py` with per-frame MoS
   computations (strain, stress, strain_rate, elastic_energy, impulse)
   and a `get_mechanics()` helper.
3. Update `series_pipeline.py` to compute and persist pressure and MoS
   columns in the augmented CSV, controlled by new transform config blocks.
4. Refactor `PressureExtractor` to read pre-computed `geo_pressure` (and
   velocity via existing `get_kinematics()`) and only aggregate.
5. Refactor `MechanicsOfSolidsExtractor` to read pre-computed MoS columns
   and only aggregate.
6. Add `pressure` and `mechanics_of_solids` config blocks under
   `transforms` in the DAG YAML.
7. Maintain backward compatibility: extractors fall back to in-memory
   computation when pre-computed columns are absent (same pattern as
   `get_kinematics()`).

### Out of Scope

- Adding pressure or MoS variables to `StatisticalExtractor` — it
  currently aggregates depth, area, velocity, acceleration and that scope
  is unchanged.
- Adding new MoS features (e.g. shear stress, viscoelastic terms).
- Changes to the clustering pipeline, comparing pipeline, or GUI.
- Changes to the on-disk output format of Stage 2b (per-feature CSVs).
- Adding new aggregation functions to existing extractors.

## Success Criteria

- [ ] Running `touch_series_transforms` with pressure and MoS transforms
  enabled produces augmented CSVs containing `geo_pressure`,
  `mos_strain`, `mos_stress_kpa`, `mos_strain_rate`,
  `mos_elastic_energy_mj`, and `mos_impulse_mns` columns alongside the
  existing `velocity_magnitude` and `acceleration_magnitude`.
- [ ] `PressureExtractor.extract()` reads the `geo_pressure` column when
  present and produces identical output to the current implementation.
- [ ] `MechanicsOfSolidsExtractor.extract()` reads MoS columns when
  present and produces identical output to the current implementation.
- [ ] Running `touch_feature_extraction` without Stage 2a (transforms
  disabled) still works — extractors fall back to in-memory computation.
- [ ] Per-feature CSV output is numerically identical whether or not the
  new Stage 2a transforms ran first.
- [ ] No regression in downstream tasks (`touch_clustering`,
  `map_receptive_fields_clustered`).

---

## Technical Design

### Approach

Follow the exact pattern established by the kinematics split:

1. **Compute module** in `representation/series_level/` with pure functions
   that take a touch group DataFrame and return per-frame pd.Series.
2. **`get_*` helper** that checks for pre-computed columns first, falling
   back to computation.  This is the single function extractors call.
3. **`series_pipeline.py`** iterates touch groups, calls the compute
   functions, and assigns results as new columns in the augmented CSV.
4. **Extractors** call the helpers and only aggregate the returned series.

**Augmented CSV columns added (all per-frame, one value per row):**

| Column | Units | Formula | Source |
|--------|-------|---------|--------|
| `geo_pressure` | mm⁻¹ | `contact_depth / contact_area` (NaN if area <= 0) | `pressure.py` |
| `mos_strain` | dimensionless | `(depth_mm * 1e-3) / skin_thickness_m` | `mechanics.py` |
| `mos_stress_kpa` | kPa | `E_kpa * strain` | `mechanics.py` |
| `mos_strain_rate` | 1/s | `(velocity_mm_s * 1e-3) / skin_thickness_m` | `mechanics.py` |
| `mos_elastic_energy_mj` | mJ | `0.5 * E * strain^2 * area_m2 * h * 1e6` | `mechanics.py` |
| `mos_impulse_mns` | mN*s | `stress * area_m2 * dt * 1e6` | `mechanics.py` |

**MoS tissue parameters** (Young's modulus, skin thickness, fps) move from
the `features.mechanics_of_solids` config block into
`transforms.mechanics_of_solids`.  The extractor config retains them as
optional overrides for the fallback path only.

**Note on `mos_strain_rate`:** This column depends on velocity, which is
computed by the kinematics transform.  `series_pipeline.py` must compute
kinematics before MoS within the same session loop.  This is naturally
ordered since kinematics runs first in the current loop.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Pre-computed columns in augmented CSV** (chosen) | Same pattern as kinematics; inspectable on disk; backward-compatible fallback; eliminates redundant computation | More columns in augmented CSV (~6 extra float columns) | **Chosen** — consistent with existing architecture |
| Separate CSV per transform type (kinematics.csv, pressure.csv, mos.csv) | Clean separation | Merge step needed before 2b; partial augmented CSVs confusing | Rejected |
| Compute all series in extractors, cache via shared state | No 2a changes needed | Violates stage boundary; cached state is fragile | Rejected |
| Only move pressure, keep MoS in extractors | Smaller scope | MoS has the same architectural problem; incomplete fix | Rejected |

### Architecture Changes

```
code/src/analysis/touch_analytics/
├── series_pipeline.py                                  # MODIFIED — add pressure + MoS transforms
├── representation/
│   └── series_level/
│       ├── kinematics.py                               # UNCHANGED
│       ├── pressure.py                                 # NEW — compute_geo_pressure, get_geo_pressure
│       └── mechanics.py                                # NEW — compute_mos_series, get_mechanics
├── representation/
│   └── feature_characterization/
│       ├── pressure.py                                 # MODIFIED — read from column, aggregate only
│       └── mos.py                                      # MODIFIED — read from columns, aggregate only

configs/
└── analyse_workflow_dag.yaml                           # MODIFIED — add transform config blocks
```

---

## Implementation Plan

### Phase 1: Pressure series transform
**Goal:** Compute `geo_pressure` per-frame in Stage 2a and persist it in
the augmented CSV.  Add a `get_geo_pressure()` helper.

**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Task 1.1 — Create `representation/series_level/pressure.py`:
  - `compute_geo_pressure(group: pd.DataFrame) -> pd.Series`: computes
    `contact_depth / contact_area` per frame with NaN for area <= 0.
  - `get_geo_pressure(group: pd.DataFrame) -> pd.Series`: returns
    `group['geo_pressure']` if present, else calls `compute_geo_pressure()`.
- [x] Task 1.2 — Update `series_pipeline._transform_session()`:
  - Read `transforms.pressure.enabled` config (default True).
  - After kinematics loop, compute `geo_pressure` per touch group via
    `compute_geo_pressure()`.
  - Assign as `df['geo_pressure']` column.
- [x] Task 1.3 — Update `run_series_transforms()` to read and log the
  pressure config alongside kinematics.

**Files Modified:**
- `code/src/analysis/touch_analytics/representation/series_level/pressure.py`
  — new file.
- `code/src/analysis/touch_analytics/series_pipeline.py` — add pressure
  computation in the session loop.

**Dependencies:** None.

### Phase 2: MoS series transforms
**Goal:** Compute MoS per-frame quantities in Stage 2a and persist them in
the augmented CSV.  Add a `get_mechanics()` helper.

**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Task 2.1 — Create `representation/series_level/mechanics.py`:
  - `compute_mos_series(group: pd.DataFrame, E_kpa, h_mm, fps) -> dict[str, pd.Series]`:
    computes strain, stress, strain_rate, elastic_energy, impulse per
    frame.  Uses `get_kinematics(group, fps)` for velocity (reads
    pre-computed when available).  Returns a dict mapping column names
    to Series.
  - `get_mechanics(group: pd.DataFrame, E_kpa, h_mm, fps) -> dict[str, pd.Series]`:
    returns pre-computed columns if all present, else calls
    `compute_mos_series()`.
  - Constants: `MOS_COLUMNS = ['mos_strain', 'mos_stress_kpa',
    'mos_strain_rate', 'mos_elastic_energy_mj', 'mos_impulse_mns']`.
- [x] Task 2.2 — Update `series_pipeline._transform_session()`:
  - Read `transforms.mechanics_of_solids` config (enabled, tissue
    params with defaults matching current `_DEFAULTS` in mos.py).
  - After kinematics + pressure loop, compute MoS per touch group.
    Requires velocity from kinematics (already computed earlier in the
    loop or available in `group['velocity_magnitude']` after assignment).
  - Assign 5 new columns to df.
- [x] Task 2.3 — Update `run_series_transforms()` to read and log the MoS
  config.

**Files Modified:**
- `code/src/analysis/touch_analytics/representation/series_level/mechanics.py`
  — new file.
- `code/src/analysis/touch_analytics/series_pipeline.py` — add MoS
  computation in the session loop.

**Dependencies:** Phase 1 (MoS reuses the same loop structure and depends
on velocity columns from kinematics).

### Phase 3: Refactor extractors to pure aggregators
**Goal:** `PressureExtractor` and `MechanicsOfSolidsExtractor` read
pre-computed columns and only aggregate.  Backward-compatible fallback
when columns are absent.

**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Task 3.1 — Refactor `PressureExtractor.extract()` in
  `feature_characterization/pressure.py`:
  - Replace lines 58-62 (inline depth/area ratio) with a call to
    `get_geo_pressure(group)` from `series_level.pressure`.
  - Velocity already uses `get_kinematics()` — no change needed.
  - The extractor body becomes: read two series, aggregate, return dict.
- [x] Task 3.2 — Refactor `MechanicsOfSolidsExtractor.extract()` in
  `feature_characterization/mos.py`:
  - Replace lines 49-64 (inline MoS computation) with a call to
    `get_mechanics(group, E, h_mm, fps)` from `series_level.mechanics`.
  - Read the 5 returned series, apply the same aggregations
    (max/mean/sum) as currently.
  - Tissue parameters remain in the extractor config for the fallback
    path — when pre-computed columns are present, the helper ignores them.
- [x] Task 3.3 — Add imports for the new helpers to both extractor modules.

**Files Modified:**
- `code/src/analysis/touch_analytics/representation/feature_characterization/pressure.py`
  — refactor to use `get_geo_pressure()`.
- `code/src/analysis/touch_analytics/representation/feature_characterization/mos.py`
  — refactor to use `get_mechanics()`.

**Dependencies:** Phases 1 and 2 (needs the helpers and column names).

### Phase 4: DAG config and documentation
**Goal:** Add the new transform config blocks so the pipeline is
controllable from YAML.

**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Task 4.1 — Add `pressure` and `mechanics_of_solids` blocks under
  `transforms` in `configs/analyse_workflow_dag.yaml`:
  ```yaml
  transforms:
    kinematics:
      enabled: true
      fps: 30.0
    pressure:
      enabled: true
    mechanics_of_solids:
      enabled: false
      youngs_modulus_kpa: 100.0
      poissons_ratio: 0.45
      skin_thickness_mm: 1.5
  ```
  MoS defaults to disabled (mirrors current `features.mechanics_of_solids`
  state).  Pressure defaults to enabled since `pressure_velocity_mean` and
  `pressure_velocity_max` are both currently active.
- [x] Task 4.2 — Docstrings on `compute_geo_pressure`, `get_geo_pressure`,
  `compute_mos_series`, `get_mechanics`.

**Files Modified:**
- `configs/analyse_workflow_dag.yaml` — new transform config blocks.

**Dependencies:** Phases 1-3.

---

## Testing Plan

### Unit Tests

- [ ] `test_compute_geo_pressure_normal` — synthetic group with known
  depth/area, assert per-frame pressure = depth/area.
- [ ] `test_compute_geo_pressure_zero_area` — frames with area=0 yield
  NaN, not errors.
- [ ] `test_get_geo_pressure_precomputed` — group with `geo_pressure`
  column returns it directly without accessing `contact_depth`/`contact_area`.
- [ ] `test_get_geo_pressure_fallback` — group without `geo_pressure`
  column computes from depth/area.
- [ ] `test_compute_mos_series_values` — synthetic group with known
  depth/area/velocity, assert strain = depth_m / h, stress = E * strain, etc.
- [ ] `test_get_mechanics_precomputed` — group with all 5 MoS columns
  returns them directly.
- [ ] `test_get_mechanics_fallback` — group without MoS columns computes
  from raw data.
- [ ] `test_pressure_extractor_from_precomputed` — pass group with
  `geo_pressure` + `velocity_magnitude` columns to
  `PressureExtractor.extract()`.  Assert output matches computing from
  raw depth/area/velocity.
- [ ] `test_mos_extractor_from_precomputed` — pass group with MoS columns
  to `MechanicsOfSolidsExtractor.extract()`.  Assert output matches
  computing from raw data.

### Integration Tests

- [ ] `test_2b_output_identical_with_and_without_pressure_mos_2a` — run
  `run_feature_extraction` on a test session twice: once with augmented
  CSV containing all new columns, once without.  Assert per-feature
  CSVs are numerically identical.
- [ ] `test_augmented_csv_contains_all_columns` — run
  `run_series_transforms` with all transforms enabled.  Assert output
  CSV has columns: `velocity_magnitude`, `acceleration_magnitude`,
  `geo_pressure`, `mos_strain`, `mos_stress_kpa`, `mos_strain_rate`,
  `mos_elastic_energy_mj`, `mos_impulse_mns`.

### Manual Verification

- [ ] Run `touch_series_transforms` from the GUI on a single session.
  Open the augmented CSV and inspect new columns — values should be
  physically plausible (geo_pressure > 0 for contact frames, strain < 1
  for typical skin deformation).
- [ ] Run `touch_feature_extraction` after transforms.  Confirm
  `pressure_velocity_mean` and `mechanics_of_solids` per-feature CSVs
  match previous output.
- [ ] Disable all new transforms in the DAG YAML.  Run
  `touch_feature_extraction` alone.  Confirm fallback works.

### Edge Cases

- [ ] Session with zero-area frames throughout — `geo_pressure` column is
  all NaN, `PressureExtractor` still produces NaN aggregates without
  errors.
- [ ] MoS transform disabled, `mechanics_of_solids` feature enabled —
  extractor falls back to in-memory computation (backward compatibility).
- [ ] Pressure transform enabled, MoS disabled — augmented CSV has
  `geo_pressure` but not MoS columns; MoS extractor falls back.

---

## Documentation Plan

- [ ] Docstrings on all new public functions.
- [ ] No changes to `CLAUDE.md` or `README.md` (internal pipeline change).

---

## Rollback Plan

1. Revert commits from this plan.  All `get_*` helpers default to fallback
   computation when columns are absent, so reverting does not break
   callers.
2. Delete any augmented CSVs produced with the new columns.  Downstream
   stages are unaffected — they read per-feature CSVs, not the augmented
   CSV.
3. No data migration — Stage 2b output format is unchanged.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| MoS tissue parameters in transforms config diverge from feature config | Medium | Medium | Transform config is the single source of truth when 2a runs.  Extractor config only used for fallback path.  Document this clearly in the DAG YAML comments |
| MoS columns depend on velocity — if kinematics transform is disabled, MoS cannot compute `strain_rate` in 2a | Low | Medium | `compute_mos_series` calls `get_kinematics()` which falls back to computing from position data.  Kinematics transform being disabled does not block MoS |
| 6 additional float columns increase augmented CSV size by ~10% for wide DataFrames | High | Low | Disk is cheap; inspectability is worth it.  Same trade-off accepted for kinematics |
| `get_geo_pressure` returns NaN-rich series from pre-computed column, masking a bug where the column was written incorrectly | Low | Medium | Integration test compares extractor output with and without 2a — must match numerically |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — Pressure series transform | 0.25 day | None |
| Phase 2 — MoS series transforms | 0.5 day | Phase 1 |
| Phase 3 — Refactor extractors | 0.25 day | Phases 1, 2 |
| Phase 4 — DAG config and docs | 0.25 day | Phases 1-3 |

---

## References

- **Predecessor plan:**
  `docs/development/plans/active/split-feature-extraction-into-2a-2b-flows.md`
  — established the 2a/2b split pattern and `get_kinematics()` helper
- **Pipeline guidelines:**
  `docs/design/timeseries_clustering_pipeline_guidelines.md` — Stage 2a/2b
  definition
- **Knowledge base — units:**
  `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
  — mm/mm² throughout pipeline, no hidden conversions
- **Current extractors:**
  - `representation/feature_characterization/pressure.py` — PressureExtractor
  - `representation/feature_characterization/mos.py` — MechanicsOfSolidsExtractor
- **Series pipeline:**
  `code/src/analysis/touch_analytics/series_pipeline.py` — Stage 2a
  orchestrator
