# Plan: Expand cluster group features and rename geo_pressure

**Date:** 2026-04-24
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-04-24 17:07
**Branch:** `feature/refactor-series-transforms-sticker-selection`

---

## Overview

The clustering pipeline can only select a subset of the touch-related columns
produced by the series transforms. `hand_position_{x,y,z}` are excluded from
statistical aggregation (never reach feature CSVs), MoS columns have no mapping
in the clustering pipeline, and `geo_pressure` should be renamed to `pressure`
for clarity. This plan unblocks all touch features for clustering and adds a
comprehensive cluster group.

## Problem Statement

Stage 2a (series transforms) computes per-frame columns from raw sensor data:
hand position, velocity, acceleration, pressure, and mechanics-of-solids
quantities. However, only `contact_depth`, `contact_area`,
`velocity_magnitude`, `acceleration_magnitude`, and `location` are usable in
cluster groups today. Two blockers exist:

1. **`hand_position_{x,y,z}`** are in `_EXCLUDE_FROM_AGGREGATION` in
   `statistical.py` (labelled "Hand position intermediates"), so they are never
   aggregated into per-touch feature CSVs.
2. **MoS columns** (`mos_strain`, `mos_stress_kpa`, `mos_strain_rate`,
   `mos_elastic_energy_mj`, `mos_impulse_mns`) pass through aggregation fine
   but have no entries in `DATA_TYPE_TO_COLUMNS` or the GUI `_DATA_TYPES` list,
   so they cannot be selected by any cluster group.

Additionally, `geo_pressure` is an unnecessarily verbose name — it should be
simply `pressure`.

## Goals

### In Scope

1. Rename `geo_pressure` to `pressure` across all code, config, and GUI
2. Un-exclude `hand_position_{x,y,z}` from statistical aggregation
3. Add `hand_position`, individual MoS, and composite `mechanics_of_solids`
   entries to `DATA_TYPE_TO_COLUMNS` and the GUI
4. Add an `all_touch_features_mean` cluster group to the DAG config

### Out of Scope

- Renaming `geo_pressure` in historical plan documents (`completed/`, `active/`)
- Adding new series transforms or extraction methods
- Changing existing cluster groups (`kinematics_mean`, etc.)
- Re-running the pipeline (user responsibility after code changes)

## Success Criteria

- [ ] Column name `pressure` (not `geo_pressure`) in augmented CSVs and feature CSVs
- [ ] `hand_position_x_mean`, `hand_position_y_mean`, `hand_position_z_mean` appear in feature CSVs after re-extraction
- [ ] Cluster group `all_touch_features_mean` selects 16 feature dimensions without warnings
- [ ] Existing `kinematics_mean` group still works unchanged
- [ ] GUI cluster group dialog offers all new data types

---

## Technical Design

### Approach

Minimal, targeted changes: remove the exclusion, add mappings, rename the
column. No new modules, no architectural changes. The `StatisticalExtractor`
auto-discovers numeric columns, so un-excluding `hand_position` is sufficient
to make it flow through. MoS columns already flow through — they just need
a mapping so the clustering pipeline can select them.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Add `hand_position` as regular columns in `DATA_TYPE_TO_COLUMNS` | Simple, consistent with existing pattern | None significant | **Chosen** |
| Add `hand_position` with special handling like `location` | Could reuse shared-column pattern | Requires changes to extraction orchestrator to write shared columns; unnecessary complexity | Rejected |
| Single `mechanics_of_solids` composite key only | Simpler mapping | No fine-grained MoS feature selection | Rejected — offer both individual and composite |

### Architecture Changes

No new modules. Changes are to configuration maps and exclusion lists:

```
statistical.py          _EXCLUDE_FROM_AGGREGATION  →  remove hand_position entries
clustering_pipeline.py  DATA_TYPE_TO_COLUMNS       →  add hand_position + MoS entries
cluster_group_dialog.py _DATA_TYPES                →  mirror DATA_TYPE_TO_COLUMNS
pressure.py (series)    function/column names      →  geo_pressure → pressure
pressure.py (feature)   import/output keys         →  geo_pressure → pressure
series_pipeline.py      import/column assignment   →  geo_pressure → pressure
analyse_workflow_dag.yaml                          →  add cluster group
```

---

## Implementation Plan

### Phase 1: Rename geo_pressure → pressure

**Goal:** Replace `geo_pressure` with `pressure` in all code, config, and GUI.

**Started:** 2026-04-24
**Completed:** 2026-04-24

- [x] Task 1.1 — Rename functions in series-level pressure module
- [x] Task 1.2 — Update series pipeline import and column assignment
- [x] Task 1.3 — Update feature characterization pressure extractor
- [x] Task 1.4 — Rename key in `DATA_TYPE_TO_COLUMNS`
- [x] Task 1.5 — Rename entry in GUI `_DATA_TYPES`
- [x] Task 1.6 — Update YAML config comment

**Files Modified:**

- `code/src/analysis/touch_analytics/representation/series_level/pressure.py`
  — `compute_geo_pressure()` → `compute_pressure()`,
    `get_geo_pressure()` → `get_pressure()`,
    column check `'geo_pressure'` → `'pressure'`
- `code/src/analysis/touch_analytics/series_pipeline.py`
  — import name, `df['geo_pressure']` → `df['pressure']`, comment
- `code/src/analysis/touch_analytics/representation/feature_characterization/pressure.py`
  — import name, output keys `geo_pressure_mean` → `pressure_mean` / `geo_pressure_max` → `pressure_max`, docstrings
- `code/src/analysis/touch_analytics/clustering_pipeline.py`
  — `'geo_pressure'` → `'pressure'` in `DATA_TYPE_TO_COLUMNS`
- `code/src/utils/gui/dag_launcher/cluster_group_dialog.py`
  — `"geo_pressure"` → `"pressure"` in `_DATA_TYPES`
- `configs/analyse_workflow_dag.yaml`
  — comment on line 47

**Dependencies:** None

### Phase 2: Un-exclude hand_position and add new data type mappings

**Goal:** Make `hand_position` and MoS columns selectable in cluster groups.

**Started:** 2026-04-24
**Completed:** 2026-04-24

- [x] Task 2.1 — Remove `hand_position_{x,y,z}` from `_EXCLUDE_FROM_AGGREGATION`
- [x] Task 2.2 — Add `hand_position`, individual MoS, and composite `mechanics_of_solids` to `DATA_TYPE_TO_COLUMNS`
- [x] Task 2.3 — Add new entries to GUI `_DATA_TYPES`

**Files Modified:**

- `code/src/analysis/touch_analytics/representation/feature_characterization/statistical.py`
  — remove `'hand_position_x'`, `'hand_position_y'`, `'hand_position_z'` and
    their comment from `_EXCLUDE_FROM_AGGREGATION`
- `code/src/analysis/touch_analytics/clustering_pipeline.py`
  — add to `DATA_TYPE_TO_COLUMNS`:
    ```python
    'hand_position':          ['hand_position_x', 'hand_position_y', 'hand_position_z'],
    'mos_strain':             ['mos_strain'],
    'mos_stress_kpa':         ['mos_stress_kpa'],
    'mos_strain_rate':        ['mos_strain_rate'],
    'mos_elastic_energy_mj':  ['mos_elastic_energy_mj'],
    'mos_impulse_mns':        ['mos_impulse_mns'],
    'mechanics_of_solids':    ['mos_strain', 'mos_stress_kpa', 'mos_strain_rate',
                               'mos_elastic_energy_mj', 'mos_impulse_mns'],
    ```
- `code/src/utils/gui/dag_launcher/cluster_group_dialog.py`
  — add `"hand_position"`, `"mos_strain"`, `"mos_stress_kpa"`, `"mos_strain_rate"`,
    `"mos_elastic_energy_mj"`, `"mos_impulse_mns"`, `"mechanics_of_solids"` to `_DATA_TYPES`

**Dependencies:** Phase 1 (pressure rename must land first to avoid mixed naming)

### Phase 3: Add comprehensive cluster group

**Goal:** Add `all_touch_features_mean` cluster group using all touch features.

**Started:** 2026-04-24
**Completed:** 2026-04-24

- [x] Task 3.1 — Add cluster group to YAML config

**Files Modified:**

- `configs/analyse_workflow_dag.yaml` — add after `kinematics_with_category`:
    ```yaml
    all_touch_features_mean:
      enabled: true
      features:
        contact_depth: [mean]
        contact_area: [mean]
        velocity_magnitude: [mean]
        acceleration_magnitude: [mean]
        pressure: [mean]
        hand_position: [mean]
        mechanics_of_solids: [mean]
        location: [mean]
      clustering_methods:
        binning:
          method: binning
          n_bins: 20
          bin_method: equal_width
          enabled: true
        type_stratified:
          method: type_stratified
          base_method: binning
          type_col: type_metadata
          n_bins: 20
          bin_method: equal_width
          enabled: true
    ```

**Feature vector (16 dimensions):**

| # | Column(s) | Source |
|---|-----------|--------|
| 1 | `contact_depth_mean` | raw |
| 2 | `contact_area_mean` | raw |
| 3 | `velocity_magnitude_mean` | hand_velocity transform |
| 4 | `acceleration_magnitude_mean` | hand_acceleration transform |
| 5 | `pressure_mean` | pressure transform |
| 6-8 | `hand_position_{x,y,z}_mean` | hand_position transform |
| 9-13 | `mos_strain_mean`, `mos_stress_kpa_mean`, `mos_strain_rate_mean`, `mos_elastic_energy_mj_mean`, `mos_impulse_mns_mean` | mechanics_of_solids transform |
| 14-16 | `mean_contact_x`, `mean_contact_y`, `mean_contact_z` | location shared cols |

Standard scaling normalizes all features before clustering.

**Dependencies:** Phase 2

---

## Testing Plan

### Manual Verification

- [ ] Re-run Stage 2b (`touch_feature_extraction`) with `force_processing: true`.
      Verify `mean/` feature CSVs contain `hand_position_x_mean` and
      `pressure_mean` (not `geo_pressure_mean`).
- [ ] Re-run Stage 2c (`touch_clustering`) for `all_touch_features_mean`.
      Check log output — no `_select_feature_columns` missing-column warnings.
- [ ] Inspect `pooled_touch_summary_clustered.csv` for all 16 feature columns
      plus `cluster_label`.
- [ ] Run `kinematics_mean` clustering — verify unchanged behavior (regression).
- [ ] Open GUI cluster group dialog — verify new data types appear in the list.

### Edge Cases

- [ ] Session with zero-area frames — `pressure_mean` may be NaN for touches
      with no valid contact area. Verify `dropna()` in the clustering pipeline
      handles this without crashing.
- [ ] MoS columns depend on `velocity_magnitude` — verify MoS values are
      physically plausible (strain < 1 for typical touch depths).

---

## Documentation Plan

- [ ] No README/CLAUDE.md updates needed (internal pipeline change)
- [ ] Knowledge base note `note-somatosensory-units-and-calculations.md`
      already documents velocity and pressure units — no update needed

---

## Rollback Plan

1. Revert the commit — all changes are in a single atomic set
2. Re-run Stage 2b + 2c with `force_processing: true` to regenerate old column
   names in feature CSVs
3. No database or external state affected

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Existing feature CSVs on disk have `geo_pressure_mean` columns — new code expects `pressure_mean` | High | Low | Re-run Stage 2b with `force_processing: true` after deploying changes |
| NaN proliferation from pressure/MoS in the 16-feature cluster group causes excessive row drops | Low | Medium | Monitor `dropna()` row count in clustering log; if significant, consider `dropna(subset=...)` on critical columns only |
| `hand_position` aggregation is meaningless without spatial reference frame | Low | Low | Standard scaling normalizes position values; spatial patterns are still captured as relative differences between touches |
