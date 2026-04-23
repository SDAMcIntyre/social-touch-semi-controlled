# Plan: Geometric Pressure Proxy

**Date:** 2026-04-22
**Completed:** 2026-04-23 19:15
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/geometric-pressure-proxy`

---

## Overview

Add two parameter-free tactile feature extractors pairing **geometric pressure** (`contact_depth / contact_area`) with **contact velocity** (`|d(sticker_position)/dt|`). Two separate feature entries are registered — one for mean aggregates, one for max — so they can be toggled and combined independently in clustering.

## Problem Statement

The existing feature set captures depth and area independently (via statistical aggregations) and derives elastic stress via a tissue model (MOS extractor). Neither captures the **ratio** of depth to area — the geometric concentration of indentation — which is directly relevant to mechanoreceptor activation patterns (SA-I afferents respond to localised pressure gradients). Pairing this ratio with contact velocity allows distinguishing touches that are simultaneously concentrated and fast from those that are slow but deep. Both metrics are parameter-free, making them robust for cross-session and cross-subject comparisons where tissue properties are unknown.

## Goals

### In Scope
1. `PressureExtractor` class implementing `FeatureExtractor`, parameterised by aggregation mode (`mean` or `max`)
2. Two registry entries: `pressure_velocity_mean` and `pressure_velocity_max`
3. Each entry outputs two columns: geometric pressure + contact velocity, under the chosen aggregation
4. Two YAML config entries under `touch_feature_extraction.options.features`

### Out of Scope
- Model-based pressure refinements (already in MOS extractor)
- New clustering profiles or feature combinations (user adds via GUI)
- Time-series output (per-frame columns)
- AUC, slope, or at-peak-depth aggregates

## Success Criteria

- [ ] `pressure_velocity_mean` and `pressure_velocity_max` appear as toggleable features in the DAG launcher GUI
- [ ] `pressure_velocity_mean` extraction produces a CSV with columns `geo_pressure_mean`, `geo_velocity_mean`
- [ ] `pressure_velocity_max` extraction produces a CSV with columns `geo_pressure_max`, `geo_velocity_max`
- [ ] Clustering and RF mapping pipelines accept both features in feature combinations without modification
- [ ] Zero-area frames produce `NaN` in pressure columns, not errors or infinities

---

## Technical Design

### Approach

`PressureExtractor` reads an `aggregation` config param (`'mean'` or `'max'`) and computes:
- **Geometric pressure:** `contact_depth / contact_area` per frame (safe division), then aggregated
- **Contact velocity:** 3D sticker position magnitude via `compute_velocity_magnitudes()` (already in `kinematics.py`), then aggregated

Both registry keys map to the same class; the feature key name drives the config injected by the orchestrator.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Geometric ratio `depth/area` | Parameter-free, empirical, no tissue assumptions | Units (mm⁻¹) are non-standard | **Chosen** |
| Extend MOS extractor with a force column (`stress * area`) | Physically grounded (Newtons) | Depends on tissue parameters, duplicates MOS scope | Rejected — separate concern |
| Hertzian contact model | Well-established indentation mechanics | Assumes elastic half-space + spherical indenter — poor fit for finger contact | Rejected — model assumptions too strong |
| Single registry entry outputting all 4 columns | Simpler | Cannot toggle mean vs. max independently in GUI/clustering | Rejected — flexibility preferred |

### Architecture Changes

**Modified file:**
- `code/src/analysis/touch_analytics/feature_extraction/pressure_extractor.py` — rewrite to `aggregation`-parameterised logic, drop slope/AUC/at-peak-depth

**Modified files:**
- `code/src/analysis/touch_analytics/feature_extraction/__init__.py` — replace `geometric_pressure` with two entries: `pressure_velocity_mean`, `pressure_velocity_max`
- `configs/analyse_workflow_dag.yaml` — replace `geometric_pressure` entry with two entries

### Output Columns

| Feature key | Column | Formula | Units |
|---|---|---|---|
| `pressure_velocity_mean` | `geo_pressure_mean` | `nanmean(depth / area)` | mm⁻¹ |
| | `geo_velocity_mean` | `nanmean(sticker velocity)` | mm/s |
| `pressure_velocity_max` | `geo_pressure_max` | `nanmax(depth / area)` | mm⁻¹ |
| | `geo_velocity_max` | `nanmax(sticker velocity)` | mm/s |

Sticker velocity is computed from `sticker_blue_position_x/y/z` via the existing `compute_velocity_magnitudes()` helper (returns mm/s, 3D magnitude).

### Division Safety

Frames where `contact_area <= 0` yield `NaN` in the pressure ratio. Aggregations use `np.nanmean` / `np.nanmax`. If the entire touch has zero area, pressure columns return `NaN`; velocity columns are unaffected.

---

## Implementation Plan

### Phase 1: Extractor (single phase)
**Goal:** Working `pressure_velocity_mean` and `pressure_velocity_max` features, end-to-end from YAML toggle to output CSV.
**Started:** 2026-04-22

**Tasks:**
- [ ] Task 1.1 — Rewrite `pressure_extractor.py`: `aggregation` param, pressure + velocity, drop slope/AUC/at-peak-depth
- [ ] Task 1.2 — Update `__init__.py`: replace `geometric_pressure` with `pressure_velocity_mean` and `pressure_velocity_max` entries
- [ ] Task 1.3 — Update DAG YAML: replace `geometric_pressure` with two entries (`enabled: false`)

**Files Modified:**
- `code/src/analysis/touch_analytics/feature_extraction/pressure_extractor.py`
- `code/src/analysis/touch_analytics/feature_extraction/__init__.py`
- `configs/analyse_workflow_dag.yaml`

**Dependencies:** `compute_velocity_magnitudes()` in `kinematics.py` (already exists, no changes needed)

---

## Testing Plan

### Manual Verification
- [ ] Enable `pressure_velocity_mean` in DAG YAML, run extraction on one session, verify columns `geo_pressure_mean` and `geo_velocity_mean` with sensible values
- [ ] Enable `pressure_velocity_max`, verify columns `geo_pressure_max` and `geo_velocity_max`
- [ ] Confirm zero-area frames produce `NaN` in pressure columns (not `inf` or error)
- [ ] Add each feature to a feature combination, run clustering, verify it merges correctly
- [ ] Open DAG launcher GUI — confirm both checkboxes appear in the features grid
- [ ] Toggle each checkbox, save config, verify YAML round-trips correctly

### Edge Cases
- [ ] Touch event where all frames have `contact_area == 0` — pressure columns `NaN`, velocity columns valid
- [ ] Single-frame touch — all aggregations return a scalar (no crash)
- [ ] Very small area values — verify no overflow in division

---

## Documentation Plan

- [ ] Column definitions documented in the extractor's docstring
- [ ] Knowledge base note `note-somatosensory-units-and-calculations.md` — add geometric pressure and contact velocity to the metrics table

---

## Rollback Plan

1. Remove `pressure_velocity_mean` and `pressure_velocity_max` entries from DAG YAML
2. Remove or revert `pressure_extractor.py`
3. Remove registry entries from `__init__.py`
4. No database/state changes — extraction outputs are new files that can be deleted

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Division by zero in area | High (expected) | Low | `np.where` guard + `nanmax`/`nanmean` |
| mm⁻¹ units confuse downstream users | Low | Low | Clear column naming (`geo_pressure_*`) + docstring |
| Very small area values produce extreme ratios | Medium | Low | Inherent to the metric; downstream clustering handles outliers via binning |
| `sticker_blue_position_x/y/z` absent in some sessions | Low | Medium | `compute_velocity_magnitudes()` will raise `KeyError` — same risk as existing `StatisticalExtractor`; no special handling needed |

---

## References

- Existing extractors: `mos_extractor.py`, `temporal_extractor.py`, `statistical_extractor.py`
- Velocity helper: `feature_extraction/kinematics.py` — `compute_velocity_magnitudes()`
- Knowledge base: `note-somatosensory-units-and-calculations.md`
- Feature extraction registry: `feature_extraction/__init__.py`
