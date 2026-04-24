# Plan: Refactor Stage 2a Series Transforms — Sticker Selection, Hand Position & Drop Used Inputs

**Date:** 2026-04-24
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/refactor-series-transforms-sticker-selection`

---

## Overview

Refactor the Stage 2a series transform pipeline to select sticker positions based
on `contact_area_metadata`, split the monolithic `kinematics` transform into three
independently toggleable transforms (`hand_position`, `hand_velocity`,
`hand_acceleration`), and add a per-transform `drop_used_inputs` option that
removes consumed input columns from the output CSV.

## Problem Statement

The current `kinematics.py` hardcodes `sticker_blue_position_{x,y,z}` for all
velocity and acceleration computation. This is incorrect for "whole hand" touches
where the green and yellow stickers track the hand — the blue sticker tracks only
the fingertip. The pipeline also lacks:

- A `hand_position_{x,y,z}` output (the resolved position used for kinematics).
- Per-transform granularity — kinematics is a single on/off toggle.
- Any mechanism to drop consumed input columns for cleaner output CSVs.

## Goals

### In Scope

1. Conditional sticker selection: if `contact_area_metadata` contains `"hand"`
   → average green + yellow stickers; otherwise → blue sticker.
2. Three separate toggleable transforms replacing `kinematics`:
   `hand_position`, `hand_velocity`, `hand_acceleration`.
3. Per-transform `drop_used_inputs` option (defaults to `false`).
4. Output filtering: collect all drop lists and remove those columns before
   writing the augmented CSV.
5. Implement the missing `get_kinematics()` function (already imported by
   `series_pipeline.py`, `mechanics.py`, and `pressure.py`).
6. Fix `StatisticalExtractor` to use `get_kinematics` instead of directly
   calling `compute_velocity_magnitudes` (so it reads pre-computed,
   sticker-aware columns from augmented CSVs).

### Out of Scope

- GUI code changes — the existing `_is_feature_dict()` renderer in
  `task_detail_panel.py` handles the new config structure automatically.
- Changes to Stage 2b feature extraction logic (only the import path in
  `statistical.py` changes).
- Interpolation changes — `interpolation.py` already handles all sticker
  columns.
- Backward-compatibility shim for old `kinematics:` config key (can be
  added later if needed; all current configs are under our control).

## Success Criteria

- [ ] `hand_position_{x,y,z}` columns in augmented CSV match avg(green, yellow)
      for "whole hand" touches and blue sticker for other contact types.
- [ ] `velocity_magnitude` and `acceleration_magnitude` are derived from
      `hand_position_*`, not hardcoded blue sticker.
- [ ] Each of the five transforms (`hand_position`, `hand_velocity`,
      `hand_acceleration`, `pressure`, `mechanics_of_solids`) can be
      independently enabled/disabled.
- [ ] `drop_used_inputs: true` on a transform removes its input columns from
      the output CSV.
- [ ] Default config (`drop_used_inputs: false`) produces identical column set
      as before, plus the new `hand_position_{x,y,z}` columns.
- [ ] `StatisticalExtractor` reads pre-computed velocity/acceleration from
      augmented CSVs via `get_kinematics`.
- [ ] GUI renders five checkboxes in the transforms section, each with a "..."
      param button, without any code changes.

---

## Technical Design

### Approach

Introduce a position resolution layer in `kinematics.py` that reads
`contact_area_metadata` per touch group and selects the appropriate sticker
columns. The resolved position is always computed internally (it's needed by
velocity, acceleration, and MoS), but only output when `hand_position.enabled`
is true. Each transform advertises its "used input" columns via module-level
constants; the pipeline collects drop lists at the end and filters the DataFrame
before writing.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Per-transform sticker selection in kinematics.py | Clean, single responsibility | — | **Chosen** |
| Sticker selection in series_pipeline.py | Keeps kinematics.py pure math | Duplicates logic for mechanics.py / extractors | Rejected |
| Global `sticker_mode` config instead of per-group | Simpler config | Wrong — contact type varies within a session | Rejected |
| Drop columns via a separate post-processing step | Decoupled | Extra I/O; can't leverage in-memory df | Rejected |

### Architecture Changes

**New functions** in `kinematics.py`:
- `resolve_hand_position(group) → DataFrame` — sticker selection + averaging
- `get_kinematics(group, fps) → (velocity, acceleration)` — fast-path
  (pre-computed columns) + slow-path (compute from scratch)

**Modified functions:**
- `compute_velocity_magnitudes(group, fps, position_cols)` — now accepts
  configurable position columns instead of hardcoded blue sticker.

**New constants** (for drop mechanism):
- `kinematics.STICKER_INPUT_COLUMNS` — 9 sticker position columns
- `kinematics.HAND_POSITION_COLUMNS` — 3 hand position columns
- `pressure.PRESSURE_INPUT_COLUMNS` — `contact_depth`, `contact_area`

**Dependency chain** (internal computation order):
```
resolve_hand_position  ← always
        ↓
compute_velocity       ← if vel OR accel OR mos enabled
        ↓
compute_acceleration   ← if accel enabled
```

`enabled` flags control output inclusion. `drop_used_inputs` flags control
input column removal. Both are applied at the final CSV-write step.

### Per-Transform Drop Mapping

| Transform | Columns dropped when `drop_used_inputs: true` |
|---|---|
| `hand_position` | 9 sticker columns: `sticker_{blue,green,yellow}_position_{x,y,z}` |
| `hand_velocity` | 3 columns: `hand_position_{x,y,z}` |
| `hand_acceleration` | 1 column: `velocity_magnitude` |
| `pressure` | 2 columns: `contact_depth`, `contact_area` |
| `mechanics_of_solids` | nothing unique (overlaps with kinematics + pressure) |

---

## Implementation Plan

### Phase 1: Position Resolution and `get_kinematics`

**Goal:** Add sticker selection logic and implement the missing `get_kinematics`
function.

- [x] Add `resolve_hand_position(group)` — reads `contact_area_metadata`,
      selects blue or avg(green, yellow), returns `hand_position_{x,y,z}`
      DataFrame. Fail-fast on missing columns.
- [x] Add `STICKER_INPUT_COLUMNS` and `HAND_POSITION_COLUMNS` constants.
- [x] Modify `compute_velocity_magnitudes` to accept `position_cols` parameter
      (default: hand position columns).
- [x] Implement `get_kinematics(group, fps)` — returns pre-computed columns if
      present, otherwise resolves position and computes from scratch.

**Files Modified:**
- `code/src/analysis/touch_analytics/representation/series_level/kinematics.py`
  — all changes in this phase

**Dependencies:** None

### Phase 2: Pipeline Integration

**Goal:** Rewire `series_pipeline.py` to use three separate transforms, compute
hand position, and implement the drop-used-inputs mechanism.

- [x] Replace single `kinematics` config parsing with `hand_position`,
      `hand_velocity`, `hand_acceleration` config blocks (each with `enabled`,
      `fps` where applicable, `drop_used_inputs`).
- [x] In the per-group loop: always call `resolve_hand_position`, augment
      group, then conditionally compute velocity and acceleration.
- [x] Pass augmented group (with `hand_position_*`) to `compute_mos_series`
      so it gets correct sticker-aware velocity via `get_kinematics`.
- [x] Merge output columns into session DataFrame based on enabled flags.
- [x] Collect drop lists from all transforms with `drop_used_inputs: true`
      and remove those columns before `df.to_csv()`.
- [x] Update `_transform_session` signature and `run_series_transforms`
      banner print.
- [x] Remove early return when kinematics is disabled — the pipeline should
      still run if only pressure or MoS is enabled (position is always
      resolved internally).

**Files Modified:**
- `code/src/analysis/touch_analytics/series_pipeline.py` — major restructure

**Dependencies:** Phase 1

### Phase 3: Config and Constants

**Goal:** Update YAML config and add drop-list constants to pressure module.

- [x] Replace `kinematics:` block in `analyse_workflow_dag.yaml` with
      `hand_position:`, `hand_velocity:`, `hand_acceleration:` blocks.
- [x] Add `drop_used_inputs: false` to all five transform blocks.
- [x] Add `PRESSURE_INPUT_COLUMNS` constant to `pressure.py`.

**Files Modified:**
- `configs/analyse_workflow_dag.yaml` — transform section restructure
- `code/src/analysis/touch_analytics/representation/series_level/pressure.py`
  — add constant only

**Dependencies:** None (can be done in parallel with Phase 1)

### Phase 4: Feature Extractor Alignment

**Goal:** Fix `StatisticalExtractor` to use `get_kinematics` so it reads
pre-computed, sticker-aware columns from augmented CSVs.

- [x] Replace `compute_velocity_magnitudes` / `compute_acceleration_magnitudes`
      imports with `get_kinematics` import.
- [x] Replace direct computation calls with `get_kinematics(group, fps=fps)`.

**Files Modified:**
- `code/src/analysis/touch_analytics/representation/feature_characterization/statistical.py`
  — import and two lines of logic

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification

- [ ] Run Stage 2a on a session with "whole hand" touches — open output CSV,
      verify `hand_position_*` ≈ avg(green, yellow sticker) for each frame.
- [ ] Run Stage 2a on a session with "one finger tip" or "two finger tips"
      touches — verify `hand_position_*` = blue sticker values.
- [ ] Enable `drop_used_inputs: true` for `hand_position` — verify 9 sticker
      columns absent from output CSV.
- [ ] Enable `drop_used_inputs: true` for `hand_velocity` — verify
      `hand_position_*` absent from output CSV.
- [ ] Enable `drop_used_inputs: true` for `hand_acceleration` — verify
      `velocity_magnitude` absent from output CSV.
- [ ] Enable `drop_used_inputs: true` for `pressure` — verify
      `contact_depth` / `contact_area` absent from output CSV.
- [ ] Run with all defaults (`drop_used_inputs: false`) — verify all original
      columns preserved plus new `hand_position_{x,y,z}`.
- [ ] Disable `hand_velocity` but enable `hand_acceleration` — verify
      acceleration is still computed correctly (velocity computed internally).
- [ ] Run full pipeline (Stage 2a → 2b) — verify `StatisticalExtractor`
      produces consistent velocity/acceleration features.
- [ ] Open GUI launcher, select `touch_series_transforms` — verify 5
      checkboxes render correctly with "..." param buttons.

### Edge Cases

- [ ] Session where `contact_area_metadata` is missing — verify fail-fast
      `KeyError` with clear message.
- [ ] Touch group where all sticker values are NaN — verify `hand_position_*`
      is NaN, velocity/acceleration are zero (from `fillna(0)` in diff).
- [ ] Session mixing "whole hand" and "one finger tip" touches — verify
      per-group sticker selection (not per-session).
- [ ] Old config with `kinematics:` key instead of new keys — verify pipeline
      uses defaults (all enabled, fps=30.0).

---

## Documentation Plan

- [ ] Update YAML config comments in `analyse_workflow_dag.yaml` to document
      new transform structure and `drop_used_inputs` semantics.
- [ ] No README/CLAUDE.md changes needed — this is internal pipeline behavior.

---

## Rollback Plan

All changes are on a feature branch. Rollback is straightforward:

1. Revert the branch — original `kinematics` config and hardcoded blue sticker
   are restored.
2. No data migration — output CSVs are regenerated on each run.
3. No breaking changes to downstream stages — `velocity_magnitude` and
   `acceleration_magnitude` column names are unchanged.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `contact_area_metadata` column missing in some sessions | Low | High | Fail-fast with clear `KeyError` message; easy to diagnose |
| Green/yellow sticker data sparse or all-NaN for hand touches | Low | Med | Produces NaN position → zero velocity; downstream handles gracefully |
| Drop mechanism accidentally removes columns needed by MoS | Low | Med | MoS has no unique drop list; its inputs overlap with position + pressure |
| Old YAML configs break | Low | Low | Defaults kick in (all enabled, fps=30.0); can add deprecation warning later |

---

## References

- Existing sticker selection precedent: `code/scripts/_5_postprocessing/set_xyz_reference_from_gestures.py:119`
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md` (velocity in mm/s, spatial values in mm)
- Active plan context: `docs/development/plans/active/split-feature-extraction-into-2a-2b-flows.md`
