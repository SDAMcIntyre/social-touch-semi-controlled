# Plan: Stage 1 — Touch Data Preparation (30 Hz → 1 kHz Interpolation)

**Created:** 2026-04-23 18:00
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/align-clustering-pipeline-with-guidelines`

---

## Overview

**What:** Introduce a formal Stage 1 (Data Preparation) phase into the analysis pipeline
that fills NaN gaps in touch columns with proper interpolation and updates the entire
pipeline to operate at 1 kHz.

**Why:** The pipeline guidelines (`docs/design/timeseries_clustering_pipeline_guidelines.md`)
define Stage 1 as responsible for temporal alignment and resampling. The aggregated
session CSVs are already at 1 kHz (the merging script forward-places 30 Hz Kinect
values at every ~33rd row), but touch columns are NaN between samples. All downstream
code uses `fps=30`, computing kinematics only on sparse non-NaN rows — velocity is
zero for 32/33 rows with a spike at each frame boundary.

**How:** A new `preparation_pipeline.py` orchestrates per-session NaN-gap interpolation
(cubic/linear) on touch columns, producing prepared CSVs. Downstream pipelines consume
these, and all `fps` defaults are updated from 30 to 1000.

## Problem Statement

The merging script (`code/scripts/_4_merging/merge_neural_and_kinect_data.py:144-153`)
upsamples Kinect data from 30 Hz to ~1 kHz by placing original values at every ~33rd
row with `scaling_nofilling=True` (default). This leaves touch columns (`contact_depth`,
`contact_area`, `sticker_blue_position_*`) as NaN between the original sample points.
Neural data (`Nerve_spike`) is at native 1 kHz.

The analysis pipeline then operates on these sparse CSVs with `fps=30`:
- `kinematics.py` computes `.diff().fillna(0) * 30` — producing mostly-zero velocity
  with spikes at 30 Hz frame boundaries
- `temporal.py` counts frames and integrates over frame indices — physically meaningless
  at 1 kHz with NaN gaps
- Feature extractors aggregate over per-frame values that are ~97% NaN

This produces numerically valid but physically inaccurate derived quantities.

## Goals

### In Scope

1. New `preparation/interpolation.py` module — NaN-gap filling with cubic/linear
   interpolation per touch group
2. New `preparation_pipeline.py` — Stage 1 orchestrator as a standalone DAG task
3. New `touch_data_preparation` DAG task inserted before `touch_series_transforms`
4. Downstream pipelines consume prepared CSVs (augmented > prepared > raw fallback chain)
5. All `fps=30.0` defaults updated to `fps=1000.0` (6 locations)
6. `temporal.py` updated with physical-time features and scaled slope window

### Out of Scope

- Denoising, smoothing, or outlier handling (future Stage 1 sub-tasks)
- Changes to the merging script (`merge_neural_and_kinect_data.py`)
- Changes to the RF mapping pipelines' existing forward-fill logic
- Downsampling or any rate other than 1 kHz
- Loading original 1 kHz neural data from external sources (already at 1 kHz in CSV)

## Success Criteria

- [ ] `touch_data_preparation` DAG task runs and produces `*_prepared.csv` files
      in `4_analysed/prepared/`
- [ ] Prepared CSVs have same row count as input but no NaN in touch columns
      within touch groups
- [ ] Continuous columns are smoothly interpolated (no step artifacts visible in plots)
- [ ] `Nerve_spike` column is untouched (already at native 1 kHz)
- [ ] Velocity magnitudes are in correct physical range (mm/s) — smooth profile
      instead of spike-at-boundary
- [ ] Scalar features (means, maxes) are statistically consistent with previous
      30 Hz pipeline results
- [ ] Idempotency: re-running with `force_processing: false` skips prepared sessions
- [ ] Pipeline runs end-to-end: preparation → series transforms → extraction → clustering

---

## Technical Design

### Approach

**"Fill NaN gaps, then update the clock."**

The aggregated CSVs are already at 1 kHz — no rows need to be added. Touch columns
have real values at every ~33rd row and NaN between them. The preparation stage fills
those gaps with per-touch-group interpolation (cubic when ≥4 sample points, linear
otherwise), then all downstream `fps` defaults are updated from 30 to 1000 so that
kinematics and physics calculations produce correct physical units.

Column handling strategy:

| Column class | Examples | Treatment |
|---|---|---|
| Continuous (30 Hz, NaN gaps) | `contact_depth`, `contact_area`, `sticker_blue_position_*`, `contact_location_*` | Cubic/linear interpolation |
| Already 1 kHz | `Nerve_spike`, `Nerve_freq`, `Nerve_TTL`, `time` | Untouched |
| Binary (forward-filled) | `led_on`, `contact_detected` | Already filled — skip |
| Identity (constant per group) | `block_order_id`, `trial_id`, `single_touch_id`, `type_metadata` | Skip |
| Unknown numeric | Any new column | Interpolate |
| Unknown non-numeric | Any new column | Forward-fill |

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Per-group `pd.interpolate(method='cubic')` on NaN gaps | Simple, uses existing DataFrame index; no row changes | Cubic may overshoot near boundaries | **Chosen** — clamp negative depth/area to 0 |
| `scipy.interpolate.interp1d` on identified sample points | More control over interpolation | Must manually identify sample indices; more code | Rejected — pandas handles NaN detection natively |
| Forward-fill only (step function) | Simplest, no overshoot | Produces staircase signal; velocity still spiky | Rejected — defeats the purpose |
| Modify merging script to interpolate at source | Single point of change | Changes upstream pipeline; affects all consumers | Rejected — out of scope |

### Architecture Changes

New files:
```
code/src/analysis/touch_analytics/
  preparation/
    interpolation.py       — core interpolation logic (~80 lines)
  preparation_pipeline.py  — Stage 1 orchestrator (~90 lines)
```

Modified files:

| File | Change |
|------|--------|
| `preparation/__init__.py` | Export `interpolate_touch_columns` |
| `__init__.py` | Export `run_preparation` in `__all__` |
| `pipeline_shared.py` | Add `DEFAULT_PREPARATION_OPTIONS` dict |
| `series_pipeline.py` | Add `prepared_dir` param; update fps default 30→1000 |
| `extraction_pipeline.py` | Add `prepared_dir` param (fallback chain: augmented > prepared > raw) |
| `representation/series_level/kinematics.py` | fps default 30→1000 (3 functions) |
| `representation/feature_characterization/mos.py` | `_DEFAULTS['fps']` 30→1000 |
| `representation/feature_characterization/pressure.py` | fps default 30→1000 |
| `representation/feature_characterization/statistical.py` | fps default 30→1000 |
| `representation/feature_characterization/temporal.py` | Add fps-aware physical-time features; scale slope window |
| `code/scripts/analysis_workflow.py` | New flow function + task registration + option forwarding |
| `configs/analyse_workflow_dag.yaml` | New task entry; updated fps and dependency |
| `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md` | Add 1 kHz preparation note |

Data flow after implementation:
```
Raw CSV (1 kHz rows, touch columns NaN between 30 Hz samples)
  → [preparation_pipeline.py] touch_data_preparation
      → load, ensure block_id, interpolate NaN gaps per touch group
      → save to 4_analysed/prepared/<session_id>_prepared.csv
  → [series_pipeline.py] touch_series_transforms (reads prepared CSV, fps=1000)
      → velocity, acceleration, pressure, MOS — now smooth at 1 kHz
  → [extraction_pipeline.py] touch_feature_extraction
      → scalar features per touch (from continuous 1 kHz data)
  → Clustering → Comparing
```

### Knowledge Base Relevance

- **`note-somatosensory-units-and-calculations.md`** — Directly relevant. Confirms
  coordinates in mm, velocity in mm/s via `displacement * fps`, capture rate 30 Hz.
  Must be updated to document the 1 kHz preparation step and its effect on derived units.
- All other knowledge base notes (forearm ICP, CuPy import, Qt signals, Kinect parallax,
  3D projection, Open3D layout) — not relevant to this feature.

---

## Implementation Plan

### Phase 1: Core Interpolation Logic
**Goal:** Implement per-touch-group NaN-gap interpolation for touch columns.
**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Task 1.1 — Create `preparation/interpolation.py` with `interpolate_touch_columns(df, method)` session-level entry point and `_interpolate_group(group, cols, method)` per-touch worker
- [x] Task 1.2 — Define column classification constants: `CONTINUOUS_COLUMNS`, `ALREADY_1KHZ`, `BINARY_FFILL`, `IDENTITY`
- [x] Task 1.3 — Implement per-group interpolation: `group[col].interpolate(method='cubic')` if ≥4 non-NaN values, else `method='linear'`; `limit_direction='both'` for boundary NaN
- [x] Task 1.4 — Handle edge cases: 1 non-NaN sample → forward-fill; 0 non-NaN → leave NaN; missing optional columns → skip; `single_touch_id == 0` → interpolate for continuity
- [x] Task 1.5 — Clamp `contact_depth` and `contact_area` to ≥ 0 after cubic interpolation (overshoot protection)
- [x] Task 1.6 — Export from `preparation/__init__.py`

**Files Modified:**
- `code/src/analysis/touch_analytics/preparation/interpolation.py` — **NEW** (~80 lines)
- `code/src/analysis/touch_analytics/preparation/__init__.py` — add export

**Dependencies:** None

### Phase 2: Preparation Pipeline Orchestrator
**Goal:** Create the Stage 1 pipeline module following existing patterns.
**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Task 2.1 — Create `preparation_pipeline.py` with `run_preparation(input_items, config, output_dir, force)` and `_prepare_session(input_file, output_dir, method, force)`
- [x] Task 2.2 — Per-session flow: idempotency check → `load_session_csv` → `ensure_block_id_column` → `interpolate_touch_columns` → save CSV
- [x] Task 2.3 — Add `DEFAULT_PREPARATION_OPTIONS` to `pipeline_shared.py`
- [x] Task 2.4 — Export `run_preparation` from `touch_analytics/__init__.py`

**Files Modified:**
- `code/src/analysis/touch_analytics/preparation_pipeline.py` — **NEW** (~90 lines)
- `code/src/analysis/touch_analytics/pipeline_shared.py` — add defaults dict
- `code/src/analysis/touch_analytics/__init__.py` — add import + `__all__` entry

**Dependencies:** Phase 1

### Phase 3: DAG Integration
**Goal:** Wire the new task into the workflow orchestrator and YAML config.
**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Task 3.1 — Add `touch_data_preparation` task to `configs/analyse_workflow_dag.yaml` before `touch_series_transforms`
- [x] Task 3.2 — Update `touch_series_transforms`: `depends_on: [touch_data_preparation]`, `fps: 1000.0`
- [x] Task 3.3 — Add `touch_data_preparation_flow` to `analysis_workflow.py` (Prefect flow function)
- [x] Task 3.4 — Insert into `available_tasks` list at index 2 (after `map_receptive_fields_simple`)
- [x] Task 3.5 — Add option forwarding in `run_batch_analysis` for `interpolation_method`

**Files Modified:**
- `configs/analyse_workflow_dag.yaml` — new task, updated deps + fps
- `code/scripts/analysis_workflow.py` — new flow, import, task list, forwarding

**Dependencies:** Phase 2

### Phase 4: Downstream Consumption Rewiring
**Goal:** Make series_pipeline and extraction_pipeline consume prepared CSVs.
**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Task 4.1 — `series_pipeline.py`: add `prepared_dir: Optional[Path]` param; check for `<session_id>_prepared.csv` before loading raw CSV
- [x] Task 4.2 — `extraction_pipeline.py`: add `prepared_dir: Optional[Path]` param; extend fallback chain to augmented > prepared > raw
- [x] Task 4.3 — `analysis_workflow.py`: derive `prepared_dir` and pass to both flow functions

**Files Modified:**
- `code/src/analysis/touch_analytics/series_pipeline.py` — add `prepared_dir` param + lookup
- `code/src/analysis/touch_analytics/extraction_pipeline.py` — add `prepared_dir` param + fallback
- `code/scripts/analysis_workflow.py` — pass `prepared_dir` to both flows

**Dependencies:** Phase 3

### Phase 5: fps Propagation (30 → 1000)
**Goal:** Update all hardcoded fps=30.0 defaults so kinematics and physics are correct.
**Started:** 2026-04-23
**Completed:** 2026-04-23

- [x] Task 5.1 — `kinematics.py`: update 3 default params `fps: float = 30.0` → `fps: float = 1000.0` (lines 7, 15, 22)
- [x] Task 5.2 — `mos.py`: update `_DEFAULTS['fps']` from 30.0 → 1000.0 (line 25)
- [x] Task 5.3 — `pressure.py`: update `config.get('fps', 30.0)` → `config.get('fps', 1000.0)` (line 52)
- [x] Task 5.4 — `statistical.py`: update `config.get('fps', 30.0)` → `config.get('fps', 1000.0)` (line 18)
- [x] Task 5.5 — `series_pipeline.py`: update `kinematics_cfg.get('fps', 30.0)` → `1000.0` (line 59)
- [x] Task 5.6 — `temporal.py`: add `fps` from config (default 1000.0); add `duration_s`, `time_to_peak_depth_s`, `time_to_peak_area_s`; update AUC to `np.trapz(depth, t / fps)` for mm·s units; scale `_SLOPE_FRAMES` to preserve ~167 ms window (use `int(slope_frames * fps / 30)` or configurable `slope_ms`)
- [x] Task 5.7 — Update `note-somatosensory-units-and-calculations.md`: add 1 kHz preparation note to Capture FPS row and Section 2 (Velocity)

**Files Modified:**
- `code/src/analysis/touch_analytics/representation/series_level/kinematics.py` — 3 fps defaults
- `code/src/analysis/touch_analytics/representation/feature_characterization/mos.py` — 1 default
- `code/src/analysis/touch_analytics/representation/feature_characterization/pressure.py` — 1 default
- `code/src/analysis/touch_analytics/representation/feature_characterization/statistical.py` — 1 default
- `code/src/analysis/touch_analytics/representation/feature_characterization/temporal.py` — fps-aware features + slope scaling
- `code/src/analysis/touch_analytics/series_pipeline.py` — 1 fallback default
- `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md` — documentation update

**Dependencies:** Phase 3

---

## Testing Plan

### Unit Tests

- [ ] `test_interpolate_group_cubic` — 5-sample touch with ~33-row NaN gaps produces smooth interpolation, no NaN remaining
- [ ] `test_interpolate_group_linear_fallback` — 3-sample touch falls back to linear interpolation
- [ ] `test_interpolate_single_sample` — 1-sample touch forward-fills to constant
- [ ] `test_interpolate_preserves_nerve_spike` — `Nerve_spike` column unchanged after interpolation
- [ ] `test_interpolate_preserves_identity_columns` — `block_order_id`, `trial_id`, `single_touch_id` unchanged
- [ ] `test_clamp_negative_depth` — cubic overshoot producing negative `contact_depth` is clamped to 0
- [ ] `test_velocity_physical_units` — velocity from interpolated data at fps=1000 matches expected mm/s range

### Integration Tests

- [ ] Run `touch_data_preparation` on one real session — verify output row count equals input, no NaN in touch columns within groups
- [ ] Run full pipeline (preparation → series → extraction → clustering) on one session — no errors
- [ ] Series pipeline loads prepared CSV when available, raw CSV when not

### Manual Verification

- [ ] Plot `contact_depth` for a single touch before/after — staircase → smooth curve
- [ ] Compare velocity profiles before/after — spike-at-boundary → continuous signal
- [ ] Compare scalar feature values (means, maxes) across old/new pipeline — similar magnitude

### Edge Cases

- [ ] Session with a 2-sample touch (minimum for linear interpolation)
- [ ] Session with only `single_touch_id == 0` frames (no actual touches)
- [ ] Session without `Nerve_spike` column (optional column)
- [ ] Session without `contact_location_*` columns (optional columns)

---

## Documentation Plan

- [ ] Update `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md` with 1 kHz preparation note
- [ ] Update `docs/design/timeseries_clustering_pipeline_guidelines.md` to note that Stage 1 is now implemented
- [ ] Add inline docstrings to new functions in `interpolation.py` and `preparation_pipeline.py`

---

## Rollback Plan

1. Set `touch_data_preparation.enabled: false` in `configs/analyse_workflow_dag.yaml`
2. Revert `touch_series_transforms.depends_on` to `[]`
3. Revert `touch_series_transforms.options.transforms.kinematics.fps` to `30.0`
4. Revert all fps defaults back to 30.0 in: `kinematics.py`, `mos.py`, `pressure.py`, `statistical.py`, `temporal.py`, `series_pipeline.py`
5. Remove `prepared_dir` parameter from `series_pipeline.py` and `extraction_pipeline.py`
6. Delete `4_analysed/prepared/` output directory

All downstream code continues to work at 30 Hz with the original defaults restored.
New files (`interpolation.py`, `preparation_pipeline.py`) can remain — they are not
called when the task is disabled.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Cubic spline overshoot (negative depth/area) | Low | Low | Clamp `contact_depth` and `contact_area` to ≥ 0 after interpolation |
| `temporal.py` slope window changes physical meaning (5 frames = 5 ms at 1 kHz vs 167 ms at 30 Hz) | Med | Med | Scale `_SLOPE_FRAMES` by fps ratio or make configurable via `slope_ms` |
| Undiscovered `fps=30` assumption elsewhere | Low | High | Grep audit found 6 locations — all listed in Phase 5. Re-audit before merge. |
| Disk space increase | — | — | Not a risk — row count unchanged, NaN→float has negligible size impact |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Core interpolation | 1-2 hours | None |
| Phase 2: Pipeline orchestrator | 1 hour | Phase 1 |
| Phase 3: DAG integration | 30 min | Phase 2 |
| Phase 4: Downstream rewiring | 1 hour | Phase 3 |
| Phase 5: fps propagation | 1-2 hours | Phase 3 |
| **Total** | **4.5-7.5 hours** | |

---

## References

- Pipeline guidelines: `docs/design/timeseries_clustering_pipeline_guidelines.md`
- Merging script: `code/scripts/_4_merging/merge_neural_and_kinect_data.py` (lines 144-153)
- Somatosensory units: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
- Related plans: `docs/development/plans/active/split-feature-extraction-into-2a-2b-flows.md`
