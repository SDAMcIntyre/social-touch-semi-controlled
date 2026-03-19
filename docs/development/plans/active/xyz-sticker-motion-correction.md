# Plan: XYZ Sticker Motion Correction

**Created:** 2026-03-19 —
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/xyz-sticker-motion-correction`

---

## Overview

**What:** A new preprocessing task `correct_xyz_stickers_motion` that filters and corrects raw 3D sticker trajectories after XYZ extraction and before downstream consumption.

**Why:** Kinect depth data contains sensor noise and occasional spikes that don't represent real human motion. Downstream kinematics (velocity, acceleration) amplify this noise via differentiation.

**How:** Strategy pattern with two filter implementations (Butterworth low-pass, Savitzky-Golay), combined with outlier detection (hard physical thresholds + statistical). Includes a "compare" mode for visual filter evaluation and a "correct" mode for production use.

## Problem Statement

- Raw XYZ sticker positions at 30fps in mm contain high-frequency depth sensor noise from the Azure Kinect
- Occasional depth lookup failures produce spikes/jumps in trajectories (physically impossible velocities/accelerations)
- No filtering step exists between extraction (`generate_xyz_stickers`) and consumers (`generate_3d_hand_in_motion`, `find_single_touches`, analysis pipeline)
- Downstream `.diff()`-based velocity/acceleration computations amplify noise through differentiation

## Goals

### In Scope
1. Outlier detection with independently toggleable hard-threshold and statistical methods
2. Two filter implementations (Butterworth, Savitzky-Golay) behind a common ABC interface
3. "Compare" mode producing overlay plots of both filters for visual evaluation
4. "Correct" mode applying the chosen filter and saving corrected CSV
5. Diagnostic plots (before/after) in both modes
6. DAG integration as a new task between `generate_xyz_stickers` and Stage 3/4

### Out of Scope
- Modifying the XYZ extraction itself (that's the extractor's job)
- Multi-sticker cross-validation (checking sticker positions against each other)
- Real-time / streaming filtering
- GPU acceleration

## Success Criteria

- [ ] Outlier detector flags physically impossible velocities/accelerations
- [ ] Both filter implementations produce visibly smoother trajectories than raw
- [ ] Compare mode generates overlay plots with raw + both filters for position, velocity, acceleration
- [ ] Correct mode saves `_xyz_corrected.csv` that downstream tasks consume seamlessly
- [ ] DAG config allows toggling mode, filter method, thresholds, and filter parameters
- [ ] Corrected data is used by downstream tasks via the existing `sticker_3d_tracking_path` context variable

---

## Technical Design

### Approach

Follow the existing strategy + factory pattern (`XYZExtractorFactory` / `XYZExtractorInterface`) to implement pluggable filters. The correction pipeline per sticker, per axis is:

1. **Detect outliers** — boolean mask (hard thresholds on velocity/acceleration AND/OR MAD-based statistical detection)
2. **Interpolate outliers** — linear interpolation from neighbours
3. **Apply smoothing filter** — Butterworth `filtfilt` (zero-phase) or Savitzky-Golay

Output is a separate `_xyz_corrected.csv` (raw file preserved). The corrected path overwrites the `sticker_3d_tracking_path` context variable so downstream tasks automatically consume corrected data.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Strategy pattern (2 filters + compare mode) | Extensible, visually comparable, follows existing patterns | More files | **Chosen** |
| Single hardcoded filter | Simpler | Can't compare, less flexible | Rejected |
| Overwrite raw CSV | Simpler pipeline | Loses raw data, can't re-run with different params | Rejected |
| Filter in 3D (joint x/y/z) | Preserves inter-axis correlation | More complex, each axis has different noise profile from depth camera | Rejected |

### Architecture Changes

New sub-package under the existing XYZ module:

```
code/src/preprocessing/stickers_analysis/xyz/motion_correction/
├── __init__.py
├── motion_filter_interface.py       — ABC: filter(signal, sampling_rate) -> filtered
├── butterworth_filter.py            — Zero-phase Butterworth low-pass (scipy.signal.butter + filtfilt)
├── savgol_filter.py                 — Savitzky-Golay polynomial smoothing (scipy.signal.savgol_filter)
├── motion_filter_factory.py         — FilterChoice enum + MotionFilterFactory
├── outlier_detector.py              — Hard threshold + MAD detection, linear interpolation
├── motion_correction_orchestrator.py — Coordinates: load -> detect -> interpolate -> filter -> save
└── diagnostics_plotter.py           — Position/velocity/acceleration overlay plots
```

New task script:
```
code/scripts/_3_preprocessing/_1_sticker_tracking/correct_xyz_stickers_motion.py
```

### Key Interfaces

**`MotionFilterInterface`** (ABC — mirrors `XYZExtractorInterface`):
```python
class MotionFilterInterface(ABC):
    @abstractmethod
    def filter(self, signal: np.ndarray, sampling_rate_hz: float) -> np.ndarray: ...

    @abstractmethod
    def name(self) -> str: ...  # Human-readable name for plot legends
```

**`ButterworthFilter`**: `scipy.signal.butter` + `scipy.signal.filtfilt` (zero-phase, no temporal lag)
- Params: `order` (default 2), `cutoff_hz` (default 6.0)

**`SavgolFilter`**: `scipy.signal.savgol_filter`
- Params: `window_length` (default 11), `polyorder` (default 3)

**`MotionFilterFactory`** (mirrors `XYZExtractorFactory`):
- `get_filter(choice, params)` — single filter instance
- `get_all_filters(params)` — list of all registered filters (for compare mode)

**`OutlierDetector`**:
- `detect(position_series) -> bool mask` — combines hard + statistical
- `interpolate_outliers(position_series, mask) -> cleaned series`
- Config dataclass: `enable_hard_threshold`, `max_velocity_mm_per_s`, `max_acceleration_mm_per_s2`, `enable_statistical`, `mad_multiplier`

**`MotionCorrectionOrchestrator`**:
- `run_correct(input_csv, output_csv, diagnostics_dir, ...)` — apply chosen filter, save corrected CSV + diagnostics
- `run_compare(input_csv, diagnostics_dir, ...)` — run all filters, produce overlay comparison plots only

### Reused Components

| Component | Location | Usage |
|-----------|----------|-------|
| `XYZDataFileHandler.load()` / `.save()` | `xyz/data_access/xyz_data_filehandler.py` | Read/write the flat sticker CSV |
| `XYZExtractorFactory` pattern | `xyz/core/xyz_extractor_factory.py` | Template for `MotionFilterFactory` |
| `scipy.signal.savgol_filter` | `utils/generic_signal_processing.py` | Already used in project |
| Velocity via `.diff()` | `feature_extraction/kinematics.py` | Same computation pattern |

### DAG YAML Config

```yaml
correct_xyz_stickers_motion:
  enabled: false
  options:
    force_processing: false
    mode: "correct"                    # "compare" or "correct"
    filter_method: "butterworth"       # "butterworth" or "savgol"
    filter_params:
      butterworth:
        order: 2
        cutoff_hz: 6.0
      savgol:
        window_length: 11
        polyorder: 3
    outlier_detection:
      enable_hard_threshold: true
      max_velocity_mm_per_s: 2000.0
      max_acceleration_mm_per_s2: 50000.0
      enable_statistical: true
      mad_multiplier: 5.0
  depends_on: [generate_xyz_stickers]
```

### Output File Layout

```
handstickers/
    ..._xyz_tracked.csv                           # raw (unchanged)
    ..._xyz_corrected.csv                         # corrected (correct mode only)
    ..._xyz_correction_diagnostics/
        sticker_blue_position_overlay.png
        sticker_blue_velocity_acceleration.png
        sticker_blue_outlier_summary.png
        sticker_green_position_overlay.png
        sticker_green_velocity_acceleration.png
        sticker_green_outlier_summary.png
        sticker_yellow_position_overlay.png
        sticker_yellow_velocity_acceleration.png
        sticker_yellow_outlier_summary.png
```

### Diagnostic Plots (per sticker)

1. **Position overlay** — 3 subplots (x, y, z axis), raw vs corrected. In compare mode: raw vs Butterworth vs Savgol overlaid
2. **Velocity & acceleration** — 2 subplots, magnitude computed from raw vs corrected positions
3. **Outlier summary** — raw signal with flagged frames marked as scatter points, colour-coded by detection method (hard threshold, statistical, or both)

---

## Implementation Plan

### Phase 1: Filter Strategy Pattern
**Goal:** Create the filter ABC, two implementations, and factory
**Started:** —
**Completed:** —

- [ ] Task 1.1 — Create `motion_correction/__init__.py` with public exports
- [ ] Task 1.2 — Create `motion_filter_interface.py` with `MotionFilterInterface` ABC (`filter()`, `name()`)
- [ ] Task 1.3 — Create `butterworth_filter.py` with `ButterworthFilter` using `scipy.signal.butter` + `filtfilt`
- [ ] Task 1.4 — Create `savgol_filter.py` with `SavgolFilter` using `scipy.signal.savgol_filter`
- [ ] Task 1.5 — Create `motion_filter_factory.py` with `FilterChoice` enum and `MotionFilterFactory`

**Files Created:**
- `code/src/preprocessing/stickers_analysis/xyz/motion_correction/__init__.py`
- `code/src/preprocessing/stickers_analysis/xyz/motion_correction/motion_filter_interface.py`
- `code/src/preprocessing/stickers_analysis/xyz/motion_correction/butterworth_filter.py`
- `code/src/preprocessing/stickers_analysis/xyz/motion_correction/savgol_filter.py`
- `code/src/preprocessing/stickers_analysis/xyz/motion_correction/motion_filter_factory.py`

**Dependencies:** None

### Phase 2: Outlier Detection
**Goal:** Detect and interpolate physically impossible motion spikes
**Started:** —
**Completed:** —

- [ ] Task 2.1 — Create `OutlierConfig` dataclass with toggleable hard/statistical parameters
- [ ] Task 2.2 — Implement hard-threshold detection (velocity > max mm/s, acceleration > max mm/s²)
- [ ] Task 2.3 — Implement statistical detection (MAD-based, configurable multiplier)
- [ ] Task 2.4 — Implement `interpolate_outliers()` with linear interpolation from neighbours

**Files Created:**
- `code/src/preprocessing/stickers_analysis/xyz/motion_correction/outlier_detector.py`

**Dependencies:** Phase 1

### Phase 3: Orchestrator & Diagnostics
**Goal:** Wire everything together and produce visual outputs
**Started:** —
**Completed:** —

- [ ] Task 3.1 — Create `DiagnosticsPlotter` with position overlay, kinematics comparison, and outlier summary plots
- [ ] Task 3.2 — Create `MotionCorrectionOrchestrator` with `run_correct()` and `run_compare()` methods
- [ ] Task 3.3 — Handle NaN gaps in raw data (explicit NaN check + interpolation before filtering)

**Files Created:**
- `code/src/preprocessing/stickers_analysis/xyz/motion_correction/diagnostics_plotter.py`
- `code/src/preprocessing/stickers_analysis/xyz/motion_correction/motion_correction_orchestrator.py`

**Dependencies:** Phases 1 and 2

### Phase 4: DAG Integration
**Goal:** Add the task to the preprocessing workflow and DAG config
**Started:** —
**Completed:** —

- [ ] Task 4.1 — Create task script `correct_xyz_stickers_motion.py` following `extract_stickers_xyz_positions.py` pattern
- [ ] Task 4.2 — Add Prefect flow `correct_xyz_stickers_motion_flow` in `preprocess_workflow_kinect_auto.py` (after `generate_xyz_stickers` flow at line 236)
- [ ] Task 4.3 — Add pipeline stage entry in `pipeline_stages` list (after `generate_xyz_stickers` at line 444), outputting to `sticker_3d_tracking_path` to override context for downstream tasks
- [ ] Task 4.4 — Add task config block to `configs/preprocess_workflow_kinect_auto_dag.yaml` (between `generate_xyz_stickers` at line 51 and Stage 3 at line 53)
- [ ] Task 4.5 — Update `depends_on` for `generate_3d_hand_in_motion` to include `correct_xyz_stickers_motion`
- [ ] Task 4.6 — Update `__init__.py` exports in `stickers_analysis/xyz/` and `stickers_analysis/`

**Files Created:**
- `code/scripts/_3_preprocessing/_1_sticker_tracking/correct_xyz_stickers_motion.py`

**Files Modified:**
- `code/scripts/preprocess_workflow_kinect_auto.py` — add flow function + pipeline stage entry
- `configs/preprocess_workflow_kinect_auto_dag.yaml` — add task config block, update `depends_on`
- `code/scripts/_3_preprocessing/_1_sticker_tracking/__init__.py` — add export
- `code/src/preprocessing/stickers_analysis/xyz/__init__.py` — add motion_correction exports
- `code/src/preprocessing/stickers_analysis/__init__.py` — add exports

**Dependencies:** Phase 3

---

## Testing Plan

### Manual Verification
- [ ] Run with `mode: "compare"` on a known session — verify 9 diagnostic plots generated (3 per sticker: position overlay, velocity/acceleration, outlier summary)
- [ ] Visually confirm both filter curves appear on position overlay plots alongside raw signal
- [ ] Run with `mode: "correct"` — verify `_xyz_corrected.csv` is created alongside raw `_xyz_tracked.csv`
- [ ] Verify corrected CSV has identical structure/columns as raw CSV
- [ ] Run downstream task `generate_3d_hand_in_motion` — verify it consumes corrected CSV via `sticker_3d_tracking_path` context variable
- [ ] Verify pipeline works with task `enabled: false` (raw path passes through unchanged)
- [ ] Test with `enable_hard_threshold: false` only (statistical detection only)
- [ ] Test with `enable_statistical: false` only (hard threshold only)

### Edge Cases
- [ ] Sticker with all-NaN frames (tracking failed entirely) — should produce empty/pass-through output without crashing
- [ ] Very short recordings (< filter window length) — should gracefully degrade or skip filtering
- [ ] Single sticker present (not all 3 colours) — should process available stickers only

---

## Documentation Plan

- [ ] Add knowledge-base note if filter parameter tuning reveals non-obvious constraints
- [ ] Update inline comments in `preprocess_workflow_kinect_auto.py` stage numbering

---

## Rollback Plan

1. Set `correct_xyz_stickers_motion: enabled: false` in DAG YAML — pipeline skips the task entirely
2. Downstream tasks fall back to raw `_xyz_tracked.csv` via unchanged context variable
3. No destructive changes — raw CSV is never modified or overwritten
4. Git revert: single feature branch, clean to revert

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Filter cutoff too aggressive, removes real motion | Medium | High | Compare mode exists specifically for visual parameter tuning; conservative default (6Hz for 30fps data) |
| NaN gaps in raw data break filter | Medium | Medium | Explicit NaN interpolation step before filtering |
| Nested dict DAG options not handled by config handler | Low | Medium | Verify `get_task_option` supports nested dict access; flatten config structure if needed |
| Large sessions slow down diagnostics plotting | Low | Low | Use matplotlib `agg` backend, save PNG not interactive |

---

## References

- Related completed plan: `docs/development/plans/completed/sticker-depth-edge-gradient-bias-correction.md` — prior XYZ quality improvement
- Existing factory pattern: `code/src/preprocessing/stickers_analysis/xyz/core/xyz_extractor_factory.py`
- Existing signal processing: `code/src/utils/generic_signal_processing.py`
- Downstream consumer: `code/src/analysis/touch_analytics/feature_extraction/kinematics.py`
