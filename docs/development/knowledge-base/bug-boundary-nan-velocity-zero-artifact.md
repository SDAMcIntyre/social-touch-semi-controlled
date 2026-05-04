# Bug: Boundary NaN sticker positions produce spurious zero velocity at touch ends

**Date:** 2026-05-05  
**Status:** Documented, not yet fixed  
**Affects:** Stage 1 (`touch_series_transforms`) and all downstream consumers of velocity/acceleration columns  

---

## Summary

The last 1–2 rows of most touch groups in the series-augmented CSV have spurious
`0.0` values in `hand_velocity_x/y/z` and `hand_acceleration_x/y/z` instead of NaN.
This is caused by a `fillna(0)` in the velocity finite-difference calculation that
cannot distinguish "first frame of a touch" (where 0 is a reasonable fill) from
"NaN position data at a touch boundary" (where 0 is incorrect). These fake zeros
propagate through all downstream stages without filtering.

---

## Root cause chain

### Step 1 — Kinect dropout at touch group boundaries (Stage 0)

The Kinect outputs position data at 30 Hz. Stage 0 (`touch_preparation`) forward-fills
sticker positions from 30 Hz up to 1 kHz. Forward-fill propagates the last valid value
forward, but stops at the boundary of a touch group. The Kinect frequently loses
tracking in the last Kinect frame before a touch ends, so the last 1–2 rows of most
touch groups have `NaN` in the sticker position columns (`sticker_blue/green/yellow_position_x/y/z`).

From CLAUDE.md:
> The Kinect often drops out before the touch group boundary ends, so the last rows
> of a touch group commonly remain NaN in the prepared CSV. Confirmed: 100% of stroke
> groups had a NaN at their final row (pre-interpolation).

Cubic interpolation (also Stage 0) only fills interior NaNs — it requires valid samples
on both sides and cannot fill trailing NaNs. So the NaN rows survive into `_prepared.csv`.

### Step 2 — `resolve_hand_position` passes NaN positions through (Stage 1)

`resolve_hand_position` in
`code/src/analysis/touch_analytics/representation/series_level/kinematics.py`
selects sticker columns and returns them as `hand_position_x/y/z`. It applies no NaN
guard — if the input sticker positions are NaN, the output `hand_position_*` values
are also NaN.

### Step 3 — `compute_velocity` fills all NaN with 0 (Stage 1)

`compute_velocity` (kinematics.py line 76):

```python
vectors = group[position_cols].diff().fillna(0) * DATA_RATE_HZ
```

`.diff()` produces NaN for two reasons:
1. The **first row of a touch group** — no prior frame to diff against. Filling with 0
   here is intentional: the touch starts with velocity = 0.
2. Any row where `hand_position_*` is NaN — diff of `(valid, NaN)` = NaN, and diff of
   `(NaN, NaN)` = NaN. These boundary NaN rows should produce NaN velocity, not zero.

`fillna(0)` cannot distinguish these two cases, so it fills both as 0. The result:
the last 1–2 rows of every touch group get `hand_velocity_x/y/z = 0.0` instead of NaN.

The same pattern propagates to `compute_acceleration` (kinematics.py line 82):

```python
vectors = vel_df[HAND_VELOCITY_COLUMNS].diff().fillna(0) * DATA_RATE_HZ
```

A NaN velocity row produces a NaN diff, which is again filled with 0. So the boundary
rows get `hand_acceleration_x/y/z = 0.0` as well.

---

## Affected rows

- **Frequency:** 100% of touch groups have at least one affected row (matches the
  confirmed Kinect dropout rate noted in CLAUDE.md).
- **Count per touch:** Typically 1–2 rows at the end of each touch group.
- **Scale:** With a 619k-row session CSV, and a typical touch duration of ~300–500 ms
  (300–500 rows at 1 kHz), a 1-row artifact represents ~0.2–0.3% of each touch —
  negligible for long strokes, more impactful for short taps.

### Relative impact by gesture type

| Gesture | Typical duration | Affected rows | % of touch |
|---|---|---|---|
| Long stroke | 500–2000 ms | 1–2 | < 0.5% |
| Short tap | 10–50 ms | 1–2 | 2–20% |

Short taps are the most vulnerable because 1–2 bad frames represent a larger fraction
of the total, and the minimum velocity across the touch is always pulled to 0.

---

## Downstream impact

The spurious zeros are **not filtered out at any downstream stage**. They are treated
as valid data throughout.

### Stage 1 — velocity scalar transforms

`compute_velocity_amplitude` (`velocity_scalar.py` lines 11–14) computes the L₂ norm
directly from the velocity components in numpy with no NaN guard. Spurious zeros
produce amplitude = 0 rather than NaN.

`compute_velocity_signed` (`velocity_scalar.py` lines 17–37) computes an SVD over the
full per-touch velocity matrix (including the boundary rows). The boundary zeros bias
the mean velocity and shift the principal axis, slightly misaligning the signed-velocity
projection for affected touches.

### Stage 2a — feature extraction

`extraction_pipeline.py` passes per-touch group DataFrames to statistical aggregation
(`mean`, `std`, `min`, `max`, `range`, `skewness`) with no NaN filtering or
`contact_detected` masking beforehand. Pandas `skipna=True` is the implicit default,
which skips real NaN values but includes the spurious 0s. Effects:

- **`mean`**: pulled toward zero; magnitude depends on touch length
- **`std`**: inflated (extra low-value outlier frame)
- **`min`**: always 0 for velocity/acceleration columns on any touch that has a
  boundary NaN row — i.e., effectively all touches
- **`skewness`**: `statistical.py` calls `.dropna()` before computing, so real NaNs
  are excluded, but the spurious zeros are still present

The currently active clustering configuration (`pressure_velocity_mean_cartesian_binning`
in `configs/analyse_workflow_dag.yaml`) uses `hand_velocity_amplitude[mean]` as a
clustering feature. This statistic is directly affected.

### RF feature space explorer

`rf_feature_space_explorer.py` reads `hand_velocity_signed` directly from the
series-augmented CSV (line 244). Spurious zeros in `hand_velocity_signed` are included
in the scatter plot and bias the `np.nanpercentile` calls used to set axis ranges.

`rf_explorer_data.py` filters frames using `contact_points > 0` (lines 234–241) but
does not filter on velocity or `contact_detected`, so frames with valid contact points
but spurious velocities are retained.

---

## Why pressure is not affected

`compute_pressure` (`pressure.py`) guards against zero/negative area:

```python
result = depth.where(area > 0, other=float('nan')) / area.where(area > 0, other=float('nan'))
```

If `contact_depth` or `contact_area` is NaN (as it would be on a boundary row where
the Kinect lost tracking), the division produces NaN — which is correct. Boundary rows
produce NaN pressure, not 0. Downstream aggregations then skip them via `skipna=True`.
Pressure is the only default-enabled series transform that handles this correctly.

---

## Why `contact_area_metadata.iloc[0]` is a secondary concern

`resolve_hand_position` reads only the first row's `contact_area_metadata` to choose
whether to use hand stickers (green + yellow average) or fingertip sticker (blue).
If the first row of a touch group has NaN metadata, `str(NaN).lower() = 'nan'` does
not contain `'hand'`, so the function silently falls through to the blue sticker path.
In practice `contact_area_metadata` is a categorical metadata column that is unlikely
to be NaN at the first row of a touch, so this is a lower-risk edge case than the
main velocity artifact.

---

## Affected files

| File | Role | How affected |
|---|---|---|
| `code/src/analysis/touch_analytics/representation/series_level/kinematics.py` | Source of the bug | `fillna(0)` on line 76 (velocity) and line 82 (acceleration) |
| `code/src/analysis/touch_analytics/series_pipeline.py` | Stage 1 orchestrator | Calls `compute_velocity` and `compute_acceleration`; writes bad values to CSV |
| `code/src/analysis/touch_analytics/representation/series_level/velocity_scalar.py` | Velocity scalars | `compute_velocity_amplitude` and `compute_velocity_signed` receive and propagate zeros |
| `code/src/analysis/touch_analytics/extraction_pipeline.py` | Stage 2a feature extraction | Statistical aggregation includes spurious zeros in all velocity/acceleration features |
| `code/src/analysis/receptive_field_mapping/rf_explorer_data.py` | RF explorer data loader | No velocity filtering; contact_points filter does not exclude affected rows |
| `code/src/analysis/receptive_field_mapping/gui/rf_feature_space_explorer.py` | RF GUI | Reads `hand_velocity_signed` with no boundary filtering |

---

## Potential fix approaches (not yet implemented)

Two general strategies:

**A — Fix at source in `compute_velocity`:** Replace the single `fillna(0)` with a
targeted fill that only zeroes the first row of each touch group, and leaves all other
NaN as NaN. This would propagate NaN correctly to all downstream consumers, but would
require verifying that downstream aggregations (which use `skipna=True`) behave
acceptably with NaN velocity frames inside a touch group.

**B — Mask at the touch-group level before aggregation:** Add a pre-aggregation filter
in `extraction_pipeline.py` (and any other consumer) that drops rows where sticker
positions or velocity values are NaN before computing per-touch statistics. This
preserves the series CSV as-is but corrects the aggregate values. The RF explorer
would need a similar filter.

Both approaches require deciding what `min(hand_velocity_amplitude)` should mean for a
touch that had a Kinect dropout on its last frame — whether to report the true minimum
over valid frames, or to flag the touch as having incomplete data.
