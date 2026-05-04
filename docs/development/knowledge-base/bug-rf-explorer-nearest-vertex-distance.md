# Bug: RF Explorer — nearest-vertex distance exceeds 15mm threshold

## Symptom

Running `explore_rf_feature_space` via the analysis workflow raises:

```
ValueError: load_explorer_data: 248 frame(s) have nearest-vertex distance
exceeding 15mm (max=29.34mm)
```

The error originates in `rf_explorer_data.py::load_explorer_data` at the
`cKDTree` distance check (line ~90).

## Status

**Fixed.** The 248 bad frames were artifacts of cubic NaN interpolation
extrapolating beyond the forearm surface at touch-group boundaries — not a
session mismatch or transform bug.

**Fix applied (2026-05-01):** Added `limit_area='inside'` to both
`pd.Series.interpolate()` calls in `_interpolate_group` in
`code/src/analysis/touch_analytics/preparation/interpolation.py`.  This
restricts interpolation to NaN values that sit between two valid anchor
values — no extrapolation at group edges.  Boundary frames that remain NaN
are dropped by downstream NaN filters.

## Root Cause

Diagnostic testing at three CSV levels (same session, same PLY):

| CSV level | Rows (after NaN filter) | Frames > 15mm |
|-----------|------------------------|---------------|
| Final postprocessing CSV (pre-analysis) | 8,003 | 0 |
| `_prepared.csv` (after preparation) | 213,937 | 248 |
| `_series_augmented.csv` (after series transforms) | 213,937 | 248 |

The preparation stage performs cubic NaN interpolation on
`contact_location_x/y/z`, filling in coordinates for frames that originally
had no contact data.  At touch-event boundaries (onset/offset), the cubic
spline overshoots beyond the forearm surface, producing contact locations up
to 29mm from the nearest mesh vertex.  Series transforms only adds columns
and does not modify contact coordinates, so the same 248 frames carry through.

### Hypotheses ruled out

1. **Session mismatch** — CSV and PLY paths confirmed to reference the same
   session.
2. **Mesh–contact transform mismatch** — the rotation is computed once and
   applied identically to both point sets; raw (pre-rotation) distances show
   the same 248 failures, confirming the issue is in the data, not the
   transform.
3. **Mesh resolution / holes** — the postprocessing CSV (same PLY) has zero
   failures, so the mesh is fine.

## Investigation Steps Taken

1. Added temporary diagnostic prints to `rf_explorer_data.py` to capture the
   CSV and PLY paths on failure.  Confirmed session IDs match.  (Prints
   removed after capture.)
2. Created `code/scripts/diagnose_rf_explorer_distance.py` — standalone
   diagnostic script with four phases (raw extents, overlap, distance
   analysis with percentile breakdown, mesh quality).
3. Ran the diagnostic script against three CSV levels to isolate the stage
   that introduces the bad frames.
4. Fixed `resolve_forearm_ply` path: removed incorrect `forearm_rf_centered/`
   subdirectory (unrelated to the distance bug but was wrong).

## Key Files

- `code/src/analysis/receptive_field_mapping/rf_explorer_data.py` — distance
  check (cKDTree, 15mm threshold)
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` —
  `launch_feature_space_explorer`, resolves CSV/PLY paths
- `code/src/analysis/touch_analytics/preparation/interpolation.py` — cubic
  NaN interpolation that produces the overshoot
- `code/scripts/diagnose_rf_explorer_distance.py` — diagnostic script
