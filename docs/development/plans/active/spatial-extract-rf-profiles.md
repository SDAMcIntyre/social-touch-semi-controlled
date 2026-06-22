# Plan: Extract 1D RF Profiles with Boundary Crossings

**Date:** 2026-06-22
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/family2-response-summary`
**Branch:** `feature/spatial-extract-rf-profiles`

---

## Overview

Add a new DAG pipeline task that extracts 1D cross-section profiles from the existing 150x150 RF heatmap grids. For each session and gesture type, the task slices the interpolated IFF grid at each constant-V coordinate, producing an IFF-vs-U profile per row, and computes where the inflection boundary contour intersects each profile. Outputs are per-gesture CSVs and two PNG figure types.

## Problem Statement

The `spatial_extract_boundaries` task produces rich 2D heatmap data (interpolated grids + inflection boundary contours) but provides no way to inspect the RF's intensity profile along individual cross-sections or to quantify where the boundary sits on each cross-section. This information is needed to characterise how the RF response rises and falls across the forearm surface and how the boundary width varies along the proximal-distal axis.

## Goals

### In Scope
1. Extract 150 constant-V cross-section profiles per session per gesture type from existing NPZ grids
2. Compute boundary intersection U-coordinates at each V-slice using scanline intersection
3. Write per-gesture CSVs with profile summary stats and boundary crossing positions
4. Render profile strip images (2D heatmap with boundary overlay and crossing markers)
5. Render representative 1D line profiles (~15 per gesture) with boundary markers

### Out of Scope
- Recomputing or modifying the upstream heatmap interpolation or boundary detection
- Profiles along arbitrary angles (only axis-aligned constant-V slices)
- Cross-session comparison of profiles (future task)
- 3D back-projection of profile data

## Success Criteria

- [ ] New DAG task `spatial_extract_rf_profiles` runs end-to-end for all sessions
- [ ] Per-gesture CSV contains 150 rows (one per V-index) with correct boundary crossings
- [ ] Boundary widths are plausible (same order of magnitude as the inflection boundary's PCA major axis)
- [ ] Profile strip PNG visually matches the existing interpolated heatmap (rotated orientation)
- [ ] Representative profile plots show IFF curves with boundary markers at correct U positions
- [ ] `pytest` passes with no import breakage

---

## Technical Design

### Approach

Load pre-computed 150x150 grids and boundary contours from the existing `spatial_extract_boundaries` NPZ files. The grid uses `np.mgrid` convention: axis 0 = U (150 values), axis 1 = V (150 values). A constant-V slice at column j gives `grid_z[:, j]` = 150 IFF values across U. Boundary crossings are found via vectorized scanline intersection on the closed contour polygon.

This is a pure read-and-derive task with no upstream modifications needed.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Slice existing grid (constant-V columns) | Simple, uses pre-computed data, no recomputation | Only axis-aligned slices | **Chosen** |
| Re-interpolate along arbitrary lines through centroid | Richer angular profiles | Requires recomputation, more complex | Rejected — future extension if needed |
| PCA-aligned slices (major/minor axis only) | Directly meaningful 2 profiles | Only 2 slices, misses spatial variation | Rejected — too limited |

### Architecture Changes

Two new modules, four existing files modified. No new abstractions or classes — follows the existing pattern of pipeline orchestrator function + renderer functions.

```
code/src/analysis/receptive_field_mapping/
    pipelines/
        rf_profile_extraction_pipeline.py  (NEW)
    rendering/
        rf_profile_renderer.py             (NEW)
```

### Grid Coordinate Convention

```
grid_u, grid_v = np.mgrid[u_min:u_max:150j, v_min:v_max:150j]

Axis 0 (i=0..149): U values — grid_u[:, 0] gives the 150 U coordinates
Axis 1 (j=0..149): V values — grid_v[0, :] gives the 150 V coordinates

Constant-V profile at column j:
  u_coords = grid_u[:, 0]        # shape (150,)
  iff_values = grid_z[:, j]      # shape (150,)
  v_value = grid_v[0, j]         # scalar (mm)
```

### Boundary Intersection Algorithm

Vectorized scanline intersection on the closed boundary polygon:

1. Close the polygon: append first point to end
2. For each segment `(u1,v1)→(u2,v2)`, check if it straddles `v_target` using one-sided strict inequality (avoids double-counting at vertices)
3. Linearly interpolate U at crossing: `u_cross = u1 + (v_target - v1) * (u2 - u1) / (v2 - v1)`
4. Return sorted array of U-intercepts

Typical result: 0 crossings (outside boundary extent) or 2 crossings (entry/exit). Non-convex boundaries may produce >2 crossings.

---

## Implementation Plan

### Phase 1: Infrastructure
**Goal:** Register the new task in the pipeline framework

- [x] Add `SPATIAL_EXTRACT_RF_PROFILES` constant to `output_dirs.py`
- [x] Add DAG config entry in `analyse_workflow_processing_dag.yaml`

**Files Modified:**
- `code/src/analysis/pipeline/output_dirs.py` — Add constant after `SPATIAL_COMPARE_RF_CENTERS`
- `configs/analyse_workflow_processing_dag.yaml` — Add task entry after `spatial_compare_rf_centers` (around line 226)

**Dependencies:** None

### Phase 2: Core Logic
**Goal:** Implement profile extraction and boundary intersection

- [x] Create `rf_profile_extraction_pipeline.py` with `run_rf_profile_extraction()` entry point
- [x] Implement `find_boundary_u_crossings()` vectorized scanline intersection
- [x] Implement per-session, per-gesture profile extraction loop
- [x] Implement CSV writing with summary stats and boundary positions
- [x] Implement sentinel-based idempotency

**Files Created:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_profile_extraction_pipeline.py`

**Key function signatures:**
```python
def run_rf_profile_extraction(
    session_configs: list[tuple[Path, Path]],
    output_dir: Path,
    force_processing: bool = False,
    iff_metric: str = "mean",
) -> None:

def find_boundary_u_crossings(
    contour_uv: np.ndarray,
    v_target: float,
) -> np.ndarray:
```

**Dependencies:** Phase 1

### Phase 3: Rendering
**Goal:** Implement the two visualisation types

- [x] Create `rf_profile_renderer.py` with two rendering functions
- [x] `render_profile_strip_with_boundary()` — 2D pcolormesh with boundary contour and crossing markers
- [x] `render_representative_profiles()` — ~15 stacked 1D line subplots with boundary markers

**Files Created:**
- `code/src/analysis/receptive_field_mapping/rendering/rf_profile_renderer.py`

**Dependencies:** Phase 2

### Phase 4: Wiring
**Goal:** Connect to the DAG execution pipeline

- [x] Add import and `__all__` entry in `receptive_field_mapping/__init__.py`
- [x] Add `@flow` function in `analysis_workflow_processing.py`
- [x] Add pipeline stage entry in `_build_pipeline_stages()`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/__init__.py` — Import + `__all__`
- `code/scripts/analysis_workflow_processing.py` — Import, flow function, stage entry

**Dependencies:** Phase 3

---

## Output Specification

### CSV Format

File: `{session_id}_rf_profiles_{gtype}.csv`
Location: `4_analysed/spatial_extract_rf_profiles/iff_{metric}/{session_id}/`

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | str | Session identifier |
| `gesture_type` | str | Gesture subset key |
| `v_index` | int | Column index in grid (0-149) |
| `v_value_mm` | float | V coordinate (mm) |
| `n_valid` | int | Non-NaN U samples in this profile |
| `profile_max` | float | Max IFF (NaN if all NaN) |
| `profile_mean` | float | Mean IFF (NaN if all NaN) |
| `n_boundary_crossings` | int | Boundary crossings at this V |
| `boundary_u_left_mm` | float | Smallest U crossing (NaN if none) |
| `boundary_u_right_mm` | float | Largest U crossing (NaN if none) |
| `boundary_width_mm` | float | right - left (NaN if <2 crossings) |
| `boundary_u_all` | str | Semicolon-separated crossing U values |

### PNG Output

Per session, per gesture type:
- `{session_id}_rf_profile_strip_{gtype}.png` — Heatmap strip with boundary overlay
- `{session_id}_rf_profiles_representative_{gtype}.png` — ~15 stacked 1D line profiles

### DAG Config Entry

```yaml
  spatial_extract_rf_profiles:
    category: spatial_sensitivity
    enabled: true
    options:
      force_processing: false
      iff_metric: both
    depends_on: [spatial_extract_boundaries]
```

---

## Testing Plan

### Unit Tests
- [ ] `find_boundary_u_crossings` with a known square polygon — verify 2 crossings at expected U values
- [ ] `find_boundary_u_crossings` with V outside polygon extent — verify empty result
- [ ] `find_boundary_u_crossings` with non-convex polygon — verify >2 crossings
- [ ] `find_boundary_u_crossings` with None/empty contour — verify empty result

### Integration Tests
- [ ] Run `run_rf_profile_extraction` on one session — verify CSV row count (150 per gesture type)
- [ ] Verify boundary_width values are consistent with the inflection boundary PCA major axis from the source NPZ

### Manual Verification
- [ ] Profile strip PNG matches the existing interpolated heatmap (same intensity pattern, different orientation)
- [ ] Boundary markers in representative profiles align with the red contour in existing heatmap PNGs
- [ ] All-NaN profiles (outside mesh) correctly skipped in representative plots

### Edge Cases
- [ ] Session with missing boundary for a gesture type — CSV has NaN boundary columns, plots omit markers
- [ ] Gesture type with very few valid profiles — renderer handles gracefully
- [ ] Non-convex boundary producing >2 crossings — `boundary_u_all` captures all, `left`/`right` are outermost

---

## Documentation Plan

- [ ] Update `code/src/analysis/CLAUDE.md` — add entry under "Receptive field mapping" for the new task
- [ ] DAG config comments (included in the YAML entry) serve as inline documentation

---

## Rollback Plan

1. Remove the DAG config entry from `analyse_workflow_processing_dag.yaml`
2. Remove the flow function and stage entry from `analysis_workflow_processing.py`
3. Remove the import from `receptive_field_mapping/__init__.py`
4. Delete the two new files (`rf_profile_extraction_pipeline.py`, `rf_profile_renderer.py`)
5. Remove the constant from `output_dirs.py`
6. No data migrations or breaking changes — purely additive task

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| NPZ key names change in upstream task | Low | High | Load gesture types from NPZ `gesture_types` key, fail-fast if grid keys missing |
| Non-convex boundaries produce confusing crossing patterns | Medium | Low | Report all crossings in `boundary_u_all`, use outermost pair for `left`/`right` |
| All-NaN profiles dominate output (forearm mesh covers small fraction of grid) | Medium | Low | Skip all-NaN rows in representative plots, report `n_valid` in CSV |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Infrastructure | Small (2 line additions) | None |
| Phase 2: Core Logic | Medium (pipeline + intersection algorithm) | Phase 1 |
| Phase 3: Rendering | Medium (2 figure types) | Phase 2 |
| Phase 4: Wiring | Small (imports + flow function) | Phase 3 |

---

## References

- Upstream task: `spatial_extract_boundaries` — `rf_population_response_field_pipeline.py`
- Grid construction: `rf_population_map_renderer.py::compute_interpolated_grid()` (line 155)
- Boundary detection: `rf_inflection_boundary.py::compute_inflection_boundary()`
- NPZ save: `rf_population_response_field_pipeline.py::_save_response_fields_npz()` (line 681)

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- code/scripts/analysis_workflow_processing.py
- code/src/analysis/pipeline/output_dirs.py
- code/src/analysis/receptive_field_mapping/__init__.py
- code/src/analysis/receptive_field_mapping/pipelines/rf_profile_extraction_pipeline.py
- code/src/analysis/receptive_field_mapping/rendering/rf_profile_renderer.py
- configs/analyse_workflow_processing_dag.yaml
- docs/development/plans/active/spatial-extract-rf-profiles.md
