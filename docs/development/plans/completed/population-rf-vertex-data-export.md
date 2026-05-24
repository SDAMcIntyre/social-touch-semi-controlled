# Plan: Population RF Vertex Data Export (NPZ)

**Date:** 2026-05-19
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-24 18:03
**Base Branch:** `feature/population-rf-inflection-boundary`
**Branch:** `feature/population-rf-vertex-data-export`

---

## Overview

Add an NPZ data export to the population RF map pipeline so that per-vertex
`(u, v, IFF_value)` data and interpolated grids are saved alongside the
existing PNG outputs. This enables a future 3D visualizer to plot the mean
IFF topology as a surface (value as Z height on the UV plane) for
investigating receptive field shape.

## Problem Statement

The population RF map pipeline (`rf_population_map_pipeline.py`) currently
produces only PNG images and a sentinel JSON. The underlying per-vertex
heatmap data and interpolated grids are computed but discarded after
rendering. To investigate RF topology in 3D or build downstream analysis
tools, this data must be re-derived from scratch — the pipeline offers no
machine-readable output.

## Goals

### In Scope

1. Export per-SLIM-vertex data `(u, v, IFF_value)` for each gesture type as
   an NPZ file per session
2. Include the 150x150 interpolated grids (`grid_u`, `grid_v`, `grid_z`)
   per gesture type — already computed, just needs saving
3. Include shared mesh data (`forearm_uv`, `forearm_faces`, `forearm_V`) and
   metadata (`session_id`, `neuron_mode`, `min_overlap_pct`, etc.)
4. Include inflection boundary data per gesture type when available
5. Record the NPZ path in the sentinel JSON for discoverability

### Out of Scope

- 3D visualizer implementation (future work — this plan provides the data it
  will consume)
- Changes to the interpolation algorithm or grid resolution
- New DAG config toggles (NPZ export is unconditional when the pipeline runs)
- Changes to PNG rendering or existing sentinel structure beyond adding the
  new path key

## Success Criteria

- [ ] Running `visualize_population_rf_maps` produces
      `{session_id}_rf_population_vertex_data.npz` in the output directory
- [ ] NPZ contains `forearm_uv` (V, 2) and `heatmap_{gtype}` (V,) with
      matching V dimension for each gesture type present
- [ ] NPZ contains `grid_u_{gtype}`, `grid_v_{gtype}`, `grid_z_{gtype}`
      each (150, 150) per gesture type
- [ ] Sentinel JSON includes `vertex_data_npz` key pointing to the NPZ path
- [ ] Existing PNG outputs are byte-identical before and after
- [ ] All existing tests pass unchanged

---

## Technical Design

### Approach

Save all per-vertex and interpolated grid data that is already computed in
the pipeline's Pass 1 loop. No new computation is required — the grid data
just needs to be accumulated in a dict alongside the existing
`gesture_boundaries` dict, then written to disk via `np.savez()`.

The NPZ uses gesture-type-suffixed keys (`heatmap_all`, `grid_z_tap`, etc.)
rather than stacked arrays because gesture subsets are dynamic (a session may
have zero touches for a gesture type, causing it to be skipped).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| NPZ with `{field}_{gtype}` keys | Simple consumer API, handles missing gestures naturally, matches existing `rf_population_grid_pipeline.py` pattern | More keys in NPZ | **Chosen** |
| NPZ with stacked 3D arrays | Compact | Requires separate index array for gesture→axis mapping; awkward for missing gestures | Rejected |
| JSON export | Human-readable | Too large for (150,150) grids; no native ndarray support | Rejected |
| HDF5 | Hierarchical grouping | New dependency (h5py), overkill for ~5 MB per session | Rejected |

### Architecture Changes

No new modules or classes. One new private function `_save_vertex_data_npz()`
added to the existing pipeline module. One optional field added to the
existing `_SessionCompositeData` dataclass.

### NPZ Array Manifest

**Shared mesh data (session-wide):**

| Key | dtype | Shape | Description |
|-----|-------|-------|-------------|
| `forearm_uv` | float64 | (V, 2) | SLIM UV coordinates per vertex (mm) |
| `forearm_faces` | int32 | (F, 3) | Triangle face indices |
| `forearm_V` | float64 | (V, 3) | 3D vertex positions (mm, Kinect frame) |

**Metadata:**

| Key | dtype | Description |
|-----|-------|-------------|
| `session_id` | str | Session identifier |
| `neuron_mode` | str | `"iff"` or `"spike"` |
| `min_overlap_pct` | float64 | Overlap percentage threshold |
| `gesture_types` | str array | Gesture types present |
| `inflection_sigma` | float64 | Smoothing sigma; NaN if disabled |

**Per-gesture arrays** (for each `gtype` in `gesture_types`):

| Key pattern | dtype | Shape | Description |
|-------------|-------|-------|-------------|
| `heatmap_{gtype}` | float64 | (V,) | NaN=uncontacted, -1.0=below threshold, positive=mean IFF/spike |
| `n_touches_{gtype}` | int64 | scalar | Touches in gesture subset |
| `threshold_{gtype}` | int64 | scalar | Minimum touch count threshold |
| `grid_u_{gtype}` | float64 | (150, 150) | Interpolated grid U coordinates |
| `grid_v_{gtype}` | float64 | (150, 150) | Interpolated grid V coordinates |
| `grid_z_{gtype}` | float64 | (150, 150) | Interpolated heatmap values |

**Per-gesture inflection boundary** (only when computed and found):

| Key pattern | dtype | Shape |
|-------------|-------|-------|
| `boundary_contour_uv_{gtype}` | float64 | (N, 2) |
| `boundary_area_uv_{gtype}` | float64 | scalar |
| `boundary_centroid_uv_{gtype}` | float64 | (2,) |

---

## Implementation Plan

### Phase 1: Data accumulation and export

**Started:** 2026-05-19
**Completed:** 2026-05-19

**Goal:** Save per-vertex and grid data as NPZ in the population RF map
pipeline.

**Tasks:**

- [x] Task 1.1 — Add `per_gesture_grids: dict = {}` alongside
      `gesture_boundaries` at line 249
- [x] Task 1.2 — Store grid tuple after `compute_interpolated_grid`:
      `per_gesture_grids[gtype] = (grid_u, grid_v, grid_z)` after line 262
- [x] Task 1.3 — Create `_save_vertex_data_npz()` function above
      `_write_sentinel()` that builds the data dict and calls `np.savez()`
- [x] Task 1.4 — Call `_save_vertex_data_npz()` after sentinel write
      (between line 300 and line 305), append NPZ path to `produced`
- [x] Task 1.5 — Add `vertex_data_npz: Path | None = None` field to
      `_SessionCompositeData` dataclass and parameter to `_write_sentinel()`
- [x] Task 1.6 — Record `vertex_data_npz` key in sentinel JSON; pass NPZ
      path at both sentinel write call sites (Pass 1 and Pass 2)

**Files Modified:**

- `code/src/analysis/receptive_field_mapping/rf_population_map_pipeline.py`
  — accumulate grids, new `_save_vertex_data_npz()`, updated sentinel/dataclass

**Dependencies:** None

**Existing code to reuse:**

- `_write_sentinel()` pattern in same file (JSON with optional keys)
- `np.savez()` pattern from `rf_population_grid_pipeline.py::_save_grid_results()`
- `InflectionBoundary` dataclass fields from `rf_inflection_boundary.py`

---

## Testing Plan

### Unit Tests

No new unit tests required — the NPZ export is a pure data-saving operation
with no logic to test beyond numpy serialization. The existing 23
inflection-boundary tests and pipeline tests remain unchanged.

### Manual Verification

- [ ] Run the analysis workflow with `visualize_population_rf_maps` enabled
      for one session
- [ ] Confirm `{session_id}_rf_population_vertex_data.npz` exists in
      `4_analysed/population_rf_maps/{session_id}/`
- [ ] Load NPZ in Python and verify:
  - `forearm_uv.shape[0] == heatmap_all.shape[0]` (same vertex count)
  - `grid_z_all.shape == (150, 150)`
  - `gesture_types` lists expected subsets
  - Boundary keys present for gestures with boundaries, absent otherwise
  - `neuron_mode` and `session_id` match expectations
- [ ] Confirm sentinel JSON contains `vertex_data_npz` key
- [ ] Confirm existing PNGs are unchanged (diff or visual comparison)
- [ ] Run `pytest code/tests/` — all tests pass

### Edge Cases

- [ ] Session with zero touches for one gesture type — that gesture's keys
      should be absent from the NPZ, and `gesture_types` should not list it
- [ ] Session with `inflection_sigma=None` — boundary keys should be absent
      for all gestures, `inflection_sigma` stored as NaN

---

## Documentation Plan

- [x] Update `code/src/analysis/CLAUDE.md` — add NPZ output to the
      "Population RF maps" bullet under Receptive field mapping

---

## Rollback Plan

1. Revert the single commit that adds the NPZ export
2. No data migration needed — the NPZ is a new artifact alongside existing
   outputs; deleting it has no downstream impact
3. Sentinel JSON remains valid without the `vertex_data_npz` key (it's
   optional)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| NPZ file too large for disk | Low | Low | Estimated ~5-6 MB compressed per session; negligible vs PNG sizes |
| Grid data references stale after code changes | Low | Low | Grid arrays are simple numpy data with no code dependency; future format changes would need a version key |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 | ~30 minutes | None |

---

## References

- Claude Code plan: `.claude/plans/in-addition-to-images-sunny-charm.md`
- Related module: `code/src/analysis/receptive_field_mapping/rf_population_grid_pipeline.py` (NPZ pattern)
- Knowledge base: `note-analysis-pipeline-coordinate-spaces.md` (units are mm throughout)

---

## Follow-on

**2026-05-20** — Plan `extract-population-rf-response-field-boundaries` renamed the NPZ file
(`{session_id}_rf_population_vertex_data.npz` → `{session_id}_population_response_fields.npz`)
and expanded its schema: all nine `InflectionBoundary` fields are now persisted (up from three),
and four new 3D-world (mm) boundary fields are added (`boundary_contour_xyz_{gtype}`,
`boundary_centroid_xyz_{gtype}`, `boundary_perimeter_xyz_mm_{gtype}`,
`boundary_area_xyz_mm2_{gtype}`) via barycentric interpolation on the SLIM mesh.
