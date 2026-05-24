# Plan: Extract Population RF Response-Field Boundaries

**Date:** 2026-05-20
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-24 18:03
**Started:** 2026-05-20
**Base Branch:** `feature/inflection-boundary-extrapolation-snapshots`
**Branch:** `feature/extract-population-rf-response-field-boundaries`

---

## Overview

Rename the `visualize_population_rf_maps` task to `extract_population_rf_response_field_boundaries` and expand its NPZ output so it contains everything needed to reconstruct each per-gesture receptive-field boundary and compute downstream metrics — including the boundary polygon and centroid in **3D world (mm)** coordinates, not just UV space.

## Problem Statement

The DAG task currently named `visualize_population_rf_maps`
(`code/src/analysis/receptive_field_mapping/pipelines/rf_population_map_pipeline.py::run_population_rf_maps`,
DAG key `visualize_population_rf_maps`) no longer just visualises. Since
`feature/population-rf-inflection-boundary` shipped, it also detects a
Laplacian zero-crossing boundary on the interpolated IFF heatmap and computes
the corresponding `InflectionBoundary` (contour polygon, centroid, area,
perimeter, circularity, PCA axes). The name no longer reflects what it does.

The NPZ export added by `feature/population-rf-vertex-data-export` carries
only three of the boundary fields (`contour_uv`, `area_uv`, `centroid_uv`).
The remaining scalar metrics live only in the **sentinel JSON**, which is an
idempotency artefact, not a clean downstream contract. Worse, every boundary
field is in dimensionless UV space — the prior plan explicitly listed
"Converting UV-space metrics to mm-scale" as out of scope. Without 3D-world
versions of the contour and centroid, downstream code cannot compute
physically-meaningful quantities (RF area in mm², RF perimeter in mm,
distance between centres across sessions).

## Goals

### In Scope
1. Rename the DAG task, Prefect flow, pipeline orchestrator function, module
   file, output folder, sentinel file and NPZ file consistently across the
   repo.
2. Persist every `InflectionBoundary` field in the NPZ (not just three).
3. Add 3D-world (mm) coordinates of the boundary contour and the centroid in
   the NPZ, computed by barycentric interpolation on the SLIM mesh.
4. Derive `perimeter` (mm) and `area` (mm²) from the 3D contour and persist
   alongside the UV-space values.
5. Update `code/src/analysis/CLAUDE.md` and the DAG config comments.

### Out of Scope
- Changing the inflection-boundary detection algorithm itself
  (`compute_inflection_boundary()` is unchanged)
- Changing PNG rendering, filenames or colour scale
- Changing the sentinel JSON structure (keys remain identical, just the
  filename is renamed)
- New downstream pipeline consuming the NPZ (separate future plan)
- Backward-compatibility shims for the old name or NPZ filename — call sites
  are updated in this change
- Per-gesture boundaries on the **per-cell grid** pipeline
  (`rf_population_grid_pipeline.py`) — separate concern

## Success Criteria

- [ ] DAG config loads with `ruamel.yaml` round-trip and comments preserved
- [ ] `python -c "from analysis.receptive_field_mapping import run_population_response_field_extraction"` succeeds
- [ ] The old DAG key, flow name, function name and module path all raise on
      import — no shims left
- [ ] Running the renamed task on a session produces, under
      `4_analysed/population_response_fields/<session_id>/`:
  - The same PNGs as before (per-gesture and composites)
  - `<session_id>_population_response_fields.npz` containing every key listed
    in the NPZ schema below
  - `<session_id>_population_response_fields_done.json` sentinel
- [ ] For at least one gesture: `boundary_contour_xyz_{gtype}.shape == (N, 3)`
      where N matches `boundary_contour_uv_{gtype}.shape == (N, 2)`
- [ ] `boundary_perimeter_xyz_mm_{gtype}` is a positive finite scalar in a
      physically sensible range (≈ 20–300 mm for forearm RFs)
- [ ] Reading only the NPZ, the contour and centroid overlay drawn on the UV
      heatmap matches the white polygon in the pipeline-rendered PNG (visual
      check)
- [ ] `pytest code/tests/` passes

---

## Technical Design

### Approach

The renderer (`rf_population_map_renderer.py`) and the boundary algorithm
(`rf_inflection_boundary.py`) stay untouched. All changes live in the
**orchestrator** (`rf_population_map_pipeline.py`, renamed) plus a single new
helper `uv_points_to_xyz()` that maps UV polygon points to 3D mesh-surface
positions via barycentric interpolation on the SLIM mesh. The SLIM cache
already contains both `uv` and `V` per vertex, so the mapping is exact for
any UV point that lies inside any SLIM triangle.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Expand NPZ in-place + new helper for UV→3D | Single output file, minimal new code, all data co-located, downstream can do mm calculations | NPZ mixes vector arrays and scalars (already does) | **Chosen** (matches user instruction "expand existing NPZ") |
| Dedicated `<session>_rf_boundaries.json` only | Lightweight, easy to read from any tooling | Cannot embed heatmap/grid arrays needed for "perfect reconstruction"; duplicates sentinel | Rejected — user explicitly asked for NPZ |
| Both NPZ + JSON | Best of both | More files to maintain; user picked NPZ-only | Rejected |
| Nearest-neighbour UV→3D (no barycentric) | Simpler | Snaps every contour vertex to a mesh vertex → noticeably jagged 3D contour and biased perimeter/area | Rejected — barycentric is exact and uses an existing dependency (`matplotlib.tri.TriFinder` already used by the renderer) |
| 3D-only outputs (drop UV) | Smaller schema | UV is needed to overlay on the existing PNGs and round-trip the white polygon; downstream may want shape metrics in UV | Rejected |

### Architecture Changes

Renaming summary:

| Layer | Current | New |
|---|---|---|
| DAG key | `visualize_population_rf_maps` | `extract_population_rf_response_field_boundaries` |
| Prefect flow | `visualize_population_rf_maps_flow` | `extract_population_rf_response_field_boundaries_flow` |
| Pipeline function | `run_population_rf_maps` | `run_population_response_field_extraction` |
| Pipeline module | `pipelines/rf_population_map_pipeline.py` | `pipelines/rf_population_response_field_pipeline.py` |
| Output folder | `4_analysed/population_rf_maps/<session_id>/` | `4_analysed/population_response_fields/<session_id>/` |
| Sentinel file | `<session_id>_rf_population_maps_done.json` | `<session_id>_population_response_fields_done.json` |
| NPZ file | `<session_id>_rf_population_vertex_data.npz` | `<session_id>_population_response_fields.npz` |
| Log prefix | `[Population RF Maps]` | `[Population Response Fields]` |

The renderer module (`rf_population_map_renderer.py`) is **not** renamed —
it remains a pure visualisation helper.

PNG filenames stay `<session_id>_rf_population_<gtype>.png` to avoid pointless
churn in downstream tooling that may already pattern-match them.

### NPZ schema (expanded)

**Shared arrays (unchanged):**
- `forearm_uv` `(n_V, 2)` float64 — SLIM mesh UV
- `forearm_faces` `(n_F, 3)` int32 — SLIM triangles
- `forearm_V` `(n_V, 3)` float64 — SLIM vertices in mm
- `session_id`, `neuron_mode`, `min_overlap_pct`, `gesture_types`,
  `inflection_sigma`

**Per-gesture heatmap arrays (unchanged):**
- `heatmap_{gtype}`, `n_touches_{gtype}`, `threshold_{gtype}`
- `grid_u_{gtype}`, `grid_v_{gtype}`, `grid_z_{gtype}`

**Per-gesture boundary fields — every `InflectionBoundary` field persisted:**

| NPZ key | Shape / dtype | Source field | Coord space |
|---|---|---|---|
| `boundary_contour_uv_{gtype}` | `(N, 2)` float64 | `contour_uv` | UV |
| `boundary_centroid_uv_{gtype}` | `(2,)` float64 | `centroid_uv` | UV |
| `boundary_perimeter_uv_{gtype}` | scalar float64 | `perimeter_uv` | UV |
| `boundary_area_uv_{gtype}` | scalar float64 | `area_uv` | UV |
| `boundary_circularity_{gtype}` | scalar float64 | `circularity` | dimensionless |
| `boundary_pca_major_uv_{gtype}` | scalar float64 | `pca_major_uv` | UV |
| `boundary_pca_minor_uv_{gtype}` | scalar float64 | `pca_minor_uv` | UV |
| `boundary_pca_orientation_deg_{gtype}` | scalar float64 | `pca_orientation_deg` | degrees |
| `boundary_mean_iff_on_contour_{gtype}` | scalar float64 | `mean_iff_on_contour` | IFF units |

**Per-gesture 3D-world (mm) boundary fields — NEW:**

| NPZ key | Shape / dtype | Description |
|---|---|---|
| `boundary_contour_xyz_{gtype}` | `(N, 3)` float64 | 3D world (mm) per contour vertex — barycentric on SLIM mesh |
| `boundary_centroid_xyz_{gtype}` | `(3,)` float64 | 3D world (mm) centroid (same mapping) |
| `boundary_perimeter_xyz_mm_{gtype}` | scalar float64 | Sum of segment lengths around the closed 3D polygon |
| `boundary_area_xyz_mm2_{gtype}` | scalar float64 | Triangle-fan area from centroid in 3D |

Gestures whose boundary is `None` omit **all** `boundary_*_{gtype}` keys for
that gesture — consistent with current behaviour.

### New helper

```python
# in code/src/analysis/receptive_field_mapping/surface/forearm_slim_uv.py
def uv_points_to_xyz(
    uv_points: np.ndarray,            # (N, 2)
    forearm_uv: np.ndarray,           # (n_V, 2) SLIM UV
    forearm_faces: np.ndarray,        # (n_F, 3) SLIM triangles
    forearm_V: np.ndarray,            # (n_V, 3) SLIM vertices in mm
) -> np.ndarray:                       # (N, 3)
    """Map each UV point to 3D via barycentric interpolation on the SLIM mesh.

    Builds a matplotlib.tri.Triangulation, uses get_trifinder() to locate the
    containing triangle for every UV point, then interpolates forearm_V with
    barycentric weights. Raises ValueError if any UV point lies outside every
    triangle (no silent fallback — pipeline policy).
    """
```

Placement rationale: the helper inverts the SLIM UV map, so it sits next to
`load_slim_uv_cache()` in `surface/forearm_slim_uv.py`. The renderer's
`compute_interpolated_grid()` already uses `matplotlib.tri.Triangulation` —
no new dependency.

### Perimeter / area in mm

Computed inline in the pipeline (small helper functions in the same
module), since they're trivial:

- `perimeter_xyz_mm = np.sum(np.linalg.norm(np.diff(contour_xyz_closed, axis=0), axis=1))`
- `area_xyz_mm2`: triangle-fan from `centroid_xyz`:
  `0.5 * Σ |((v_i − c) × (v_{i+1} − c))|`

Triangle-fan area approximates the area of the closed contour on the
slightly-curved forearm surface. Accurate when the boundary lies in a
near-flat patch — typical for inflection-ring RFs, which are localised.

---

## Implementation Plan

### Phase 1: Helper + NPZ schema expansion
**Goal:** Get the data right before the renaming churn

- [x] Add `uv_points_to_xyz()` to
      `code/src/analysis/receptive_field_mapping/surface/forearm_slim_uv.py`
- [x] Re-export from `surface/__init__.py` if other modules export-by-default
      from there (check first)
- [x] In `rf_population_map_pipeline.py::_save_vertex_data_npz`, persist all
      `InflectionBoundary` scalar fields (not just three) when boundary is
      not `None`
- [x] In the same function, compute `boundary_contour_xyz_{gtype}`,
      `boundary_centroid_xyz_{gtype}`, `boundary_perimeter_xyz_mm_{gtype}`,
      `boundary_area_xyz_mm2_{gtype}` via the new helper + inline length/area
- [ ] Verify the NPZ via the round-trip check in **Manual Verification**

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/surface/forearm_slim_uv.py` — add
  `uv_points_to_xyz()`
- `code/src/analysis/receptive_field_mapping/surface/__init__.py` — re-export
  if needed
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_map_pipeline.py` —
  expand `_save_vertex_data_npz`

**Dependencies:** None

### Phase 2: Rename
**Goal:** Make the names match what the task does

- [x] Rename file → `pipelines/rf_population_response_field_pipeline.py`
- [x] Rename `run_population_rf_maps` → `run_population_response_field_extraction`
- [x] Rename `_save_vertex_data_npz` → `_save_response_fields_npz`
- [x] Update sentinel filename → `<session_id>_population_response_fields_done.json`
- [x] Update NPZ filename → `<session_id>_population_response_fields.npz`
- [x] Update output folder → `4_analysed/population_response_fields/<session_id>/`
- [x] Update log prefix → `[Population Response Fields]`
- [x] Update `__init__.py` exports + `__all__`
- [x] Rename flow `visualize_population_rf_maps_flow` →
      `extract_population_rf_response_field_boundaries_flow` and its
      `@flow(name=...)` in `code/scripts/analysis_workflow_processing.py`
- [x] Update the dispatch-table entry around line 1097 to use the new DAG
      key (`extract_population_rf_response_field_boundaries`) including all
      `dag_handler.get_task_options(...)` lookups
- [x] Rename DAG key in `configs/analyse_workflow_processing_dag.yaml:133`
      and update the comment block lines 129–132
- [x] Grep for any other config or task whose `depends_on:` mentions the old
      DAG key (none found at planning time, but verify before merging)
- [x] Update docstrings at `analysis_workflow_viewers.py:214–215` and
      `rf_cluster_gui_launchers.py:357`

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/pipelines/rf_population_map_pipeline.py` →
  renamed
- `code/src/analysis/receptive_field_mapping/__init__.py` — update exports
- `code/scripts/analysis_workflow_processing.py` — rename flow, update
  dispatch table
- `code/scripts/analysis_workflow_viewers.py` — docstring updates
- `code/src/analysis/receptive_field_mapping/pipelines/rf_cluster_gui_launchers.py` —
  docstring update
- `configs/analyse_workflow_processing_dag.yaml` — rename DAG key and comment

**Dependencies:** Phase 1

### Phase 3: Documentation
**Goal:** Keep the docs in sync

- [x] Update `code/src/analysis/CLAUDE.md` — the "Population RF maps" bullet
      and the orchestration paragraph listing DAG task names
- [x] Update `docs/development/plans/active/population-rf-inflection-boundary.md`
      and `docs/development/plans/active/population-rf-vertex-data-export.md`
      with a short "follow-on" note referencing this plan (so the lineage is
      discoverable)

**Files Modified:**
- `code/src/analysis/CLAUDE.md`
- Cross-references in the two active plan docs

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [x] `uv_points_to_xyz()` on a synthetic mesh (e.g. unit square, 2
      triangles) — verify barycentric interpolation matches a hand-computed
      reference for a known UV point
- [x] `uv_points_to_xyz()` raises `ValueError` for a UV point outside every
      triangle (no silent fallback)

### Integration Tests
- [ ] No new ones — exercise via the manual end-to-end run below;
      `pytest code/tests/` must still pass unchanged after the rename

### Manual Verification
- [ ] In the DAG config: enable
      `extract_population_rf_response_field_boundaries.enabled: true` with
      `force_processing: true`, then run
      `python code/scripts/launch_pipeline_gui.py` and pick a session whose
      prerequisites (`map_single_touch_rf`, `set_rf_camera_settings`,
      `precompute_forearm_slim_uv`) are complete
- [ ] Confirm `4_analysed/population_response_fields/<session_id>/` is
      created with PNGs + NPZ + sentinel JSON
- [ ] Open the NPZ with `np.load(...)` and check every key in the schema
      table is present for at least one gesture with a boundary
- [ ] Round-trip overlay check:
      ```python
      import numpy as np, matplotlib.pyplot as plt
      d = np.load(npz_path, allow_pickle=True)
      gt = 'all'
      plt.scatter(d['forearm_uv'][:,0], d['forearm_uv'][:,1],
                  c=d[f'heatmap_{gt}'], s=1)
      plt.plot(d[f'boundary_contour_uv_{gt}'][:,0],
               d[f'boundary_contour_uv_{gt}'][:,1], 'w-')
      plt.scatter(*d[f'boundary_centroid_uv_{gt}'], c='r', s=40)
      plt.savefig('roundtrip_check.png')
      ```
      The overlay must coincide with the white polygon in the pipeline PNG
- [ ] Sanity-check `boundary_perimeter_xyz_mm_{gtype}` falls in 20–300 mm
      and `boundary_area_xyz_mm2_{gtype}` is positive

### Edge Cases
- [ ] Gesture with `boundary = None` (e.g. detection fails) — NPZ omits all
      `boundary_*_{gtype}` keys for that gesture, sentinel records `null`
- [ ] `inflection_sigma: null` in DAG — no boundaries computed; NPZ contains
      only the shared + heatmap fields; PNGs render without overlays
- [ ] Contour with very few vertices (e.g. N < 4) — `uv_points_to_xyz()`
      still works on a small array

---

## Documentation Plan

- [x] Update `code/src/analysis/CLAUDE.md` — Population RF maps bullet and
      DAG task list
- [ ] Update the DAG YAML comment block above the renamed task to describe
      the dual visualisation + boundary-extraction output
- [x] Cross-reference from the two prior plan docs in `active/`
- [ ] No changelog file convention exists for this repo (per recent merges);
      skipping

---

## Rollback Plan

1. **Before merging the feature branch:**
   - Revert all commits on `feature/extract-population-rf-response-field-boundaries`
   - The two prior shipped artefacts (boundary detection + initial NPZ
     export) are on the base branches, so they remain functional

2. **After merging but before downstream consumers depend on the new names:**
   - Single revert commit of the merge; downstream still works because the
     old code is reinstated wholesale

3. **After downstream consumers depend on the new NPZ schema:**
   - Roll forward — re-introduce only the schema fields that consumers use,
     drop the renaming if it caused the regression

Data considerations: no migrations, no breaking changes to upstream stages.
Re-running the renamed task regenerates outputs deterministically.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `uv_points_to_xyz()` returns slightly different 3D positions than `slim_V[nearest_orig_for_slim]` (the existing nearest-vertex mapping) | Med | Low | Document that the new mapping is barycentric (sub-vertex precision) and intentionally more accurate; sanity-check on a synthetic mesh |
| Old DAG key still referenced somewhere not caught by grep (e.g. in a notebook, in `configs/launcher.yaml`) | Low | Med | Final grep before merging the rename commit; also run the pipeline end-to-end to surface missing wiring |
| Triangle-fan area underestimates area for non-convex or strongly curved contours | Low | Low | Inflection-ring contours are near-convex by construction (Laplacian zero-crossing around a peak); the prior `area_uv` uses the same shoelace approach |
| `ruamel.yaml` round-trip breaks DAG comments during the rename | Low | Med | Edit YAML manually with a precise `Edit` (not programmatic dump) so original formatting is preserved |
| User had a typo and "rf_response_field" should be just "rf_boundaries" | Low | Med | The exact name was confirmed via AskUserQuestion in the planning conversation; surface during review if questioned |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Helper + NPZ schema | ~2 h | None |
| Phase 2: Rename | ~1 h | Phase 1 |
| Phase 3: Documentation | ~30 min | Phase 2 |

---

## References

- Related plan (boundary detection, shipped): `docs/development/plans/active/population-rf-inflection-boundary.md`
- Related plan (initial NPZ export, shipped): `docs/development/plans/active/population-rf-vertex-data-export.md`
- Source-of-truth dataclass: `code/src/analysis/receptive_field_mapping/metrics/rf_inflection_boundary.py::InflectionBoundary`
- SLIM UV cache loader: `code/src/analysis/receptive_field_mapping/surface/forearm_slim_uv.py::load_slim_uv_cache`
- Conversation scratch plan (research notes): `C:\Users\basil\.claude\plans\analyse-the-visualise-population-hazy-phoenix.md`
