# Plan: RF-Centered Coordinate Origin

**Date:** 2026-03-21
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-03-21 18:25
**Branch:** `feature/rf-centered-origin`

---

## Overview

**What:** A new postprocessing task (`center_on_receptive_field`) that translates all spatial data so the coordinate origin sits at the receptive field center of the recorded neuron.

**Why:** The current PCA-calibrated coordinate system is anatomy-based but session-specific in absolute position. Each session records from a different neuron at a different forearm location, making raw spatial coordinates incomparable across sessions. Centering on the RF enables cross-session spatial comparison of contact patterns and neural responses.

**How:** Reuse the existing `RFMappingEngine` (selectivity scoring + DBSCAN clustering) to compute the RF center as the selectivity-weighted centroid of the dominant cluster. Translate all spatial columns and the forearm PLY by that offset. Integrate as a DAG task between `project_contacts_onto_forearm` and `aggregate_session`.

## Problem Statement

- After postprocessing, each session's spatial data is in a PCA-calibrated frame that is anatomically aligned but not neuron-centered
- Cross-session comparison of contact positions relative to the receptive field requires a common spatial reference
- The user's initial approach (forward-fill contact data, histogram at spike times, weighted average) has methodological issues:
  - **Stimulus-response confound:** Naive spike-weighted averaging computes P(position|spike) instead of P(spike|position), biasing the center toward high-stimulus-density regions
  - **Mesh density artifact:** Histogramming mesh-snapped vertices conflates spatial frequency with mesh topology
  - **Temporal smearing:** Forward-filling contact_points freezes position for ~33ms per kinect frame, creating quantization error proportional to stroke velocity
  - **Partial redundancy:** `contact_detected` is already forward-filled during merge (line 159 of `merge_neural_and_kinect_data.py`)
- The codebase already contains a more principled RF mapping approach (`rf_mapping_engine.py`) that corrects for stimulus distribution bias

## Goals

### In Scope
1. Compute per-session RF center using selectivity-weighted DBSCAN cluster centroid (reusing existing `RFMappingEngine`)
2. Translate all spatial columns in block CSVs (`contact_location_x/y/z`, `contact_points`, sticker positions) by the RF center offset
3. Translate the PCA-calibrated forearm PLY by the same offset and save to `forearm_rf_centered/`
4. Write per-session metadata JSON with RF center coordinates, cluster info, and status
5. Integrate as DAG task between `project_contacts_onto_forearm` and `aggregate_session`
6. Graceful fallback when RF estimation fails (no cluster above threshold): pass data through untranslated

### Out of Scope
- Conduction delay correction (already handled during merge synchronization)
- Interpolation of `contact_points` between 30Hz kinect frames (separate concern)
- Modifying the existing RF mapping analysis module
- Rotation or scaling of coordinates — translation only
- GUI changes

## Success Criteria

- [ ] New task `center_on_receptive_field` appears in DAG between `project_contacts_onto_forearm` and `aggregate_session`
- [ ] RF center computed via selectivity-weighted centroid of dominant DBSCAN cluster
- [ ] All `_XYZ_GROUPS` columns and `contact_points` translated by RF center offset in output CSVs
- [ ] Forearm PLY translated and saved to `forearm_rf_centered/`
- [ ] Per-session `rf_center_origin.json` written with center coordinates and cluster metadata
- [ ] `aggregate_session` consumes RF-centered CSVs
- [ ] Sessions where RF estimation fails produce warning log and pass data through unchanged

---

## Technical Design

### Approach

The RF center is estimated using the existing `RFMappingEngine`:

1. Load all block CSVs from `blocks_contact_projected/` via `load_grouped_spatial_data()` (no sub-grouping — single "all" group)
2. Compute selectivity: `spike_count / total_count` per contact point (corrects for stimulus distribution)
3. Run DBSCAN to cluster high-selectivity points
4. Select dominant cluster (largest by point count)
5. RF center = selectivity-weighted centroid: `np.average(cluster.points, weights=cluster.selectivity_scores, axis=0)`

Translation uses the 4x4 rigid transform pattern from `csv_spatial_transformer.py`:
- Build a 4x4 translation matrix from the RF center offset
- Apply to all `_XYZ_GROUPS` columns and `contact_points` using `transform_spatial_columns_in_place()`
- Apply same translation to forearm PLY vertices

**Why this approach:** The selectivity ratio P(spike|position) corrects for non-uniform stimulus distributions that bias naive spike-weighted averaging. DBSCAN provides noise rejection. The 4x4 transform pattern is the established codebase convention for spatial transformations.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Selectivity-weighted DBSCAN centroid | Corrects stimulus bias, reuses existing code, noise-rejected | Requires enough touches for DBSCAN convergence | **Chosen** |
| Naive spike-weighted contact centroid (original proposal) | Simple to implement | Biased by stimulus distribution, conflates mesh density, temporal smearing | Rejected |
| Anatomy-based landmark (forearm centroid) | Neuron-independent, reproducible | Does not achieve RF alignment for cross-session comparison | Rejected |
| Manual RF identification (probe mapping) | Gold standard | Not automated, not available in this dataset | Rejected |

### Architecture Changes

```
code/scripts/_5_postprocessing/
    center_on_receptive_field.py        -- NEW: main task function
code/scripts/_5_postprocessing/__init__.py   -- add export
code/scripts/postprocess_workflow_kinect_auto.py  -- add flow + pipeline stage
configs/postprocess_workflow_kinect_auto_dag.yaml -- add task entry
```

**Reused modules (no modifications):**
- `analysis.receptive_field_mapping.rf_data_loader` — `load_grouped_spatial_data()`
- `analysis.receptive_field_mapping.rf_mapping_engine` — `RFMappingEngine.compute_selectivity()`, `.cluster_receptive_field()`
- `analysis.receptive_field_mapping.rf_mapping_config` — `RFMappingConfig`, `SelectivityDBSCANConfig`, `GroupedSpatialData`
- `preprocessing.forearm_extraction.registration.csv_spatial_transformer` — `parse_contact_points()`, `serialize_contact_points()`, `_XYZ_GROUPS`, `transform_spatial_columns_in_place()`
- `utils.should_process_task` — `should_process_task()`, `clean_task_outputs()`

### Knowledge Base Constraints

- **Units:** All spatial data in mm (from Kinect SDK). No conversion needed. RF center will be in mm.
- **4x4 transform pattern:** Use `csv_spatial_transformer.py` functions for all spatial column translation (established convention).
- **No-op identity check:** If RF center is zero or computation fails, check `np.allclose(T, np.eye(4))` and skip transform.
- **CuPy import order:** Not needed — RF mapping uses sklearn (CPU), no CuPy dependency chain.

---

## Implementation Plan

### Phase 1: Core Task Function
**Goal:** Implement `center_on_receptive_field.py` with RF center computation and spatial translation
**Started:** 2026-03-21
**Completed:** 2026-03-21

- [x] Task 1.1 — Create `center_on_receptive_field.py` with helper `_compute_rf_center(block_csvs, config) -> Tuple[np.ndarray, dict]` that loads grouped spatial data, runs selectivity + DBSCAN, returns RF center and metadata dict
- [x] Task 1.2 — Implement `_translate_single_csv(input_csv, output_csv, translation_matrix)` that applies 4x4 translation to all spatial columns using `transform_spatial_columns_in_place()` pattern
- [x] Task 1.3 — Implement `_translate_forearm_ply(input_ply, output_ply, offset)` using Open3D
- [x] Task 1.4 — Implement main entry `center_on_receptive_field(input_files, forearm_ply_path, output_dir, forearm_output_dir, rf_origin_path, *, force_processing) -> List[Path]`
- [x] Task 1.5 — Handle edge case: no DBSCAN cluster found — log warning, write `status: "no_cluster_found"` in JSON, copy input files unchanged

**Files Modified:**
- `code/scripts/_5_postprocessing/center_on_receptive_field.py` — NEW (~150 lines)

**Dependencies:** None

### Phase 2: Pipeline Integration
**Goal:** Wire the new task into the DAG and workflow runner
**Started:** 2026-03-21
**Completed:** 2026-03-21

- [x] Task 2.1 — Add `from .center_on_receptive_field import center_on_receptive_field` to `code/scripts/_5_postprocessing/__init__.py`
- [x] Task 2.2 — Add `center_on_receptive_field` task entry to `configs/postprocess_workflow_kinect_auto_dag.yaml` with `depends_on: [project_contacts_onto_forearm]`; update `aggregate_session.depends_on` to `[center_on_receptive_field]`
- [x] Task 2.3 — Add `center_on_receptive_field_flow()` Prefect flow wrapper and pipeline stage dict in `code/scripts/postprocess_workflow_kinect_auto.py` (between steps 4 and 5)
- [x] Task 2.4 — Update `aggregate_session` stage params to read from `blocks_rf_centered/` (the new task's output dir)

**Files Modified:**
- `code/scripts/_5_postprocessing/__init__.py` — add 1 import line
- `configs/postprocess_workflow_kinect_auto_dag.yaml` — add task entry (~6 lines), update aggregate depends_on
- `code/scripts/postprocess_workflow_kinect_auto.py` — add flow function (~15 lines), add pipeline stage dict (~15 lines), update aggregate stage input context key

**Dependencies:** Phase 1

---

## Testing Plan

### Manual Verification
- [ ] Run postprocessing workflow on a single session (e.g., ST14-01) with new task enabled
- [ ] Verify `rf_center_origin.json` contains plausible RF center coordinates (within forearm bounding box, typically 10-200mm range per axis)
- [ ] Spot-check output CSVs in `blocks_rf_centered/`: contact_location values should be shifted relative to `blocks_contact_projected/` by exactly the RF center offset
- [ ] Load `forearm_rf_centered/*.ply` in Open3D — verify it is visually translated (origin near center of high-selectivity region)
- [ ] Verify `aggregate_session` produces a valid aggregated CSV from the RF-centered blocks
- [ ] Run with `force_processing: false` — confirm skipping when outputs are up-to-date

### Edge Cases
- [ ] Session with no spikes: should produce `status: "no_cluster_found"`, pass-through data unchanged
- [ ] Session with very sparse contacts (few unique points): DBSCAN may not form clusters — verify graceful fallback
- [ ] Session with multiple DBSCAN clusters: verify largest cluster (by point count) is selected for center computation

---

## Documentation Plan

- [ ] Add knowledge-base note: `docs/development/knowledge-base/note-rf-centered-origin.md` documenting the methodology, why naive spike-weighted averaging was rejected, and the selectivity-weighted approach
- [ ] Update `configs/README.md` with new DAG task description

---

## Rollback Plan

1. Set `center_on_receptive_field: enabled: false` in `postprocess_workflow_kinect_auto_dag.yaml`
2. Revert `aggregate_session.depends_on` to `[project_contacts_onto_forearm]`
3. No data migration needed — original `blocks_contact_projected/` data is never modified; the new task writes to a separate `blocks_rf_centered/` directory

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| DBSCAN fails to cluster for some sessions (too few touches/spikes) | Medium | Medium | Graceful fallback: pass data through untranslated, log warning, write status in JSON |
| Selectivity threshold too aggressive (default 0.3) | Low | Medium | Make threshold configurable via DAG task options; can be tuned per-experiment |
| `load_grouped_spatial_data` expects per-trial CSVs + summary CSV (analysis format), not raw block CSVs | Medium | High | May need to adapt the data loading to work with block-level CSVs directly; verify input format compatibility in Phase 1 |
| `contact_points` parsing performance on large session CSVs | Low | Low | Same parsing already used by RF mapping module at analysis time; known to be adequate |

---

## References

- Related Plans: `docs/development/plans/completed/receptive-field-mapping.md`
- Related Plans: `docs/development/plans/completed/contact-point-forearm-projection.md`
- Knowledge Base: `docs/development/knowledge-base/note-forearm-icp-registration.md` (4x4 transform pattern)
- Knowledge Base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md` (mm units)
- Literature: "Estimating Receptive Fields from Responses to Natural Stimuli with Asymmetric Intensity Distributions" (PLOS ONE 2008) — stimulus bias correction rationale
