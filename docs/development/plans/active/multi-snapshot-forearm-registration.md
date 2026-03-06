# Plan: Multi-Snapshot Forearm Registration

**Date:** 2026-03-04
**Author:** Basil Duvernoy
**Status:** Active
**Branch:** `feature/multi-snapshot-forearm-registration`

---

## Overview

When a neuron session uses multiple forearm reference snapshots (because the participant's
arm shifted on the armrest over time), the spatial contact data computed per block is
expressed in different coordinate frames. This feature adds a session-level registration
step that aligns all forearm point cloud snapshots to a common reference frame, produces
a single unified forearm point cloud, and transforms the affected per-block spatial fields
so that the compiled neuron data has consistent 3D coordinates.

## Problem Statement

The forearm frame-averaging feature introduced support for multiple forearm reference
snapshots per session (different `representative_frame_id` values). Each block's
somatosensory characteristics (`contact_location_x/y/z`, `contact_points`) are computed
relative to the forearm mesh that was active for that block. When different blocks use
different forearm snapshots, these fields live in different coordinate frames. Simply
concatenating blocks during session aggregation produces spatially inconsistent data —
the same physical skin location gets different 3D coordinates depending on which forearm
snapshot was active. This breaks cross-block analyses such as receptive field determination.

## Goals

### In Scope

1. Register all forearm point cloud snapshots of a session to a common reference frame
   using point-to-plane ICP
2. Produce a single unified forearm point cloud per session, saved as a PLY file
3. Persist per-snapshot 4x4 rigid transform matrices in a dedicated file so that any
   data referenced to an individual forearm can be transformed to the unified frame
4. Extend the forearm data model (`ForearmParameters`, `ForearmCatalog`) to load both
   individual forearm snapshots and the unified registered cloud
5. Transform `contact_location_x/y/z` and `contact_points` in each block's somatosensory
   CSV to the common frame via a per-block flow in the automated pipeline
6. Integrate registration as the final step of `preprocess_pipeline_extract_forearm_manual.py`
   (produce-and-persist during manual forearm extraction)
7. Integrate spatial transformation as a per-block `@flow` in
   `preprocess_workflow_kinect_auto.py`, after `compute_somatosensory_characteristics`
   and before `unify_processed_data`
8. Ensure backward compatibility: single-forearm sessions are a no-op; original
   per-block CSVs are never modified

### Out of Scope

- Re-running somatosensory characteristics computation against the unified forearm
- GPU-accelerated registration
- Global registration (FPFH + RANSAC) as the *default* path — implemented as the optional ``"global"`` method in ``ForearmRegistrator.register_global_then_local()``
- Transforming sticker XYZ, contact depth, or contact area (these are unaffected)

## Success Criteria

- [x] Sessions with multiple forearm snapshots produce a unified registered point cloud
      PLY and a per-snapshot transforms file during manual forearm extraction
- [x] `ForearmCatalog` can load both individual forearm clouds and the unified registered
      cloud from the same session directory
- [x] Per-block `_unified_registered.csv` files are produced by the automated pipeline
      with correctly transformed spatial columns
- [x] Sessions with a single forearm snapshot skip registration entirely (no-op)
- [ ] ICP registration fitness > 0.9 on real session data
- [x] `contact_points` re-serialization is round-trip compatible with
      `determine_receptive_field.py:parse_contact_points()`
- [x] Merging pipeline automatically consumes `_unified_registered.csv` when available,
      falling back to `_unified.csv`
- [x] Original per-block CSVs are never modified

---

## Technical Design

### Approach

Use Open3D's point-to-plane ICP registration to align forearm point clouds. The forearm
PLY files already contain normals, and the shifts are small (same sensor viewing the same
arm on an armrest), so ICP converges reliably without a global registration step. The
first forearm snapshot (lowest `representative_frame_id`) serves as the canonical
reference, consistent with the existing `remap_lowest_to_zero` convention.

Registration is performed once, during manual forearm extraction, and persists two
artifacts to disk:
1. **Unified registered point cloud** — a single PLY combining all aligned snapshots
2. **Per-snapshot transform file** — a dedicated JSON mapping each
   `representative_frame_id` to its 4x4 rigid transform into the unified frame

The automated pipeline later consumes these artifacts: after somatosensory characteristics
are computed for a block, the relevant transform is loaded from the persisted file and
applied to the block's contact columns, producing a `_unified_registered.csv`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Point-to-plane ICP (Open3D)** | Already a dependency; normals available; robust for small shifts | Needs reasonable initial alignment | **Chosen** — initial alignment is inherent (same sensor, same arm) |
| **Sticker-landmark alignment** | Explicit correspondences; fast | Stickers in camera space not forearm space; variable visibility per block | Rejected |
| **FPFH + RANSAC global registration** | Works without initial alignment | Overkill for small shifts; much slower | Rejected (optional fallback in future) |

### Architecture Changes

New subpackage for registration logic:

```
code/src/preprocessing/forearm_extraction/registration/
    __init__.py
    forearm_registrator.py      -- ForearmRegistrator class (ICP logic)
    csv_spatial_transformer.py  -- transform_unified_csv() function
```

Extended data model:

```
code/src/preprocessing/forearm_extraction/models/forearm_catalog.py   -- new method: get_unified_pointcloud()
code/src/preprocessing/forearm_extraction/models/forearm_parameters.py -- (if needed for unified cloud metadata)
code/src/preprocessing/forearm_extraction/data_access/forearm_frame_parameters_filehandler.py -- load/save unified cloud reference
```

Integration points:
- `code/scripts/preprocess_pipeline_extract_forearm_manual.py` — registration as final
  step of `run_session()`, producing the unified PLY and transform file
- `code/scripts/preprocess_workflow_kinect_auto.py` — new per-block `@flow`
  `transform_to_registered_frame` after `compute_somatosensory_characteristics`
- `configs/preprocess_workflow_kinect_auto_dag.yaml` — new DAG task entry
  `transform_to_registered_frame` between `compute_somatosensory_characteristics` and
  `unify_processed_data`
- `code/scripts/merging_pipeline_neuron_to_kinect_auto.py` — prefer registered CSV in
  `resolve_filenames()`

### Persisted Artifacts

Saved to `{session_processed_path}/forearm_pointclouds/` during manual forearm extraction:

| Artifact | Filename Pattern | Content |
|----------|-----------------|---------|
| Unified registered point cloud | `{session_id}_unified_registered.ply` | All forearm snapshots aligned and merged (voxel-downsampled) |
| Per-snapshot transforms | `{session_id}_registration_transforms.json` | JSON mapping `representative_frame_id` to 4x4 transform matrix, plus fitness scores and canonical reference ID |

### Affected Data Fields

| Field | Why affected | Source |
|-------|-------------|--------|
| `contact_location_x` | Centroid X of contacting forearm triangles | `objects_interaction_processor.py:181` |
| `contact_location_y` | Centroid Y of contacting forearm triangles | `objects_interaction_processor.py:182` |
| `contact_location_z` | Centroid Z of contacting forearm triangles | `objects_interaction_processor.py:183` |
| `contact_points` | Raw 3D forearm mesh vertices in contact | `objects_interaction_processor.py:174,178` |

**Unaffected:** `contact_depth`, `contact_area`, `contact_detected` (scalar/relative),
sticker XYZ (camera space), `contact_normals` (visualization only, not persisted).

---

## Implementation Plan

### Phase 1: Core Registration Module

**Goal:** Implement ICP-based forearm registration as a standalone, testable module.

- [x] Task 1.1 — Create `ForearmRegistrator` class in `forearm_registrator.py`:
  constructor takes canonical `o3d.geometry.PointCloud`; `register()` runs point-to-plane
  ICP returning 4x4 transform + fitness; `register_all()` registers multiple clouds;
  `build_unified_cloud()` concatenates transformed clouds and voxel-downsamples
- [x] Task 1.2 — Create `transform_unified_csv()` in `csv_spatial_transformer.py`:
  reads somatosensory CSV, applies 4x4 rigid transform to `contact_location_x/y/z`,
  parses and transforms `contact_points` string column, writes
  `_unified_registered.csv`. Reuse `parse_contact_points` pattern from
  `determine_receptive_field.py:50`
- [x] Task 1.3 — Create `__init__.py` for `registration/` subpackage with exports

**Files Created:**
- `code/src/preprocessing/forearm_extraction/registration/__init__.py`
- `code/src/preprocessing/forearm_extraction/registration/forearm_registrator.py`
- `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py`

**Dependencies:** None

### Phase 2: Data Model Extension

**Goal:** Extend the forearm data model to expose both individual and unified forearms.

- [x] Task 2.1 — Extend `ForearmCatalog` with a `get_unified_pointcloud()` method that
  loads the `{session_id}_unified_registered.ply` from the pointclouds directory.
  Returns `None` if the file does not exist (single-forearm session / not yet registered)
- [x] Task 2.2 — Add a `load_registration_transforms()` utility (in the catalog or a
  dedicated loader) that reads `{session_id}_registration_transforms.json` and returns
  the per-snapshot transform mapping. Returns `None` if the file does not exist
- [x] Task 2.3 — Update `ForearmFrameParametersFileHandler` or the metadata JSON schema
  if needed to reference the unified cloud (e.g., a session-level field). Ensure
  backward compatibility — existing metadata files without registration data load
  without error
- [x] Task 2.4 — Update `__init__.py` exports for any new public API

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/models/forearm_catalog.py`
- `code/src/preprocessing/forearm_extraction/data_access/forearm_frame_parameters_filehandler.py` (if needed)
- `code/src/preprocessing/forearm_extraction/__init__.py`

**Dependencies:** None (parallel with Phase 1)

### Phase 3: Manual Extraction Integration

**Goal:** Add registration as the final step of `preprocess_pipeline_extract_forearm_manual.py`.

- [x] Task 3.1 — Add a `register_session_forearms()` function that: loads all extracted
  forearm PLY files for the session, skips if single forearm (no-op), selects canonical
  (lowest `representative_frame_id`), runs `ForearmRegistrator.register_all()`, saves
  the unified registered point cloud PLY, and saves the per-snapshot transforms JSON
- [x] Task 3.2 — Call `register_session_forearms()` at the end of `run_session()`, after
  `_execute_all_batches()` completes. This is a produce-and-persist step — the unified
  cloud and transform file are written to `{session_processed_path}/forearm_pointclouds/`

**Files Modified:**
- `code/scripts/preprocess_pipeline_extract_forearm_manual.py` — new function + call at
  end of `run_session()`

**Dependencies:** Phase 1, Phase 2

### Phase 4: Automated Pipeline Integration

**Goal:** Add a per-block `@flow` in the automated pipeline that applies pre-computed
transforms to somatosensory contact data.

- [x] Task 4.1 — Create `@flow` function `transform_to_registered_frame` in
  `preprocess_workflow_kinect_auto.py` that: loads the registration transforms from disk
  (via the catalog/loader from Phase 2), determines which forearm snapshot was used for
  this block, applies the corresponding 4x4 transform to the somatosensory CSV using
  `transform_unified_csv()` from Phase 1, and returns the path to the transformed CSV.
  If no transforms file exists (single forearm), passes the original path through as-is
- [x] Task 4.2 — Wire the new flow into `pipeline_stages` in
  `run_single_session_pipeline()`: insert `transform_to_registered_frame` after
  `compute_somatosensory_characteristics`, consuming `somatosensory_chars_path` and
  producing `registered_somatosensory_path`. Update the `unify_processed_data` stage
  to consume `registered_somatosensory_path` instead of `somatosensory_chars_path`
- [x] Task 4.3 — Add `transform_to_registered_frame` task to
  `configs/preprocess_workflow_kinect_auto_dag.yaml`:
  `depends_on: [compute_somatosensory_characteristics]`; update `unify_processed_data`
  to depend on `transform_to_registered_frame` instead of
  `compute_somatosensory_characteristics`

**Files Modified:**
- `code/scripts/preprocess_workflow_kinect_auto.py` — new `@flow` + pipeline stage
- `configs/preprocess_workflow_kinect_auto_dag.yaml` — new task entry + updated
  dependency for `unify_processed_data`

**Dependencies:** Phase 1, Phase 2

### Phase 5: Downstream Integration

**Goal:** Ensure downstream pipelines consume registered data when available.

- [x] Task 5.1 — Update `merging_pipeline_neuron_to_kinect_auto.py` `resolve_filenames()`
  to prefer `_unified_registered.csv` over `_unified.csv` when the registered version
  exists (backward-compatible fallback)

**Files Modified:**
- `code/scripts/merging_pipeline_neuron_to_kinect_auto.py` — prefer registered CSV

**Dependencies:** Phase 4

### Phase 6: Documentation

**Goal:** Document the feature and capture any non-obvious constraints.

- [x] Task 6.1 — Move this plan to `active/` and add inline docstrings to new modules
- [ ] Task 6.2 — Add knowledge-base note if ICP convergence constraints or other
  non-obvious patterns emerge during implementation

**Dependencies:** Phase 5

---

## Testing Plan

### Unit Tests
- [ ] `ForearmRegistrator.register()` with two clouds offset by a known rigid transform
  — verify recovered transform matches ground truth within tolerance
- [ ] `ForearmRegistrator.register()` with identical cloud — verify identity transform
- [ ] `transform_unified_csv()` with identity transform — output equals input
- [ ] `transform_unified_csv()` with known transform — `contact_location_x/y/z`
  correctly transformed
- [ ] `contact_points` re-serialization round-trip compatible with
  `determine_receptive_field.py:parse_contact_points()`
- [ ] `ForearmCatalog.get_unified_pointcloud()` returns cloud when PLY exists, `None`
  when it does not
- [ ] `load_registration_transforms()` returns mapping when JSON exists, `None` when not

### Integration Tests
- [ ] Run manual extraction on session with 2+ snapshots — unified PLY and transforms
  JSON written to disk
- [ ] Run automated pipeline on same session — `_unified_registered.csv` produced per
  block with correctly transformed spatial columns
- [ ] Merging pipeline picks up `_unified_registered.csv` when it exists
- [ ] Single-forearm session produces no registration artifacts (no-op in both scripts)

### Manual Verification
- [ ] Visually compare unified registered point cloud with individual forearm clouds
  in Open3D viewer
- [ ] Receptive field heatmaps from `determine_receptive_field.py` spatially coherent
  across blocks after registration

### Edge Cases
- [ ] Session with only 1 forearm snapshot — no-op, pipeline proceeds as before
- [ ] Block using fallback forearm (no direct match) — correct transform applied
- [ ] Block with all-NaN contact columns — transformer handles gracefully
- [ ] `contact_points` column containing `"[]"` — preserved as-is
- [ ] Existing metadata JSON without registration fields — loads without error
  (backward compatibility)

---

## Documentation Plan

- [x] Add inline docstrings to all new modules
- [ ] Add knowledge-base note if non-obvious constraints emerge (e.g., ICP convergence
  requirements for forearm shapes)

---

## Rollback Plan

All changes are Python scripts, YAML configs, and new files. No data migrations.

1. **Before deployment:** All changes on `feature/multi-snapshot-forearm-registration`
   branch; `main`/`dev` untouched.
2. **Data considerations:** Original per-block CSVs are never modified. Registration
   artifacts (unified PLY, transforms JSON, `_unified_registered.csv`) are purely
   additive.
3. **Rollback procedure:** `git revert` the merge commit. Delete any
   `_unified_registered.csv` files, `{session_id}_unified_registered.ply` files, and
   `{session_id}_registration_transforms.json` files from processed output directories.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| ICP diverges for large forearm shifts | Low | High | Add optional FPFH+RANSAC coarse alignment as fallback; log fitness and warn if < threshold |
| Re-serialized `contact_points` format breaks downstream parser | Medium | Medium | Explicit round-trip test against `parse_contact_points()` from `determine_receptive_field.py` |
| Merging pipeline picks up old `_unified.csv` instead of registered version | Medium | Medium | Prefer-registered fallback logic in `resolve_filenames()` |
| Existing analysis breaks due to changed coordinate frame | Low | High | Originals never modified; only new `_unified_registered.csv` created; downstream consumers opt in |
| Existing forearm metadata JSON missing registration fields | Low | Low | Backward-compatible loading — missing fields default to `None` |

---

## References

- Related Plans: `docs/development/plans/completed/forearm-frame-averaging.md`
- Related Plans: `docs/development/plans/pending/pipeline-dag-validation-fixes.md`
- `code/src/preprocessing/forearm_extraction/models/forearm_catalog.py` — forearm loading
- `code/src/preprocessing/forearm_extraction/models/forearm_parameters.py` — data model
- `code/src/preprocessing/forearm_extraction/data_access/forearm_frame_parameters_filehandler.py` — metadata I/O
- `code/src/preprocessing/motion_analysis/tactile_quantification/model/objects_interaction_processor.py` — contact field computation
- `code/scripts/_5_postprocessing/determine_receptive_field.py` — downstream `contact_points` consumer
- `code/scripts/preprocess_pipeline_extract_forearm_manual.py` — manual forearm extraction (registration host)
- `code/scripts/preprocess_workflow_kinect_auto.py` — automated pipeline (transform flow host)
