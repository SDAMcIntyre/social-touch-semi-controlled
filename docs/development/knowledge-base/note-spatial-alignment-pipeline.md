# Dev Note: Spatial Alignment Pipeline — Forearm Pointclouds & Contact Points

**Problem class:** Tracing the complete spatial data pipeline from raw Kinect
capture through forearm alignment and contact point transformations to
receptive field heatmaps.

| Field | Value |
|-------|-------|
| Scope | End-to-end spatial pipeline (4 phases, 5 coordinate spaces) |
| Related notes | [note-forearm-icp-registration.md](note-forearm-icp-registration.md) (ICP-specific detail) |
| Related notes | [note-somatosensory-units-and-calculations.md](note-somatosensory-units-and-calculations.md) |

---

## 1. Coordinate Space Progression

Every spatial datum passes through up to 5 coordinate spaces, applied sequentially:

| # | Space | Entered By | Units |
|---|-------|-----------|-------|
| 1 | **Kinect native** | Raw Azure Kinect SDK capture | mm |
| 2 | **ICP-registered** | 4x4 rigid transform from `registration_transforms.json` | mm |
| 3 | **PCA-calibrated** | Two-stage PCA rotation (tap Z + stroke XY) | mm |
| 4 | **Projected** | KD-tree nearest-vertex snap onto forearm surface | mm |
| 5 | **RF-centered** | Translation so receptive-field center = origin | mm |

**Columns transformed at each step:**

| Column Group | Columns |
|---|---|
| Contact location | `contact_location_x/y/z` |
| Sticker blue | `sticker_blue_position_x/y/z` |
| Sticker green | `sticker_green_position_x/y/z` |
| Sticker yellow | `sticker_yellow_position_x/y/z` |
| Contact points (multi-valued) | `contact_points` — serialized as `[[x1 y1 z1] [x2 y2 z2] ...]` |

---

## 2. Phase A: Preprocessing

### A1. Forearm Pointcloud Extraction & Registration (Manual DAG)

**DAG**: `configs/preprocess_workflow_kinect_manual_dag.yaml`

```
extract_forearm -> curate_forearm -> clean_forearm -> define_normals -> build_mesh -> register_forearms
```

| Step | Input | Output | Directory |
|---|---|---|---|
| extract through define_normals | Raw depth frames | `{stem}_with_normals.ply` | `forearm_pointclouds/` |
| **register_forearms** | All `*_with_normals.ply` in session | `{session_id}_unified_registered.ply` + `{session_id}_registration_transforms.json` | `forearm_pointclouds/` |

Registration loads per-snapshot PLYs keyed as `"<video_stem>:<frame_id>"`,
selects a canonical reference (lowest frame_id by default), runs ICP (6
methods available), and saves per-snapshot 4x4 transforms + fitness scores.
See [note-forearm-icp-registration.md](note-forearm-icp-registration.md) for
full ICP detail.

**Key files:**
- `code/src/preprocessing/forearm_extraction/registration/forearm_registrator.py`
- `code/src/preprocessing/forearm_extraction/registration/register_session_forearms.py`

### A2. Contact Point Generation (Auto DAG)

**DAG**: `configs/preprocess_workflow_kinect_auto_dag.yaml`
**Task**: `compute_somatosensory_characteristics`

Collision detection between hand mesh and forearm terrain
(`objects_interaction_processor.py`):
- Broad phase: AABB crop using hand bounding box
- Narrow phase: signed-distance raycasting for penetrating vertices

**Produces per frame** (all in Space 1):
- `contact_points` — serialized vertex coords
- `contact_location_x/y/z` — mean of contact vertices
- `contact_area` (mm^2), `contact_depth` (mm), `contact_detected` (0/1)

### A3. Unification

**Task**: `unify_processed_data` — aggregates all per-block streams (stickers,
hand model, contacts, LED sync, trials, single touches) into a single CSV per
block. All data remains in Space 1.

---

## 3. Phase B: Merging (Neural + Kinect)

**DAG**: `configs/merging_pipeline_neuron_to_kinect_auto_dag.yaml`

| Task | What | Output |
|---|---|---|
| `unify_dataset` | Cross-correlates neural spike timestamps with Kinect frames; adds `Nerve_spike` column | `blocks_merged/*.csv` |
| `filter_by_neural_quality` | Filters by xlsx quality annotations | Filtered `blocks_merged/*.csv` |

Coordinate space: Still Space 1. The merge adds `Nerve_spike` but does not
alter spatial values.

---

## 4. Phase C: Postprocessing (Spatial Transform Pipeline)

**DAG**: `configs/postprocess_workflow_kinect_auto_dag.yaml`

```
apply_icp_registration          [Space 1 -> 2]
        |
        v
deduplicate_xy                  [Space 2, filtering only]
        |
        v
project_contacts_onto_forearm   [Space 2, snap to deduped forearm]
        |
        v
set_xyz_reference_from_gestures [Space 2 -> 3]
        |
        v
export_forearm_pca_calibrated   [PLY: Space 2 -> 3]
        |
        v
center_on_receptive_field       [Space 3 -> 4]
        |
        v
aggregate_session               [Space 4, concatenation only]
```

### C0. apply_icp_registration (Space 1 -> 2)

**Script**: `code/scripts/_5_postprocessing/apply_icp_registration.py`

- **Input**: `blocks_merged/*.csv` + `{session_id}_registration_transforms.json`
- **Output**: `blocks_registered/*.csv`
- **Transform**: 4x4 rigid matrices via `apply_rigid_transform(points, T)` = `points @ R.T + t`

Transform key resolution (`csv_spatial_transformer.py`):
- Same block -> earliest snapshot (lowest frame_id)
- Earlier block -> latest snapshot from most recent preceding block
- Before all snapshots -> pass through unchanged
- Multiple snapshots in one block -> rows segmented by `frame_index`

Single-forearm sessions: no transforms JSON; files copied unchanged.

### C1. deduplicate_xy (Space 2, filtering only)

- **Input**: `blocks_registered/*.csv` + `{session_id}_unified_registered.ply`
- **Output**: `blocks_registered_deduped/*.csv` + `forearm_deduped/{session_id}_unified_registered.ply`
- **Transform**: Remove duplicate (x,y) points keeping lowest z (DBSCAN clustering)
- **Forearm**: Single `_unified_registered.ply` deduplicated once for entire session

### C2. project_contacts_onto_forearm (Space 2, snap to surface)

**Script**: `code/scripts/_5_postprocessing/project_contacts_onto_forearm.py`

- **Input**: `blocks_registered_deduped/*.csv` + `forearm_deduped/{session_id}_unified_registered.ply`
- **Output**: `blocks_registered_projected/*.csv` + `projection_stats.csv`
- **Transform**: KD-tree nearest-neighbor snap to forearm vertex (XY distance)
- **Columns**: `contact_points` (snapped) + `contact_location_x/y/z` (recomputed mean)
- **Stats**: mean/median/p95/max displacement in mm

### C3. set_xyz_reference_from_gestures (Space 2 -> 3)

**Script**: `code/scripts/_5_postprocessing/set_xyz_reference_from_gestures.py`

- **Input**: `blocks_registered_projected/*.csv`
- **Output**: `blocks_pca_calibrated/*.csv` + `pca-xyz_transformation-matrices.json`

PCA calibration (`PCACalibrationEngine`):
1. Tapping gestures -> Z-axis (normal to skin)
2. Stroking gestures (post step-1 rotation) -> X-axis (along forearm)

### C4. export_forearm_pca_calibrated (PLY: Space 2 -> 3)

**Script**: `code/scripts/_5_postprocessing/export_forearm_pca_calibrated.py`

- **Input**: `{session_id}_unified_registered.ply` + `pca-xyz_transformation-matrices.json`
- **Output**: `forearm_pca_calibrated/{session_id}_forearm.ply`
- **Fallback**: If no `_unified_registered.ply`, uses any single PLY in `forearm_pointclouds/`

### C5. center_on_receptive_field (Space 3 -> 4)

**Script**: `code/scripts/_5_postprocessing/center_on_receptive_field.py`

- **Input**: `blocks_pca_calibrated/*.csv` + `forearm_pca_calibrated/{session_id}_forearm.ply`
- **Output**: `blocks_rf_centered/*.csv` + `forearm_rf_centered/{session_id}_forearm.ply` + `rf_center_origin.json`
- **Transform**: Pure translation `T[:3,3] = -rf_center`

RF center estimation:
1. Per-point selectivity = spike_count / total_count
2. DBSCAN on high-selectivity points
3. RF center = selectivity-weighted centroid of largest cluster
4. On failure: data passes through unchanged

### C6. aggregate_session

- **Input**: `blocks_rf_centered/*.csv`
- **Output**: `{session_id}_semicontrolled_aggregated_session.csv`

---

## 5. Phase D: Analysis

**DAG**: `configs/analyse_workflow_dag.yaml`

### D1. map_receptive_fields_simple

**Script**: `code/src/analysis/receptive_field_mapping/rf_simple_pipeline.py`

- **Input**: `*_aggregated_session.csv` (Space 5) + `forearm_rf_centered/{session_id}_forearm.ply`
- **Output**: `4_analysed/receptive_field_maps_simple/{session_id}/spike_positions.csv` + heatmap PNG

Process:
1. Forward-fills `contact_points` within each `(block_order_id, trial_id, single_touch_id)` group
2. Filters `Nerve_spike == 1` rows, parses contact_points
3. KD-tree vertex snapping on forearm PLY (aggregates by vertex index)
4. Renders heatmap via `rf_cluster_visualizer.py`

Sanity check: mean snap distance > 2 mm -> fallback to exact-float groupby.

### D2. Clustered RF Mapping

`touch_feature_extraction` -> `touch_clustering` -> `map_receptive_fields_clustered`

Cluster-specific heatmaps using same forearm PLY and contact data, all in
RF-centered space.

---

## 6. File Flow Summary (Per Session)

```
PHASE A (Preprocessing)
  forearm_pointclouds/{stem}_with_normals.ply              [Space 1]
         |  register_forearms
         v
  forearm_pointclouds/{session}_unified_registered.ply     [Space 1, aligned]
  forearm_pointclouds/{session}_registration_transforms.json
  + collision detection -> contact_points in per-block CSVs [Space 1]

PHASE B (Merging)
  blocks_merged/*.csv                                      [Space 1 + Nerve_spike]

PHASE C (Postprocessing)
  C0: blocks_registered/*.csv                              [Space 2]
  C1: blocks_registered_deduped/*.csv                      [Space 2, filtered]
      forearm_deduped/{session}_unified_registered.ply
  C2: blocks_registered_projected/*.csv                    [Space 2, on-surface]
      projection_stats.csv
  C3: blocks_pca_calibrated/*.csv                          [Space 3]
      pca-xyz_transformation-matrices.json
  C4: forearm_pca_calibrated/{session}_forearm.ply         [Space 3]
  C5: blocks_rf_centered/*.csv                             [Space 4]
      forearm_rf_centered/{session}_forearm.ply             [Space 4]
      rf_center_origin.json
  C6: {session}_semicontrolled_aggregated_session.csv      [Space 4]

PHASE D (Analysis)
  spike_positions.csv                                      [Space 5]
  {session}_rf_simple.png                                  [Space 5, rendered]
```

---

## 7. Key Invariants

1. **Transform JSON keys** are `"<video_stem>:<frame_id>"` composites.

2. **Scheduled transforms** (C1) allow multiple transforms within a single
   block CSV. Handled by `transform_spatial_columns_scheduled()`.

3. **contact_points serialization**: `[[x1 y1 z1] [x2 y2 z2]]` with
   space-separated floats. Parser: `parse_contact_points()` using regex
   `\[([^\]]+)\]`. Serializer: `serialize_contact_points()` with `%.1f`.

4. **Forearm PLY fallback** in C3: `_unified_registered.ply` -> any `.ply`.
   Single-forearm sessions skip ICP entirely.

5. **RF-centered PLY resolution** in D1: `resolve_forearm_ply()` looks
   exclusively in `forearm_rf_centered/`.

6. **Forward-fill** in D1: `contact_points` is forward-filled within each
   touch group to bridge the 30 Hz / 1 kHz sampling mismatch.
