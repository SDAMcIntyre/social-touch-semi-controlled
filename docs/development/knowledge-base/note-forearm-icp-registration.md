# Dev Note: ICP Registration Constraints for Forearm Point Clouds

**Problem class:** Aligning multiple forearm point cloud snapshots to a common
reference frame for spatially consistent cross-block analysis.

| Field | Value |
|-------|-------|
| Introduced in | `forearm_extraction/registration/forearm_registrator.py` |
| Open3D version | 0.19.0 |
| Platform | Windows / WSL2 |
| Related plan | [multi-snapshot-forearm-registration.md](../plans/active/multi-snapshot-forearm-registration.md) |

---

## 1. Context

When a neuron session uses multiple forearm reference snapshots (because the
participant's arm shifted on the armrest), each block's somatosensory contact
data (`contact_location_x/y/z`, `contact_points`) lives in a different
coordinate frame.  Registration aligns all forearm point clouds to a single
canonical frame so that downstream analyses (e.g., receptive field
determination) produce spatially coherent results.

---

## 2. Why Point-to-Plane ICP

Point-to-plane ICP is a good fit for this problem because all three
preconditions are naturally satisfied:

1. **Small initial displacement** — the forearm stays on the same armrest and
   is captured by the same Kinect sensor across snapshots.  Shifts are
   translational on the order of a few centimetres with minimal rotation.
   This eliminates the need for a global registration step (FPFH + RANSAC).
2. **Normals already available** — the forearm PLY files exported during
   manual extraction include per-vertex normals, which point-to-plane ICP
   requires.  No additional normal estimation is needed.
3. **Open3D already a project dependency** — `o3d.pipelines.registration.registration_icp`
   with `TransformationEstimationPointToPlane` is available without adding
   any new packages.

---

## 3. Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `max_correspondence_distance` | `0.10` (10 cm) | Wide threshold accommodates the full range of forearm shifts seen in practice; point-to-plane ICP converges correctly even with a loose initial threshold |
| `max_iteration` | `200` | Conservative upper bound; convergence typically occurs within 50-80 iterations for these cloud sizes |
| `voxel_size` | `0.002` (2 mm) | Applied when downsampling the unified (merged) cloud to control output density; not used during pairwise registration |

> **Note:** An earlier version of this note listed `max_correspondence_distance = 0.01` (1 cm).
> The actual code default has always been `0.10` (10 cm); the note was incorrect.
> 1 cm is unnecessarily tight for forearm-scale shifts and risks stalling convergence
> when the arm has moved more than a centimetre between snapshots.

---

## 4. Canonical Reference Convention

The forearm snapshot with the **lowest `representative_frame_id`** is held
fixed (identity transform).  All other snapshots are registered to it.

This convention is consistent with the `remap_lowest_to_zero` pattern used
throughout the pipeline for ordering and normalising frame indices.  It
ensures deterministic behaviour: the canonical reference does not change if
snapshots are added or removed at the high end.

---

## 5. Alternative Registration Methods

Six registration methods are available via the ``method`` parameter of
``register_all()`` / ``register_all_to_average()``, and the
``registration_method`` parameter of ``register_session_forearms()``.

| Method key | Description | When to use |
|------------|-------------|-------------|
| `"vanilla"` *(default)* | Point-to-plane ICP, equal weights | Small shifts, high overlap (> 95 %) |
| `"robust"` | Point-to-plane ICP + Tukey kernel | Fringe artefacts bias vanilla result |
| `"multiscale"` | Coarse-to-fine ICP at 3 scales | Larger shifts or multi-cm displacement |
| `"generalized"` | Generalized ICP (GICP) | Dense, noisy clouds; uncertain surface normals |
| `"trimmed"` | Pre-filter fringe points by NN distance, then ICP | Known partial overlap (arm shift exposes different side) |
| `"global"` | FPFH + RANSAC global, then ICP refinement | Large displacement where identity init fails |

### TukeyLoss parameter guidance

`register_robust()` accepts a `kernel_k` parameter (default `0.01`, i.e. 1 cm).
Residuals beyond ~3k are effectively zeroed out.

- If the cloud has many fringe points and fitness is systematically low, try
  `kernel_k=0.02` to widen the inlier zone.
- If surfaces are very close and you want aggressive outlier rejection, try
  `kernel_k=0.005`.

### Multiscale resolution levels

``register_multiscale()`` uses three fixed levels:
``[(voxel=0.01, max_dist=0.05), (voxel=0.005, max_dist=0.02), (voxel=0.002, max_dist=0.01)]``.
Each level starts from the transform returned by the previous level.

### FPFH+RANSAC note

``register_global_then_local()`` was previously rejected as overkill for small
shifts (see section 2).  It is now implemented as an *optional* method for
edge cases where identity initialisation fails, not as the default path.

---

## 6. Fitness Threshold

A fitness score of **0.9** is the warning threshold.  Fitness is the fraction
of source points that have a correspondence within `max_correspondence_distance`
in the target cloud.

- **>= 0.9** — registration is considered reliable; proceed normally.
- **< 0.9** — a warning is logged.  Common causes: large arm shift between
  snapshots, significant occlusion differences, or noisy edge regions.

If fitness is consistently low across sessions, use
``registration_method="global"`` to enable FPFH + RANSAC coarse alignment
before ICP refinement (see section 5 for details).

---

## 7. Data Flow

Registration and transformation are split across two pipeline stages:

```
Manual extraction pipeline                 Automated pipeline (per block)
─────────────────────────                  ──────────────────────────────
preprocess_pipeline_extract_               preprocess_workflow_kinect_
  forearm_manual.py                          auto.py

  ┌─────────────────────┐                   ┌──────────────────────────┐
  │ Extract forearm PLYs│                   │ compute_somatosensory_   │
  │ (per snapshot)      │                   │   characteristics        │
  └────────┬────────────┘                   └────────────┬─────────────┘
           │                                             │
  ┌────────▼────────────┐                   ┌────────────▼─────────────┐
  │ Register all PLYs   │──── produces ───▶ │ transform_to_registered_ │
  │ (ICP pairwise)      │   PLY + JSON      │   frame                  │
  └────────┬────────────┘                   │ (applies pre-computed    │
           │                                │  4x4 to contact columns) │
  Outputs:                                  └────────────┬─────────────┘
  • {session}_unified_registered.ply                     │
  • {session}_registration_transforms.json    Outputs:   │
                                              • {block}_unified_registered.csv
```

**Manual extraction** runs registration once per session, producing:
- A unified registered PLY (all snapshots aligned and voxel-downsampled)
- A transforms JSON mapping each `representative_frame_id` to its 4x4 rigid
  transform and fitness score, plus a `"parameters"` block recording the ICP
  settings used:

```json
{
    "mode": "reference",
    "canonical_key": "video_stem:42",
    "parameters": {
        "registration_method": "vanilla",
        "max_correspondence_distance": 0.10,
        "icp_max_iteration": 200
    },
    "transforms": {
        "video_stem:42": { "matrix_4x4": [[...]], "fitness": 1.0 }
    }
}
```

The `"parameters"` key is present in all registrations run after
`workbench-driven-registration` was implemented (2026-03-05).  Earlier
artifacts without the key continue to load correctly via
`ForearmRegistrator.load_transforms` (additive schema change).

**Automated pipeline** reads the pre-computed transforms and applies them
per block to the somatosensory CSV's spatial columns.

---

## 8. No-Op for Single-Forearm Sessions

Both the registration step and the per-block transformation step gracefully
skip when only one forearm snapshot exists:

- **Registration** (`register_session_forearms`) — detects a single PLY,
  logs a skip message, produces no artifacts.
- **Transformation** (`transform_to_registered_frame`) — detects no
  transforms JSON on disk, passes the original somatosensory CSV path
  through without modification.

This ensures backward compatibility: existing single-forearm sessions
continue to work identically with no additional configuration.

---

## 9. Reusable Pattern

When adding ICP-based registration to a new point cloud alignment task in
this codebase, use this checklist:

- [ ] **Verify normals exist** in the source PLY files.  Point-to-plane ICP
  requires normals on the target cloud at minimum; having them on both clouds
  improves convergence.
- [ ] **Set `max_correspondence_distance`** based on expected displacement
  magnitude.  Too large admits false correspondences; too small misses valid
  ones during early iterations.
- [ ] **Choose a deterministic canonical reference** (e.g., lowest frame ID)
  rather than selecting the "best" cloud by some metric — determinism
  simplifies debugging and ensures reproducible transforms.
- [ ] **Persist transforms to disk** rather than re-computing them.
  Registration is expensive relative to applying a 4x4 matrix; splitting
  compute-once from apply-many avoids redundant work.
- [ ] **Log fitness scores** and set a warning threshold.  Silent convergence
  failures produce subtly wrong spatial data that is difficult to diagnose
  downstream.
- [ ] **Handle the single-input no-op** explicitly.  Registration with one
  cloud is undefined; skip it cleanly and document the skip path.

---

## 10. References

| Document | Location |
|----------|----------|
| Feature plan | [multi-snapshot-forearm-registration.md](../plans/active/multi-snapshot-forearm-registration.md) |
| Registration module | `code/src/preprocessing/forearm_extraction/registration/forearm_registrator.py` |
| CSV transformer | `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py` |
| Manual extraction host | `code/scripts/preprocess_pipeline_extract_forearm_manual.py` |
| Automated pipeline host | `code/scripts/preprocess_workflow_kinect_auto.py` |
| Downstream consumer | `code/scripts/_5_postprocessing/determine_receptive_field.py` |
