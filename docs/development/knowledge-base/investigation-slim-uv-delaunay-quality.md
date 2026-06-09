# Investigation: SLIM UV Delaunay quality vs. preprocessing Delaunay quality

**Date:** 2026-05-22
**Author:** Basil Duvernoy
**Status:** Open — Q1 answered, revised hypotheses

---

## Problem

The SLIM UV pipeline with `mesh_method: delaunay` produces a poor-quality
triangulation, while the preprocessing forearm extraction pipeline uses the
same 2.5D Delaunay approach and produces a high-quality mesh.

This investigation documents the differences found between the two systems
and the open questions to answer before fixing the issue.

---

## What was investigated

Both systems use `scipy.spatial.Delaunay` with an XY projection. The
triangulation algorithm itself is identical. The differences are below.

---

## Differences between the two systems

### 1. Edge filtering

| System | Filter |
|--------|--------|
| **Preprocessing** | **None.** All convex-hull triangles are kept, including elongated boundary ones. |
| **SLIM UV** | `max_edge_mm = 3.0 × avg_nn`, where `avg_nn` = mean 1-NN distance of the whole cloud (computed via `pcd.compute_nearest_neighbor_distance()`). |

The `3.0 × avg_nn` formula assumes the input point cloud is **uniformly
sampled**. With uniform 5 mm input (see §2 below), `avg_nn ≈ 5 mm` →
`max_edge_mm ≈ 15 mm`. This should accommodate most internal edges
(`√2 × 5 ≈ 7 mm`), but curvature inflates 3D distances (see §5 below).

**Code location:** `rf_surface_utils.py::_build_and_cache_delaunay()`.

### 2. Input point distribution

| System | Input |
|--------|-------|
| **Preprocessing** | Raw Kinect depth cloud voxel-downsampled to a **uniform 5 mm grid** (`ArmSegmentation`, leaf = 5 mm). Homogeneous density by construction. |
| **SLIM UV** | `{session_id}_forearm.ply` loaded directly with **no additional downsampling**. See Q1 answer below — this PLY **is** the voxel-downsampled point cloud (uniform 5 mm), not mesh vertices. |

**Chain of custody for `*_forearm.ply`:**

1. Preprocessing: Kinect depth → `voxel_down_sample(leaf_size=5.0)` → uniform 5 mm
   (`arm_segmentation.py:155`)
2. Aggregation: `shutil.copy2()` — bit-for-bit copy, no reprocessing
   (`aggregate_blocks_session.py:83`)
3. Postprocessing: PCA rotation (rigid) + RF translation (rigid) — no resampling
   (`center_on_receptive_field.py:180–196`)
4. SLIM UV: loaded directly via Open3D (`rf_data_loader.py:30–63`)

**Input density is approximately uniform at both systems.** The edge-filtering
difference (§1) and the 3D-vs-2D edge measurement issue (§5) are the primary
suspects, not the point distribution.

### 3. Coordinate space

RF-centering is **translation-only** (no rotation). Forearm orientation is
unchanged from Kinect camera coordinates. XY projection is equally valid.
**Not the cause of the quality difference.**

### 4. Post-triangulation cleaning

Preprocessing: no cleaning. Elongated boundary triangles are tolerated.

SLIM UV: the 8-step `clean_mesh` pipeline (designed for BPA artifacts) runs
after meshing. With Delaunay + aggressive edge filtering producing holes, the
hole-filler may introduce distortion or disconnected regions that corrupt the
SLIM UV parameterisation.

### 5. 3D edge measurement on a 2D-projected triangulation

The Delaunay triangulation is computed on the **XY projection** (2D), but
`build_delaunay_mesh()` measures edge lengths in **3D**
(`np.linalg.norm(v1 - v0, axis=1)` over full XYZ vertices). On the curved
forearm surface, two points that are close in XY can be far apart in Z due
to the forearm's cylindrical geometry.

With `avg_nn ≈ 5 mm` → `max_edge_mm ≈ 15 mm`, Delaunay-valid triangles in
high-curvature regions (forearm edges, wrist transition) can exceed the 3D
threshold even though their 2D projection is well-formed. This
**systematically removes triangles at curved boundaries**, creating holes in
the regions that matter most for surface continuity.

The preprocessing pipeline avoids this entirely: no edge filtering → no
curvature-dependent triangle rejection.

**Code location:** `rf_surface_utils.py::build_delaunay_mesh()`, lines 252–270.

### 6. `clean_mesh` scope mismatch (BPA vs. Delaunay)

A 2.5D Delaunay triangulation is **manifold by construction**: it always
produces a valid planar triangulation without non-manifold edges, degenerate
triangles, or pinch vertices. The `clean_mesh` pipeline was designed for BPA
meshes, which commonly exhibit these artifacts.

Running BPA-oriented cleanup on a Delaunay mesh is:

- **Unnecessary** for manifold repair, non-manifold edge removal, and
  degenerate-triangle removal (these artifacts cannot exist in Delaunay
  output).
- **Harmful** when edge filtering has already created holes: the cascade
  (largest-component isolation → pinch repair up to 10 passes → sliver
  removal → hole filling) amplifies the initial damage rather than
  mitigating it.

**Code location:** `slim_helpers.py::clean_mesh()`, lines 645–851.

---

## Open questions — to answer before fixing

### Q1 ~~(most critical)~~: What IS `{session_id}_forearm.ply`? — ANSWERED

**Answer: (a)** — the voxel-downsampled Kinect point cloud, **not** mesh
vertices. Density is approximately **uniform at 5 mm**.

**Evidence:** Full chain-of-custody traced through code (see §2 above). The
PLY originates from `ArmSegmentation.voxel_down_sample(leaf_size=5.0)` and
passes through only `shutil.copy2` copies and rigid transforms (PCA rotation,
RF translation) before reaching the SLIM UV pipeline. No mesh-vertex
extraction or resampling occurs at any stage.

**Implication:** The original primary hypothesis (non-uniform density making
`avg_nn` unreliable) is **not supported**. The quality difference must come
from the edge filtering itself (§1, §5) and/or the `clean_mesh` cascade (§6),
not the input distribution.

### Q2: How many triangles does edge filtering remove?

Log `n_faces_before_filter` and `n_faces_after_filter` inside
`build_delaunay_mesh()` to see how aggressively the threshold cuts.

**Already instrumented:** `_build_and_cache_delaunay()` logs `avg_nn` and
`max_edge_mm` at INFO level. Add face-count logging to `build_delaunay_mesh()`
after the filter step (it currently logs only vertex/face counts of the
final mesh).

### Q3: Does removing edge filtering (+ skipping `clean_mesh`) fix quality?

Run with `max_edge_mm` disabled **and** `clean_mesh` skipped for the Delaunay
path. Since Delaunay output is manifold by construction (§6), the BPA cleanup
steps are unnecessary and may be causing secondary damage.

**How to test:** In `configs/analyse_workflow_processing_dag.yaml`, uncomment
and set `max_edge_mm: 200.0` and re-run with `interactive: true`. Also
temporarily bypass `clean_mesh` in the Delaunay code path. If the raw
unfiltered Delaunay looks clean, the fix is to disable both by default for
`mesh_method: delaunay`.

### Q4: ~~Should preprocessing apply voxel downsampling before SLIM UV Delaunay?~~ — MOOT

**Unnecessary.** Q1 shows the PLY is already voxel-downsampled at 5 mm
(uniform). Adding a second downsample would lose resolution without changing
the density distribution.

---

## Code locations

| Element | File | Notes |
|---------|------|-------|
| `_build_and_cache_delaunay()` | `code/src/analysis/receptive_field_mapping/surface/rf_surface_utils.py` | avg_nn + max_edge_mm computation, PLY loading |
| `build_delaunay_mesh()` | same file | XY projection, Delaunay, edge filter, winding fix |
| `define_forearm_mesh()` (preprocessing) | `code/scripts/_3_preprocessing/_3_forearm_extraction/define_forearm_mesh.py` | No edge filter; voxel-downsampled input |
| `ArmSegmentation` | `code/src/preprocessing/forearm_extraction/arm_segmentation.py` | Voxel 5 mm, DBSCAN, HSV filter |
| DAG config | `configs/analyse_workflow_processing_dag.yaml` | `mesh_method: delaunay`, commented `max_edge_mm` |

---

## Hypothesis ranking

1. **Primary:** 3D edge measurement on curved surface (§5) — curvature inflates
   3D distances beyond `max_edge_mm` even with uniform 5 mm input, systematically
   removing triangles at high-curvature regions.
2. **Secondary:** `clean_mesh` cascade (§6) amplifies edge-filter damage —
   component isolation, pinch repair, and hole filling compound the initial
   holes into larger mesh corruption.
3. **Tertiary:** The `3.0 × avg_nn` multiplier is inherently too tight for
   convex-hull boundary triangles, which are legitimately elongated.
4. ~~**Deprioritised:**~~ Non-uniform input density — Q1 shows input IS uniform
   at 5 mm. This was the original primary hypothesis but is no longer supported.

---

## Suggested fix (pending Q2–Q3)

If Q3 confirms edge filtering + `clean_mesh` are the cause:

- **Option A (recommended):** Remove edge filtering for Delaunay by default
  (`max_edge_mm` → `None`). Expose as optional DAG param for power users.
  This matches the preprocessing pipeline, which produces excellent results
  with no filtering.
- **Option B:** Skip `clean_mesh` entirely for `mesh_method: delaunay`. A
  2.5D Delaunay is manifold by construction — the BPA-oriented cleanup steps
  are unnecessary. Combine with Option A for the cleanest approach.
- ~~Option C:~~ Voxel-downsample before Delaunay — **moot** per Q4 (input is
  already 5 mm uniform).
