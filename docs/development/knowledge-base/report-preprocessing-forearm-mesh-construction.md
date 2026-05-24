# Report: Preprocessing Forearm Mesh Construction

**Date:** 2026-05-22
**Author:** Basil Duvernoy

---

## Purpose

This report documents how the forearm triangle mesh is built in the
preprocessing pipeline. It traces the full path from raw Kinect depth
frames to a persisted OBJ mesh file, covering every intermediate stage,
the key classes and functions involved, and the design decisions behind
the approach.

---

## Overview

The preprocessing forearm extraction pipeline converts Kinect MKV
recordings into forearm triangle meshes. The pipeline has four stages:

```
Depth Averaging  -->  Arm Segmentation  -->  Normal Estimation  -->  Mesh Construction
(FrameDepthAverager)  (ArmSegmentation)    (PointCloudModel)     (define_forearm_mesh)
```

**Input:** Raw MKV depth video from Azure Kinect.
**Output:** Wavefront OBJ triangle mesh with vertex colors.

---

## Stage 1 -- Depth Frame Averaging

**File:** `code/src/preprocessing/forearm_extraction/depth_averaging/frame_depth_averager.py`
**Class:** `FrameDepthAverager`
**Entry point:** `FrameDepthAverager.average(mkv, frame_ids)`

### What it does

Computes a per-pixel mean of the Kinect's `transformed_depth_point_cloud`
across multiple frames to reduce noise. Color is averaged separately.

### Algorithm

1. Iterates over sorted frame IDs from the `KinectMKV` context.
2. For each frame, extracts the XYZ point cloud (shape `H x W x 3`) and
   builds a validity mask: `Z > 0` and no NaN values.
3. Maintains running accumulators (`running_sum`, `valid_count`) so memory
   stays `O(H x W x 3)` regardless of frame count.
4. Averages color frames (BGR -> RGB, scaled to `[0, 1]`) across all
   frames that have valid color data.
5. Excludes pixels with zero valid observations from the final cloud.

### Output

An `o3d.geometry.PointCloud` with averaged XYZ positions and averaged RGB
colors. Pixels with no valid depth across any frame are excluded.

### Error handling

- Individual frames that fail to load or have no depth are skipped with a
  warning.
- Raises `ValueError` if no frames yielded valid depth data at all.
- If no color frames are available, assigns white (`[1, 1, 1]`) as the
  fallback color.

---

## Stage 2 -- Arm Segmentation

**File:** `code/src/preprocessing/forearm_extraction/arm_segmentation.py`
**Class:** `ArmSegmentation`
**Entry points:** `preprocess(pcd, box_corners)` then `extract_arm(pcd)`

### What it does

Isolates the forearm point cloud from background objects, the table
surface, and non-skin regions.

### Algorithm

**Step 2a -- Voxel downsampling** (default leaf size: 5.0 mm)

Reduces point density uniformly via Open3D's `voxel_down_sample()`. This
controls the computational cost of subsequent stages and the final mesh
vertex count.

**Step 2b -- Bounding-box crop**

Crops the cloud to an axis-aligned bounding box defined by `box_corners`
(XY bounds from the session config) and optional Z bounds from params.

**Step 2c -- HSV skin-color filter**

Converts each vertex color from RGB to HSV and applies range filters on
hue, saturation, and value. The hue filter supports cyclic wrap-around
(e.g. `[330, 30]` selects pink/red across the 0/360 boundary).

Default ranges:
- Hue: `[335, 25]` degrees (red/pink skin tones, wrapping through 0)
- Saturation: `[0.1, 1.0]`
- Value: `[0.0, 1.0]`

**Step 2d -- DBSCAN clustering**

Runs Open3D's `cluster_dbscan()` (default `eps=18.0` mm,
`min_cluster_size=50`) on the filtered cloud. Selects the largest cluster
by vertex count, assuming it is the forearm.

### Interactive mode

When `interactive=True`, each step launches an Open3D GUI window with
slider controls for real-time parameter tuning. The hue filter uses a
custom circular hue-wheel widget with draggable handles. A hover readout
displays HSV values under the cursor.

### Output

An `o3d.geometry.PointCloud` containing only the forearm vertices with
their original RGB colors restored (clustering visualization colors are
not propagated).

---

## Stage 3 -- Normal Estimation

**File:** `code/src/preprocessing/forearm_extraction/normals_estimation/point_cloud_model.py`
**Class:** `PointCloudModel`
**Entry points:** `load(filepath)` then `compute_normals()` then `save()`

### What it does

Estimates surface normals for each vertex of the segmented forearm point
cloud and persists the result as a PLY file with normals embedded.

### Algorithm

1. **Load** the segmented point cloud via `o3d.io.read_point_cloud()`.
2. **Estimate normals** using one of two search strategies:
   - *KDTree hybrid search* (default): `radius=0.1`, `max_nn=100` --
     finds up to 100 nearest neighbours within a radius of 0.1 (units
     match the point cloud, typically metres).
   - *KDTree KNN*: pure k-nearest-neighbours with `k=100`.
3. **Orient normals** for consistency:
   - If `align_with_viewpoint=True`: orients normals to face a given
     viewpoint vector (e.g. the Kinect camera position).
   - Otherwise: uses Open3D's `orient_normals_consistent_tangent_plane(k)`
     for local tangent-plane consistency across the surface.
4. **Optional post-processing**: centering (subtract cloud centroid),
   scaling, and manual normal flip (toggled via the GUI).

### Output

A PLY file (`*_with_normals.ply`) containing XYZ positions, RGB colors,
and normal vectors. A companion JSON metadata file stores the processing
parameters.

---

## Stage 4 -- Mesh Construction (Core)

**File:** `code/scripts/_3_preprocessing/_3_forearm_extraction/define_forearm_mesh.py`
**Function:** `define_forearm_mesh(source, output_path, show, force_processing)`

### What it does

Converts the normal-augmented point cloud into a triangle mesh using 2.5D
Delaunay triangulation, corrects normal orientation, and exports the
result as an OBJ file.

### Idempotency guard (lines 51-62)

Before doing any work, checks whether the output mesh is already
up-to-date relative to the input PLY via `should_process_task()`. If the
output exists and is newer than the input, refreshes its mtime and returns
`None` (no reprocessing). This integrates with the pipeline's
idempotency mechanism.

### Step 4.1 -- Input loading (lines 64-93)

Accepts either a file path or a raw `(N, 3)` NumPy array.

When given a file path:
1. Reads the PLY via `o3d.io.read_point_cloud()`.
2. Validates the cloud is non-empty.
3. Extracts three arrays:
   - `points`: `np.asarray(pcd.points)` -- shape `(N, 3)`, the XYZ
     positions.
   - `input_normals`: `np.asarray(pcd.normals)` -- shape `(N, 3)`, if
     present.
   - `input_colors`: `np.asarray(pcd.colors)` -- shape `(N, 3)`, float
     RGB in `[0, 1]`, if present.

Validates that `points` is a 2D array with exactly 3 columns.

### Step 4.2 -- 2D projection (lines 95-96)

```python
xy_points = points[:, 0:2]
```

Projects the 3D point cloud onto the XY plane by discarding the Z
coordinate. This is the "2.5D" in "2.5D Delaunay triangulation": the
triangulation is computed in 2D, but the resulting mesh uses the original
3D vertex positions.

**Why this works:** The forearm is captured from a fixed Kinect viewpoint
roughly aligned with the Z axis. From this perspective, the forearm
surface is approximately a height field -- each (X, Y) location maps to
at most one Z value. The 2D Delaunay triangulation of the XY projection
therefore produces a valid surface triangulation of the 3D points.

**Limitation:** This approach fails for surfaces with self-occlusion or
folds where multiple Z values share the same (X, Y). For the forearm
captured from a single viewpoint, this does not occur.

### Step 4.3 -- Delaunay triangulation (lines 98-100)

```python
from scipy.spatial import Delaunay
tri = Delaunay(xy_points)
```

`scipy.spatial.Delaunay` computes the Delaunay triangulation of the 2D
point set. The result `tri.simplices` is an `(M, 3)` integer array where
each row contains the indices of three vertices forming a triangle.

**Properties of Delaunay triangulation:**
- Maximises the minimum angle of all triangles (avoids thin slivers where
  possible given the point distribution).
- Produces the convex hull of the 2D point set -- all boundary triangles
  extend to the convex hull.
- Runtime: `O(N log N)` via the Qhull library underneath SciPy.

**Side effect:** Because Delaunay covers the convex hull, elongated
triangles can appear at concavities in the forearm boundary (e.g. between
fingers) where the convex hull bridges across empty space. These are
tolerated in the preprocessing mesh and cleaned up downstream if the mesh
enters the SLIM UV parameterization pipeline.

### Step 4.4 -- Trimesh assembly (lines 102-105)

```python
mesh = trimesh.Trimesh(
    vertices=points,        # (N, 3) -- original 3D positions
    faces=tri.simplices,    # (M, 3) -- triangle indices from Delaunay
    vertex_colors=input_colors,
)
```

Combines the original 3D vertex positions (not the 2D projections) with
the face connectivity from the 2D triangulation into a `trimesh.Trimesh`
object. Vertex colors are propagated directly -- no texture mapping or UV
coordinates are involved.

### Step 4.5 -- Normal orientation correction (lines 107-125)

The face winding order from Delaunay is arbitrary (it triangulates in 2D
and has no concept of "outward" in 3D). This step aligns the mesh face
normals with the known surface normals from Stage 3.

**Primary path -- input normals available (lines 108-118):**

1. Compute face normals from the mesh geometry: `mesh.face_normals`
   (shape `(M, 3)`).
2. For each face, average the input vertex normals of its three vertices
   to get the expected face normal direction.
3. Compute the dot product between each generated face normal and its
   expected direction:
   ```python
   dots = np.einsum('ij,ij->i', generated_face_normals, avg_input_normals)
   ```
4. If more than half the dot products are negative (normals point
   opposite to expectations), the entire mesh winding is inverted:
   ```python
   mesh.invert()  # reverses face winding order
   ```

**Fallback path -- no input normals (lines 119-123):**

Assumes the forearm surface faces upward (positive Z). If the mean Z
component of face normals is negative, flips the mesh.

**Final consistency pass (line 125):**

```python
mesh.fix_normals()
```

Trimesh's `fix_normals()` ensures consistent winding across the entire
mesh by propagating the orientation from face to face along shared edges.

### Step 4.6 -- Persistence (lines 127-137)

Exports the mesh to the output path via `trimesh.export()`. The format is
inferred from the file extension (typically `.obj` for Wavefront OBJ).
The parent directory is created if it does not exist.

### Step 4.7 -- Optional visualization (lines 139-178)

When `show=True`, converts the Trimesh to an Open3D `TriangleMesh` via
the helper function `trimesh_to_open3d()` (lines 14-34) and displays it
in an Open3D viewer window. The viewer provides:

- **F key**: flip mesh orientation (reverses triangle winding and
  recomputes normals).
- **W key**: toggle wireframe overlay (shader-dependent).

The `trimesh_to_open3d()` conversion:
1. Copies vertices and faces to Open3D vector types.
2. Converts vertex colors from Trimesh's uint8 `[0, 255]` to Open3D's
   float `[0.0, 1.0]` format.
3. Transfers vertex normals from cache if available; otherwise calls
   `compute_vertex_normals()`.

---

## Output File Naming

The output mesh filename follows the pattern:

```
{video_stem}_{frame_id}_mesh.obj
```

Example:
```
2022-06-17_ST16-05_semicontrolled_block-order01_kinect_frame_0007_mesh.obj
```

The file is stored alongside the input PLY in the session's
`forearm_pointclouds/` directory.

---

## Mesh Loading (Downstream Access)

**File:** `code/src/preprocessing/forearm_extraction/models/forearm_catalog.py`
**Class:** `ForearmCatalog`
**Method:** `_load_mesh(params)` (line 79)

Downstream consumers access the mesh through `ForearmCatalog`, which:

1. Constructs the expected filename from `ForearmParameters` (video stem +
   frame ID + `_mesh.obj`).
2. Loads via `o3d.io.read_triangle_mesh()`.
3. Validates the mesh is non-empty.
4. Returns an `o3d.geometry.TriangleMesh` or `None`.

Public accessors:
- `get_meshes_for_video(video_filename)` -- returns all meshes for a video
  as a `Dict[frame_id, TriangleMesh]`.
- `get_first_mesh()` -- returns the first loadable mesh in the catalog.
- `find_closest_reference(video_filename, use_mesh=True)` -- finds the
  mesh from the nearest block number for cross-block fallback.

---

## Data Flow Diagram

```
                        Kinect MKV
                            |
                            v
            +-------------------------------+
            |     FrameDepthAverager        |
            |  Per-pixel depth + color mean |
            |  across N frames              |
            +-------------------------------+
                            |
                    o3d.PointCloud (XYZ + RGB)
                            |
                            v
            +-------------------------------+
            |       ArmSegmentation         |
            |  1. Voxel downsample          |
            |  2. Bounding-box crop         |
            |  3. HSV skin-color filter     |
            |  4. DBSCAN largest cluster    |
            +-------------------------------+
                            |
                    o3d.PointCloud (forearm only)
                            |
                            v
            +-------------------------------+
            |       PointCloudModel         |
            |  KDTree normal estimation     |
            |  Tangent-plane orientation    |
            |  Optional: center + scale     |
            +-------------------------------+
                            |
                    *_with_normals.ply (XYZ + RGB + normals)
                            |
                            v
            +-------------------------------+
            |     define_forearm_mesh()      |
            |  1. Load PLY via Open3D       |
            |  2. Project XY for 2D Delaunay|
            |  3. scipy.spatial.Delaunay    |
            |  4. Trimesh(verts, faces)     |
            |  5. Normal orientation fix    |
            |  6. Export to OBJ             |
            +-------------------------------+
                            |
                    *_mesh.obj (triangle mesh)
                            |
                            v
            +-------------------------------+
            |       ForearmCatalog          |
            |  Load + index for downstream  |
            |  analysis pipelines           |
            +-------------------------------+
```

---

## Design Decisions

### Why 2.5D Delaunay instead of 3D surface reconstruction?

The forearm is captured from a single Kinect viewpoint, making it a
height field (single-valued Z for each XY). A 2D Delaunay triangulation
on the XY projection is:

- **Simple:** One function call (`scipy.spatial.Delaunay`), no parameter
  tuning.
- **Fast:** `O(N log N)` via Qhull.
- **Deterministic:** No iterative convergence, no random seeds.
- **Well-conditioned:** Delaunay maximises minimum angles, avoiding
  extreme slivers.

Alternatives like Ball Pivoting Algorithm (BPA) or Poisson reconstruction
handle arbitrary 3D surfaces but require careful parameter tuning and
produce non-manifold artifacts that need extensive cleanup (see
`report-slim-uv-mesh-cleaning-steps.md` for the eight cleaning steps
required when BPA is used upstream of SLIM UV parameterization).

### Why vertex colors instead of texture mapping?

The Kinect provides per-pixel RGB aligned with depth. Propagating these
as vertex colors is lossless at the mesh resolution and avoids the
complexity of UV parameterization at the preprocessing stage. UV mapping
is performed later in the analysis pipeline (SLIM) where it serves a
different purpose (receptive field mapping onto a 2D surface).

### Why Trimesh for construction and Open3D for visualization?

- **Trimesh** provides `fix_normals()`, `invert()`, and robust
  import/export -- better mesh repair primitives than Open3D.
- **Open3D** provides the GPU-accelerated visualization window with
  key callbacks. The original Trimesh viewer had COM/MTA threading
  issues and blank-window bugs on Windows.

---

## Key Source Files

| Stage | File | Class / Function | Lines |
|-------|------|------------------|-------|
| Depth averaging | `code/src/preprocessing/forearm_extraction/depth_averaging/frame_depth_averager.py` | `FrameDepthAverager.average()` | 15-106 |
| Arm segmentation | `code/src/preprocessing/forearm_extraction/arm_segmentation.py` | `ArmSegmentation.preprocess()`, `extract_arm()` | 133-212 |
| Normal estimation | `code/src/preprocessing/forearm_extraction/normals_estimation/point_cloud_model.py` | `PointCloudModel.compute_normals()` | 78-106 |
| Mesh construction | `code/scripts/_3_preprocessing/_3_forearm_extraction/define_forearm_mesh.py` | `define_forearm_mesh()` | 36-179 |
| Trimesh-to-O3D | `code/scripts/_3_preprocessing/_3_forearm_extraction/define_forearm_mesh.py` | `trimesh_to_open3d()` | 14-34 |
| Mesh loading | `code/src/preprocessing/forearm_extraction/models/forearm_catalog.py` | `ForearmCatalog._load_mesh()` | 79-98 |

---

## Related Documents

- Knowledge base: [`report-slim-uv-mesh-cleaning-steps.md`](report-slim-uv-mesh-cleaning-steps.md) -- the eight mesh cleaning steps required to prepare BPA meshes for SLIM UV parameterization
- Knowledge base: [`note-kinect-depth-access-single-path.md`](note-kinect-depth-access-single-path.md) -- why all depth reads must go through `KinectFrame`
- Knowledge base: [`note-azure-kinect-rgb-depth-parallax.md`](note-azure-kinect-rgb-depth-parallax.md) -- the RGB/depth parallax correction applied in `KinectFrame`
- Knowledge base: [`investigation-mesh-flattening-algorithms-survey.md`](investigation-mesh-flattening-algorithms-survey.md) -- literature survey of flattening algorithms (context for why SLIM was chosen downstream)
