# 3D-to-2D Surface Projection Algorithms for RF Heatmaps

## Symptom

RF heatmaps rendered as 3D scatter plots suffer from depth ambiguity, inconsistent
cross-session views, and non-standard format for publication. A true 2D projection
of the forearm surface is needed.

## Investigation

Five families of algorithms were evaluated for projecting a ~20–60 mm diameter
forearm surface patch from 3D to 2D `(u, v)` coordinates in mm. The evaluation
criteria were: distortion at the RF center, distortion at patch edges, implementation
complexity, dependency footprint, and whether a mesh is required.

## Context

This builds upon the tangent-plane alignment feature
(`docs/development/plans/active/tangent-plane-rf-alignment.md`), which added a
render-time 3×3 rotation R aligning each session's forearm surface normal to the
camera Z-axis with the PCA longitudinal axis as X.

### Current data

| Data | Format | Typical size | Source |
|------|--------|-------------|--------|
| Forearm PLY | Open3D PointCloud (XYZ mm, normals, optional RGB) | 10k–50k vertices | `define_normals.py` |
| Forearm mesh | Trimesh (2.5D Delaunay on XY, preserves Z) | Same vertices, ~2× triangles | `define_forearm_mesh.py` |
| Spike counts | DataFrame `(x, y, z, spike_count)` | 50–500 rows, snapped to PLY vertices | `rf_simple_pipeline.py` |
| RF center | `np.ndarray(3,)` = origin after centering | Always `[0, 0, 0]` | `center_on_receptive_field.py` |
| Tangent rotation R | `np.ndarray(3, 3)` orthonormal | Per-session | `tangent_plane_alignment.py` |
| Coordinate system | PCA-calibrated, mm, origin at RF center | X = longitudinal, Y/Z vary | After centering + PCA |
| RF patch diameter | ~20–60 mm | Local neighborhood around origin | Empirical |

### Desired output

- `(u, v)` coordinates in mm for every 3D point (forearm vertices + spike contacts)
- `u` = longitudinal (proximal–distal), `v` = circumferential (medial–lateral)
- RF hotspot at `(0, 0)`
- Cross-session maps directly comparable (same orientation, same units)
- Distortion minimized at RF center, quantifiable at edges
- Two toggleable renderings: 2D scatter plot and interpolated heatmap image

---

## Algorithm Catalogue

Each algorithm is documented with: principle, mathematical formulation, distortion
characteristics, implementation path, dependencies, and suitability.

---

### Algorithm 1 — Tangent-Plane Projection (Drop-Z)

**Method name:** `"tangent_plane"`

**Principle:**
After the existing tangent-plane rotation R aligns the local surface normal to Z
and the longitudinal axis to X, discard the Z coordinate:
`u, v = rotated[:, 0], rotated[:, 1]`. This is orthogonal projection onto the
tangent plane at the RF center.

**Mathematical formulation:**
1. Compute R via `compute_tangent_plane_rotation(forearm_vertices, contact_centroid)`
2. Rotate: `p_rot = points @ R.T`
3. Project: `u, v = p_rot[:, 0], p_rot[:, 1]`

**Distortion characteristics:**
- **At RF center:** Exactly zero — the tangent plane is exact at the point of tangency.
- **At edges:** Distances are compressed because the curved surface is flattened.
  For a cylindrical forearm of radius `r` and a patch of half-width `d`:
  - Max Z excursion: `Δz = r - √(r² - d²)`
  - For 60 mm patch on 45 mm-radius forearm: `Δz ≈ 11.5 mm`, ~5–10% distance
    compression at edges
  - Distortion grows quadratically with distance from center: `O(d²/r)`
- **Angles:** Preserved near center, mild distortion at edges.
- **Areas:** Slightly compressed at edges (surface area > projected area).

**Implementation sketch:**
```python
def project_tangent_plane(points_3d, forearm_vertices, contact_centroid, k=50):
    R = compute_tangent_plane_rotation(forearm_vertices, contact_centroid, k)
    if R is None:
        return None, None
    rotated = align_points(points_3d, R)
    return rotated[:, :2], R
```

**Dependencies:** None beyond existing code (`tangent_plane_alignment.py`).

**Suitability:** Excellent for 20–60 mm patches. Simplest possible approach with
negligible distortion at the scale of interest. Recommended as the default/baseline.

---

### Algorithm 2 — Exponential Map (Logarithmic Map)

**Method name:** `"exponential_map"`

**Principle:**
The exponential map is the natural parameterization of a geodesic neighborhood on a
surface. For each point P, compute the geodesic distance `d` and the geodesic
direction `θ` from the RF center, then map to 2D polar coordinates:
`u = d·cos(θ)`, `v = d·sin(θ)`. The inverse (log map) maps surface points back to
the tangent plane while preserving geodesic distances from the center.

**Mathematical formulation:**
1. Build mesh from forearm PLY (available from `define_forearm_mesh.py`)
2. Compute geodesic distances from RF center to all vertices via Dijkstra on mesh
   edges (or the heat method)
3. For direction: compute tangent-plane rotation R at the center, then for each
   vertex find the initial direction of the shortest geodesic path from center to
   vertex and project onto the tangent plane to get angle `θ`
4. Map: `u = d·cos(θ)`, `v = d·sin(θ)`

**Simplified practical approach (Dijkstra + tangent projection):**
1. Compute shortest paths on mesh graph via `scipy.sparse.csgraph.shortest_path`
2. For angle: use tangent-plane rotation R, take direction vector from center to the
   first mesh neighbor on the shortest path, project onto tangent plane
3. Convert polar `(d, θ)` → Cartesian `(u, v)`

**Implementation sketch:**
```python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

def project_exponential_map(mesh_vertices, mesh_faces, center_idx, R):
    # Build adjacency with edge lengths as weights
    edges = set()
    for face in mesh_faces:
        for i in range(3):
            a, b = face[i], face[(i+1) % 3]
            edges.add((a, b))
    rows, cols, weights = [], [], []
    for a, b in edges:
        d = np.linalg.norm(mesh_vertices[a] - mesh_vertices[b])
        rows.extend([a, b]); cols.extend([b, a]); weights.extend([d, d])
    n = len(mesh_vertices)
    graph = csr_matrix((weights, (rows, cols)), shape=(n, n))

    # Geodesic distances from center
    dists = shortest_path(graph, indices=center_idx)

    # Angles: project displacement onto tangent plane
    displacements = mesh_vertices - mesh_vertices[center_idx]
    tangent_coords = displacements @ R.T
    angles = np.arctan2(tangent_coords[:, 1], tangent_coords[:, 0])

    # Polar → Cartesian
    u = dists * np.cos(angles)
    v = dists * np.sin(angles)
    return np.stack([u, v], axis=1)
```

**Distortion characteristics:**
- **Distances from center:** Exactly preserved (geodesic distance is the radial
  coordinate).
- **Angles at center:** Exactly preserved.
- **Distances between non-center points:** Approximately preserved; error grows with
  curvature variation across the patch.
- **Improvement over drop-Z:** ~1–3% better distance preservation at patch edges for
  typical forearm curvature.
- **Singularities:** None for a single connected patch around the center.

**Dependencies:**
- `scipy.sparse.csgraph` (already in project)
- Mesh from `define_forearm_mesh.py`
- Tangent-plane rotation R from `tangent_plane_alignment.py`

**Advanced alternative:** The heat method for geodesic distances (`potpourri3d`
Python package, or `geometry-central` C++ with Python bindings) is faster for large
meshes but adds a dependency.

**Suitability:** Good upgrade over drop-Z when distortion precision matters. The
improvement is marginal for 20–60 mm patches but provides mathematically rigorous
geodesic distance preservation.

---

### Algorithm 3 — Geodesic MDS (Multi-Dimensional Scaling)

**Method name:** `"geodesic_mds"`

**Principle:**
Compute all-pairs geodesic distances on the mesh surface, then use MDS to find a 2D
embedding that best preserves those distances. Unlike the exponential map (which is
center-centric), MDS optimizes distance preservation globally across all point pairs.

**Mathematical formulation:**
1. Build mesh graph, compute all-pairs geodesic distances → distance matrix `D` of
   shape `(N, N)`
2. Apply classical (metric) MDS or SMACOF iterative MDS:
   - Classical MDS: double-center `D²`, eigendecompose, take top-2 eigenvectors
   - SMACOF: iteratively minimize stress `Σ(d_ij - ||x_i - x_j||)²`
3. Result: 2D coordinates `(u, v)` for each vertex

**Scikit-learn implementation:**
```python
from sklearn.manifold import MDS

def project_geodesic_mds(mesh_vertices, mesh_faces, random_state=42):
    graph = build_mesh_graph(mesh_vertices, mesh_faces)
    D = shortest_path(graph, directed=False)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=random_state)
    uv = mds.fit_transform(D)
    return uv
```

**Isomap alternative** (geodesic-aware manifold learning):
```python
from sklearn.manifold import Isomap

def project_isomap(mesh_vertices, n_neighbors=10):
    iso = Isomap(n_components=2, n_neighbors=n_neighbors)
    uv = iso.fit_transform(mesh_vertices)
    return uv
```

**Distortion characteristics:**
- **Global distance preservation:** Best among all methods — optimizes a global
  objective.
- **Local angles/areas:** Not explicitly preserved.
- **Center symmetry:** Not guaranteed — MDS doesn't know about the RF center;
  post-processing may be needed to center and orient the result.
- **Determinism:** Classical MDS is deterministic; SMACOF depends on initialization.

**Computational cost:**
- All-pairs shortest path: `O(N² log N)` with Dijkstra, `O(N³)` with Floyd-Warshall
- N = 10k vertices: ~100M distance pairs → seconds to minutes
- N = 50k vertices: ~2.5B pairs → minutes to hours, ~20 GB memory (float64)
- **Mitigation:** Subsample to ~1k–5k vertices, compute MDS on subsample, interpolate
  remaining points via nearest-neighbor mapping.

**Dependencies:**
- `scikit-learn` (MDS or Isomap)
- `scipy.sparse.csgraph` for geodesic distances
- Mesh from pipeline

**Suitability:** Overkill for local 20–60 mm patches. Best suited for full-forearm
parameterization. Computational cost and non-determinism make it less practical than
the exponential map for center-centric RF visualization.

---

### Algorithm 4 — LSCM / ARAP (Mesh Parameterization)

**Method name:** `"lscm"` / `"arap"`

**Principle:**
Mesh parameterization methods from computational geometry that unfold a 3D triangle
mesh onto a 2D domain:
- **LSCM (Least-Squares Conformal Maps):** Minimizes angular distortion. Preserves
  local angles (conformal), but areas may shrink or grow.
- **ARAP (As-Rigid-As-Possible):** Minimizes a combination of angular and area
  distortion via iterative local/global optimization. Each triangle is mapped as
  close to a rigid transformation as possible.

**Mathematical formulation (LSCM):**
1. For each triangle with 3D vertices `(p1, p2, p3)` and 2D targets `(u1, u2, u3)`:
   the Jacobian `J` of the 3D→2D map should satisfy the Cauchy-Riemann equations
2. Minimize `Σ_triangles area_t · ||J_t - closest_conformal_J||²`
3. Reduces to a sparse linear system (one linear solve, no iteration)
4. Free boundary — only 2 vertices need pinning for uniqueness

**Mathematical formulation (ARAP):**
1. Local step: for each triangle, find the closest rotation `R_t` to the current
   Jacobian `J_t`
2. Global step: fix rotations, solve for vertex positions that stitch triangles
3. Iterate until convergence (typically 5–20 iterations)
4. Energy: `Σ_triangles area_t · ||J_t - R_t||²`

**Python libraries:**

| Library | LSCM | ARAP | Python API | Notes |
|---------|------|------|------------|-------|
| **libigl** | Yes | Yes (deformation) | `pip install libigl` | Most mature; `igl.lscm()` |
| **Easy3D** | Yes | No | `pip install easy3d` | `SurfaceMeshParameterization.lscm()` |
| **CGAL** | Yes | Yes | C++ (pybind wrappers) | Full suite, heavy dependency |
| **Custom** | Moderate | Complex | numpy/scipy | LSCM: ~100 lines; ARAP: ~300 lines |

**libigl example (LSCM):**
```python
import igl
import numpy as np

def project_lscm(vertices, faces):
    boundary = igl.boundary_loop(faces)
    b = np.array([boundary[0], boundary[len(boundary) // 2]])
    bc = np.array([[0.0, 0.0], [1.0, 0.0]])
    _, uv = igl.lscm(vertices, faces, b, bc)
    return uv
```

**Distortion comparison:**

| Property | LSCM | ARAP |
|----------|------|------|
| Angle preservation | Excellent (conformal) | Good |
| Area preservation | Poor | Good |
| Distance preservation | Moderate | Good |
| Bijectivity guarantee | No (foldovers possible) | No |
| Determinism | Yes (single linear solve) | Depends on initialization |
| Boundary | Free (natural) | Fixed or free |

**Suitability:** Designed for full-surface UV mapping. For a local 20–60 mm patch,
significantly more complex than needed. Prefer these if the patch is large (>80 mm),
forearm curvature is high, or area preservation matters for quantitative analysis.

---

### Algorithm 5 — Cylindrical Unwrapping

**Method name:** `"cylindrical_unwrap"`

**Principle:**
Fit a cylinder to the forearm point cloud, then unwrap using cylindrical coordinates:
`u = r·θ` (arc length along circumference), `v = h` (height along axis). Exploits
the forearm's roughly cylindrical geometry.

**Mathematical formulation:**
1. **Fit cylinder:** axis direction `a`, center point `c`, radius `r`. Use PCA for
   axis (largest eigenvector ≈ longitudinal axis, already available), then optimize
   `(c, r)` by minimizing `Σ(||p_i - proj_axis(p_i)|| - r)²`
2. **Project to cylindrical coordinates:** for each point `p`:
   - Height: `h = (p - c) · a`
   - Radial direction: `d = p - c - h·a`, normalize →
     `θ = atan2(d_y', d_x')` in a local frame perpendicular to `a`
   - `u = r·θ` (arc length), `v = h`
3. **Handle seam:** `atan2` wrapping at `±π` creates a discontinuity. Place the seam
   opposite the RF center (far side of forearm).

**Implementation sketch:**
```python
def project_cylindrical_unwrap(points_3d, forearm_vertices):
    axis = np.array([1.0, 0.0, 0.0])  # PCA longitudinal axis
    center = np.mean(forearm_vertices, axis=0)

    h = (points_3d - center) @ axis
    radial = points_3d - center - np.outer(h, axis)

    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])
    cos_theta = radial @ y_axis
    sin_theta = radial @ z_axis
    theta = np.arctan2(sin_theta, cos_theta)

    r = np.linalg.norm(radial, axis=1).mean()
    u = r * theta  # arc length (circumferential)
    v = h          # height (longitudinal)
    return np.stack([v, u], axis=1)
```

**Python libraries for cylinder fitting:**
- `pyransac3d` — RANSAC-based (`pip install pyransac3d`)
- `cylinder_fitting` — least-squares (`pip install cylinder_fitting`)
- `py-cylinder-fitting` — Nelder-Mead (`pip install py-cylinder-fitting`)
- Custom: PCA axis + `scipy.optimize.minimize` for radius/center

**Distortion characteristics:**
- **Longitudinal:** Zero — height is preserved exactly.
- **Circumferential:** Depends on cylinder fit quality. Tapered forearm (wrist→elbow)
  has varying radius → arc length `r·θ` is wrong for non-constant `r`.
  Fix: use per-point radius `r_i · θ_i`.
- **Seam artifact:** Discontinuity at `±π`. Manageable if RF is far from seam.
- **Non-cylindrical regions:** Poor for elbow, wrist, or highly curved areas.

**Suitability:** Good for mid-forearm sessions where geometry is approximately
cylindrical. PCA calibration already provides the axis. Main advantage: geometrically
intuitive, produces an unwrapped map matching physical forearm anatomy.

---

## Comparison Summary

| Property | Tangent-Plane | Exponential Map | Geodesic MDS | LSCM | ARAP | Cylindrical |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| Implementation complexity | Trivial | Moderate | Moderate | Moderate | High | Low |
| New dependencies | None | None | sklearn | libigl/Easy3D | libigl/CGAL | Optional |
| Requires mesh | No | Yes | Yes | Yes | Yes | No |
| Distance preservation (center) | Exact | Exact | Global opt | Moderate | Good | Exact (axial) |
| Distance preservation (edges) | ~90–95% | ~97–99% | ~95–99% | ~90–95% | ~95–98% | Variable |
| Angle preservation | Good | Good | Not explicit | Excellent | Good | Poor at seam |
| Area preservation | Compressed | Good | Not explicit | Poor | Good | Variable |
| Deterministic | Yes | Yes | Seed-dependent | Yes | Init-dependent | Yes |
| Compute time (10k pts) | <1 ms | ~100 ms | ~10 s | ~100 ms | ~1 s | <1 ms |
| Best for patch size | <80 mm | <100 mm | Full surface | Full surface | Full surface | Any (if cylindrical) |

## Pipeline Architecture Vision

The future pipeline should support method selection by name:

```python
PROJECTION_METHODS = {
    "tangent_plane": project_tangent_plane,
    "exponential_map": project_exponential_map,
    "geodesic_mds": project_geodesic_mds,
    "lscm": project_lscm,
    "arap": project_arap,
    "cylindrical_unwrap": project_cylindrical_unwrap,
}

def project_to_2d(points_3d, method="tangent_plane", **kwargs):
    return PROJECTION_METHODS[method](points_3d, **kwargs)
```

Each method returns `np.ndarray` of shape `(N, 2)` with `(u, v)` in mm. The renderer
is method-agnostic — it receives `(u, v)` coordinates and spike counts and produces
either a scatter plot or interpolated heatmap based on output options.

## Key Existing Code to Reuse

| Module | Function/Class | Reuse for |
|--------|---------------|-----------|
| `tangent_plane_alignment.py` | `compute_tangent_plane_rotation()`, `align_points()` | Algorithms 1 and 2 |
| `define_forearm_mesh.py` | Delaunay mesh generation | Algorithms 2, 3, 4 |
| `rf_cluster_visualizer.py` | `render_forearm_heatmap()` pattern | 2D renderer template |
| `rf_simple_pipeline.py` | Vertex snapping via KDTree | Mapping spike points to mesh vertices |
| `center_on_receptive_field.py` | RF center computation | Origin for all methods |

## References

- [Flattening-Net: Deep Regular 2D Representation for 3D Point Cloud Analysis](https://arxiv.org/abs/2212.08892)
- [Flatten Anything: Unsupervised Neural Surface Parameterization](https://arxiv.org/html/2405.14633v1)
- [CGAL Surface Mesh Parameterization](https://doc.cgal.org/latest/Surface_mesh_parameterization/index.html)
- [Easy3D Mesh Parameterization (LSCM)](https://3d.bk.tudelft.nl/liangliang/software/easy3d_doc/python/auto_tutorials/tutorial_405_mesh_parameterization.html)
- [Least Squares Conformal Maps (Lévy et al., 2002)](https://www.cs.jhu.edu/~misha/Fall09/Levy02.pdf)
- [A Local/Global Approach to Mesh Parameterization — ARAP (Liu et al., 2008)](https://cs.harvard.edu/~sjg/papers/arap.pdf)
- [libigl Python tutorial — parameterization](https://libigl.github.io/libigl-python-bindings/tut-chapter4/)
- [Surface Extraction and Flattening for Anatomical Visualization (EPFL thesis)](https://lspwww.epfl.ch/publications/gigaserver/thesis-saroul.pdf)
- [Geodesic Computations for Fast Surface Remeshing and Parameterization](https://link.springer.com/chapter/10.1007/3-7643-7384-9_18)
- [Computing Harmonic Maps and Conformal Maps on Point Clouds](https://arxiv.org/pdf/2009.09383)
- [Fast Exact and Approximate Geodesics on Meshes (Surazhsky et al.)](https://hhoppe.com/geodesics.pdf)
- [Heat distance & logarithmic map — geometry-central](https://geometry-central.net/pointcloud/algorithms/heat_solver/)
- [Local Surface Parameterizations via Geodesic Splines](https://arxiv.org/html/2410.06330v1)
- [scikit-learn MDS](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html)
- [scikit-learn Manifold Learning comparison](https://scikit-learn.org/stable/modules/manifold.html)
- [pyRANSAC-3D — cylinder fitting](https://github.com/leomariga/pyRANSAC-3D)
- [cylinder_fitting package](https://github.com/xingjiepan/cylinder_fitting)
- [Open3D Surface Reconstruction](https://www.open3d.org/docs/release/tutorial/geometry/surface_reconstruction.html)
- [Cortical Surface Segmentation and Mapping (NIH)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4587756/)
- [ARAP++ extension (Frontiers)](https://link.springer.com/article/10.1631/FITEE.1500184)
- [Implementing ARAP Flattening (Igarashi et al.)](https://www-ui.is.s.u-tokyo.ac.jp/~takeo/papers/takeo_jgt09_arapFlattening.pdf)
