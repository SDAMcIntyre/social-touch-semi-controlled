# MeshLab Mesh Cleanup Workflow

Post-processing workflow for forearm meshes produced by
`standalone_mkv_to_forearm_mesh.py` → `standalone_mesh_edge_filter.py`.

## Prerequisites

Meshes from the reconstruction pipeline often have **duplicate vertices** —
adjacent triangles each own a private copy of shared corner positions instead
of referencing a single vertex. This must be fixed before smoothing, otherwise
triangles drift apart.

## Workflow

### 1. Merge duplicate vertices

**Filters → Cleaning and Repairing → Merge Close Vertices**

Set the threshold relative to the mesh scale (check bounding box dimensions
first). A value of 0.0001 or smaller is usually safe. This welds co-located
vertices so triangles share topology.

### 2. Remove unwanted triangles

Use the toolbar selection tools:

- **Select Faces/Vertices with a Brush** — click and drag to paint over faces.
  Scroll wheel resizes the brush.
- **Select Faces in a Rectangular Region** — drag a box selection.

Selected faces turn red. Press **Delete** (or Filters → Selection → Delete
Selected Faces and Vertices) to remove them.

Tips:

- Hold **Ctrl** while brushing to deselect.
- Switch to wireframe mode to avoid accidentally selecting back-faces.

### 3. Smooth the surface

**Filters → Smoothing, Fairing and Deformation → Taubin Smooth**

Taubin smoothing is a shrinkage-free Gaussian approximation. Default
parameters (λ = 0.5, μ = −0.53) work well. Start with 10 iterations; increase
to 20–50 for heavier smoothing.

**MeshLab has no undo for filters.** Before smoothing, either save a backup
(File → Export Mesh As) or duplicate the layer (Filters → Mesh Layer →
Duplicate Current Layer) so you can revert by switching back to it.

### 4. Export

**File → Export Mesh As** → save as `.obj` (or `.ply`).
