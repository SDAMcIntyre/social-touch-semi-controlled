# RF Cluster Visualization — Code Overview

## Purpose

This note documents how the cluster-based RF visualization pipeline works:
which files do what, how data flows from extraction artifacts into rendered
heatmap PNGs, and the internal structure of the main rendering function.

## Module Map

All files live in `code/src/analysis/receptive_field_mapping/`.

| File | Lines | Role |
|------|-------|------|
| `rf_cluster_pipeline.py` | ~1000 | Orchestrator — `run_cluster_rf_extraction()` writes intermediate artifacts; `run_cluster_rf_visualization()` reads them, computes metrics, calls the renderer per session×cluster |
| `rf_cluster_visualizer.py` | ~585 | Main renderer — `render_forearm_heatmap()` dispatches to 2D or 3D path |
| `rf_2d_renderer.py` | ~320 | 2D heatmap rendering (gridded interpolation, disjoint mask, colourmap) |
| `rf_surface_utils.py` | ~140 | Mesh I/O: `load_or_build_forearm_mesh()`, `map_scalars_to_mesh()`, `apply_rotation_to_mesh()` |
| `tangent_plane_alignment.py` | ~65 | `compute_tangent_plane_rotation()`, `align_points()` — rotates forearm surface normal to Z-axis |
| `rf_projection.py` | ~190 | `project_to_2d()` dispatcher — tangent-plane and cylindrical-unwrap methods |
| `rf_metrics.py` | ~610 | `compute_rf_metrics()` — 30+ quantitative RF properties (hull area, Gaussian fit, etc.) |
| `rf_extraction_io.py` | ~300 | Save/load helpers for intermediate artifacts (`.npy`, `.csv`, `.json`) |
| `rf_data_loader.py` | ~350 | `load_forearm_vertices()` with `.npy` sidecar cache; `parse_contact_points()` |

## Data Flow: Pipeline → Visualizer

```
run_cluster_rf_visualization()                  [rf_cluster_pipeline.py]
│
├─ For each (combo_name, clusterer_name) pair:
│   ├─ Load neuron_touches.json                 (all-cluster touch counts per session)
│   ├─ Load sessions_metadata.json              (forearm PLY paths per session)
│   │
│   ├─ For each cluster_dir:
│   │   ├─ Load spike_counts.csv (pooled)       → compute_rf_metrics() → rf_metrics.json
│   │   ├─ Load cluster_description.json
│   │   │
│   │   └─ For each session in cluster:
│   │       ├─ Load session_spike_counts.csv    (per-session spike DataFrame)
│   │       ├─ Load cluster_contacts_xyz.npy    (M, 3) cluster contact points
│   │       ├─ Load neuron_contacts_xyz.npy     (N, 3) all-cluster contact points
│   │       ├─ Build RFRenderContext
│   │       │
│   │       ├─ render_forearm_heatmap(          [rf_cluster_visualizer.py]
│   │       │     display_metric="spike_count")
│   │       └─ render_forearm_heatmap(
│   │             display_metric="spike_ratio")
│   │
│   └─ Write rf_visualization_summary.json      (sentinel)
```

## `RFRenderContext` Dataclass

Carries neuron-scoped metadata needed by the renderer:

| Field | Type | Description |
|-------|------|-------------|
| `neuron_touches` | int | Total touches for this session across **all** clusters |
| `neuron_cluster_touches` | int | Touches for this session in **this** cluster |
| `neuron_contacts_xyz` | (N, 3) | All contact points across all clusters (session-wide) |
| `neuron_cluster_contacts_xyz` | (M, 3) | Contact points for this cluster only |
| `feature_ranges` | dict | `{name: {min, max, mean}}` from `cluster_description.json` |

**Hull invariant:** `heatmap_spike_points ⊆ neuron_cluster_contacts_xyz ⊆ neuron_contacts_xyz`
(validated in the pipeline before calling the renderer).

## `render_forearm_heatmap()` Internal Structure

The 442-line function has two major paths, selected by `projection_method`:

### 2D Path (projection_method is not None) — lines 228–301

Early return. Prepares data and delegates entirely to `rf_2d_renderer.render_2d_heatmap()`:

1. Load forearm vertices via `load_forearm_vertices()`
2. Project spike XYZ → UV via `project_to_2d()` using neuron-wide centroid as projection origin
3. Project forearm vertices → UV (background scatter)
4. Compute ratio counts if `display_metric="spike_ratio"`
5. Project hull contact arrays → UV
6. Call `render_2d_heatmap()` with all projected data

### 3D Path (projection_method is None) — lines 303–584

Renders a 3D matplotlib figure with a dark theme. The path has several
branches depending on what data is available:

```
┌─ Setup dark figure + 3D axes                          [303–318]
│
├─ Load forearm PLY vertices                             [326–341]
│   ├─ Compute tangent-plane rotation R                  [344]
│   │
│   ├─ If mesh builds successfully:                      [346–430]
│   │   ├─ Apply rotation R to mesh
│   │   ├─ Draw grey trisurf (forearm background)
│   │   ├─ Map spike scalars to mesh vertices            (map_scalars_to_mesh)
│   │   ├─ Choose colour driver:
│   │   │   ├─ spike_ratio → map ratio values, linear norm [0, 1]
│   │   │   └─ spike_count → map counts, LogNorm
│   │   ├─ Build per-face RGBA from vertex colours
│   │   ├─ Draw coloured trisurf overlay
│   │   └─ Add colorbar                                  → cbar_created = True
│   │
│   └─ If mesh fails (scatter fallback):                 [432–452]
│       ├─ Subsample forearm, exclude spike-nearby points
│       ├─ Apply rotation R if available
│       └─ Draw grey scatter background
│
├─ If cbar_created is False (scatter spike overlay):     [458–493]
│   ├─ Rotate spike points if R exists
│   ├─ Choose colour driver (spike_ratio or spike_count)
│   └─ Draw coloured scatter with colorbar
│
├─ Draw convex hull perimeters (3D wireframe):           [495–525]
│   ├─ _draw_hull_3d() for neuron-all-clusters hull      (cyan #00aaff)
│   ├─ _draw_hull_3d() for neuron∩cluster hull           (orange #ffaa00)
│   └─ Metadata text overlay (touch counts, percentages)
│
├─ Camera orientation:                                   [527–536]
│   ├─ If R exists → view_init(90°, −90°) (top-down after rotation)
│   └─ Else → compute normal from forearm vertices → _normal_to_view_angles()
│
├─ Axis labels + title                                   [538–548]
├─ Interactive point-size slider (if interactive=True)   [550–575]
└─ Save PNG + optional plt.show()                        [577–584]
```

### Two Rendering Strategies (3D)

The 3D path tries **mesh colouring** first (preferred): spike counts are
interpolated onto mesh vertices via KD-tree, producing smooth per-face
RGBA on a trisurf. If mesh building fails (missing trimesh, degenerate
geometry), it falls back to **scatter rendering**: forearm as grey points,
spikes as a coloured scatter overlay.

The `cbar_created` flag tracks which strategy succeeded so only one
colorbar is drawn.

### Colour Metrics

| `display_metric` | Colour driver | Normalisation | Range |
|-------------------|---------------|---------------|-------|
| `"spike_count"` | `spike_count` column | `LogNorm(vmin=max(1, min), vmax=max)` | [1, max_count] |
| `"spike_ratio"` | `unique_touch_spike_count / neuron_cluster_touches` | Linear | [0.0, 1.0] |

Both use the `RdYlBu_r` colourmap.

## Helper Functions

| Function | Lines | Purpose |
|----------|-------|---------|
| `_format_metadata_overlay()` | 13 | Builds "Touches: X / Y (Z%)" string |
| `_draw_hull_3d()` | 51 | Draws scipy ConvexHull edges as 3D polylines; gracefully skips < 3 points or QhullError |
| `_normal_to_view_angles()` | 18 | Converts surface normal → (elev, azim) for `ax.view_init()` |

## Key Design Decisions

- **Neuron-wide projection centroid:** All clusters for one neuron share
  the same 2D coordinate frame (centroid = mean of `neuron_contacts_xyz`,
  not per-cluster spike centroid). This ensures hull enclosure is preserved
  after projection.

- **Tangent-plane rotation in 3D:** When available, R aligns the forearm
  surface to the XY plane, then the camera looks straight down (90°, −90°).
  This gives a consistent top-down view across sessions.

- **Separate extraction/visualization sentinels:** Changing
  `projection_method` only re-renders (skips CSV parsing). See
  `rf_extraction_io.visualization_is_up_to_date()`.
