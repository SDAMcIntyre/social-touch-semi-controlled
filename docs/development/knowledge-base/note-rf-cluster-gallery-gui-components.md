# RF Cluster Gallery Viewer — GUI Component Reference

## Purpose

This note catalogues every visual component in the RF Cluster Gallery Viewer
(`gui/rf_cluster_gallery_viewer.py`), its widget type, position in the layout,
and what it controls. Companion to
[note-rf-cluster-visualization-overview.md](note-rf-cluster-visualization-overview.md)
which covers the static rendering pipeline.

## Layout Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│ TOOLBAR                                                              │
│ Mode: [By Cluster ▼]  Select: [cluster_0 ▼]  [Load All] [Export] [X]│
├────────┬─────────────────────────────────────────────┬───────────────┤
│        │                                             │               │
│ SIDE-  │  3D VIEW                                    │  SETTINGS     │
│ BAR    │  (PyVista QtInteractor)                     │  PANEL        │
│        │                                             │               │
│ 160 px │  ┌───────────────────────────────────────┐  │  260 px       │
│ fixed  │  │                                       │  │  fixed        │
│        │  │  Forearm mesh + spike heatmap          │  │               │
│ Scroll │  │  + hull overlays                       │  │  Cluster Info │
│ area   │  │  + text overlay                        │  │  Background   │
│ of     │  │  + scalar bar                          │  │  Forearm      │
│ thumb- │  │                                       │  │  Contact Pts  │
│ nail   │  │                                       │  │  Display      │
│ cards  │  │                                       │  │               │
│        │  └───────────────────────────────────────┘  │  [Process]    │
│        │                                             │  [Process All]│
└────────┴─────────────────────────────────────────────┴───────────────┘
```

Root layout: `QVBoxLayout` → toolbar row + `QSplitter(Horizontal)` with three
children. Stretch factors: sidebar 0, 3D view 1, settings 0. Initial sizes:
160, 900, 260 pixels.

## Toolbar

| Widget | Type | Purpose |
|--------|------|---------|
| Mode | `QComboBox` | Switch navigation axis: **By Cluster** (sidebar shows sessions for one cluster) or **By Session** (sidebar shows clusters for one session) |
| Select | `QComboBox` | Choose the fixed axis value (cluster label or session ID). Expanding size policy. |
| Load All | `QPushButton` | Pre-compute Delaunay meshes, heatmaps, and perimeter hulls for every cell so thumbnail switching is instant |
| Export images | `QPushButton` | Batch-export a PNG screenshot per cell to `_gallery_exports/` under the output directory |
| Close | `QPushButton` | Close the viewer (persists cameras and thresholds on close) |

## Sidebar (Thumbnail Panel)

Fixed-width (160 px) `QScrollArea` containing a vertical stack of
`_ThumbnailWidget` cards.

### `_ThumbnailWidget`

| Property | Detail |
|----------|--------|
| Type | `QWidget` subclass (lines 167-213) |
| Label | `QLabel` below a placeholder icon area — shows session ID (By Cluster mode) or `cluster_<label>` (By Session mode) |
| Tooltip | Touch type, touch count, RF area (mm^2) if available |
| Click | Left-click calls `_on_thumbnail_click(key)` → loads the cell into the 3D view |
| Selected state | Blue background (`#3399ff`) via stylesheet |
| Noise state | Dark grey background (`#444444`) for noise clusters (label -1) |
| Default state | No special styling |

Sidebar is repopulated when mode or selection changes. The first card is
auto-loaded on repopulation.

## 3D View (PyVista QtInteractor)

Central panel, stretches to fill available space. Renders a single cell
(one session x cluster pair) at a time.

### Scene Layers

Built by `_build_scene(cell)` (lines 922-1117). Layers are added in order:

| Layer | Name | Condition | Description |
|-------|------|-----------|-------------|
| Forearm surface | `"forearm"` | `render_as_surface=True` and Delaunay mesh succeeds | Delaunay triangulated mesh with smooth shading. Three sub-paths for colouring (see below). |
| Forearm point cloud | `"forearm"` | `render_as_surface=False` or Delaunay fails | Point cloud of forearm vertices. Supports vertex RGB or flat colour. |
| Contact points | `"contacts"` | Point-cloud mode only, spike data present | Scatter of spike positions, coloured by spike metric via `contact_cmap`. |
| Neuron hull | `"hull_neuron"` | `hull_display_mode != "off"`, neuron contacts available | Perimeter wireframe (alpha-shape edges) or point cloud, coloured `hull_neuron_color` (default cyan `#00aaff`) |
| Cluster hull | `"hull_cluster"` | `hull_display_mode != "off"`, cluster contacts available | Same as neuron hull but for cluster-only contacts, coloured `hull_cluster_color` (default orange `#ffaa00`) |
| Hull points | `"hull_neuron_pts"` / `"hull_cluster_pts"` | `hull_show_points=True` | Additional point overlay on hull contact positions |
| Cluster info text | `"cluster_info"` | `cluster_description` present | Upper-left text overlay with cluster description summary |
| Scalar bar | (auto) | `show_scalar_bar=True` | Colour-scale legend for the spike metric |
| Axes | (auto) | `show_axes=True` | 3D orientation axes widget |

### Forearm Surface Colouring Modes

When rendering as surface with a Delaunay mesh, three colouring branches exist:

| Branch | Condition | Result |
|--------|-----------|--------|
| Spike + vertex colours | Spike data present, vertex colours match vertex count, `use_vertex_colors=True` | Vertex RGB blended with spike heatmap via `_compose_spike_vertex_colors()`. Spike values mapped through `cmap`, composited onto original PLY vertex colours. |
| Spike, no vertex colours | Spike data present, vertex colour mismatch or disabled | Scalar-based colouring via `cmap` with `clim=[0, max_val]`. NaN vertices shown in `forearm_color`. |
| No spike data | Empty spike DataFrame | Plain vertex-colour mesh or flat `forearm_color` |

### Camera Behaviour

| Scenario | Camera |
|----------|--------|
| `tangent_rotation` available | `view_xy()` — face-on to the tangent plane (surface normal aligned to Z) |
| No tangent rotation | `view_isometric()` fallback |
| Returning to a previously viewed session | Camera restored from `_session_cameras` dict |
| First view of a session | Auto face-on or isometric |

Camera position is captured when leaving a cell and restored when returning.

## Settings Panel (Right)

Fixed-width (260 px) `QScrollArea` with a vertical stack of `QGroupBox`
sections, plus a pinned button bar at the bottom.

### Cluster Info

| Widget | Type | Detail |
|--------|------|--------|
| Info label | `QLabel` | Read-only. Shows `n_touches`, dominant gesture type with percentage, and feature ranges (display_ranges or raw feature_ranges from `cluster_description`). Word-wrap enabled, 11 px font. |

### Background

| Widget | Type | Setting | Default |
|--------|------|---------|---------|
| Colour button | `QPushButton` (60x24) | `bg_color` | `"black"` |

Clicking opens `QColorDialog`.

### Forearm

| Widget | Type | Setting | Default | Detail |
|--------|------|---------|---------|--------|
| Colour button | `QPushButton` | `forearm_color` | `"lightgrey"` | Flat colour when vertex colours disabled |
| Surface toggle | `QCheckBox` | `render_as_surface` | `True` | Switches between Delaunay mesh and point cloud. Disables point-size and spheres controls when surface mode is on. |
| Max edge | `QDoubleSpinBox` | Delaunay threshold | `20.0 mm` | Range 1.0-200.0, step 1.0, suffix " mm". Per-session persistent. Controls maximum triangle edge length in Delaunay triangulation. |
| Point size | `QSlider` | `forearm_size` | `3` | Range 1-20. Only active in point-cloud mode. |
| Opacity | `QSlider` | `forearm_opacity` | `1.0` (100%) | Range 0-100, mapped to 0.0-1.0 |
| Spheres | `QCheckBox` | `forearm_spheres` | `False` | Render points as spheres vs flat. Only active in point-cloud mode. |
| Vertex colours | `QCheckBox` | `use_vertex_colors` | `True` | Use original PLY vertex RGB colours on the mesh |

### Contact Points

| Widget | Type | Setting | Default | Detail |
|--------|------|---------|---------|--------|
| Colormap | `QComboBox` | `cmap` / `contact_cmap` | `"jet"` | Options: YlOrRd, viridis, plasma, inferno, magma, coolwarm, RdBu_r, jet |
| Metric | `QComboBox` | `display_metric` | `"spike_ratio"` | `spike_count` (raw counts) or `spike_ratio` (unique_touch_spike_count) |
| Point size | `QSlider` | `contact_size` | `5.0` | Range 1-30. Used for contact scatter in point-cloud mode. |
| Spheres | `QCheckBox` | `contact_spheres` | `False` | Render contact points as spheres |

### Display

| Widget | Type | Setting | Default | Detail |
|--------|------|---------|---------|--------|
| Scalar bar | `QCheckBox` | `show_scalar_bar` | `True` | Toggle colour-scale legend |
| Hull mode | `QComboBox` | `hull_display_mode` | `"perimeter"` | Perimeter (alpha-shape wireframe), Point cloud, Off |
| Line width | `QDoubleSpinBox` | `hull_line_width` | `2.0` | Range 0.5-20.0, step 0.5. Controls wireframe thickness. |
| Show points | `QCheckBox` | `hull_show_points` | `False` | Overlay raw contact points on top of hull |
| Blob sep | `QDoubleSpinBox` | `hull_blob_sep_mm` | `20.0 mm` | Range 1.0-200.0, step 5.0. KDTree distance for separating disconnected contact blobs before alpha-shape. |
| Alpha radius | `QDoubleSpinBox` | `hull_alpha_mm` | `15.0 mm` | Range 1.0-100.0, step 1.0. Alpha-shape radius for perimeter edge computation. |
| Neuron hull colour | `QPushButton` | `hull_neuron_color` | `"#00aaff"` (cyan) | Colour of the all-clusters hull |
| Cluster hull colour | `QPushButton` | `hull_cluster_color` | `"#ffaa00"` (orange) | Colour of the single-cluster hull |
| Axes | `QCheckBox` | `show_axes` | `False` | Toggle 3D orientation axes widget |

### Action Buttons (pinned below scroll area)

| Button | Behaviour |
|--------|-----------|
| Process | Apply staged settings to current cell — rebuilds scene, persists threshold. No-op if no pending changes. |
| Process All | Pre-compute Delaunay mesh for all visible cells, apply settings to current cell, persist thresholds. |

Settings are staged (buffered) on widget change. The scene is only rebuilt
when Process or Process All is clicked, except for `render_as_surface` which
triggers an immediate rebuild.

## Persistence

State that survives across viewer sessions:

| Data | File | Scope |
|------|------|-------|
| Camera positions | `session_cameras.json` | Per session_id: position, focal_point, up_vector, view_angle |
| Delaunay thresholds | `delaunay_thresholds.json` | Per session_id: max edge length (mm) |
| Exported images | `_gallery_exports/*.png` | `{session_id}__cluster_{label}.png` |

Saved on close (`closeEvent`) via `rf_extraction_io` helpers.

## Caching

Three in-memory caches make cell-to-cell navigation instant after first
computation:

| Cache | Key | Value | Purpose |
|-------|-----|-------|---------|
| `_delaunay_cache` | `(session_id, threshold)` | `trimesh.Trimesh` or `None` | Avoid recomputing Delaunay triangulation per session |
| `_heatmap_cache` | `(session_id, cluster_label, threshold, metric_col)` | `np.ndarray` (vertex scalars) | Avoid recomputing KD-tree scalar mapping |
| `_perimeter_cache` | `(session_id, cluster_key, role, blob_sep_mm, alpha_mm)` | `pv.PolyData` or `None` | Avoid recomputing alpha-shape perimeter edges |

The **Load All** button pre-warms all three caches for every cell.

## Data Model

### `GalleryData` (`rf_gallery_data.py`)

Container for one combo/clusterer pair:

| Field | Type | Description |
|-------|------|-------------|
| `combo_name` | `str` | Feature combination name |
| `clusterer_name` | `str` | Clustering algorithm name |
| `output_base_dir` | `Path` | Root output directory for persistence files |
| `cells` | `Dict[Tuple[str, str], GalleryCell]` | `(session_id, cluster_label)` → cell |
| `session_ids` | `List[str]` | Sorted unique session IDs |
| `cluster_labels` | `List[str]` | Sorted unique cluster labels (numeric ascending, noise last) |
| `gesture_type` | `Optional[str]` | Gesture type for per-type clustering, or `None` |

### `GalleryCell` (`rf_gallery_data.py`)

One renderable (session, cluster) pair:

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | `str` | Session identifier |
| `cluster_label` | `str` | Cluster label (numeric or "noise") |
| `cluster_folder` | `str` | Directory name (e.g. `cluster_00`) |
| `is_noise` | `bool` | True for noise clusters (label -1 or "noise") |
| `forearm_mesh` | `Optional[trimesh.Trimesh]` | Pre-built mesh (deferred to viewer) |
| `forearm_vertices` | `Optional[np.ndarray]` | (N, 3) forearm point cloud |
| `forearm_vertex_colors` | `Optional[np.ndarray]` | (N, 3) RGB uint8 from PLY |
| `tangent_rotation` | `Optional[np.ndarray]` | 3x3 rotation aligning surface normal to Z |
| `spike_counts_df` | `pd.DataFrame` | Columns: x, y, z, spike_count, unique_touch_spike_count |
| `neuron_contacts_xyz` | `np.ndarray` | (N, 3) all-cluster contact points for this session |
| `cluster_contacts_xyz` | `np.ndarray` | (M, 3) this-cluster-only contact points |
| `neuron_touches` | `int` | Total touches across all clusters |
| `neuron_cluster_touches` | `int` | Touches in this cluster |
| `cluster_description` | `dict` | Cluster metadata (n_touches, gesture distribution, feature ranges) |
| `rf_metrics` | `Optional[dict]` | Quantitative RF metrics (area, shape, response, concentration) |

## Helper Functions

| Function | Lines | Purpose |
|----------|-------|---------|
| `_separate_blobs(pts_2d, max_dist_mm)` | 90-101 | Split disconnected point groups by KDTree distance → connected-components labelling |
| `_alpha_shape_edges(points_2d, alpha)` | 104-131 | Compute alpha-shape boundary edges from 2D Delaunay triangulation |
| `_build_perimeter_mesh(pts_3d, blob_sep_mm, alpha_mm)` | 134-164 | Orchestrator: project to 2D → separate blobs → alpha-shape per blob → assemble PyVista line mesh |

## Key Source Locations

| Component | File | Lines |
|-----------|------|-------|
| `_GallerySettings` dataclass | `rf_cluster_gallery_viewer.py` | 64-86 |
| `_ThumbnailWidget` | `rf_cluster_gallery_viewer.py` | 167-213 |
| `RFClusterGalleryViewer` | `rf_cluster_gallery_viewer.py` | 216-1282 |
| `_build_ui()` | `rf_cluster_gallery_viewer.py` | 262-286 |
| `_build_toolbar()` | `rf_cluster_gallery_viewer.py` | 288-316 |
| `_build_sidebar()` | `rf_cluster_gallery_viewer.py` | 318-338 |
| `_build_right_settings_panel()` | `rf_cluster_gallery_viewer.py` | 340-585 |
| `_build_scene()` | `rf_cluster_gallery_viewer.py` | 922-1117 |
| `_on_preload_all_clicked()` | `rf_cluster_gallery_viewer.py` | 1151-1196 |
| `_on_export_all_clicked()` | `rf_cluster_gallery_viewer.py` | 1202-1247 |
| `GalleryCell` / `GalleryData` | `rf_gallery_data.py` | 34-74 |
| `load_gallery_data()` | `rf_gallery_data.py` | 123-282 |
