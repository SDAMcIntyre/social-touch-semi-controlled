# RF Feature-Space Explorer — GUI Component Reference

## Purpose

This note catalogues every visual component in the RF Feature-Space Explorer
(`gui/rf_feature_space_explorer.py`), its widget type, position in the layout,
and what it controls. The explorer presents a 2D scatter plot
(pressure x velocity_signed, colored by gesture type) alongside an interactive
PyVista 3D forearm mesh with a spike-density heatmap. A draggable/resizable
rectangle on the scatter filters which frames contribute to the 3D heatmap;
gesture-type checkboxes provide additional filtering.

Companion to
[note-rf-cluster-gallery-gui-components.md](note-rf-cluster-gallery-gui-components.md)
which covers the gallery viewer (a different GUI in the same `gui/` directory).

## Layout Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│ TOOLBAR                                                              │
│ Session: [session_id ▼]                                              │
├──────────────────────────────┬───────────────────────────────────────┤
│                              │                                       │
│  SCATTER PLOT (40%)          │  3D VIEW (60%)                        │
│  (Matplotlib)                │  (PyVista QtInteractor)               │
│                              │                                       │
│  ┌────────────────────────┐  │  ┌─────────────────────────────────┐  │
│  │  * * * *    * * *      │  │  │                                 │  │
│  │  * ┌─────────┐ *      │  │  │  Forearm mesh                   │  │
│  │  * │ Filter  │ *      │  │  │  + spike-density heatmap ("hot") │  │
│  │    │  Rect   │        │  │  │  + scalar bar                    │  │
│  │    └─────────┘        │  │  │                                 │  │
│  │  * * *    * *   *     │  │  │                                 │  │
│  │                        │  │  │                                 │  │
│  └────────────────────────┘  │  └─────────────────────────────────┘  │
│  X: Pressure                 │                                       │
│  Y: Hand Velocity (signed)   │                                       │
├──────────────────────────────┴───────────────────────────────────────┤
│ BOTTOM BAR                                                           │
│ [✓ tap] [✓ stroke_proximal] [✓ stroke_distal] [✓ stroke_unknown]    │
│                                              Frames: 1234 / 5678    │
└──────────────────────────────────────────────────────────────────────┘
```

Root layout: `QVBoxLayout` -> toolbar + `QSplitter(Horizontal)` + bottom bar.
Splitter initial sizes: 40% left (scatter), 60% right (3D view).
Stretch factors: left 2, right 3.

## Toolbar

| Widget | Type | Purpose |
|--------|------|---------|
| Session label | `QLabel` | Static text "Session:" |
| Session selector | `QComboBox` | Switch between loaded sessions. Each entry is a `(label, ExplorerData)` pair. Triggers full scatter + 3D + filter-rect reload on change. |

The toolbar is non-movable (`QToolBar`, `setMovable(False)`).

**Signal:** `currentIndexChanged` -> `_on_session_changed()` -> `_load_session()`.

## Scatter Plot (Left Panel)

Matplotlib `Figure` + `FigureCanvasQTAgg` embedded in a `QVBoxLayout`.

### Data mapping

| Axis | Data field | Label |
|------|-----------|-------|
| X | `ExplorerData.pressure` | "Pressure" |
| Y | `ExplorerData.velocity_signed` | "Hand Velocity (signed)" |

### Gesture-type series

Each gesture type is plotted as a separate scatter series with its own color:

| Gesture type | Color |
|-------------|-------|
| `tap` | `tab:blue` |
| `stroke_proximal` | `tab:orange` |
| `stroke_distal` | `tab:green` |
| `stroke_unknown` | `tab:red` |

Point size: 3, alpha: 0.3, rasterized. Any gesture types not in the known set
are plotted with matplotlib's default color cycle. Legend at `loc="best"`,
`markerscale=3`, `fontsize=8`.

### Performance

Background caching via `canvas.copy_from_bbox()` enables blit-based updates
when the filter rectangle moves, avoiding full redraws. The background is
recaptured on every `draw_event`.

## DraggableFilterRect

Custom interactive overlay on the matplotlib scatter. Not a Qt widget — it is
a matplotlib `Rectangle` patch with mouse-event handlers connected to the
canvas.

### Appearance

| Property | Value |
|----------|-------|
| Edge color | white |
| Face color | white |
| Alpha | 0.15 |
| Line width | 1.5 |
| Z-order | 5 |

### Interaction zones

8-zone hit testing for drag and resize, plus interior move:

| Zone | Action | Detection |
|------|--------|-----------|
| `nw`, `ne`, `sw`, `se` | Corner resize (both axes) | Distance from corner < 5% of rect size (`_CORNER_RADIUS_FRAC`) |
| `n`, `s`, `e`, `w` | Edge resize (single axis) | Distance from edge < 5% of rect size (`_EDGE_PROXIMITY_FRAC`) |
| `move` | Translate (both axes) | Interior click, not near any corner or edge |

### Initial bounds

Computed from the data percentiles on first draw:

| Bound | Value |
|-------|-------|
| x_min | 25th percentile of `pressure` |
| x_max | 75th percentile of `pressure` |
| y_min | 25th percentile of `velocity_signed` |
| y_max | 75th percentile of `velocity_signed` |

### Callback

`on_changed(x_min, x_max, y_min, y_max)` fires on every mouse-motion event
during a drag. The callback blits the updated rectangle patch and starts a
30 ms single-shot `QTimer` to debounce the expensive 3D heatmap update.

### Mouse event connections

| Event | Handler |
|-------|---------|
| `button_press_event` | `_on_press()` — hit-test and record drag mode + press position |
| `motion_notify_event` | `_on_motion()` — update rectangle geometry, fire callback |
| `button_release_event` | `_on_release()` — clear drag state |

## 3D View (Right Panel)

PyVista `QtInteractor` (from `pyvistaqt`), stretches to fill available space.

### Scene contents

| Layer | Mesh name | Description |
|-------|-----------|-------------|
| Forearm surface | `"forearm"` | Delaunay-triangulated forearm mesh (tangent-plane-rotated) |
| Scalar field | `"spike_density"` | Per-vertex spike count, mapped via `"hot"` colormap |
| Scalar bar | (auto) | Color-scale legend for spike density |

### Rendering

- Colormap: `"hot"`
- Scalar bar: shown by default (`show_scalar_bar=True`)
- Camera: `view_xy()` (face-on to the tangent plane after rotation)

### Live update

On every filter update (`_apply_filter_update`):
1. Compute `spike_density` via `np.bincount(frame_vertex_idx[mask], weights=spikes[mask])`
2. Assign to `mesh["spike_density"]`
3. Call `plotter.render()`

## Bottom Bar

Horizontal layout (`QHBoxLayout`) below the splitter, with small margins
(4, 2, 4, 2).

### Gesture-type checkboxes

| Widget | Type | Default | Purpose |
|--------|------|---------|---------|
| `tap` | `QCheckBox` | Checked | Include/exclude tap frames from filter |
| `stroke_proximal` | `QCheckBox` | Checked | Include/exclude proximal strokes |
| `stroke_distal` | `QCheckBox` | Checked | Include/exclude distal strokes |
| `stroke_unknown` | `QCheckBox` | Checked | Include/exclude unknown strokes |

**Signal:** `stateChanged` -> `_on_checkbox_changed()` -> `_apply_filter_update()`
(immediate, no debounce).

### Frame counter

| Widget | Type | Purpose |
|--------|------|---------|
| Frame label | `QLabel` | Displays `"Frames: {filtered_count} / {total_count}"`. Updated on every filter change. Right-aligned via preceding `addStretch()`. |

## Interactive Flow

```
User drags/resizes rectangle
  │
  ├──> _on_rect_changed()
  │      ├── blit updated rectangle patch (fast, no full redraw)
  │      └── start 30ms QTimer (debounce)
  │
  └──> QTimer.timeout
         └──> _apply_filter_update()

User toggles gesture checkbox
  │
  └──> _on_checkbox_changed()
         └──> _apply_filter_update() (immediate)

_apply_filter_update():
  1. rect_mask = pressure in [x_min, x_max] AND velocity in [y_min, y_max]
  2. type_mask = gesture_type in {checked types}
  3. mask = rect_mask AND type_mask
  4. spike_density = np.bincount(frame_vertex_idx[mask], weights=spikes[mask])
  5. mesh["spike_density"] = spike_density
  6. plotter.render()
  7. frame_label.setText(f"Frames: {mask.sum()} / {n_frames}")
```

Note: `blockSignals(True/False)` is used on checkboxes during the read loop
to prevent recursive signal emission.

## Session Management

| Method | Trigger | Behaviour |
|--------|---------|-----------|
| `_on_session_changed(index)` | Combo box selection | Validates index, loads the corresponding `ExplorerData` |
| `_load_session(new_data)` | Called by above | Full reset: clears rect and mesh references, resets frame label, redraws scatter, rebuilds 3D mesh, re-initialises filter rect |

Each session has its own pre-loaded `ExplorerData` with a pre-computed Delaunay
mesh and tangent rotation. Switching is instant aside from the scatter redraw
and 3D scene rebuild.

## Deferred Initialization

The window defers heavy initialization until the first `showEvent`:

| Step | Method | Detail |
|------|--------|--------|
| 1. Position | `showEvent()` | Move to primary screen origin, maximize window |
| 2. Deferred start | `QTimer.singleShot(0, _deferred_start)` | Initialize PyVista interactor, sync render window size |
| 3. Draw | `_deferred_start()` | Draw scatter, render 3D, init filter rect |

On close (`closeEvent`), the PyVista plotter is explicitly closed to release
VTK resources.

## Data Model

### `ExplorerSessionData` (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `forearm_mesh` | `pv.PolyData` | Delaunay mesh, rotated to tangent plane |
| `vertices` | `np.ndarray` (N, 3) | Rotated vertex positions |
| `tangent_rotation` | `np.ndarray` (3, 3) | Rotation matrix aligning surface normal to +Z |

### `ExplorerData` (dataclass)

| Field | Type | Description |
|-------|------|-------------|
| `pressure` | `np.ndarray` (n_frames,) float64 | Per-frame pressure values |
| `velocity_signed` | `np.ndarray` (n_frames,) float64 | Per-frame signed hand velocity |
| `gesture_types` | `np.ndarray` (n_frames,) object | Per-frame gesture labels |
| `spikes` | `np.ndarray` (n_frames,) bool | Per-frame nerve spike flag |
| `frame_vertex_idx` | `np.ndarray` (n_frames,) int | Index of nearest forearm vertex per frame |
| `session_data` | `ExplorerSessionData` | Shared mesh and rotation data |

Property: `n_frames` -> `len(pressure)`.

### `load_explorer_data(series_csv_path, forearm_ply_path, max_edge_mm=20.0)`

Loading pipeline:

1. Read series-augmented CSV, filter to frames with valid `contact_location_x`
2. Extract arrays: `pressure`, `hand_velocity_signed`, `gesture_type`, `Nerve_spike`, contact XYZ
3. Load forearm vertices from PLY (cached via `.npy` sidecar)
4. Build Delaunay mesh (`max_edge_mm=20.0`)
5. Compute tangent-plane rotation (SVD on 50 nearest vertices to contact centroid)
6. Rotate vertices and contact points
7. KDTree: find nearest mesh vertex per contact frame
8. Drop frames with nearest-vertex distance > 15 mm (mesh boundary sparsity)

## Key Source Locations

| Component | File | Lines |
|-----------|------|-------|
| `DraggableFilterRect` | `gui/rf_feature_space_explorer.py` | 49-227 |
| `RFFeatureSpaceExplorer` | `gui/rf_feature_space_explorer.py` | 229-545 |
| `_build_ui()` | `gui/rf_feature_space_explorer.py` | 274-305 |
| `_build_toolbar()` | `gui/rf_feature_space_explorer.py` | 307-317 |
| `_build_bottom_bar()` | `gui/rf_feature_space_explorer.py` | 319-338 |
| `_draw_scatter()` | `gui/rf_feature_space_explorer.py` | 344-383 |
| `_render_3d()` | `gui/rf_feature_space_explorer.py` | 392-414 |
| `_apply_filter_update()` | `gui/rf_feature_space_explorer.py` | 435-465 |
| `_init_filter_rect()` | `gui/rf_feature_space_explorer.py` | 471-489 |
| `ExplorerSessionData` | `rf_explorer_data.py` | 19-23 |
| `ExplorerData` | `rf_explorer_data.py` | 26-37 |
| `load_explorer_data()` | `rf_explorer_data.py` | 40-120 |
| `launch_feature_space_explorer()` | `rf_cluster_pipeline.py` | 817-865 |
| `explore_rf_feature_space_flow()` | `analysis_workflow.py` | 442-459 |
