# RF GUI Comparison: Cluster Gallery Viewer vs Feature-Space Explorer

## Purpose & Concept

| Aspect | Gallery Viewer | Feature-Space Explorer |
|--------|---------------|----------------------|
| **Goal** | Browse pre-computed cluster heatmaps across sessions/clusters | Interactively explore pressure x velocity feature space and see how filtering affects the 3D spike heatmap |
| **Interaction model** | Select-and-view: pick a (session, cluster) cell from a sidebar | Filter-and-explore: drag a rectangle on a scatter plot to define feature ranges |
| **Navigation** | Dual-axis: "By Cluster" or "By Session" via toolbar combo | Single-axis: session combo only |
| **Primary question answered** | "What does each cluster's RF look like per session?" | "How does spike density change when I restrict pressure/velocity ranges?" |

## Layout Architecture

| Aspect | Gallery Viewer | Feature-Space Explorer |
|--------|---------------|----------------------|
| **Layout** | Toolbar + 3-panel horizontal splitter (sidebar, 3D, settings) | Toolbar + 2-panel horizontal splitter (scatter, 3D) + bottom bar |
| **Panel count** | 3 (sidebar 160px, 3D stretches, settings 260px) | 2 (scatter 40%, 3D 60%) |
| **Fixed panels** | Sidebar (160px) + Settings (260px) | None — both panels resize |
| **Settings location** | Dedicated right panel with grouped controls | No settings panel — all parameters are hardcoded |
| **Bottom bar** | None | Gesture-type checkboxes + frame counter |

## Toolbar

| Feature | Gallery Viewer | Feature-Space Explorer |
|---------|---------------|----------------------|
| **Mode selector** | Yes — By Cluster / By Session | No |
| **Entity selector** | Yes — cluster or session depending on mode | Yes — session only |
| **Action buttons** | Load All, Export, Close | None |
| **Complexity** | 5 widgets | 2 widgets (label + combo) |

## 3D View (PyVista)

| Aspect | Gallery Viewer | Feature-Space Explorer |
|--------|---------------|----------------------|
| **Rendering modes** | Surface (Delaunay) OR point cloud, user-selectable | Surface (Delaunay) only |
| **Colormap** | Configurable (8 options: YlOrRd, viridis, plasma, etc.) | Fixed: `"hot"` |
| **Scene layers** | 6+ (forearm, contacts, neuron hull, cluster hull, hull points, info text, scalar bar, axes) | 3 (forearm surface, spike density scalar, scalar bar) |
| **Scalar data** | `spike_count` or `spike_ratio` (user-selected) | `spike_density` (bincount with spike mask) |
| **Hull overlays** | Yes — alpha-shape perimeter with blob separation | No |
| **Text overlays** | Cluster description in upper-left | No |
| **Camera** | Per-session persistence (saved to JSON), tangent-plane face-on or isometric | `view_xy()` only, no persistence |
| **Live update** | Staged — requires "Process" button click | Immediate — 30ms debounced from filter rect drag |

## Interactivity & Filtering

| Aspect | Gallery Viewer | Feature-Space Explorer |
|--------|---------------|----------------------|
| **Primary interaction** | Click thumbnail -> load cell | Drag/resize rectangle on scatter plot |
| **Filter mechanism** | Mode/selection combo narrows the sidebar | Rectangle bounds + gesture checkboxes |
| **Custom widget** | `_ThumbnailWidget` (sidebar cards) | `DraggableFilterRect` (matplotlib patch with 8-zone hit testing) |
| **Debouncing** | None (explicit Process button) | 30ms QTimer on rectangle changes |
| **Gesture filtering** | Implicit (per-type clustering upstream) | Explicit checkboxes in bottom bar |

## Settings & Configurability

| Setting Category | Gallery Viewer | Feature-Space Explorer |
|-----------------|---------------|----------------------|
| **Background color** | Configurable via color picker | Hardcoded |
| **Forearm color** | Configurable | Hardcoded |
| **Surface/point-cloud toggle** | Yes | No (surface only) |
| **Delaunay max edge** | Per-session spin box (1-200mm) | Hardcoded 20mm at load time |
| **Point size** | Slider (1-20) | N/A |
| **Opacity** | Slider (0-100%) | N/A |
| **Hull display** | 3 modes + 6 parameters | N/A |
| **Scalar bar toggle** | Yes | Always on |
| **Total configurable params** | ~18 | 0 |

## Data Model

| Aspect | Gallery Viewer | Feature-Space Explorer |
|--------|---------------|----------------------|
| **Container** | `GalleryData` -> dict of `GalleryCell` | `ExplorerData` with `ExplorerSessionData` |
| **Granularity** | (session, cluster) pairs | Per-frame arrays (pressure, velocity, spikes, vertex indices) |
| **Mesh ownership** | Optional per-cell, deferred to viewer | Pre-computed at load in `ExplorerSessionData` |
| **Key fields** | forearm vertices/colors, spike counts DF, hull contacts, cluster description, RF metrics | pressure, velocity_signed, gesture_types, spikes, frame_vertex_idx |
| **Load function** | `load_gallery_data()` (lines 123-282) | `load_explorer_data()` (lines 40-120) |

## Persistence & Caching

| Aspect | Gallery Viewer | Feature-Space Explorer |
|--------|---------------|----------------------|
| **Camera persistence** | `session_cameras.json` | None |
| **Threshold persistence** | `delaunay_thresholds.json` | None |
| **Export** | Batch PNG export to `_gallery_exports/` | None |
| **In-memory caches** | 3 (Delaunay, heatmap, perimeter) | None explicit — mesh pre-computed at load |

## Complexity

| Metric | Gallery Viewer | Feature-Space Explorer |
|--------|---------------|----------------------|
| **Source lines (viewer)** | ~1282 | ~545 |
| **Helper functions** | 3 (blob separation, alpha-shape, perimeter builder) | 0 |
| **UI builder methods** | 5 | 5 |
| **Data model classes** | 2 (`GalleryData`, `GalleryCell`) | 2 (`ExplorerData`, `ExplorerSessionData`) |

## Summary

The **Gallery Viewer** is a full-featured inspection tool — richly configurable,
multi-axis browsable, with persistent state and export. It answers "what does
each cluster look like?" across every session.

The **Feature-Space Explorer** is a lightweight interactive analysis tool —
minimal settings, no persistence, but with a unique interactive filtering
paradigm (draggable rectangle on a feature scatter). It answers "which
feature-space regions drive neural responses?" within a single session at a time.

They share the same 3D rendering backbone (PyVista `QtInteractor`, Delaunay
meshes, tangent-plane rotation) but diverge completely in navigation model,
configurability depth, and the kind of question they help answer.
