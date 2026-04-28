# Plan: RF Cluster Gallery Viewer

**Date:** 2026-04-27
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/rf-cluster-extraction-visualization-split`
**Started:** 2026-04-27
**Phase 2 Completed:** 2026-04-27
**Phase 3 Completed:** 2026-04-27
**Phase 4 Completed:** 2026-04-27

---

## Overview

**What:** An interactive PyQt5 gallery viewer for exploring RF cluster heatmaps after visualization completes. Shows a scrollable thumbnail sidebar alongside a single interactive 3D PyVista view, with two navigation modes: by-cluster (all sessions for one cluster) and by-session (all clusters for one session).

**Why:** The current pipeline renders static PNGs without any way to compare heatmaps across sessions or clusters interactively. Researchers need to visually inspect how receptive fields vary across sessions within a cluster, and across clusters within a session, with the ability to rotate and zoom the 3D forearm surface.

**How:** Build a `QMainWindow` subclass following the existing `RFCameraAnglePicker` pattern — thumbnail snapshots in a sidebar, full interactive `pyvistaqt.QtInteractor` in the main area, loading data from the extraction artifacts already on disk.

## Problem Statement

After `run_cluster_rf_visualization()` finishes, the only output is a set of disconnected PNG files scattered across cluster directories. To compare results, the researcher must open individual files in an image viewer and mentally map between them. There is no way to:

- See all sessions for a cluster at a glance
- See all clusters for a session at a glance
- Interactively rotate or zoom into a heatmap
- Switch between different views without leaving the pipeline

## Goals

### In Scope

1. PyQt5 gallery viewer (`RFClusterGalleryViewer`) with thumbnail sidebar + interactive 3D main view
2. Two navigation modes: "By Cluster" and "By Session"
3. Load all data from extraction artifacts — forearm mesh, spike counts, hull contacts, metadata
4. Auto-generate thumbnails via PyVista screenshot on viewer startup
5. Noise cluster (label -1) included with dimmed visual marker
6. Settings panel for colormap, opacity, point size, display metric toggle
7. Pipeline integration: optional launch after each cluster group via DAG config

### Out of Scope

- Re-rendering or modifying the static PNGs on disk
- Live parameter changes that trigger pipeline re-runs
- 2D projection rendering in the gallery (3D only)
- Editing or saving data from the viewer
- Multi-combo/multi-clusterer comparison (viewer scoped to one combo/clusterer pair)

## Success Criteria

- [x] Gallery viewer launches after visualization completes when enabled in DAG config
- [x] "By Cluster" mode: selecting a cluster shows thumbnails for all sessions; clicking one loads the 3D view
- [x] "By Session" mode: selecting a session shows thumbnails for all clusters; clicking one loads the 3D view
- [x] 3D view shows forearm mesh + spike heatmap with hull wireframes, matching the static PNG content
- [x] Noise cluster appears with dimmed label/border in thumbnail sidebar
- [x] Settings panel controls (colormap, opacity, display metric) update the 3D view live
- [x] Viewer is blocking — pipeline waits for user to close the window before continuing
- [ ] Viewer handles edge cases: empty clusters, missing forearm PLY, single-session datasets

---

## Technical Design

### Approach

Follow the `RFCameraAnglePicker` architecture: standalone `QMainWindow` with `pyvistaqt.QtInteractor`, data loaded upfront into dataclasses, scene rebuilt on navigation. The key difference is the thumbnail sidebar and two-axis navigation (cluster × session grid vs camera picker's flat session list).

**Data source:** Extraction artifacts from `run_cluster_rf_extraction()` — the viewer reads `.npy`, `.csv`, and `.json` files from the output directory. It never touches aggregated CSVs or `clustering_dir`.

**Thumbnail generation:** On startup, iterate all cells (session × cluster pairs with data), render each scene briefly in the `QtInteractor`, capture a screenshot, store as `QPixmap`. This is a one-time cost (~0.5s per cell).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Thumbnail grid + single 3D view** | Scales to any count, low GPU, overview + detail | Thumbnail generation adds startup time | **Chosen** |
| Multi-pane 3D grid (N×M QtInteractors) | All views interactive simultaneously | Heavy GPU/memory, limited to ~9 panes | Rejected — doesn't scale |
| Static PNG gallery (QLabel grid) | Simplest, fastest load | No interactive 3D, no rotation/zoom | Rejected — user requires 3D navigation |
| Single 3D view + combo only | Minimal code, lightest | No visual overview, no comparison | Rejected — user requires grid overview |

### Architecture Changes

**New files:**

| File | Purpose |
|------|---------|
| `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py` | Main `QMainWindow` widget |
| `code/src/analysis/receptive_field_mapping/rf_gallery_data.py` | Data loading: scan output dir, load artifacts into dataclasses |

**Modified files:**

| File | Changes |
|------|---------|
| `code/src/analysis/receptive_field_mapping/gui/__init__.py` | Export `RFClusterGalleryViewer` |
| `code/src/analysis/receptive_field_mapping/__init__.py` | Export gallery viewer if needed |
| `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` | Add gallery launch call after visualization |
| `code/scripts/analysis_workflow.py` | Pass `gallery_viewer` config to visualization flow |
| `configs/analyse_workflow_dag.yaml` | Add `gallery_viewer` option to `visualize_receptive_fields_clustered` |

**Reused from existing code:**

| Utility | File | Purpose |
|---------|------|---------|
| `mesh_to_pyvista()` | `rf_surface_utils.py` | Convert trimesh → PyVista PolyData |
| `load_or_build_forearm_mesh()` | `rf_surface_utils.py` | Load forearm mesh from PLY |
| `map_scalars_to_mesh()` | `rf_surface_utils.py` | Interpolate spike counts to mesh vertices |
| `load_forearm_vertices()` | `rf_data_loader.py` | Load forearm vertices from .npy cache |
| `load_neuron_contacts()` | `rf_extraction_io.py` | Load neuron contact arrays |
| `load_cluster_session_data()` | `rf_extraction_io.py` | Load per-session spike data + cluster contacts |
| `load_neuron_touches()` | `rf_extraction_io.py` | Load touch counts per session |
| `load_neuron_cluster_touches()` | `rf_extraction_io.py` | Load cluster-specific touch counts |
| `compute_tangent_plane_rotation()` | `tangent_plane_alignment.py` | Align forearm surface for consistent camera |
| `align_points()` | `tangent_plane_alignment.py` | Apply rotation to points |
| `apply_rotation_to_mesh()` | `rf_surface_utils.py` | Apply rotation to mesh |
| QApplication reuse pattern | `rf_camera_angle_task.py:239-246` | Safe app creation |

### Data Model

```python
@dataclass
class GalleryCell:
    """One renderable cell: a specific (session, cluster) pair."""
    session_id: str
    cluster_label: str
    cluster_folder: str          # e.g. "cluster_05"
    is_noise: bool               # True if cluster_label == -1 or "noise"

    # Geometry
    forearm_mesh: Optional[trimesh.Trimesh]
    forearm_vertices: Optional[np.ndarray]   # (N, 3)
    tangent_rotation: Optional[np.ndarray]   # (3, 3) or None

    # Spike data
    spike_counts_df: pd.DataFrame            # (x, y, z, spike_count, unique_touch_spike_count)

    # Hull contacts
    neuron_contacts_xyz: np.ndarray          # (N, 3) all-cluster
    cluster_contacts_xyz: np.ndarray         # (M, 3) this cluster

    # Metadata
    neuron_touches: int
    neuron_cluster_touches: int
    cluster_description: dict
    rf_metrics: Optional[dict]

    # Thumbnail (populated after first render)
    thumbnail: Optional[QPixmap] = None


@dataclass
class GalleryData:
    """All data for one combo/clusterer pair."""
    combo_name: str
    clusterer_name: str
    cells: Dict[Tuple[str, str], GalleryCell]  # (session_id, cluster_label) → cell
    session_ids: List[str]                      # ordered
    cluster_labels: List[str]                   # ordered, noise last
```

### Widget Layout

```
+-------------------------------------------------------+
| [Mode: By Cluster v]  [Select: cluster_05 v]  [Close] |  ← Toolbar
+---------------+---------------------------------------+
|               |                                       |
| ┌───────────┐ |                                       |
| │ thumb S1  │ |                                       |
| │ session_1 │ |     [Interactive 3D PyVista View]     |
| └───────────┘ |                                       |
| ┌───────────┐ |     Forearm mesh + spike heatmap      |
| │ thumb S2  │ |     Hull wireframes                   |
| │ session_2 │ |     Rotate / Zoom / Pan               |
| └───────────┘ |                                       |
| ┌───────────┐ |                                       |
| │ thumb S3  │ |                                       |
| │ session_3 │ |                                       |
| └───────────┘ |                                       |
| ┌───────────┐ |                                       |
| │ [dimmed]  │ |                                       |  ← noise marker
| │ noise     │ |                                       |
| └───────────┘ |                                       |
|   (scroll)    |                                       |
+---------------+---+-------------------------------+---+
|                    | Settings ▼                    |   |  ← Collapsible
|                    | Colormap: [YlOrRd v]          |   |
|                    | Metric:   [spike_count v]     |   |
|                    | Opacity:  [====•=====]        |   |
+--------------------+-------------------------------+---+
```

### 3D Scene Building

For each `GalleryCell`, the scene build follows the `RFCameraAnglePicker._build_scene()` pattern:

1. `plotter.clear()`
2. Set background (from settings)
3. Load forearm mesh → `mesh_to_pyvista()` → apply tangent rotation if available
4. Map spike scalars to mesh vertices → `map_scalars_to_mesh()`
5. Add forearm mesh: `plotter.add_mesh(mesh_pv, scalars="spike_values", cmap=cmap, clim=..., nan_color="lightgrey", smooth_shading=True)`
6. Add hull wireframes:
   - Neuron contacts hull (cyan `#00aaff`, dashed)
   - Cluster contacts hull (orange `#ffaa00`, dashed)
   - Use `scipy.spatial.ConvexHull` → extract edges → `pv.PolyData(lines=...)` → `plotter.add_mesh(..., style='wireframe')`
7. Set camera: top-down if tangent rotation applied, else normal-based
8. `plotter.render()`

### Thumbnail Generation

On startup, after loading all `GalleryCell` data:

```python
for cell in cells_to_display:
    self._build_scene(cell)
    img = self.plotter.screenshot(return_cimg=True)
    pixmap = numpy_to_qpixmap(img).scaled(THUMB_W, THUMB_H, Qt.KeepAspectRatio)
    cell.thumbnail = pixmap
```

Estimated cost: ~0.3-0.5s per cell at 200×150 resolution. For 5 sessions × 10 clusters = 50 cells, ~15-25s startup. Acceptable for a review tool.

### DAG Config

```yaml
visualize_receptive_fields_clustered:
  enabled: true
  options:
    force_processing: true
    cluster_groups: [pressure_velocity_mean]
    projection_method: cylindrical_unwrap
    disjoint_mask_distance_mm: 8.0
    gallery_viewer: true           # NEW — launch gallery after each group
    camera_angle_mode:
      auto:
        enabled: false
  depends_on: [extract_receptive_fields_clustered]
```

### Knowledge Base Constraints

- **Qt signal recursion** (`note-qt-itemchanged-signal-recursion.md`): Guard all `setData()` / `setText()` calls inside signal handlers with `blockSignals(True/False)`. Applies to cluster/session combo box switching.
- **Open3D SceneWidget layout** (`note-open3d-scenewidget-layout.md`): Don't nest the `QtInteractor` inside auto-layout containers. Make it a direct child and manage geometry via `resizeEvent()` if needed. The `RFCameraAnglePicker` handles this correctly — follow the same pattern.

---

## Implementation Plan

### Phase 1: Data Loading Module
**Goal:** Build `rf_gallery_data.py` that scans the extraction output directory and loads all artifacts into `GalleryData` / `GalleryCell` dataclasses.

**Tasks:**
- [ ] 1.1 — Define `GalleryCell` and `GalleryData` dataclasses
- [ ] 1.2 — Implement `load_gallery_data(output_dir, combo_name, clusterer_name) → GalleryData` that:
  - Scans `cluster_*` directories
  - For each cluster, loads `cluster_description.json`, `neuron_cluster_touches.json`, `rf_metrics.json`
  - For each session subdirectory, loads `session_spike_counts.csv`, `cluster_contacts_xyz.npy`
  - Loads session-level `neuron_contacts_xyz.npy`, `forearm_vertices.npy`
  - Loads forearm mesh via `load_or_build_forearm_mesh()`
  - Computes tangent plane rotation per session
  - Orders clusters numerically (noise last)
- [ ] 1.3 — Handle edge cases: missing forearm PLY (cell rendered without mesh), empty cluster (cell skipped), missing session data (warning logged)

**Files:**
- `code/src/analysis/receptive_field_mapping/rf_gallery_data.py` — **NEW**

**Dependencies:** None (reads from existing extraction artifacts)

### Phase 2: Gallery Viewer Widget
**Goal:** Build the `RFClusterGalleryViewer` QMainWindow with thumbnail sidebar and interactive 3D view.

**Tasks:**
- [x] 2.1 — Create `RFClusterGalleryViewer(QMainWindow)` with constructor accepting `GalleryData`
- [x] 2.2 — Build toolbar: mode combo ("By Cluster" / "By Session"), selection combo (cluster or session list depending on mode), close button
- [x] 2.3 — Build thumbnail sidebar: `QScrollArea` with vertical `QVBoxLayout` of `QLabel` widgets. Each thumbnail shows a small snapshot + text label. Noise clusters get a dimmed border/background.
- [x] 2.4 — Build main 3D view: `pyvistaqt.QtInteractor` as the central widget
- [x] 2.5 — Implement `_build_scene(cell: GalleryCell)`:
  - Clear plotter
  - Add forearm mesh with spike scalars (spike_count or spike_ratio based on settings)
  - Add hull wireframes (neuron + cluster)
  - Set camera (top-down after tangent rotation, or normal-based)
  - Render
- [x] 2.6 — Implement thumbnail click → load cell into 3D view
- [x] 2.7 — Implement mode switching: changing "By Cluster" ↔ "By Session" repopulates the selection combo and thumbnail sidebar
- [x] 2.8 — Implement thumbnail generation on startup: iterate visible cells, render + screenshot each, populate thumbnails

**Files:**
- `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py` — **NEW**
- `code/src/analysis/receptive_field_mapping/gui/__init__.py` — export new class

**Dependencies:** Phase 1

### Phase 3: Pipeline Integration
**Goal:** Wire the gallery viewer into the visualization pipeline and DAG config.

**Tasks:**
- [x] 3.1 — Add `gallery_viewer` option to `visualize_receptive_fields_clustered` in `analyse_workflow_dag.yaml`
- [x] 3.2 — In `run_cluster_rf_visualization()`: after rendering all PNGs for a combo/clusterer pair, if `gallery_viewer=True`, call the gallery launch function
- [x] 3.3 — Create `launch_gallery_viewer(output_dir, combo_name, clusterer_name)` in `rf_cluster_pipeline.py` or a dedicated module:
  - Load `GalleryData` via `load_gallery_data()`
  - Create/reuse `QApplication`
  - Instantiate `RFClusterGalleryViewer(data)`
  - `viewer.show()` + `app.exec_()`
- [x] 3.4 — Pass `gallery_viewer` option through `analysis_workflow.py` Prefect flow to the pipeline function

**Files:**
- `configs/analyse_workflow_dag.yaml` — add option
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — launch call
- `code/scripts/analysis_workflow.py` — pass option through

**Dependencies:** Phase 2

### Phase 4: Settings Panel and Polish
**Goal:** Add rendering controls and UX polish.

**Tasks:**
- [x] 4.1 — Create `_GallerySettings` dataclass: `bg_color`, `forearm_color`, `forearm_opacity`, `contact_cmap`, `contact_size`, `display_metric` (spike_count / spike_ratio), `show_scalar_bar`, `show_hull_wireframes`
- [x] 4.2 — Build collapsible settings panel at the bottom of the window (or right sidebar): colormap dropdown, metric toggle, opacity slider, hull toggle
- [x] 4.3 — Settings changes trigger `_apply_setting_change()` → rebuild scene with preserved camera
- [x] 4.4 — Highlight selected thumbnail in the sidebar (blue border or background tint)
- [x] 4.5 — Add metadata tooltip on thumbnail hover: cluster description summary, touch count, RF area
- [x] 4.6 — Noise cluster thumbnails: dimmed border (grey), italic label, sorted last in the list

**Files:**
- `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py` — extend

**Dependencies:** Phase 3

---

## Testing Plan

### Unit Tests
- [ ] `test_rf_gallery_data.py` — `load_gallery_data()` with a mock output directory: correct cell count, correct ordering, noise detection
- [ ] `test_rf_gallery_data.py` — Missing forearm PLY → cell created with `forearm_mesh=None`
- [ ] `test_rf_gallery_data.py` — Empty cluster (no session subdirs) → cluster skipped with warning
- [ ] `test_rf_gallery_data.py` — Single-session dataset → one cell per cluster

### Integration Tests
- [ ] Launch gallery viewer with a real output directory → window opens, thumbnails visible, 3D view renders
- [ ] Switch modes → sidebar repopulates correctly

### Manual Verification
- [ ] Run full pipeline with `gallery_viewer: true` → viewer pops up after visualization
- [ ] "By Cluster" mode: select a cluster → all sessions visible, click one → 3D view loads
- [ ] "By Session" mode: select a session → all clusters visible, click one → 3D view loads
- [ ] Rotate/zoom/pan the 3D view → responsive, no crashes
- [ ] Settings panel: change colormap → 3D view updates, change metric → colours update
- [ ] Noise cluster: appears last in sidebar with dimmed visual marker
- [ ] Close viewer → pipeline continues to next cluster group (or finishes)

### Edge Cases
- [ ] Cluster with zero sessions → not shown in sidebar
- [ ] Session missing forearm PLY → 3D view shows spike points only (point cloud fallback)
- [ ] Single cluster in group → "By Cluster" mode shows one entry, "By Session" mode still works
- [ ] Very large dataset (50+ cells) → thumbnail generation doesn't freeze UI (consider progress bar)

---

## Documentation Plan

- [ ] Update `docs/development/knowledge-base/note-rf-cluster-visualization-overview.md` to mention the gallery viewer
- [ ] Inline docstrings for `RFClusterGalleryViewer`, `GalleryCell`, `GalleryData`, `load_gallery_data()`
- [ ] No user guide needed — viewer is self-explanatory and launched via existing DAG config

---

## Rollback Plan

1. **Before deployment:**
   - The `gallery_viewer` DAG config defaults to `false` — existing workflows unaffected
   - No changes to existing visualization pipeline logic; the viewer is additive

2. **Rollback procedure:**
   - Set `gallery_viewer: false` in DAG config
   - Delete the two new files (`rf_gallery_data.py`, `rf_cluster_gallery_viewer.py`)
   - Revert the 3 modified files (pipeline launch call, workflow pass-through, DAG config)

3. **Data considerations:**
   - No new files written to disk. The viewer reads existing extraction artifacts.
   - No migrations needed.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Thumbnail generation too slow for large datasets (50+ cells) | Med | Med | Add progress bar during startup; consider lazy thumbnail generation (render on scroll) |
| PyVista `QtInteractor` conflicts with existing `QApplication` state | Low | High | Follow proven `QApplication.instance()` reuse pattern from `rf_camera_angle_task.py` |
| VTK/PyVista not installed in all environments | Med | High | Gate import behind `try/except`; skip gallery viewer with warning if unavailable |
| `blockSignals` oversight causes infinite recursion | Low | Med | Knowledge base constraint documented; follow `RFCameraAnglePicker` signal patterns exactly |
| Forearm mesh loading OOM for many sessions | Low | Med | Meshes are ~50k vertices each (~1.2 MB); 50 sessions = ~60 MB — manageable |
| Tangent plane rotation inconsistent across sessions | Low | Med | Each session's rotation is computed independently (same as existing pipeline) |

---

## References

- Existing GUI pattern: `code/src/analysis/receptive_field_mapping/gui/rf_camera_angle_picker.py`
- Pipeline integration pattern: `code/src/analysis/receptive_field_mapping/rf_camera_angle_task.py`
- Extraction artifacts: `code/src/analysis/receptive_field_mapping/rf_extraction_io.py`
- Visualization overview: `docs/development/knowledge-base/note-rf-cluster-visualization-overview.md`
- Knowledge base: `docs/development/knowledge-base/note-qt-itemchanged-signal-recursion.md`
