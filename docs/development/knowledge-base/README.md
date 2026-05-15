# Knowledge Base

Problem-resolution notes for engineering challenges encountered in this
codebase.  Each note captures a **specific problem, its root cause, the
constraints that shaped the fix, and the reusable pattern** that resolves the
class of problem — not just the single instance.

These are developer-facing documents, not usage guides.

---

## Index

### Engineering notes (`note-*`)

| Note | Problem class | Key files |
|------|--------------|-----------|
| [Open3D SceneWidget Layout — Nested Widget Sizing](note-open3d-scenewidget-layout.md) | Embedding a `SceneWidget` inside `gui.Vert` causes it to collapse on first mouse interaction (grey background / widget disappears). | `arm_segmentation.py` — `_make_hue_range_circle`, `_display_pointcloud` |
| [CuPy Import Order with Preprocessing Packages](note-cupy-import-order.md) | Importing CuPy after the `preprocessing` package tree crashes with `TypeError: Alias 'bool8' was removed in NumPy 2.0`. | Entry-point scripts, `neural_kinect_scene_viewer.py` |
| [ICP Registration Constraints for Forearm Point Clouds](note-forearm-icp-registration.md) | Aligning multiple forearm point cloud snapshots to a common reference frame using point-to-plane ICP. | `forearm_registrator.py`, `csv_spatial_transformer.py` |
| [Somatosensory Metric Units and Calculations](note-somatosensory-units-and-calculations.md) | Coordinate system (mm from Kinect SDK), velocity (mm/frame), contact depth (mm), and contact area (mm²) derivations and known labeling bug. | `objects_interaction_processor.py`, `touch_analysis.py` |
| [Qt `itemChanged` Signal Recursion](note-qt-itemchanged-signal-recursion.md) | `setData()` called inside an `itemChanged` handler re-emits `itemChanged`, causing infinite recursion. Guard with `blockSignals(True/False)`. | `task_panel.py` — `_on_item_changed` |
| [Azure Kinect RGB ↔ Depth Parallax (near range)](note-azure-kinect-rgb-depth-parallax.md) | Per-pixel correspondence between color frame and `transformed_depth` is offset by ~10 px median / 25 px worst at 500–800 mm; downstream consumers that index `point_cloud[v_rgb, u_rgb]` inherit the error. | `xyz_extractor_centroid.py`, `xyz_extractor_ellipse_depth.py`, `kinect_pointcloud_wrapper.py`, `kinect_rgb_depth_viewer.py` |
| [Kinect depth access — single path](note-kinect-depth-access-single-path.md) | The parallax correction inside `KinectFrame` is only effective if every depth read flows through `KinectFrame` / `KinectMKV` / `KinectPointCloudView`; direct `pyk4a` imports outside `data_access/` silently bypass it. Enforced via an allowlist + grep audit. | `kinect_mkv_manager.py`, `kinect_pointcloud_wrapper.py`, `kinect_rgb_depth_viewer.py`, and the `pyk4a` allowlist |
| [3D-to-2D Surface Projection Algorithms for RF Heatmaps](note-3d-to-2d-surface-projection-algorithms.md) | Algorithm catalogue (tangent-plane, exponential map, geodesic MDS, LSCM/ARAP, cylindrical unwrap) for projecting forearm point clouds to 2D `(u, v)` coordinates. Includes distortion analysis, implementation sketches, and pipeline architecture vision. | `tangent_plane_alignment.py`, `define_forearm_mesh.py`, `rf_cluster_visualizer.py` |
| [RF Cluster Gallery Viewer — GUI Component Reference](note-rf-cluster-gallery-gui-components.md) | Catalogue of every visual component in the interactive PyQt5 + PyVista gallery viewer: layout, widgets, settings, 3D scene layers, caching, persistence, and data model. | `gui/rf_cluster_gallery_viewer.py`, `rf_gallery_data.py` |
| [RF Feature-Space Explorer — GUI Component Reference](note-rf-feature-space-explorer-gui-components.md) | Catalogue of visual components in the interactive scatter + 3D heatmap explorer: layout, widgets, filter rectangle, gesture checkboxes, session switching, and data model. | `gui/rf_feature_space_explorer.py`, `rf_explorer_data.py` |
| [Matplotlib Blitting in NeuralDataPanel — Blit + Threshold-Resnap Pattern](note-neural-kinect-viewer-blitting.md) | Per-frame cursor update via `copy_from_bbox` + `blit` reduces 30–80 ms full redraws to ~1–3 ms; `tight_layout=True` desync trap and invalidation-site discipline. | `neural_kinect_scene_viewer.py` — `NeuralDataPanel` |

### Bug reports (`bug-*`)

| Report | Summary | Related note |
|--------|---------|--------------|
| [bug-cupy-bool8-import-order.md](bug-cupy-bool8-import-order.md) | `bool8` TypeError caused by CuPy import order | [note-cupy-import-order.md](note-cupy-import-order.md) |
| [bug-neural-kinect-viewer-initial-render.md](bug-neural-kinect-viewer-initial-render.md) | Neural Kinect Viewer initial render issue | — |
| [bug-hue-circle-drag-not-working.md](bug-hue-circle-drag-not-working.md) | Hue circle drag interaction not working | — |
| [bug-rf-explorer-nearest-vertex-distance.md](bug-rf-explorer-nearest-vertex-distance.md) | RF Explorer crashes: contact frames >15mm from forearm mesh vertices | — |

---

## Structure of each note

1. **Symptom** — observable failure, as it appeared during development.
2. **Investigation** — what was tried, what was eliminated.
3. **Root Cause** — the underlying engine/framework constraint.
4. **Architecture Constraints** — why the naive fix doesn't work.
5. **Fix Applied** — the minimal code change and the reasoning behind it.
6. **Reusable Pattern** — a checklist / template for future occurrences.
7. **References** — related bug reports, plans, and source locations.

---

## Relevance check — subagent pattern

The planning procedure requires a knowledge-base relevance check before any
plan is finalised.  The check is performed by an Explore subagent launched in
parallel with other pre-planning research:

**Subagent prompt template:**
> Read `docs/development/knowledge-base/README.md`. For each note listed in
> the index, assess whether its problem class overlaps with the feature being
> planned. Read the full content of any relevant notes and return a concise
> summary of what applies: which constraints to respect, which approaches were
> rejected and why, and which code patterns to reuse.

The subagent output feeds directly into the plan's **Technical Design** section
(Alternatives Considered, Architecture Constraints).  If no notes are relevant,
the check still satisfies the planning procedure — record "no applicable notes"
in the plan.

---

## Adding a new note

**Engineering notes** (root-cause analysis and reusable patterns):
1. Create `docs/development/knowledge-base/note-<slug>.md` using the 7-section
   structure above.
2. Add a row to the **Engineering notes** table in this file.

**Bug reports** (incident records):
1. Create `docs/development/knowledge-base/bug-<slug>.md`.
2. Add a row to the **Bug reports** table in this file.
3. If the bug has a corresponding engineering note, add a cross-link in the
   "Related note" column and link back to the bug from the note's References
   section.
