# Plan: Postprocessing Visualization Pipeline

**Created:** 2026-03-13
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/postprocessing-visualization-pipeline`

---

## Overview

**What:** A new visualization pipeline for postprocessed data, showing PCA-calibrated forearm surfaces with projected contact points — without raw Kinect MKV data.

**Why:** The existing `NeuralKinectViewer` is tightly coupled to raw Kinect MKV playback (FramePreloader, GPU cropping, depth-to-pointcloud). Postprocessed data lives in PCA-calibrated space and needs a dedicated viewer that works directly from CSVs and the calibrated forearm PLY.

**How:** A new `PostprocessedSceneViewer` class (single class, two modes) with a standalone workflow script and DAG config, following the same architectural patterns as `NeuralKinectViewer` (in-place VTK actor updates, deferred render) but without any MKV dependency.

## Problem Statement

After the postprocessing pipeline (ICP registration → PCA calibration → forearm export → contact projection), all spatial data exists in a unified PCA-calibrated coordinate frame. There is currently no way to visualize this final output. The existing `NeuralKinectViewer` cannot serve this purpose because it requires raw MKV files and renders data in the original Kinect camera space, not PCA space. A dedicated viewer is needed to inspect the quality of postprocessed data — particularly contact point projections on the forearm surface.

The hand mesh is a specific challenge: unlike stickers and contact points (which are transformed through the postprocessing pipeline and stored in CSVs), the hand mesh remains in its original Kinect camera space. Rendering it alongside PCA-calibrated data requires chaining ICP + PCA transforms on its vertices at render time.

## Goals

### In Scope
1. Simple viewer mode: static PCA-calibrated forearm PLY + per-frame contact points + neural panels
2. Advanced viewer mode: adds sticker spheres (from CSV, already PCA-calibrated) + hand mesh (ICP+PCA transformed at render time)
3. Standalone workflow script with its own DAG config
4. Neural overlay panels (NeuralDataPanel + StickerVelocityCompass in advanced mode)

### Out of Scope
- Raw Kinect MKV playback or depth-to-pointcloud rendering
- GPU-accelerated point cloud cropping (no point cloud to crop)
- Modifying the existing `NeuralKinectViewer`
- Cross-session forearm unification

## Success Criteria

- [ ] Simple mode renders: forearm PLY (static) + contact points (per-frame) + NeuralDataPanel
- [ ] Advanced mode additionally renders: sticker spheres + hand mesh with correct ICP+PCA transforms
- [ ] Frame slider navigates kinect frames extracted from the postprocessed CSV
- [ ] Hand mesh in advanced mode aligns spatially with stickers and contact points (visual check)
- [ ] Pipeline launches correctly via `--dag-config` with both tasks independently toggleable
- [ ] Sessions missing required files (forearm PLY, hand motion NPZ, etc.) are skipped with warnings

---

## Technical Design

### Approach

Create a new `PostprocessedSceneViewer` class with a `mode` parameter (`"simple"` | `"advanced"`). The viewer uses in-place VTK actor mutation (same pattern as `NeuralKinectViewer`) but is fundamentally simpler: no MKV, no FramePreloader, no GPU cropping. Data comes entirely from postprocessed CSVs and the PCA-calibrated forearm PLY.

For the hand mesh transform chain (advanced mode only):
1. Load per-frame vertices from `HandMotionManager[frame_idx]` (Kinect camera space)
2. Apply the active ICP 4x4 rigid transform for the current frame (from `get_transform_schedule()`)
3. Apply PCA calibration via `PCACalibrationEngine.apply_full_transform()` (two-step: Z-align then XY-align)

This produces hand mesh vertices in the same PCA-calibrated space as stickers and contact points.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| New standalone viewer class | Clean, focused, no MKV baggage | Some UI code similar to NeuralKinectViewer | **Chosen** |
| Adapt NeuralKinectViewer with optional MKV | Reuses existing code | MKV coupling is deep (preloader, GPU crop, LOD stride); would bloat an already 1577-line class | Rejected |
| Extend SceneViewer framework | Reusable abstractions | SceneViewer uses `plotter.clear()` per frame (inferior pattern); would need heavy refactoring | Rejected |

### Architecture Changes

New modules:

```
code/src/postprocessing/gui/
├── __init__.py
└── postprocessed_scene_viewer.py    — PostprocessedSceneViewer class (~500-700 lines)

code/scripts/
└── postprocess_visualization.py     — Workflow script (entry point)

configs/
└── postprocess_visualization_dag.yaml
```

**Reused utilities (no modifications needed):**
- `PCACalibrationEngine.apply_full_transform()` — `code/src/postprocessing/xyz_reference_from_gestures/calibration_pca_engine.py`
- `apply_rigid_transform()`, `get_transform_schedule()` — `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py`
- `HandMotionManager` — `code/src/preprocessing/motion_analysis/hand_tracking/models/hand_motion_manager.py`
- `ForearmCatalog`, `ForearmFrameParametersFileHandler`, `get_forearms_with_fallback` — `code/src/preprocessing/forearm_extraction/`
- `_parse_contact_points_cell()` — `code/src/merging/gui/neural_kinect_scene_viewer.py` (will extract or duplicate)
- `NeuralDataPanel`, `StickerVelocityCompass` — `code/src/merging/gui/` (imported cross-package)
- `XYZDataFileHandler` — `code/src/preprocessing/stickers_analysis/`
- `DagConfigHandler`, `resolve_session_configs` — `code/src/utils/pipeline/`
- `KinectConfig`, `KinectConfigFileHandler` — `code/src/primary_processing/`

**Key technical notes:**
- `apply_full_transform()` does NOT mutate its input: it creates a new array in `_apply_z_alignment` (line 73: `return (coords - mean_1) @ R1.T`), then modifies `coords_step1` in-place (lines 92-93). Safe to call without `.copy()` on source.
- `get_transform_schedule()` returns `[(start_frame, T_4x4), ...]` — use `bisect` to resolve active transform per frame.
- `CalibrationResult.from_dict(json_dict)` loads the PCA matrices from `pca-xyz_transformation-matrices.json`.
- Output CSV filenames from `project_contacts_onto_forearm` match input names: `output_dir / input_csv.name`.

---

## Implementation Plan

### Phase 1: Viewer Class Foundation
**Goal:** Create the `PostprocessedSceneViewer` with simple mode (forearm + contact points + neural panel)
**Started:** —
**Completed:** —

- [ ] Create `code/src/postprocessing/gui/__init__.py`
- [ ] Create `code/src/postprocessing/gui/postprocessed_scene_viewer.py`:
  - Data loading: read postprocessed CSV, extract kinect rows (`dropna(subset=['time_kinect'])`), parse contact points per frame
  - Load forearm PLY via `open3d.io.read_point_cloud()` → convert to PyVista mesh (static, rendered once)
  - Build Qt UI: PyVista interactor + frame slider + NeuralDataPanel (imported from `merging.gui`)
  - `_init_actors()`: register forearm mesh (static) + contact point PolyData (dynamic, DeepCopy per frame)
  - `_update_frame(idx)`: update contact points, neural panel cursor, render
  - Deferred initial render via `showEvent` + `QTimer.singleShot(0, ...)` (per knowledge base)
  - `ResetCameraClippingRange()` before `plotter.render()` (per knowledge base)
- [ ] Extract `_parse_contact_points_cell()` into a shared location or duplicate in the new module

**Files Modified:**
- `code/src/postprocessing/gui/__init__.py` — **new**
- `code/src/postprocessing/gui/postprocessed_scene_viewer.py` — **new**

**Dependencies:** None

### Phase 2: Advanced Mode (Stickers + Hand Mesh)
**Goal:** Add sticker rendering and hand mesh with ICP+PCA transform chain
**Started:** —
**Completed:** —

- [ ] Add sticker sphere actors (3 per color, `SetPosition()` per frame from CSV columns — already PCA-calibrated)
- [ ] Add `StickerVelocityCompass` widgets (imported from `merging.gui`)
- [ ] Add hand mesh actor (`_mesh_hand`), lazy-loaded from `HandMotionManager`
- [ ] Implement transform chain helper:
  ```python
  def _transform_hand_to_pca(self, vertices: np.ndarray, frame_idx: int) -> np.ndarray:
      # 1. Resolve active ICP transform via bisect on schedule
      icp_T = resolve_icp_for_frame(self._icp_schedule, frame_idx)
      coords = vertices.copy()  # safety: HandMotionManager may cache
      if icp_T is not None:
          coords = apply_rigid_transform(coords, icp_T)
      # 2. Apply PCA calibration
      coords = PCACalibrationEngine.apply_full_transform(coords, self._pca_calib)
      return coords
  ```
- [ ] Load ICP schedule from `registration_transforms.json` via `ForearmCatalog.load_registration_transforms()` + `get_transform_schedule()`
- [ ] Load PCA calibration from `pca-xyz_transformation-matrices.json` via `CalibrationResult.from_dict()`
- [ ] Update `_update_frame()`: skip stickers/handmesh in simple mode, render them in advanced mode
- [ ] In-place hand mesh vertex update: same triangle-count optimization as `NeuralKinectViewer` (mesh.points setter when face count unchanged, full DeepCopy otherwise)

**Files Modified:**
- `code/src/postprocessing/gui/postprocessed_scene_viewer.py` — extend with advanced mode

**Dependencies:** Phase 1

### Phase 3: Workflow Script + DAG Integration
**Goal:** Wire everything into a launchable pipeline
**Started:** —
**Completed:** —

- [ ] Create `code/scripts/postprocess_visualization.py`:
  - Follow pattern of `merging_pipeline_neuron_to_kinect_visualisation.py`
  - CuPy early-init (imports from `preprocessing` trigger the issue)
  - `resolve_postprocessed_paths(config: KinectConfig)` → derive all paths:
    - `contact_projected_csv`: `session_merged_output_dir / "blocks_contact_projected" / {csv_name}`
    - `forearm_pca_ply`: `session_merged_output_dir / "forearm_pca_calibrated" / "{session_id}_forearm_pca_calibrated.ply"`
    - `pca_calib_json`: `session_merged_output_dir / "blocks_pca_calibrated" / "pca-xyz_transformation-matrices.json"`
    - `hand_motion_path`: `video_processed_output_dir / "kinematics_analysis" / "{stem}_handmodel_motion.npz"`
    - `forearm_metadata_path`: `session_processed_output_dir / "forearm_pointclouds" / "{session_id}_arm_roi_metadata.json"`
  - `run_single_session_pipeline()` for simple mode + `run_single_session_pipeline_advanced()` for advanced mode
  - Batch dispatcher (sequential, one viewer at a time)
  - `QCoreApplication.processEvents()` between viewers (VTK context cleanup)
- [ ] Create `configs/postprocess_visualization_dag.yaml`:
  ```yaml
  parameters:
    kinect_configs: valid_configs_ST13-01
  tasks:
    view_postprocessed_simple:
      enabled: true
      description: "View PCA-calibrated forearm + projected contact points + neural panels"
      options: {}
      depends_on: []
    view_postprocessed_advanced:
      enabled: false
      description: "View forearm + contacts + stickers + hand mesh (ICP+PCA transformed)"
      options: {}
      depends_on: []
  ```

**Files Modified:**
- `code/scripts/postprocess_visualization.py` — **new**
- `configs/postprocess_visualization_dag.yaml` — **new**

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] `_transform_hand_to_pca()` produces expected output for known ICP matrix + PCA calibration on synthetic vertices
- [ ] `resolve_icp_for_frame()` returns correct transform for frame indices at boundaries of the schedule
- [ ] Contact point parsing from CSV cells handles `"[]"`, `NaN`, valid arrays, and malformed strings

### Integration Tests
- [ ] Workflow script resolves all paths correctly from a valid `KinectConfig`
- [ ] Viewer launches and closes cleanly without errors for both simple and advanced modes
- [ ] DAG handler correctly skips disabled tasks and runs enabled ones

### Manual Verification
- [ ] Run simple mode on one session — confirm forearm PLY renders as static surface, contact points update per frame, NeuralDataPanel tracks cursor
- [ ] Run advanced mode on same session — confirm stickers appear as spheres at correct PCA-calibrated positions, hand mesh aligns with stickers/contact points
- [ ] Visually compare hand mesh position in advanced mode vs. sticker positions — they should be spatially coherent
- [ ] Slider navigation: drag, play/pause, keyboard arrows all function correctly
- [ ] Toggle visibility of individual actors (forearm, contacts, stickers, hand mesh)

### Edge Cases
- [ ] Session with no forearm PLY (Stage 3 returned None) → warning logged, viewer skipped
- [ ] Session with no hand motion NPZ → advanced mode falls back gracefully (hand mesh disabled, stickers still shown)
- [ ] Session with no registration transforms → advanced mode skips hand mesh (no ICP available)
- [ ] CSV rows where all contact_points are empty → frame renders with no contact geometry
- [ ] Toggle between simple/advanced tasks in DAG config → only the enabled task runs

---

## Documentation Plan

- [ ] No README/CLAUDE.md changes needed (internal pipeline addition)
- [ ] Inline docstrings in new modules
- [ ] Module-level docstring in `postprocessed_scene_viewer.py` documenting the transform chain

---

## Rollback Plan

1. Delete the 4 new files (2 Python, 1 YAML, 1 `__init__.py`)
2. No existing files are modified — rollback has zero impact on the rest of the codebase
3. No data is generated or modified by visualization (read-only pipeline)

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Hand mesh frame index misalignment with CSV kinect frames | Med | High | HandMotionManager uses original kinect frame indices; CSV `time_kinect` rows map 1:1. Add assertion on frame count match at load time. |
| ICP transform schedule frame indices don't match CSV frame indices | Low | High | Both use the same video stem for resolution. Log transform schedule at startup for visual verification. |
| `NeuralDataPanel` import from `merging.gui` creates circular dependency | Low | Low | Both packages import from `preprocessing`; no circular path exists. If issues arise, extract panel to `utils/gui/`. |
| Hand mesh visually misaligned despite correct transforms | Med | Med | Add a debug mode that renders untransformed hand mesh alongside transformed one for comparison. Start with a known-good session where sticker alignment is verified. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Viewer Foundation | ~400 lines new code | None |
| Phase 2: Advanced Mode | ~200 lines additions | Phase 1 |
| Phase 3: Workflow + DAG | ~150 lines new code + YAML | Phase 2 |

---

## References

- Related Plans: `docs/development/plans/pending/contact-point-forearm-projection.md`
- Existing viewer: `code/src/merging/gui/neural_kinect_scene_viewer.py`
- PCA engine: `code/src/postprocessing/xyz_reference_from_gestures/calibration_pca_engine.py`
- Knowledge base: `docs/development/knowledge-base/note-cupy-import-order.md`
