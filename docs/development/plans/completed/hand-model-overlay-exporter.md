# Plan: Hand-Model Overlay Exporter GUI

**Date:** 2026-05-11
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-05-12 10:42
**Base Branch:** `feature/session-comparison-heatmap`
**Branch:** `feature/hand-model-overlay-exporter`

---

## Overview

Add a new viewer in the **Preprocess [Visualisation]** workflow that lets the
researcher pick any (session, block-order) pair from two dropdowns and export
an MP4 of the RGB video with the MANO hand-mesh wireframe overlaid on top.
The renderer logic already exists as a proof-of-concept under
`code/scripts/__misc/`; this plan promotes it into a proper module and wraps
it in a PyQt5 GUI driven by a Save-As dialog with a prefilled filename.

## Problem Statement

The `track_hands_model` task in `preprocess_workflow_kinect_auto` produces
per-frame MANO mesh data (`vertices_pixel`, `faces`) saved as
`{rgb_stem}_handmodel_tracked_hands.pkl` under `kinematics_analysis/`, but
there is currently no integrated way to *visually verify* that output. The
only existing tools are:

- `code/src/preprocessing/motion_analysis/hand_tracking/gui/handmesh_3d_viewer.py` —
  a static 3D scatter/trisurf viewer (no video, no 2D RGB context).
- `code/scripts/__misc/hamer_semi-controlled_generate-video.py` — a
  one-off script with hard-coded paths in `__main__` and no GUI wrapper.

For quality control of the `track_hands_model` step researchers need to be
able to scrub a representative block, see the mesh overlaid on the actual
RGB hand, and ideally export the video for review or sharing. Today that
requires editing the hard-coded paths in the misc script for every block.

## Goals

### In Scope
1. A PyQt5 GUI window with two cascading dropdowns (Session → Block-order)
   that lists all blocks in the resolved `kinect_configs` of the DAG.
2. An **Export Video** button that opens a `QFileDialog.getSaveFileName`
   dialog with the filename prefilled to a canonical convention.
3. Headless overlay rendering driven from the GUI with a progress dialog
   and Cancel support.
4. A new viewer task `view_hand_model_overlay` registered in the Preprocess
   [Visualisation] DAG config so the GUI is launched from the existing
   launcher.
5. Extracting `BatchVideoRenderer` from the misc script into a clean,
   reusable module under `code/src/preprocessing/motion_analysis/hand_tracking/`.

### Out of Scope
- In-window video playback / scrubbing (export-only; no on-the-fly preview
  of the overlay is required for this feature).
- GPU-accelerated rendering (the CPU `cv2.polylines` path used by
  `BatchVideoRenderer` is sufficient for the volumes encountered).
- Batch-export of all blocks at once (per-block manual export is what the
  user asked for — keep the surface minimal).
- Re-running `track_hands_model`. The GUI assumes the pickle already
  exists; it surfaces clear feedback when it does not.

## Success Criteria

- [ ] New task `view_hand_model_overlay` appears in the Preprocess
      [Visualisation] DAG, defaults to `enabled: false`, and is recognised
      by the existing launcher GUI.
- [ ] When enabled, the dispatcher opens **one** GUI window with the full
      set of resolved blocks (session-level, not per-block).
- [ ] Session dropdown lists every distinct `session_id` from the
      configured blocks; Block-order dropdown lists the `block_id`s of the
      currently-selected session.
- [ ] Export button is disabled when the RGB MP4 or hand-model pickle for
      the selected (session, block) does not exist, with a status-bar
      message identifying which file is missing (fail-fast convention).
- [ ] Save-As dialog opens with default filename
      `{rgb_stem}_handmodel_overlay.mp4` in the block's
      `kinematics_analysis/` folder.
- [ ] Rendered MP4 has the yellow wireframe MANO mesh visible on each
      frame that has tracking data, with a "No Tracking Data" label on
      frames where it does not.
- [ ] Progress dialog ticks per frame; pressing Cancel cleanly releases
      the `cv2.VideoWriter` and leaves a (partial) MP4 on disk.

---

## Technical Design

### Approach

Promote the existing `BatchVideoRenderer` POC into a proper module, then
wrap it in a thin PyQt5 GUI that mirrors the cascading-combo + export
pattern already used by
`code/src/analysis/receptive_field_mapping/gui/touch_playback_explorer.py`.
Register a new task `view_hand_model_overlay` in the Preprocess
visualisation DAG and short-circuit `run_batch_sequentially` for it — the
GUI is session-level (runs once for the whole batch) rather than per-block.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|---|---|---|---|
| **Promote `BatchVideoRenderer` to a module + new GUI wrapper** | Reuses validated render logic; matches existing viewer pattern; small surface area | Requires the `preprocessing/` scope-toggle to land code under that tree | **Chosen** |
| Add the GUI inside `code/scripts/__misc/` and call it from the dispatcher | No scope-toggle needed | `__misc/` is documented as scratch; future maintainers will not look there; not importable from packages | Rejected |
| Add overlay rendering to the existing `handmesh_3d_viewer.py` | Single GUI file for hand-tracking outputs | Conflates 3D-scene viewing with 2D RGB export; doubles widget complexity; the 3D viewer is Tkinter while the rest of preprocessing GUI code is PyQt5 | Rejected |
| Make `view_hand_model_overlay` a per-block task inside `run_single_session_visualization` | Symmetric with existing viewer tasks | Forces N windows for N blocks; researcher wants to *choose* a block from dropdowns | Rejected |

### Knowledge-base relevance check

| Note | Relevance | Application |
|---|---|---|
| [note-cupy-import-order.md](../../knowledge-base/note-cupy-import-order.md) | Applicable — the GUI module imports from `preprocessing.common` (`VideoMP4Manager`, `ColorFormat`) | The dispatcher entry-point script must keep its current import order; no new CuPy imports added in the renderer or GUI |
| [note-qt-itemchanged-signal-recursion.md](../../knowledge-base/note-qt-itemchanged-signal-recursion.md) | Applicable — cascading session→block combos must not re-emit on programmatic repopulation | Use `blockSignals(True)` while clearing+refilling the block combo on session change |
| [note-rf-cluster-gallery-gui-components.md](../../knowledge-base/note-rf-cluster-gallery-gui-components.md) | Reference only — informs widget layout choices | Adopt the toolbar + status-bar pattern; export-button gating mirrors the gallery's "Export images" handler |

Other knowledge-base notes (parallax, ICP, somatosensory units, 3D-to-2D
projection, RF explorer) do not overlap with this feature's problem class.

### Architecture Changes

New files:

```
code/src/preprocessing/motion_analysis/hand_tracking/
├── handmesh_overlay_renderer.py          # NEW — extracted renderer + data manager
└── gui/
    └── handmesh_overlay_exporter.py      # NEW — PyQt5 GUI with dropdowns + export
```

Modified files:

```
configs/preprocess_workflow_kinect_visualisation_dag.yaml   # +1 task
code/scripts/preprocess_workflow_kinect_visualisation.py    # +dispatcher branch
code/scripts/__misc/hamer_semi-controlled_generate-video.py # reduced to a CLI shim or removed
```

Module API:

```python
# handmesh_overlay_renderer.py
class HandTrackingDataManager:
    def __init__(self, pkl_path: Path) -> None: ...
    def get_hand_geometry(self, frame_index: int) -> Optional[tuple[np.ndarray, np.ndarray]]: ...

class BatchVideoRenderer:
    def __init__(self, video_path: Path, data_path: Path, output_path: Path,
                 target_fps: int = 30) -> None: ...
    def render(self, progress_cb: Optional[Callable[[int, int], bool]] = None) -> None:
        """Render the overlay video.

        progress_cb is called per frame with (current, total). If it returns
        True the render is cancelled, the writer released, and a partial MP4
        remains on disk. On any other failure the writer is released and the
        exception re-raised (fail-fast convention).
        """
```

```python
# handmesh_overlay_exporter.py
def launch_handmesh_overlay_exporter(blocks: list[KinectConfig]) -> None:
    """Open the GUI modally and block until the window closes."""

class HandmeshOverlayExporter(QMainWindow):
    # Toolbar: [Session combo] [Block combo] [Export Video button]
    # Status bar: RGB path / pickle path / existence
    # _on_session_changed → repopulates block combo with blockSignals guard
    # _on_block_changed → resolves paths, enables/disables Export
    # _on_export_clicked → QFileDialog → QProgressDialog → BatchVideoRenderer.render(...)
```

Dispatcher hook in `preprocess_workflow_kinect_visualisation.py::__main__`:

```python
if main_dag_handler.can_run('view_hand_model_overlay'):
    configs = [
        KinectConfig(
            config_data=KinectConfigFileHandler.load_and_resolve_config(bf),
            database_path=project_data_root,
        )
        for bf in block_files
    ]
    from preprocessing.motion_analysis.hand_tracking.gui.handmesh_overlay_exporter \
        import launch_handmesh_overlay_exporter
    launch_handmesh_overlay_exporter(configs)
    main_dag_handler.mark_completed('view_hand_model_overlay')
```

---

## Implementation Plan

### Phase 1: Extract renderer module
**Goal:** Land `BatchVideoRenderer` as a real, importable module with a
progress-callback hook.

- [x] Task 1.1 — Create
      `code/src/preprocessing/motion_analysis/hand_tracking/handmesh_overlay_renderer.py`
      with `HandTrackingDataManager` and `BatchVideoRenderer` copied from the
      misc script.
- [x] Task 1.2 — Convert string paths to `Path`, add type hints, add a
      `progress_cb: Optional[Callable[[int, int], bool]]` parameter to
      `render`.
- [x] Task 1.3 — Replace the blanket `except Exception: log` swallow inside
      `render` with a `try/finally` that calls `writer.release()` and
      re-raises the exception, complying with the fail-fast convention.
- [x] Task 1.4 — Either delete
      `code/scripts/__misc/hamer_semi-controlled_generate-video.py` or
      reduce its `__main__` to a thin shim that imports from the new
      module.

**Files Modified:**
- `code/src/preprocessing/motion_analysis/hand_tracking/handmesh_overlay_renderer.py` — new module
- `code/scripts/__misc/hamer_semi-controlled_generate-video.py` — reduced or removed

**Dependencies:** None.

### Phase 2: GUI module
**Goal:** Land the PyQt5 exporter window with cascading dropdowns and the
Save-As / progress flow.

- [x] Task 2.1 — Create
      `code/src/preprocessing/motion_analysis/hand_tracking/gui/handmesh_overlay_exporter.py`
      with `HandmeshOverlayExporter(QMainWindow)` and the
      `launch_handmesh_overlay_exporter(blocks)` function.
- [x] Task 2.2 — Build `_configs: dict[tuple[str, str], KinectConfig]`
      index from the input list (keyed by `(session_id, block_id)`); build
      `_session_to_blocks: dict[str, list[str]]` for combo population.
- [x] Task 2.3 — Wire cascading combos: `_on_session_changed` clears and
      repopulates `_block_combo` under a `blockSignals(True)` guard
      (per [note-qt-itemchanged-signal-recursion.md](../../knowledge-base/note-qt-itemchanged-signal-recursion.md)).
- [x] Task 2.4 — Implement `_on_block_changed` to resolve `rgb_video_path`
      and `pkl_path`, update the status bar, and enable/disable the export
      button based on `Path.exists()`.
- [x] Task 2.5 — Implement `_on_export_clicked`:
      `QFileDialog.getSaveFileName` with the canonical prefilled name and
      directory, then drive `BatchVideoRenderer.render(progress_cb=...)`
      through a `QProgressDialog` whose Cancel button maps to the
      callback's `True` return.

**Files Modified:**
- `code/src/preprocessing/motion_analysis/hand_tracking/gui/handmesh_overlay_exporter.py` — new module
- (no existing files in this phase)

**Dependencies:** Phase 1.

### Phase 3: DAG integration
**Goal:** Wire the new GUI into the existing Preprocess [Visualisation]
workflow so it launches from `launch_pipeline_gui.py`.

- [x] Task 3.1 — Append `view_hand_model_overlay` task to
      `configs/preprocess_workflow_kinect_visualisation_dag.yaml`
      (enabled: false, description, depends_on: []). Use `ruamel.yaml`
      semantics — preserve existing comments and key order when editing.
- [x] Task 3.2 — In
      `code/scripts/preprocess_workflow_kinect_visualisation.py::__main__`,
      add the dispatcher branch shown above before
      `run_batch_sequentially`. Mark the task completed via
      `main_dag_handler.mark_completed`.
- [x] Task 3.3 — Sanity-check: launch the pipeline GUI and confirm the new
      task is listed under Preprocess → Kinect [Visualisation] with an
      Enabled checkbox.

**Files Modified:**
- `configs/preprocess_workflow_kinect_visualisation_dag.yaml` — +1 task
- `code/scripts/preprocess_workflow_kinect_visualisation.py` — dispatcher hook

**Dependencies:** Phases 1 and 2.

---

## Testing Plan

### Unit Tests
- [ ] `HandTrackingDataManager.get_hand_geometry` returns `None` for
      out-of-range frame indices and for frames with empty / missing
      `hands` lists.
- [ ] `HandTrackingDataManager.get_hand_geometry` returns vertices/faces
      with the expected dtypes (`float32`, `int32`) for a synthetic
      single-frame pickle.

(Both tests use a tiny synthetic pickle stitched in the test — no real
data file required. `conftest.py` already stubs `preprocessing.*` package
roots so the module can be imported without OpenGL/Kinect deps.)

### Integration Tests
- [ ] Not applicable — `BatchVideoRenderer.render` writes an actual MP4
      via `cv2.VideoWriter` and the GUI requires a Qt event loop, both
      out of scope for headless `pytest`. Manual verification covers
      this.

### Manual Verification
- [ ] **Scope toggle**: comment out the `permissions.deny` array in
      `.claude/settings.local.json` so files under `preprocessing/` are
      reachable for the assistant to edit.
- [ ] **Enable**: set `view_hand_model_overlay.enabled: true` in
      `configs/preprocess_workflow_kinect_visualisation_dag.yaml` and
      point `parameters.kinect_configs` at a directory that contains at
      least one block whose pickle already exists.
- [ ] **Launcher**: run
      `python code/scripts/launch_pipeline_gui.py` → Preprocess → Kinect
      [Visualisation] → Run.
- [ ] **Dropdowns**: confirm session combo lists every distinct
      `session_id` and that changing session repopulates the block combo.
- [ ] **Status**: status bar shows the resolved RGB + pickle paths and
      flags missing files.
- [ ] **Save-As**: click Export → dialog opens with default filename
      `{stem}_handmodel_overlay.mp4` in the `kinematics_analysis/` folder
      of the selected block.
- [ ] **Render**: progress dialog ticks per frame; output MP4 plays in any
      standard player and shows the yellow wireframe on hand frames.
- [ ] **Cancel**: pressing Cancel during render leaves a partial MP4 and
      closes the writer cleanly (no zombie file handles).

### Edge Cases
- [ ] Selected block has the RGB MP4 but no pickle → Export disabled,
      status bar reports "Hand model: missing".
- [ ] Selected block has the pickle but no RGB MP4 → Export disabled,
      status bar reports "RGB: missing".
- [ ] Frames within the pickle that contain `api_response.error` or empty
      `hands` lists → those frames render with a "No Tracking Data" label
      (existing renderer behaviour, kept).
- [ ] User selects an output filename outside the
      `kinematics_analysis/` folder → still works (no path constraint).

---

## Documentation Plan

- [ ] No README/CLAUDE.md update required — the feature follows existing
      Preprocess viewer conventions already documented in
      `code/src/preprocessing` and the root CLAUDE.md.
- [ ] No new knowledge-base note required — the implementation does not
      introduce a new reusable pattern beyond what is already captured
      in `note-qt-itemchanged-signal-recursion.md`.
- [ ] Inline docstrings on `BatchVideoRenderer.render` and
      `launch_handmesh_overlay_exporter` are sufficient; no separate user
      guide needed.

---

## Rollback Plan

The change is additive — it does not modify any existing render path,
DAG, or output. Rollback is straightforward:

1. **DAG**: revert
   `configs/preprocess_workflow_kinect_visualisation_dag.yaml` to drop
   the `view_hand_model_overlay` task.
2. **Dispatcher**: revert the `__main__` branch in
   `preprocess_workflow_kinect_visualisation.py`.
3. **Modules**: delete the two new files under
   `code/src/preprocessing/motion_analysis/hand_tracking/`.
4. **Misc script**: restore
   `code/scripts/__misc/hamer_semi-controlled_generate-video.py` from
   git if it was deleted in Phase 1.

No data migrations, no on-disk artifacts to clean up (the only outputs
are MP4s the user explicitly saved via the Save-As dialog).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Scope-toggle requirement (must comment out `preprocessing/` deny in `.claude/settings.local.json` for the assistant to edit the new files) | High | Low | Document it explicitly in the Manual Verification section; restore the deny on completion |
| `VideoMP4Manager` is unavailable in tests because `conftest.py` stubs `preprocessing.*` | Low | Low | Confine unit tests to `HandTrackingDataManager` (pure-Python, no video dep); manual verification covers the renderer |
| Large blocks (>30k frames) render slowly with `cv2.polylines` on a single thread | Medium | Low | Progress dialog with Cancel button; document the partial-MP4 outcome on cancel |
| `BatchVideoRenderer` constructor logs FileNotFoundError at INFO level today | Low | Low | Phase 1 audit replaces the blanket `except` with an explicit raise; GUI gates Export on `Path.exists()` to surface missing inputs before render |
| GUI launched per-block by accident if the dispatcher hook is mis-placed inside `run_single_session_visualization` | Low | Medium | The hook lives in `__main__`, **before** `run_batch_sequentially`; success criterion #2 verifies "one window for the whole batch" |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — Renderer module | 1–2 hours | None |
| Phase 2 — GUI module | 2–3 hours | Phase 1 |
| Phase 3 — DAG integration | < 1 hour | Phases 1 + 2 |

---

## References

- Misc POC: `code/scripts/__misc/hamer_semi-controlled_generate-video.py`
- 3D viewer (sibling location convention):
  `code/src/preprocessing/motion_analysis/hand_tracking/gui/handmesh_3d_viewer.py`
- GUI template (cascading combos + export):
  `code/src/analysis/receptive_field_mapping/gui/touch_playback_explorer.py`
- DAG dispatcher: `code/scripts/preprocess_workflow_kinect_visualisation.py`
- DAG config: `configs/preprocess_workflow_kinect_visualisation_dag.yaml`
- Knowledge-base notes consulted:
  - [note-cupy-import-order.md](../../knowledge-base/note-cupy-import-order.md)
  - [note-qt-itemchanged-signal-recursion.md](../../knowledge-base/note-qt-itemchanged-signal-recursion.md)
  - [note-rf-cluster-gallery-gui-components.md](../../knowledge-base/note-rf-cluster-gallery-gui-components.md)
