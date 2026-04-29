# Plan: RF Gallery — Batch 3D Image Export & Manual Picker Deprecation

**Date:** 2026-04-28
**Author:** Basil Duvernoy
**Status:** Completed
**Branch:** `feature/rf-gallery-batch-image-export`

---

## Overview

Add an **"Export images"** button to the RF cluster gallery viewer that batch-renders every cell (session × cluster) to PNG using a reproducible auto face-on tangent camera. As a follow-on, deprecate the now-redundant manual camera-angle picker GUI (`RFCameraAnglePicker`) — the gallery has surpassed it for interactive use — while keeping the auto camera task that still produces the `camera_params.json` reference artifact.

## Problem Statement

The gallery viewer (`rf_cluster_gallery_viewer.py`) is the primary post-visualization tool for reviewing per-cluster heatmaps on the forearm surface. There is currently **no way to export the 3D scene as an image** — capturing a figure requires a manual screenshot per cell, which does not scale to N_sessions × N_clusters cells.

In parallel, the older "camera angle mode" feature has two sub-modes:
- **Auto** — computes a face-on tangent-plane camera and writes `camera_params.json`.
- **Manual** — launches `RFCameraAnglePicker` GUI for interactive viewpoint selection.

Since the gallery viewer was built it has duplicated everything the manual picker offers (multi-session navigation, richer rendering, threshold control, in-memory per-session camera) and added more (Delaunay surfaces, hull perimeters, cluster mode). The manual picker has therefore become functionally redundant. Auto mode remains useful as the only producer of the `camera_params.json` reference artifact.

## Goals

### In Scope
1. Add an **"Export images"** button to the gallery toolbar that batch-renders **every** cell in `gallery_data.cells` to PNG.
2. Use a reproducible **auto face-on tangent camera** for each export (matching the geometry the gallery already computes via `tangent_rotation` + `view_xy()`).
3. Write PNGs under `{output_base_dir}/{combo_name}/{clusterer_name}/_gallery_exports/` co-located with existing gallery artifacts.
4. **Deprecate the manual camera-angle picker**: delete `RFCameraAnglePicker` GUI and the `manual` branch of `pick_rf_camera_angle_batch()`. Keep auto mode + `camera_params.json` writing.
5. Flatten the DAG YAML key from `camera_angle_mode.auto.enabled` to `camera_angle_mode.enabled` (auto is the only mode left).

### Out of Scope
- Consuming `camera_params.json` from the gallery export (export uses its own auto camera independently).
- Persisting per-session/per-cluster cameras *from* the gallery to disk.
- Configurable PNG resolution, image format other than PNG, or per-cell scene customization at export time.
- A progress dialog with cancel button (status-bar message is sufficient).
- Migrating any historical `camera_params.json` files written under the legacy schema.

## Success Criteria

- [ ] Toolbar shows a new **"Export images"** button next to the existing Mode/Select/Close controls.
- [ ] Clicking the button writes one PNG per `(session_id, cluster_label)` cell to `{output_base_dir}/{combo}/{clusterer}/_gallery_exports/{session}__cluster_{label}.png`.
- [ ] Every exported image is rendered with the same auto face-on tangent camera (reproducible across runs).
- [ ] After the batch completes, the originally displayed cell and per-session camera state are restored.
- [ ] Status bar shows `Exporting i/n: …` during the run and a completion message at the end.
- [ ] Toolbar is disabled during export to prevent re-entry; re-enabled in `finally`.
- [ ] `RFCameraAnglePicker` GUI file is deleted; no remaining imports reference it.
- [ ] `pick_rf_camera_angle_batch()` accepts a single `enabled: bool` parameter; the `mode` parameter and the manual branch are gone.
- [ ] DAG YAML uses `camera_angle_mode: { enabled: true|false }` at both call sites; the legacy nested `auto.enabled` shape raises a clear error rather than silently degrading.
- [ ] Auto mode still writes `camera_params.json` per session as before.

---

## Technical Design

### Approach

**Part A — Batch export (gallery viewer):**
Add a toolbar button. The slot iterates `self._gallery_data.cells` deterministically. For each cell it calls `self._build_scene(cell)` directly (bypassing `_load_cell`'s camera-restore branch), forces the auto face-on camera (`view_xy()` if `cell.tangent_rotation is not None`, else `view_isometric()`), processes pending Qt events, and saves a PNG via `self.plotter.screenshot(path)`. The originally displayed cell is restored on completion.

Why this approach:
- Reuses the existing `_build_scene()` rendering pipeline — single source of truth for what a "gallery image" looks like.
- The auto face-on logic already exists at `rf_cluster_gallery_viewer.py:1054-1056`; no new geometry math needed.
- `_delaunay_cache` and `_heatmap_cache` warm naturally as we iterate, keeping the loop reasonably fast.
- No coupling to camera-angle-mode artifacts — export is fully self-contained and reproducible.

**Part B — Manual picker deprecation:**
Delete `RFCameraAnglePicker` outright. Trim `pick_rf_camera_angle_batch()` to a single auto code path. Update call sites in `analysis_workflow.py` and the YAML keys in `analyse_workflow_dag.yaml`. Per the fail-fast convention, the YAML loader must raise on the legacy `auto.enabled` shape rather than silently mapping it onto the new key.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Auto face-on tangent camera per cell | Reproducible; matches existing camera-angle auto logic; no external dependency | Cannot honour user-set viewpoints | **Chosen** |
| Honour `camera_params.json` if present, else auto | Respects manual viewpoint choices | Ties export to a deprecated artifact; introduces fallback branching | Rejected |
| Use whatever is in `_session_cameras` (gallery's in-memory state) | "WYSIWYG" for cells the user navigated | Non-reproducible; cells never visited fall back to auto anyway → mixed cameras | Rejected |
| Currently-visible cells only (mode-filtered) | Faster; matches sidebar contents | Requires multiple clicks to cover the whole grid | Rejected — user wants the whole grid in one click |
| Both options via dialog | Maximum flexibility | Extra UX surface for negligible benefit | Rejected |
| Keep manual picker as is, just add export | Smallest change | Leaves redundant code path; YAML still confusing | Rejected — gallery has surpassed picker |
| Fully remove camera-angle-mode (incl. auto + DAG task) | Cleanest end-state | `camera_params.json` artifact is lost; larger blast radius | Rejected for now — keep auto as cheap reference |

### Architecture Changes

**Part A — single file change (gallery viewer):**

- **`code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py`**
  - `_build_toolbar()` (`:285-304`): add `QPushButton("Export images")` between Select combo and Close button.
  - New method `_on_export_all_clicked()`: iterates `sorted(self._gallery_data.cells.keys())`, renders each, screenshots to PNG, restores original state.
  - Helper `_set_auto_face_on_camera(cell)`: extracts the existing tangent-rotation + `view_xy()` / `view_isometric()` + `reset_camera()` logic so it can be called outside the regular `_build_scene` flow.

**Part B — deprecation:**

- **Delete** `code/src/analysis/receptive_field_mapping/gui/rf_camera_angle_picker.py`.
- **`code/src/analysis/receptive_field_mapping/rf_camera_angle_task.py`**: drop the `RFCameraAnglePicker` import and the manual-mode branch in `pick_rf_camera_angle_batch()`. Replace `mode: str` parameter with `enabled: bool`. Function becomes auto-only.
- **`code/scripts/analysis_workflow.py`** (lines 337-348, 426-437, 621-627): update call sites; remove branching that reads `camera_angle_mode.auto.enabled`.
- **`configs/analyse_workflow_dag.yaml`** (lines 272-274 and 301-304): replace
  ```yaml
  camera_angle_mode:
    auto:
      enabled: true
  ```
  with
  ```yaml
  camera_angle_mode:
    enabled: true
  ```
  Drop the comment block referencing manual mode.

PNG output directory:
```
{output_base_dir}/{combo_name}/{clusterer_name}/_gallery_exports/
    └── {session_id}__cluster_{cluster_label}.png
```

---

## Implementation Plan

### Phase 1: Toolbar button + export skeleton
**Goal:** Wire the button and render-loop scaffolding without changing rendering output.
**Started:** 2026-04-29 **Completed:** 2026-04-29

- [x] Task 1.1 — Add `QPushButton("Export images")` in `_build_toolbar()`.
- [x] Task 1.2 — Add `_on_export_all_clicked()` slot stub that prints planned export keys and counts.
- [x] Task 1.3 — Add `_set_auto_face_on_camera(cell)` helper (extract tangent + `view_xy()` / `view_isometric()` + `reset_camera()` from `_build_scene`).
- [x] Task 1.4 — Snapshot/restore logic for the originally displayed cell and `_session_cameras` dict.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py` — toolbar, new slot, helper.

**Dependencies:** None.

### Phase 2: Batch render + screenshot
**Goal:** Produce the PNG files.
**Started:** 2026-04-29 **Completed:** 2026-04-29

- [x] Task 2.1 — Iterate `sorted(self._gallery_data.cells.keys())`; for each call `_build_scene(cell)` then `_set_auto_face_on_camera(cell)`.
- [x] Task 2.2 — Call `QApplication.processEvents()` between render and screenshot so PyVista flushes.
- [x] Task 2.3 — Save PNG via `self.plotter.screenshot(str(out_path))` to the computed path. `mkdir(parents=True, exist_ok=True)` for the export dir.
- [x] Task 2.4 — Status-bar progress: `statusBar().showMessage(f"Exporting {i}/{n}: {sid}__cluster_{lbl}")`. Final completion message with file count.
- [x] Task 2.5 — Disable the toolbar at start, re-enable in `finally` so failures don't leave it locked. Per fail-fast convention, do not swallow exceptions inside the loop — raise after attempting cleanup of the active cell.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py` — body of `_on_export_all_clicked()`.

**Dependencies:** Phase 1.

### Phase 3: Deprecate manual camera-angle picker
**Goal:** Remove redundant code paths.
**Started:** 2026-04-29 **Completed:** 2026-04-29

- [x] Task 3.1 — Delete `code/src/analysis/receptive_field_mapping/gui/rf_camera_angle_picker.py`.
- [x] Task 3.2 — Edit `rf_camera_angle_task.py`: remove the picker import, replace `mode: str` with `enabled: bool`, drop the manual branch; auto-only path remains.
- [x] Task 3.3 — Edit `code/scripts/analysis_workflow.py` call sites (3 locations) to pass the new boolean and drop the mode-selection branching.
- [x] Task 3.4 — Edit `configs/analyse_workflow_dag.yaml` (2 locations) to use `camera_angle_mode: { enabled: true|false }`. Per fail-fast: ensure the YAML loader raises a clear error on the legacy `auto.enabled` shape.
- [x] Task 3.5 — Search for any remaining references to `RFCameraAnglePicker`, `camera_angle_mode.auto`, or manual-mode docs and clean up.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/gui/rf_camera_angle_picker.py` — *deleted*.
- `code/src/analysis/receptive_field_mapping/rf_camera_angle_task.py` — drop manual branch, simplify signature.
- `code/scripts/analysis_workflow.py` — adapt call sites.
- `configs/analyse_workflow_dag.yaml` — flatten YAML key at both task call sites.

**Dependencies:** Phase 2 (so the gallery export is in place before the picker disappears).

---

## Testing Plan

### Unit Tests
- [ ] Add a test for the YAML schema migration: a config with the legacy `camera_angle_mode.auto.enabled` shape must raise (fail-fast convention), not silently load.
- [ ] Add a test for `pick_rf_camera_angle_batch(enabled=False)` — should be a no-op (no `camera_params.json` written).
- [ ] Add a test for `pick_rf_camera_angle_batch(enabled=True)` — should still produce `camera_params.json` with the expected keys (`camera_position`, `focal_point`, `up_vector`, `view_angle`).

### Integration Tests
- [ ] Run `pytest code/tests/` — confirm existing tests still pass after the picker removal.

### Manual Verification
- [ ] Launch the pipeline GUI → enable `visualize_receptive_fields_clustered` with `gallery_viewer: true` → open the gallery.
- [ ] Click **"Export images"**. Confirm `_gallery_exports/` is populated with `len(gallery_data.cells)` PNGs, named `{session}__cluster_{label}.png`.
- [ ] Spot-check 2-3 PNGs visually against the live gallery view (same cluster, same threshold, face-on tangent orientation).
- [ ] After export completes, verify the originally displayed cell is back, the per-session camera memory is intact, and the toolbar is re-enabled.
- [ ] Run the full DAG end-to-end after Phase 3 — confirm `camera_params.json` is still produced when `camera_angle_mode.enabled: true`, and no errors related to the removed manual branch.
- [ ] Manually replace `camera_angle_mode.enabled: true` with the legacy `camera_angle_mode.auto.enabled: true` in a test YAML — confirm a clear error is raised.

### Edge Cases
- [ ] Cell with `forearm_vertices is None` or `tangent_rotation is None` — `_set_auto_face_on_camera` falls back to `view_isometric()`; PNG should still render.
- [ ] Pressing **"Export images"** twice in rapid succession — toolbar disable should prevent the second click from reaching the slot.
- [ ] An `IOError` on PNG write (e.g. read-only directory) — must raise immediately and trigger the `finally` cleanup so the gallery remains usable.

---

## Documentation Plan

- [x] Update `code/src/analysis/CLAUDE.md` — note the deprecation of `RFCameraAnglePicker` and the YAML key flattening.
- [x] Update inline docstring of `pick_rf_camera_angle_batch()` to reflect the simplified signature.
- [ ] No changes needed to root `CLAUDE.md` (no architectural shift) or `README.md` (no new top-level commands).

---

## Rollback Plan

1. **Before deployment:** all changes are local; `git restore` on the modified files reverts the gallery and DAG.
2. **After deployment:**
   - Phase A is purely additive (new button) — revert the relevant commit to remove it.
   - Phase B touches existing tasks. If unwanted, revert the commits in reverse order. The YAML key flattening must be reverted in tandem with the `pick_rf_camera_angle_batch()` signature change to keep the loader and the config in sync.
3. **Data considerations:** `_gallery_exports/` directories created during testing can be deleted at any time — they are derived artifacts. `camera_params.json` files produced by auto mode remain valid and unchanged.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Render lags behind screenshot, producing blank PNGs | Med | Med | Call `QApplication.processEvents()` between `_build_scene` and `screenshot`; verify with manual spot-check. |
| Existing per-session camera state in `_session_cameras` corrupts after restore | Low | Med | Snapshot the dict before the loop, restore at the end inside `finally`. |
| Iterating all cells thrashes Delaunay / heatmap caches on memory-tight systems | Low | Low | Caches are keyed by `(session_id, threshold)` — bounded by N_sessions, not N_cells. Acceptable. |
| Removing the manual picker breaks an undocumented researcher workflow | Low | Med | Status: feature has been superseded by the gallery; verify with researcher before merging Part B. Auto + `camera_params.json` retained as the reference artifact. |
| Legacy YAML configs in the wild silently break | Med | Low | Fail-fast: the loader raises a clear error on `camera_angle_mode.auto.enabled`. Update any tracked configs in the same PR. |
| `RFCameraAnglePicker` is referenced from a place not yet found | Low | Low | Phase 3 Task 3.5 is a final grep to catch stragglers before merging. |

---

## References

- Related plan (in progress): `docs/development/plans/active/rf-gallery-global-process.md` — global deferred-processing model in the gallery; this plan builds on the same toolbar.
- Related plan (in progress): `docs/development/plans/active/rf-cluster-gallery-viewer.md` — original gallery-viewer plan.
- Related plan (in progress): `docs/development/plans/active/rf-cluster-extraction-visualization-split.md` — the extraction/visualization split that introduced the gallery launch path.
- Code: `code/src/analysis/receptive_field_mapping/gui/rf_cluster_gallery_viewer.py` (toolbar, render pipeline).
- Code: `code/src/analysis/receptive_field_mapping/rf_camera_angle_task.py` (auto camera logic to retain).
- Config: `configs/analyse_workflow_dag.yaml` (lines 272-274, 301-304).
