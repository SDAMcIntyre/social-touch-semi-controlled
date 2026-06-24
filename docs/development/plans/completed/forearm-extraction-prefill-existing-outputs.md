# Plan: Forearm Extraction Manual — Pre-fill GUIs with Existing Outputs

**Date:** 2026-04-14
**Author:** Basil Duvernoy
**Status:** Completed
**Completed:** 2026-04-17 15:34
**Branch:** `feature/forearm-extraction-prefill-existing-outputs`

---

## Overview

When the manual forearm extraction pipeline is re-run with `force_processing: true`, each interactive task (`extract_forearm`, `curate_forearm`, `define_normals`, `register_forearms`) currently opens its GUI with factory defaults, forcing the operator to manually retune parameters they already validated. This plan extends the pre-fill pattern already used by `curate_forearm` (which loads `_curation_metadata.json` and passes `existing_removed_indices=` to its GUI) to every task in the pipeline, so prior saved outputs become the GUI defaults. The operator can then click Save/Accept to validate without rework, or edit and re-save normally. Validation is invisible — no separate preview modal.

## Problem Statement

The manual forearm extraction DAG (`configs/preprocess_pipeline_extract_forearm_manual_dag.yaml`) has six tasks. When an operator needs to tweak a single frame, they set `force_processing: true` and re-run. Today:

- `extract_forearm` silently re-applies prior params to sliders (via `self.params` deep-update at `arm_segmentation.py:123`), but the re-save always happens even on pure validation → mtime refreshes, but there is no way to close without save when outputs are fine.
- `curate_forearm` already pre-fills prior removed indices — the pattern to mirror.
- `clean_forearm` and `build_mesh` re-run deterministic algorithms (byte-identical result, wasted work).
- `define_normals` re-opens with factory slider defaults despite `_with_normals_metadata.json` containing the exact saved parameters.
- `register_forearms` (`RegistrationWorkbench.__init__` at `registration_workbench.py:107`) hard-codes defaults at lines 131–134 and widget values at 183/193/200/206/213; saved `_registration_transforms.json` is never read back.

This friction discourages fine-grained iteration and makes forced re-runs costly.

## Goals

### In Scope

1. Add a shared `refresh_output_mtimes()` helper in `code/src/utils/should_process_task.py`.
2. For each task with a GUI (extract, curate, normals, register), seed GUI state from saved outputs when they exist.
3. For deterministic no-GUI tasks (clean, mesh), skip reprocessing and refresh output mtimes when outputs exist under `force=True`.
4. For GUI tasks, when the operator closes without editing, refresh output mtimes so the staleness check in `should_process_task` does not refire.
5. Verify the `register_forearms` workbench supports the existing JSON schema documented in `docs/development/knowledge-base/note-forearm-icp-registration.md` (section 7).

### Out of Scope

- Any visual "preview + accept/reject" modal. Per the operator's direction, validation must be hidden inside the existing GUIs.
- Changes to the `should_process_task` decision tree itself.
- Changes to the automatic (non-manual) pipeline.
- Re-saving byte-identical outputs via re-processing — we touch mtimes instead.
- Support for arbitrary back-compat metadata schemas — assume schemas written by the same codebase version.

## Success Criteria

- [x] `refresh_output_mtimes()` exists in `should_process_task.py` and mirrors the `p.touch()` / `PermissionError` handling at lines 88–93.
- [ ] With `force_processing: true`, each of extract/curate/normals/register GUIs opens pre-populated with prior saved values.
- [ ] Closing any GUI without edits leaves output file contents identical (SHA-256 unchanged) but mtimes newer.
- [ ] Deterministic tasks (clean, mesh) skip reprocessing when outputs exist, touching mtimes only.
- [ ] A subsequent run with `force_processing: false` reports all tasks up-to-date (no GUIs open) on the touched outputs.
- [ ] Genuine input staleness still cascades downstream — input modifications trigger the staleness check correctly.

---

## Technical Design

### Approach

Adopt the `curate_forearm` pattern uniformly:

1. In each task script, capture `outputs_existed = all(p.exists() for p in output_paths)` *before* calling `should_process_task`.
2. If `should_process_task` returns True and `outputs_existed`, load the outputs into a "prior state" bundle and pass it to the GUI/model constructor.
3. The GUI opens with those values seeded into its controls.
4. On save: outputs are overwritten as today (mtime refreshes naturally).
5. On close-without-save (cancel): if `outputs_existed`, call `refresh_output_mtimes(output_paths)` and return without modifying contents.

For `clean_forearm` and `build_mesh`, there is no GUI, so the pattern reduces to: when `outputs_existed`, skip reprocessing entirely and call `refresh_output_mtimes`.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Hidden pre-fill in existing GUIs (this plan) | Matches operator's direction; reuses existing GUIs; no new surfaces | Each GUI needs its own small change | **Chosen** |
| Separate preview/validate modal before GUI | Uniform; centralised | Operator explicitly rejected this: "No hard coded display for validation" | Rejected |
| Change `should_process_task` to silently skip when outputs exist | Zero per-task work | Loses the ability to re-edit at all; contradicts `force_processing: true` intent | Rejected |
| Re-save byte-identical outputs on accept | Simple — no touch helper | Wastes I/O; for extract_forearm's PLY and normals re-save, round-trip may not be byte-identical | Rejected |

### Architecture Changes

**New helper:** `refresh_output_mtimes(output_paths: PathInput) -> None` in `code/src/utils/should_process_task.py`. Public, no side-effects beyond `Path.touch()`. Uses the same `PathLike`/`PathInput` aliases as the rest of that module.

**Constructor additions:**

- `PointCloudModel.__init__` (normals estimation) — add `is_centered`, `scale_factor`, `normals_flipped` kwargs. k_neighbors / radius / hybrid_tree / align_with_viewpoint / viewpoint already supported.
- `RegistrationWorkbench.__init__` — add `existing_state: Optional[dict] = None`; store on `self._existing_state`; seed `_last_mode`/`_last_method`/`_last_max_dist`/`_last_max_iter` when provided.
- `ArmSegmentation` — add a public `was_modified: bool` attribute set by the interactive-process callback so the caller can distinguish edit-then-save from close-without-edit.

**Widget seeding points (inside `RegistrationWorkbench._build_gui`):**

- Line 183 — `_combo_mode.selected_index`
- Line 193 — `_combo_canonical.selected_index`
- Line 200 — `_combo_method.selected_index`
- Line 206 — `_edit_max_dist.double_value`
- Line 213 — `_edit_max_iter.int_value`

Widgets do not exist in `__init__`, so seeding must happen inside `_build_gui`. The cached `_last_*` attributes (lines 131–134) are seeded in `__init__` because they are plain attributes.

---

## Implementation Plan

### Phase 1: Shared helper
**Goal:** Introduce the mtime-refresh primitive used by every task.

- [x] 1.1 — Add `refresh_output_mtimes()` to `code/src/utils/should_process_task.py`; mirror `touch()` and PermissionError handling from lines 88–93.
- [x] 1.2 — Add a unit test covering: single Path, list of Paths, missing Path (no-op), read-only Path (logs and continues).

**Files Modified:**
- `code/src/utils/should_process_task.py` — add helper.
- `code/tests/test_should_process_task.py` (create if missing) — unit test.

**Dependencies:** None.

### Phase 2: Deterministic no-GUI tasks
**Goal:** Apply the touch-and-skip shortcut to `clean_forearm` and `build_mesh`.

- [x] 2.1 — In `clean_forearm_pointcloud.py`, after `should_process_task` returns True (around line 44), if all outputs exist, call `refresh_output_mtimes` and `return`.
- [x] 2.2 — In `define_forearm_mesh.py`, after the existing `should_process_task` gate (around line 57), if `output_path.exists()`, call `refresh_output_mtimes([output_path])` and `return None`.

**Files Modified:**
- `code/scripts/_3_preprocessing/_3_forearm_extraction/clean_forearm_pointcloud.py` — touch-and-skip.
- `code/scripts/_3_preprocessing/_3_forearm_extraction/define_forearm_mesh.py` — touch-and-skip.

**Dependencies:** Phase 1.

### Phase 3: Curate — close-without-save mtime refresh
**Goal:** Round out the already-working pre-fill by handling cancellation.

- [x] 3.1 — In `curate_forearm_pointcloud.py`, before opening the GUI, capture `outputs_existed`. In the `if not validated[0]` branch (line 80), if `outputs_existed`, call `refresh_output_mtimes([output_path, meta_path])` before returning.

**Files Modified:**
- `code/scripts/_3_preprocessing/_3_forearm_extraction/curate_forearm_pointcloud.py` — add mtime refresh on cancel.

**Dependencies:** Phase 1.

### Phase 4: Extract — add `was_modified` and cancel path
**Goal:** Distinguish edit+save from close-without-edit in `ArmSegmentation`, and skip re-save on the cancel path.

- [x] 4.1 — Add `ArmSegmentation.was_modified: bool = False` attribute.
- [x] 4.2 — In the interactive-process callback (around `arm_segmentation.py:905`), set `was_modified = True` whenever the user applies a change.
- [x] 4.3 — In `extract_participant_forearm.py` (around line 139–155), capture `outputs_existed` before instantiating `ArmSegmentation`. If `outputs_existed and not segmenter.was_modified`, call `refresh_output_mtimes([output_ply_path, output_params_path])` and skip `PointCloudDataHandler.save`.

**Fallback if 4.1/4.2 proves invasive:** unconditionally re-save (byte-identical JSON should round-trip fine; PLY may not — acceptable for the operator since parameters are what drive the re-run). Document the fallback if adopted.

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/arm_segmentation.py` — add `was_modified` flag.
- `code/scripts/_3_preprocessing/_3_forearm_extraction/extract_participant_forearm.py` — skip re-save on cancel; mtime refresh.

**Dependencies:** Phase 1.

### Phase 5: Define normals — seed model from metadata
**Goal:** Load `_with_normals_metadata.json` and seed `PointCloudModel` parameters so sliders open with prior values.

- [x] 5.1 — Extend `PointCloudModel.__init__` with `is_centered`, `scale_factor`, `normals_flipped` kwargs.
- [x] 5.2 — In `define_normals.py` after line 25, if `output_metadata_path.exists()`, read JSON and extract `processing_parameters.{k_neighbors_for_normals, radius_for_hybrid, used_hybrid_tree, aligned_with_viewpoint, viewpoint_vector}` and `final_transformations.{is_centered, scale_factor, normals_flipped}`; pass into `PointCloudModel(**seeded)`.
- [x] 5.3 — Audit `PointCloudVisualizer` widget setup and ensure initial values read from `self.controller.model.<attr>` rather than hard-coded literals.
- [x] 5.4 — Ensure the viewpoint slider range is computed from cloud bounds *before* the seeded viewpoint is applied (order matters — the seeded value must fit in the range).
- [x] 5.5 — Expose a `controller.saved: bool` (or repurpose `_running`) so the script can distinguish save from close-without-save.
- [x] 5.6 — In `define_normals.py`, if `outputs_existed and not controller.saved`, call `refresh_output_mtimes([output_ply_path, output_metadata_path])`.

**Files Modified:**
- `code/scripts/_3_preprocessing/_3_forearm_extraction/define_normals.py` — metadata load, seeded model construction, cancel branch.
- `code/src/preprocessing/forearm_extraction/normals_estimation/point_cloud_model.py` — constructor kwargs.
- `code/src/preprocessing/forearm_extraction/normals_estimation/point_cloud_controller.py` — expose `saved`.
- `code/src/preprocessing/forearm_extraction/normals_estimation/point_cloud_visualizer.py` — initial widget values from model.

**Dependencies:** Phase 1.

### Phase 6: Register forearms — seed workbench from transforms JSON
**Goal:** Pre-populate `RegistrationWorkbench` controls from `_registration_transforms.json`.

- [x] 6.1 — In `register_session_forearms.py` near line 88, if `transforms_path.exists()`, load JSON; extract `mode`, `canonical_key`, `parameters.registration_method`, `parameters.max_correspondence_distance`, `parameters.icp_max_iteration`; pass as `existing_state` dict to the workbench.
- [x] 6.2 — Add `existing_state: Optional[dict] = None` parameter to `RegistrationWorkbench.__init__` (line 107); store on `self._existing_state`.
- [x] 6.3 — In `__init__`, seed `_last_mode`, `_last_method`, `_last_max_dist`, `_last_max_iter` (lines 131–134) from `existing_state` when provided.
- [x] 6.4 — In `_build_gui`, at widget-creation lines 183 / 193 / 200 / 206 / 213, use `existing_state` values when present; fall back to current hard-coded defaults otherwise.
- [x] 6.5 — After `workbench.show()` in `register_session_forearms.py` (around line 137), if `outputs_existed and workbench.accepted is False`, call `refresh_output_mtimes([unified_ply_path, transforms_path])`.

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/registration/register_session_forearms.py` — load existing transforms, cancel branch.
- `code/src/preprocessing/forearm_extraction/registration/registration_workbench.py` — constructor parameter, `_build_gui` seeding.

**Dependencies:** Phase 1.

---

## Testing Plan

### Unit Tests

- [x] `refresh_output_mtimes` — single Path updates mtime.
- [x] `refresh_output_mtimes` — list of Paths updates all mtimes.
- [x] `refresh_output_mtimes` — missing file is a no-op and does not raise.
- [x] `refresh_output_mtimes` — read-only file logs and continues (no raise).
- [ ] `PointCloudModel(**seeded_kwargs)` — all seeded attributes are round-tripped.
- [ ] `RegistrationWorkbench(clouds, existing_state=...)` — `_last_*` attributes reflect `existing_state`.

### Integration Tests

- [ ] Pipeline fresh run — all outputs created; mtimes recorded.
- [ ] Pipeline forced re-run with close-without-edit on every GUI — output contents identical (SHA-256), mtimes newer.
- [ ] Pipeline non-forced re-run after 2 — all tasks report up-to-date.
- [ ] Staleness cascade — touch a sticker-ROI JSON; non-forced re-run triggers downstream reprocessing as today.

### Manual Verification

- [ ] Delete outputs for one session, run with `force_processing: true` on all tasks — all GUIs open with factory defaults; validate each normally.
- [ ] Re-run the same session with `force_processing: true` — confirm:
  - [ ] `extract_forearm` GUI opens with prior HSV/DBSCAN slider values.
  - [ ] `curate_forearm` GUI opens with prior removed points highlighted.
  - [ ] `define_normals` GUI opens with prior k_neighbors/radius/viewpoint slider values.
  - [ ] `register_forearms` workbench opens with prior mode/method/max_dist/max_iter values.
- [ ] Close each GUI without any changes. Confirm outputs' SHA-256 unchanged but mtimes newer.
- [ ] For `clean_forearm` / `build_mesh`: no GUI appears; log line indicates skip; mtimes are refreshed.
- [ ] Edit one parameter in `define_normals` GUI and save. Confirm outputs change.

### Edge Cases

- [ ] Outputs exist but one file is missing — `outputs_existed` is False; task runs normally with factory defaults.
- [ ] Read-only output file — `refresh_output_mtimes` logs and continues; task still returns.
- [ ] Metadata file has extra/missing keys — graceful fallback to defaults for missing keys; unknown keys ignored.
- [ ] Single-forearm session for `register_forearms` — no-op path preserved (per `note-forearm-icp-registration.md` section 8).
- [ ] First-run-only: `outputs_existed` is False → all tasks behave exactly as today.

---

## Documentation Plan

- [ ] Update `docs/development/knowledge-base/note-forearm-icp-registration.md` section 7 to note that the `"parameters"` block is now read back at workbench construction time.
- [ ] Add a short knowledge-base note `note-task-prefill-pattern.md` describing the capture-outputs_existed / seed-defaults / touch-on-cancel pattern so future tasks can reuse it.
- [ ] Update inline docstrings of the four modified GUI constructors to document the new kwargs.
- [ ] No CLAUDE.md change needed (no architectural shift).

---

## Rollback Plan

1. **Before deployment:**
   - Phase 1 alone is safe and can be released independently; subsequent phases are per-task and isolated.
   - If a phase regresses a task, revert the commit(s) for that phase; other phases remain in place.

2. **Data considerations:**
   - No migrations; no on-disk schema changes.
   - `refresh_output_mtimes` only updates mtimes, never content — safe to back out.

3. **Rollback procedure:**
   - `git revert` the offending phase commit(s) on the feature branch before merging to dev.
   - If already merged: revert the merge commit; re-open this plan for rework.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `clean_task_outputs` is called between `should_process_task` and GUI open in one of the target tasks, deleting the outputs we want to reload | Low | High | Grep every target task for `clean_task_outputs` before implementing Phases 4–6; guard any call on `not outputs_existed`. |
| `ArmSegmentation.was_modified` plumbing is more invasive than expected (threading, callback chains) | Medium | Medium | Fallback: always re-save on close; document in the plan. mtime refresh is the real goal, byte-identical content is acceptable. |
| `PointCloudController.load_point_cloud` unconditionally recomputes normals, producing slightly different bytes on save | Medium | Low | Seeding model attrs before the first recompute should yield identical numerical output on the same cloud. Verify on one real cloud; if not identical, rely on mtime refresh + no-save on cancel path. |
| `RegistrationWorkbench` seeding order issue — widgets don't exist in `__init__` | Low | Low | Seed inside `_build_gui`; `_last_*` cache attributes in `__init__`. Called out explicitly in Phase 6. |
| Metadata schemas evolve and old metadata is missing keys | Low | Low | Use `.get(key, default)` on every load; fall back to factory defaults. |
| Operator confuses "no GUI opens" (deterministic skip) with "pipeline failed" | Low | Low | Log a clear one-line "✅ Outputs up-to-date, skipping" message in clean/mesh skip branches. |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 — shared helper | 0.5 day | None |
| Phase 2 — deterministic tasks (clean, mesh) | 0.5 day | Phase 1 |
| Phase 3 — curate cancel path | 0.25 day | Phase 1 |
| Phase 4 — extract `was_modified` + cancel | 1 day | Phase 1 |
| Phase 5 — normals seeding | 1 day | Phase 1 |
| Phase 6 — workbench seeding | 1 day | Phase 1 |

Total: ~4 days.

---

## References

- Source plan (scratch): `C:\Users\basil\.claude\plans\vectorized-plotting-bear.md`
- Knowledge base: [note-forearm-icp-registration.md](../../knowledge-base/note-forearm-icp-registration.md) — registration transforms JSON schema (section 7).
- DAG config: `configs/preprocess_pipeline_extract_forearm_manual_dag.yaml`
- Pipeline orchestrator: `code/scripts/preprocess_pipeline_extract_forearm_manual.py`
- Existing pattern to mirror: `code/scripts/_3_preprocessing/_3_forearm_extraction/curate_forearm_pointcloud.py` (lines 62–70, 80).
