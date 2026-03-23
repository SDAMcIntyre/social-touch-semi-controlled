# Plan: Postprocess Forearm PLY Cleanup

**Created:** 2026-03-23 18:00
**Approved:** ---
**Completed:** ---
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/postprocess-forearm-ply-cleanup`

---

## Overview

**What:** Two small fixes to the postprocessing pipeline: simplify forearm PLY filenames and have the aggregate task copy the final forearm PLY to the session root.
**Why:** Forearm PLY filenames are redundant with their parent folder names, and the final forearm output is buried in a subfolder rather than sitting alongside the aggregated session CSV.
**How:** Rename `{session_id}_forearm_pca_calibrated.ply` to `{session_id}_forearm.ply` across all consumers; extend `aggregate_session_blocks` to copy the latest forearm PLY to the session root.

## Problem Statement

- Forearm PLY files are named `{session_id}_forearm_pca_calibrated.ply` inside folders like `forearm_pca_calibrated/` and `forearm_rf_centered/`. The `_pca_calibrated` suffix is redundant with the folder name.
- The `aggregate_session` task (stage 6) produces only an aggregated CSV in the session root. The companion forearm PLY (the other key session-level output) remains buried in a subfolder, making it inconvenient to locate the two final outputs together.

## Goals

### In Scope

1. Simplify forearm PLY filename to `{session_id}_forearm.ply`
2. Copy the latest forearm PLY (RF-centered if available, else PCA-calibrated) to the session root during aggregation

### Out of Scope

- Renaming block CSV files or other output filenames
- Changing the folder structure (`forearm_pca_calibrated/`, `forearm_rf_centered/`)
- Migrating or deleting old-named files on disk

## Success Criteria

- [ ] Stage 3 (`export_forearm_pca_calibrated`) writes `{session_id}_forearm.ply`
- [ ] Stage 5 (`center_on_receptive_field`) auto-inherits the new filename
- [ ] Visualization (`postprocess_visualization.py`) and analysis (`analysis_workflow.py`) resolve the new filename
- [ ] After `aggregate_session`, the session root contains both the aggregated CSV and a copy of the forearm PLY
- [ ] `should_process_task` freshness check covers the forearm PLY copy

---

## Technical Design

### Approach

**Fix 1 (filename):** Pure string replacement in 3 files (5 edit sites). `center_on_receptive_field.py` uses `forearm_ply_path.name` so it auto-inherits.

**Fix 2 (aggregate copies forearm):**
- Add a `_resolve_latest_forearm_ply()` helper in the workflow that checks `forearm_rf_centered/` then `forearm_pca_calibrated/` for `*.ply` files.
- Add optional `forearm_ply_path` parameter to `aggregate_session_blocks()`.
- Include the forearm PLY in `should_process_task` input/output checks.
- Copy via `shutil.copy2()` after the CSV write.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Resolve forearm PLY in aggregate params lambda | Simple, uses existing glob; filename-agnostic | Relies on filesystem state at call time | Chosen |
| Propagate forearm PLY path through context from stage 5 | Explicit data flow | Requires changing `center_on_receptive_field` return type; breaks if stage 5 is skipped | Rejected |

### Architecture Changes

No new modules. Minor signature extension to `aggregate_session_blocks()`.

---

## Implementation Plan

### Phase 1: Filename simplification + aggregate forearm copy
**Goal:** Apply both fixes
**Started:** 2026-03-23 18:30
**Completed:** 2026-03-23 18:30

**Tasks:**

- [x] Task 1.1 --- Rename forearm PLY output in `export_forearm_pca_calibrated.py` (line 83 + docstring line 5)
- [x] Task 1.2 --- Update filename reference in `postprocess_visualization.py` (lines 65, 85)
- [x] Task 1.3 --- Update filename reference in `analysis_workflow.py` (line 379)
- [x] Task 1.4 --- Add `forearm_ply_path` parameter to `aggregate_session_blocks()`, include in freshness check, copy after CSV write
- [x] Task 1.5 --- Update `aggregate_session_blocks_flow` wrapper to pass `forearm_ply_path`
- [x] Task 1.6 --- Add `_resolve_latest_forearm_ply()` helper and update stage 6 params lambda in `postprocess_workflow_kinect_auto.py`

**Files Modified:**

- `code/scripts/_5_postprocessing/export_forearm_pca_calibrated.py` --- Rename output from `_forearm_pca_calibrated.ply` to `_forearm.ply`
- `code/scripts/postprocess_visualization.py` --- Update filename in path resolution and docstring
- `code/scripts/analysis_workflow.py` --- Update filename in `_load_forearm_pcd()`
- `code/scripts/_4_merging/aggregate_blocks_session.py` --- Add `forearm_ply_path` param, freshness check, `shutil.copy2` logic
- `code/scripts/postprocess_workflow_kinect_auto.py` --- Add `_resolve_latest_forearm_ply()` helper, update flow wrapper and stage 6 params

**Dependencies:** None

---

## Testing Plan

### Manual Verification

- [ ] Run postprocess pipeline on a session with RF centering enabled --- confirm `{session_id}_forearm.ply` appears in `forearm_pca_calibrated/`, `forearm_rf_centered/`, and the session root
- [ ] Run on a session where `center_on_receptive_field` is disabled --- confirm the copy comes from `forearm_pca_calibrated/`
- [ ] Run `postprocess_visualization` on a processed session --- confirm it resolves the new filename
- [ ] Confirm `should_process_task` skips re-aggregation when outputs are up-to-date

### Edge Cases

- [ ] No forearm PLY exists (e.g. no forearm scan) --- aggregate completes normally, no copy attempted
- [ ] Empty input_paths --- existing early-return guard fires before forearm logic

---

## Documentation Plan

- [ ] No external documentation changes needed (internal pipeline detail)

---

## Rollback Plan

1. Revert the feature branch commits
2. Old-named files on disk remain valid; new-named files will be regenerated on next force-reprocess

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Old-named forearm PLY files remain on disk | High | Low | They are harmless; next force-reprocess overwrites with new name |
| Visualization or analysis script fails to find renamed PLY | Low | Med | All 3 consumer sites are updated in the same commit |
