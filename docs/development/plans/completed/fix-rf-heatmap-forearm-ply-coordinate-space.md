# Plan: Fix RF Heatmap Forearm PLY Coordinate Space Mismatch

**Created:** 2026-04-13 18:00
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** In Progress
**Branch:** `feature/rf-heatmap-visualization-restyle`

---

## Overview

**What:** Fix the forearm PLY path used by the RF heatmap visualizers so it matches the coordinate space of the contact points.
**Why:** The visualizer loads the PCA-calibrated PLY (`forearm_pca_calibrated/`) but contact points in the aggregated CSV have been RF-centered (translated by the RF center offset), causing a visible spatial shift.
**How:** Change both RF pipelines to load from `forearm_rf_centered/` with fallback to `forearm_pca_calibrated/`, mirroring the existing `_resolve_latest_forearm_ply()` pattern.

## Problem Statement

- When zooming into RF heatmaps, spike contact points are systematically shifted from the forearm surface mesh — close but not coincident.
- Root cause: the postprocessing pipeline applies 5 stages to contact points (ICP registration → PCA calibration → forearm export → contact projection → **RF centering**), but the visualizer loads the forearm PLY from stage 3 output (`forearm_pca_calibrated/`), skipping the RF centering translation applied in stage 5.
- The aggregated CSV (stage 6 input) is built from `blocks_rf_centered/`, so contact points are in RF-centered space while the forearm PLY is only in PCA-calibrated space.
- The shift magnitude equals the RF center offset vector — typically a few mm.

## Goals

### In Scope
1. Fix `rf_cluster_pipeline.py` to load the RF-centered forearm PLY
2. Fix `rf_simple_pipeline.py` to load the RF-centered forearm PLY
3. Use a fallback pattern (RF-centered → PCA-calibrated) for robustness

### Out of Scope
- Investigating the RGB-D frame lag (separate idea: `rgb-d-frame-lag-compensation.md`)
- Fixing serialization precision (`.1f` quantization — ~0.05mm, negligible)
- Forward-fill temporal lag (separate issue, causes minor jitter not systematic shift)
- Updating the idea doc status (will be done at completion)

## Success Criteria

- [ ] `rf_cluster_pipeline.py` loads forearm PLY from `forearm_rf_centered/` (with fallback)
- [ ] `rf_simple_pipeline.py` loads forearm PLY from `forearm_rf_centered/` (with fallback)
- [ ] Heatmap spike contact points align with the forearm surface (no systematic shift)
- [ ] Heatmaps still render when only `forearm_pca_calibrated/` exists (fallback works)

---

## Technical Design

### Approach

Extract a shared helper function `resolve_forearm_ply(session_dir, session_id)` that checks `forearm_rf_centered/` first, falls back to `forearm_pca_calibrated/`. This mirrors the existing `_resolve_latest_forearm_ply()` in `postprocess_workflow_kinect_auto.py:144` but is session-aware (takes `session_id` to construct the expected filename).

Both pipelines call this helper instead of hardcoding the path.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Shared helper with fallback | DRY, robust, matches existing pattern | Adds one function | Chosen |
| Hardcode `forearm_rf_centered/` directly | Simplest change | No fallback if RF centering was skipped; duplicated in 2 files | Rejected |
| Reuse `_resolve_latest_forearm_ply()` directly | Zero new code | It's private to `postprocess_workflow_kinect_auto.py`, uses glob (no session_id filter), wrong abstraction layer | Rejected |

### Architecture Changes

- New helper in `code/src/analysis/receptive_field_mapping/rf_data_loader.py` — this module already contains shared loading utilities for RF mapping
- Two call-site changes in `rf_cluster_pipeline.py` and `rf_simple_pipeline.py`

---

## Implementation Plan

### Phase 1: Add helper and fix both pipelines
**Goal:** Resolve the coordinate space mismatch
**Started:** 2026-04-13
**Completed:** 2026-04-13

- [x] Task 1.1 — Add `resolve_forearm_ply(session_dir: Path, session_id: str) -> Optional[Path]` to `rf_data_loader.py`. Check `forearm_rf_centered/{session_id}_forearm.ply` first, fall back to `forearm_pca_calibrated/{session_id}_forearm.ply`, return `None` if neither exists.
- [x] Task 1.2 — In `rf_cluster_pipeline.py:356-358`, replace the hardcoded `forearm_pca_calibrated` path with a call to `resolve_forearm_ply()`.
- [x] Task 1.3 — In `rf_simple_pipeline.py:133`, replace the hardcoded `forearm_pca_calibrated` path with a call to `resolve_forearm_ply()`.

**Files Modified:**
- `code/src/analysis/receptive_field_mapping/rf_data_loader.py` — Add `resolve_forearm_ply()` helper
- `code/src/analysis/receptive_field_mapping/rf_cluster_pipeline.py` — Use helper at line ~356
- `code/src/analysis/receptive_field_mapping/rf_simple_pipeline.py` — Use helper at line ~133

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Run the RF cluster pipeline on a session that has both `forearm_rf_centered/` and `forearm_pca_calibrated/` directories — confirm the heatmap loads the RF-centered PLY and contact points sit on the forearm surface
- [ ] Run on a session that only has `forearm_pca_calibrated/` — confirm fallback works and heatmap still renders
- [ ] Zoom into the heatmap and visually confirm spike points are coincident with the forearm mesh (no systematic shift)

### Edge Cases
- [ ] Session where RF centering failed (no cluster found) — `forearm_rf_centered/` should still exist as a copy of PCA-calibrated PLY (see `center_on_receptive_field.py:256-258`), so the helper should find it

---

## Documentation Plan

- [ ] Update idea doc `docs/development/plans/ideas/contact-point-forearm-spatial-discrepancy.md` — note root cause identified and fixed

---

## Rollback Plan

1. Revert the 3 file changes — the hardcoded `forearm_pca_calibrated/` path is restored
2. No data changes, no migrations, no breaking changes

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Some sessions lack `forearm_rf_centered/` directory | Medium | Low | Fallback to `forearm_pca_calibrated/` ensures rendering still works |
| Helper returns `None` (no PLY at all) | Low | Low | Existing code already handles missing PLY gracefully (skips rendering) |

---

## References

- Related idea: `docs/development/plans/ideas/contact-point-forearm-spatial-discrepancy.md`
- Related idea: `docs/development/plans/ideas/rgb-d-frame-lag-compensation.md`
- Existing pattern: `_resolve_latest_forearm_ply()` in `code/scripts/postprocess_workflow_kinect_auto.py:144`
- Correct usage: `rf_camera_angle_task.py:133` already uses `forearm_rf_centered/`
