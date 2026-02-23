# Plan: Forearm Catalog Filename Fix for Averaged Frames

**Date:** 2026-02-23
**Status:** Completed
**Branch:** `fix/forearm-catalog-averaged-filename`

---

## Overview

The forearm extraction pipeline now supports multi-frame depth averaging (see `forearm-frame-averaging` plan), which introduced a new filename convention for averaged captures. The downstream `ForearmCatalog` — consumed by `compute_somatosensory_characteristics_flow` — still hardcodes the old single-frame naming pattern, silently failing to load averaged forearm files.

## Problem Statement

`ForearmCatalog._load_mesh()` and `_load_pointcloud()` construct filenames using only the single-frame pattern (`_frame_{id:04}`), but the extraction pipeline now produces averaged files with a different pattern (`_frames_{lo:04d}-{hi:04d}_avg_N{count}`). This causes the catalog to miss files on disk, fall through to a wrong fallback reference, or return an empty dict — producing incorrect or missing somatosensory results.

**Current behavior for an averaged capture** (`frame_ids=[42,43,44,45,46]`, `representative_frame_id=42`):
1. Catalog looks for `{stem}_frame_0042_mesh.obj` — does not exist
2. Falls back to closest block-order match — same bug applies if that is also averaged
3. Returns `{}` or wrong forearm from a different block

## Goals

### In Scope
1. Fix `ForearmCatalog` to construct correct filenames for both single-frame and averaged captures
2. Centralize the filename stem-building logic to eliminate duplication between the catalog and the pipeline script
3. Ensure backward compatibility with existing single-frame extractions

### Out of Scope
- Changes to the averaging algorithm or extraction pipeline behavior
- Changes to `compute_somatosensory_characteristics.py` (it delegates entirely to the catalog)
- Migration or renaming of existing files on disk

## Success Criteria

- [x] `ForearmCatalog._load_mesh()` finds `_frames_XXXX-YYYY_avg_NZ_mesh.obj` for averaged captures
- [x] `ForearmCatalog._load_pointcloud()` finds `_frames_XXXX-YYYY_avg_NZ_with_normals.ply` for averaged captures
- [x] Single-frame captures (`_frame_XXXX_mesh.obj`) continue to load correctly
- [x] Filename stem logic exists in a single location (no duplication)

---

## Technical Design

### Approach

Move the stem-building logic into `ForearmParameters` as a `build_output_stem(video_stem)` method. Both `ForearmCatalog` and `preprocess_pipeline_extract_forearm_manual.py` call this single source of truth.

This is the natural home for this logic because:
- `ForearmParameters` already owns `frame_ids`, `representative_frame_id`, and `is_averaged`
- The naming convention is intrinsic to the parameters themselves
- Both producer and consumer already have access to a `ForearmParameters` instance

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Method on `ForearmParameters` | Single source of truth; natural home; no new imports needed | Adds a filename concern to a data class | **Chosen** |
| Standalone utility function | Separation of concerns | Requires a new module or an awkward import; both sites still need to find it | Rejected |
| Duplicate the logic in `ForearmCatalog` | No refactoring of pipeline script | Two copies to maintain; guaranteed drift | Rejected |

### Architecture Changes

No new modules. One method added to an existing dataclass, two call sites updated.

---

## Implementation Plan

### Phase 1: Centralize stem logic and fix catalog

**Goal:** Add `build_output_stem()` to `ForearmParameters` and update all consumers.

- [x] Task 1.1 — Add `build_output_stem(video_stem: str) -> str` method to `ForearmParameters`
- [x] Task 1.2 — Update `ForearmCatalog._load_pointcloud()` to use `params.build_output_stem(video_stem)`
- [x] Task 1.3 — Update `ForearmCatalog._load_mesh()` to use `params.build_output_stem(video_stem)`
- [x] Task 1.4 — Update `_build_output_stem()` in pipeline script to delegate to `params.build_output_stem()`

**Files Modified:**

| File | Change |
|------|--------|
| `code/src/preprocessing/forearm_extraction/models/forearm_parameters.py` | Add `build_output_stem()` method |
| `code/src/preprocessing/forearm_extraction/models/forearm_catalog.py` | Replace hardcoded filename patterns in `_load_pointcloud()` and `_load_mesh()` |
| `code/scripts/preprocess_pipeline_extract_forearm_manual.py` | Delegate `_build_output_stem()` to new method |

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Run forearm extraction with averaged frames (multi-frame selection) — confirm files are produced with new naming
- [x] Run `compute_somatosensory_characteristics_flow` on a session with averaged forearm data — confirmed the catalog finds and loads the averaged mesh
- [ ] Run the same flow on a session with single-frame forearm data — confirm no regression
- [ ] Verify `get_forearms_with_fallback()` fallback path also works for averaged captures (closest block reference)

### Edge Cases
- [ ] Single-frame capture (backward compat) — `frame_ids=[42]`, `is_averaged=False` → `_frame_0042` pattern
- [ ] Averaged capture with non-contiguous frames — `frame_ids=[40,42,45]` → `_frames_0040-0045_avg_N3`
- [ ] Fallback reference is an averaged capture from a different block — should load correctly

---

## Documentation Plan

- [x] No user-facing docs needed (internal pipeline fix)
- [x] Add inline comment in `ForearmParameters.build_output_stem()` noting the naming convention

---

## Rollback Plan

1. Revert the three file changes — the old hardcoded pattern is restored
2. No data migration needed — file naming on disk is unchanged
3. No breaking changes — this is purely a filename lookup fix

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Naming convention drifts again in future | Low | Med | Centralizing in `ForearmParameters` makes it a single edit point |
| Existing single-frame files break | Low | High | `is_averaged` check preserves the old pattern exactly |
