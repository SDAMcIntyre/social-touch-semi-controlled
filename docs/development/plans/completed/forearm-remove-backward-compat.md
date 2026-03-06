# Plan: Remove Forearm Pipeline Backward Compatibility

**Date:** 2026-03-04
**Author:** Claude (AI-assisted)
**Status:** Completed
**Completion Date:** 2026-03-06
**Branch:** `refactor/forearm-remove-backward-compat`

---

## Overview

**What:** Remove all backward-compatibility layers from the forearm extraction pipeline.
**Why:** The old single-field `frame_id` format is no longer in use. The compat code adds complexity and makes the data model ambiguous (two ways to access the same value).
**How:** Delete the `frame_id` property, simplify the file handler to accept only the new JSON format, migrate all call sites to `representative_frame_id`, and collapse the extraction branching into a single `FrameDepthAverager` path.

## Problem Statement

The forearm extraction pipeline was recently extended from single-frame to multi-frame depth averaging (see `docs/development/plans/completed/forearm-frame-averaging.md`). That feature shipped with full backward compatibility at three layers:

1. **Data model** — a `frame_id` property alias that transparently returns `representative_frame_id`
2. **File handler** — three-path JSON loading that auto-converts old `"frame_id": int` format to the new `"frame_ids": list` + `"representative_frame_id": int` format
3. **Extraction logic** — dual paths branching on `is_averaged`, where single-frame captures bypass `FrameDepthAverager`

Now that the new version is stable and no old-format JSON files remain on disk, this compatibility code is dead weight that obscures the data model and adds unnecessary branching.

## Goals

### In Scope
1. Remove the `frame_id` backward-compat property from `ForearmParameters`
2. Remove old-format `"frame_id"` JSON loading from `ForearmFrameParametersFileHandler`
3. Migrate all 7 call sites from `frame_id` to `representative_frame_id`
4. Collapse the `is_averaged` extraction branching — all captures flow through `FrameDepthAverager`

### Out of Scope
- Migration script for old-format JSON files (confirmed none exist on disk)
- Changes to the `is_averaged` property itself (still used by `build_output_stem()` and `FrameBatch.description`)
- Changes to downstream consumers (`get_forearms_with_fallback`, scene viewers, somatosensory pipeline)
- Changes to unrelated `frame_id` fields in sticker analysis modules

## Success Criteria

- [x] `ForearmParameters` has no `frame_id` property — accessing it raises `AttributeError`
- [x] `ForearmFrameParametersFileHandler.load()` requires both `"frame_ids"` and `"representative_frame_id"` keys
- [x] `is_valid_structure()` rejects JSON entries with only `"frame_id"` (old format)
- [x] `extract_forearm()` uses `FrameDepthAverager.average()` for all captures (no `is_averaged` branching)
- [x] Single-frame extraction produces identical output PLY to the pre-change version
- [x] All 7 former `frame_id` call sites compile and run correctly with `representative_frame_id`
- [x] No remaining references to the string `".frame_id"` in the forearm extraction package (except unrelated sticker modules)

---

## Technical Design

### Approach

Direct removal of each backward-compat layer with 1:1 substitution at call sites. The `frame_id` property was always a transparent proxy for `representative_frame_id`, so every call site replacement is a mechanical substitution with no behavioral change. The extraction path collapse is safe because `FrameDepthAverager.average()` with N=1 produces output identical to `frame.generate_o3d_point_cloud()` (same source data, same validity mask, same color conversion — the division by 1 is a no-op).

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Remove all compat layers at once | Clean single change; no intermediate state | Slightly larger diff | **Chosen** |
| Keep `frame_id` as a deprecated alias with warning | Gradual migration | Adds warning noise; no external consumers to protect | Rejected |
| Keep extraction branching | Less change | Retains dead code path; confirmed identical output for N=1 | Rejected |

### Architecture Changes

No new files. Five files modified (net code deletion):

```
code/src/preprocessing/forearm_extraction/
├── models/forearm_parameters.py              -- delete frame_id property
├── models/forearm_catalog.py                 -- 5 call sites: frame_id -> representative_frame_id
├── data_access/forearm_frame_parameters_filehandler.py  -- simplify load() and is_valid_structure()

code/scripts/
├── _3_preprocessing/_3_forearm_extraction/
│   └── extract_participant_forearm.py        -- collapse extraction branching, update frame_id ref
└── preprocess_pipeline_extract_forearm_manual.py  -- update FrameBatch.description
```

---

## Implementation Plan

### Phase 1: Remove `frame_id` Property and Migrate Call Sites
**Goal:** Delete the backward-compat property and update all consumers.

- [x] 1.1 — Delete the `frame_id` property (lines 35-38) from `forearm_parameters.py`
- [x] 1.2 — Update `forearm_catalog.py` (5 sites):
  - Line 110: `pointclouds[params.frame_id]` -> `pointclouds[params.representative_frame_id]`
  - Line 126: `meshes[params.frame_id]` -> `meshes[params.representative_frame_id]`
  - Line 138: `f"Frame {params.frame_id}"` -> `f"Frame {params.representative_frame_id}"`
  - Line 152: `f"Frame {params.frame_id}"` -> `f"Frame {params.representative_frame_id}"`
  - Line 196: `closest_params.frame_id` -> `closest_params.representative_frame_id`
- [x] 1.3 — Update `extract_participant_forearm.py` line 231: `mkv[video_config.frame_id]` -> `mkv[video_config.representative_frame_id]`
- [x] 1.4 — Update `preprocess_pipeline_extract_forearm_manual.py` line 60: `f"frame {p.frame_id}"` -> `f"frame {p.representative_frame_id}"`

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/models/forearm_parameters.py` -- delete property
- `code/src/preprocessing/forearm_extraction/models/forearm_catalog.py` -- 5 substitutions
- `code/scripts/_3_preprocessing/_3_forearm_extraction/extract_participant_forearm.py` -- 1 substitution
- `code/scripts/preprocess_pipeline_extract_forearm_manual.py` -- 1 substitution

**Dependencies:** None

### Phase 2: Simplify File Handler
**Goal:** Remove old-format JSON support from the loader and validator.

- [x] 2.1 — Simplify `load()` (lines 65-75): replace three-path branching with direct reads of `frame_ids` and `representative_frame_id`. Keep the guard that resets `representative_frame_id` to `min(frame_ids)` if not a member (data-integrity check, not backward compat).
- [x] 2.2 — Simplify `is_valid_structure()` (lines 128-140): require `"frame_ids"` (non-empty list) and `"representative_frame_id"` (int). Remove the `elif "frame_id" in item` branch.

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/data_access/forearm_frame_parameters_filehandler.py`

**Dependencies:** None (parallel with Phase 1)

### Phase 3: Collapse Extraction Path Branching
**Goal:** Remove the `is_averaged` check from extraction — all captures use `FrameDepthAverager.average()`.

- [x] 3.1 — In `extract_participant_forearm.py`, replace lines 231-244 (the `if video_config.is_averaged` block) with a unified path that always calls `FrameDepthAverager.average(mkv, video_config.frame_ids, color_frame_id=video_config.representative_frame_id)`. Keep loading the representative frame for ROI cuboid computation (line 248) and monitoring display (line 262).
- [x] 3.2 — Replace the conditional print (lines 234-237) with an unconditional log: `f"   Loading {len(video_config.frame_ids)} frame(s) (representative: {video_config.representative_frame_id})..."`

**Files Modified:**
- `code/scripts/_3_preprocessing/_3_forearm_extraction/extract_participant_forearm.py`

**Dependencies:** Phase 1 (the `frame_id` reference on line 231 must be replaced first)

---

## Testing Plan

### Manual Verification
- [x] Run the forearm extraction pipeline on a single-frame capture. Compare output PLY to pre-change output (should be identical).
- [x] Run the forearm extraction pipeline on a multi-frame averaged capture. Verify identical results.
- [x] Launch `neural_kinect_scene_viewer.py` — confirm forearm point clouds/meshes load correctly in the 3D viewer.
- [x] Verify `ForearmFrameParametersFileHandler.load()` successfully loads an existing new-format JSON file.

### Edge Cases
- [x] JSON file with `representative_frame_id` not in `frame_ids` — guard should still reset it to `min(frame_ids)`
- [x] Empty JSON file (empty list) — `is_valid_structure()` should return `True`
- [x] Single-frame capture `frame_ids=[N]` — `FrameDepthAverager` processes it without error

---

## Documentation Plan

- [x] No CLAUDE.md changes needed
- [x] No new user guide needed — this is an internal cleanup
- [x] Update the completed forearm-frame-averaging plan with a note referencing this cleanup

---

## Rollback Plan

All changes are mechanical substitutions and code deletions within the forearm extraction sub-package. Rollback is a simple `git revert` of the commit. No data format changes are introduced (only old-format reading is removed; writing was already new-format only).

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Old-format JSON files exist somewhere unexpected | Low | Med | User confirmed none exist. If discovered, the file handler will raise a clear `KeyError` for `"frame_ids"`, pointing to the problem. |
| `FrameDepthAverager` with N=1 has subtle behavioral difference | Low | Med | Code inspection confirms identical logic (same source array, same validity mask, same color conversion). Verified by manual PLY comparison in testing plan. |
| A caller outside the 7 identified sites accesses `frame_id` | Low | Low | Project-wide grep confirms no other callers in the forearm package. Any missed site produces an immediate `AttributeError`. |

---

## References

- Predecessor plan: `docs/development/plans/completed/forearm-frame-averaging.md`
- Data model: `code/src/preprocessing/forearm_extraction/models/forearm_parameters.py`
- File handler: `code/src/preprocessing/forearm_extraction/data_access/forearm_frame_parameters_filehandler.py`
- Extraction logic: `code/scripts/_3_preprocessing/_3_forearm_extraction/extract_participant_forearm.py`
- Catalog: `code/src/preprocessing/forearm_extraction/models/forearm_catalog.py`
- Pipeline orchestrator: `code/scripts/preprocess_pipeline_extract_forearm_manual.py`
