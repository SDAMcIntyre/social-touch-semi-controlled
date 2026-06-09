# Plan: Standalone MKV-to-Forearm-Mesh Tool

**Date:** 2026-06-09
**Author:** Basil Duvernoy
**Status:** In Progress
**Base Branch:** `feature/rename-iff-to-response-tuning`
**Branch:** `feature/standalone-mkv-to-forearm-mesh`

---

## Overview

A standalone proof-of-concept script that extracts a forearm mesh from a single Kinect MKV frame, fully decoupled from the existing pipeline (no DAG config, no Prefect flow, no session configs). The user navigates the video, picks a frame, draws an ROI, tunes segmentation parameters interactively, and gets a saved mesh file — all in one invocation.

## Problem Statement

The existing "extract forearm [manual]" pipeline requires session configs, DAG YAML, pre-annotated forearm parameters, and Prefect orchestration to produce a forearm mesh. There is no quick way to open an arbitrary MKV recording, browse to a specific frame, and export a forearm mesh for ad-hoc visualisation or proof-of-concept work. This forces researchers to set up the full pipeline infrastructure even for one-off mesh exports.

## Goals

### In Scope
1. Single-script tool: MKV file in → forearm mesh out
2. Interactive frame browsing and selection via existing Tkinter GUI
3. Interactive ROI drawing via existing OpenCV GUI
4. Interactive forearm segmentation parameter tuning via existing Open3D GUI
5. Automated cleaning, normal estimation, and mesh generation
6. Optional saving of intermediate point clouds for debugging

### Out of Scope
- Manual curation step (the ArmSegmentation GUI provides sufficient control)
- Multi-frame averaging (single frame only for this POC)
- ICP registration across multiple snapshots
- Integration with the pipeline DAG or launcher GUI
- GPU acceleration (CuPy) — not needed for single-frame processing

## Success Criteria

- [ ] Script runs with only an MKV path as required argument
- [ ] Frame picker, ROI selector, and segmentation GUI all launch and function correctly
- [ ] Valid `.obj` mesh file is saved to the output directory
- [ ] Mesh opens correctly in MeshLab or Open3D viewer
- [ ] No modifications to existing pipeline code

---

## Technical Design

### Approach

A single Python script at `code/scripts/__misc/standalone_mkv_to_forearm_mesh.py` that chains six sequential steps, reusing existing GUI components and processing functions. Data flows in-memory between steps; only the final mesh (and optionally intermediate PLYs) is persisted to disk.

The script uses `cv2.VideoCapture` for the frame picker (duck-typed against `VideoFrameSelector`), then opens the same MKV with `KinectMKV` for depth/point-cloud access on the selected frame. This avoids loading all depth frames just for preview browsing.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Chain existing functions in a standalone script | Minimal new code (~60 lines of glue), reuses tested components | Requires careful import bridging | **Chosen** |
| Build a unified PyQt5 GUI with all steps in one window | Polished UX, single window | Significant new GUI code, overkill for POC | Rejected |
| Add a "standalone mode" flag to the existing pipeline | No new script | Couples POC to pipeline infrastructure, complicates existing code | Rejected |

### Architecture Changes

No new modules or classes. One new standalone script that imports from existing packages:

```
code/scripts/__misc/standalone_mkv_to_forearm_mesh.py  (NEW)
  imports from:
    preprocessing.common.gui.video_frame_selector     → VideoFrameSelector
    preprocessing.common.gui.frame_roi_rotatable       → FrameROIRotatable
    preprocessing.common.data_access.kinect_mkv_manager → KinectMKV, KinectFrame
    preprocessing.forearm_extraction.arm_segmentation  → ArmSegmentation
    preprocessing.forearm_extraction.models.forearm_parameters → RegionOfInterest, Point
```

Functions copied/adapted from existing scripts (not imported, to avoid pulling in pipeline dependencies like `should_process_task`):
- `get_3d_cuboid_from_roi()` from `extract_participant_forearm.py` (~6 lines)
- AABB-from-rotated-rectangle math from `define_extraction_parameters.py:266-289` (~15 lines)
- XY-dedup cleaning logic from `clean_forearm_pointcloud.py:62-81` (~15 lines)

### Knowledge Base Constraints

- **Kinect depth single-path**: All depth reads go through `KinectMKV` with `parallax_correction=True` (default)
- **CuPy import order**: CuPy imported before any `preprocessing` imports (guarded try/except)
- **2.5D Delaunay**: Confirmed as the correct approach for height-field forearm data
- **Open3D SceneWidget layout**: `ArmSegmentation` already handles this correctly; no new Open3D GUI code needed

---

## Implementation Plan

### Phase 1: Standalone Script
**Goal:** Create the complete script with all six interactive/automated steps
**Started:** 2026-06-09
**Completed:** 2026-06-09

**Tasks:**
- [x] Task 1.1 — Create script with CuPy guard, sys.path setup, and argparse CLI
- [x] Task 1.2 — Implement frame selection step using `cv2.VideoCapture` + `VideoFrameSelector`
- [x] Task 1.3 — Implement ROI selection step using `FrameROIRotatable` + AABB conversion helper
- [x] Task 1.4 — Implement forearm extraction step using `KinectMKV` + `ArmSegmentation`
- [x] Task 1.5 — Implement in-memory cleaning (XY-dedup, keep lowest Z)
- [x] Task 1.6 — Implement in-memory normal estimation (Open3D `estimate_normals` + `orient_normals_consistent_tangent_plane`)
- [x] Task 1.7 — Implement mesh generation using `define_forearm_mesh()` with temp PLY bridge
- [x] Task 1.8 — Implement `--save-intermediates` flag for optional intermediate PLY output
- [x] Task 1.9 — Wire up `main()` orchestrator and summary output

**Files Modified:**
- `code/scripts/__misc/standalone_mkv_to_forearm_mesh.py` — New file (~180 lines)

**Dependencies:** None

---

## Testing Plan

### Manual Verification
- [ ] Run script with a real MKV file; verify frame picker opens and selects correctly
- [ ] Verify ROI selector opens on the colour frame; draw rectangle and confirm
- [ ] Verify ArmSegmentation interactive GUI opens; adjust parameters and continue
- [ ] Verify automated steps (clean → normals → mesh) complete without errors
- [ ] Verify mesh viewer opens showing the final result
- [ ] Verify `.obj` file is saved and loads correctly in MeshLab
- [ ] Run with `--save-intermediates` and verify intermediate PLY files are saved
- [ ] Run with `--output-dir` and verify custom output directory is used

### Edge Cases
- [ ] Cancel during frame selection (should exit cleanly)
- [ ] Cancel during ROI selection (should exit cleanly)
- [ ] Empty segmentation result (should print diagnostic and exit, not crash)

---

## Documentation Plan

- [ ] Script includes usage instructions in its argparse `--help` output

---

## Rollback Plan

Single new file with no modifications to existing code. Rollback is deleting the script:
1. `git rm code/scripts/__misc/standalone_mkv_to_forearm_mesh.py`

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Tkinter root conflicts with OpenCV/Open3D windows | Low | Med | Destroy Tkinter root before launching OpenCV/Open3D steps (existing `FrameROISquare._setup_window` handles this) |
| `cv2.VideoCapture` frame count differs from `KinectMKV` total_frames | Low | Low | Frame count mismatch only affects preview slider range, not the actual depth extraction (which uses `KinectMKV[frame_idx]`) |
| `define_forearm_mesh` import pulls in `utils.should_process_task` | Med | Low | Import at function scope or inline the Delaunay+Trimesh logic (~20 lines) if import chain is problematic |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 | ~2 hours | None |

---

## References

- Existing pipeline: `configs/preprocess_pipeline_extract_forearm_manual_dag.yaml`
- Extraction script: `code/scripts/_3_preprocessing/_3_forearm_extraction/extract_participant_forearm.py`
- Mesh script: `code/scripts/_3_preprocessing/_3_forearm_extraction/define_forearm_mesh.py`
- Knowledge base: `docs/development/knowledge-base/note-kinect-depth-access-single-path.md`
- Knowledge base: `docs/development/knowledge-base/report-preprocessing-forearm-mesh-construction.md`

---

## Modified Files

<!-- auto-generated by /plan-implement — do not edit manually -->
- `code/scripts/__misc/standalone_mkv_to_forearm_mesh.py`
- `docs/development/plans/active/standalone-mkv-to-forearm-mesh.md`
