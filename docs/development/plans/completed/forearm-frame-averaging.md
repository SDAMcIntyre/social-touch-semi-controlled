# Plan: Forearm Frame Averaging

**Date:** 2026-02-19
**Author:** Claude (AI-assisted)
**Status:** Completed
**Branch:** `feature/forearm-frame-averaging`

---

## Overview

**What:** Extend the forearm extraction pipeline to allow a user to select a group of frames (a consecutive range or a set of individual frames) and produce a single averaged point cloud from them, instead of extracting from one frame.
**Why:** Azure Kinect ToF depth noise is ~5–6 mm std dev per pixel per frame. As documented in `docs/kinect_frame_averaging_insight.docx`, averaging N frames reduces this by √N — averaging just 4 frames roughly halves the error. The current pipeline is hard-coded to single-frame extraction throughout the data model, GUI, and backend.
**How:** Extend the `ForearmParameters` data model to store a list of frame IDs and an explicit `representative_frame_id` (the temporal position the averaged capture stands for, defaulting to the earliest frame in the group); enhance the `VideoFramesSelector` GUI with range and custom-group selection modes and a "Mark as Representative" button; add a `FrameDepthAverager` backend that computes a per-pixel mean of `transformed_depth_point_cloud` across N frames; wire everything together in the extraction orchestrator. The output artifact set (PLY → cleaned PLY → normals PLY → mesh OBJ) is unchanged — each logical forearm capture still produces exactly one set of files, just optionally derived from an averaged depth.

---

## Problem Statement

The extraction pipeline at every layer assumes exactly one frame per capture:

- **Data model:** `ForearmParameters.frame_id: int` — a single integer
- **File handler:** JSON schema validates a single `"frame_id"` key per entry
- **GUI:** `VideoFramesSelector.selected_frames: set` — a flat set of independent frame indices; each selected frame becomes its own independent `ForearmParameters` entry
- **Backend:** `extract_forearm()` opens the MKV and calls `mkv[video_config.frame_id]` — one frame, one point cloud
- **Orchestrator naming:** `f"{video_stem}_frame_{params.frame_id:04d}"` — single-frame filename
- **Temporal meaning:** `frame_id` also serves as the temporal anchor for downstream consumers (merging pipeline, neural-kinect alignment) — it marks "when" in the video timeline this forearm capture occurred. For averaged groups this concept must be preserved explicitly, as there is no single natural frame index.

There is no concept of a "frame group for averaging" anywhere in the stack. Adding averaging without addressing all these layers would leave the architecture inconsistent.

---

## Goals

### In Scope
1. Extend `ForearmParameters` model: replace `frame_id: int` with `frame_ids: List[int]` and a new `representative_frame_id: int` field; keep a `frame_id` property (returning `representative_frame_id`) for backward-compatible internal and downstream access
2. Update `ForearmFrameParametersFileHandler` to read and write the new `frame_ids` and `representative_frame_id` fields, with a backward-compatibility path for old JSON files that use `frame_id`
3. Enhance `VideoFramesSelector` GUI with three modes: **Single Frame** (legacy, one click = one group), **Range** (mark start/end → add as group), and **Custom Group** (accumulate non-consecutive frames → finalize as group); add a **Mark as Representative** button to override the temporal anchor for any group
4. Add a `FrameDepthAverager` class that loads N frames from a `KinectMKV`, computes a per-pixel mean of `transformed_depth_point_cloud` across valid pixels, and returns a single `o3d.PointCloud`
5. Update `extract_forearm()` to route through `FrameDepthAverager` when `len(frame_ids) > 1`, preserving the existing single-frame path when `len(frame_ids) == 1`
6. Update output filenames to reflect frame groups (e.g., `_frames_0042-0047_avg_N6` for a 6-frame range, `_frame_0042` for a single frame); `representative_frame_id` is stored in the JSON params file rather than the filename

### Out of Scope
- GPU-accelerated frame averaging (out of scope; a potential future optimization)
- Motion-artifact detection or masking between averaged frames (user is responsible for selecting static frames)
- Averaging post-PLY files already on disk (this plan averages at the depth array level, before PLY generation)
- Changes to `clean_forearm_pointcloud`, `define_normals`, `define_forearm_mesh`, or any downstream pipeline step
- Changing any other preprocessing module (LED analysis, sticker analysis, trial segmentation, etc.)
- Batch/automatic frame range selection (always manual/interactive)

---

## Success Criteria

- [ ] `ForearmParameters(frame_ids=[42], representative_frame_id=42)` has `frame_id == 42` and `is_averaged == False`
- [ ] `ForearmParameters(frame_ids=[42, 43, 44, 45, 46], representative_frame_id=42)` has `frame_id == 42` and `is_averaged == True`
- [ ] `ForearmParameters(frame_ids=[42, 43, 44, 45, 46], representative_frame_id=44)` has `frame_id == 44` — user-overridden representative is respected
- [ ] Old JSON (`"frame_id": 42`) loaded by the new `ForearmFrameParametersFileHandler` produces `frame_ids=[42]` and `representative_frame_id=42`
- [ ] New JSON (`"frame_ids": [42, 43, 44, 45, 46], "representative_frame_id": 42`) round-trips correctly through save/load
- [ ] New JSON with `frame_ids` but no `representative_frame_id` key defaults to `min(frame_ids)` on load
- [ ] `VideoFramesSelector` in **Range** mode allows the user to mark a start frame, mark an end frame, and confirm the range as a named group; the group appears in the groups listbox with `[rep: XXXX]` shown
- [ ] `VideoFramesSelector` in **Custom Group** mode allows adding non-consecutive frames one by one, then finalizing as a group; representative defaults to `min` of the group
- [ ] `VideoFramesSelector` in **Single Frame** mode (default) behaves identically to the current "Select/Deselect Frame" workflow, each click adding a 1-frame group; `representative_frame_id == frame_ids[0]` always
- [ ] "Mark as Representative" button updates the representative of the currently selected group in the listbox to the current slider frame; group entry updates in real time
- [ ] "Mark as Representative" button is disabled when the current slider frame is not a member of the selected group
- [ ] `FrameDepthAverager` uses `representative_frame_id` as `color_frame_id`; produces a point cloud within floating-point tolerance of the existing single-frame extraction path when N=1
- [ ] `FrameDepthAverager` correctly ignores pixels where Z=0 or NaN in any frame (uses only valid-pixel mean)
- [ ] End-to-end pipeline run with a 5-frame averaged group produces a `_frames_XXXX-YYYY_avg_N5` PLY; the JSON params file contains `"representative_frame_id"` matching the user's choice
- [ ] End-to-end pipeline run with a single frame still produces a `_frame_XXXX` PLY (unchanged naming and behavior)
- [ ] Pressing `←` in `VideoFramesSelector` moves the slider one frame back; pressing `→` moves it one frame forward; at boundaries (frame 0 or last frame) the press is a no-op
- [ ] While the frame entry has keyboard focus, `←`/`→` perform normal text-cursor movement inside the entry (not frame navigation)
- [ ] Typing a valid frame number in the entry and pressing `Enter` jumps the slider and video canvas to that frame immediately; `frame_info_lbl` and status bar update
- [ ] Typing a number outside `[0, total_frames - 1]` and pressing `Enter` clamps silently to the nearest boundary with no error dialog
- [ ] Typing non-numeric text and pressing `Enter` is a no-op (no crash, no navigation change)
- [ ] The frame entry stays in sync with the slider: navigating via keyboard or dragging the slider keeps the displayed number current (unless the entry currently has focus)
- [ ] A group indicator strip sits directly above the slider, spanning its full width, with a dark background
- [ ] Each frame belonging to any group shows a 1 px red vertical line in the strip at its proportional x-position
- [ ] Adding or removing a group immediately redraws the strip
- [ ] When the window is resized horizontally the strip redraws to reflect the new width at the next group change or `<Configure>` event
- [ ] In **Single Frame** mode, navigating to a frame already in any group and clicking "Add as Group" does nothing (no duplicate group is created)
- [ ] In **Range** mode, attempting to add a range that overlaps an existing group shows a warning dialog listing the conflicting frames and does not add the group
- [ ] In **Custom Group** mode, clicking "Add Frame to Group" on a frame already claimed by any group skips it silently; the pending count does not increase
- [ ] A frame released by "Remove Selected Group" becomes claimable again immediately in all three modes

---

## Technical Design

### Approach

Averaging happens at the **depth point cloud array level** inside the MKV context, before any point cloud object is constructed. The `KinectFrame.transformed_depth_point_cloud` property exposes an `(H, W, 3)` float32 array of XYZ values in camera space. For N frames, these arrays are loaded sequentially, a validity mask (`Z > 0 and not NaN`) is applied per pixel per frame, and a per-pixel weighted mean is computed. The averaged `(H, W, 3)` array is then used to construct a single `o3d.PointCloud`, with color taken from the frame at `representative_frame_id`. This averaged point cloud enters the existing pipeline unchanged.

The data model extends `frame_id: int` to `frame_ids: List[int]` plus a new `representative_frame_id: int` field. The `representative_frame_id` carries the temporal meaning that `frame_id` previously held: it is the index downstream consumers (merging pipeline, neural-kinect alignment) use to position this forearm capture in time. It defaults to `min(frame_ids)` (the earliest frame in the group) and can be overridden by the user in the GUI. A `@property frame_id` returning `self.representative_frame_id` preserves all existing internal and downstream access patterns without any call-site changes. A `@property is_averaged` returning `len(frame_ids) > 1` gates the averaging code path.

The GUI adds a "Frame Groups" panel to `VideoFramesSelector`. The internal data structure changes from a flat `set` of integers to a `List[List[int]]` of groups alongside a parallel `List[int]` of per-group representatives. `MultiVideoFramesSelector.all_selected_frames` changes accordingly from `{video_path: [42, 67]}` to `{video_path: {"groups": [[42], [67, 68, 69, 70]], "representatives": [42, 67]}}`.

#### Averaging Algorithm

Given N frames, each with `xyz_i ∈ R^{H×W×3}`:

```
mask_i[h, w]      = (xyz_i[h, w, 2] > 0) AND NOT any(nan(xyz_i[h, w]))
valid_count[h, w] = Σ_i mask_i[h, w]
avg_xyz[h, w]     = Σ_i (xyz_i[h, w] * mask_i[h, w]) / max(valid_count[h, w], 1)
final_mask[h, w]  = valid_count[h, w] >= 1
```

This is computed with a **running accumulator** (Welford-style sequential loading) to avoid loading all N `(H, W, 3)` arrays simultaneously into memory.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Average `transformed_depth_point_cloud` XYZ arrays per pixel | Directly averages the 3D positions used downstream; handles per-pixel validity naturally | Slightly more memory per frame (3×float32 vs 1×uint16) | **Chosen** |
| Average raw `transformed_depth` uint16 maps, then reproject once | Minimal memory; works at raw sensor level | Requires access to `K4A` calibration to re-project; averaging integers loses sub-mm precision; more complex | Rejected |
| Average N independent `o3d.PointCloud` objects via voxel downsampling | Fully modular, reuses existing output | Loses XY pixel grid structure; point cloud densities differ per frame; requires registration even for near-static scenes | Rejected |
| Average finished PLY files post-extraction | Keeps extraction unchanged; fully independent | Would require two-pass pipeline; same registration problem as above; doubles disk I/O | Rejected |

### Architecture Changes

**New file:**
```
code/src/preprocessing/forearm_extraction/
└── depth_averaging/
    ├── __init__.py
    └── frame_depth_averager.py          ← FrameDepthAverager class
```

**Modified files:**
```
code/src/preprocessing/forearm_extraction/
├── models/forearm_parameters.py         ← frame_id → frame_ids: List[int] + representative_frame_id: int
├── data_access/forearm_frame_parameters_filehandler.py  ← schema update + backward compat
└── gui/multivideo_frames_selector.py    ← enhanced VideoFramesSelector + groups panel + representative picker

code/scripts/_3_preprocessing/_3_forearm_extraction/
├── define_extraction_parameters.py      ← group-aware parameter building
└── extract_participant_forearm.py       ← multi-frame averaging code path

code/scripts/preprocess_pipeline_extract_forearm_manual.py  ← filename logic for groups
```

**Unchanged files** (no modification required):
```
clean_forearm_pointcloud.py
define_normals.py
define_forearm_mesh.py
kinect_mkv_manager.py
video_mp4_manager.py
```

---

## Implementation Plan

### Phase 1: Data Model & File Handler
**Goal:** Extend `ForearmParameters` to carry a list of frame IDs and update JSON persistence to match, with full backward compatibility.

**Tasks:**
- [x] 1.1 — In `forearm_parameters.py`, rename `frame_id: int` to `frame_ids: List[int]` and add `representative_frame_id: int` (both imported from `typing`); add `@property frame_id(self) -> int` returning `self.representative_frame_id`; add `@property is_averaged(self) -> bool` returning `len(self.frame_ids) > 1`; update `sort_forearm_parameters_by_video_and_frame` key to use `p.representative_frame_id`
- [x] 1.2 — In `forearm_frame_parameters_filehandler.py`, update `save()`: `asdict()` will serialize both `frame_ids` and `representative_frame_id` automatically — verify and remove any legacy `frame_id` remnants
- [x] 1.3 — In `forearm_frame_parameters_filehandler.py`, update `load()` with three backward-compat cases:
  - Old JSON has `"frame_id"` (int) only → `frame_ids = [data["frame_id"]]`, `representative_frame_id = data["frame_id"]`
  - New JSON has `"frame_ids"` and `"representative_frame_id"` → use both directly
  - New JSON has `"frame_ids"` but no `"representative_frame_id"` (partially migrated) → default `representative_frame_id = min(frame_ids)`
- [x] 1.4 — In `forearm_frame_parameters_filehandler.py`, update `is_valid_structure()`: accept old format (`"frame_id"` int only) and new format (`"frame_ids"` non-empty list + optional `"representative_frame_id"` int)

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/models/forearm_parameters.py`
- `code/src/preprocessing/forearm_extraction/data_access/forearm_frame_parameters_filehandler.py`

**Dependencies:** None

---

### Phase 2: Averaging Backend
**Goal:** Implement the `FrameDepthAverager` class and update `extract_forearm()` to use it for multi-frame groups.

**Tasks:**
- [x] 2.1 — Create `code/src/preprocessing/forearm_extraction/depth_averaging/__init__.py` (exports `FrameDepthAverager`)
- [x] 2.2 — Create `code/src/preprocessing/forearm_extraction/depth_averaging/frame_depth_averager.py`:
  - Class `FrameDepthAverager` with static method `average(mkv: KinectMKV, frame_ids: List[int], color_frame_id: int) -> o3d.geometry.PointCloud`
  - `color_frame_id` is required and comes from `video_config.representative_frame_id`; this is the frame whose color image is used for the averaged point cloud's colors
  - Uses running accumulator: for each `frame_id` in sorted order, load `frame.transformed_depth_point_cloud`, build validity mask (`Z > 0 and not NaN`), add to running sum and count arrays; discard the frame object after accumulation
  - After all frames: `avg_xyz = running_sum / np.maximum(valid_count, 1)`, mask out pixels with `valid_count == 0`
  - Separately load `mkv[color_frame_id].color` for `pcd.colors`
  - Returns `o3d.geometry.PointCloud` with points and colors set (same structure as `KinectFrame.generate_o3d_point_cloud()`)
- [x] 2.3 — In `extract_participant_forearm.py`, update `extract_forearm()`:
  - Import `FrameDepthAverager`
  - Always load the representative frame first: `representative_frame: KinectFrame = mkv[video_config.frame_id]` (via the `frame_id` property, which returns `representative_frame_id`)
  - Use `representative_frame` for `get_3d_cuboid_from_roi` (ROI cuboid, unchanged)
  - If `video_config.is_averaged`: replace point cloud generation with `point_cloud = FrameDepthAverager.average(mkv, video_config.frame_ids, color_frame_id=video_config.representative_frame_id)`
  - Else: `point_cloud = representative_frame.generate_o3d_point_cloud()` (existing path, zero behavior change)
- [x] 2.4 — In `preprocess_pipeline_extract_forearm_manual.py`, update `_process_single_forearm_frame()` filename generation:
  - If `params.is_averaged`: `base_filename = f"{video_stem}_frames_{min(params.frame_ids):04d}-{max(params.frame_ids):04d}_avg_N{len(params.frame_ids)}"`
  - Else (unchanged): `base_filename = f"{video_stem}_frame_{params.frame_id:04d}"`

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/depth_averaging/__init__.py` (new)
- `code/src/preprocessing/forearm_extraction/depth_averaging/frame_depth_averager.py` (new)
- `code/scripts/_3_preprocessing/_3_forearm_extraction/extract_participant_forearm.py`
- `code/scripts/preprocess_pipeline_extract_forearm_manual.py`

**Dependencies:** Phase 1

---

### Phase 3: GUI Enhancement
**Goal:** Add range and custom-group selection modes to `VideoFramesSelector`; update `MultiVideoFramesSelector` to work with frame groups.

**Tasks:**
- [x] 3.1 — Redesign `VideoFramesSelector` layout: split into left side (existing video canvas + slider) and right side (new "Frame Groups" panel containing a listbox of defined groups and group-management buttons). Geometry: ~1050 × 700 px total.
- [x] 3.2 — Add a "Selection Mode" `ttk.Notebook` or `tk.IntVar`-backed radio group with three tabs/options: **Single Frame** | **Range** | **Custom Group**
- [x] 3.3 — **Single Frame mode** (default): clicking "Add as Group" adds current frame as a 1-frame group `[current_frame_idx]` with `representative = current_frame_idx`; if a group for this exact single frame already exists, clicking removes it (toggle behavior matching current UX)
- [x] 3.4 — **Range mode**: "Set Start" button records `range_start = current_frame_idx`; "Set End" button records `range_end = current_frame_idx`; start/end labels update to show current values; "Add Range as Group" button validates `start <= end` and appends `list(range(range_start, range_end + 1))` to groups with `representative = range_start` (earliest frame); clears range markers
- [x] 3.5 — **Custom Group mode**: "Add Frame to Group" button appends `current_frame_idx` to a staging buffer `pending_group: List[int]`; "Finalize Group" button appends `sorted(set(pending_group))` to groups with `representative = min(pending_group)` (earliest added); clears buffer; staging count shown in a label
- [x] 3.6 — Groups listbox (right panel): each entry shows `"Group N: frame XXXX"` (single) or `"Group N: frames XXXX–YYYY (M frames) [rep: RRRR]"` (range/custom); "Remove Selected Group" button deletes the highlighted listbox entry from `groups` and `group_representatives`
- [x] 3.7 — **"Mark as Representative" button** (in the groups panel, below the listbox): enabled only when a group is selected in the listbox AND the current slider frame is a member of that group; on click, updates `self.group_representatives[selected_idx] = self.current_frame_idx` and refreshes the listbox entry to show the new `[rep: XXXX]`; disabled with a tooltip explanation otherwise
- [x] 3.8 — Update `_update_status()` to reflect the current groups state: `f"Groups defined: {len(self.groups)} | Frames covered: {sum(len(g) for g in self.groups)}"`
- [x] 3.9 — Change internal data structure: `self.selected_frames: set` → `self.groups: List[List[int]]` and `self.group_representatives: List[int]` (parallel list, same length); `_on_proceed()` stores both
- [x] 3.10 — Update `MultiVideoFramesSelector`:
  - `all_selected_frames: Dict[str, List[int]]` → `all_selected_groups: Dict[str, dict]` where each value is `{"groups": List[List[int]], "representatives": List[int]}`
  - In `open_frame_selector()`: pass existing groups and representatives for the video as `initial_groups` and `initial_representatives`; after the selector closes, read `selector.groups` and `selector.group_representatives`
  - Status label: show group count per video (`"[N Groups]"`)
  - In `validate_and_close()`: no logic change needed
- [x] 3.11 — Add `initial_groups: Optional[List[List[int]]]` and `initial_representatives: Optional[List[int]]` parameters to `VideoFramesSelector.__init__()` to pre-populate when re-opening an existing session

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/gui/multivideo_frames_selector.py`

**Dependencies:** None (can be developed in parallel with Phase 2)

---

### Phase 4: Integration
**Goal:** Connect the new GUI group structure to `ForearmParameters` creation in `define_extraction_parameters.py`; ensure end-to-end flow is correct.

**Tasks:**
- [x] 4.1 — In `define_forearm_extraction_parameters()`, update the iteration after multi-selector closes: iterate over `multi_selector.all_selected_groups` where each entry is `(video_path, {"groups": ..., "representatives": ...})`; for each group at index `i` create `ForearmParameters(frame_ids=groups[i], representative_frame_id=representatives[i], ...)`
- [x] 4.2 — ROI display: show `representative_frame_id` in `FrameROISquare` for the user to draw the ROI (the representative frame is now an explicit choice, not derived heuristically); update the window title to `f"ROI for {video_filename} — rep. frame {representative_frame_id} (group of {len(group)} frames)"` for multi-frame groups and `f"ROI for {video_filename} — frame {representative_frame_id}"` for single-frame groups
- [x] 4.3 — Update pre-loading logic: when metadata already exists and is loaded, reconstruct groups and representatives from `ForearmParameters` objects:
  ```python
  groups_by_video = defaultdict(lambda: {"groups": [], "representatives": []})
  for params in parameters_list:
      groups_by_video[params.video_filename]["groups"].append(params.frame_ids)
      groups_by_video[params.video_filename]["representatives"].append(params.representative_frame_id)
  ```
  Pass these as `initial_groups` and `initial_representatives` to `MultiVideoFramesSelector`
- [x] 4.4 — Update `preprocess_pipeline_extract_forearm_manual.py` log messages: for averaged groups log `f"frames {params.frame_ids} (representative: {params.representative_frame_id})"` rather than the single `params.frame_id`

**Files Modified:**
- `code/scripts/_3_preprocessing/_3_forearm_extraction/define_extraction_parameters.py`
- `code/scripts/preprocess_pipeline_extract_forearm_manual.py` (log messages only, minor)

**Dependencies:** Phase 1, Phase 2, Phase 3

---

### Phase 5: GUI Navigation Enhancements
**Goal:** Make frame navigation in `VideoFramesSelector` more precise and ergonomic: keyboard arrow stepping, a direct jump-to-frame entry, and a visual group indicator strip rendered above the slider.

**Tasks:**
- [x] 5.1 — Add a `tk.Canvas` group indicator strip (height 16 px, `bg="#1a1a1a"`) packed directly above the slider in the left panel (between the video canvas and the slider); implement `_redraw_group_indicator()`: iterate all `self.groups`, for each `frame_id` compute `x = int(fid / (self.total_frames - 1) * canvas_width)` and draw a 1 px red vertical line spanning the full canvas height; guard against `canvas_width <= 1` or `total_frames <= 1`; bind `<Configure>` on the strip canvas so any resize triggers a redraw; call `_redraw_group_indicator()` at the end of `_refresh_groups_listbox()` and once during `__init__` after `_build_ui()`
- [x] 5.2 — Replace `frame_info_lbl` (standalone label) with a compact navigation row packed in its place: a `ttk.Label("Frame:")`, a `ttk.Entry` (`self.frame_entry`, width 6, justify `tk.RIGHT`), a `ttk.Label(f"/ {self.total_frames - 1}")`, and keep a `ttk.Label` for `self.frame_info_lbl` or drop it in favour of the entry display — the entry itself shows the current frame number; bind `<Return>` and `<KP_Enter>` to `_on_frame_entry_submit()`
- [x] 5.3 — Implement `_on_frame_entry_submit()`: parse entry text as `int`, clamp to `[0, total_frames - 1]` (ignoring non-integer text silently), call `self.slider.set(idx)` (which fires `_on_slider_move` for full update), then call `self.parent.focus_set()` to restore keyboard navigation focus
- [x] 5.4 — Implement `_sync_frame_entry(idx: int)`: update the entry text only when `self.parent.focus_get() is not self.frame_entry` (avoids overwriting in-progress user input); call this from `_on_slider_move()` and once with `idx=0` at the end of `_build_ui()` to populate the initial value
- [x] 5.5 — Bind `<Left>` and `<Right>` keyboard events on `self.parent` (the Toplevel); in each handler return early if `self.parent.focus_get() is self.frame_entry` (so cursor movement in the entry is unaffected); otherwise compute `new_idx = max(0, current - 1)` or `min(total_frames - 1, current + 1)` and call `self.slider.set(new_idx)`
- [x] 5.6 — Enforce frame exclusivity across all groups: add a helper `_claimed_frames(self) -> set` that returns the union of all frame indices across `self.groups`; enforce in each mode: **Single Frame** — skip silently if the frame is already claimed; **Range** — when "Add Range as Group" is clicked, compute the overlap with claimed frames; if any overlap exists show a `messagebox.showwarning` listing the conflicting frames and abort (user must adjust start/end); **Custom Group** — in `_custom_add_frame()` skip the current frame silently if it is already claimed and update the pending label; in `_custom_finalize_group()` deduplicate against claimed frames before appending (overlap with already-committed groups is impossible at this point since the check happens at add-time, but apply `sorted(set(pending_group))` minus claimed frames as a safety net)

**Files Modified:**
- `code/src/preprocessing/forearm_extraction/gui/multivideo_frames_selector.py` — `VideoFramesSelector._build_ui()` layout, new `_redraw_group_indicator()`, new `_on_frame_entry_submit()`, new `_sync_frame_entry()`, updated `_on_slider_move()`, updated `_refresh_groups_listbox()`, keyboard bindings in `__init__`

**Dependencies:** Phase 3

---

## Testing Plan

### Unit Tests
- [ ] `ForearmParameters(frame_ids=[42], representative_frame_id=42)`: `frame_id == 42`, `is_averaged == False`
- [ ] `ForearmParameters(frame_ids=[42, 43, 44], representative_frame_id=42)`: `frame_id == 42`, `is_averaged == True`
- [ ] `ForearmParameters(frame_ids=[42, 43, 44], representative_frame_id=43)`: `frame_id == 43` — user-chosen representative respected
- [ ] `ForearmFrameParametersFileHandler.load()` on old-format entry (`"frame_id": 42`) produces `frame_ids=[42]`, `representative_frame_id=42`
- [ ] `ForearmFrameParametersFileHandler.load()` on new-format entry (`"frame_ids": [42,43,44], "representative_frame_id": 43`) produces correct object with `frame_id == 43`
- [ ] `ForearmFrameParametersFileHandler.load()` on entry with `"frame_ids"` but no `"representative_frame_id"` defaults `representative_frame_id = min(frame_ids)`
- [ ] `ForearmFrameParametersFileHandler` round-trip: save then load preserves `frame_ids` and `representative_frame_id` exactly
- [ ] `is_valid_structure()` returns `True` for old format, new format with representative, and new format without representative (all three)
- [ ] `FrameDepthAverager.average()` with `frame_ids=[N], color_frame_id=N` produces the same `(points, colors)` arrays as `KinectFrame.generate_o3d_point_cloud()` for frame N (within float32 tolerance)
- [ ] `FrameDepthAverager.average()` uses the color from `color_frame_id`, not the first frame — verify by passing `color_frame_id` pointing to a frame with a distinct color
- [ ] `FrameDepthAverager.average()` with all-zero depth at a pixel in every frame: that pixel is excluded from the output point cloud
- [ ] `FrameDepthAverager.average()` with a pixel valid in only 2 of 5 frames: average computed over those 2 frames only; pixel is included in output
- [ ] `FrameDepthAverager.average()` with synthetic frames where known noise is added: output std dev is lower than input std dev

### Integration Tests
- [ ] Full pipeline run with `frame_ids=[N]` (single frame): output filenames match `_frame_NNNN` pattern; PLY file is non-empty and passes through `clean_forearm_pointcloud`, `define_normals`, `define_forearm_mesh` without errors
- [ ] Full pipeline run with `frame_ids=[N, N+1, N+2, N+3, N+4]` (5-frame range): output filenames contain `_frames_NNNN-MMMM_avg_N5`; PLY file is non-empty and passes through all downstream steps
- [ ] Full pipeline run with a custom non-consecutive group (e.g., frames 10, 15, 20): `frame_ids=[10, 15, 20]`; output file named `_frames_0010-0020_avg_N3`; pipeline completes

### Manual Verification
- [ ] Launch `VideoFramesSelector` in **Single Frame** mode: navigate to frame 42, click "Add as Group" → groups listbox shows `"Group 1: frame 0042"`; navigate to frame 67, click "Add as Group" → `"Group 2: frame 0067"`; click "Save & Proceed" → `groups == [[42], [67]]`, `representatives == [42, 67]`
- [ ] Launch `VideoFramesSelector` in **Range** mode: navigate to frame 42, click "Set Start"; navigate to frame 47, click "Set End"; click "Add Range as Group" → listbox shows `"Group 1: frames 0042–0047 (6 frames) [rep: 0042]"` (default representative = start)
- [ ] In the above Range group: navigate to frame 44, click "Mark as Representative" → listbox updates to `"Group 1: frames 0042–0047 (6 frames) [rep: 0044]"`
- [ ] Navigate to frame 99 (outside the group), select Group 1 in the listbox → "Mark as Representative" button is disabled
- [ ] Launch `VideoFramesSelector` in **Custom Group** mode: navigate to frames 10, 15, 20 and click "Add Frame to Group" for each → staging shows "3 frames pending"; click "Finalize Group" → listbox shows `"Group 1: 3 frames [custom] [rep: 0010]"` (default representative = min)
- [ ] Navigate to frame 15, select Group 1, click "Mark as Representative" → listbox updates to `"Group 1: 3 frames [custom] [rep: 0015]"`
- [ ] Click "Remove Selected Group" with a group highlighted → group and its representative are removed from both internal lists and listbox
- [ ] Open `MultiVideoFramesSelector` with two videos, define one group per video with non-default representatives, validate → `all_selected_groups` carries correct groups and representatives per video
- [ ] Re-open an existing session (metadata file exists): groups and representatives pre-populate correctly from saved `frame_ids` and `representative_frame_id` fields; user can adjust representatives
- [ ] Run end-to-end on a session with one averaged group: confirm JSON params file contains `"representative_frame_id"` matching the user's choice; inspect the averaged PLY in the viewer — surface should appear smoother than a single-frame PLY
- [ ] Press `←` repeatedly from frame 5 → slider decrements one frame at a time down to 0; pressing again at 0 is a no-op
- [ ] Press `→` repeatedly near the end of the video → slider increments one frame at a time up to `total_frames - 1`; pressing again is a no-op
- [ ] Click into the frame entry, press `←` → cursor moves left inside the text box; the slider does not change
- [ ] Type `42` in the frame entry and press `Enter` → slider jumps to frame 42; video canvas and status bar update; entry shows `42`
- [ ] Type `99999` (beyond last frame) and press `Enter` → slider moves to `total_frames - 1` without error
- [ ] Type `abc` and press `Enter` → nothing changes, no error dialog
- [ ] Drag the slider while the entry does not have focus → entry updates in real time; navigate with arrows → entry updates
- [ ] Add a Range group (frames 42–47) → six red vertical bars appear in the indicator strip at the correct proportional positions
- [ ] Remove that group → red bars disappear immediately
- [ ] Resize the window horizontally → bars redraw at the correct proportional positions after the next group change
- [ ] Add Group 1 = frames 10–15; switch to Single Frame mode, navigate to frame 12, click "Add as Group" → nothing happens (frame is already claimed)
- [ ] Switch to Range mode, set start=14, end=20, click "Add Range as Group" → warning dialog mentions frames 14 and 15 as conflicts; no new group is added; set start=16, end=20 → group adds successfully
- [ ] Switch to Custom Group mode, add frames 5, 12, 25 → pending shows 2 frames (frame 12 skipped); finalize → Group 2 = [5, 25]
- [ ] Remove Group 1; navigate to frame 12, click "Add as Group" in Single Frame mode → Group 3 = [12] is created successfully

### Edge Cases
- [ ] Group with a single frame in **Range** mode (start == end) → valid; stored as `frame_ids=[N]`, `representative_frame_id=N`; `is_averaged == False`
- [ ] Attempting to add a frame that is already claimed by any existing group (in any mode) → silently skipped or warned; the duplicate is never stored
- [ ] Removing a group containing frames that overlap another (hypothetical inconsistency from future code) → `_claimed_frames()` always recomputes from scratch, so any freed frames become claimable after the removal
- [ ] Pixel with Z=0 in all N frames → pixel absent from output point cloud (not set to 0)
- [ ] Range including a dropped/missing frame from the MKV (frame returned as None/empty) → that frame's contribution is skipped; averaging continues over remaining valid frames; logged as a warning
- [ ] Very large range (N=50 frames) → running accumulator keeps memory to O(H×W×3) regardless of N; no out-of-memory error
- [ ] Custom group with duplicate frame indices added in GUI → `sorted(set(pending_group))` deduplicates before finalizing; representative recalculates to `min` of deduplicated list if it was pointing to a removed duplicate
- [ ] Attempting to "Finalize Group" with empty pending buffer → button is disabled or shows warning
- [ ] Attempting to "Add Range as Group" with `range_end < range_start` → validation error message; no group added
- [ ] User attempts to mark a frame outside the group as representative → "Mark as Representative" button is disabled; guard also in the click handler to prevent any race condition
- [ ] `representative_frame_id` set to a frame that is outside `frame_ids` — e.g. through manual JSON edit → `load()` detects the inconsistency, logs a warning, and resets `representative_frame_id = min(frame_ids)` rather than crashing

---

## Documentation Plan

- [ ] Update `docs/development/plans/README.md` (create if absent) with this plan listed under Active
- [ ] Add changelog entry: `docs/changelogs/forearm-frame-averaging.md` when shipped
- [ ] No CLAUDE.md changes needed (no new CuPy usage patterns introduced)
- [ ] No new user guide needed; the GUI changes are self-explanatory with the mode labels

---

## Rollback Plan

All changes are isolated to the forearm extraction sub-package and two scripts. No shared infrastructure is modified.

1. **Data model rollback:** Revert `forearm_parameters.py` and `forearm_frame_parameters_filehandler.py` to re-establish `frame_id: int`. Any JSON files already written with `"frame_ids"` must be deleted and regenerated interactively with the old code.

2. **Backend rollback:** Revert `extract_participant_forearm.py` and delete the `depth_averaging/` directory. The `FrameDepthAverager` is only called from `extract_forearm()` when `is_averaged == True`, so reverting the model (step 1) makes this branch unreachable regardless.

3. **GUI rollback:** Revert `multivideo_frames_selector.py` to the previous version. The old flat-set `VideoFramesSelector` is self-contained.

4. **Integration rollback:** Revert `define_extraction_parameters.py` and `preprocess_pipeline_extract_forearm_manual.py`.

No database migrations, shared infrastructure changes, or CI/CD modifications are involved.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Forearm moves between averaged frames, causing ghosting/blur | Med | High | UI warning in groups panel when N > 9: "Large group — ensure forearm is static across all frames"; no technical guard (user responsibility) |
| Memory spike from large frame groups | Med | Med | Use running accumulator (load one frame at a time, add to sum/count, discard) instead of stacking all N arrays; O(H×W×3) memory regardless of N |
| `_KinectReader` seek overhead for non-consecutive custom groups | Low | Med | `_KinectReader` already optimizes forward seeks; for non-consecutive groups, frames are accessed in sorted order to minimize backward seeks; document that non-consecutive groups should be kept small |
| Old JSON metadata breaks after model change | Low | High | Backward-compat loader: detect `"frame_id"` key → wrap as `[frame_id]`; `is_valid_structure()` accepts both formats; tested explicitly in unit tests |
| ROI drawn on `representative_frame_id` may not encompass the forearm at all group frames | Low | Low | The user explicitly chose the representative frame, so they have visibility; for large ranges, the ROI window title shows the full group span so the user is reminded to draw a generous ROI |
| `FrameDepthAverager` produces an `o3d.PointCloud` without colors if the representative frame's color fails to decode | Low | Low | Fallback: if `representative_frame_id` frame color is None, use a uniform white color array; log a warning |
| GUI becomes too complex with three modes + representative picker | Low | Med | Default mode is **Single Frame** (zero learning curve for existing users); Range and Custom Group modes are opt-in; "Mark as Representative" button is in the groups panel and only enabled when relevant, so it does not clutter the primary workflow |
| User forgets to set a representative frame; default `min(frame_ids)` is semantically wrong for their use case | Low | Low | Default is documented in tooltip and listbox label; user can always re-open the selector and adjust before running the extraction |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1: Data Model & File Handler | Small (~1–2 h) | None |
| Phase 2: Averaging Backend | Medium (~3–4 h) | Phase 1 |
| Phase 3: GUI Enhancement | Medium-Large (~4–6 h) | None (parallel with Phase 2) |
| Phase 4: Integration & Wiring | Small (~1–2 h) | Phase 1, 2, 3 |
| Phase 5: GUI Navigation Enhancements | Small (~1–2 h) | Phase 3 |

---

## References

- `docs/kinect_frame_averaging_insight.docx` — depth noise analysis and √N averaging rationale
- `code/src/preprocessing/common/data_access/kinect_mkv_manager.py` — `KinectFrame.transformed_depth_point_cloud` property
- `code/src/preprocessing/forearm_extraction/models/forearm_parameters.py` — current data model
- `code/src/preprocessing/forearm_extraction/gui/multivideo_frames_selector.py` — current GUI
- `code/scripts/_3_preprocessing/_3_forearm_extraction/extract_participant_forearm.py` — current extraction backend
- `code/scripts/preprocess_pipeline_extract_forearm_manual.py` — pipeline orchestrator

---
