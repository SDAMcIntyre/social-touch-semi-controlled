# Plan: Kinect parallax correction at the data-access layer

**Created:** 2026-04-14 12:00
**Approved:** —
**Completed:** —
**Author:** Basil
**Status:** In Progress
**Branch:** `feature/kinect-parallax-tolerance-band`

---

## Overview

**What:** Apply the calibrated median RGB↔depth parallax shift
(`(dv, du) = (−1, +10)` px) inside `KinectFrame` so that
`transformed_depth` and `transformed_depth_point_cloud` return arrays
already aligned to the color frame, with NaN-padded edges.

**Why:** The audit recorded in
[`note-azure-kinect-rgb-depth-parallax.md`](../../knowledge-base/note-azure-kinect-rgb-depth-parallax.md)
showed the data-access layer is **unaccounted** — every consumer that
reads `point_cloud[v_rgb, u_rgb]` silently inherits a ~10 px median
offset that becomes a metres-scale XYZ error near depth edges.

**How:** Add a constructor flag `parallax_correction: bool = True` to
`KinectFrame` (and forward it through `KinectMKV` / `_KinectReader`).
When enabled, properties wrap the cached raw arrays with a single
NaN-padded shift function. The audit / measurement tool opts out.

---

## Problem Statement

`pyk4a.transformation.depth_image_to_color_camera()` corrects the
color/IR baseline rigidly, but the result is sub-pixel-accurate only
on smooth surfaces. At our 500–800 mm operating range the residual
band is **median Δu = +10 px, Δv = −1 px**, worst case ±25 px,
97 % sign-consistent (n = 35 measurements; see KB note §5).

`KinectFrame.transformed_depth_point_cloud` is the public per-pixel
paired API. Every downstream module that pairs it with `KinectFrame.color`
inherits the offset. The KB note §6 audit table lists the affected
consumers (xyz extractors, depth-to-tiff, forearm extraction, the
point-cloud wrapper, the scene viewer). Two follow-up idea files
proposed handling this per-consumer; this plan instead pushes the
correction down into the data-access layer so a single fix benefits
**every** RGB-paired consumer at once.

The change is **explicit by construction**: future readers of
`KinectFrame` see the constructor parameter and know the data is no
longer raw pyk4a output. Calibration / measurement code must opt out
explicitly with `parallax_correction=False` — making the audit tool's
deviation from the default visible in code review.

## Goals

### In Scope
1. New constructor parameter `parallax_correction: bool = True` on
   `KinectFrame` and `KinectMKV` (passed through `_KinectReader`).
2. When enabled, both `transformed_depth` and
   `transformed_depth_point_cloud` apply a NaN-padded shift of
   `(dv = −1, du = +10)` so that index `(v, u)` returns the surface
   visible at color pixel `(v, u)`.
3. Centralise the shift constants and the shift function in one
   module so any future re-calibration touches a single place.
4. Audit / measurement tool (`kinect_rgb_depth_viewer.py` +
   `audit_rgb_depth_registration.py`) **must** opt out so it
   continues to *measure* the raw offset, not absorb it.
5. Update KB note §6 audit table: `KinectFrame` and
   `KinectPointCloudView` move from `unaccounted` to
   `mitigated (median shift)`.
6. Unit tests for the shift function and the flag plumbing.

### Out of Scope
- Per-frame or per-pixel calibration of the parallax (the shift is
  the median, not a calibrated map).
- Replacing the area-sampling strategy in
  `xyz_extractor_ellipse_depth.py` — that module remains "mitigated"
  by its own ellipse mask + z-guard; the median shift is additive
  benefit, not a replacement.
- Operating ranges outside 500–800 mm.
- `KinectFrame.depth` (the raw depth in depth-camera geometry) — the
  parallax does not apply to it, so it is left untouched.
- Backwards-compatibility shims for callers that already manually
  shift; if any exist, they must be removed when discovered (this is
  in-scope cleanup, but no such caller is currently known).

## Success Criteria

- [ ] `KinectFrame(capture, color_format)` returns parallax-corrected
      depth/point-cloud by default.
- [ ] `KinectFrame(capture, color_format, parallax_correction=False)`
      returns raw pyk4a output, byte-identical to current behaviour.
- [ ] `KinectMKV(path, parallax_correction=False)` propagates the
      flag to every yielded `KinectFrame`.
- [ ] Re-running `audit_rgb_depth_registration.py` (post-opt-out) on
      a previously-measured session reproduces a Δu median ≈ +10 px,
      proving the audit is not silently zeroed.
- [ ] Existing consumer scripts run end-to-end on one ST13 session
      without crashing on the new NaN edges.
- [ ] KB note §6 audit table updated with the new statuses.

---

## Technical Design

### Approach

**Single-source the shift.** Add
`code/src/preprocessing/common/data_access/parallax_correction.py`
exposing:

```python
PARALLAX_SHIFT_PX_RGB_TO_DEPTH = (-1, +10)   # (dv, du), median over
                                              # 500–800 mm, n=35
PARALLAX_CORRECTION_VERSION = "median_shift_v1"

def apply_parallax_shift(arr: np.ndarray) -> np.ndarray:
    """
    Return a copy of `arr` (H, W) or (H, W, C) such that
    out[v, u, ...] == arr[v + dv, u + du, ...] when in bounds, NaN
    otherwise. Float dtype; integer inputs are promoted to float32 so
    NaN can be represented.
    """
```

The transform: `out[v, u] = src[v + dv, u + du]` with `(dv, du) = (−1, +10)`.

This means a depth value originally located at raw pixel `(v_raw, u_raw)`
is relocated to corrected pixel `(v_raw − dv, u_raw − du)`, e.g.
**raw `(500, 500)` → corrected `(501, 490)`**. Top row (v=0) and the
right-most 10 columns (u ∈ [W−10, W)) become NaN by construction.

**Thread the flag through the data-access layer.**
`KinectFrame.__init__` gains `parallax_correction: bool = True`. Both
`transformed_depth` and `transformed_depth_point_cloud` properties
wrap the cached raw value when the flag is True; the shifted result
is itself cached so repeated access does not re-shift.
`_KinectReader` accepts and stores the flag, forwarding it via
`_get_frame_from_capture`. `KinectMKV.__init__` accepts it and
forwards to `_KinectReader`.

**Audit tool opt-out.**
`audit_rgb_depth_registration.py` constructs
`KinectMKV(path, parallax_correction=False)`. The viewer itself does
not know or care; it just gets raw frames.

**`KinectPointCloudView` is unchanged in code.** It already reads
`frame_object.transformed_depth_point_cloud`; once the underlying
property is corrected, the wrapper's `.color` / `.points` pair becomes
correctly aligned automatically. Only its docstring changes — to
reference the correction.

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| Apply shift in `KinectPointCloudView` only | Smallest surface; matches the original idea-doc scope | Most consumers (xyz extractors, forearm extraction, depth-to-tiff, frame_depth_averager, mkv_stream_analyze) bypass the wrapper and stay unaccounted | Rejected |
| New attribute `transformed_depth_point_cloud_rgb_aligned` on `KinectFrame` | Old behaviour fully preserved; explicit by name | Two parallel attributes ⇒ readers easy to forget which to use; doesn't fix the default behaviour | Rejected |
| **Constructor flag, default ON, replaces returned value** | One flag flips behaviour; default is the safe answer; opt-out is one keyword for the audit tool | Behaviour change to existing callers; audit tool must remember to opt out | **Chosen** |
| Shift via `np.roll` | One-liner | Wraps garbage from the opposite edge into the corrected 10 columns | Rejected |
| Crop both color and depth to drop the 10 px strip | No NaNs; pairing trivially valid | Changes `(H, W)` for downstream code; high blast radius | Rejected |

### Architecture Constraints (from KB note)

- The error has a **direction** (+u-biased), not just a magnitude — a
  scalar shift is the right kind of correction for the median, but
  callers integrating over a region still need their own slack for
  the IQR / worst case.
- The 500–800 mm operating range is fixed; if a future pipeline
  operates >1 m, the band must be re-measured (KB note §4).
- The audit tool must measure raw output, not absorb it (KB note §6
  classifies it `n/a` for that reason).

### Architecture Changes

```
code/src/preprocessing/common/data_access/
├── __init__.py
├── kinect_mkv_manager.py         — KinectFrame, KinectMKV, _KinectReader
│                                   gain `parallax_correction: bool = True`;
│                                   transformed_depth / point_cloud properties
│                                   wrap with apply_parallax_shift when True
├── kinect_pointcloud_wrapper.py  — docstring update only
└── parallax_correction.py        — NEW: constants + apply_parallax_shift()
```

---

## Implementation Plan

### Phase 1: Correction primitive
**Goal:** Pure, tested function for the shift.
**Started:** 2026-04-14
**Completed:** 2026-04-14

**Tasks:**
- [x] Create `parallax_correction.py` with
      `PARALLAX_SHIFT_PX_RGB_TO_DEPTH`, `PARALLAX_CORRECTION_VERSION`,
      and `apply_parallax_shift()`.
- [x] Module docstring links to `note-azure-kinect-rgb-depth-parallax.md`
      and states the operating range.
- [x] Unit tests in `code/tests/test_parallax_correction.py`:
  - shape preservation for `(H, W)` and `(H, W, 3)` inputs
  - NaN region geometry: row 0 + cols `W-10..W-1` become NaN
  - in-bounds equivalence: `out[v, u] == src[v − 1, u + 10]`
  - integer dtype promoted to float32 (so NaN is representable)
  - explicit pixel mapping test: a sentinel value placed at
    `src[499, 510]` reappears at `out[500, 500]`

**Files Modified:**
- `code/src/preprocessing/common/data_access/parallax_correction.py` — new
- `code/tests/test_parallax_correction.py` — new

**Dependencies:** None

### Phase 2: KinectFrame / KinectMKV plumbing
**Goal:** Thread the flag through the data-access layer; default ON.
**Started:** 2026-04-14
**Completed:** 2026-04-14

**Tasks:**
- [x] `KinectFrame.__init__`: add `parallax_correction: bool = True`,
      store on `self`.
- [x] `KinectFrame.transformed_depth`: when flag is True, return
      `apply_parallax_shift(raw)`; cache the corrected result alongside
      the raw cache.
- [x] `KinectFrame.transformed_depth_point_cloud`: same wrapping.
- [x] `_KinectReader.__init__`: accept and store the flag.
- [x] `_KinectReader._get_frame_from_capture`: pass the flag to
      `KinectFrame(...)`.
- [x] `KinectMKV.__init__`: add `parallax_correction: bool = True`,
      forward to `_KinectReader`.
- [x] Class docstrings on `KinectFrame` and `KinectMKV` describe the
      default and link to the KB note.
- [x] `KinectPointCloudView` docstring updated to mention the
      correction (no behavioural change in this file).

**Files Modified:**
- `code/src/preprocessing/common/data_access/kinect_mkv_manager.py` — flag plumbing + property wrapping
- `code/src/preprocessing/common/data_access/kinect_pointcloud_wrapper.py` — docstring only

**Dependencies:** Phase 1

### Phase 3: Audit opt-out + downstream sanity check
**Goal:** Audit pipeline keeps measuring raw offset; consumer scripts run.
**Started:** 2026-04-14
**Completed:** —

**Tasks:**
- [x] `audit_rgb_depth_registration.py`: construct
      `KinectMKV(path, parallax_correction=False)`.
      (`KinectRgbDepthViewer` now accepts `parallax_correction` and
      defaults it to `False`; the audit script passes it explicitly.)
- [ ] Re-run audit on one previously-measured session; confirm the
      generated CSV's Δu median ≈ +10 px (within the prior IQR).
- [ ] Smoke-test consumer scripts on one ST13 session (no exceptions,
      output files written):
      `view_xyz_stickers_with_depth_data.py`,
      `view_somatosensory_3d_scene.py`,
      `extract_depth_to_tiff.py`,
      `extract_participant_forearm.py`.
- [x] Update KB note §6 audit table:
      `KinectFrame` (new row) and `KinectPointCloudView` reclassified
      from `unaccounted` to `mitigated (median shift)`; rationale
      column cites the new module + flag.
- [x] Mark related ideas as superseded by this plan in their `Status:`
      headers (preserve, do not delete):
      `kinect-pointcloud-wrapper-parallax-tolerance.md`,
      `xyz-extractor-centroid-parallax-tolerance.md`.

**Files Modified:**
- `code/scripts/_3_preprocessing/_0_kinect_diagnostics/audit_rgb_depth_registration.py` — opt out
- `docs/development/knowledge-base/note-azure-kinect-rgb-depth-parallax.md` — table refresh
- `docs/development/plans/ideas/kinect-pointcloud-wrapper-parallax-tolerance.md` — status note
- `docs/development/plans/ideas/xyz-extractor-centroid-parallax-tolerance.md` — status note

**Dependencies:** Phase 2

---

## Testing Plan

### Unit Tests
- [ ] `apply_parallax_shift` shape preservation for 2-D and 3-D inputs
- [ ] NaN region geometry: row 0 + cols `[W-10, W)` are NaN, others finite
- [ ] In-bounds equivalence: `out[v, u] == src[v − 1, u + 10]`
- [ ] Integer dtype promotion to float32
- [ ] Sentinel mapping: `src[500, 500]` value lands at `out[501, 490]`
- [ ] `KinectFrame(parallax_correction=False)` returns array equal to
      the raw `_capture.transformed_depth_point_cloud`
- [ ] `KinectFrame()` (default) returns the shifted array; row 0 NaN,
      right 10 cols NaN, interior matches raw at `(v−1, u+10)`

### Integration Tests
- [ ] Open one MKV via `KinectMKV(path)`, iterate, generate Open3D
      colored point cloud — no exceptions; ~10/W of points dropped due
      to edge NaN, otherwise unchanged
- [ ] Open same MKV via `KinectMKV(path, parallax_correction=False)`,
      run `audit_rgb_depth_registration.py` against a known measurement
      CSV — recovered Δu median ≈ +10 px (within prior IQR)

### Manual Verification
- [ ] Run `view_xyz_stickers_with_depth_data.py` on an ST13 session
      that historically showed visible XYZ jumps for small stickers;
      visually confirm the corrected XYZ traces sit closer to the
      sticker depth blob than the raw run

### Edge Cases
- [ ] Frame where `transformed_depth_point_cloud is None`: property
      returns None, no shift attempted
- [ ] Frame with all-zero point cloud (start of recording): shift
      returns zeros, NaN edges
- [ ] Repeated property access uses the cached corrected array (do
      not re-shift each call)

---

## Documentation Plan

- [ ] Module docstring on `parallax_correction.py` linking to
      `note-azure-kinect-rgb-depth-parallax.md` and stating the
      operating range
- [ ] Class docstrings on `KinectFrame` and `KinectMKV` describing
      the default behaviour and pointing to the opt-out flag
- [ ] Update KB note §6 audit table
- [ ] Mark superseded ideas with a `Status: Superseded by …` note
- [ ] No `CLAUDE.md` change required

---

## Rollback Plan

1. **Before deployment:** revert the Phase-2 / Phase-3 commits;
   delete `parallax_correction.py` and its test.
2. **Data considerations:** the change is in-memory at frame load —
   no migrations, no on-disk format change. Audit CSVs already on
   disk under `F:/_tmp/kinect-measurement-offsets/` are unaffected.
3. **Rollback procedure:** `git revert` the implementation commits;
   audit tool will go back to the (pre-correction) default.

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Audit tool not opted out — future calibration silently sees ~0 px offset | Med | High | Phase-3 task explicitly toggles the flag and re-runs against stored CSVs to confirm the +10 px median is recovered |
| Edge NaNs break a consumer that assumed `point_cloud[..., 2] > 0` was the only validity check | Low–Med | Med | Smoke-test the four main consumer scripts in Phase 3; existing `valid_mask` patterns already combine `> 0` with `~np.isnan` (see `KinectFrame.generate_o3d_point_cloud`) |
| Median shift is wrong-direction for an out-of-spec session (>1 m) | Low | Med | Operating range stated on the constants and in module docstring; future >1 m work must re-measure (KB note §7 reusable pattern) |
| Caching the corrected array doubles per-frame memory | Low | Low | Same lifetime as the existing raw cache; one frame at a time |

---

## Timeline

| Phase | Estimated Effort | Dependencies |
|-------|-----------------|--------------|
| Phase 1 | ~1 hour | None |
| Phase 2 | ~1 hour | Phase 1 |
| Phase 3 | ~1 hour (+ smoke-test runtime) | Phase 2 |

---

## References

- KB note: [`docs/development/knowledge-base/note-azure-kinect-rgb-depth-parallax.md`](../../knowledge-base/note-azure-kinect-rgb-depth-parallax.md)
- Idea (wrapper): [`docs/development/plans/ideas/kinect-pointcloud-wrapper-parallax-tolerance.md`](../ideas/kinect-pointcloud-wrapper-parallax-tolerance.md)
- Idea (centroid): [`docs/development/plans/ideas/xyz-extractor-centroid-parallax-tolerance.md`](../ideas/xyz-extractor-centroid-parallax-tolerance.md)
- Completed parent: [`docs/development/plans/completed/kinect-rgb-depth-parallax-tolerance-band.md`](../completed/kinect-rgb-depth-parallax-tolerance-band.md)
- Audit script:
  `code/scripts/_3_preprocessing/_0_kinect_diagnostics/analyse_rgb_depth_offset_dataset.py`
