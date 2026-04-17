# Kinect depth access — single path through `KinectFrame` / `KinectMKV`

## 1. Symptom

Production code that reads Kinect depth pixels occasionally returns XYZ
values that are metres off the true surface at depth edges, even after the
median RGB↔depth parallax shift is applied inside `KinectFrame`.

## 2. Investigation

The fix implemented by `plans/active/kinect-frame-parallax-correction.md`
moves the `(dv, du) = (−1, +10)` shift into `KinectFrame.transformed_depth`
and `KinectFrame.transformed_depth_point_cloud` so that every consumer of
the `KinectFrame` / `KinectMKV` / `KinectPointCloudView` API receives
parallax-corrected arrays by default. The correction is effective **only
for readers that use the data-access layer**. Any code that imports
`pyk4a` directly and calls `capture.transformed_depth` or
`capture.transformed_depth_point_cloud` sees raw output and silently
reintroduces the ~10 px median offset.

A two-agent audit (see
`plans/pending/kinect-depth-access-guardrails.md`) catalogued every depth
reader in the repo. Most go through the data-access layer. A small number
import `pyk4a` directly and are listed in the allowlist below.

## 3. Root cause

The correction is applied inside `KinectFrame`, not inside `pyk4a`. There
is no mechanism in `pyk4a` itself that prevents a caller from reading raw
depth. The invariant (*all depth reads flow through `KinectFrame` /
`KinectMKV` / `KinectPointCloudView`*) is therefore a **project-level
convention**, not a library guarantee. It survives only if every new
contribution respects it.

## 4. Architecture constraints

- `pyk4a` is still required for MKV *metadata* (recording configuration,
  serial number) and for structural *validation* of MKV streams — these
  are not frame-pixel reads and do not benefit from the correction.
- RGB-only extraction paths (e.g. `extract_color_to_mp4.py`) read
  `capture.color` and never touch depth, so the correction is irrelevant
  to them.
- The audit / calibration widget `KinectRgbDepthViewer` **must** be able
  to opt out and see raw data; otherwise it would absorb the very offset
  it is trying to measure.

The invariant therefore cannot be enforced by a blanket "no `pyk4a`
outside `data_access/`" rule. It needs an explicit allowlist.

## 5. Fix applied

**Invariant.** All code that reads Kinect depth pixels MUST go through
`code/src/preprocessing/common/data_access/`:

- `KinectFrame` / `KinectMKV` — the blessed per-frame and per-recording
  reader
- `KinectPointCloudView` — the RGB↔XYZ paired-pixel wrapper

Direct `pyk4a` imports outside `code/src/preprocessing/common/data_access/`
are allowed only in the files enumerated in the allowlist below, and new
entries require a plan + review that states *why* the raw path is needed
and *what* it is reading (depth vs color vs metadata vs import probe).

### Allowlist

| File | What it uses `pyk4a` for | Why raw `pyk4a` is OK |
|---|---|---|
| `code/src/primary_processing/mkv_video_management/mkv_stream_analyze.py` | `PyK4APlayback`, `K4AException`, `ColorResolution`, `DepthMode` — iterates an MKV to validate stream structure and frame counts. | Structural validation, not pixel processing. Output is integrity metadata, never forwarded to downstream consumers as depth. |
| `code/scripts/_2_primary_processing/_2_generate_rgb_depth_video/generate_mkv_stream_analysis.py` | `K4AException` — imported to catch exceptions raised by `MKVStreamAnalyzer` (which is itself allowlisted above). | Exception-handling import only; no playback, no pixel reads. |
| `code/scripts/_2_primary_processing/_2_generate_rgb_depth_video/extract_color_to_mp4.py` | `PyK4APlayback`, `K4AException`, `PyK4ACapture`, `ColorResolution` — iterates an MKV to extract the **color** stream to MP4. | Reads `capture.color` only; never touches `capture.depth` / `.transformed_depth` / `.transformed_depth_point_cloud`. |
| `code/src/preprocessing/stickers_analysis/xyz/models/xyz_metadata_model.py` | `PyK4APlayback` — reads `playback.configuration` (camera_fps, color_format, color_resolution, depth_mode). | Metadata only; no frame reads, no pixel data. |
| `code/scripts/_3_preprocessing/_3_forearm_extraction/debug_opengl.py` | `__import__("pyk4a")` (line 107) and `from pyk4a import PyK4APlayback` (line 227) — both inside diagnostic availability probes. | Import-only; no playback opened, no captures, no pixels. |

Any file not in this allowlist that imports `pyk4a` is a regression.

### Audit-mode opt-out

Exactly one explicit pathway exists for code that *must* see raw depth:

```python
KinectMKV(mkv_path, parallax_correction=False)              # low-level
KinectRgbDepthViewer(mkv_path, parallax_correction=False)   # GUI widget
```

The only current caller of that opt-out is
`code/scripts/_3_preprocessing/_0_kinect_diagnostics/audit_rgb_depth_registration.py`
(and its flat-wall variant). New callers must:

- Pass `parallax_correction=False` **explicitly** (no default).
- State in a comment or docstring why raw data is required.
- Be diagnostic / calibration code, not production data.

Default values for `parallax_correction` are `True` everywhere
(`KinectFrame`, `KinectMKV`, `KinectRgbDepthViewer`).

## 6. Reusable pattern

When reviewing a PR that adds a new reader of Kinect data:

- [ ] Run `grep -rn "from pyk4a\|import pyk4a" code/`. Compare against the
      allowlist in §5. Any new row is a regression unless the PR updates
      this note with a justified entry.
- [ ] If the new code reads depth (anything from `capture.depth` /
      `capture.transformed_depth` / `capture.transformed_depth_point_cloud`
      / `KinectFrame.depth` / `KinectFrame.transformed_depth*`), confirm it
      goes through `KinectFrame` / `KinectMKV` / `KinectPointCloudView`
      with `parallax_correction=True` (the default).
- [ ] If the new code passes `parallax_correction=False`, confirm it is a
      diagnostic / calibration tool and the rationale is stated in code.
- [ ] If the new code uses `pyk4a` for non-depth access (metadata, stream
      structure, color-only, import probe), add it to the allowlist above
      as part of the same PR, with a one-line justification.

## 7. References

| Document | Location |
|---|---|
| Correction primitive | `code/src/preprocessing/common/data_access/parallax_correction.py` |
| Data-access layer | `code/src/preprocessing/common/data_access/kinect_mkv_manager.py` |
| Paired-pixel wrapper | `code/src/preprocessing/common/data_access/kinect_pointcloud_wrapper.py` |
| Parent plan (correction) | `docs/development/plans/active/kinect-frame-parallax-correction.md` |
| This plan (guardrails) | `docs/development/plans/pending/kinect-depth-access-guardrails.md` |
| Parallax characterisation | `docs/development/knowledge-base/note-azure-kinect-rgb-depth-parallax.md` |
