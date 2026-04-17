# Dev Note: Azure Kinect RGB ↔ depth parallax at near range

**Problem class:** Per-pixel correspondence between the Azure Kinect color
frame and `transformed_depth` / `transformed_depth_point_cloud` is **not**
sub-pixel accurate at this project's operating range (~500–800 mm). A
systematic horizontal offset of ~10 px (median) with a worst case of
~25 px is present and direction-consistent across sessions and devices.

| Field | Value |
|-------|-------|
| Resolved in | (none — characterised, not fixed) |
| Affected versions | All Azure Kinect color/depth captures via pyk4a 1.5.0 used in this project |
| Platform | Windows 11, Azure Kinect DK |
| Operating range characterised | **500–800 mm only** (the only range used by downstream pipelines) |
| Related plan | `docs/development/plans/completed/kinect-rgb-depth-parallax-tolerance-band.md` |

---

## 1. Symptom

When the same physical feature (sticker centre, finger pad, fabric corner)
is clicked on the RGB panel and on the `transformed_depth` panel of
`KinectRgbDepthViewer`, the two pixel coordinates do not coincide.
Instead, the depth-panel click lands several pixels to the **right** of
the RGB-panel click (Δu > 0), with a small, mostly-zero vertical
component. The size of the offset depends on where in the frame the
feature sits and how close it is to a depth discontinuity, but the sign
is essentially constant.

Because every downstream module that reads
`point_cloud[v_rgb, u_rgb]` (or equivalently uses an RGB-derived pixel
mask to sample depth) treats the RGB-frame pixel as a depth-frame index,
this offset turns into an XYZ error whose magnitude depends on the local
depth gradient.

---

## 2. Investigation path

Two measurement campaigns with `KinectRgbDepthViewer`:

- **Phase 1** (3 CSVs under `F:/_tmp/kinect-measurement-offsets/phase-1/`):
  single-click measurements on multiple sessions to scope the offset.
- **Phase 2** (5 CSVs under `F:/_tmp/kinect-measurement-offsets/phase-2/`):
  median-of-5-clicks per feature with a 5×5 depth-gradient
  `near_edge` flag, restricted to operator-judged "flat" body regions
  (forearm, palm; `near_edge=False` on 24/27 rows).

The flat-wall verification originally specified in the parent plan was
**not** run: a one-distance flat-wall capture cannot quantify
range-dependence, and our downstream consumers operate only in the
near-field bin where the original ambiguity (parallax vs miscalibration)
does not change what they need to absorb. We therefore characterise the
operating range directly.

Aggregation script:
`code/scripts/_3_preprocessing/_0_kinect_diagnostics/analyse_rgb_depth_offset_dataset.py`.
Run on the combined Phase-1 + Phase-2 root, filtering to
`500 mm ≤ z_mm ≤ 800 mm` and dropping rows with `|Δ| > 100 px`
(catches a single ST13 misclick where the depth-axis click landed in a
NaN region — see Reusable Pattern §6.5).

---

## 3. Root Cause

The Azure Kinect color sensor and IR/depth sensor are physically offset
on the device. pyk4a's `transformation.color_image_to_depth_camera()` /
`depth_image_to_color_camera()` corrects this rigidly — but the
correction is exact only on a smooth surface whose depth is known
everywhere. Wherever the scene has a depth discontinuity (an arm edge,
a finger silhouette, a sticker-on-skin step), the rigid reprojection
becomes ambiguous because two real surfaces project to the same RGB
pixel from different angles. pyk4a fills the resulting "occlusion gap"
in the transformed depth from the nearer surface, which means the
**transformed depth value at RGB pixel (u, v) is the depth of the
near-camera surface, not necessarily of the surface visible in RGB at
(u, v)**. The visible offset is the parallax between the two camera
viewpoints projected onto the depth-edge geometry.

This is a known and documented Azure Kinect behaviour (see references
§7); it is **not** a sensor defect on our specific devices.

The horizontal `+Δu` sign is consistent with the color/IR baseline
direction on the Azure Kinect DK.

---

## 4. Architecture Constraints

- **No firmware fix.** The parallax is geometric; pyk4a's
  transformation is the best closed-form correction available without
  per-frame per-pixel ray casting.
- **Per-pixel paired access is the natural API.** `KinectFrame.color`
  and `KinectFrame.transformed_depth_point_cloud` share a `(H, W)`
  grid; downstream code that wants both naturally uses the same
  index. Removing this affordance is not feasible.
- **The error has a *direction*, not just a magnitude.** A scalar
  tolerance hides this; consumers that integrate over a small region
  in pixel space need to be biased in the −u direction (away from the
  baseline) if they want to recover the visible-RGB feature, or
  accept the +u-shifted depth blob if they want the true sticker
  surface.
- **Operating range is fixed.** All current pipelines (sticker XYZ
  extraction, contact projection, depth-weighted aggregation) run on
  hand-on-forearm geometry within ~50–80 cm of the camera. This note
  characterises that range only; if the operating range ever extends
  to >1 m, the band must be re-measured.

---

## 5. Quantified tolerance band (500–800 mm operating range)

Pooled across 8 CSVs (Phase-1 + Phase-2), n = 35 measurements in the
500–800 mm bin after dropping one |Δ| > 100 px misclick:

| Metric | Value |
|--------|-------|
| Median \|Δ\| | **10.20 px** |
| IQR \|Δ\| | **[7.77, 14.50] px** |
| Worst observed \|Δ\| | **25.06 px** |
| Median Δu | **+10.00 px** |
| Median Δv | **−1.00 px** |
| Δu sign-consistency | **97 %** (modal sign = +) |
| Rows with `near_edge=True` | 3 / 35 |

Plot: `F:/_tmp/kinect-measurement-offsets/aggregate_near_field.png`
(top: |Δ| histogram with median + IQR + worst overlaid; bottom:
Δu vs u_rgb, colour = v_rgb).

Recommended downstream tolerance phrasing:

> At 500–800 mm camera distance, the Azure Kinect RGB pixel `(u, v)`
> may be displaced from its corresponding `transformed_depth` pixel
> by a median **±10 px** with worst case **±25 px**, biased in the
> **+u** direction (97 % sign-consistency). Consumers that look up
> `point_cloud[v, u]` from an RGB-derived feature must either
> (a) absorb this envelope, or (b) sample over a region ≥ ±25 px in
> u and apply a depth-plausibility filter to discard the wrong
> surface.

---

## 6. Downstream-consumer audit checklist

For each module that pairs RGB-derived pixel coordinates with
transformed-depth (or its point-cloud equivalent), the table below
records how the ±10 px median / ±25 px worst-case envelope is
handled.

Statuses:
- **absorbed** — the module applies a region-aggregation or
  tolerance that exceeds the worst-case envelope, so the parallax is
  invisible to downstream code.
- **mitigated** — the module reduces the impact (e.g. via a small
  spatial mask or a depth-plausibility filter) but does not fully
  cover the worst-case envelope.
- **unaccounted** — the module reads `point_cloud[v_rgb, u_rgb]`
  directly without compensating, and downstream XYZ inherits the
  parallax-induced error.
- **n/a** — the module does not pair RGB with depth.

| File | Paired-access site | Status | Rationale |
|------|--------------------|--------|-----------|
| `code/src/preprocessing/stickers_analysis/xyz/core/xyz_extractor_centroid.py` | `get_xyz_from_point_cloud(point_cloud, px, py)` (line ~344) — single `point_cloud[iy, ix]` lookup at RGB-derived sticker centroid. | **unaccounted** | Single-pixel lookup with no spatial slack and no depth-plausibility check. At the median +10 px offset the lookup lands ~10 px outside the sticker depth blob — likely on background or sticker edge — and returns a z value that may be metres off. |
| `code/src/preprocessing/stickers_analysis/xyz/core/xyz_extractor_ellipse_depth.py` | `_sample_depth_within_mask` (line ~160) — fills an ellipse mask at RGB-derived `(px, py)` then samples `point_cloud[mask]`; `_aggregate_depth` then keeps only `z ≤ z_min + 10 mm`. | **mitigated** | The ellipse mask provides spatial slack proportional to the sticker's RGB extent; the 10 mm sticker-size z-guard rejects background pixels that bleed in via parallax. Fails when ellipse minor axis ≲ 10 px (mask cannot reach the depth blob even after offset). |
| `code/src/preprocessing/stickers_analysis/ellipse/gui/ellipse_fit_view_gui.py` | None — overlays ellipses on a single stream of frames the caller passed in. | **n/a** | Single-stream visualisation; no depth involvement. |
| `code/src/preprocessing/common/data_access/kinect_mkv_manager.py` (`KinectFrame`) | `transformed_depth` and `transformed_depth_point_cloud` properties — the two outputs paired with `color` by every downstream consumer. | **mitigated (median shift)** | `KinectFrame.__init__` accepts `parallax_correction: bool = True`; when enabled, both properties apply `apply_parallax_shift()` from `parallax_correction.py` (`(dv, du) = (−1, +10)`, NaN-padded edges) before returning the array. Audit/measurement code must opt out with `parallax_correction=False`. |
| `code/src/preprocessing/common/data_access/kinect_pointcloud_wrapper.py` | `_create_frame_view` (line ~42) — flattens `(H, W, 3)` color and `(H, W, 3)` point cloud as parallel `(N, 3)` vectors, implicit per-pixel pairing. | **mitigated (median shift)** | The wrapper reads `frame_object.transformed_depth_point_cloud`, which now returns the parallax-corrected array when `parallax_correction=True` (the default on `KinectFrame`). No code change required in the wrapper itself; correction is inherited from `KinectFrame`. |
| `code/src/merging/gui/neural_kinect_scene_viewer.py` | Consumes `KinectPointCloudView[idx]` for the colored point cloud rendered by PyVista. | **mitigated** (visual only) | The 10 px texture mismatch on a 1280×720 colored cloud is below visual perception at viewer scale. **However**, any *measurement* derived from the colored cloud (e.g. picking a 3D point by its colour) inherits the offset; if such a path is added, it must be re-classified `unaccounted`. |
| `code/src/preprocessing/common/gui/kinect_rgb_depth_viewer.py` | Both panels indexed independently by user clicks; offset is the *measurement*, not a hidden bias. | **n/a** | This is the audit tool. The offset is observed, not absorbed. |

Modules flagged `unaccounted` get one idea file each under
`docs/development/plans/ideas/` (see References §7).

---

## 7. Reusable pattern

When introducing or reviewing code that reads
`transformed_depth[v, u]` or `point_cloud[v, u, :]` from a pixel
`(u, v)` derived from the **color frame**:

- [ ] **Quantify the local depth gradient first.** A 5×5 patch with
  `np.nanmax - np.nanmin > 20 mm` means the lookup is on a depth
  edge and the parallax-induced jump can be metres, not pixels.
  `KinectRgbDepthViewer`'s `near_edge` column flags this; reuse the
  same 5×5 / 20 mm rule in batch code.
- [ ] **Prefer area sampling over single-pixel lookup.** A mask of
  ≥ ±25 px in u (the worst-case offset) almost always contains the
  intended surface somewhere; combine with a depth-plausibility
  filter (e.g. the sticker-diameter z-guard from
  `xyz_extractor_ellipse_depth.py`) to discard the wrong surface.
- [ ] **Account for the +u bias.** If you must pick a single pixel,
  shift the lookup by −10 px in u as a first-order correction (or
  by `(−10, +1)` for u/v together). This is a heuristic, not a
  calibrated offset — it is the *median* of the band, not a
  per-frame correction.
- [ ] **Validate at the operating range you actually use.** This
  band is characterised at 500–800 mm only. If you operate outside
  that range, re-measure with `KinectRgbDepthViewer` first.
- [ ] **Treat the depth-axis click in the audit tool with caution.**
  If the operator clicks on a NaN-dominated pixel (off-image in the
  depth panel, or inside a depth hole), the viewer falls back in a
  way that produces an absurd `(Δu, Δv)`. The aggregation script's
  `|Δ| > 100 px` filter catches these; the underlying viewer
  failure mode is documented as a known limitation rather than
  fixed, since it is operator-detectable on inspection.

---

## 8. References

| Document | Location |
|----------|----------|
| Parent plan (audit) | `docs/development/plans/completed/kinect-rgb-depth-registration-audit.md` |
| Direct plan (this band) | `docs/development/plans/completed/kinect-rgb-depth-parallax-tolerance-band.md` |
| Measurement CSVs (Phase 1) | `F:/_tmp/kinect-measurement-offsets/phase-1/` |
| Measurement CSVs (Phase 2) | `F:/_tmp/kinect-measurement-offsets/phase-2/` |
| Aggregation plot | `F:/_tmp/kinect-measurement-offsets/aggregate_near_field.png` |
| Aggregation script | `code/scripts/_3_preprocessing/_0_kinect_diagnostics/analyse_rgb_depth_offset_dataset.py` |
| Audit viewer | `code/src/preprocessing/common/gui/kinect_rgb_depth_viewer.py` |
| Audit viewer launcher | `code/scripts/_3_preprocessing/_0_kinect_diagnostics/audit_rgb_depth_registration.py` |
| Upstream — color/depth alignment inaccuracy | [microsoft/Azure-Kinect-Sensor-SDK #1058](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/1058) |
| Upstream — IR↔color transformation imperfections | [microsoft/Azure-Kinect-Sensor-SDK #1201](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/1201) |
| Upstream — third-party verification | [Depthkit: Azure Kinect alignment verification](https://docs.depthkit.tv/docs/azure-kinect-alignment-verification/) |
| Upstream — Microsoft Learn | [Azure Kinect image transformations](https://learn.microsoft.com/en-us/azure/kinect-dk/use-image-transformation) |
| Follow-up idea (centroid extractor) | `docs/development/plans/ideas/xyz-extractor-centroid-parallax-tolerance.md` |
| Follow-up idea (point-cloud wrapper) | `docs/development/plans/ideas/kinect-pointcloud-wrapper-parallax-tolerance.md` |
