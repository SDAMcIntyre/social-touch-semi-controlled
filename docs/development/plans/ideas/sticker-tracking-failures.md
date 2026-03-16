# Idea: Sticker Depth Edge-Gradient Bias Correction

**Date:** 2026-02-23
**Status:** Idea

## Summary

Stickers sit near the edges of the hand (fingertips, knuckles). The Kinect depth sensor produces
a gradual depth gradient at object boundaries — pixels near sticker edges blend the hand's true
depth with the background depth behind it, pulling the reported position away from the hand
surface. The current extractor samples a single centre pixel, making it vulnerable to this bias.
Replacing single-pixel sampling with ellipse-based multi-pixel sampling and a
homogeneity-adaptive aggregation strategy would improve depth accuracy on every frame, not just
failure cases.

## Root cause

`get_xyz_from_point_cloud()` in `xyz_extractor_centroid.py` samples a **single pixel** from the
transformed depth point cloud at the sticker's detected 2D centre. Its only guard is rejecting
all-zero coordinates (sensor dropout). It has no awareness of whether that pixel sits on a clean
surface or on a depth edge ramp.

The Kinect depth sensor does not produce sharp depth discontinuities at object boundaries.
Instead, it creates a smooth gradient from the true surface depth to the background depth —
like a camera looking down at a cube on a table, where the cube's edges show a ramp rather than
a step. On the hand, this means sticker-edge pixels blend the hand surface with whatever is
behind it, systematically biasing the reported z away from the camera.

The existing quality gate (`score` threshold at 0.3, ellipse-fit quality) does not help because
the 2D ellipse fits correctly — the problem only surfaces at the 3D lift step. Likewise,
`should_process_row()` only rejects `"Failed"`, `"Black Frame"`, and `"Ignored"` status rows;
edge-biased frames pass through as valid.

ST13-03 block 07 notes "position of blue stickers which can lack of accuracy" — this is likely
the edge-gradient effect.

## Rough Approach

### Ellipse-based multi-pixel depth sampling

Sample **all depth points** within the sticker's detected ellipse rather than a single centre
pixel. The ellipse from the 2D tracker already defines the sticker's spatial extent — use it as
a mask over the point cloud.

### Homogeneity-based adaptive aggregation

Extract all non-zero z values within the ellipse mask, then assess their spread:

1. **Homogeneous case** (low spread, e.g., std < threshold) — the sticker sits cleanly on the
   hand surface with no edge contamination. The depth values agree, so any robust central
   estimate (median) is reliable. This also serves as a positive confidence signal that the
   tracking succeeded for this frame.

2. **Spread-out case** (high spread) — the ellipse straddles an object edge and includes a mix
   of hand-surface depths and edge-gradient / background depths. Since the gradient always
   biases **away from the camera**, the correct hand-surface depth is the **shallowest**
   (nearest-to-camera) cluster. Strategy: select only the upper z values — e.g., take the lower
   percentile (10th–25th) of the z distribution, or fit a bimodal split and keep the near
   cluster.

The spread itself is a per-frame quality signal: frames with high spread can be flagged even if
the selected depth value seems plausible.

### Why the ellipse, not the ROI

The ellipse is the right region to sample because it tightly follows the sticker's 2D shape —
unlike the ROI bounding box, which includes background pixels by definition. Using the fitted
ellipse as a mask confines depth sampling to pixels that the 2D tracker already identified as
belonging to the sticker.

## Where to hook in

This is a change to the extractor itself. `get_xyz_from_point_cloud()` currently takes a single
`(px, py)` and samples one pixel. It would need to accept the ellipse parameters (centre, axes,
angle) or a pre-computed pixel mask, sample all points within it, and return an aggregated
position plus a spread metric.

The ellipse parameters are available in the tracked data upstream (from the ellipse fitting
step) and flow through `tracked_obj_row` into the extractor.

## Notes

- This is the more feasible of the two sticker depth issues — it works within a single frame
  and does not depend on temporal context or neighbouring frames.
- Should be tackled before the background-snap problem (see `sticker-background-snap.md`),
  since the spread metric produced here may also help detect background snaps.
- Related ideas: sticker-background-snap (separate failure mode, consecutive-frame problem),
  green-sticker-forearm-separation (geometric ambiguity, different root cause),
  hand-mesh-scaling-sticker-placement (needs clean depth data, benefits from this fix).
- Smoothing sticker positions (Savitzky-Golay) is a downstream concern — should be applied
  after both edge-bias correction and any background-snap handling.
