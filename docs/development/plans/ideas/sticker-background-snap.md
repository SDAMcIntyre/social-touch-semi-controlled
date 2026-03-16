# Idea: Sticker Background-Snap Detection and Recovery

**Date:** 2026-02-23
**Status:** Idea

## Summary

The Kinect occasionally reports sticker depth values that belong to the background surface behind
the hand rather than the hand itself. The 2D tracking succeeds and the frame status passes as
valid, but the retrieved z value is completely wrong. Unlike edge-gradient bias (a per-frame
correction problem), background snaps typically persist across **consecutive frames**, which makes
temporal interpolation from neighbouring frames unreliable as a recovery strategy.

## Root cause

When the Kinect fails to return valid depth for the hand surface at the sticker's pixel location,
`get_xyz_from_point_cloud()` picks up a non-zero depth value from the background behind the hand.
The function has no way to distinguish this from a valid hand depth — the value is non-zero and
within sensor range.

The failure tends to be sustained: the Kinect loses depth for the relevant patch of hand surface
for a stretch of frames, not just a single frame. This means the rolling-median and
frame-interpolation strategies that work for isolated outliers are ineffective here — there are
no nearby valid frames to interpolate from during the failure window.

## Known occurrences

- ST13-02 blocks 09, 11 (blue sticker). Block 11 also affects the green sticker.
- ST13-03 blocks 01, 05, 07 (blue sticker).
- ST13-03 block 01 is the clearest example: "the RGB-D blue sticker position is on the
  background — only one part of the index has an ok depth value."

## Key challenge

Because the failure spans consecutive frames, detection and recovery cannot rely on temporal
neighbours:

- **Rolling median** against recent frames will adapt to the wrong depth if the failure is long
  enough.
- **Linear interpolation** across the gap requires valid frames on both sides and assumes smooth
  motion — unreliable for gaps longer than a few frames.
- **Last-valid carry-forward** freezes the position, which is wrong if the hand is moving.

Detection must therefore work **within a single frame**, or use non-temporal cues.

## Rough Approach — detection strategies to investigate

1. **Depth-spread metric from edge-gradient correction** — if the ellipse-based multi-pixel
   sampling (see `sticker-tracking-failures.md`) is implemented first, the spread metric it
   produces may flag background-snap frames: when most or all pixels within the sticker ellipse
   report background depth, the spread will either be very low (uniformly wrong) or the
   aggregated depth will be far from the other stickers. This needs investigation on the known-bad
   blocks.

2. **Inter-sticker coherence check** — the pairwise 3D distances between hand stickers are
   approximately stable across frames. If a single sticker suddenly reports a position that
   breaks expected inter-sticker distances while the others remain coherent, flag it. This does
   not require temporal context — it compares stickers within the same frame.

3. **Absolute depth range check** — if the sticker reports a z value that places it on or near
   the known background plane (table / wall), flag it. Requires knowing the approximate
   background depth for the recording, which could be estimated from the first few frames or
   from the forearm reference.

4. **Velocity capping** — even though the failure spans multiple frames, the *onset* is a sudden
   jump. A per-sticker velocity threshold could flag the transition frame, and then a
   state-machine approach could mark subsequent frames as suspect until the sticker returns to
   a plausible position relative to the other stickers.

## Recovery — open question

Recovery is harder than detection for this failure mode. Options to explore:

- **NaN the affected stretch** and let downstream code handle the gap. Honest but loses data.
- **Constrained interpolation** using the other stickers' positions as anchors — e.g., infer the
  missing sticker's position from the known rigid-body geometry of the hand stickers. Requires
  a hand model or at least stable inter-sticker distances.
- **Accept the data loss** for blocks where the gap is too long to recover meaningfully, and
  flag it in metadata so downstream analysis can exclude those segments.

## Notes

- Lower priority than edge-gradient bias correction — that fix is more feasible and may provide
  the spread metric needed for detection here.
- Reported as "Issue C" in the technical notes.
- These are fundamentally a Kinect hardware limitation; the goal is damage control, not
  elimination.
- Related ideas: sticker-tracking-failures (edge-gradient bias, tackle first),
  green-sticker-forearm-separation (geometric ambiguity, different root cause),
  hand-mesh-scaling-sticker-placement (needs clean depth data).
