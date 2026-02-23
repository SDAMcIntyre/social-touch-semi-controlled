# Idea: Hand Mesh Scaling and Sticker Placement Calibration

**Date:** 2026-02-23
**Status:** Idea

## Summary

For participant ST13-03, the hand mesh is too large relative to the actual hand: 4 real digits fit
within 3 mesh digits and the thumb. Additionally, green and yellow sticker spheres are not aligned
with their expected positions on the hand model. Both issues corrupt contact-point localisation.

## Rough Approach

Two sub-problems to address:
1. **Mesh scaling** — expose or infer a per-participant scale factor so the hand mesh matches the
   actual hand dimensions. Could use anatomical landmarks (fingertip-to-wrist distance) recorded
   at setup time.
2. **Sticker alignment** — improve the registration between sticker world positions and hand-model
   vertices, possibly by averaging several reference frames to reduce Kinect Z noise, then
   minimising reprojection error per sticker.

## Notes

- Explicitly flagged as "GLOBAL PROBLEM FOR 13-03" in the technical notes.
- The +z noise from the Kinect Depth sensor means a single-frame registration will be unreliable;
  multi-frame averaging is recommended.
- Smoothing sticker positions prior to somatosensory evaluation was also suggested as a related
  measure.
- Closely related to the sticker-tracking failure idea — bad depth frames must be excluded before
  averaging.
