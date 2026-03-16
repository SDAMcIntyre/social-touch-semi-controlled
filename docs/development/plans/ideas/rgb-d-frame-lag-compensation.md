# Idea: RGB-D Frame Lag Compensation

**Date:** 2026-02-23
**Status:** Idea

## Summary

The Kinect RGB-D stream appears to lag behind by approximately 1 frame relative to the sticker
tracking data. This temporal misalignment affects depth and contact point accuracy across multiple
blocks in ST13-02 and ST13-03 sessions.

## Rough Approach

Identify the lag during data loading or pre-processing and shift the RGB-D frame index by the
observed offset before pairing it with the corresponding sticker data. Could be a fixed 1-frame
offset or estimated dynamically from a cross-correlation of a shared signal (e.g., a visible
contact event).

## Notes

- Reported as "Issue B" in the technical notes; present in ST13-02 blocks 08–11 and ST13-03
  blocks 06–07.
- The GUI annotation already notes "Kinect data seems to lag behind by 1 frame".
- Must verify whether the offset is constant across sessions and participants before applying a
  global constant correction.
- Related to Issue A (ref arm +z shift) but independent; both can co-exist in the same block.
