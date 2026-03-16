# Idea: Green Sticker / Forearm Separation Strategy

**Date:** 2026-02-23
**Status:** Idea

## Summary

At certain hand angles (observed in ST13-02 block 03), the green sticker on the experimenter's
hand blends visually with the participant's forearm, causing the tracker to assign it to the
forearm rather than the hand. A separation strategy is needed to keep the green sticker reliably
associated with the experimenter's hand.

## Rough Approach

Options to investigate:
- **Spatial constraint** — reject green sticker candidates whose position is inconsistent with
  the known experimenter-hand workspace (e.g., too far from other hand stickers or within the
  forearm bounding volume).
- **Temporal continuity** — prefer candidate positions that minimise displacement from the
  previous valid frame, anchoring the tracker to the hand trajectory.
- **Multi-sticker coherence** — use the relative geometry of all hand stickers to validate each
  individual assignment; flag candidates that break expected inter-sticker distances.

## Notes

- Low priority; this is an edge case tied to specific hand orientations.
- The fix should not affect blocks where the green sticker tracks cleanly.
- Related to the broader sticker-tracking failure idea, but the cause here is geometric
  ambiguity rather than a Kinect depth error.
