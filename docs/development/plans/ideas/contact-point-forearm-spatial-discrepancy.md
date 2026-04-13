# Idea: Contact Point / Forearm Pointcloud Spatial Discrepancy

**Date:** 2026-04-13
**Status:** Idea

## Summary

When zooming into the RF heatmap, spike contact points are spatially shifted from
the nearest forearm pointcloud (PLY) vertex — close but not coincident. This
discrepancy needs investigation to understand why contact positions extracted from
the aggregated CSV do not land exactly on the forearm surface mesh.

## Rough Approach

- Trace the contact point coordinates end-to-end: from raw Kinect capture, through
  forward-fill interpolation, CSV aggregation, and into the visualizer
- Trace the forearm PLY vertices: from raw depth frames, through PCA calibration,
  to the final PLY written to disk
- Identify where the two coordinate spaces diverge — possible causes:
  - Forward-fill of contact positions across frames where the forearm moves
  - PCA calibration applied to the PLY but not to contact points (or vice versa)
  - Temporal misalignment between the frame that produced the contact point and
    the frame used for the PLY reconstruction
  - Floating-point quantisation during CSV round-tripping

## Notes

- Discovered while testing the RF heatmap restyle (`feature/rf-heatmap-visualization-restyle`)
- The exact-match forearm point filtering approach does not work because of this
  shift — a radius-based approach (KDTree, ~2mm) is needed as a workaround until
  the root cause is resolved
- Related: `docs/development/plans/ideas/rgb-d-frame-lag-compensation.md` may
  describe a related temporal offset issue
