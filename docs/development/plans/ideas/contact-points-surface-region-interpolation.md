# Idea: Physically Meaningful Interpolation of contact_points

## Summary

Replace the current forward-fill of `contact_points` with a physically grounded
interpolation that propagates contact regions between consecutive 30 Hz Kinect frames.

## Rough Approach

Each `contact_points` value is a set of 3D vertices sampled from the forearm surface
that define the contact patch at that frame. Two consecutive non-NaN frames at t and
t+1 (30 Hz) represent two snapshots of the same (or adjacent) contact region.

The interpolation would proceed as follows for each pair (t, t+1):

1. **Region matching** — match the contact patch at t to the one at t+1 one-to-one.
   Options: centroid alignment, ICP, or a shape-based correspondence (e.g. Hungarian
   matching on nearest-neighbour pairs). The patches live on the forearm surface so
   geodesic distance may be more appropriate than Euclidean.

2. **Transformation interpolation** — estimate the per-vertex displacement field (or a
   rigid/affine transform of the patch) from t to t+1. Interpolate that transformation
   linearly over the ~33 missing 1 kHz frames to produce intermediate patch geometries.

3. **Surface re-projection** — for each intermediate frame, project the estimated patch
   geometry back onto the forearm mesh. Extract the actual forearm surface points
   (vertices or sampled XYZ) that fall within the projected region. These become the
   `contact_points` for that frame.

## Notes

- The forearm surface mesh is available via the postprocessing infrastructure
  (`project_contacts_onto_forearm.py`, KD-tree projection).
- The current forward-fill (Stage 1 `interpolation.py`, `BINARY_FFILL`) is the
  placeholder until this is implemented.
- The interpolated `contact_location_x/y/z` (centroid, already cubic-interpolated)
  can serve as a sanity check: the centroid of the re-projected region should track it.
- Edge cases: patch appears/disappears between frames (contact onset/offset) —
  interpolation should only apply to frame pairs where both endpoints are non-empty.
