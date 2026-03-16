# Plan: Hand-Side Contact Detection for Deep Penetration

**Created:** 2026-03-11 00:00
**Approved:** —
**Completed:** —
**Author:** Basil Duvernoy
**Status:** Draft — PENDING FURTHER REVIEW (algorithm not finalized)
**Branch:** `feature/hand-side-contact-detection`

---

## Overview

**What:** Add a second contact detection pass that measures which parts of the hand mesh are below the forearm surface, complementing the existing forearm-side detection.
**Why:** The current algorithm severely underestimates contact area when a finger deeply penetrates the rigid forearm mesh — exactly the case where real contact area is largest.
**How:** Use Open3D's `RaycastingScene.compute_closest_points()` to project hand vertices onto the forearm surface and classify them as above/below using the face normal dot product.

## Problem Statement

The current `ObjectsInteractionProcessor` detects contact by finding forearm mesh triangles that are "inside" the hand mesh (signed distance < 0). This works for shallow contact but fails for deep penetration:

- The forearm is a 2.5D terrain surface (non-watertight), not a volume
- When a finger pushes deep, it passes through the rigid forearm mesh
- Only forearm triangles at the "entry ring" (where the finger crosses the surface) are detected
- The submerged finger shaft and tip generate **zero contact points** — no forearm geometry exists there
- Result: deeper pressing → less detected contact area (inverted from physical reality)

In reality, the forearm skin deforms to conform around the pressing finger. The physical contact patch is approximately the hand surface area that lies below the undeformed forearm surface.

## Goals

### In Scope
1. Detect hand mesh vertices/triangles that are below the forearm surface
2. Compute contact area from the submerged hand surface (the physical contact patch)
3. Compute penetration depth as distance below the forearm surface
4. Preserve existing forearm-side metrics as legacy columns for validation
5. Maintain backward compatibility with all downstream consumers

### Out of Scope
- Soft-tissue deformation simulation (non-rigid forearm)
- Pressure or force estimation from penetration depth
- GPU acceleration of the new detection (CPU Open3D is sufficient for 778 vertices)
- Changes to the visualization GUI beyond consuming the new contact_points

## Success Criteria

- [ ] Deep-penetration frames produce larger `contact_area` than shallow-touch frames
- [ ] `contact_area` is monotonically related to penetration depth (deeper → larger)
- [ ] Shallow/grazing contact still produces small, reasonable area values
- [ ] Existing downstream scripts (`touch_analysis.py`, `merge_neural_and_kinect_data.py`) work without modification
- [ ] Both new (hand-side) and legacy (forearm-side) metrics appear in output CSV
- [ ] Visual inspection confirms contact_points cover the submerged finger surface, not just the entry ring

---

## Technical Design

### Approach — UNDER REVIEW

> **Status**: The algorithm below is the current best candidate but has a known
> limitation on the lateral drop-off of the forearm mesh. Do not implement
> without further review and sign-off.

**Core idea**: Build a `RaycastingScene` from the forearm mesh (cached per reference update). For each frame, cast rays from hand vertices in the forearm's mean outward direction. If a ray hits the forearm surface, the hand vertex is below (submerged).

```python
# Mean outward direction of the forearm surface
mean_normal = np.mean(forearm_vertex_normals, axis=0)
mean_normal /= np.linalg.norm(mean_normal)

# Cast rays from all hand vertices in outward direction
ray_dirs = np.tile(mean_normal, (N, 1))
rays = np.hstack([hand_verts, ray_dirs]).astype(np.float32)
result = self._ref_raycasting_scene.cast_rays(o3d.core.Tensor(rays))

t_hit = result['t_hit'].numpy()   # inf if no intersection
below_mask = np.isfinite(t_hit)   # hit → forearm above → submerged
penetration_depth = t_hit         # distance to surface along ray
```

Hand triangles where all 3 vertices are submerged contribute to the contact patch. Their summed area is the contact area; the maximum penetration_depth is the contact depth.

### Why ray casting, not closest-point

**Rejected: `compute_closest_points()` + face normal dot product.**
The forearm mesh is a 2.5D Delaunay surface of a roughly cylindrical forearm. It includes steep lateral drop-offs where the surface curves down sharply. For deep penetration, the globally closest forearm surface point can be on the lateral side rather than the top entry surface. The side face normal points horizontally, causing the dot-product above/below test to give incorrect classifications.

Ray casting avoids this: a ray from the hand vertex in the outward direction checks whether the forearm surface is *above* the point (between the point and the camera), regardless of which surface region is geometrically nearest.

### Known limitation — single mean normal direction

Using the forearm's mean vertex normal as a single ray direction for all hand vertices is approximate. It works well for the top surface and moderate slopes, but may misclassify hand vertices near extreme lateral edges where the surface is nearly perpendicular to the mean normal. Practical impact is low (contact occurs on the visible top surface), but this warrants further analysis.

**Open questions for review:**
1. Should the ray direction be per-vertex (local forearm normal at nearest point) or global (mean normal)? Per-vertex reintroduces the closest-point lateral problem for determining the ray direction.
2. Is `t_hit` (distance along mean normal) a good proxy for penetration depth, or should depth be measured differently (e.g., perpendicular to the local surface at the hit point)?
3. How to handle hand vertices that are past the lateral extent of the forearm surface in XY — no ray hit, but physically still near/in-contact with the forearm edge?

### Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| `cast_rays()` along mean forearm normal | Robust to lateral drop-off; simple; works on non-watertight meshes | Single ray direction is approximate; depth measurement is along ray, not perpendicular to local surface | **Current candidate** |
| `compute_closest_points()` + face normal dot product | Surface-accurate closest point; normals from API | **Fails on lateral drop-off**: closest point shifts to side face for deep penetration, wrong normal used | Rejected |
| KDTree nearest-vertex + vertex normal dot product | Simple; no new API | Same lateral problem as closest-point; also `crop()` may lose normals | Rejected |
| `compute_signed_distance()` on forearm mesh | Single API call | **Requires watertight mesh** — forearm is non-watertight | Rejected |
| Per-vertex ray direction (local normal at nearest point) | Handles curved surfaces better | Reintroduces closest-point problem for finding the local normal direction | Needs investigation |
| Virtual deformation simulation | Physically most accurate | Enormous complexity; out of scope | Rejected |

### Architecture Changes

No new modules or classes. Changes are contained within `ObjectsInteractionProcessor`:
- New cached attributes: `self._ref_raycasting_scene`, `self._ref_mean_outward_normal` (rebuilt on `update_reference()`)
- New private method: `_detect_hand_side_contact()`
- Modified output dict: primary keys use hand-side values; legacy keys get `_forearm_side` suffix
- `empty_structure()` updated with new legacy columns

### Knowledge Base Constraints

From `note-somatosensory-units-and-calculations.md`:
- All coordinates in **mm** — no unit conversions
- Triangle contact uses "all 3 vertices" strict rule
- Cross-product area accumulation pattern to reuse

From `note-forearm-icp-registration.md`:
- Contact computation operates in the registered coordinate frame (post-ICP)
- No additional frame transformations needed

---

## Implementation Plan

### Phase 1: Infrastructure
**Goal:** Add the forearm RaycastingScene cache to `ObjectsInteractionProcessor`
**Started:** —
**Completed:** —

- [ ] In `update_reference()`: build `RaycastingScene` from forearm mesh, store as `self._ref_raycasting_scene`
- [ ] In `update_reference()`: compute and cache `self._ref_mean_outward_normal` from forearm vertex normals
- [ ] Add `EPSILON` as a class-level constant (currently hardcoded in method body)

**Files Modified:**
- `code/src/preprocessing/motion_analysis/tactile_quantification/model/objects_interaction_processor.py` — add cached scene + mean normal

**Dependencies:** None

### Phase 2: Hand-Side Detection Method
**Goal:** Implement the core `_detect_hand_side_contact()` algorithm
**Started:** —
**Completed:** —

> **NOTE**: This pseudocode reflects the ray-casting approach which is still
> under review. Do not implement without sign-off on the open questions in
> the Technical Design section.

- [ ] Extract hand vertices and triangles as numpy arrays
- [ ] Cast rays from all hand vertices in the mean outward normal direction
- [ ] Classify: `t_hit < inf` → submerged (forearm surface is above)
- [ ] Select hand triangles where all 3 vertices are submerged
- [ ] Compute: area (cross product sum), depth (max t_hit or TBD), location (mean centroid), contact_points (unique submerged vertices)
- [ ] Return metrics dict

**Pseudocode:**
```python
def _detect_hand_side_contact(self, input_mesh):
    hand_verts = np.asarray(input_mesh.vertices).astype(np.float32)
    hand_tris = np.asarray(input_mesh.triangles)

    # Cast rays from each hand vertex in the forearm outward direction
    ray_dirs = np.tile(self._ref_mean_outward_normal, (len(hand_verts), 1))
    rays = np.hstack([hand_verts, ray_dirs]).astype(np.float32)

    result = self._ref_raycasting_scene.cast_rays(o3d.core.Tensor(rays))
    t_hit = result['t_hit'].numpy()  # inf = no hit

    # Hit → forearm surface is above → hand vertex is submerged
    below_mask = np.isfinite(t_hit)

    tri_below = below_mask[hand_tris]
    submerged_mask = np.all(tri_below, axis=1)

    if not np.any(submerged_mask):
        return None  # no hand-side contact

    active_tris = hand_tris[submerged_mask]
    v0 = hand_verts[active_tris[:, 0]]
    v1 = hand_verts[active_tris[:, 1]]
    v2 = hand_verts[active_tris[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    # Depth: t_hit measures distance along mean normal to surface
    # TODO: review whether this is the right depth metric
    tri_depths = t_hit[active_tris]
    contact_depth = float(np.max(tri_depths[np.isfinite(tri_depths)]))
    contact_area = float(np.sum(areas))

    centroids = (v0 + v1 + v2) / 3.0
    mean_loc = np.mean(centroids, axis=0)

    unique_idx = np.unique(active_tris)
    contact_points = hand_verts[unique_idx]

    return {
        "contact_area": contact_area,
        "contact_depth": contact_depth,
        "contact_location_x": float(mean_loc[0]),
        "contact_location_y": float(mean_loc[1]),
        "contact_location_z": float(mean_loc[2]),
        "contact_points": contact_points.tolist(),
    }
```

**Files Modified:**
- `code/src/preprocessing/motion_analysis/tactile_quantification/model/objects_interaction_processor.py` — add `_detect_hand_side_contact()`

**Dependencies:** Phase 1

### Phase 3: Integration
**Goal:** Wire hand-side detection into `_calculate_intersection_volume()` and update output schema
**Started:** —
**Completed:** —

- [ ] Call `_detect_hand_side_contact(input_mesh)` after existing forearm-side detection
- [ ] Restructure output dict: hand-side → primary keys; forearm-side → `*_forearm_side` suffix
- [ ] `contact_detected = 1` if either side detects contact
- [ ] Update `empty_structure()` with legacy columns (default 0.0)
- [ ] Update `contact_info` visualization dict to provide hand-side contact points

**Files Modified:**
- `code/src/preprocessing/motion_analysis/tactile_quantification/model/objects_interaction_processor.py` — modify `_calculate_intersection_volume()` and `empty_structure()`

**Dependencies:** Phase 2

### Phase 4: Downstream Verification
**Goal:** Confirm all downstream consumers work with the new output schema
**Started:** —
**Completed:** —

- [ ] Verify `ObjectsInteractionController` column reordering handles new columns
- [ ] Verify `touch_analysis.py` reads `contact_area`/`contact_depth` correctly
- [ ] Verify `merge_neural_and_kinect_data.py` reads `contact_area` correctly
- [ ] Verify `csv_spatial_transformer.py` transforms `contact_location_x/y/z` and `contact_points`
- [ ] Verify `neural_kinect_scene_viewer.py` parses `contact_points` from CSV

**Files to check (read-only):**
- `code/src/preprocessing/motion_analysis/tactile_quantification/core/objects_interaction_controller.py`
- `code/src/analysis/touch_analytics/touch_analysis.py`
- `code/scripts/_4_merging/merge_neural_and_kinect_data.py`
- `code/src/preprocessing/forearm_extraction/registration/csv_spatial_transformer.py`
- `code/src/merging/gui/neural_kinect_scene_viewer.py`

**Dependencies:** Phase 3

### Phase 5: Documentation
**Goal:** Document the algorithm change
**Started:** —
**Completed:** —

- [ ] Update `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md` — add hand-side contact detection description, note legacy columns
- [ ] Create `docs/development/knowledge-base/note-hand-side-contact-detection.md` — full problem/solution write-up

**Files Modified:**
- `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
- `docs/development/knowledge-base/note-hand-side-contact-detection.md` (new)
- `docs/development/knowledge-base/README.md` — add index entry

**Dependencies:** Phase 3

---

## Testing Plan

### Manual Verification
- [ ] Run somatosensory quantification on a session with known deep-penetration frames
- [ ] Compare `contact_area` (new) vs `contact_area_forearm_side` (old) — new should be larger for deep penetration
- [ ] Verify shallow contact produces similar values for both metrics
- [ ] Visually inspect contact_points in `ObjectsInteractionVisualizer` — should cover submerged finger surface, not just entry ring
- [ ] Run full pipeline through `merge_neural_and_kinect_data.py` and `touch_analysis.py` — no errors

### Edge Cases
- [ ] No contact (hand in free space) — both sides return empty, `contact_detected = 0`
- [ ] Shallow grazing contact — small area from both sides
- [ ] Deep single-finger penetration — key improvement case: large hand-side area
- [ ] Multiple fingers in contact — each contributes independently
- [ ] Hand near forearm edge — `MAX_FOREARM_PROXIMITY_MM` prevents false matches
- [ ] Reference mesh change mid-sequence — `RaycastingScene` rebuilt correctly

---

## Documentation Plan

- [ ] Update knowledge base note on somatosensory units/calculations
- [ ] Create new knowledge base note for hand-side contact detection algorithm
- [ ] Update knowledge base README index

---

## Rollback Plan

1. The existing forearm-side detection is preserved unchanged in the code
2. Legacy columns (`contact_area_forearm_side`, `contact_depth_forearm_side`) provide the old values
3. To revert primary metrics to old behavior: swap the dict key assignment (forearm-side → primary, hand-side → suffixed)
4. No database migrations or external state changes

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Forearm face normals inconsistently oriented (inverted winding) | Low | High — flips above/below classification | Forearm OBJs use `trimesh.fix_normals()`; add warning log if mean normal Z < 0 |
| `compute_closest_points()` matches hand vertex to wrong forearm region near mesh edges | Low | Medium — false contact at boundary | `MAX_FOREARM_PROXIMITY_MM` threshold filters distant matches |
| Downstream consumer breaks on new legacy columns | Low | Medium — extra columns in CSV | All downstream code accesses specific column names, not positional; extra columns are harmless |
| Performance regression from per-frame raycasting queries | Low | Low — 778 queries is negligible | `RaycastingScene` cached per reference; only 778 hand vertices queried per frame |

---

## References

- Open3D 0.19.0 `RaycastingScene` API: https://www.open3d.org/docs/release/python_api/open3d.t.geometry.RaycastingScene.html
- Knowledge base: `docs/development/knowledge-base/note-somatosensory-units-and-calculations.md`
- Knowledge base: `docs/development/knowledge-base/note-forearm-icp-registration.md`
- Core file: `code/src/preprocessing/motion_analysis/tactile_quantification/model/objects_interaction_processor.py`
