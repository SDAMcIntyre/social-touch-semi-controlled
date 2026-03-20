# Dev Note: Somatosensory metric units and calculations

**Problem class:** Understanding the coordinate system and derived units for
velocity, contact depth, and contact area — the three primary somatosensory
metrics computed from Kinect point-cloud data.

| Field | Value |
|-------|-------|
| Scope | `objects_interaction_processor.py`, `touch_analysis.py`, `sticker_velocity_compass.py` |
| Upstream source | Azure Kinect SDK (`k4a_transformation_depth_image_to_point_cloud`) via pyk4a |
| Capture FPS | 30 Hz |

---

## 1. Coordinate system origin

All 3D geometry in the pipeline originates from the Azure Kinect SDK function
`k4a_transformation_depth_image_to_point_cloud`, accessed in Python via
`pyk4a.PyK4ACapture.transformed_depth_point_cloud`.

The SDK stores **signed 16-bit XYZ values in millimeters** for every pixel.
No unit conversion is applied anywhere in the pipeline — forearm meshes,
sticker positions, and hand meshes all remain in **mm**.

**SDK reference:**
[k4a_transformation_depth_image_to_point_cloud](https://microsoft.github.io/Azure-Kinect-Sensor-SDK/master/group___functions_ga7385eb4beb9d8892e8a88cf4feb3be70.html)
— *"Transforms the depth image into a 3 channel image where each pixel
represents the corresponding 3d coordinates of the point cloud. [...]
three signed 16 bit values [...] in millimeters."*

**Code path:** `pyk4a` → `KinectFrame.transformed_depth_point_cloud`
(`kinect_mkv_manager.py:61`) → consumed unchanged by forearm extraction,
sticker XYZ extraction, and tactile quantification.

---

## 2. Velocity

| Property | Value |
|----------|-------|
| Unit | **mm/s** |

**Calculation** — `kinematics.py: compute_velocity_magnitudes()`:

```python
velocity_magnitudes = compute_velocity_magnitudes(group, fps=fps)
```

Frame-to-frame Euclidean displacement of the blue sticker multiplied by the
capture frame rate (fps, default 30). The result is a physical velocity in
mm/s.

---

## 3. Contact depth

| Property | Value |
|----------|-------|
| Unit | **mm** |

**Calculation** — `objects_interaction_processor.py:118–170`:

1. Build an Open3D `RaycastingScene` from the hand mesh.
2. Compute signed distances from forearm terrain vertices to the hand mesh
   surface (`scene.compute_signed_distance`). Negative distance = penetration.
3. Select terrain triangles where **all 3 vertices** have signed distance
   < epsilon (strict penetration check).
4. `contact_depth = max(|signed_distance|)` across all penetrating vertex
   distances.

The signed-distance function returns values in the same coordinate units as
the input meshes — millimeters.

---

## 4. Contact area

| Property | Value |
|----------|-------|
| Unit | **mm²** |

**Calculation** — `objects_interaction_processor.py:148–160`:

1. **Broad phase:** AABB crop of the forearm terrain mesh using the hand
   mesh's bounding box.
2. **Narrow phase:** Signed-distance check of cropped terrain vertices
   against the hand mesh (Open3D tensor raycasting).
3. **Triangle selection:** A terrain triangle is "in contact" if all 3
   vertices have negative signed distance (< epsilon).
4. **Area accumulation:** Standard cross-product triangle area, summed over
   all penetrating triangles:

```python
cross_prod = np.cross(v1 - v0, v2 - v0)
active_areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
total_contact_area = np.sum(active_areas)
```

Since vertex coordinates are in mm, the cross product yields mm² and the
0.5 scalar preserves that unit.

---

## 5. Known labeling bug

`compute_somatosensory_characteristics.py:150` labels the contact-area
plot axis as `'Area (cm^2)'`. Based on the data pipeline (no unit
conversion from mm anywhere), the correct unit is **mm²**. This is a
labeling error.

---

## 6. Reusable pattern

When adding new metrics derived from Kinect point-cloud geometry:

- **Positions** are in mm (inherited from the SDK).
- **Velocity** is in mm/s. Use `compute_velocity_magnitudes(group, fps=fps)`
  from `kinematics.py` — it handles the fps multiplication internally.
- **Areas** computed from mesh triangles are in mm².
- **Volumes** (if ever needed) would be in mm³.
- Always verify there is no hidden unit conversion by tracing the data path
  from `transformed_depth_point_cloud` to the metric output.

---

## 7. References

- Azure Kinect SDK — [depth_image_to_point_cloud](https://microsoft.github.io/Azure-Kinect-Sensor-SDK/master/group___functions_ga7385eb4beb9d8892e8a88cf4feb3be70.html)
- Azure Kinect SDK — [image transformations guide](https://learn.microsoft.com/en-us/azure/kinect-dk/use-image-transformation)
- `code/src/preprocessing/motion_analysis/tactile_quantification/model/objects_interaction_processor.py`
- `code/src/analysis/touch_analytics/touch_analysis.py`
- `code/src/merging/gui/sticker_velocity_compass.py`
- `code/src/preprocessing/common/data_access/kinect_mkv_manager.py`
- `code/src/preprocessing/forearm_extraction/arm_segmentation.py` (line 71: `dbscan_eps` documented as mm)
