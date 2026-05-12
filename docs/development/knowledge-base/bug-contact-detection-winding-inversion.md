# Bug: Contact Detection — False Positives from Negative Procrustes Scale

## Status

**Fixed.** Scale guard added to `_calculate_alignment_procrustes` in
`hand_motion_manager.py` (Phase 2, 2026-05-11). Cross-block fallback warning
added to `get_forearms_with_fallback` in `forearm_catalog.py` (Phase 3,
2026-05-11).

## Root Cause

`HandMotionManager._calculate_alignment_procrustes` computes a scaled-Procrustes
alignment between sticker reference coordinates (P) and tracked sticker
coordinates (Q):

```
scale = Σ(P_rotated · Q) / Σ(P · P)
```

The denominator (sum of squares) is always positive. The numerator can be
negative for degenerate sticker frames — collinear sticker layouts, tracking
noise, or a re-labelling flip produce `P_rotated · Q < 0`, yielding `scale < 0`.

## Effect

In `HandMotionManager.__getitem__`, the rotation block of the 4×4 transform
matrix is multiplied by the scalar scale before applying it to the MANO mesh:

```python
matrix[:3, :3] *= scale   # scale < 0 → det(matrix) < 0 → winding FLIPPED
mesh.transform(matrix)
```

A negative determinant inverts the face winding order of the transformed mesh.
Open3D's `RaycastingScene.compute_signed_distance` determines interior/exterior
via winding-number ray casting, so an inverted mesh swaps "inside" and "outside":
forearm vertices that are geometrically outside the hand receive a negative
signed distance, triggering `inside_mask = all(dist < 1e-5)` everywhere.
This produces false positive `contact_points` at every triangle of the forearm
mesh for every affected frame, corrupting `contact_detected`, `contact_area`,
`contact_depth`, and `contact_location_x/y/z` in the somatosensory CSV.

## Discovery

First observed at frame 502 of session
`2022-06-16_ST15-01_semicontrolled_block-order04_kinect`: the Neural-Kinect
Viewer showed two dense contact-point clusters (one per finger) while the MANO
handmesh was visually detached from the reference forearm PLY.

Phase 1 diagnosis established that the visible symptom in that specific run was
caused by a stale merged CSV that predated a somatosensory reprocess — neither
H1 (negative scale) nor H2 (cross-block forearm fallback) were active for that
session after reprocessing. The NPZ for block-order04 contained no negative-scale
frames after reprocessing. However, the winding-inversion mechanism is real and
would silently corrupt any session whose NPZ does contain negative-scale frames.

## Fix Applied

**H1 — scale guard** (`hand_motion_manager.py`,
`_calculate_alignment_procrustes`): if `scale <= 0`, log a warning and
substitute the previous frame's scale; raise `ValueError` if no previous scale
exists (first frame cannot be degenerate). Additionally warn if
`scale` falls outside `[0.3, 5.0]` (plausible range for hand-size variation).

**H2 — cross-block fallback warning** (`forearm_catalog.py`,
`get_forearms_with_fallback`): emit `logging.warning` when a preceding-block
PLY is used as a fallback for the current block, printing source block and
representative frame ID so the operator can decide whether to capture a new PLY.

## Diagnostic Tool

`code/scripts/__misc/inspect_handmodel_scales.py` — standalone CLI that loads
any `*_handmodel_motion.npz`, prints summary statistics (min, max, mean, std,
total frames), and prints a table of frames where `scale <= 0` or
`abs(scale) > 3.0`.

```
python code/scripts/__misc/inspect_handmodel_scales.py path/to/file_handmodel_motion.npz
```

## Key Files

- `code/src/preprocessing/motion_analysis/hand_tracking/models/hand_motion_manager.py`
  — `_calculate_alignment_procrustes`, `__getitem__`
- `code/src/preprocessing/motion_analysis/tactile_quantification/model/objects_interaction_processor.py`
  — signed-distance contact detection
- `code/src/preprocessing/forearm_extraction/models/forearm_catalog.py`
  — `get_forearms_with_fallback`
- `code/scripts/__misc/inspect_handmodel_scales.py` — diagnostic script
