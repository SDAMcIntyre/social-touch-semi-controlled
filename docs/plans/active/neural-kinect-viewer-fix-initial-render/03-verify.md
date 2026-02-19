# Fix Initial Render — Phase 3: Verify & Clean Up

## Goal

Confirm the fix works across all modes, then remove diagnostic code and
archive the plan.

---

## Step 3.1 — Manual test checklist

Run the viewer and verify each scenario:

```bash
cd /mnt/f/GitHub/social-touch-semi-controlled
python code/scripts/view_merged_neural_kinect.py
```

| # | Scenario | Expected result | Pass? |
|---|---|---|---|
| 1 | Window opens (frame 0 may be empty) | Axes widget + origin sphere visible; no fully blank viewport | |
| 2 | Slider to first frame with data | Point cloud / forearm / stickers appear **without** camera interaction | |
| 3 | Slider back to empty frame, then forward | Data disappears then reappears; no permanent blank state | |
| 4 | Play button | Smooth playback; data appears as soon as non-empty frames are reached | |
| 5 | Zoom / pan / rotate | Camera moves normally; no flicker, no reset | |
| 6 | Toggle visibility checkboxes | Objects appear / disappear correctly | |
| 7 | Close + reopen viewer | No `wglMakeCurrent` errors; second instance renders correctly | |
| 8 | Run without merged CSV (pure 3D mode) | Point cloud + forearm + stickers visible; no neural panel; no crash | |

### Running test 8 (pure 3D mode)

If the entry-point script requires a merged CSV path, either:
- Temporarily edit `view_merged_neural_kinect.py` to pass
  `merged_csv_path=None`, or
- Create a minimal test harness that instantiates `NeuralKinectViewer`
  without it.

---

## Step 3.2 — Check camera position is preserved

After data first appears, read the camera readout in the right panel:

- `pos` should be near `contact_centroid + (0, 0, -400)`
- `up` should be near `(0.375, -0.904, -0.201)`

If the camera has jumped to VTK's default isometric view, the fix
accidentally called `reset_camera()` instead of
`ResetCameraClippingRange()`.

---

## Step 3.3 — Remove diagnostic instrumentation

After all tests pass, remove from `neural_kinect_scene_viewer.py`:

- All `print(f"[DIAG ...")` lines (if any remain from Phase 1 diagnostics)
- All `print(f"[DEBUG ...")` lines (if any were added for bounding proxy)
- The `resizeEvent` debug probe (if added)

Ensure the only additions remaining are:
- `self.plotter.renderer.ResetCameraClippingRange()` in `_update_frame`
- (If Phase 2 applied) The `_bounds_proxy` actor + `_bounds_proxy_active`
  flag + removal logic

---

## Step 3.4 — Clean up investigation doc

Delete the now-resolved investigation document:

```bash
rm docs/bugs/neural-kinect-viewer-initial-render.md
```

---

## Step 3.5 — Archive this plan

Move the plan from `active` to `done`:

```bash
mkdir -p docs/plans/done
mv docs/plans/active/fix-initial-render.md docs/plans/done/
mv docs/plans/active/neural-kinect-viewer-fix-initial-render/ docs/plans/done/
```

---

## Regression risks

| Risk | Mitigation |
|---|---|
| `ResetCameraClippingRange` per frame causes perf regression | Measured: O(10 actors) ≈ sub-microsecond. No risk. |
| Camera position changes unexpectedly | `ResetCameraClippingRange` only changes `clipping_range`, not position/focal/up. Verified in Step 3.2. |
| Bounding proxy visible as wireframe | Uses `opacity=0.0` (or `0.001`). Not visible. |
| `opacity=0.0` excluded from VTK bounds | Fall back to `opacity=0.001`. See Phase 2 troubleshooting. |
| `contact_centroid` is NaN | Phase 2 notes a fallback: `np.array([0.0, 0.0, 500.0])`. |
| Proxy never removed (all frames empty) | Harmless — invisible actor with no performance cost. |
