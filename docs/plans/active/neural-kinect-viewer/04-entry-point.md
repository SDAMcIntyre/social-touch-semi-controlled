# NeuralKinectViewer — 04: Entry Point & Verification (Group 7)

## Group 7 🔴 — Standalone Entry Point Script

**File:** `code/scripts/view_merged_neural_kinect.py`
**Dependencies:** Group 5 (NeuralKinectViewer) | **Sequential:** YES

Style mirrors `view_somatosensory_3d_scene.py` exactly: hardcoded paths in `__main__`, with comments.

```python
"""
view_merged_neural_kinect.py
----------------------------
Standalone visualization tool for merged neural + Kinect recordings.

Inputs required:
  - xyz_csv_path       : *_handstickers_xyz_tracked.csv
  - kinect_video_path  : *.mkv  (raw Kinect recording)
  - forearm_pointcloud_dir + forearm_metadata_path
  - rgb_video_path     : filename used as catalog lookup key (e.g. "*.mp4" stem)
  - hand_motion_path   : *_handmodel_motion.npz
  - merged_csv_path    : *_merged_data.csv (set to None for pure-3D mode)

Features:
  - GPU-cropped point cloud (CuPy, ±400mm AABB around contact centroid)
  - Background MKV pre-loading (8-frame ring buffer)
  - Per-sticker velocity compass widgets
  - Live Nerve_freq / contact_depth / contact_area time-series panel
  - Camera centered on contact region
"""
import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication

from preprocessing.common.gui.neural_kinect_scene_viewer import NeuralKinectViewer


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Define paths — edit these for each recording
    # -------------------------------------------------------------------------
    recording_name = "2022-06-15_ST14-01_block-order-07"

    xyz_csv = Path(
        'F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled'
        '/2_processed/kinect/2022-06-15_ST14-01/block-order-07/handstickers'
        '/2022-06-15_ST14-01_semicontrolled_block-order07_kinect_handstickers_xyz_tracked.csv'
    )
    kinect_video_path = Path(
        'F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled'
        '/1_primary/kinect/2022-06-15_ST14-01/block-order-07'
        '/2022-06-15_ST14-01_semicontrolled_block-order07_kinect.mkv'
    )
    forearm_pointcloud_dir = Path(
        'F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled'
        '/2_processed/kinect/2022-06-15_ST14-01/forearm_pointclouds'
    )
    forearm_metadata_path = forearm_pointcloud_dir / '2022-06-15_ST14-01_arm_roi_metadata.json'
    rgb_video_path = Path('2022-06-15_ST14-01_semicontrolled_block-order07_kinect.mp4')
    hand_motion_path = Path(
        'F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled'
        '/2_processed/kinect/2022-06-15_ST14-01/block-order-07/kinematics_analysis'
        '/2022-06-15_ST14-01_semicontrolled_block-order07_kinect_handmodel_motion.npz'
    )

    # Optional: provide merged CSV for neural overlay.
    # Set to None to run in pure-3D mode (no neural panel or compass widgets).
    merged_csv_path = Path(
        'F:/liu-onedrive-nospecial-carac/_Teams/Social touch Kinect MNG/data/semi-controlled'
        '/3_merged/2022-06-15_ST14-01/sessions'
        '/2022-06-15_ST14-01_semicontrolled_block-order-07_merged_data.csv'
    )
    # merged_csv_path = None  # ← uncomment to disable neural overlay

    # -------------------------------------------------------------------------
    # Launch viewer
    # -------------------------------------------------------------------------
    app = QApplication.instance() or QApplication(sys.argv)

    viewer = NeuralKinectViewer(
        xyz_csv_path=xyz_csv,
        kinect_mkv_path=kinect_video_path,
        forearm_pointcloud_dir=forearm_pointcloud_dir,
        forearm_metadata_path=forearm_metadata_path,
        rgb_video_path=rgb_video_path,
        hand_motion_path=hand_motion_path,
        recording_name=recording_name,
        merged_csv_path=merged_csv_path,
        crop_half_size_mm=400.0,
    )
    viewer.show()
    sys.exit(app.exec_())
```

---

## Verification & Testing Checklist

### Smoke Tests (run immediately after implementation)

- [ ] `python code/scripts/view_merged_neural_kinect.py` launches without exception
- [ ] Window title shows `"Neural-Kinect Viewer | 2022-06-15_ST14-01_block-order-07"`
- [ ] Recording name visible as white text in upper-left of 3D viewport
- [ ] Frame slider movement updates the 3D scene
- [ ] Camera does NOT reset on frame change (no `clear()` being called)
- [ ] Compass widgets appear in right panel (one per sticker color)
- [ ] Neural data panel visible at bottom with 3 signal rows
- [ ] Red cursor line in neural panel moves when slider is dragged

### Performance Tests

- [ ] Move slider by 1 frame → frame update completes in < 33ms (watch console for timing)
- [ ] Press Play → playback feels smooth at ~30fps (no obvious jank)
- [ ] Move slider by 100 frames → cache miss, fallback sync load visible but recovers
- [ ] After 5s of Play, buffer is fully preloaded and playback is consistently smooth

### GPU/CuPy Tests

- [ ] With CuPy installed: add `print(f"Points rendered: {pts.shape[0]}")` in `_update_frame()` — should be ~50k-150k, not 800k
- [ ] Without CuPy: rename/uninstall cupy temporarily → viewer still works with CPU fallback → console shows warning

### Neural Overlay Tests (requires merged CSV)

- [ ] `Nerve_freq` plot is visible and non-flat (verify with a recording that has spikes)
- [ ] `contact_depth` and `contact_area` plots show expected shape
- [ ] Cursor position in neural panel correlates visually with sticker position in 3D

### Degradation Tests (no merged CSV)

- [ ] Set `merged_csv_path = None` → viewer launches, no neural panel, no compass widgets
- [ ] All 3D elements (Kinect cloud, forearm, hand, stickers) still visible

### Close/Cleanup Tests

- [ ] Close window → no Python exception in terminal
- [ ] Close window → console shows "Closed '{mkv_path}'" within 3 seconds (preloader joined)
- [ ] Re-open viewer in same Python session (if QApplication reused) → no crash

---

## Agent Integration Suggestions

When implementing this plan, consider:
- **Groups 1 & 2** can be developed and unit-tested in isolation with a small synthetic dataset before integrating with real MKV files
- **Group 3** (NeuralDataPanel) can be tested with a fake `merged_df = pd.DataFrame({'Nerve_freq': np.random.rand(1000), 'contact_depth': np.random.rand(1000), 'contact_area': np.random.rand(1000)})`
- **Group 5** is the largest; tackle `_init_actors()` first, then `_update_frame()` without GPU crop, then add CuPy
- Test the actor-update pattern early: verify that `plotter.camera_set` remains True between frame updates (proves no `clear()` is happening)

---

## Success Criteria

1. **Reactivity**: Frame slider update visually responds within one display refresh (~33ms at 30fps)
2. **No camera reset**: The user's camera angle is preserved across frame changes
3. **Neural overlay**: When merged CSV is provided, all three signals are visible with a moving cursor
4. **Compass**: Each sticker has a compass widget showing direction and Z-color of velocity
5. **Recording identity**: Window title and 3D text show the recording name
6. **Graceful degradation**: Works without CuPy (CPU fallback) and without merged CSV (pure 3D mode)
7. **Clean exit**: Window closes without hang or exception
