# NeuralKinectViewer — 03: Integration (Groups 5 & 6)

## Group 5 🔴 — NeuralKinectViewer Main Class

**File:** `code/src/preprocessing/common/gui/neural_kinect_scene_viewer.py`
**Dependencies:** Groups 1, 2, 3, 4 | **Sequential:** YES

### Full Constructor Signature

```python
class NeuralKinectViewer(QMainWindow):
    def __init__(
        self,
        xyz_csv_path: Path,
        kinect_mkv_path: Path,
        forearm_pointcloud_dir: Path,
        forearm_metadata_path: Path,
        rgb_video_path: Path,          # just the filename (str or Path) for catalog lookup
        hand_motion_path: Path,
        recording_name: str,
        merged_csv_path: Optional[Path] = None,
        crop_half_size_mm: float = 400.0,
        parent=None
    ):
```

### Constructor Body — Step by Step

```
1. super().__init__(parent)
2. self.setWindowTitle(f"Neural-Kinect Viewer | {recording_name}")
3. self._recording_name = recording_name
4. self._crop_half_size = crop_half_size_mm
5. self._cupy_available = _detect_cupy()  # module-level function
6. Load stickers:
     stickers_df_dict = XYZDataFileHandler.load(xyz_csv_path)
     self._custom_colors = define_custom_colors(stickers_df_dict.keys())
     self._stickers_xyz_dict = {
         k: df[['x_mm', 'y_mm', 'z_mm']].to_numpy()
         for k, df in stickers_df_dict.items()
     }
7. Load forearm:
     forearm_params = ForearmFrameParametersFileHandler.load(forearm_metadata_path)
     catalog = ForearmCatalog(forearm_params, forearm_pointcloud_dir)
     self._forearms_dict = get_forearms_with_fallback(catalog, rgb_video_path)
     self._sorted_forearm_keys = sorted(self._forearms_dict.keys())
8. Load hand motion (LAZY — store manager, NOT eager loop):
     self._hand_manager = HandMotionManager()
     self._hand_manager.load(str(hand_motion_path))
     self._last_hand_frame = -1
     self._last_hand_mesh = None
9. Load merged_df if path provided:
     self.merged_df = pd.read_csv(merged_csv_path) if merged_csv_path else None
10. Compute contact_centroid:
     self.contact_centroid = self._compute_contact_centroid()
11. Open MKV (manual context management):
     self._mkv = KinectMKV(kinect_mkv_path)
     self._mkv.__enter__()
     self._point_cloud_view = KinectPointCloudView(self._mkv)
     self._total_frames = len(self._point_cloud_view)
12. Launch preloader:
     self._preloader = FramePreloader(self._point_cloud_view, buffer_size=8)
     self._preloader.seek(0)
     self._preloader.start()
13. Build UI: self._build_ui()
14. Init actors: self._init_actors()
15. Init play timer: self._play_timer = QTimer(); self._play_timer.timeout.connect(self._play_advance)
16. Neural scale factor: self._neural_scale = len(self.merged_df) / self._total_frames
    if self.merged_df is not None else 1.0
```

### `_compute_contact_centroid()` Method

```python
def _compute_contact_centroid(self) -> np.ndarray:
    if self.merged_df is not None and 'contact_detected' in self.merged_df.columns:
        contact_rows = self.merged_df[self.merged_df['contact_detected'] == 1]
        sticker_positions = []
        for name in self._stickers_xyz_dict:
            for axis in ['x_mm', 'y_mm', 'z_mm']:
                col = f"{name}_{axis}"
                if col in contact_rows.columns:
                    sticker_positions.append(contact_rows[col].values)
        if sticker_positions:
            all_pos = np.stack(sticker_positions, axis=1)
            centroid = np.nanmean(all_pos, axis=0)
            if centroid.shape == (3,) and not np.any(np.isnan(centroid)):
                return centroid
    # Fallback: global mean of all sticker positions
    all_xyz = np.concatenate([pos for pos in self._stickers_xyz_dict.values()], axis=0)
    return np.nanmean(all_xyz, axis=0)
```

### `_build_ui()` Method

```python
central = QWidget(); self.setCentralWidget(central)
outer = QVBoxLayout(central)

# Top: 3D + right panel
top_widget = QWidget(); top_layout = QHBoxLayout(top_widget)
plotter_widget = QWidget(); plotter_layout = QVBoxLayout(plotter_widget)
self.plotter = QtInteractor(plotter_widget)
self.plotter.set_background('midnightblue')
plotter_layout.addWidget(self.plotter.interactor)
top_layout.addWidget(plotter_widget, stretch=4)

# Right panel (scroll area, fixed width 220px)
self._right_panel = QWidget(); self._right_panel_layout = QVBoxLayout(self._right_panel)
self._build_right_panel_controls()
scroll = QScrollArea(); scroll.setWidget(self._right_panel)
scroll.setWidgetResizable(True); scroll.setFixedWidth(220)
top_layout.addWidget(scroll, stretch=1)
outer.addWidget(top_widget, stretch=4)

# Middle: frame controls
outer.addWidget(self._build_frame_controls())

# Bottom: neural panel (only if merged_df)
self.neural_panel = None
if self.merged_df is not None:
    self.neural_panel = NeuralDataPanel(self.merged_df, self._total_frames)
    outer.addWidget(self.neural_panel)
```

### `_build_right_panel_controls()` Method

For each scene object (kinect, forearms, hand_meshes, each sticker):
- `QGroupBox(label)` containing `QCheckBox("Visible")` connected to `self._visibility[name]`
- For point-cloud objects: `QSlider` for point size connected to `self._point_sizes[name]`

For each sticker (when `merged_df` is available):
- `StickerVelocityCompass(name, color)` stored in `self._compass_widgets: Dict[str, StickerVelocityCompass]`
- Wrapped in a labeled `QGroupBox`, added to the right panel

Init dicts:
```python
self._visibility = {name: True for name in ['kinect_point_cloud', 'forearms',
                                              'hand_meshes'] + list(self._stickers_xyz_dict.keys())}
self._point_sizes = {'kinect_point_cloud': 2.0, 'forearms': 8.0}
self._compass_widgets: Dict[str, StickerVelocityCompass] = {}
```

### `_build_frame_controls()` Method

Returns `QWidget` with `QHBoxLayout`:
- `QLabel("Frame:")`, `QSlider(Qt.Horizontal)` → `self.frame_slider` (connected to `_on_slider_change`)
- `QLabel(f"N / {self._total_frames}")` → `self.frame_label` (fixed width 100px)
- `QPushButton("Recenter")` → `self._recenter_view()`
- `QPushButton("▶ Play")` → `self._toggle_play()` (toggles `self._play_timer`)
- `QLabel("Crop ±")`, `QSpinBox(100, 2000, step=50, default=400)` → `self.crop_spinbox` (connected to `self._on_crop_changed`)

### `_init_actors()` Method

```python
def _init_actors(self):
    # Static (created once, never updated)
    self.plotter.add_text(self._recording_name, position='upper_left',
                          font_size=10, color='white', name='recording_label')
    self.plotter.add_axes(interactive=False, line_width=150, box=True)
    self.plotter.add_mesh(pv.Sphere(radius=2.0), color='yellow',
                          name='origin_sphere', pickable=False)

    # Dynamic placeholder actors (empty mesh = named slot exists)
    empty = pv.PolyData(np.empty((0, 3), dtype=np.float32))
    self.plotter.add_mesh(empty, name='kinect_point_cloud',
                          render_points_as_spheres=False, point_size=2.0)
    self.plotter.add_mesh(empty, name='forearms',
                          render_points_as_spheres=True, point_size=8.0)
    self.plotter.add_mesh(empty, name='hand_meshes', style='wireframe')
    for name, color in self._custom_colors.items():
        sphere = pv.Sphere(radius=4.0, center=(0, 0, 0))
        self.plotter.add_mesh(sphere, color=color, name=f'sticker_{name}')

    # Camera centered on contact
    self.plotter.camera.focal_point = self.contact_centroid.tolist()
    self.plotter.camera.position = [
        self.contact_centroid[0],
        self.contact_centroid[1] - 800,
        self.contact_centroid[2] + 400
    ]
    self.plotter.reset_camera()
    self.plotter.camera_set = True
```

### `_update_frame(frame_idx: int)` Method — The Hot Path

```python
def _update_frame(self, frame_idx: int):
    self.current_index = frame_idx

    # 1. Kinect point cloud (GPU-cropped)
    pc_data = self._preloader.get_frame(frame_idx)
    if pc_data and pc_data.points is not None and pc_data.points.shape[0] > 0:
        pts, cols = self._crop_pointcloud_gpu(pc_data.points, pc_data.color,
                                              self.contact_centroid, self._crop_half_size)
        if pts.shape[0] > 0 and self._visibility['kinect_point_cloud']:
            cloud = pv.PolyData(pts)
            cloud['colors'] = cols.astype(np.uint8)
            self.plotter.add_mesh(cloud, scalars='colors', rgb=True,
                                  name='kinect_point_cloud',
                                  render_points_as_spheres=False,
                                  point_size=self._point_sizes['kinect_point_cloud'])

    # 2. Forearm (only update on key change)
    forearm_key = self._bisect_forearm(frame_idx)
    if forearm_key != getattr(self, '_last_forearm_key', None):
        self._last_forearm_key = forearm_key
        o3d_pc = self._forearms_dict.get(forearm_key)
        if o3d_pc and o3d_pc.has_points() and self._visibility['forearms']:
            pts = np.asarray(o3d_pc.points)
            cloud = pv.PolyData(pts)
            if o3d_pc.has_colors():
                cols = (np.asarray(o3d_pc.colors) * 255).astype(np.uint8)
                cloud['colors'] = cols
                self.plotter.add_mesh(cloud, scalars='colors', rgb=True,
                                      name='forearms', render_points_as_spheres=True,
                                      point_size=self._point_sizes['forearms'])
            else:
                self.plotter.add_mesh(cloud, color='gray', name='forearms',
                                      render_points_as_spheres=True,
                                      point_size=self._point_sizes['forearms'])

    # 3. Hand mesh (lazy per-frame transform)
    o3d_mesh = self._get_hand_mesh(frame_idx)
    if o3d_mesh and o3d_mesh.has_triangles() and self._visibility['hand_meshes']:
        verts = np.asarray(o3d_mesh.vertices)
        tris = np.asarray(o3d_mesh.triangles)
        faces = np.hstack([np.full((len(tris), 1), 3, dtype=tris.dtype), tris])
        mesh = pv.PolyData(verts, faces)
        self.plotter.add_mesh(mesh, color='white', style='wireframe', name='hand_meshes')

    # 4. Stickers + compasses
    for name, positions in self._stickers_xyz_dict.items():
        pos = positions[frame_idx] if frame_idx < len(positions) else None
        if pos is None or np.any(np.isnan(pos)):
            continue
        color = self._custom_colors.get(name, 'magenta')
        if self._visibility.get(name, True):
            sphere = pv.Sphere(radius=4.0, center=pos.tolist())
            self.plotter.add_mesh(sphere, color=color, name=f'sticker_{name}')
        if name in self._compass_widgets and frame_idx > 0:
            prev_pos = positions[frame_idx - 1]
            if not np.any(np.isnan(prev_pos)):
                self._compass_widgets[name].update_velocity(pos - prev_pos)

    # 5. Single render call
    self.plotter.render()

    # 6. Neural panel cursor
    if self.neural_panel is not None:
        self.neural_panel.update_cursor(frame_idx, self._neural_scale)

    # 7. Frame label
    self.frame_label.setText(f"{frame_idx + 1} / {self._total_frames}")

    # 8. Tell preloader to look ahead
    self._preloader.seek(frame_idx + 1)
```

### Supporting Methods

**`_crop_pointcloud_gpu(xyz, colors, center, half_size)`:**
```python
def _crop_pointcloud_gpu(self, xyz, colors, center, half_size):
    if self._cupy_available:
        import cupy as cp
        xyz_gpu = cp.asarray(xyz.astype(np.float32))
        c_gpu = cp.asarray(center.astype(np.float32))
        mask = (cp.abs(xyz_gpu - c_gpu) <= half_size).all(axis=1) & (xyz_gpu[:, 2] > 0)
        mask_np = cp.asnumpy(mask)
        return cp.asnumpy(xyz_gpu[mask]), colors[mask_np]
    else:
        mask = (np.abs(xyz - center) <= half_size).all(axis=1) & (xyz[:, 2] > 0)
        return xyz[mask], colors[mask]
```

**`_bisect_forearm(frame_idx)`:**
```python
import bisect
def _bisect_forearm(self, frame_idx: int) -> Optional[int]:
    if not self._sorted_forearm_keys:
        return None
    i = bisect.bisect_right(self._sorted_forearm_keys, frame_idx) - 1
    return self._sorted_forearm_keys[i] if i >= 0 else None
```

**`_get_hand_mesh(frame_idx)`** (lazy, cached):
```python
def _get_hand_mesh(self, frame_idx: int):
    if frame_idx == self._last_hand_frame:
        return self._last_hand_mesh
    try:
        mesh = self._hand_manager[frame_idx]
        self._last_hand_frame = frame_idx
        self._last_hand_mesh = mesh
        return mesh
    except Exception:
        return None
```

**`_recenter_view()`:**
```python
def _recenter_view(self):
    self.plotter.camera.focal_point = self.contact_centroid.tolist()
    self.plotter.reset_camera()
    self.plotter.render()
```

**`_toggle_play()`:**
```python
def _toggle_play(self):
    fps = getattr(self._mkv._reader, 'fps', 30)
    if self._play_timer.isActive():
        self._play_timer.stop()
        self.play_button.setText("▶ Play")
    else:
        self._play_timer.start(int(1000 / fps))
        self.play_button.setText("⏸ Pause")

def _play_advance(self):
    nxt = (self.current_index + 1) % self._total_frames
    self.frame_slider.setValue(nxt)  # triggers _on_slider_change → _update_frame
```

**`_on_slider_change(value)`:**
```python
def _on_slider_change(self, value):
    self._update_frame(value)
```

**`_on_crop_changed(value)`:**
```python
def _on_crop_changed(self, value):
    self._crop_half_size = float(value)
```

**`closeEvent(event)`:**
```python
def closeEvent(self, event):
    self._play_timer.stop()
    self._preloader.stop()
    self._preloader.join(timeout=2.0)
    try:
        self._mkv.__exit__(None, None, None)
    except Exception:
        pass
    super().closeEvent(event)
```

### Module-Level CuPy Detector

```python
def _detect_cupy() -> bool:
    try:
        import cupy as cp
        cp.array([1.0])  # test CUDA is accessible
        return True
    except Exception:
        print("CuPy not available or CUDA not accessible. Using CPU-based point cloud filtering.")
        return False
```

---

## Group 6 🔴 — Update `common/__init__.py`

**File:** `code/src/preprocessing/common/__init__.py`
**Dependencies:** Group 5 | **Sequential:** YES

Check what is currently exported, then add:
```python
from .gui.neural_kinect_scene_viewer import NeuralKinectViewer
```

And add `"NeuralKinectViewer"` to `__all__` if that list is defined.

No other changes to this file.
