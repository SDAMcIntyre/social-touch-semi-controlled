# NeuralKinectViewer — 01: Foundation (Groups 1 & 2)

## Group 1 🟢 — Sticker Velocity Compass Widget

**File:** `code/src/preprocessing/common/gui/sticker_velocity_compass.py`
**Dependencies:** None | **Parallelizable:** YES

### Class: `StickerVelocityCompass(QWidget)`

```python
from PyQt5.QtWidgets import QWidget, QSizePolicy
from PyQt5.QtCore import Qt, QPointF, QRectF, QSize
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont
import numpy as np
from typing import Optional
```

**Constructor:** `(sticker_name: str, sticker_color: str, parent=None)`
- Fixed size: `120 × 140 px` via `setFixedSize(120, 140)`
- Store `self._sticker_color = QColor(sticker_color)`, `self._sticker_name`
- `self._v_xyz: Optional[np.ndarray] = None`
- `self._scale_mm_per_frame: float = 5.0` (tunable)

**`update_velocity(v_xyz: np.ndarray) → None`**
- Stores `self._v_xyz = v_xyz`
- Calls `self.update()` to trigger repaint

**`paintEvent(event) → None`** — rendering logic:
1. `painter = QPainter(self)` with antialiasing
2. Define outer rect: `r = self.rect().adjusted(4, 4, -4, -4)` (leaves border margin)
3. **Outer border ring** (sticker color): `painter.setPen(QPen(self._sticker_color, 4))`, `painter.setBrush(Qt.NoBrush)`, draw ellipse on `r`
4. **Dark background fill**: `painter.setBrush(QBrush(QColor(30, 30, 30)))`, `painter.setPen(Qt.NoPen)`, fill `r.adjusted(4,4,-4,-4)`
5. Compute `center = r.center()`, `radius = min(r.width(), r.height()) / 2 - 8`
6. If `self._v_xyz` is not None and non-zero:
   - `vx, vy, vz = self._v_xyz`
   - Arrow direction: `theta = atan2(-vy, vx)` (flip Y for screen coords)
   - `mag_xy = sqrt(vx**2 + vy**2)`
   - `arrow_len = min(mag_xy / self._scale_mm_per_frame * radius, radius * 0.85)`
   - **Z-to-color**: `z_norm = clip(vz / 3.0, -1, 1)`; `r_c = int(128 + 127*z_norm)`, `b_c = int(128 - 127*z_norm)`, `arrow_color = QColor(r_c, 50, b_c)`
   - Draw line from center to `(center.x + cos(theta)*arrow_len, center.y + sin(theta)*arrow_len)`
   - Draw filled arrowhead triangle (3 points, same Z-color)
7. **Abs velocity label**: small gray text at bottom of circle interior
8. **Sticker name label**: 8pt white text below the circle, centered

**`sizeHint() → QSize`**: returns `QSize(120, 140)`

---

## Group 2 🟢 — FramePreloader (Background MKV Thread)

**Location:** Inside `code/src/preprocessing/common/gui/neural_kinect_scene_viewer.py`
**Dependencies:** `KinectPointCloudView` (existing) | **Parallelizable:** YES

### Class: `FramePreloader(threading.Thread)`

```python
import threading, queue, collections
from typing import Optional, Any
```

**Constructor:** `(point_cloud_view, buffer_size: int = 8)`
- `self._source = point_cloud_view` — ONLY this thread calls `_source[idx]`
- `self._seek_queue = queue.Queue(maxsize=1)`
- `self._buffer = collections.OrderedDict()` — `{frame_idx: PointCloudData}`
- `self._lock = threading.Lock()`
- `self._stop_event = threading.Event()`
- `self._next_frame = 0`
- `super().__init__(daemon=True)`

**`seek(frame_idx: int)`** — called from main thread:
```python
def seek(self, frame_idx: int):
    try:
        self._seek_queue.put_nowait(frame_idx)
    except queue.Full:
        try: self._seek_queue.get_nowait()
        except queue.Empty: pass
        self._seek_queue.put_nowait(frame_idx)
```

**`get_frame(frame_idx: int)`** — called from main thread:
```python
def get_frame(self, frame_idx: int) -> Optional[Any]:
    with self._lock:
        if frame_idx in self._buffer:
            return self._buffer[frame_idx]
    # Cache miss: synchronous fallback (rare — large jump)
    return self._source[frame_idx]
```

**`stop()`**: `self._stop_event.set()`

**`run()`** — worker loop:
```python
def run(self):
    while not self._stop_event.is_set():
        # 1. Check for seek request (resets position)
        try:
            new_target = self._seek_queue.get_nowait()
            with self._lock:
                self._buffer.clear()
            self._next_frame = new_target
        except queue.Empty:
            pass

        # 2. Fill buffer ahead if not full
        with self._lock:
            buffered = len(self._buffer)

        if buffered < self._buffer_size:
            total = len(self._source)
            if self._next_frame < total:
                try:
                    frame_data = self._source[self._next_frame]
                    with self._lock:
                        if len(self._buffer) >= self._buffer_size:
                            self._buffer.popitem(last=False)  # evict oldest
                        self._buffer[self._next_frame] = frame_data
                    self._next_frame += 1
                except Exception:
                    self._next_frame += 1  # skip corrupt frame
            else:
                self._stop_event.wait(0.05)  # wait at end of file
        else:
            self._stop_event.wait(0.01)  # buffer full, idle
```

### Thread Safety Constraint (CRITICAL)
- **Main thread**: only calls `preloader.get_frame()` and `preloader.seek()`
- **Preloader thread**: only one that calls `self._source[idx]` (i.e., `KinectPointCloudView.__getitem__`)
- `pyk4a` is NOT thread-safe — this single-producer design is mandatory
