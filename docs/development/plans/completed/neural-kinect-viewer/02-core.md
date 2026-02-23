# NeuralKinectViewer — 02: Core Components (Groups 3 & 4)

## Group 3 🟡 — NeuralDataPanel (Matplotlib Time-Series)

**Location:** Inside `code/src/preprocessing/common/gui/neural_kinect_scene_viewer.py`
**Dependencies:** None structurally | **Parallelizable:** YES

### Class: `NeuralDataPanel(QWidget)`

**Imports:**
```python
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
```

**Constructor:** `(merged_df: pd.DataFrame, total_kinect_frames: int, parent=None)`

```python
def __init__(self, merged_df, total_kinect_frames, parent=None):
    super().__init__(parent)
    self._total_kinect_frames = total_kinect_frames

    # Create matplotlib figure (3 rows sharing x-axis)
    self.fig = Figure(figsize=(12, 2.5), tight_layout=True)
    self.fig.patch.set_facecolor('#1a1a2e')  # dark background to match scene
    self.canvas = FigureCanvasQTAgg(self.fig)

    layout = QVBoxLayout(self)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(self.canvas)
    self.setFixedHeight(200)

    self._setup_axes(merged_df)
```

**`_setup_axes(merged_df)` method:**
```python
def _setup_axes(self, merged_df):
    axes = self.fig.subplots(3, 1, sharex=True)
    self.ax_freq, self.ax_depth, self.ax_area = axes

    x = np.arange(len(merged_df))
    colors = [('#ff9500', 'Nerve_freq', 'IFF (Hz)'),
              ('#00d4ff', 'contact_depth', 'Depth (mm)'),
              ('#39ff14', 'contact_area', 'Area (mm²)')]

    self._cursor_lines = []
    for ax, (color, col, ylabel) in zip(axes, colors):
        ax.set_facecolor('#0d0d1a')
        if col in merged_df.columns:
            ax.plot(x, merged_df[col].fillna(0), color=color, lw=0.8)
        ax.set_ylabel(ylabel, color='white', fontsize=7)
        ax.tick_params(colors='white', labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')
        line = ax.axvline(x=0, color='red', lw=1.5, alpha=0.9)
        self._cursor_lines.append(line)

    self.canvas.draw()
```

**`update_cursor(frame_idx: int, scale_factor: float)` method:**
```python
def update_cursor(self, frame_idx, scale_factor):
    sample_idx = int(frame_idx * scale_factor)
    for line in self._cursor_lines:
        line.set_xdata([sample_idx])
    self.canvas.draw_idle()  # non-blocking, deduplicates rapid calls
```

- `scale_factor = len(merged_df) / total_kinect_frames` — computed once in `NeuralKinectViewer.__init__()`
- `draw_idle()` rather than `draw()` prevents matplotlib jank at 30fps

---

## Group 4 🟡 — Extract `define_custom_colors()`

**Dependencies:** None | **Parallelizable:** YES

### What to do

The function `define_custom_colors()` is currently defined inline in the entry-point script
`code/scripts/_3_preprocessing/_4_somatosensory_quantification/view_somatosensory_3d_scene.py` (lines 35-62).
It needs to be a shared utility, accessible from both the old viewer and the new one.

**Step 1:** Copy the function verbatim into `neural_kinect_scene_viewer.py` as a module-level function:

```python
def define_custom_colors(string_list) -> dict[str, str]:
    """Searches an iterable of strings for standard color keywords."""
    STANDARD_COLORS = {
        "red", "green", "blue", "yellow", "orange", "purple", "pink",
        "black", "white", "brown", "gray", "grey", "cyan", "magenta", "violet"
    }
    found_colors = {}
    for item in string_list:
        item_lower = item.lower()
        for color in STANDARD_COLORS:
            if color in item_lower:
                found_colors[item] = color
    return found_colors
```

**Step 2:** In `view_somatosensory_3d_scene.py`, replace the function definition with an import:
```python
from preprocessing.common.gui.neural_kinect_scene_viewer import define_custom_colors
```

Delete lines 35-62 (the original function body) from that script.
