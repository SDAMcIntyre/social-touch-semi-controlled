# Standard library imports
from dataclasses import dataclass, field
from typing import List, Tuple

# Third-party imports
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


@dataclass
class LineSpec:
    """Specification for a single line within a subplot."""
    label: str
    data: np.ndarray           # 1-D array, one value per frame
    color: str = "#ffffff"
    linewidth: float = 0.8


@dataclass
class SubplotSpec:
    """Specification for one subplot row inside a TimeSeriesPanel."""
    ylabel: str
    lines: List[LineSpec] = field(default_factory=list)


class TimeSeriesPanel(QWidget):
    """
    A reusable Matplotlib panel that stacks an arbitrary number of subplots
    (shared X axis) and displays a red vertical cursor line that tracks the
    current frame.

    Layout
    ------
    - A thin toolbar row above the canvas contains per-subplot visibility
      checkboxes on the left, a stretch, and a ± zoom spinbox on the right.
    - The canvas height is ``max(180, min(55 * n_visible + 40, 400))`` px.

    Zoom mechanics
    --------------
    - Mouse-wheel over the canvas zooms in/out (scroll-up → zoom in).
    - The spinbox shows the current half-window in seconds and stays in sync
      with wheel events.
    - Zoom is clamped between 0.5 s and half the total recording length.

    Parameters
    ----------
    subplot_specs:
        One ``SubplotSpec`` per subplot row.
    total_frames:
        Total number of frames in the recording (determines X-axis length).
    fps:
        Playback frame-rate used to convert frames ↔ seconds.
    default_zoom_seconds:
        Initial half-window width shown around the cursor (in seconds).
    parent:
        Optional Qt parent widget.
    """

    def __init__(
        self,
        subplot_specs: List[SubplotSpec],
        total_frames: int,
        *,
        fps: float = 30.0,
        default_zoom_seconds: float = 15.0,
        parent=None,
    ):
        super().__init__(parent)

        self._subplot_specs: List[SubplotSpec] = subplot_specs
        self._visible: List[bool] = [True] * len(subplot_specs)
        self._total_frames: int = total_frames
        self._fps: float = fps
        self._total_duration: float = total_frames / fps

        # Zoom state (in seconds)
        self._zoom_half_window_s: float = default_zoom_seconds
        self._max_half_window_s: float = max(0.5, self._total_duration / 2.0)
        self._current_frame: int = 0

        # Touch-band state
        self._touch_boundaries: List[Tuple[float, float, int]] = []
        self._touch_bands_visible: bool = True
        self._touch_spans: List = []

        # Panel height fixed from total subplot count — never changes on toggle
        self._canvas_height_px: int = max(180, min(55 * len(subplot_specs) + 40, 400))
        self.fig = Figure(figsize=(12, self._canvas_height_px / 96.0), tight_layout=True)
        self.fig.patch.set_facecolor('#1a1a2e')
        self.canvas = FigureCanvasQTAgg(self.fig)

        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar row: visibility checkboxes (left) + stretch + ± spinbox (right)
        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(4, 1, 4, 1)
        btn_layout.setSpacing(6)

        for i, spec in enumerate(subplot_specs):
            cb = QCheckBox(spec.ylabel)
            cb.setChecked(True)
            cb.setStyleSheet("color: black; font-size: 8pt;")
            cb.toggled.connect(lambda checked, idx=i: self._on_toggle(idx, checked))
            btn_layout.addWidget(cb)

        self._touch_bands_cb = QCheckBox("Touch bands")
        self._touch_bands_cb.setChecked(True)
        self._touch_bands_cb.setStyleSheet("color: black; font-size: 8pt;")
        self._touch_bands_cb.setVisible(False)
        self._touch_bands_cb.toggled.connect(self._on_touch_bands_toggled)
        btn_layout.addWidget(self._touch_bands_cb)

        btn_layout.addStretch()

        btn_layout.addWidget(QLabel("±"))
        self._window_spinbox = QDoubleSpinBox()
        self._window_spinbox.setMinimum(0.5)
        _max_secs = min(total_frames / fps / 2.0, 3600.0)
        self._window_spinbox.setMaximum(_max_secs)
        self._window_spinbox.setSingleStep(0.5)
        self._window_spinbox.setValue(default_zoom_seconds)
        self._window_spinbox.setSuffix(" s")
        self._window_spinbox.setFixedWidth(70)
        self._window_spinbox.setFixedHeight(18)
        self._window_spinbox.setStyleSheet("font-size: 8pt;")
        self._window_spinbox.valueChanged.connect(self._on_zoom_window_changed)
        btn_layout.addWidget(self._window_spinbox)

        layout.addWidget(btn_row)

        self.canvas.installEventFilter(self)
        layout.addWidget(self.canvas)

        self._axes = None
        self._cursor_lines = []
        self.setFixedHeight(self._canvas_height_px + 22)  # +22 for toolbar row
        self._rebuild_axes()

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _rebuild_axes(self) -> None:
        """Clear the figure and re-create only the visible subplots."""
        self.fig.clear()
        self._axes = None
        self._cursor_lines = []
        self._touch_spans = []

        visible_specs = [s for s, v in zip(self._subplot_specs, self._visible) if v]
        n = len(visible_specs)
        if n == 0:
            self.canvas.draw()
            return

        axes = self.fig.subplots(n, 1, sharex=True)
        if n == 1:
            axes = [axes]
        self._axes = axes

        x = np.arange(self._total_frames) / self._fps

        for i, (ax, spec) in enumerate(zip(axes, visible_specs)):
            ax.set_facecolor('#0d0d1a')
            for line_spec in spec.lines:
                data = np.asarray(line_spec.data)
                ax.plot(x, data, color=line_spec.color, lw=line_spec.linewidth,
                        label=line_spec.label)
            if any(ls.label for ls in spec.lines):
                ax.legend(loc='upper right', fontsize=6,
                          facecolor='#1a1a2e', labelcolor='white')
            ax.set_ylabel(spec.ylabel, color='white', fontsize=7)
            ax.tick_params(colors='white', labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333355')
            if i == n - 1:
                ax.set_xlabel('Time (s)', color='white', fontsize=7)
            cursor = ax.axvline(x=self._current_frame / self._fps,
                                color='red', lw=1.5, alpha=0.9)
            self._cursor_lines.append(cursor)

        if self._touch_boundaries and self._touch_bands_visible:
            _band_colors = ('#ff4444', '#44ff44')
            for touch_order, (start, end, _touch_id) in enumerate(self._touch_boundaries):
                color = _band_colors[touch_order % 2]
                for ax in axes:
                    span = ax.axvspan(start, end, color=color, alpha=0.12, zorder=0)
                    self._touch_spans.append(span)

        self._apply_xlim()
        self.canvas.draw()

    def _on_toggle(self, idx: int, checked: bool) -> None:
        self._visible[idx] = checked
        self._rebuild_axes()

    def _on_touch_bands_toggled(self, checked: bool) -> None:
        self._touch_bands_visible = checked
        self._rebuild_axes()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_touch_boundaries(
        self, boundaries: List[Tuple[float, float, int]]
    ) -> None:
        """
        Set touch-boundary intervals for background shading.

        Parameters
        ----------
        boundaries:
            List of ``(start_s, end_s, touch_id)`` tuples, where *start_s*
            and *end_s* are in seconds (matching the panel x-axis).  Bands
            are drawn in alternating red/green order determined by list
            position, not by the ``touch_id`` value.
        """
        self._touch_boundaries = boundaries
        self._touch_bands_cb.setVisible(True)
        self._rebuild_axes()

    def update_cursor(self, frame_idx: int) -> None:
        """
        Move the red vertical cursor to *frame_idx* and re-centre the zoom
        window around it.  Uses ``draw_idle()`` so it is safe to call at
        playback frame-rate.
        """
        self._current_frame = frame_idx
        t = frame_idx / self._fps
        for line in self._cursor_lines:
            line.set_xdata([t])
        self._apply_xlim()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Zoom helpers
    # ------------------------------------------------------------------

    def _apply_xlim(self) -> None:
        t = self._current_frame / self._fps
        lo = max(0.0, t - self._zoom_half_window_s)
        hi = min(self._total_duration, t + self._zoom_half_window_s)
        if self._axes is not None:
            self._axes[0].set_xlim(lo, hi)

    def _on_zoom_window_changed(self, seconds: float) -> None:
        self._zoom_half_window_s = seconds
        self._apply_xlim()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Event filter — mouse-wheel zoom
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event) -> bool:
        if obj is self.canvas and event.type() == QEvent.Wheel:
            delta = event.angleDelta().y()
            if delta == 0:
                return False
            notches = delta / 120.0              # typically ±1 per detent
            factor = 1.25 ** (-notches)          # scroll-up shrinks window
            new_half_s = self._zoom_half_window_s * factor
            new_half_s = max(0.5, min(new_half_s, self._max_half_window_s))
            self._zoom_half_window_s = new_half_s
            # Sync spinbox without re-triggering _on_zoom_window_changed
            self._window_spinbox.blockSignals(True)
            self._window_spinbox.setValue(new_half_s)
            self._window_spinbox.blockSignals(False)
            self._apply_xlim()
            self.canvas.draw_idle()
            return True    # consume event
        return False
