"""PyQt5 QMainWindow viewer for touch preparation pipeline output."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from pyvistaqt import QtInteractor

from analysis.touch_analytics.gui.preparation_viewer_data import (
    PreparationTouchData,
    PreparationViewerData,
)

logger = logging.getLogger(__name__)

_GESTURE_COLORS = {
    'tap': 'green',
    'stroke_proximal': 'blue',
    'stroke_distal': 'red',
    'stroke_unknown': 'grey',
}

_DEFAULT_SIGNALS = [
    'contact_depth',
    'contact_area',
    'contact_location_x',
    'Nerve_freq',
    'Nerve_spike',
]

_SIGNAL_LABELS = {
    'contact_depth': 'Depth',
    'contact_area': 'Area',
    'contact_location_x': 'Loc X',
    'contact_location_y': 'Loc Y',
    'contact_location_z': 'Loc Z',
    'Nerve_freq': 'Freq',
    'Nerve_spike': 'Spikes',
}


class TouchPreparationViewer(QMainWindow):
    def __init__(
        self,
        data: PreparationViewerData,
        sessions: Optional[List[Tuple[str, PreparationViewerData]]] = None,
        title: str = "Touch Preparation Viewer",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)

        self._data = data
        self._sessions: List[Tuple[str, PreparationViewerData]] = (
            sessions if sessions is not None else [(data.session_id, data)]
        )

        self._initialized = False
        self._current_touch: Optional[PreparationTouchData] = None
        self._current_frame = 0
        self._pending_slider_value = 0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)

        self._slider_drag_timer = QTimer(self)
        self._slider_drag_timer.setSingleShot(True)
        self._slider_drag_timer.timeout.connect(self._on_slider_settle)

        self._plotter: Optional[QtInteractor] = None
        self._forearm_actor = None
        self._centroid_actor = None
        self._cursor_lines: list = []
        self._axes: list = []

        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        self._build_toolbar()

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        self._splitter = QSplitter(Qt.Vertical)

        self._plotter = QtInteractor(self)
        self._splitter.addWidget(self._plotter.interactor)

        self._fig = Figure(tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.mpl_connect('button_press_event', self._on_canvas_click)
        self._splitter.addWidget(self._canvas)

        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 2)

        main_layout.addWidget(self._splitter, stretch=1)
        main_layout.addLayout(self._build_frame_controls())

        self.resize(1200, 700)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addWidget(QLabel("Session:"))
        self._session_combo = QComboBox()
        self._session_combo.setMinimumWidth(160)
        for label, _ in self._sessions:
            self._session_combo.addItem(label)
        self._session_combo.currentIndexChanged.connect(self._on_session_changed)
        toolbar.addWidget(self._session_combo)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Block:"))
        self._block_combo = QComboBox()
        self._block_combo.setMinimumWidth(80)
        self._block_combo.currentIndexChanged.connect(self._on_block_changed)
        toolbar.addWidget(self._block_combo)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Trial:"))
        self._trial_combo = QComboBox()
        self._trial_combo.setMinimumWidth(80)
        self._trial_combo.currentIndexChanged.connect(self._on_trial_changed)
        toolbar.addWidget(self._trial_combo)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Touch:"))
        self._touch_combo = QComboBox()
        self._touch_combo.setMinimumWidth(200)
        self._touch_combo.currentIndexChanged.connect(self._on_touch_changed)
        toolbar.addWidget(self._touch_combo)

        toolbar.addSeparator()

        self._gesture_label = QLabel("")
        self._gesture_label.setFixedWidth(120)
        toolbar.addWidget(self._gesture_label)

    def _build_frame_controls(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Frame:"))

        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.setEnabled(False)
        self._frame_slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self._frame_slider, stretch=1)

        self._frame_label = QLabel("0 / 0")
        self._frame_label.setFixedWidth(80)
        layout.addWidget(self._frame_label)

        separator = QWidget()
        separator.setFixedWidth(16)
        layout.addWidget(separator)

        self._play_btn = QPushButton("▶ Play")
        self._play_btn.clicked.connect(self._toggle_play)
        layout.addWidget(self._play_btn)

        layout.addWidget(QLabel("Speed:"))
        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(0.1, 10.0)
        self._speed_spin.setValue(1.0)
        self._speed_spin.setSingleStep(0.25)
        self._speed_spin.setSuffix("x")
        layout.addWidget(self._speed_spin)

        return layout

    # ------------------------------------------------------------------ #
    # Combo cascade helpers
    # ------------------------------------------------------------------ #

    def _populate_block_combo(self, data: PreparationViewerData) -> None:
        self._block_combo.blockSignals(True)
        self._block_combo.clear()
        for bid in data.block_order_ids:
            self._block_combo.addItem(f"Block {bid}")
        self._block_combo.blockSignals(False)
        if data.block_order_ids:
            self._populate_trial_combo(data, data.block_order_ids[0])

    def _populate_trial_combo(self, data: PreparationViewerData, block_id: str) -> None:
        self._trial_combo.blockSignals(True)
        self._trial_combo.clear()
        for tid in data.trial_ids_by_block.get(block_id, []):
            self._trial_combo.addItem(f"Trial {tid}")
        self._trial_combo.blockSignals(False)
        trial_ids = data.trial_ids_by_block.get(block_id, [])
        if trial_ids:
            self._populate_touch_combo(data, block_id, trial_ids[0])

    def _populate_touch_combo(
        self, data: PreparationViewerData, block_id: str, trial_id: int
    ) -> None:
        self._touch_combo.blockSignals(True)
        self._touch_combo.clear()
        for td in data.touches_by_block_trial.get((block_id, trial_id), []):
            self._touch_combo.addItem(f"Touch {td.single_touch_id} ({td.gesture_type})")
        self._touch_combo.blockSignals(False)
        if self._touch_combo.count() > 0:
            self._touch_combo.setCurrentIndex(0)
        if self._initialized:
            touches = data.touches_by_block_trial.get((block_id, trial_id), [])
            if touches:
                self._load_touch(touches[0])
                self._set_frame(0)

    def _current_block_id(self) -> Optional[str]:
        idx = self._block_combo.currentIndex()
        if idx < 0 or idx >= len(self._data.block_order_ids):
            return None
        return self._data.block_order_ids[idx]

    def _current_trial_id(self) -> Optional[int]:
        bid = self._current_block_id()
        if bid is None:
            return None
        trial_ids = self._data.trial_ids_by_block.get(bid, [])
        idx = self._trial_combo.currentIndex()
        if idx < 0 or idx >= len(trial_ids):
            return None
        return trial_ids[idx]

    # ------------------------------------------------------------------ #
    # 3D view
    # ------------------------------------------------------------------ #

    def _render_forearm(self) -> None:
        self._plotter.clear()
        self._plotter.set_background("black")
        cloud = pv.PolyData(self._data.forearm_vertices)
        self._plotter.add_mesh(cloud, color=[0.3, 0.3, 0.3], point_size=3, name="forearm")
        self._plotter.view_xy()
        self._plotter.render()
        self._centroid_actor = None

    def _update_3d_frame(self, idx: int) -> None:
        if self._current_touch is None:
            return
        pts = self._current_touch.frame_contact_pts[idx]
        if len(pts) > 0:
            cloud = pv.PolyData(pts)
            self._plotter.add_mesh(
                cloud, color="red", point_size=8,
                render_points_as_spheres=True, name="contacts",
            )
        else:
            empty = pv.PolyData(np.zeros((1, 3), dtype=np.float64))
            self._plotter.add_mesh(empty, color="red", point_size=1, name="contacts")
        self._plotter.render()

    # ------------------------------------------------------------------ #
    # Time-series panel
    # ------------------------------------------------------------------ #

    def _plot_signals(self, touch: PreparationTouchData) -> None:
        self._fig.clear()
        self._axes = []
        self._cursor_lines = []

        signals_to_plot = [s for s in _DEFAULT_SIGNALS if s in touch.signals]
        if not signals_to_plot:
            self._canvas.draw()
            return

        t = touch.time
        n = len(signals_to_plot)

        for i, col in enumerate(signals_to_plot):
            sharex = self._axes[0] if i > 0 else None
            ax = self._fig.add_subplot(n, 1, i + 1, sharex=sharex)
            self._axes.append(ax)

            vals = touch.signals[col]
            if col == 'Nerve_spike':
                spike_times = t[vals > 0]
                ax.vlines(spike_times, 0, 1, color='black', linewidth=0.5)
                ax.set_ylim(-0.1, 1.1)
            else:
                ax.plot(t, vals, linewidth=1)

            ax.set_ylabel(_SIGNAL_LABELS.get(col, col), fontsize=8)
            ax.tick_params(labelsize=7)

            if i < n - 1:
                ax.tick_params(labelbottom=False)

        if self._axes:
            self._axes[-1].set_xlabel("Time (s)", fontsize=8)

        self._fig.tight_layout()

        for ax in self._axes:
            vline = ax.axvline(x=t[0], color='red', linewidth=0.8, alpha=0.7)
            self._cursor_lines.append(vline)

        self._canvas.draw()

    def _update_cursor(self, idx: int) -> None:
        if self._current_touch is None or not self._cursor_lines:
            return
        t = self._current_touch.time[idx]
        for line in self._cursor_lines:
            line.set_xdata([t, t])
        self._canvas.draw_idle()

    def _on_canvas_click(self, event) -> None:
        if event.inaxes is None or self._current_touch is None:
            return
        t_click = event.xdata
        if t_click is None:
            return
        idx = int(np.argmin(np.abs(self._current_touch.time - t_click)))
        self._stop()
        self._set_frame(idx)

    # ------------------------------------------------------------------ #
    # Touch loading and frame control
    # ------------------------------------------------------------------ #

    def _load_touch(self, touch: PreparationTouchData) -> None:
        self._current_touch = touch
        self._current_frame = 0
        self._stop()

        n = len(touch.time)
        self._frame_slider.blockSignals(True)
        self._frame_slider.setRange(0, max(0, n - 1))
        self._frame_slider.setValue(0)
        self._frame_slider.setEnabled(n > 0)
        self._frame_slider.blockSignals(False)
        self._frame_label.setText(f"1 / {n}")

        color = _GESTURE_COLORS.get(touch.gesture_type, 'grey')
        self._gesture_label.setText(touch.gesture_type)
        self._gesture_label.setStyleSheet(f"color: {color}; font-weight: bold;")

        if self._initialized:
            self._plot_signals(touch)
            self._update_3d_frame(0)

    def _set_frame(self, idx: int) -> None:
        if self._current_touch is None:
            return
        n = len(self._current_touch.time)
        idx = max(0, min(idx, n - 1))
        self._current_frame = idx
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(idx)
        self._frame_slider.blockSignals(False)
        self._frame_label.setText(f"{idx + 1} / {n}")
        self._update_3d_frame(idx)
        self._update_cursor(idx)

    # ------------------------------------------------------------------ #
    # Playback
    # ------------------------------------------------------------------ #

    def _toggle_play(self) -> None:
        if self._timer.isActive():
            self._stop()
        else:
            speed = self._speed_spin.value()
            interval = max(16, round(33 / speed))
            self._timer.start(interval)
            self._play_btn.setText("⏸ Pause")

    def _stop(self) -> None:
        self._timer.stop()
        self._play_btn.setText("▶ Play")

    def _advance_frame(self) -> None:
        if self._current_touch is None:
            self._stop()
            return
        speed = self._speed_spin.value()
        step = max(1, round(speed))
        next_frame = self._current_frame + step
        n = len(self._current_touch.time)
        if next_frame >= n:
            self._stop()
            next_frame = n - 1
        self._set_frame(next_frame)

    # ------------------------------------------------------------------ #
    # Slider throttle
    # ------------------------------------------------------------------ #

    def _on_slider_changed(self, value: int) -> None:
        if self._current_touch is not None:
            n = len(self._current_touch.time)
            self._frame_label.setText(f"{value + 1} / {n}")
        self._pending_slider_value = value
        self._slider_drag_timer.start(80)

    def _on_slider_settle(self) -> None:
        self._stop()
        self._set_frame(self._pending_slider_value)

    # ------------------------------------------------------------------ #
    # Signal handlers
    # ------------------------------------------------------------------ #

    def _on_session_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._sessions):
            raise ValueError(
                f"_on_session_changed: index {index} out of range "
                f"for {len(self._sessions)} sessions"
            )
        _, new_data = self._sessions[index]
        self._data = new_data
        if self._initialized:
            self._render_forearm()
        self._populate_block_combo(new_data)

    def _on_block_changed(self, index: int) -> None:
        if not self._initialized or index < 0:
            return
        bid = self._current_block_id()
        if bid is None:
            return
        self._populate_trial_combo(self._data, bid)

    def _on_trial_changed(self, index: int) -> None:
        if not self._initialized or index < 0:
            return
        bid = self._current_block_id()
        if bid is None:
            return
        trial_ids = self._data.trial_ids_by_block.get(bid, [])
        if index >= len(trial_ids):
            return
        trial_id = trial_ids[index]
        self._populate_touch_combo(self._data, bid, trial_id)

    def _on_touch_changed(self, index: int) -> None:
        if not self._initialized or index < 0:
            return
        bid = self._current_block_id()
        tid = self._current_trial_id()
        if bid is None or tid is None:
            return
        touches = self._data.touches_by_block_trial.get((bid, tid), [])
        if index >= len(touches):
            return
        self._load_touch(touches[index])
        self._set_frame(0)

    # ------------------------------------------------------------------ #
    # Qt lifecycle
    # ------------------------------------------------------------------ #

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._initialized:
            QTimer.singleShot(0, self._deferred_start)

    def _deferred_start(self) -> None:
        self._initialized = True
        self._render_forearm()
        h = self._splitter.height()
        if h > 0:
            self._splitter.setSizes([h // 3, h - h // 3])
        self._populate_block_combo(self._data)

    def closeEvent(self, event) -> None:
        self._stop()
        if self._plotter is not None:
            try:
                self._plotter.close()
            except Exception:
                pass
        super().closeEvent(event)
