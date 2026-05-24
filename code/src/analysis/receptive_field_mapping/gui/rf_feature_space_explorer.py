"""RF Feature-Space Explorer GUI — scatter + 3D forearm heatmap with live filter.

Presents a 2D scatter (pressure × velocity_signed, colored by gesture type)
alongside an interactive PyVista 3D forearm mesh with a spike-density heatmap.
A draggable/resizable rectangle on the scatter filters which frames contribute
to the 3D heatmap; gesture-type checkboxes provide additional filtering.
"""

import json
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pyvista as pv
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
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
from pyvistaqt import QtInteractor

from analysis.receptive_field_mapping.data.rf_explorer_data import ExplorerData

from analysis.pipeline.shared_constants import GESTURE_TYPES

logger = logging.getLogger(__name__)

_GESTURE_TYPES_WITH_UNKNOWN_WITH_UNKNOWN = (*GESTURE_TYPES, 'stroke_unknown')

_GESTURE_COLORS = {
    "tap": "tab:blue",
    "stroke_proximal": "tab:orange",
    "stroke_distal": "tab:green",
    "stroke_unknown": "tab:red",
}

_CORNER_RADIUS_FRAC = 0.05
_EDGE_PROXIMITY_FRAC = 0.05


class DraggableFilterRect:
    """Matplotlib rectangle with 8-zone drag/resize and a change callback.

    Zones: 4 corners ('nw', 'ne', 'sw', 'se'), 4 edges ('n', 's', 'e', 'w'),
    and interior ('move').  Corners are detected by distance < a fraction of
    rectangle size; edges by proximity to each side.

    Parameters
    ----------
    ax:
        The matplotlib Axes to add the rectangle to.
    x0, y0, width, height:
        Initial rectangle geometry in data coordinates.
    on_changed:
        Called with (x_min, x_max, y_min, y_max) after every geometry update.
    """

    def __init__(
        self,
        ax,
        x0: float,
        y0: float,
        width: float,
        height: float,
        on_changed: Callable[[float, float, float, float], None],
    ) -> None:
        self._ax = ax
        self._on_changed = on_changed
        self._drag_mode: Optional[str] = None
        self._press_xy: Optional[Tuple[float, float]] = None
        self._rect_at_press: Optional[Tuple[float, float, float, float]] = None

        self._patch = Rectangle(
            (x0, y0),
            width,
            height,
            linewidth=1.5,
            edgecolor="royalblue",
            facecolor="royalblue",
            alpha=0.10,
            linestyle="--",
            zorder=5,
        )
        ax.add_patch(self._patch)

        canvas = ax.figure.canvas
        canvas.setMouseTracking(True)
        self._cid_press = canvas.mpl_connect("button_press_event", self._on_press)
        self._cid_motion = canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._cid_release = canvas.mpl_connect("button_release_event", self._on_release)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) in data coordinates."""
        x0 = self._patch.get_x()
        y0 = self._patch.get_y()
        w = self._patch.get_width()
        h = self._patch.get_height()
        x_min, x_max = (x0, x0 + w) if w >= 0 else (x0 + w, x0)
        y_min, y_max = (y0, y0 + h) if h >= 0 else (y0 + h, y0)
        return x_min, x_max, y_min, y_max

    def set_bounds(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        """Set rectangle geometry from bounds and fire on_changed."""
        self._patch.set_xy((x_min, y_min))
        self._patch.set_width(x_max - x_min)
        self._patch.set_height(y_max - y_min)
        self._on_changed(x_min, x_max, y_min, y_max)

    def disconnect(self) -> None:
        """Disconnect all canvas callbacks and remove the patch from the axes."""
        canvas = self._ax.figure.canvas
        canvas.mpl_disconnect(self._cid_press)
        canvas.mpl_disconnect(self._cid_motion)
        canvas.mpl_disconnect(self._cid_release)
        try:
            self._patch.remove()
        except ValueError:
            pass

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

    def _hit_test(self, xdata: float, ydata: float) -> Optional[str]:
        x0 = self._patch.get_x()
        y0 = self._patch.get_y()
        w = self._patch.get_width()
        h = self._patch.get_height()
        x1 = x0 + w
        y1 = y0 + h
        x_lo, x_hi = (x0, x1) if w >= 0 else (x1, x0)
        y_lo, y_hi = (y0, y1) if h >= 0 else (y1, y0)

        if not (x_lo <= xdata <= x_hi and y_lo <= ydata <= y_hi):
            return None

        corner_dx = abs(w) * _CORNER_RADIUS_FRAC
        corner_dy = abs(h) * _CORNER_RADIUS_FRAC
        edge_dx = abs(w) * _EDGE_PROXIMITY_FRAC
        edge_dy = abs(h) * _EDGE_PROXIMITY_FRAC

        near_left = abs(xdata - x_lo) < corner_dx
        near_right = abs(xdata - x_hi) < corner_dx
        near_bottom = abs(ydata - y_lo) < corner_dy
        near_top = abs(ydata - y_hi) < corner_dy

        if near_left and near_bottom:
            return "sw"
        if near_left and near_top:
            return "nw"
        if near_right and near_bottom:
            return "se"
        if near_right and near_top:
            return "ne"

        near_left_edge = abs(xdata - x_lo) < edge_dx
        near_right_edge = abs(xdata - x_hi) < edge_dx
        near_bottom_edge = abs(ydata - y_lo) < edge_dy
        near_top_edge = abs(ydata - y_hi) < edge_dy

        if near_left_edge:
            return "w"
        if near_right_edge:
            return "e"
        if near_bottom_edge:
            return "s"
        if near_top_edge:
            return "n"

        return "move"

    # ------------------------------------------------------------------
    # Mouse event handlers
    # ------------------------------------------------------------------

    def _on_press(self, event) -> None:
        if event.inaxes is not self._ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        mode = self._hit_test(event.xdata, event.ydata)
        if mode is None:
            x_min, x_max, y_min, y_max = self.get_bounds()
            hw = (x_max - x_min) / 2
            hh = (y_max - y_min) / 2
            new_x0 = event.xdata - hw
            new_y0 = event.ydata - hh
            self._patch.set_xy((new_x0, new_y0))
            self._on_changed(new_x0, new_x0 + 2 * hw, new_y0, new_y0 + 2 * hh)
            return
        self._drag_mode = mode
        self._press_xy = (event.xdata, event.ydata)
        x0 = self._patch.get_x()
        y0 = self._patch.get_y()
        self._rect_at_press = (x0, y0, self._patch.get_width(), self._patch.get_height())

    def _on_motion(self, event) -> None:
        if self._drag_mode is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        dx = event.xdata - self._press_xy[0]
        dy = event.ydata - self._press_xy[1]
        x0, y0, w, h = self._rect_at_press

        mode = self._drag_mode

        if mode == "move":
            self._patch.set_xy((x0 + dx, y0 + dy))
        elif mode == "e":
            self._patch.set_width(w + dx)
        elif mode == "w":
            self._patch.set_xy((x0 + dx, y0))
            self._patch.set_width(w - dx)
        elif mode == "n":
            self._patch.set_height(h + dy)
        elif mode == "s":
            self._patch.set_xy((x0, y0 + dy))
            self._patch.set_height(h - dy)
        elif mode == "ne":
            self._patch.set_width(w + dx)
            self._patch.set_height(h + dy)
        elif mode == "nw":
            self._patch.set_xy((x0 + dx, y0))
            self._patch.set_width(w - dx)
            self._patch.set_height(h + dy)
        elif mode == "se":
            self._patch.set_width(w + dx)
            self._patch.set_xy((x0, y0 + dy))
            self._patch.set_height(h - dy)
        elif mode == "sw":
            self._patch.set_xy((x0 + dx, y0 + dy))
            self._patch.set_width(w - dx)
            self._patch.set_height(h - dy)

        x_min, x_max, y_min, y_max = self.get_bounds()
        self._on_changed(x_min, x_max, y_min, y_max)

    def _on_release(self, event) -> None:
        self._drag_mode = None
        self._press_xy = None
        self._rect_at_press = None


class RFFeatureSpaceExplorer(QMainWindow):
    """QMainWindow showing scatter (pressure × velocity_signed) + 3D forearm heatmap.

    Parameters
    ----------
    explorer_data:
        Pre-loaded data for the initial session.
    sessions:
        Optional list of ``(label, ExplorerData)`` tuples for the session
        selector.  If ``None``, defaults to ``[("Session 1", explorer_data)]``.
    title:
        Window title override.
    parent:
        Optional Qt parent widget.
    """

    def __init__(
        self,
        explorer_data: ExplorerData,
        sessions: Optional[List[Tuple[str, ExplorerData]]] = None,
        title: str = "RF Feature-Space Explorer",
        settings_path: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)

        self._data = explorer_data
        self._sessions: List[Tuple[str, ExplorerData]] = (
            sessions if sessions is not None else [("Session 1", explorer_data)]
        )
        self._settings_path = settings_path
        self._initialized = False
        self._heatmap_mode: str = "spike"
        raw_max = float(np.max(explorer_data.iff)) if len(explorer_data.iff) > 0 else 0.0
        self._max_iff: float = raw_max if raw_max > 0.0 else 1.0
        self._rect: Optional[DraggableFilterRect] = None
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(30)
        self._filter_timer.timeout.connect(self._apply_filter_update)
        self._cloud: Optional[pv.PolyData] = None
        self._actor = None
        self._scatter_xlim: Tuple[float, float] = (0.0, 1.0)
        self._scatter_ylim: Tuple[float, float] = (0.0, 1.0)
        self._controls_updating: bool = False

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_toolbar()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        splitter = QSplitter(Qt.Vertical)

        self._plotter = QtInteractor(self)
        splitter.addWidget(self._plotter.interactor)

        scatter_container = QWidget()
        scatter_layout = QVBoxLayout(scatter_container)
        scatter_layout.setContentsMargins(0, 0, 0, 0)
        scatter_layout.setSpacing(2)

        self._figure = Figure()
        self._figure.patch.set_facecolor("white")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)
        self._ax.set_facecolor("white")
        self._canvas.setMinimumSize(350, 180)
        scatter_layout.addWidget(self._canvas, stretch=1)

        scatter_layout.addLayout(self._build_rect_controls())
        scatter_layout.addLayout(self._build_bottom_bar())

        splitter.addWidget(scatter_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter, stretch=1)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Session")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addWidget(QLabel("Session:"))
        self._session_combo = QComboBox()
        for label, _ in self._sessions:
            self._session_combo.addItem(label)
        self._session_combo.currentIndexChanged.connect(self._on_session_changed)
        toolbar.addWidget(self._session_combo)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Mode:"))
        self._heatmap_mode_combo = QComboBox()
        self._heatmap_mode_combo.addItem("Spike density")
        self._heatmap_mode_combo.addItem("IFF (Hz)")
        self._heatmap_mode_combo.currentIndexChanged.connect(self._on_heatmap_mode_changed)
        toolbar.addWidget(self._heatmap_mode_combo)

        toolbar.addSeparator()
        save_btn = QPushButton("Save filter dims")
        save_btn.setToolTip(
            "Save current velocity width and pressure height as startup defaults"
        )
        save_btn.clicked.connect(self._save_settings)
        toolbar.addWidget(save_btn)

    def _build_rect_controls(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 2)
        bar.setSpacing(6)

        bar.addWidget(QLabel("Velocity:"))
        self._cx_slider = QSlider(Qt.Horizontal)
        self._cx_slider.setRange(0, 1000)
        self._cx_slider.setFixedWidth(120)
        self._cx_slider.valueChanged.connect(self._update_rect_from_controls)
        bar.addWidget(self._cx_slider)
        self._cx_val_label = QLabel("—")
        self._cx_val_label.setFixedWidth(52)
        bar.addWidget(self._cx_val_label)

        bar.addWidget(QLabel("Pressure:"))
        self._cy_slider = QSlider(Qt.Horizontal)
        self._cy_slider.setRange(0, 1000)
        self._cy_slider.setFixedWidth(120)
        self._cy_slider.valueChanged.connect(self._update_rect_from_controls)
        bar.addWidget(self._cy_slider)
        self._cy_val_label = QLabel("—")
        self._cy_val_label.setFixedWidth(52)
        bar.addWidget(self._cy_val_label)

        bar.addStretch()

        bar.addWidget(QLabel("V. width:"))
        self._w_spinbox = QDoubleSpinBox()
        self._w_spinbox.setDecimals(3)
        self._w_spinbox.setRange(0.001, 1e6)
        self._w_spinbox.setSingleStep(0.1)
        self._w_spinbox.setFixedWidth(80)
        self._w_spinbox.valueChanged.connect(self._update_rect_from_controls)
        bar.addWidget(self._w_spinbox)

        bar.addWidget(QLabel("P. height:"))
        self._h_spinbox = QDoubleSpinBox()
        self._h_spinbox.setDecimals(3)
        self._h_spinbox.setRange(0.001, 1e6)
        self._h_spinbox.setSingleStep(0.1)
        self._h_spinbox.setFixedWidth(80)
        self._h_spinbox.valueChanged.connect(self._update_rect_from_controls)
        bar.addWidget(self._h_spinbox)

        return bar

    def _build_bottom_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 2)

        self._checkboxes: dict[str, QCheckBox] = {}
        for gtype in _GESTURE_TYPES_WITH_UNKNOWN:
            cb = QCheckBox(gtype)
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_checkbox_changed)
            self._checkboxes[gtype] = cb
            bar.addWidget(cb)

        bar.addStretch()

        self._frame_label = QLabel(
            f"Frames: {self._data.n_frames} / {self._data.n_frames}"
        )
        bar.addWidget(self._frame_label)

        return bar

    # ------------------------------------------------------------------
    # Scatter
    # ------------------------------------------------------------------

    def _draw_scatter(self) -> None:
        self._ax.cla()

        checked_types: set[str] = set()
        if hasattr(self, "_checkboxes"):
            for gtype, cb in self._checkboxes.items():
                if cb.isChecked():
                    checked_types.add(gtype)
        else:
            checked_types = set(_GESTURE_TYPES_WITH_UNKNOWN)

        unique_types = np.unique(self._data.gesture_types)

        vel_amp = np.abs(self._data.velocity_signed)

        for gtype in _GESTURE_TYPES_WITH_UNKNOWN:
            if gtype not in unique_types or gtype not in checked_types:
                continue
            mask = self._data.gesture_types == gtype
            color = _GESTURE_COLORS.get(gtype, "grey")
            self._ax.scatter(
                vel_amp[mask],
                self._data.pressure[mask],
                c=color,
                alpha=0.3,
                s=3,
                rasterized=True,
                label=gtype,
            )

        other_types = [t for t in unique_types if t not in _GESTURE_TYPES_WITH_UNKNOWN]
        for gtype in other_types:
            mask = self._data.gesture_types == gtype
            self._ax.scatter(
                vel_amp[mask],
                self._data.pressure[mask],
                alpha=0.3,
                s=3,
                rasterized=True,
                label=gtype,
            )

        x_lo, x_hi = float(np.nanpercentile(vel_amp, 1)), float(
            np.nanpercentile(vel_amp, 99)
        )
        y_lo, y_hi = float(np.nanpercentile(self._data.pressure, 1)), float(
            np.nanpercentile(self._data.pressure, 99)
        )
        if np.isfinite(x_lo) and np.isfinite(x_hi) and x_lo < x_hi:
            self._ax.set_xlim(x_lo, x_hi)
        if np.isfinite(y_lo) and np.isfinite(y_hi) and y_lo < y_hi:
            self._ax.set_ylim(y_lo, y_hi)

        self._ax.set_xlabel("Hand velocity amplitude (mm/s)")
        self._ax.set_ylabel("Pressure (depth/area, mm⁻¹)")
        self._ax.legend(loc="best", markerscale=3, fontsize=8)
        self._figure.tight_layout()

        if self._rect is not None:
            self._ax.add_patch(self._rect._patch)

        self._canvas.draw()

        self._scatter_xlim = self._ax.get_xlim()
        self._scatter_ylim = self._ax.get_ylim()

    # ------------------------------------------------------------------
    # 3D render
    # ------------------------------------------------------------------

    def _current_heatmap_clim(self) -> Tuple[float, float]:
        if self._heatmap_mode == "iff":
            return (0.0, self._max_iff)
        return (0.0, 1.0)

    def _on_heatmap_mode_changed(self, index: int) -> None:
        self._heatmap_mode = "iff" if index == 1 else "spike"
        self._render_3d()
        if self._rect is not None:
            self._apply_filter_update()

    def _render_3d(self) -> None:
        self._plotter.clear()
        self._plotter.set_background("black")

        vertices = self._data.session_data.forearm_vertices
        n_verts = len(vertices)

        cp_spikes = self._data.spikes[self._data.cp_frame_idx]
        cp_iff = self._data.iff[self._data.cp_frame_idx]
        weights = cp_iff if self._heatmap_mode == "iff" else cp_spikes.astype(float)
        vertex_val_sum = np.bincount(
            self._data.cp_vertex_idx,
            weights=weights,
            minlength=n_verts,
        )
        vertex_n_contacts = np.bincount(
            self._data.cp_vertex_idx,
            minlength=n_verts,
        ).astype(float)
        vertex_density = np.divide(
            vertex_val_sum, vertex_n_contacts,
            out=np.zeros(n_verts), where=vertex_n_contacts > 0,
        )

        vertex_density_display = vertex_density.astype(float)
        vertex_density_display[vertex_n_contacts == 0] = np.nan

        self._cloud = pv.PolyData(vertices)
        self._cloud["heatmap"] = vertex_density_display

        self._actor = self._plotter.add_mesh(
            self._cloud,
            scalars="heatmap",
            cmap="jet",
            clim=self._current_heatmap_clim(),
            nan_color=[0.3, 0.3, 0.3],
            show_scalar_bar=True,
            render_points_as_spheres=False,
            point_size=3,
            smooth_shading=True,
            name="forearm",
            copy_mesh=False,
        )

        self._plotter.view_xy()
        self._plotter.render()

    # ------------------------------------------------------------------
    # Filter update
    # ------------------------------------------------------------------

    def _on_rect_changed(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> None:
        if self._rect is None:
            return
        self._update_controls_from_rect(x_min, x_max, y_min, y_max)
        self._canvas.draw_idle()
        self._filter_timer.start()

    def _on_checkbox_changed(self, _state: int) -> None:
        self._draw_scatter()
        self._apply_filter_update()

    def _apply_filter_update(self) -> None:
        if self._rect is None or self._cloud is None or self._actor is None:
            return

        x_min, x_max, y_min, y_max = self._rect.get_bounds()
        vel_amp = np.abs(self._data.velocity_signed)
        rect_mask = (
            (vel_amp >= x_min)
            & (vel_amp <= x_max)
            & (self._data.pressure >= y_min)
            & (self._data.pressure <= y_max)
        )

        checked_types: set[str] = set()
        for gt, cb in self._checkboxes.items():
            cb.blockSignals(True)
            if cb.isChecked():
                checked_types.add(gt)
            cb.blockSignals(False)

        type_mask = np.isin(self._data.gesture_types, list(checked_types))
        frame_mask = rect_mask & type_mask

        n_verts = len(self._data.session_data.forearm_vertices)
        cp_mask = frame_mask[self._data.cp_frame_idx]
        cp_spikes = self._data.spikes[self._data.cp_frame_idx]
        cp_iff = self._data.iff[self._data.cp_frame_idx]
        active_cp_vertices = self._data.cp_vertex_idx[cp_mask]
        weights = cp_iff[cp_mask] if self._heatmap_mode == "iff" else cp_spikes[cp_mask].astype(float)
        vertex_val_sum = np.bincount(
            active_cp_vertices,
            weights=weights,
            minlength=n_verts,
        )
        vertex_n_contacts = np.bincount(
            active_cp_vertices,
            minlength=n_verts,
        ).astype(float)
        vertex_ratio = np.divide(
            vertex_val_sum, vertex_n_contacts,
            out=np.zeros(n_verts), where=vertex_n_contacts > 0,
        )

        vertex_ratio_display = vertex_ratio.copy()
        vertex_ratio_display[vertex_n_contacts == 0] = np.nan

        self._cloud["heatmap"] = vertex_ratio_display
        self._cloud.Modified()
        self._actor.mapper.scalar_range = self._current_heatmap_clim()
        self._plotter.render()
        self._frame_label.setText(f"Frames: {frame_mask.sum()} / {self._data.n_frames}")

    # ------------------------------------------------------------------
    # Draggable rectangle initialization
    # ------------------------------------------------------------------

    def _init_filter_rect(self) -> None:
        vel_amp = np.abs(self._data.velocity_signed)
        p25_x = float(np.nanpercentile(vel_amp, 25))
        p75_x = float(np.nanpercentile(vel_amp, 75))
        p25_y = float(np.nanpercentile(self._data.pressure, 25))
        p75_y = float(np.nanpercentile(self._data.pressure, 75))

        self._rect = DraggableFilterRect(
            self._ax,
            x0=p25_x,
            y0=p25_y,
            width=p75_x - p25_x,
            height=p75_y - p25_y,
            on_changed=self._on_rect_changed,
        )

        self._update_controls_from_rect(p25_x, p75_x, p25_y, p75_y)
        self._canvas.draw_idle()
        self._apply_filter_update()

    # ------------------------------------------------------------------
    # Controls ↔ rect synchronisation
    # ------------------------------------------------------------------

    def _slider_to_x(self, val: int) -> float:
        lo, hi = self._scatter_xlim
        return lo + val / 1000.0 * (hi - lo)

    def _x_to_slider(self, x: float) -> int:
        lo, hi = self._scatter_xlim
        if hi == lo:
            return 500
        return int(round(float(np.clip((x - lo) / (hi - lo) * 1000, 0, 1000))))

    def _slider_to_y(self, val: int) -> float:
        lo, hi = self._scatter_ylim
        return lo + val / 1000.0 * (hi - lo)

    def _y_to_slider(self, y: float) -> int:
        lo, hi = self._scatter_ylim
        if hi == lo:
            return 500
        return int(round(float(np.clip((y - lo) / (hi - lo) * 1000, 0, 1000))))

    def _update_controls_from_rect(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> None:
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        self._controls_updating = True
        self._cx_slider.setValue(self._x_to_slider(cx))
        self._cy_slider.setValue(self._y_to_slider(cy))
        self._cx_val_label.setText(f"{cx:.2f}")
        self._cy_val_label.setText(f"{cy:.2f}")
        self._w_spinbox.setValue(max(0.001, w))
        self._h_spinbox.setValue(max(0.001, h))
        self._controls_updating = False

    def _update_rect_from_controls(self, *_) -> None:
        if self._controls_updating or self._rect is None:
            return
        cx = self._slider_to_x(self._cx_slider.value())
        cy = self._slider_to_y(self._cy_slider.value())
        w = self._w_spinbox.value()
        h = self._h_spinbox.value()
        self._rect.set_bounds(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _load_settings(self) -> None:
        if self._settings_path is None or not self._settings_path.exists():
            return
        try:
            with open(self._settings_path) as f:
                settings = json.load(f)
            w = settings.get("velocity_width")
            h = settings.get("pressure_height")
            if w is not None:
                self._w_spinbox.setValue(float(w))
            if h is not None:
                self._h_spinbox.setValue(float(h))
        except Exception as exc:
            logger.warning(
                "_load_settings: could not load %s — %s", self._settings_path, exc
            )

    def _save_settings(self) -> None:
        if self._settings_path is None:
            return
        try:
            settings = {
                "velocity_width": self._w_spinbox.value(),
                "pressure_height": self._h_spinbox.value(),
            }
            self._settings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._settings_path, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as exc:
            logger.warning(
                "_save_settings: could not save %s — %s", self._settings_path, exc
            )

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _load_session(self, new_data: ExplorerData) -> None:
        self._data = new_data
        self._rect = None
        self._cloud = None
        self._actor = None
        raw_max = float(np.max(self._data.iff)) if len(self._data.iff) > 0 else 0.0
        self._max_iff = raw_max if raw_max > 0.0 else 1.0
        self._frame_label.setText(
            f"Frames: {self._data.n_frames} / {self._data.n_frames}"
        )
        self._draw_scatter()
        self._render_3d()
        self._init_filter_rect()
        self._load_settings()

    def _on_session_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._sessions):
            raise ValueError(
                f"RFFeatureSpaceExplorer: session index {index} out of range "
                f"(have {len(self._sessions)} sessions)"
            )
        _, new_data = self._sessions[index]
        self._load_session(new_data)

    # ------------------------------------------------------------------
    # Deferred initialization
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._initialized:
            self._initialized = True
            primary = QApplication.primaryScreen()
            if primary is not None:
                self.move(primary.geometry().topLeft())
            self.showMaximized()
            QTimer.singleShot(0, self._deferred_start)

    def _deferred_start(self) -> None:
        try:
            self._plotter.interactor.Initialize()
        except Exception:
            pass
        sz = self._plotter.interactor.size()
        if sz.width() > 0 and sz.height() > 0:
            self._plotter.render_window.SetSize(sz.width(), sz.height())
        self._draw_scatter()
        self._render_3d()
        self._init_filter_rect()
        self._load_settings()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._save_settings()
        self._plotter.close()
        super().closeEvent(event)
