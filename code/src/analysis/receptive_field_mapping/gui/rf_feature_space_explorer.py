"""RF Feature-Space Explorer GUI — scatter + 3D forearm heatmap with live filter.

Presents a 2D scatter (pressure × velocity_signed, colored by gesture type)
alongside an interactive PyVista 3D forearm mesh with a spike-density heatmap.
A draggable/resizable rectangle on the scatter filters which frames contribute
to the 3D heatmap; gesture-type checkboxes provide additional filtering.
"""

import logging
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
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from analysis.receptive_field_mapping.rf_explorer_data import ExplorerData

logger = logging.getLogger(__name__)

_GESTURE_TYPES = ["tap", "stroke_proximal", "stroke_distal", "stroke_unknown"]

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
            edgecolor="white",
            facecolor="white",
            alpha=0.15,
            zorder=5,
        )
        ax.add_patch(self._patch)

        canvas = ax.figure.canvas
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("button_release_event", self._on_release)

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
        mode = self._hit_test(event.xdata, event.ydata)
        if mode is None:
            return
        self._drag_mode = mode
        self._press_xy = (event.xdata, event.ydata)
        x0 = self._patch.get_x()
        y0 = self._patch.get_y()
        self._rect_at_press = (x0, y0, self._patch.get_width(), self._patch.get_height())

    def _on_motion(self, event) -> None:
        if self._drag_mode is None or event.inaxes is not self._ax:
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
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)

        self._data = explorer_data
        self._sessions: List[Tuple[str, ExplorerData]] = (
            sessions if sessions is not None else [("Session 1", explorer_data)]
        )
        self._initialized = False
        self._rect: Optional[DraggableFilterRect] = None
        self._bg = None
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(30)
        self._filter_timer.timeout.connect(self._apply_filter_update)
        self._cloud: Optional[pv.PolyData] = None

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

        # Overlay container: 3D interactor and scatter canvas share cell (0, 0)
        overlay_container = QWidget()
        grid = QGridLayout(overlay_container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        self._plotter = QtInteractor(overlay_container)
        grid.addWidget(self._plotter.interactor, 0, 0)

        self._figure = Figure()
        self._figure.patch.set_alpha(0.0)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)
        self._ax.set_facecolor((0.1, 0.1, 0.1, 0.7))
        self._canvas.setFixedSize(350, 280)
        self._canvas.setAttribute(Qt.WA_TranslucentBackground, True)
        self._canvas.setStyleSheet("background: transparent;")
        grid.addWidget(self._canvas, 0, 0, Qt.AlignBottom | Qt.AlignLeft)

        root.addWidget(overlay_container, stretch=1)

        bottom_bar = self._build_bottom_bar()
        root.addLayout(bottom_bar)

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

    def _build_bottom_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 2)

        self._checkboxes: dict[str, QCheckBox] = {}
        for gtype in _GESTURE_TYPES:
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

        unique_types = np.unique(self._data.gesture_types)

        for gtype in _GESTURE_TYPES:
            if gtype not in unique_types:
                continue
            mask = self._data.gesture_types == gtype
            color = _GESTURE_COLORS.get(gtype, "grey")
            self._ax.scatter(
                self._data.pressure[mask],
                self._data.velocity_signed[mask],
                c=color,
                alpha=0.3,
                s=3,
                rasterized=True,
                label=gtype,
            )

        other_types = [t for t in unique_types if t not in _GESTURE_TYPES]
        for gtype in other_types:
            mask = self._data.gesture_types == gtype
            self._ax.scatter(
                self._data.pressure[mask],
                self._data.velocity_signed[mask],
                alpha=0.3,
                s=3,
                rasterized=True,
                label=gtype,
            )

        x_lo, x_hi = float(np.percentile(self._data.pressure, 1)), float(
            np.percentile(self._data.pressure, 99)
        )
        y_lo, y_hi = float(np.percentile(self._data.velocity_signed, 1)), float(
            np.percentile(self._data.velocity_signed, 99)
        )
        self._ax.set_xlim(x_lo, x_hi)
        self._ax.set_ylim(y_lo, y_hi)

        self._ax.set_xlabel("Pressure")
        self._ax.set_ylabel("Hand Velocity (signed)")
        self._ax.legend(loc="best", markerscale=3, fontsize=8)
        self._figure.tight_layout()
        self._canvas.draw()

        self._bg = self._canvas.copy_from_bbox(self._ax.bbox)
        self._canvas.mpl_connect("draw_event", self._on_draw_event)

    def _on_draw_event(self, event) -> None:
        self._bg = self._canvas.copy_from_bbox(self._ax.bbox)

    # ------------------------------------------------------------------
    # 3D render
    # ------------------------------------------------------------------

    def _render_3d(self) -> None:
        self._plotter.clear()

        vertices = self._data.session_data.forearm_vertices
        n_verts = len(vertices)

        spike_density = np.bincount(
            self._data.frame_vertex_idx,
            weights=self._data.spikes.astype(float),
            minlength=n_verts,
        )

        self._cloud = pv.PolyData(vertices)
        self._cloud["spike_density"] = spike_density

        self._plotter.add_mesh(
            self._cloud,
            scalars="spike_density",
            cmap="hot",
            show_scalar_bar=True,
            render_points_as_spheres=True,
            point_size=5,
            name="forearm",
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
        canvas = self._canvas
        if self._bg is not None:
            canvas.restore_region(self._bg)
        self._ax.draw_artist(self._rect._patch)
        canvas.blit(self._ax.bbox)
        self._filter_timer.start()

    def _on_checkbox_changed(self, _state: int) -> None:
        self._apply_filter_update()

    def _apply_filter_update(self) -> None:
        if self._rect is None or self._cloud is None:
            return

        x_min, x_max, y_min, y_max = self._rect.get_bounds()
        rect_mask = (
            (self._data.pressure >= x_min)
            & (self._data.pressure <= x_max)
            & (self._data.velocity_signed >= y_min)
            & (self._data.velocity_signed <= y_max)
        )

        checked_types: set[str] = set()
        for gt, cb in self._checkboxes.items():
            cb.blockSignals(True)
            if cb.isChecked():
                checked_types.add(gt)
            cb.blockSignals(False)

        type_mask = np.isin(self._data.gesture_types, list(checked_types))
        mask = rect_mask & type_mask

        n_verts = len(self._data.session_data.forearm_vertices)
        vertex_spikes = np.bincount(
            self._data.frame_vertex_idx[mask],
            weights=self._data.spikes[mask].astype(float),
            minlength=n_verts,
        )

        cloud = pv.PolyData(self._data.session_data.forearm_vertices)
        cloud["spike_density"] = vertex_spikes
        self._plotter.add_mesh(
            cloud,
            scalars="spike_density",
            cmap="hot",
            show_scalar_bar=True,
            render_points_as_spheres=True,
            point_size=5,
            name="forearm",
        )
        self._cloud = cloud
        self._plotter.render()
        self._frame_label.setText(f"Frames: {mask.sum()} / {self._data.n_frames}")

    # ------------------------------------------------------------------
    # Draggable rectangle initialization
    # ------------------------------------------------------------------

    def _init_filter_rect(self) -> None:
        p25_x = float(np.percentile(self._data.pressure, 25))
        p75_x = float(np.percentile(self._data.pressure, 75))
        p25_y = float(np.percentile(self._data.velocity_signed, 25))
        p75_y = float(np.percentile(self._data.velocity_signed, 75))

        self._rect = DraggableFilterRect(
            self._ax,
            x0=p25_x,
            y0=p25_y,
            width=p75_x - p25_x,
            height=p75_y - p25_y,
            on_changed=self._on_rect_changed,
        )

        if self._bg is not None:
            self._canvas.restore_region(self._bg)
        self._ax.draw_artist(self._rect._patch)
        self._canvas.blit(self._ax.bbox)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _load_session(self, new_data: ExplorerData) -> None:
        self._data = new_data
        self._rect = None
        self._cloud = None
        self._frame_label.setText(
            f"Frames: {self._data.n_frames} / {self._data.n_frames}"
        )
        self._draw_scatter()
        self._render_3d()
        self._init_filter_rect()

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

    def closeEvent(self, event) -> None:  # noqa: N802
        self._plotter.close()
        super().closeEvent(event)
