"""Touch Population Explorer GUI — per-touch scatter + 3D forearm heatmap.

Presents a 2D scatter (one dot per single touch, configurable axes) alongside
an interactive PyVista 3D forearm mesh with a spike-density / IFF heatmap.
A draggable/resizable rectangle on the scatter filters which touches contribute
to the 3D heatmap; gesture-type checkboxes provide additional filtering.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pyvista as pv
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from analysis.receptive_field_mapping.gui.rf_feature_space_explorer import DraggableFilterRect
from analysis.receptive_field_mapping.touch_population_data import PopulationData

logger = logging.getLogger(__name__)

_GESTURE_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

_HEATMAP_MODES = ["Spike density", "Mean IFF (Hz)", "Cumulative IFF (Hz)"]
_MODE_KEYS = ["spike_density", "mean_iff", "cumulative_iff"]


class TouchPopulationExplorer(QMainWindow):
    """QMainWindow showing per-touch scatter + 3D forearm heatmap.

    Parameters
    ----------
    population_data:
        Pre-loaded data for the initial session.
    sessions:
        Optional list of ``(label, PopulationData)`` tuples for the session
        selector.  If ``None``, defaults to ``[("Session 1", population_data)]``.
    title:
        Window title override.
    parent:
        Optional Qt parent widget.
    """

    def __init__(
        self,
        population_data: PopulationData,
        sessions: Optional[List[Tuple[str, PopulationData]]] = None,
        title: str = "Touch Population Explorer",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)

        self._data = population_data
        self._sessions: List[Tuple[str, PopulationData]] = (
            sessions if sessions is not None else [("Session 1", population_data)]
        )
        self._initialized = False
        self._heatmap_mode: str = "spike_density"
        self._rect: Optional[DraggableFilterRect] = None
        self._filter_timer = QTimer(self)
        self._filter_timer.setSingleShot(True)
        self._filter_timer.setInterval(30)
        self._filter_timer.timeout.connect(self._apply_filter_update)
        self._cloud: Optional[pv.PolyData] = None
        self._actor = None
        self._camera_states: dict[int, object] = {}

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

        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(4)

        self._figure = Figure()
        self._figure.patch.set_facecolor("white")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._ax = self._figure.add_subplot(111)
        self._ax.set_facecolor("white")
        self._canvas.setMinimumSize(350, 180)
        bottom_layout.addWidget(self._canvas, stretch=1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.setSpacing(4)
        right_layout.setAlignment(Qt.AlignTop)

        right_layout.addWidget(QLabel("Gesture types:"))
        self._checkboxes: dict[str, QCheckBox] = {}
        self._gesture_colors: dict[str, str] = {}
        unique_types = [str(g) for g in np.unique(self._data.gesture_types)]
        for i, gtype in enumerate(unique_types):
            cb = QCheckBox(gtype)
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_checkbox_changed)
            self._checkboxes[gtype] = cb
            self._gesture_colors[gtype] = _GESTURE_COLORS[i % len(_GESTURE_COLORS)]
            right_layout.addWidget(cb)

        right_layout.addStretch()

        self._touch_count_label = QLabel("N touches shown: —")
        right_layout.addWidget(self._touch_count_label)

        bottom_layout.addWidget(right_panel)

        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter, stretch=1)

    def _build_toolbar(self) -> None:
        from PyQt5.QtWidgets import QToolBar
        toolbar = QToolBar("Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addWidget(QLabel("Session:"))
        self._session_combo = QComboBox()
        for label, _ in self._sessions:
            self._session_combo.addItem(label)
        self._session_combo.currentIndexChanged.connect(self._on_session_changed)
        toolbar.addWidget(self._session_combo)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Heatmap:"))
        self._heatmap_mode_combo = QComboBox()
        for mode_label in _HEATMAP_MODES:
            self._heatmap_mode_combo.addItem(mode_label)
        self._heatmap_mode_combo.currentIndexChanged.connect(self._on_heatmap_mode_changed)
        toolbar.addWidget(self._heatmap_mode_combo)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("X axis:"))
        self._x_axis_combo = QComboBox()
        self._x_axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        toolbar.addWidget(self._x_axis_combo)

        toolbar.addWidget(QLabel("Y axis:"))
        self._y_axis_combo = QComboBox()
        self._y_axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        toolbar.addWidget(self._y_axis_combo)

        self._populate_axis_combos()

    def _populate_axis_combos(self) -> None:
        all_features = list(self._data.feature_names)
        self._x_axis_combo.blockSignals(True)
        self._y_axis_combo.blockSignals(True)
        self._x_axis_combo.clear()
        self._y_axis_combo.clear()
        for name in all_features:
            self._x_axis_combo.addItem(name)
            self._y_axis_combo.addItem(name)
        # Default: X = first feature, Y = second feature (if available)
        x_idx = 0
        y_idx = min(1, len(all_features) - 1) if len(all_features) > 1 else 0
        self._x_axis_combo.setCurrentIndex(x_idx)
        self._y_axis_combo.setCurrentIndex(y_idx)
        self._x_axis_combo.blockSignals(False)
        self._y_axis_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Scatter
    # ------------------------------------------------------------------

    def _x_feature(self) -> str:
        return self._x_axis_combo.currentText()

    def _y_feature(self) -> str:
        return self._y_axis_combo.currentText()

    def _draw_scatter(self) -> None:
        self._ax.cla()

        x_name = self._x_feature()
        y_name = self._y_feature()
        x_data = self._data.get_feature_array(x_name)
        y_data = self._data.get_feature_array(y_name)

        checked_types = {gtype for gtype, cb in self._checkboxes.items() if cb.isChecked()}
        unique_types = [str(g) for g in np.unique(self._data.gesture_types)]

        for gtype in unique_types:
            if gtype not in checked_types:
                continue
            mask = self._data.gesture_types == gtype
            color = self._gesture_colors.get(gtype, "grey")

            spike_mask = mask & self._data.spike_elicited
            no_spike_mask = mask & ~self._data.spike_elicited

            if no_spike_mask.any():
                self._ax.scatter(
                    x_data[no_spike_mask],
                    y_data[no_spike_mask],
                    c=color,
                    alpha=0.3,
                    s=8,
                    rasterized=True,
                )
            if spike_mask.any():
                self._ax.scatter(
                    x_data[spike_mask],
                    y_data[spike_mask],
                    c=color,
                    alpha=1.0,
                    s=8,
                    rasterized=True,
                    label=gtype,
                )

        # Set axis limits from full data (all visible types).
        visible_mask = np.isin(self._data.gesture_types, list(checked_types))
        if visible_mask.any():
            x_vis = x_data[visible_mask]
            y_vis = y_data[visible_mask]
            x_lo = float(np.nanpercentile(x_vis, 1))
            x_hi = float(np.nanpercentile(x_vis, 99))
            y_lo = float(np.nanpercentile(y_vis, 1))
            y_hi = float(np.nanpercentile(y_vis, 99))
            if np.isfinite(x_lo) and np.isfinite(x_hi) and x_lo < x_hi:
                self._ax.set_xlim(x_lo, x_hi)
            if np.isfinite(y_lo) and np.isfinite(y_hi) and y_lo < y_hi:
                self._ax.set_ylim(y_lo, y_hi)

        self._ax.set_xlabel(x_name)
        self._ax.set_ylabel(y_name)
        self._ax.legend(loc="best", markerscale=2, fontsize=8)
        self._figure.tight_layout()

        if self._rect is not None:
            self._ax.add_patch(self._rect._patch)

        self._canvas.draw()

    # ------------------------------------------------------------------
    # 3D render
    # ------------------------------------------------------------------

    def _heatmap_clim(self) -> Tuple[float, float]:
        mode = self._heatmap_mode
        if mode == "mean_iff":
            raw_max = float(np.max(self._data.cp_iff)) if len(self._data.cp_iff) > 0 else 0.0
            max_iff = raw_max if raw_max > 0.0 else 1.0
            return (0.0, max_iff)
        if mode == "cumulative_iff":
            n_verts = len(self._data.forearm_vertices)
            val = np.bincount(self._data.cp_vertex_idx, weights=self._data.cp_iff, minlength=n_verts)
            top = float(np.nanpercentile(val[val > 0], 99)) if np.any(val > 0) else 1.0
            return (0.0, top if top > 0.0 else 1.0)
        # spike_density: ratio in [0, 1]
        return (0.0, 1.0)

    def _on_heatmap_mode_changed(self, index: int) -> None:
        if index < 0 or index >= len(_MODE_KEYS):
            raise ValueError(
                f"TouchPopulationExplorer: heatmap mode index {index} out of range"
            )
        self._heatmap_mode = _MODE_KEYS[index]
        self._render_3d()
        if self._rect is not None:
            self._apply_filter_update()

    def _render_3d(self) -> None:
        session_idx = self._session_combo.currentIndex()
        if self._cloud is not None and session_idx in self._camera_states:
            try:
                self._camera_states[session_idx] = self._plotter.camera.copy()
            except Exception:
                pass

        self._plotter.clear()
        self._plotter.set_background("black")

        vertices = self._data.forearm_vertices
        n_verts = len(vertices)

        cp_mask_all = np.ones(len(self._data.cp_touch_idx), dtype=bool)
        heatmap_val = self._compute_heatmap(cp_mask_all, n_verts)

        self._cloud = pv.PolyData(vertices)
        self._cloud["heatmap"] = heatmap_val

        self._actor = self._plotter.add_mesh(
            self._cloud,
            scalars="heatmap",
            cmap="jet",
            clim=self._heatmap_clim(),
            nan_color=[0.3, 0.3, 0.3],
            show_scalar_bar=True,
            render_points_as_spheres=False,
            point_size=3,
            smooth_shading=True,
            name="forearm",
            copy_mesh=False,
        )

        if session_idx in self._camera_states:
            try:
                self._plotter.camera = self._camera_states[session_idx]
            except Exception:
                self._plotter.view_xy()
        else:
            self._plotter.view_xy()

        self._plotter.render()

    def _compute_heatmap(self, cp_mask: np.ndarray, n_verts: int) -> np.ndarray:
        active_verts = self._data.cp_vertex_idx[cp_mask]
        contact_count = np.bincount(active_verts, minlength=n_verts).astype(float)

        mode = self._heatmap_mode
        if mode == "mean_iff":
            val = (
                np.bincount(active_verts, weights=self._data.cp_iff[cp_mask], minlength=n_verts)
                / np.maximum(contact_count, 1)
            )
        elif mode == "cumulative_iff":
            val = np.bincount(active_verts, weights=self._data.cp_iff[cp_mask], minlength=n_verts)
        elif mode == "spike_density":
            val = (
                np.bincount(active_verts, weights=self._data.cp_spike[cp_mask].astype(float), minlength=n_verts)
                / np.maximum(contact_count, 1)
            )
        else:
            raise ValueError(f"TouchPopulationExplorer: unknown heatmap mode '{mode}'")

        result = val.astype(float)
        result[contact_count == 0] = np.nan
        return result

    # ------------------------------------------------------------------
    # Filter update
    # ------------------------------------------------------------------

    def _on_rect_changed(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> None:
        if self._rect is None:
            return
        self._canvas.draw_idle()
        self._filter_timer.start()

    def _on_checkbox_changed(self, _state: int) -> None:
        self._draw_scatter()
        self._apply_filter_update()

    def _apply_filter_update(self) -> None:
        if self._rect is None or self._cloud is None or self._actor is None:
            return

        x_name = self._x_feature()
        y_name = self._y_feature()
        x_data = self._data.get_feature_array(x_name)
        y_data = self._data.get_feature_array(y_name)

        x_min, x_max, y_min, y_max = self._rect.get_bounds()
        rect_mask = (
            (x_data >= x_min)
            & (x_data <= x_max)
            & (y_data >= y_min)
            & (y_data <= y_max)
        )

        checked_types = {gtype for gtype, cb in self._checkboxes.items() if cb.isChecked()}
        type_mask = np.isin(self._data.gesture_types, list(checked_types))
        touch_mask = rect_mask & type_mask

        n_verts = len(self._data.forearm_vertices)
        cp_mask = np.isin(self._data.cp_touch_idx, np.where(touch_mask)[0])

        heatmap_val = self._compute_heatmap(cp_mask, n_verts)

        self._cloud["heatmap"] = heatmap_val
        self._cloud.Modified()
        self._actor.mapper.scalar_range = self._heatmap_clim()
        self._plotter.render()

        n_shown = int(touch_mask.sum())
        n_total = len(self._data.gesture_types)
        self._touch_count_label.setText(f"N touches shown: {n_shown} / {n_total}")

    # ------------------------------------------------------------------
    # Filter rectangle initialization
    # ------------------------------------------------------------------

    def _init_filter_rect(self) -> None:
        x_name = self._x_feature()
        y_name = self._y_feature()
        x_data = self._data.get_feature_array(x_name)
        y_data = self._data.get_feature_array(y_name)

        p25_x = float(np.nanpercentile(x_data, 25))
        p75_x = float(np.nanpercentile(x_data, 75))
        p25_y = float(np.nanpercentile(y_data, 25))
        p75_y = float(np.nanpercentile(y_data, 75))

        if not (np.isfinite(p25_x) and np.isfinite(p75_x) and np.isfinite(p25_y) and np.isfinite(p75_y)):
            logger.warning(
                "TouchPopulationExplorer: percentiles for '%s'/'%s' are NaN — "
                "skipping filter rectangle initialization",
                x_name,
                y_name,
            )
            return

        if p75_x <= p25_x:
            p75_x = p25_x + 1.0
        if p75_y <= p25_y:
            p75_y = p25_y + 1.0

        self._rect = DraggableFilterRect(
            self._ax,
            x0=p25_x,
            y0=p25_y,
            width=p75_x - p25_x,
            height=p75_y - p25_y,
            on_changed=self._on_rect_changed,
        )
        self._canvas.draw_idle()
        self._apply_filter_update()

    # ------------------------------------------------------------------
    # Axis changes
    # ------------------------------------------------------------------

    def _on_axis_changed(self, _index: int) -> None:
        self._rect = None
        self._draw_scatter()
        self._init_filter_rect()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _load_session(self, session_idx: int, new_data: PopulationData) -> None:
        self._data = new_data
        self._rect = None
        self._cloud = None
        self._actor = None

        # Rebuild gesture checkboxes for the new session's gesture types.
        unique_types = [str(g) for g in np.unique(self._data.gesture_types)]
        old_checked = {gtype for gtype, cb in self._checkboxes.items() if cb.isChecked()}

        # Remove old checkboxes from layout.
        for cb in self._checkboxes.values():
            cb.stateChanged.disconnect()
            cb.setParent(None)
        self._checkboxes.clear()
        self._gesture_colors.clear()

        # The right panel layout is the second widget added to bottom_layout.
        # We need to find the right_layout to add new checkboxes.
        # Walk through the central widget's hierarchy to find the right panel.
        central = self.centralWidget()
        splitter = central.layout().itemAt(0).widget()
        bottom = splitter.widget(1)
        right_panel = bottom.layout().itemAt(1).widget()
        right_layout = right_panel.layout()

        # Remove all items from right_layout except the stretch and touch count label.
        while right_layout.count() > 0:
            item = right_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        right_layout.addWidget(QLabel("Gesture types:"))
        for i, gtype in enumerate(unique_types):
            cb = QCheckBox(gtype)
            cb.setChecked(True)  # always default to checked on session load
            cb.stateChanged.connect(self._on_checkbox_changed)
            self._checkboxes[gtype] = cb
            self._gesture_colors[gtype] = _GESTURE_COLORS[i % len(_GESTURE_COLORS)]
            right_layout.addWidget(cb)

        right_layout.addStretch()
        self._touch_count_label = QLabel("N touches shown: —")
        right_layout.addWidget(self._touch_count_label)

        # Repopulate axis combos (features may differ between sessions).
        self._populate_axis_combos()

        if not self._data.feature_names:
            # No Stage 3 features for this session — show informative message,
            # still render the 3D heatmap with all contact points.
            self._ax.cla()
            self._ax.set_axis_off()
            self._ax.text(
                0.5,
                0.5,
                "No touch features — run feature extraction first",
                ha="center",
                va="center",
                transform=self._ax.transAxes,
                fontsize=11,
                color="grey",
            )
            self._figure.tight_layout()
            self._canvas.draw()
            self._render_3d()
            return

        self._draw_scatter()
        self._render_3d()
        self._init_filter_rect()

    def _on_session_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._sessions):
            raise ValueError(
                f"TouchPopulationExplorer: session index {index} out of range "
                f"(have {len(self._sessions)} sessions)"
            )
        _, new_data = self._sessions[index]
        self._load_session(index, new_data)

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

        if not self._data.feature_names:
            # No Stage 3 features available — show informative message in scatter area,
            # but still render the 3D heatmap with all contact points.
            self._ax.cla()
            self._ax.set_axis_off()
            self._ax.text(
                0.5,
                0.5,
                "No touch features — run feature extraction first",
                ha="center",
                va="center",
                transform=self._ax.transAxes,
                fontsize=11,
                color="grey",
            )
            self._figure.tight_layout()
            self._canvas.draw()
            self._render_3d()
            return

        self._draw_scatter()
        self._render_3d()
        self._init_filter_rect()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._plotter.close()
        super().closeEvent(event)
