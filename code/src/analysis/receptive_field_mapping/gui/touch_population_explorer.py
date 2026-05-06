"""Touch Population Explorer GUI — per-touch scatter + 3D forearm heatmap.

Presents a 2D scatter (one dot per single touch, configurable axes) alongside
an interactive PyVista 3D forearm mesh with a spike-density / IFF heatmap.
A draggable/resizable rectangle on the scatter filters which touches contribute
to the 3D heatmap; gesture-type checkboxes provide additional filtering.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QCompleter,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from analysis.receptive_field_mapping.gui.rf_feature_space_explorer import DraggableFilterRect
from analysis.receptive_field_mapping.touch_population_data import PopulationData, PopulationRFData

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

# RF mode labels/keys are built dynamically from neuron_mode; these are the static ones above.

_DEFAULT_X_FEATURE = "hand_velocity_amplitude_mean_during_iff"
_DEFAULT_Y_FEATURE = "pressure_mean_during_iff"

_FEATURE_UNITS: dict[str, str] = {
    # velocity (mm/s)
    "hand_velocity_x": "mm/s",
    "hand_velocity_y": "mm/s",
    "hand_velocity_z": "mm/s",
    "hand_velocity_amplitude": "mm/s",
    "hand_velocity_signed": "mm/s",
    "velocity": "mm/s",
    "speed": "mm/s",
    # acceleration (mm/s²)
    "hand_acceleration_x": "mm/s²",
    "hand_acceleration_y": "mm/s²",
    "hand_acceleration_z": "mm/s²",
    # geometry
    "contact_area": "mm²",
    "area": "mm²",
    "contact_depth": "mm",
    "depth": "mm",
    "distance": "mm",
    "radius": "mm",
    "perimeter": "mm",
    # mechanics of solids
    "mos_stress_kpa": "kPa",
    "mos_strain_rate": "1/s",
    "mos_elastic_energy_mj": "mJ",
    "mos_impulse_mns": "mN·s",
    "mos_strain": "",
    # pressure / neural
    "pressure": "N/mm²",
    "Nerve_freq": "Hz",
    "force": "N",
    # shape descriptors (dimensionless)
    "angle": "°",
    "curvature": "1/mm",
    "eccentricity": "",
    "aspect_ratio": "",
    "duration": "s",
}

_AGG_SUFFIXES: list[str] = [
    # IFF-windowed (longest first to avoid partial matches)
    "_mean_during_iff",
    "_mean_before_iff",
    # standard statistical
    "_skewness",
    "_median",
    "_initial",
    "_range",
    "_first",
    "_final",
    "_mean",
    "_last",
    "_kurt",
    "_skew",
    "_std",
    "_sum",
    "_max",
    "_min",
    "_iqr",
    "_cv",
]


def _feature_unit(name: str) -> str:
    base = name
    for suffix in _AGG_SUFFIXES:
        if name.endswith(suffix):
            base = name[: -len(suffix)]
            break
    return _FEATURE_UNITS.get(base, "")


def _feature_display(name: str) -> str:
    unit = _feature_unit(name)
    return f"{name} ({unit})" if unit else name


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
        rf_sessions: Optional[List[Optional[PopulationRFData]]] = None,
        title: str = "Touch Population Explorer",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)

        self._data = population_data
        self._sessions: List[Tuple[str, PopulationData]] = (
            sessions if sessions is not None else [("Session 1", population_data)]
        )
        self._rf_sessions: List[Optional[PopulationRFData]] = (
            rf_sessions if rf_sessions is not None else []
        )
        # Initialise RF data for the first session (index 0).
        self._rf_data: Optional[PopulationRFData] = (
            self._rf_sessions[0] if self._rf_sessions else None
        )
        self._initialized = False
        self._heatmap_mode: str = "spike_density"
        self._rect: Optional[DraggableFilterRect] = None
        self._controls_updating: bool = False
        self._single_touch_mode: bool = False
        self._vertex_threshold: int = 1
        self._threshold_ratio_mode: bool = True
        self._threshold_ratio_value: int = 50
        self._n_filtered: int = 0
        self._selected_touch_idx: int | None = None
        self._single_touch_cid: int | None = None
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

        scatter_container = QWidget()
        scatter_layout = QVBoxLayout(scatter_container)
        scatter_layout.setContentsMargins(0, 0, 0, 0)
        scatter_layout.setSpacing(2)

        self._figure = Figure()
        self._figure.patch.set_facecolor("white")
        self._scatter_canvas = FigureCanvasQTAgg(self._figure)
        self._canvas = self._scatter_canvas
        self._ax = self._figure.add_subplot(111)
        self._ax.set_facecolor("white")
        self._scatter_canvas.setMinimumSize(350, 180)
        scatter_layout.addWidget(self._scatter_canvas, stretch=1)

        scatter_layout.addLayout(self._build_rect_controls())

        bottom_layout.addWidget(scatter_container, stretch=1)

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

        self._threshold_row_layout = QHBoxLayout()
        self._threshold_row_layout.setSpacing(4)
        self._threshold_row_layout.addWidget(QLabel("Min overlaps:"))
        self._threshold_spinbox = QSpinBox()
        self._threshold_spinbox.setMinimum(1)
        self._threshold_spinbox.setMaximum(100)
        self._threshold_spinbox.setValue(50)
        self._threshold_spinbox.valueChanged.connect(self._on_threshold_changed)
        self._threshold_row_layout.addWidget(self._threshold_spinbox)
        self._threshold_mode_btn = QPushButton("%")
        self._threshold_mode_btn.setCheckable(True)
        self._threshold_mode_btn.setChecked(True)
        self._threshold_mode_btn.setFixedWidth(28)
        self._threshold_mode_btn.toggled.connect(self._on_threshold_mode_toggled)
        self._threshold_row_layout.addWidget(self._threshold_mode_btn)
        self._threshold_suffix_label = QLabel("% of filtered")
        self._threshold_row_layout.addWidget(self._threshold_suffix_label)
        right_layout.addLayout(self._threshold_row_layout)

        right_layout.addStretch()

        self._touch_count_label = QLabel("N touches shown: —")
        right_layout.addWidget(self._touch_count_label)

        bottom_layout.addWidget(right_panel)

        splitter.addWidget(bottom)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter, stretch=1)

        # Sync the heatmap mode dropdown for the initial session's RF data.
        self._sync_heatmap_modes()

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
        for mode_label, mode_key in zip(_HEATMAP_MODES, _MODE_KEYS):
            self._heatmap_mode_combo.addItem(mode_label, mode_key)
        self._heatmap_mode_combo.currentIndexChanged.connect(self._on_heatmap_mode_changed)
        toolbar.addWidget(self._heatmap_mode_combo)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("X axis:"))
        self._x_axis_combo = self._make_searchable_combo()
        self._x_axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        toolbar.addWidget(self._x_axis_combo)

        toolbar.addWidget(QLabel("Y axis:"))
        self._y_axis_combo = self._make_searchable_combo()
        self._y_axis_combo.currentIndexChanged.connect(self._on_axis_changed)
        toolbar.addWidget(self._y_axis_combo)

        toolbar.addSeparator()
        self._single_touch_btn = QPushButton("Single touch")
        self._single_touch_btn.setCheckable(True)
        self._single_touch_btn.toggled.connect(self._set_single_touch_mode)
        toolbar.addWidget(self._single_touch_btn)

        self._export_touches_btn = QPushButton("Export touches")
        self._export_touches_btn.clicked.connect(self._on_export_touches_clicked)
        toolbar.addWidget(self._export_touches_btn)

        self._populate_axis_combos()

    @staticmethod
    def _make_searchable_combo() -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)
        completer = combo.completer()
        completer.setCompletionMode(QCompleter.PopupCompletion)
        completer.setFilterMode(Qt.MatchContains)
        return combo

    def _populate_axis_combos(self) -> None:
        all_features = list(self._data.feature_names)
        self._x_axis_combo.blockSignals(True)
        self._y_axis_combo.blockSignals(True)
        self._x_axis_combo.clear()
        self._y_axis_combo.clear()
        for name in all_features:
            self._x_axis_combo.addItem(name)
            self._y_axis_combo.addItem(name)
        if _DEFAULT_X_FEATURE in all_features:
            x_idx = all_features.index(_DEFAULT_X_FEATURE)
        else:
            x_idx = 0
        if _DEFAULT_Y_FEATURE in all_features:
            y_idx = all_features.index(_DEFAULT_Y_FEATURE)
        else:
            y_idx = min(1, len(all_features) - 1) if len(all_features) > 1 else 0
        self._x_axis_combo.setCurrentIndex(x_idx)
        self._y_axis_combo.setCurrentIndex(y_idx)
        self._x_axis_combo.blockSignals(False)
        self._y_axis_combo.blockSignals(False)

    def _build_rect_controls(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 2, 4, 2)
        bar.setSpacing(6)

        bar.addWidget(QLabel("W:"))
        self._w_spinbox = QDoubleSpinBox()
        self._w_spinbox.setRange(0.001, 1e9)
        self._w_spinbox.setSingleStep(0.1)
        self._w_spinbox.setDecimals(4)
        self._w_spinbox.valueChanged.connect(self._update_rect_from_controls)
        bar.addWidget(self._w_spinbox)

        bar.addWidget(QLabel("H:"))
        self._h_spinbox = QDoubleSpinBox()
        self._h_spinbox.setRange(0.001, 1e9)
        self._h_spinbox.setSingleStep(0.1)
        self._h_spinbox.setDecimals(4)
        self._h_spinbox.valueChanged.connect(self._update_rect_from_controls)
        bar.addWidget(self._h_spinbox)

        bar.addStretch()
        return bar

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
                margin_x = (x_hi - x_lo) * 0.1
                self._ax.set_xlim(x_lo - margin_x, x_hi + margin_x)
            if np.isfinite(y_lo) and np.isfinite(y_hi) and y_lo < y_hi:
                margin_y = (y_hi - y_lo) * 0.1
                self._ax.set_ylim(y_lo - margin_y, y_hi + margin_y)

        self._ax.set_xlabel(x_name)
        self._ax.set_ylabel(y_name)
        self._ax.legend(loc="best", markerscale=2, fontsize=8)
        self._figure.tight_layout()

        if self._rect is not None:
            self._ax.add_patch(self._rect._patch)

        if self._single_touch_mode and self._selected_touch_idx is not None:
            sel_idx = self._selected_touch_idx
            if str(self._data.gesture_types[sel_idx]) in checked_types:
                self._ax.scatter(
                    [x_data[sel_idx]],
                    [y_data[sel_idx]],
                    s=80,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=2,
                    zorder=10,
                )

        self._canvas.draw()

    # ------------------------------------------------------------------
    # 3D render
    # ------------------------------------------------------------------

    def _heatmap_clim(self) -> Tuple[float, float]:
        mode = self._heatmap_mode
        if self._rf_data is not None and mode == f"rf_mean_{self._rf_data.neuron_mode}":
            return (0.0, self._rf_data.session_max_value)
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
        if index < 0 or index >= self._heatmap_mode_combo.count():
            raise ValueError(
                f"TouchPopulationExplorer: heatmap mode index {index} out of range"
            )
        # Derive the mode key from the combo item's user data (set by _sync_heatmap_modes),
        # falling back to the static _MODE_KEYS for built-in modes.
        item_data = self._heatmap_mode_combo.itemData(index)
        if item_data is not None:
            self._heatmap_mode = item_data
        elif index < len(_MODE_KEYS):
            self._heatmap_mode = _MODE_KEYS[index]
        else:
            raise ValueError(
                f"TouchPopulationExplorer: heatmap mode index {index} has no key"
            )
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

        if self._rf_data is not None and self._heatmap_mode == f"rf_mean_{self._rf_data.neuron_mode}":
            n_touches = len(self._data.gesture_types)
            heatmap_val = self._compute_rf_heatmap(list(range(n_touches)), n_verts)
            if not self._single_touch_mode:
                cp_mask_all = np.ones(len(self._data.cp_touch_idx), dtype=bool)
                unique_touch_count = self._compute_unique_touch_count(cp_mask_all, n_verts)
                heatmap_val = self._apply_vertex_threshold(heatmap_val, unique_touch_count)
        else:
            cp_mask_all = np.ones(len(self._data.cp_touch_idx), dtype=bool)
            heatmap_val = self._compute_heatmap(cp_mask_all, n_verts)
            if not self._single_touch_mode:
                unique_touch_count = self._compute_unique_touch_count(cp_mask_all, n_verts)
                heatmap_val = self._apply_vertex_threshold(heatmap_val, unique_touch_count)

        self._cloud = pv.PolyData(vertices)
        self._cloud["heatmap"] = heatmap_val

        self._actor = self._plotter.add_mesh(
            self._cloud,
            scalars="heatmap",
            cmap="jet",
            clim=self._heatmap_clim(),
            nan_color=[0.3, 0.3, 0.3],
            below_color=[0.75, 0.75, 0.75],
            show_scalar_bar=True,
            scalar_bar_args={"title": "", "n_labels": 5, "color": "white", "fmt": "%.3g", "below_label": "below threshold"},
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

        if not self._single_touch_mode:
            n_total = len(self._data.gesture_types)
            self._update_threshold_range(n_total)

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

    def _compute_rf_heatmap(self, touch_indices: list, n_verts: int) -> np.ndarray:
        """Compute mean RF heatmap across *touch_indices* using pre-loaded RF maps.

        For each vertex contacted by at least one selected touch, computes the
        mean of the per-touch RF values (mean neuron response per vertex index).
        Vertices not contacted by any selected touch are set to NaN.

        Parameters
        ----------
        touch_indices:
            List of integer touch indices (into ``self._rf_data`` lists).
        n_verts:
            Total number of forearm mesh vertices.
        """
        if self._rf_data is None:
            raise ValueError(
                "_compute_rf_heatmap: called with self._rf_data=None — RF mode is not available"
            )
        result = np.zeros(n_verts, dtype=np.float64)
        count = np.zeros(n_verts, dtype=np.int64)
        for idx in touch_indices:
            verts = self._rf_data.rf_vertex_indices[idx]
            vals = self._rf_data.rf_values[idx]
            if len(verts) == 0:
                continue
            np.add.at(result, verts, vals)
            np.add.at(count, verts, 1)
        nonzero = count > 0
        result[nonzero] /= count[nonzero]
        result[~nonzero] = np.nan
        return result

    def _sync_heatmap_modes(self) -> None:
        """Add or remove the RF heatmap mode item from the dropdown.

        Called after ``self._rf_data`` changes (session load or session switch).
        When RF data is available, adds the RF mode item if not already present.
        When RF data is unavailable, removes the RF mode item if present and
        falls back to ``"spike_density"`` if the current mode was an RF mode.
        """
        combo = self._heatmap_mode_combo
        combo.blockSignals(True)

        if self._rf_data is not None:
            rf_key = f"rf_mean_{self._rf_data.neuron_mode}"
            rf_label = f"RF Mean {self._rf_data.neuron_mode.upper()}"

            # Check whether the RF mode item is already in the combo (by key).
            rf_idx = None
            for i in range(combo.count()):
                if combo.itemData(i) == rf_key:
                    rf_idx = i
                    break

            if rf_idx is None:
                # Remove any stale RF mode items (different neuron_mode from a prior session).
                indices_to_remove = []
                for i in range(combo.count()):
                    key = combo.itemData(i)
                    if isinstance(key, str) and key.startswith("rf_mean_"):
                        indices_to_remove.append(i)
                for i in reversed(indices_to_remove):
                    combo.removeItem(i)
                combo.addItem(rf_label, rf_key)
                # Auto-select the RF mode if the current selection is the spike_density default.
                if combo.currentIndex() == 0 and (
                    combo.itemData(0) == "spike_density" or combo.itemData(0) is None
                ):
                    new_rf_idx = combo.count() - 1
                    combo.setCurrentIndex(new_rf_idx)
                    self._heatmap_mode = rf_key
        else:
            # Remove RF mode item(s) if present.
            indices_to_remove = []
            for i in range(combo.count()):
                key = combo.itemData(i)
                if isinstance(key, str) and key.startswith("rf_mean_"):
                    indices_to_remove.append(i)
            for i in reversed(indices_to_remove):
                combo.removeItem(i)

            # If the current mode was an RF mode, fall back to spike_density.
            if self._heatmap_mode.startswith("rf_mean_"):
                self._heatmap_mode = "spike_density"
                for i in range(combo.count()):
                    if combo.itemData(i) == "spike_density":
                        combo.setCurrentIndex(i)
                        break

        combo.blockSignals(False)

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

    def _update_controls_from_rect(
        self, x_min: float, x_max: float, y_min: float, y_max: float
    ) -> None:
        self._controls_updating = True
        self._w_spinbox.setValue(x_max - x_min)
        self._h_spinbox.setValue(y_max - y_min)
        self._controls_updating = False

    def _update_rect_from_controls(self, *_) -> None:
        if self._controls_updating or self._rect is None:
            return
        x_min, x_max, y_min, y_max = self._rect.get_bounds()
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = self._w_spinbox.value()
        h = self._h_spinbox.value()
        self._rect.set_bounds(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
        self._scatter_canvas.draw_idle()

    def _on_checkbox_changed(self, _state: int) -> None:
        self._draw_scatter()
        if self._single_touch_mode:
            self._apply_single_touch_display()
        else:
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
        active_touch_indices = list(np.where(touch_mask)[0])
        cp_mask = np.isin(self._data.cp_touch_idx, active_touch_indices)

        if self._rf_data is not None and self._heatmap_mode == f"rf_mean_{self._rf_data.neuron_mode}":
            heatmap_val = self._compute_rf_heatmap(active_touch_indices, n_verts)
        else:
            heatmap_val = self._compute_heatmap(cp_mask, n_verts)

        n_shown = int(touch_mask.sum())
        unique_touch_count = self._compute_unique_touch_count(cp_mask, n_verts)
        self._update_threshold_range(n_shown)
        heatmap_val = self._apply_vertex_threshold(heatmap_val, unique_touch_count)

        self._cloud["heatmap"] = heatmap_val
        self._cloud.Modified()
        self._actor.mapper.scalar_range = self._heatmap_clim()
        self._plotter.render()

        n_total = len(self._data.gesture_types)
        self._touch_count_label.setText(f"N touches shown: {n_shown} / {n_total}")

    def _compute_unique_touch_count(self, cp_mask: np.ndarray, n_verts: int) -> np.ndarray:
        """Return per-vertex count of unique touches in the masked contact points.

        Parameters
        ----------
        cp_mask:
            Boolean mask of shape ``(C,)`` over the full contact-point arrays
            (``self._data.cp_vertex_idx``, ``self._data.cp_touch_idx``).
        n_verts:
            Total number of forearm mesh vertices.

        Returns
        -------
        np.ndarray
            Shape ``(n_verts,)`` int64 — number of distinct touches that
            contacted each vertex among the masked contact points.
        """
        vertex_idx = self._data.cp_vertex_idx[cp_mask]
        touch_idx = self._data.cp_touch_idx[cp_mask]

        if len(vertex_idx) == 0:
            return np.zeros(n_verts, dtype=np.int64)

        # Encode (vertex, touch) pairs as a single integer for fast deduplication.
        max_touch = int(touch_idx.max()) + 1
        key = vertex_idx * max_touch + touch_idx
        unique_keys = np.unique(key)

        # Decode vertex indices from deduplicated keys.
        unique_verts = unique_keys // max_touch

        return np.bincount(unique_verts, minlength=n_verts).astype(np.int64)

    def _effective_threshold(self, n_filtered: int) -> int:
        """Return the effective integer threshold given the current mode and *n_filtered*."""
        if self._threshold_ratio_mode:
            return max(1, round(self._threshold_ratio_value / 100 * n_filtered))
        return self._vertex_threshold

    def _apply_vertex_threshold(
        self, heatmap_val: np.ndarray, unique_touch_count: np.ndarray
    ) -> np.ndarray:
        """Return a copy of *heatmap_val* with below-threshold contacted vertices set to -1.0.

        Only contacted vertices (those with ``heatmap_val != 0.0``) that have
        fewer unique touches than ``self._vertex_threshold`` are set to -1.0.
        Uncontacted vertices (``heatmap_val == 0.0`` or NaN) are left unchanged
        so that the dark-grey uncontacted colour is preserved.

        Parameters
        ----------
        heatmap_val:
            Float array of shape ``(n_verts,)`` produced by ``_compute_heatmap``
            or ``_compute_rf_heatmap``.
        unique_touch_count:
            Integer array of shape ``(n_verts,)`` produced by
            ``_compute_unique_touch_count``.

        Returns
        -------
        np.ndarray
            Modified copy of *heatmap_val*.
        """
        result = heatmap_val.copy()
        # A vertex is "contacted" when it has a positive unique-touch count.
        contacted = unique_touch_count > 0
        below_threshold = contacted & (unique_touch_count < self._effective_threshold(self._n_filtered))
        result[below_threshold] = -1.0
        return result

    def _update_threshold_range(self, n_filtered_touches: int) -> None:
        """Update the threshold spinbox range and suffix label to reflect *n_filtered_touches*.

        In ratio mode, the spinbox range stays 1–100 and the value is not changed;
        only the suffix label is updated. In absolute mode, the spinbox maximum is
        set to *n_filtered_touches* and the current value is clamped if it exceeds
        the new maximum.

        Uses ``blockSignals`` guards to avoid re-entering ``_on_threshold_changed``.

        Parameters
        ----------
        n_filtered_touches:
            Number of touches currently passing all filters.
        """
        self._n_filtered = n_filtered_touches
        spinbox = self._threshold_spinbox
        spinbox.blockSignals(True)
        try:
            if self._threshold_ratio_mode:
                spinbox.setEnabled(n_filtered_touches > 0)
                spinbox.setMinimum(1)
                spinbox.setMaximum(100)
                self._threshold_suffix_label.setText("% of filtered")
            else:
                if n_filtered_touches == 0:
                    spinbox.setEnabled(False)
                    spinbox.setMinimum(1)
                    spinbox.setMaximum(1)
                    spinbox.setValue(1)
                    self._threshold_suffix_label.setText("/ 1")
                else:
                    spinbox.setEnabled(True)
                    new_max = max(1, n_filtered_touches)
                    spinbox.setMinimum(1)
                    spinbox.setMaximum(new_max)
                    effective = self._effective_threshold(n_filtered_touches)
                    if effective > new_max:
                        spinbox.setValue(new_max)
                        self._vertex_threshold = new_max
                    self._threshold_suffix_label.setText(f"/ {new_max}")
        finally:
            spinbox.blockSignals(False)

    def _on_threshold_changed(self, value: int) -> None:
        """Handle threshold spinbox value changes.

        In ratio mode stores to ``_threshold_ratio_value``; in absolute mode
        stores to ``_vertex_threshold``. Then triggers a heatmap update.

        Parameters
        ----------
        value:
            New spinbox integer value.
        """
        if self._threshold_ratio_mode:
            self._threshold_ratio_value = value
        else:
            self._vertex_threshold = value
        self._apply_filter_update()

    def _on_threshold_mode_toggled(self, checked: bool) -> None:
        """Toggle between ratio (checked=True) and absolute (checked=False) threshold mode.

        Converts the current spinbox value to the equivalent in the new mode so
        that the effective threshold is preserved across the toggle.

        Parameters
        ----------
        checked:
            ``True`` means ratio mode (spinbox shows percentage 1–100).
            ``False`` means absolute mode (spinbox shows raw overlap count).
        """
        spinbox = self._threshold_spinbox
        spinbox.blockSignals(True)
        try:
            if checked:
                ratio = round(self._vertex_threshold / max(1, self._n_filtered) * 100)
                ratio = max(1, min(100, ratio))
                self._threshold_ratio_mode = True
                self._threshold_ratio_value = ratio
                spinbox.setMinimum(1)
                spinbox.setMaximum(100)
                spinbox.setValue(ratio)
                self._threshold_suffix_label.setText("% of filtered")
            else:
                abs_val = max(1, round(self._threshold_ratio_value / 100 * self._n_filtered))
                self._threshold_ratio_mode = False
                self._vertex_threshold = abs_val
                spinbox.setMinimum(1)
                spinbox.setMaximum(max(1, self._n_filtered))
                spinbox.setValue(abs_val)
                self._threshold_suffix_label.setText(f"/ {self._n_filtered}")
        finally:
            spinbox.blockSignals(False)
        self._apply_filter_update()

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
        self._update_controls_from_rect(p25_x, p75_x, p25_y, p75_y)
        self._canvas.draw_idle()
        self._apply_filter_update()

    # ------------------------------------------------------------------
    # Single-touch mode
    # ------------------------------------------------------------------

    def _set_single_touch_mode(self, enabled: bool) -> None:
        if enabled:
            if self._rect is not None:
                self._rect.disconnect()
                self._rect = None
            self._single_touch_cid = self._scatter_canvas.figure.canvas.mpl_connect(
                "button_press_event", self._on_scatter_click_single_touch
            )
            self._single_touch_mode = True
            self._w_spinbox.setVisible(False)
            self._h_spinbox.setVisible(False)
            # Hide threshold row widgets in single-touch mode.
            for i in range(self._threshold_row_layout.count()):
                item = self._threshold_row_layout.itemAt(i)
                if item is not None and item.widget() is not None:
                    item.widget().hide()
            self._selected_touch_idx = None
            self._draw_scatter()
        else:
            if self._single_touch_cid is not None:
                self._scatter_canvas.figure.canvas.mpl_disconnect(self._single_touch_cid)
                self._single_touch_cid = None
            self._single_touch_mode = False
            self._selected_touch_idx = None
            self._w_spinbox.setVisible(True)
            self._h_spinbox.setVisible(True)
            # Restore threshold row widgets when returning to population mode.
            for i in range(self._threshold_row_layout.count()):
                item = self._threshold_row_layout.itemAt(i)
                if item is not None and item.widget() is not None:
                    item.widget().show()
            self._init_filter_rect()
            self._on_rect_changed(*self._rect.get_bounds()) if self._rect is not None else None

    def _on_scatter_click_single_touch(self, event) -> None:
        if event.inaxes is None:
            return

        x_name = self._x_feature()
        y_name = self._y_feature()
        x_data = self._data.get_feature_array(x_name)
        y_data = self._data.get_feature_array(y_name)

        checked_types = {gtype for gtype, cb in self._checkboxes.items() if cb.isChecked()}
        visible_mask = np.isin(self._data.gesture_types, list(checked_types))
        visible_indices = np.where(visible_mask)[0]

        if len(visible_indices) == 0:
            return

        x_lim = self._ax.get_xlim()
        y_lim = self._ax.get_ylim()
        x_range = x_lim[1] - x_lim[0]
        y_range = y_lim[1] - y_lim[0]
        if x_range == 0.0:
            x_range = 1.0
        if y_range == 0.0:
            y_range = 1.0

        dx = (x_data[visible_indices] - event.xdata) / x_range
        dy = (y_data[visible_indices] - event.ydata) / y_range
        nearest_local = int(np.argmin(dx**2 + dy**2))
        self._selected_touch_idx = int(visible_indices[nearest_local])

        self._draw_scatter()
        self._apply_single_touch_display()

    def _apply_single_touch_display(self) -> None:
        if self._selected_touch_idx is None:
            return
        if self._cloud is None or self._actor is None:
            return

        n_verts = len(self._data.forearm_vertices)
        sel_idx = self._selected_touch_idx

        if self._rf_data is not None and self._heatmap_mode == f"rf_mean_{self._rf_data.neuron_mode}":
            heatmap_val = self._compute_rf_heatmap([sel_idx], n_verts)
        else:
            cp_mask = self._data.cp_touch_idx == sel_idx
            heatmap_val = self._compute_heatmap(cp_mask, n_verts)

        self._cloud["heatmap"] = heatmap_val
        self._cloud.Modified()
        self._actor.mapper.scalar_range = self._heatmap_clim()
        self._plotter.render()

        gtype = str(self._data.gesture_types[sel_idx])
        self._touch_count_label.setText(f"Touch {sel_idx} ({gtype})")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _compose_touch_image(
        self,
        screenshot_arr: np.ndarray,
        touch_idx: int,
        x_feature: str,
        y_feature: str,
    ) -> "Image.Image":
        """Compose an annotated PNG: white header strip above the PyVista screenshot.

        Parameters
        ----------
        screenshot_arr:
            H×W×3 or H×W×4 uint8 array from ``self._plotter.screenshot(return_img=True)``.
        touch_idx:
            Index into ``self._data.touch_triple_keys`` and ``self._data.gesture_types``.
        x_feature:
            Current X-axis feature name (used to label the header).
        y_feature:
            Current Y-axis feature name (used to label the header).

        Returns
        -------
        PIL.Image.Image
            RGB image with a ~60px white header strip above the screenshot.
        """
        if screenshot_arr.shape[2] == 4:
            screenshot_arr = screenshot_arr[:, :, :3]

        screenshot_img = Image.fromarray(screenshot_arr, mode="RGB")
        img_w, img_h = screenshot_img.size

        header_h = 60
        header_img = Image.new("RGB", (img_w, header_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(header_img)

        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()

        triple = self._data.touch_triple_keys[touch_idx]
        block_id = int(triple[0])
        trial_id = int(triple[1])
        touch_id = int(triple[2])
        gesture_type = str(self._data.gesture_types[touch_idx])
        n_contacts = int(np.sum(self._data.cp_touch_idx == touch_idx))

        line1 = (
            f"Block {block_id} | Trial {trial_id} | Touch {touch_id}"
            f" | {gesture_type} | {n_contacts} contacts"
        )

        x_display = _feature_display(x_feature)
        y_display = _feature_display(y_feature)

        if x_feature in self._data.feature_names:
            x_val = float(self._data.get_feature_array(x_feature)[touch_idx])
            x_str = f"{x_val:.3g}"
        else:
            x_str = "N/A"

        if y_feature in self._data.feature_names:
            y_val = float(self._data.get_feature_array(y_feature)[touch_idx])
            y_str = f"{y_val:.3g}"
        else:
            y_str = "N/A"

        if self._threshold_ratio_mode:
            threshold_str = f"{self._threshold_ratio_value}%"
        else:
            ratio = round(self._vertex_threshold / max(1, self._n_filtered) * 100)
            threshold_str = f"{min(ratio, 100)}%"

        line2 = (
            f"X: {x_display} = {x_str} | Y: {y_display} = {y_str}"
            f" | Min overlaps: {threshold_str}"
        )

        margin = 8
        draw.text((margin, margin), line1, fill=(0, 0, 0), font=font)
        draw.text((margin, margin + 22), line2, fill=(0, 0, 0), font=font)

        composed = Image.new("RGB", (img_w, header_h + img_h))
        composed.paste(header_img, (0, 0))
        composed.paste(screenshot_img, (0, header_h))
        return composed

    def _on_export_touches_clicked(self) -> None:
        """Batch-export one annotated PNG per gesture-type-filtered touch."""
        checked_types = {gtype for gtype, cb in self._checkboxes.items() if cb.isChecked()}
        type_mask = np.isin(self._data.gesture_types, list(checked_types))
        filtered_indices = list(np.where(type_mask)[0])

        if not filtered_indices:
            self.statusBar().showMessage("Export cancelled — no touches match the current gesture filter.")
            return

        out_dir_str = QFileDialog.getExistingDirectory(
            self, "Select export directory", ""
        )
        if not out_dir_str:
            return

        out_dir = Path(out_dir_str)

        x_feature = self._x_feature()
        y_feature = self._y_feature()

        session_idx = self._session_combo.currentIndex()
        saved_camera = None
        if self._cloud is not None:
            try:
                saved_camera = self._plotter.camera.copy()
            except Exception:
                pass

        saved_selected_idx = self._selected_touch_idx
        saved_single_touch_mode = self._single_touch_mode

        if not self._single_touch_mode:
            self._single_touch_mode = True
            self._selected_touch_idx = None
            self._render_3d()
            if saved_camera is not None:
                try:
                    self._plotter.camera = saved_camera
                except Exception:
                    pass

        n = len(filtered_indices)
        self._export_touches_btn.setEnabled(False)
        try:
            for i, touch_idx in enumerate(filtered_indices, start=1):
                triple = self._data.touch_triple_keys[touch_idx]
                block_id = int(triple[0])
                trial_id = int(triple[1])
                touch_id = int(triple[2])

                self.statusBar().showMessage(
                    f"Exporting {i}/{n} — Block {block_id}, Trial {trial_id}, Touch {touch_id}…"
                )
                QApplication.processEvents()

                self._selected_touch_idx = touch_idx
                self._apply_single_touch_display()
                QApplication.processEvents()

                screenshot_arr = self._plotter.screenshot(return_img=True)
                if screenshot_arr is None:
                    raise ValueError(
                        f"_on_export_touches_clicked: screenshot returned None for touch index {touch_idx}"
                    )

                composed = self._compose_touch_image(
                    screenshot_arr, touch_idx, x_feature, y_feature
                )

                fname = f"touch_B{block_id}_T{trial_id}_S{touch_id}.png"
                composed.save(out_dir / fname)

            self.statusBar().showMessage(
                f"Export done — {n} image(s) saved to {out_dir}"
            )
        finally:
            self._single_touch_mode = saved_single_touch_mode
            self._selected_touch_idx = saved_selected_idx

            if not saved_single_touch_mode:
                self._single_touch_mode = False
                self._render_3d()
                if self._rect is None:
                    self._init_filter_rect()
                else:
                    self._apply_filter_update()
            else:
                if saved_selected_idx is not None:
                    self._apply_single_touch_display()

            if saved_camera is not None:
                try:
                    self._plotter.camera = saved_camera
                    self._plotter.render()
                except Exception:
                    pass

            self._export_touches_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Axis changes
    # ------------------------------------------------------------------

    def _on_axis_changed(self, _index: int) -> None:
        if self._rect is not None:
            self._rect.disconnect()
        self._rect = None
        self._draw_scatter()
        if not self._single_touch_mode:
            self._init_filter_rect()

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _load_session(self, session_idx: int, new_data: PopulationData) -> None:
        if self._single_touch_cid is not None:
            self._scatter_canvas.figure.canvas.mpl_disconnect(self._single_touch_cid)
            self._single_touch_cid = None
        self._single_touch_mode = False
        self._selected_touch_idx = None
        self._single_touch_btn.blockSignals(True)
        self._single_touch_btn.setChecked(False)
        self._single_touch_btn.blockSignals(False)
        self._w_spinbox.setVisible(True)
        self._h_spinbox.setVisible(True)

        self._data = new_data
        self._rect = None
        self._cloud = None
        self._actor = None

        # Update RF data for the new session (None if rf_sessions not set or index out of range).
        if self._rf_sessions and session_idx < len(self._rf_sessions):
            self._rf_data = self._rf_sessions[session_idx]
        else:
            self._rf_data = None
        self._sync_heatmap_modes()

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

        # Remove all items from right_layout.
        # Sub-layouts (QHBoxLayout) require explicit child-widget removal before deletion.
        while right_layout.count() > 0:
            item = right_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
            else:
                sub_layout = item.layout()
                if sub_layout is not None:
                    while sub_layout.count() > 0:
                        child_item = sub_layout.takeAt(0)
                        child_widget = child_item.widget()
                        if child_widget is not None:
                            child_widget.setParent(None)
                    sub_layout.deleteLater()

        right_layout.addWidget(QLabel("Gesture types:"))
        for i, gtype in enumerate(unique_types):
            cb = QCheckBox(gtype)
            cb.setChecked(True)  # always default to checked on session load
            cb.stateChanged.connect(self._on_checkbox_changed)
            self._checkboxes[gtype] = cb
            self._gesture_colors[gtype] = _GESTURE_COLORS[i % len(_GESTURE_COLORS)]
            right_layout.addWidget(cb)

        # Rebuild threshold spinbox row and reset threshold to ratio 50%.
        self._threshold_ratio_mode = True
        self._threshold_ratio_value = 50
        self._n_filtered = 0
        self._vertex_threshold = 1
        self._threshold_row_layout = QHBoxLayout()
        self._threshold_row_layout.setSpacing(4)
        self._threshold_row_layout.addWidget(QLabel("Min overlaps:"))
        self._threshold_spinbox = QSpinBox()
        self._threshold_spinbox.setMinimum(1)
        self._threshold_spinbox.setMaximum(100)
        self._threshold_spinbox.setValue(50)
        self._threshold_spinbox.valueChanged.connect(self._on_threshold_changed)
        self._threshold_row_layout.addWidget(self._threshold_spinbox)
        self._threshold_mode_btn = QPushButton("%")
        self._threshold_mode_btn.setCheckable(True)
        self._threshold_mode_btn.setChecked(True)
        self._threshold_mode_btn.setFixedWidth(28)
        self._threshold_mode_btn.toggled.connect(self._on_threshold_mode_toggled)
        self._threshold_row_layout.addWidget(self._threshold_mode_btn)
        self._threshold_suffix_label = QLabel("% of filtered")
        self._threshold_row_layout.addWidget(self._threshold_suffix_label)
        right_layout.addLayout(self._threshold_row_layout)

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
