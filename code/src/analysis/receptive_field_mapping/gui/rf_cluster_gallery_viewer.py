"""RF Cluster Gallery Viewer — interactive PyQt5 thumbnail gallery + 3D view.

Loads data from ``GalleryData`` (produced by ``load_gallery_data()``) and
presents a scrollable thumbnail sidebar alongside a single interactive
PyVista 3D view.  Two navigation modes: "By Cluster" (all sessions for
one cluster) and "By Session" (all clusters for one session).
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from scipy.spatial import ConvexHull

from analysis.receptive_field_mapping.rf_cluster_pipeline import description_summary_line
from analysis.receptive_field_mapping.rf_gallery_data import GalleryCell, GalleryData
from analysis.receptive_field_mapping.rf_surface_utils import (
    apply_rotation_to_mesh,
    map_scalars_to_mesh,
    mesh_to_pyvista,
)

logger = logging.getLogger(__name__)

THUMB_W = 120
THUMB_H = 90
SIDEBAR_W = 160

_COLORMAPS = ["YlOrRd", "viridis", "plasma", "inferno", "magma", "coolwarm", "RdBu_r", "jet"]

_MODE_BY_CLUSTER = "By Cluster"
_MODE_BY_SESSION = "By Session"


@dataclass
class _GallerySettings:
    bg_color: str = "black"
    forearm_color: str = "lightgrey"
    forearm_opacity: float = 1.0
    render_as_surface: bool = True
    forearm_size: int = 3
    forearm_spheres: bool = False
    contact_cmap: str = "jet"
    contact_size: float = 5.0
    contact_spheres: bool = False
    cmap: str = "jet"
    display_metric: str = "spike_ratio"
    show_scalar_bar: bool = True
    show_hull_wireframes: bool = True
    hull_max_edge_mm: float = 15.0
    hull_neuron_color: str = "#00aaff"
    hull_cluster_color: str = "#ffaa00"
    show_axes: bool = False


def _numpy_to_qpixmap(img: np.ndarray) -> QPixmap:
    """Convert HxWx3 or HxWx4 uint8 numpy array to QPixmap."""
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    bytes_per_line = 3 * w
    qimage = QImage(img.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)


def _build_convex_hull_mesh(
    points: np.ndarray,
    max_edge_length: Optional[float] = None,
) -> Optional[pv.PolyData]:
    """Build a PyVista wireframe polydata from the convex hull of *points*.

    Returns None when hull computation fails or the point set is too small.
    """
    if points is None or len(points) < 4:
        return None
    try:
        hull = ConvexHull(points)
    except Exception:
        return None

    edge_set = set()
    for simplex in hull.simplices:
        for i in range(len(simplex)):
            a = simplex[i]
            b = simplex[(i + 1) % len(simplex)]
            edge_set.add((min(a, b), max(a, b)))

    if max_edge_length is not None:
        edge_set = {
            (a, b) for a, b in edge_set
            if np.linalg.norm(points[a] - points[b]) <= max_edge_length
        }

    if not edge_set:
        return None

    lines = []
    for a, b in edge_set:
        lines.extend([2, a, b])

    hull_pv = pv.PolyData(points)
    hull_pv.lines = np.array(lines, dtype=np.int_)
    return hull_pv


class _ThumbnailWidget(QWidget):
    """A clickable thumbnail card: image on top, text label below."""

    def __init__(
        self,
        key: Tuple[str, str],
        label_text: str,
        is_noise: bool,
        on_click,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._key = key
        self._on_click = on_click
        self._is_noise = is_noise
        self._selected = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._img_label = QLabel()
        self._img_label.setFixedSize(THUMB_W, THUMB_H)
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setStyleSheet("background-color: #333;")
        layout.addWidget(self._img_label, alignment=Qt.AlignHCenter)

        self._text_label = QLabel(label_text)
        self._text_label.setAlignment(Qt.AlignCenter)
        self._text_label.setWordWrap(True)
        if is_noise:
            self._text_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self._text_label)

        self.setFixedWidth(SIDEBAR_W - 8)
        self._update_style()

    def set_thumbnail(self, pixmap: QPixmap) -> None:
        self._img_label.setPixmap(pixmap)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._update_style()

    def _update_style(self) -> None:
        if self._selected:
            self.setStyleSheet("QWidget { background-color: #1a4a8a; border: 2px solid #4488ff; }")
        elif self._is_noise:
            self.setStyleSheet("QWidget { background-color: #2a2a2a; border: 1px solid #555; }")
        else:
            self.setStyleSheet("QWidget { background-color: #3a3a3a; border: 1px solid #666; }")

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._on_click(self._key)
        super().mousePressEvent(event)


class RFClusterGalleryViewer(QMainWindow):
    """Interactive gallery viewer for RF cluster heatmaps.

    Accepts a ``GalleryData`` instance (loaded via ``load_gallery_data()``)
    and presents a thumbnail sidebar alongside a single interactive 3D
    PyVista view.

    Parameters
    ----------
    gallery_data:
        Pre-loaded data for one combo/clusterer pair.
    parent:
        Optional Qt parent widget.
    """

    def __init__(self, gallery_data: GalleryData, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(
            f"RF Cluster Gallery — {gallery_data.combo_name} / {gallery_data.clusterer_name}"
        )

        self._gallery_data = gallery_data
        self._settings = _GallerySettings()
        self._current_cell: Optional[GalleryCell] = None
        self._thumb_widgets: Dict[Tuple[str, str], _ThumbnailWidget] = {}
        self._initial_render_done = False
        self._thumbnail_keys: List[Tuple[str, str]] = []
        self._thumbnail_index: int = 0
        self._session_cameras: Dict[str, dict] = {}

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        root.addLayout(self._build_toolbar())

        body_splitter = QSplitter(Qt.Horizontal)

        sidebar_container = self._build_sidebar()
        body_splitter.addWidget(sidebar_container)

        self.plotter = QtInteractor(central)
        body_splitter.addWidget(self.plotter.interactor)

        body_splitter.addWidget(self._build_right_settings_panel())

        body_splitter.setStretchFactor(0, 0)
        body_splitter.setStretchFactor(1, 1)
        body_splitter.setStretchFactor(2, 0)
        body_splitter.setSizes([SIDEBAR_W, 900, 260])

        root.addWidget(body_splitter, stretch=1)

    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()

        bar.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems([_MODE_BY_CLUSTER, _MODE_BY_SESSION])
        self._mode_combo.currentTextChanged.connect(self._on_mode_changed)
        bar.addWidget(self._mode_combo)

        bar.addWidget(QLabel("Select:"))
        self._select_combo = QComboBox()
        self._select_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._select_combo.currentIndexChanged.connect(self._on_selection_changed)
        bar.addWidget(self._select_combo, stretch=1)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        bar.addWidget(close_btn)

        return bar

    def _build_sidebar(self) -> QWidget:
        container = QWidget()
        container.setFixedWidth(SIDEBAR_W)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._scroll_content = QWidget()
        self._scroll_layout = QVBoxLayout(self._scroll_content)
        self._scroll_layout.setContentsMargins(4, 4, 4, 4)
        self._scroll_layout.setSpacing(6)
        self._scroll_layout.addStretch()

        self._scroll_area.setWidget(self._scroll_content)
        layout.addWidget(self._scroll_area)

        return container

    def _build_right_settings_panel(self) -> QScrollArea:
        s = self._settings

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFixedWidth(260)
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)

        # --- Background ---
        bg_box = QGroupBox("Background")
        bg_lay = QVBoxLayout(bg_box)
        self._bg_color_btn = self._color_button(s.bg_color)
        self._bg_color_btn.clicked.connect(
            lambda: self._pick_color("bg_color", self._bg_color_btn)
        )
        bg_lay.addWidget(self._bg_color_btn)
        settings_layout.addWidget(bg_box)

        # --- Forearm ---
        fa_box = QGroupBox("Forearm")
        fa_lay = QVBoxLayout(fa_box)

        self._fa_color_btn = self._color_button(s.forearm_color)
        self._fa_color_btn.clicked.connect(
            lambda: self._pick_color("forearm_color", self._fa_color_btn)
        )
        row = QHBoxLayout()
        row.addWidget(QLabel("Color:"))
        row.addWidget(self._fa_color_btn)
        fa_lay.addLayout(row)

        self._fa_surface_cb = QCheckBox("Render as surface")
        self._fa_surface_cb.setChecked(s.render_as_surface)
        self._fa_surface_cb.stateChanged.connect(self._on_surface_toggle)
        fa_lay.addWidget(self._fa_surface_cb)

        self._fa_size_slider = self._slider("Point size:", 1, 20, s.forearm_size, fa_lay)
        self._fa_size_slider.valueChanged.connect(
            lambda v: self._set_and_rebuild("forearm_size", v)
        )

        self._fa_opacity_slider = self._slider(
            "Opacity:", 0, 100, int(s.forearm_opacity * 100), fa_lay
        )
        self._fa_opacity_slider.valueChanged.connect(
            lambda v: self._set_and_rebuild("forearm_opacity", v / 100.0)
        )

        self._fa_spheres_cb = QCheckBox("Render as spheres")
        self._fa_spheres_cb.setChecked(s.forearm_spheres)
        self._fa_spheres_cb.stateChanged.connect(
            lambda st: self._set_and_rebuild("forearm_spheres", st == Qt.Checked)
        )
        fa_lay.addWidget(self._fa_spheres_cb)
        settings_layout.addWidget(fa_box)

        # --- Contact Points ---
        cp_box = QGroupBox("Contact Points")
        cp_lay = QVBoxLayout(cp_box)

        row_cmap = QHBoxLayout()
        row_cmap.addWidget(QLabel("Colormap:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(_COLORMAPS)
        self._cmap_combo.setCurrentText(s.cmap)
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        row_cmap.addWidget(self._cmap_combo)
        cp_lay.addLayout(row_cmap)

        row_metric = QHBoxLayout()
        row_metric.addWidget(QLabel("Metric:"))
        self._metric_combo = QComboBox()
        self._metric_combo.addItems(["spike_count", "spike_ratio"])
        self._metric_combo.setCurrentText(s.display_metric)
        self._metric_combo.currentTextChanged.connect(
            lambda t: self._set_and_rebuild("display_metric", t)
        )
        row_metric.addWidget(self._metric_combo)
        cp_lay.addLayout(row_metric)

        self._cp_size_slider = self._slider("Point size:", 1, 30, int(s.contact_size), cp_lay)
        self._cp_size_slider.valueChanged.connect(
            lambda v: self._set_and_rebuild("contact_size", float(v))
        )

        self._cp_spheres_cb = QCheckBox("Render as spheres")
        self._cp_spheres_cb.setChecked(s.contact_spheres)
        self._cp_spheres_cb.stateChanged.connect(
            lambda st: self._set_and_rebuild("contact_spheres", st == Qt.Checked)
        )
        cp_lay.addWidget(self._cp_spheres_cb)
        settings_layout.addWidget(cp_box)

        # --- Display ---
        disp_box = QGroupBox("Display")
        disp_lay = QVBoxLayout(disp_box)

        self._scalar_bar_cb = QCheckBox("Scalar bar")
        self._scalar_bar_cb.setChecked(s.show_scalar_bar)
        self._scalar_bar_cb.stateChanged.connect(
            lambda st: self._set_and_rebuild("show_scalar_bar", st == Qt.Checked)
        )
        disp_lay.addWidget(self._scalar_bar_cb)

        self._hull_cb = QCheckBox("Hull wireframes")
        self._hull_cb.setChecked(s.show_hull_wireframes)
        self._hull_cb.stateChanged.connect(
            lambda st: self._set_and_rebuild("show_hull_wireframes", st == Qt.Checked)
        )
        disp_lay.addWidget(self._hull_cb)

        row_edge = QHBoxLayout()
        row_edge.addWidget(QLabel("Max hull edge:"))
        self._hull_edge_spin = QDoubleSpinBox()
        self._hull_edge_spin.setRange(1.0, 100.0)
        self._hull_edge_spin.setSingleStep(1.0)
        self._hull_edge_spin.setDecimals(1)
        self._hull_edge_spin.setValue(s.hull_max_edge_mm)
        self._hull_edge_spin.setSuffix(" mm")
        self._hull_edge_spin.valueChanged.connect(
            lambda v: self._set_and_rebuild("hull_max_edge_mm", v)
        )
        row_edge.addWidget(self._hull_edge_spin)
        disp_lay.addLayout(row_edge)

        row_hn = QHBoxLayout()
        row_hn.addWidget(QLabel("Neuron hull:"))
        self._hull_neuron_color_btn = self._color_button(s.hull_neuron_color)
        self._hull_neuron_color_btn.clicked.connect(
            lambda: self._pick_color("hull_neuron_color", self._hull_neuron_color_btn)
        )
        row_hn.addWidget(self._hull_neuron_color_btn)
        disp_lay.addLayout(row_hn)

        row_hc = QHBoxLayout()
        row_hc.addWidget(QLabel("Cluster hull:"))
        self._hull_cluster_color_btn = self._color_button(s.hull_cluster_color)
        self._hull_cluster_color_btn.clicked.connect(
            lambda: self._pick_color("hull_cluster_color", self._hull_cluster_color_btn)
        )
        row_hc.addWidget(self._hull_cluster_color_btn)
        disp_lay.addLayout(row_hc)

        self._axes_cb = QCheckBox("Axes")
        self._axes_cb.setChecked(s.show_axes)
        self._axes_cb.stateChanged.connect(
            lambda st: self._set_and_rebuild("show_axes", st == Qt.Checked)
        )
        disp_lay.addWidget(self._axes_cb)
        settings_layout.addWidget(disp_box)

        settings_layout.addStretch()
        settings_scroll.setWidget(settings_widget)

        self._on_surface_toggle(Qt.Checked if s.render_as_surface else Qt.Unchecked)

        return settings_scroll

    # ------------------------------------------------------------------
    # Widget helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _color_button(color_name: str) -> QPushButton:
        btn = QPushButton()
        btn.setFixedSize(60, 24)
        qc = QColor(color_name)
        btn.setStyleSheet(f"background-color: {qc.name()};")
        return btn

    @staticmethod
    def _slider(label: str, lo: int, hi: int, val: int, layout: QVBoxLayout) -> QSlider:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        sl = QSlider(Qt.Horizontal)
        sl.setMinimum(lo)
        sl.setMaximum(hi)
        sl.setValue(val)
        row.addWidget(sl)
        layout.addLayout(row)
        return sl

    def _pick_color(self, attr: str, btn: QPushButton) -> None:
        current = QColor(getattr(self._settings, attr))
        color = QColorDialog.getColor(current, self, f"Pick {attr}")
        if color.isValid():
            btn.setStyleSheet(f"background-color: {color.name()};")
            self._set_and_rebuild(attr, color.name())

    def _on_surface_toggle(self, state) -> None:
        is_surface = state == Qt.Checked
        self._fa_size_slider.setEnabled(not is_surface)
        self._fa_spheres_cb.setEnabled(not is_surface)
        self._set_and_rebuild("render_as_surface", is_surface)

    def _on_cmap_changed(self, text: str) -> None:
        self._settings.cmap = text
        self._settings.contact_cmap = text
        self._apply_setting_change()

    # ------------------------------------------------------------------
    # Sidebar population
    # ------------------------------------------------------------------

    def _current_mode(self) -> str:
        return self._mode_combo.currentText()

    def _populate_select_combo(self) -> None:
        self._select_combo.blockSignals(True)
        self._select_combo.clear()
        if self._current_mode() == _MODE_BY_CLUSTER:
            for label in self._gallery_data.cluster_labels:
                self._select_combo.addItem(f"cluster_{label}", userData=label)
        else:
            for sid in self._gallery_data.session_ids:
                self._select_combo.addItem(sid, userData=sid)
        self._select_combo.blockSignals(False)

    def _visible_keys(self) -> List[Tuple[str, str]]:
        """Return (session_id, cluster_label) pairs visible under the current selection."""
        mode = self._current_mode()
        idx = self._select_combo.currentIndex()
        if idx < 0:
            return []

        if mode == _MODE_BY_CLUSTER:
            cluster_label = self._select_combo.itemData(idx)
            return [
                (sid, cluster_label)
                for sid in self._gallery_data.session_ids
                if (sid, cluster_label) in self._gallery_data.cells
            ]
        else:
            session_id = self._select_combo.itemData(idx)
            return [
                (session_id, cl)
                for cl in self._gallery_data.cluster_labels
                if (session_id, cl) in self._gallery_data.cells
            ]

    def _repopulate_sidebar(self) -> None:
        for w in self._thumb_widgets.values():
            w.setParent(None)
        self._thumb_widgets.clear()

        stretch_item = self._scroll_layout.takeAt(self._scroll_layout.count() - 1)

        keys = self._visible_keys()
        for key in keys:
            cell = self._gallery_data.cells[key]
            mode = self._current_mode()
            if mode == _MODE_BY_CLUSTER:
                text = cell.session_id
            else:
                text = f"cluster_{cell.cluster_label}"

            widget = _ThumbnailWidget(
                key=key,
                label_text=text,
                is_noise=cell.is_noise,
                on_click=self._on_thumbnail_click,
            )
            if cell.thumbnail is not None:
                widget.set_thumbnail(cell.thumbnail)

            tooltip_lines = []
            if cell.cluster_description:
                desc_type = cell.cluster_description.get("touch_type", "")
                if desc_type:
                    tooltip_lines.append(f"Type: {desc_type}")
            tooltip_lines.append(f"Touches: {cell.neuron_cluster_touches}")
            if cell.rf_metrics and "rf_area_mm2" in cell.rf_metrics:
                area = cell.rf_metrics["rf_area_mm2"]
                tooltip_lines.append(f"RF area: {area:.1f} mm²")
            widget.setToolTip("\n".join(tooltip_lines))

            self._scroll_layout.addWidget(widget)
            self._thumb_widgets[key] = widget

        self._scroll_layout.addItem(stretch_item)

        if keys:
            first_key = keys[0]
            self._load_cell(first_key)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _on_mode_changed(self, _text: str) -> None:
        self._populate_select_combo()
        self._repopulate_sidebar()

    def _on_selection_changed(self, _index: int) -> None:
        self._repopulate_sidebar()

    def _on_thumbnail_click(self, key: Tuple[str, str]) -> None:
        self._load_cell(key)

    def _load_cell(self, key: Tuple[str, str]) -> None:
        if self._current_cell is not None:
            self._session_cameras[self._current_cell.session_id] = self._capture_camera()

        for k, w in self._thumb_widgets.items():
            w.set_selected(k == key)

        self._current_cell = self._gallery_data.cells[key]
        self._build_scene(self._current_cell)

        stored_cam = self._session_cameras.get(self._current_cell.session_id)
        if stored_cam is not None:
            self._restore_camera(stored_cam)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _set_and_rebuild(self, attr: str, value) -> None:
        setattr(self._settings, attr, value)
        self._apply_setting_change()

    def _apply_setting_change(self) -> None:
        if self._current_cell is None:
            return
        cam = self._capture_camera()
        self._build_scene(self._current_cell)
        self._restore_camera(cam)

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self, cell: GalleryCell) -> None:
        self.plotter.clear()
        self.plotter.set_background(self._settings.bg_color)

        s = self._settings
        metric_col = (
            "spike_count"
            if s.display_metric == "spike_count"
            else "unique_touch_spike_count"
        )

        if s.render_as_surface and cell.forearm_mesh is not None:
            mesh_pv = mesh_to_pyvista(cell.forearm_mesh)
            if cell.tangent_rotation is not None:
                mesh_pv = mesh_to_pyvista(
                    apply_rotation_to_mesh(cell.forearm_mesh, cell.tangent_rotation)
                )

            df = cell.spike_counts_df
            if len(df) > 0 and metric_col in df.columns:
                contact_pts = df[["x", "y", "z"]].to_numpy()
                scalar_vals = df[metric_col].to_numpy().astype(np.float64)

                target_mesh = cell.forearm_mesh
                if cell.tangent_rotation is not None:
                    target_mesh = apply_rotation_to_mesh(
                        cell.forearm_mesh, cell.tangent_rotation
                    )

                spike_values = map_scalars_to_mesh(target_mesh, contact_pts, scalar_vals)
                if cell.tangent_rotation is not None:
                    contact_pts = (cell.tangent_rotation @ contact_pts.T).T
                mesh_pv["spike_values"] = spike_values
                max_val = float(np.nanmax(spike_values)) if not np.all(np.isnan(spike_values)) else 1.0

                sbar_args = {"title": s.display_metric} if s.show_scalar_bar else None
                self.plotter.add_mesh(
                    mesh_pv,
                    scalars="spike_values",
                    cmap=s.cmap,
                    clim=[0, max_val],
                    nan_color=s.forearm_color,
                    opacity=s.forearm_opacity,
                    smooth_shading=True,
                    scalar_bar_args=sbar_args,
                    show_scalar_bar=s.show_scalar_bar,
                    name="forearm",
                )
            else:
                self.plotter.add_mesh(
                    mesh_pv,
                    color=s.forearm_color,
                    opacity=s.forearm_opacity,
                    smooth_shading=True,
                    name="forearm",
                )

        elif cell.forearm_vertices is not None:
            pts = cell.forearm_vertices
            if cell.tangent_rotation is not None:
                pts = (cell.tangent_rotation @ pts.T).T
            cloud = pv.PolyData(pts)
            self.plotter.add_mesh(
                cloud,
                color=s.forearm_color,
                point_size=s.forearm_size,
                opacity=s.forearm_opacity,
                render_points_as_spheres=s.forearm_spheres,
                name="forearm",
            )

            df = cell.spike_counts_df
            if len(df) > 0 and metric_col in df.columns:
                contact_pts = df[["x", "y", "z"]].to_numpy()
                scalar_vals = df[metric_col].to_numpy().astype(np.float64)
                if cell.tangent_rotation is not None:
                    contact_pts = (cell.tangent_rotation @ contact_pts.T).T
                contact_cloud = pv.PolyData(contact_pts)
                max_val = float(np.nanmax(scalar_vals)) if len(scalar_vals) > 0 else 1.0
                contact_cloud["scalars"] = scalar_vals
                sbar_args = {"title": s.display_metric} if s.show_scalar_bar else None
                self.plotter.add_mesh(
                    contact_cloud,
                    scalars="scalars",
                    cmap=s.contact_cmap,
                    clim=[0, max_val],
                    point_size=s.contact_size,
                    render_points_as_spheres=s.contact_spheres,
                    scalar_bar_args=sbar_args,
                    show_scalar_bar=s.show_scalar_bar,
                    name="contacts",
                )

        if s.show_hull_wireframes:
            neuron_pts = cell.neuron_contacts_xyz
            if cell.tangent_rotation is not None and neuron_pts is not None and len(neuron_pts) >= 4:
                neuron_pts = (cell.tangent_rotation @ neuron_pts.T).T
            hull_neuron = _build_convex_hull_mesh(neuron_pts, max_edge_length=s.hull_max_edge_mm)
            if hull_neuron is not None:
                self.plotter.add_mesh(
                    hull_neuron, color=s.hull_neuron_color, style="wireframe", line_width=2,
                    name="hull_neuron",
                )

            cluster_pts = cell.cluster_contacts_xyz
            if cell.tangent_rotation is not None and cluster_pts is not None and len(cluster_pts) >= 4:
                cluster_pts = (cell.tangent_rotation @ cluster_pts.T).T
            hull_cluster = _build_convex_hull_mesh(cluster_pts, max_edge_length=s.hull_max_edge_mm)
            if hull_cluster is not None:
                self.plotter.add_mesh(
                    hull_cluster, color=s.hull_cluster_color, style="wireframe", line_width=2,
                    name="hull_cluster",
                )

        if s.show_axes:
            self.plotter.add_axes()

        if cell.cluster_description:
            summary = description_summary_line(cell.cluster_description, separator='\n')
            if summary:
                self.plotter.add_text(
                    summary,
                    position="upper_left",
                    font_size=10,
                    color="white",
                    name="cluster_info",
                )

        if cell.tangent_rotation is not None:
            self.plotter.view_xy()
        else:
            self.plotter.view_isometric()

        self.plotter.render()

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def _capture_camera(self) -> dict:
        cam = self.plotter.camera
        return {
            "camera_position": list(cam.position),
            "focal_point": list(cam.focal_point),
            "up_vector": list(cam.up),
            "view_angle": float(cam.view_angle),
        }

    def _restore_camera(self, params: dict) -> None:
        self.plotter.camera.position = params["camera_position"]
        self.plotter.camera.focal_point = params["focal_point"]
        self.plotter.camera.up = params["up_vector"]
        self.plotter.camera.view_angle = params["view_angle"]
        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

    # ------------------------------------------------------------------
    # Thumbnail generation
    # ------------------------------------------------------------------

    def _generate_thumbnails(self) -> None:
        self._thumbnail_keys = list(self._gallery_data.cells.keys())
        self._thumbnail_index = 0
        self._generate_next_thumbnail()

    def _generate_next_thumbnail(self) -> None:
        if self._thumbnail_index >= len(self._thumbnail_keys):
            if self._current_cell is not None:
                self._build_scene(self._current_cell)
            return

        key = self._thumbnail_keys[self._thumbnail_index]
        cell = self._gallery_data.cells[key]
        self._build_scene(cell)
        try:
            img = self.plotter.screenshot()
            pixmap = _numpy_to_qpixmap(img).scaled(
                THUMB_W, THUMB_H, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            cell.thumbnail = pixmap
            widget = self._thumb_widgets.get(key)
            if widget is not None:
                widget.set_thumbnail(pixmap)
        except Exception:
            logger.exception("Failed to capture thumbnail for %s/%s", key[0], key[1])

        self._thumbnail_index += 1
        QTimer.singleShot(0, self._generate_next_thumbnail)

    # ------------------------------------------------------------------
    # Deferred start
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._initial_render_done:
            self._initial_render_done = True
            primary = QApplication.primaryScreen()
            if primary is not None:
                self.move(primary.geometry().topLeft())
            self.showMaximized()
            QTimer.singleShot(0, self._deferred_start)

    def _deferred_start(self) -> None:
        try:
            self.plotter.interactor.Initialize()
        except Exception:
            pass
        sz = self.plotter.interactor.size()
        if sz.width() > 0 and sz.height() > 0:
            self.plotter.render_window.SetSize(sz.width(), sz.height())

        self._populate_select_combo()
        self._repopulate_sidebar()

        QTimer.singleShot(0, self._generate_thumbnails)
