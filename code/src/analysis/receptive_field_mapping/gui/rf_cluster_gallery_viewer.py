"""RF Cluster Gallery Viewer — interactive PyQt5 thumbnail gallery + 3D view.

Loads data from ``GalleryData`` (produced by ``load_gallery_data()``) and
presents a scrollable thumbnail sidebar alongside a single interactive
PyVista 3D view.  Two navigation modes: "By Cluster" (all sessions for
one cluster) and "By Session" (all clusters for one session).
"""

import logging
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
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
from matplotlib import colormaps
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, KDTree, QhullError

from analysis.receptive_field_mapping.rf_cluster_pipeline import description_summary_line
from analysis.receptive_field_mapping.rf_extraction_io import (
    load_delaunay_thresholds,
    load_session_cameras,
    save_delaunay_thresholds,
    save_session_cameras,
)
from analysis.receptive_field_mapping.rf_gallery_data import GalleryCell, GalleryData
from analysis.receptive_field_mapping.rf_surface_utils import (
    build_delaunay_mesh,
    map_scalars_to_mesh,
    mesh_to_pyvista,
)

logger = logging.getLogger(__name__)

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
    use_vertex_colors: bool = True
    contact_cmap: str = "jet"
    contact_size: float = 5.0
    contact_spheres: bool = False
    cmap: str = "jet"
    display_metric: str = "spike_ratio"
    show_scalar_bar: bool = True
    hull_display_mode: str = "perimeter"
    hull_show_points: bool = False
    hull_blob_sep_mm: float = 20.0
    hull_alpha_mm: float = 15.0
    hull_line_width: float = 2.0
    hull_neuron_color: str = "#00aaff"
    hull_cluster_color: str = "#ffaa00"
    show_axes: bool = False



def _separate_blobs(pts_2d: np.ndarray, max_dist_mm: float) -> np.ndarray:
    n = len(pts_2d)
    tree = KDTree(pts_2d)
    pairs = list(tree.query_pairs(max_dist_mm))
    if not pairs:
        return np.arange(n)
    rows, cols = zip(*pairs)
    data = np.ones(len(rows))
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    return labels


def _alpha_shape_edges(pts_2d: np.ndarray, alpha_mm: float) -> set:
    if len(pts_2d) < 3:
        return set()
    try:
        tri = Delaunay(pts_2d)
    except QhullError:
        return set()
    edge_count: dict = {}
    for simplex in tri.simplices:
        a, b, c = simplex
        pa, pb, pc = pts_2d[a], pts_2d[b], pts_2d[c]
        ab = pb - pa
        ac = pc - pa
        area = abs(ab[0] * ac[1] - ab[1] * ac[0]) / 2.0
        if area <= 0:
            continue
        side_a = np.linalg.norm(pb - pc)
        side_b = np.linalg.norm(pc - pa)
        side_c = np.linalg.norm(pa - pb)
        R = (side_a * side_b * side_c) / (4.0 * area)
        if R < alpha_mm:
            for edge in (
                (min(a, b), max(a, b)),
                (min(b, c), max(b, c)),
                (min(a, c), max(a, c)),
            ):
                edge_count[edge] = edge_count.get(edge, 0) + 1
    return {e for e, cnt in edge_count.items() if cnt == 1}


def _build_perimeter_mesh(
    points: np.ndarray,
    blob_sep_mm: float,
    alpha_mm: float,
) -> Optional[pv.PolyData]:
    if points is None or len(points) < 3:
        return None
    pts_2d = points[:, :2]
    labels = _separate_blobs(pts_2d, blob_sep_mm)
    lines: list = []
    for label in np.unique(labels):
        mask = labels == label
        blob_2d = pts_2d[mask]
        global_indices = np.where(mask)[0]
        if len(blob_2d) < 3:
            continue
        local_edges = _alpha_shape_edges(blob_2d, alpha_mm)
        for local_a, local_b in local_edges:
            lines.extend([2, global_indices[local_a], global_indices[local_b]])
    if not lines:
        return None
    lines_arr = np.array(lines, dtype=np.int_)
    n_edges = len(lines_arr) // 3
    edge_pairs = lines_arr.reshape(n_edges, 3)[:, 1:]
    unique_verts, inverse = np.unique(edge_pairs.ravel(), return_inverse=True)
    new_lines = np.hstack(
        [np.full((n_edges, 1), 2, dtype=np.int_), inverse.reshape(n_edges, 2)]
    ).ravel()
    mesh = pv.PolyData(points[unique_verts])
    mesh.lines = new_lines
    return mesh


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

        self._text_label = QLabel(label_text)
        self._text_label.setAlignment(Qt.AlignCenter)
        self._text_label.setWordWrap(True)
        if is_noise:
            self._text_label.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(self._text_label)

        self.setFixedWidth(SIDEBAR_W - 8)
        self._update_style()

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
        _type_suffix = f" — {gallery_data.gesture_type}" if gallery_data.gesture_type else ""
        self.setWindowTitle(
            f"RF Cluster Gallery — {gallery_data.combo_name} / {gallery_data.clusterer_name}{_type_suffix}"
        )

        self._gallery_data = gallery_data
        self._settings = _GallerySettings()
        self._current_cell: Optional[GalleryCell] = None
        self._thumb_widgets: Dict[Tuple[str, str], _ThumbnailWidget] = {}
        self._initial_render_done = False
        self._session_cameras: Dict[str, dict] = self._load_cameras()

        self._default_threshold: float = 20.0
        self._session_thresholds: Dict[str, float] = self._load_thresholds()
        self._delaunay_cache: Dict[Tuple[str, float], Optional["trimesh.Trimesh"]] = {}
        self._heatmap_cache: Dict[Tuple[str, str, float, str], np.ndarray] = {}
        self._perimeter_cache: Dict[
            Tuple[str, Optional[str], str, float, float], Optional[pv.PolyData]
        ] = {}

        self._last_rendered_settings: Optional[_GallerySettings] = None
        self._last_rendered_threshold: Optional[float] = None

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

        self._preload_btn = QPushButton("Load All")
        self._preload_btn.setToolTip("Pre-compute Delaunay meshes, heatmaps, and perimeter hulls for every cell so navigation is instant.")
        self._preload_btn.clicked.connect(self._on_preload_all_clicked)
        bar.addWidget(self._preload_btn)

        self._export_btn = QPushButton("Export images")
        self._export_btn.clicked.connect(self._on_export_all_clicked)
        bar.addWidget(self._export_btn)

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

    def _build_right_settings_panel(self) -> QWidget:
        s = self._settings

        outer = QWidget()
        outer.setFixedWidth(260)
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)

        # --- Cluster Info ---
        info_box = QGroupBox("Cluster Info")
        info_lay = QVBoxLayout(info_box)
        self._cluster_info_label = QLabel("—")
        self._cluster_info_label.setWordWrap(True)
        self._cluster_info_label.setAlignment(Qt.AlignTop)
        self._cluster_info_label.setStyleSheet("font-size: 11px;")
        info_lay.addWidget(self._cluster_info_label)
        settings_layout.addWidget(info_box)

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

        row_threshold = QHBoxLayout()
        row_threshold.addWidget(QLabel("Max edge:"))
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(1.0, 200.0)
        self._threshold_spin.setSingleStep(1.0)
        self._threshold_spin.setDecimals(1)
        self._threshold_spin.setValue(self._default_threshold)
        self._threshold_spin.setSuffix(" mm")
        row_threshold.addWidget(self._threshold_spin)
        fa_lay.addLayout(row_threshold)

        self._fa_size_slider = self._slider("Point size:", 1, 20, s.forearm_size, fa_lay)
        self._fa_size_slider.valueChanged.connect(
            lambda v: self._stage_setting("forearm_size", v)
        )

        self._fa_opacity_slider = self._slider(
            "Opacity:", 0, 100, int(s.forearm_opacity * 100), fa_lay
        )
        self._fa_opacity_slider.valueChanged.connect(
            lambda v: self._stage_setting("forearm_opacity", v / 100.0)
        )

        self._fa_spheres_cb = QCheckBox("Render as spheres")
        self._fa_spheres_cb.setChecked(s.forearm_spheres)
        self._fa_spheres_cb.stateChanged.connect(
            lambda st: self._stage_setting("forearm_spheres", st == Qt.Checked)
        )
        fa_lay.addWidget(self._fa_spheres_cb)

        self._fa_vertex_colors_cb = QCheckBox("Use vertex colors")
        self._fa_vertex_colors_cb.setChecked(s.use_vertex_colors)
        self._fa_vertex_colors_cb.stateChanged.connect(
            lambda st: self._stage_setting("use_vertex_colors", st == Qt.Checked)
        )
        fa_lay.addWidget(self._fa_vertex_colors_cb)
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
            lambda t: self._stage_setting("display_metric", t)
        )
        row_metric.addWidget(self._metric_combo)
        cp_lay.addLayout(row_metric)

        self._cp_size_slider = self._slider("Point size:", 1, 30, int(s.contact_size), cp_lay)
        self._cp_size_slider.valueChanged.connect(
            lambda v: self._stage_setting("contact_size", float(v))
        )

        self._cp_spheres_cb = QCheckBox("Render as spheres")
        self._cp_spheres_cb.setChecked(s.contact_spheres)
        self._cp_spheres_cb.stateChanged.connect(
            lambda st: self._stage_setting("contact_spheres", st == Qt.Checked)
        )
        cp_lay.addWidget(self._cp_spheres_cb)
        settings_layout.addWidget(cp_box)

        # --- Display ---
        disp_box = QGroupBox("Display")
        disp_lay = QVBoxLayout(disp_box)

        self._scalar_bar_cb = QCheckBox("Scalar bar")
        self._scalar_bar_cb.setChecked(s.show_scalar_bar)
        self._scalar_bar_cb.stateChanged.connect(
            lambda st: self._stage_setting("show_scalar_bar", st == Qt.Checked)
        )
        disp_lay.addWidget(self._scalar_bar_cb)

        _hull_mode_items = ["Perimeter", "Point cloud", "Off"]
        _hull_mode_values = ["perimeter", "pointcloud", "off"]
        row_hull_mode = QHBoxLayout()
        row_hull_mode.addWidget(QLabel("Hull:"))
        self._hull_mode_combo = QComboBox()
        self._hull_mode_combo.addItems(_hull_mode_items)
        self._hull_mode_combo.setCurrentIndex(
            _hull_mode_values.index(s.hull_display_mode)
            if s.hull_display_mode in _hull_mode_values else 0
        )
        self._hull_mode_combo.currentIndexChanged.connect(
            lambda i: self._stage_setting("hull_display_mode", _hull_mode_values[i])
        )
        row_hull_mode.addWidget(self._hull_mode_combo)
        disp_lay.addLayout(row_hull_mode)

        row_lw = QHBoxLayout()
        row_lw.addWidget(QLabel("Line width:"))
        self._hull_lw_spin = QDoubleSpinBox()
        self._hull_lw_spin.setRange(0.5, 20.0)
        self._hull_lw_spin.setSingleStep(0.5)
        self._hull_lw_spin.setDecimals(1)
        self._hull_lw_spin.setValue(s.hull_line_width)
        self._hull_lw_spin.valueChanged.connect(
            lambda v: self._stage_setting("hull_line_width", v)
        )
        row_lw.addWidget(self._hull_lw_spin)
        disp_lay.addLayout(row_lw)

        self._hull_show_pts_cb = QCheckBox("Show points")
        self._hull_show_pts_cb.setChecked(s.hull_show_points)
        self._hull_show_pts_cb.stateChanged.connect(
            lambda st: self._stage_setting("hull_show_points", st == Qt.Checked)
        )
        disp_lay.addWidget(self._hull_show_pts_cb)

        row_blob = QHBoxLayout()
        row_blob.addWidget(QLabel("Blob sep:"))
        self._hull_blob_spin = QDoubleSpinBox()
        self._hull_blob_spin.setRange(1.0, 200.0)
        self._hull_blob_spin.setSingleStep(5.0)
        self._hull_blob_spin.setDecimals(1)
        self._hull_blob_spin.setValue(s.hull_blob_sep_mm)
        self._hull_blob_spin.setSuffix(" mm")
        self._hull_blob_spin.valueChanged.connect(
            lambda v: self._stage_setting("hull_blob_sep_mm", v)
        )
        row_blob.addWidget(self._hull_blob_spin)
        disp_lay.addLayout(row_blob)

        row_alpha = QHBoxLayout()
        row_alpha.addWidget(QLabel("Alpha radius:"))
        self._hull_alpha_spin = QDoubleSpinBox()
        self._hull_alpha_spin.setRange(1.0, 100.0)
        self._hull_alpha_spin.setSingleStep(1.0)
        self._hull_alpha_spin.setDecimals(1)
        self._hull_alpha_spin.setValue(s.hull_alpha_mm)
        self._hull_alpha_spin.setSuffix(" mm")
        self._hull_alpha_spin.valueChanged.connect(
            lambda v: self._stage_setting("hull_alpha_mm", v)
        )
        row_alpha.addWidget(self._hull_alpha_spin)
        disp_lay.addLayout(row_alpha)

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
            lambda st: self._stage_setting("show_axes", st == Qt.Checked)
        )
        disp_lay.addWidget(self._axes_cb)
        settings_layout.addWidget(disp_box)

        settings_layout.addStretch()
        settings_scroll.setWidget(settings_widget)
        outer_layout.addWidget(settings_scroll, stretch=1)

        btn_bar = QWidget()
        btn_layout = QHBoxLayout(btn_bar)
        btn_layout.setContentsMargins(8, 6, 8, 6)
        self._process_btn = QPushButton("Process")
        self._process_all_btn = QPushButton("Process All")
        btn_layout.addWidget(self._process_btn)
        btn_layout.addWidget(self._process_all_btn)
        self._process_btn.clicked.connect(self._on_process_current)
        self._process_all_btn.clicked.connect(self._on_process_all)
        outer_layout.addWidget(btn_bar)

        self._on_surface_toggle(Qt.Checked if s.render_as_surface else Qt.Unchecked)

        return outer

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
            setattr(self._settings, attr, color.name())

    def _on_surface_toggle(self, state) -> None:
        is_surface = state == Qt.Checked
        self._fa_size_slider.setEnabled(not is_surface)
        self._fa_spheres_cb.setEnabled(not is_surface)
        self._set_and_rebuild("render_as_surface", is_surface)

    def _on_cmap_changed(self, text: str) -> None:
        self._settings.cmap = text
        self._settings.contact_cmap = text

    def _update_cluster_info(self, cell: GalleryCell) -> None:
        desc = cell.cluster_description or {}
        lines = []

        n = desc.get('n_touches')
        if n is not None:
            lines.append(f"n_touches: {n}")

        gtd = desc.get('gesture_type_distribution') or {}
        if gtd:
            dominant = max(gtd, key=gtd.get)
            lines.append(f"type: {dominant} ({gtd[dominant]:.0%})")

        dr = desc.get('display_ranges') or {}
        if dr:
            for label, r in dr.items():
                lines.append(f"{label}: [{r['min']}, {r['max']}]")
        else:
            fr = desc.get('feature_ranges') or {}
            for col, r in fr.items():
                lines.append(f"{col}: [{r['min']}, {r['max']}]")

        self._cluster_info_label.setText("\n".join(lines) if lines else "—")

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
        self._update_cluster_info(self._current_cell)

        self._threshold_spin.blockSignals(True)
        self._threshold_spin.setValue(
            self._current_threshold(self._current_cell.session_id)
        )
        self._threshold_spin.blockSignals(False)

        self._build_scene(self._current_cell)
        self._record_rendered_state()

        stored_cam = self._session_cameras.get(self._current_cell.session_id)
        if stored_cam is not None:
            self._restore_camera(stored_cam)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _stage_setting(self, attr: str, value) -> None:
        setattr(self._settings, attr, value)

    def _set_and_rebuild(self, attr: str, value) -> None:
        setattr(self._settings, attr, value)
        self._apply_setting_change()

    def _record_rendered_state(self) -> None:
        self._last_rendered_settings = replace(self._settings)
        if self._current_cell is not None:
            self._last_rendered_threshold = self._current_threshold(self._current_cell.session_id)

    def _has_pending_changes(self) -> bool:
        if self._last_rendered_settings is None:
            return True
        if self._settings != self._last_rendered_settings:
            return True
        if self._current_cell is None:
            return False
        return self._threshold_spin.value() != self._last_rendered_threshold

    def _apply_setting_change(self) -> None:
        if self._current_cell is None:
            return
        cam = self._capture_camera()
        self._build_scene(self._current_cell)
        self._restore_camera(cam)
        self._record_rendered_state()

    def _on_process_current(self) -> None:
        if self._current_cell is None:
            return
        if not self._has_pending_changes():
            return
        self._session_thresholds[self._current_cell.session_id] = self._threshold_spin.value()
        self._apply_setting_change()
        self._save_thresholds()

    def _on_process_all(self) -> None:
        if self._current_cell is None:
            return
        self._session_thresholds[self._current_cell.session_id] = self._threshold_spin.value()
        for key in self._visible_keys():
            self._get_delaunay_mesh(self._gallery_data.cells[key])
        if self._has_pending_changes():
            self._apply_setting_change()
        self._save_thresholds()

    # ------------------------------------------------------------------
    # Delaunay threshold persistence
    # ------------------------------------------------------------------

    def _load_thresholds(self) -> Dict[str, float]:
        return load_delaunay_thresholds(self._gallery_data.output_base_dir)

    def _save_thresholds(self) -> None:
        save_delaunay_thresholds(
            self._gallery_data.output_base_dir, self._session_thresholds
        )

    # ------------------------------------------------------------------
    # Camera persistence
    # ------------------------------------------------------------------

    def _load_cameras(self) -> Dict[str, dict]:
        return load_session_cameras(self._gallery_data.output_base_dir)

    def _save_cameras(self) -> None:
        save_session_cameras(
            self._gallery_data.output_base_dir, self._session_cameras
        )

    def _current_threshold(self, session_id: str) -> float:
        return self._session_thresholds.get(session_id, self._default_threshold)

    def _get_delaunay_mesh(self, cell: GalleryCell):
        if cell.forearm_vertices is None:
            return None
        threshold = self._current_threshold(cell.session_id)
        key = (cell.session_id, threshold)
        if key not in self._delaunay_cache:
            vertices = cell.forearm_vertices
            if cell.tangent_rotation is not None:
                vertices = (cell.tangent_rotation @ vertices.T).T
            self._delaunay_cache[key] = build_delaunay_mesh(vertices, max_edge_mm=threshold)
        return self._delaunay_cache[key]

    def _get_heatmap(
        self,
        cell: GalleryCell,
        mesh: "trimesh.Trimesh",
        threshold: float,
        metric_col: str,
    ) -> np.ndarray:
        key = (cell.session_id, cell.cluster_label, threshold, metric_col)
        if key not in self._heatmap_cache:
            df = cell.spike_counts_df
            contact_pts = df[["x", "y", "z"]].to_numpy()
            scalar_vals = df[metric_col].to_numpy().astype(np.float64)
            if cell.tangent_rotation is not None:
                contact_pts = (cell.tangent_rotation @ contact_pts.T).T
            self._heatmap_cache[key] = map_scalars_to_mesh(
                mesh, contact_pts, scalar_vals
            )
        return self._heatmap_cache[key]

    def _get_perimeter_mesh(
        self,
        cell: GalleryCell,
        role: str,
        raw_pts: np.ndarray,
        blob_sep_mm: float,
        alpha_mm: float,
    ) -> Optional[pv.PolyData]:
        cluster_key = None if role == "neuron" else cell.cluster_label
        key = (cell.session_id, cluster_key, role, blob_sep_mm, alpha_mm)
        if key not in self._perimeter_cache:
            pts = raw_pts
            if cell.tangent_rotation is not None:
                pts = (cell.tangent_rotation @ pts.T).T
            self._perimeter_cache[key] = _build_perimeter_mesh(pts, blob_sep_mm, alpha_mm)
        return self._perimeter_cache[key]

    @staticmethod
    def _compose_spike_vertex_colors(
        spike_values: np.ndarray,
        max_val: float,
        cmap_name: str,
        vertex_colors: np.ndarray,
    ) -> np.ndarray:
        nan_mask = np.isnan(spike_values)
        rgb = vertex_colors.copy()

        if not np.all(nan_mask) and max_val > 0:
            cmap = colormaps[cmap_name]
            normed = np.clip(spike_values[~nan_mask] / max_val, 0.0, 1.0)
            mapped = cmap(normed)[:, :3]
            rgb[~nan_mask] = (mapped * 255).astype(np.uint8)

        return rgb

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

        delaunay_mesh = self._get_delaunay_mesh(cell) if s.render_as_surface else None

        has_vertex_colors = (
            s.use_vertex_colors
            and cell.forearm_vertex_colors is not None
        )

        if delaunay_mesh is not None:
            # delaunay_mesh is already in rotated space (rotation applied in _get_delaunay_mesh)
            mesh_pv = mesh_to_pyvista(delaunay_mesh)

            vc_match = (
                has_vertex_colors
                and len(cell.forearm_vertex_colors) == len(delaunay_mesh.vertices)
            )

            df = cell.spike_counts_df
            has_spike_data = len(df) > 0 and metric_col in df.columns

            if has_spike_data:
                threshold = self._current_threshold(cell.session_id)
                spike_values = self._get_heatmap(
                    cell, delaunay_mesh, threshold, metric_col
                )
                max_val = float(np.nanmax(spike_values)) if not np.all(np.isnan(spike_values)) else 1.0

                if vc_match:
                    rgb = self._compose_spike_vertex_colors(
                        spike_values, max_val, s.cmap, cell.forearm_vertex_colors
                    )
                    mesh_pv["rgb"] = rgb
                    self.plotter.add_mesh(
                        mesh_pv,
                        scalars="rgb",
                        rgb=True,
                        opacity=s.forearm_opacity,
                        smooth_shading=True,
                        name="forearm",
                    )
                else:
                    mesh_pv["spike_values"] = spike_values
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
            elif vc_match:
                mesh_pv["rgb"] = cell.forearm_vertex_colors
                self.plotter.add_mesh(
                    mesh_pv,
                    scalars="rgb",
                    rgb=True,
                    opacity=s.forearm_opacity,
                    smooth_shading=True,
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

            vc_match = (
                has_vertex_colors
                and len(cell.forearm_vertex_colors) == len(cell.forearm_vertices)
            )

            if vc_match:
                cloud["rgb"] = cell.forearm_vertex_colors
                self.plotter.add_mesh(
                    cloud,
                    scalars="rgb",
                    rgb=True,
                    point_size=s.forearm_size,
                    opacity=s.forearm_opacity,
                    render_points_as_spheres=s.forearm_spheres,
                    name="forearm",
                )
            else:
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

        if s.hull_display_mode != "off":
            for raw_pts, color, name, role in [
                (cell.neuron_contacts_xyz, s.hull_neuron_color, "hull_neuron", "neuron"),
                (cell.cluster_contacts_xyz, s.hull_cluster_color, "hull_cluster", "cluster"),
            ]:
                if raw_pts is None or len(raw_pts) == 0:
                    continue
                if s.hull_display_mode == "perimeter":
                    mesh = self._get_perimeter_mesh(
                        cell, role, raw_pts, s.hull_blob_sep_mm, s.hull_alpha_mm
                    )
                    if mesh is not None:
                        self.plotter.add_mesh(
                            mesh, color=color, style="wireframe", line_width=s.hull_line_width, name=name
                        )
                else:
                    pts = raw_pts
                    if cell.tangent_rotation is not None:
                        pts = (cell.tangent_rotation @ pts.T).T
                    cloud = pv.PolyData(pts)
                    self.plotter.add_mesh(
                        cloud, color=color, point_size=s.contact_size, name=name
                    )

        if s.hull_show_points:
            for raw_pts, color, name in [
                (cell.neuron_contacts_xyz, s.hull_neuron_color, "hull_neuron_pts"),
                (cell.cluster_contacts_xyz, s.hull_cluster_color, "hull_cluster_pts"),
            ]:
                if raw_pts is None or len(raw_pts) == 0:
                    continue
                pts = raw_pts
                if cell.tangent_rotation is not None:
                    pts = (cell.tangent_rotation @ pts.T).T
                self.plotter.add_mesh(
                    pv.PolyData(pts), color=color, point_size=s.contact_size, name=name
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

    def _set_auto_face_on_camera(self, cell: GalleryCell) -> None:
        if cell.tangent_rotation is not None:
            self.plotter.view_xy()
        else:
            self.plotter.view_isometric()
        self.plotter.reset_camera()

    # ------------------------------------------------------------------
    # Batch preload
    # ------------------------------------------------------------------

    def _on_preload_all_clicked(self) -> None:
        """Pre-warm all caches for every cell so switching is instant."""
        cells = self._gallery_data.cells
        if not cells:
            return

        s = self._settings
        metric_col = (
            "spike_count" if s.display_metric == "spike_count" else "unique_touch_spike_count"
        )

        all_keys = sorted(cells.keys())
        n = len(all_keys)

        self._preload_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        try:
            for i, key in enumerate(all_keys, start=1):
                cell = cells[key]
                self.statusBar().showMessage(
                    f"Preloading {i}/{n}: {cell.session_id} / cluster_{cell.cluster_label}"
                )
                QApplication.processEvents()

                mesh = self._get_delaunay_mesh(cell)

                if mesh is not None:
                    df = cell.spike_counts_df
                    if len(df) > 0 and metric_col in df.columns:
                        threshold = self._current_threshold(cell.session_id)
                        self._get_heatmap(cell, mesh, threshold, metric_col)

                if s.hull_display_mode == "perimeter":
                    for raw_pts, role in [
                        (cell.neuron_contacts_xyz, "neuron"),
                        (cell.cluster_contacts_xyz, "cluster"),
                    ]:
                        if raw_pts is not None and len(raw_pts) >= 3:
                            self._get_perimeter_mesh(
                                cell, role, raw_pts, s.hull_blob_sep_mm, s.hull_alpha_mm
                            )

            self.statusBar().showMessage(f"Preload complete: {n} cells cached.")
        finally:
            self._preload_btn.setEnabled(True)
            self._export_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Batch export
    # ------------------------------------------------------------------

    def _on_export_all_clicked(self) -> None:
        cells = self._gallery_data.cells
        if not cells:
            self.statusBar().showMessage("No cells to export.")
            return

        export_dir = self._gallery_data.output_base_dir / "_gallery_exports"
        export_dir.mkdir(parents=True, exist_ok=True)

        sorted_keys = sorted(cells.keys())
        n = len(sorted_keys)

        saved_cell = self._current_cell
        if saved_cell is not None:
            self._session_cameras[saved_cell.session_id] = self._capture_camera()
        saved_cameras = dict(self._session_cameras)

        self._export_btn.setEnabled(False)
        try:
            for i, (sid, lbl) in enumerate(sorted_keys, start=1):
                self.statusBar().showMessage(f"Exporting {i}/{n}: {sid}__cluster_{lbl}")
                QApplication.processEvents()

                cell = cells[(sid, lbl)]
                self._build_scene(cell)
                stored_cam = saved_cameras.get(cell.session_id)
                if stored_cam is not None:
                    self._restore_camera(stored_cam)
                else:
                    self._set_auto_face_on_camera(cell)
                QApplication.processEvents()

                out_path = export_dir / f"{sid}__cluster_{lbl}.png"
                self.plotter.screenshot(str(out_path))

            self.statusBar().showMessage(
                f"Export complete: {n} image(s) saved to {export_dir}"
            )
        finally:
            self._session_cameras = saved_cameras
            if saved_cell is not None:
                self._build_scene(saved_cell)
                stored_cam = self._session_cameras.get(saved_cell.session_id)
                if stored_cam is not None:
                    self._restore_camera(stored_cam)
            self._export_btn.setEnabled(True)

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

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._current_cell is not None:
            self._session_cameras[self._current_cell.session_id] = self._capture_camera()
        self._save_cameras()
        self._save_thresholds()
        self.plotter.close()
        super().closeEvent(event)

