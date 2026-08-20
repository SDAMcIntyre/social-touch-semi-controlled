"""
postprocessing_stage_viewer.py
-------------------------------
Single-window PyQt5+PyVista viewer for postprocessed data with a stage
dropdown that switches between all 5 postprocessing coordinate stages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
import pandas as pd
import pyvista as pv
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from merging.contact_depth_field_series import ContactDepthFieldLoader
from merging.gui.neural_kinect_scene_viewer import NeuralDataPanel, contact_polydata

# The scalar array name, the colourbar title and the colourmap have exactly one
# definition in the tree, shared with the tactile-quantification depth viewer
# and the Neural+Kinect viewer.  Restating any of them here would let this
# window disagree with the other two about what it is showing.
from preprocessing.motion_analysis.tactile_quantification.gui.contact_depth_field_viewer import (
    COLORMAP,
    CONTACT_SCALAR_NAME,
    SCALAR_BAR_TITLE,
)

# Re-exported: the labels are defined in the Qt-free policy leaf beside this
# module, because that leaf names stages in its validation errors and a second
# copy of the six strings would drift from this one.
from .stage_depth_field import STAGE_LABELS  # noqa: F401

# All depth-field policy — which space a stage must declare, what to say when
# there is none, and how a slider position becomes a frame index — lives in that
# same leaf, and is testable without a window because of it.
from .stage_depth_field import (
    FOREARM_DEPTH_SCALAR_NAME,
    StageDepthField,
    depth_frame_at_position,
    forearm_depth_scalars,
    kinect_frame_at_position,
    kinect_frame_indices,
    resolve_stage_depth_field,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CAMERA_FRAME_STAGES = {0, 1, 2, 3}
_PCA_FRAME_STAGES = {4, 5}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class StagePaths:
    """Paths and metadata for one postprocessing stage.

    Attributes:
        stage_label: The dropdown entry this stage is shown under.
        csv_path: The stage's merged-data CSV, or ``None`` when the session has
            no merged output directory.
        forearm: The forearm surface for this stage — a PLY path or an
            already-loaded point cloud.
        coordinate_frame: ``"camera"`` or ``"pca"``.
        depth_field_loader: Zero-argument callable returning this stage's
            :class:`~merging.contact_depth_field_series.ContactDepthFieldSeries`,
            or ``None`` when the sidecar is absent.  A **loader**, never a path
            and never a loaded table: the caller resolves six of these before
            the window exists, and reading six stages of 10^5-10^6 vertices
            before the first pixel is exactly the regression the lazy contract
            was introduced to prevent.  The field itself defaults to ``None``
            so every existing construction site stays valid; ``None`` means the
            caller wired no depth field at all, which is indistinguishable from
            an absent sidecar as far as this widget is concerned.
        depth_field_path: Where ``depth_field_loader`` was built to look.  Used
            **only** to name the file in messages and errors — nothing in this
            module opens it, and the widget derives no parquet path of its own.
            It travels beside the loader because a loader deliberately hides its
            path and the loaded series carries none either, yet "wrong
            coordinate space" without "in which file" is not actionable.  The
            two fields are set together or not at all.
    """

    stage_label: str
    csv_path: Optional[Path]
    forearm: Optional[Union[Path, "o3d.geometry.PointCloud"]]
    coordinate_frame: str  # "camera" or "pca"
    depth_field_loader: Optional[ContactDepthFieldLoader] = None
    depth_field_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _parse_contact_points_cell(cell) -> Optional[np.ndarray]:
    if not isinstance(cell, str):
        return None
    s = cell.strip()
    if s in ("", "[]", "nan"):
        return None
    rows = re.findall(r"\[\s*([-\d\s.,eE+]+)\s*\]", s)
    points: List[List[float]] = []
    for row in rows:
        parts = row.replace(",", " ").split()
        if len(parts) == 3:
            try:
                points.append([float(p) for p in parts])
            except ValueError:
                pass
    return np.array(points, dtype=np.float64) if points else None


def _empty_contact_polydata(with_depth_scalars: bool) -> pv.PolyData:
    """A zero-point contact dataset for a frame with no contact.

    Args:
        with_depth_scalars: Whether the dataset must carry the depth array.
            When the actor is mapped to :data:`CONTACT_SCALAR_NAME`, the array
            has to exist even at zero length, or the mapper loses its binding
            the first time a no-contact frame is shown and the colours never
            come back.  When the actor renders flat, the array must be absent
            rather than zero-filled: no depth is not depth zero.

    Returns:
        An empty :class:`pyvista.PolyData`, built by the same function that
        builds the non-empty ones so the two cannot drift apart.
    """
    points = np.empty((0, 3), dtype=np.float32)
    depths = np.empty((0,), dtype=np.float64) if with_depth_scalars else None
    return contact_polydata(points, depths)


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------


class PostprocessingStageViewer(QMainWindow):
    """Single-window viewer with a stage dropdown for the 5 postprocessing stages.

    Parameters
    ----------
    stage_paths:
        One ``StagePaths`` per stage (length must equal ``len(STAGE_LABELS)``).
    recording_name:
        Displayed in the 3D scene text label and window title.
    initial_stage:
        Index of the stage to display on launch (default: 0).
    parent:
        Optional Qt parent widget.
    """

    def __init__(
        self,
        stage_paths: List[StagePaths],
        recording_name: str,
        initial_stage: int = 0,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        if len(stage_paths) != len(STAGE_LABELS):
            raise ValueError(
                f"stage_paths must have {len(STAGE_LABELS)} entries, "
                f"got {len(stage_paths)}"
            )
        if not (0 <= initial_stage < len(stage_paths)):
            raise ValueError(
                f"initial_stage {initial_stage} out of range [0, {len(stage_paths)})"
            )

        pv.global_theme.allow_empty_mesh = True

        self._stage_paths = stage_paths
        self._recording_name = recording_name
        self._current_stage_idx: int = initial_stage
        self.current_index: int = 0
        self._initial_render_done = False
        self._bounds_proxy_active = False
        # The view preference, deliberately NOT the same attribute as the data
        # fact (``self._depth_series is not None``).  It lives on the viewer, not
        # on the stage, so switching to a stage without a sidecar leaves it
        # untouched and the next stage that has one opens coloured again.
        self._colour_contact_by_depth: bool = True
        # The forearm's own layer preference, and a separate attribute for the
        # same reason: it is a view preference, not a data fact.  Default OFF,
        # unlike the contact layer -- the PLY's own vertex colours are the
        # anatomical context the contact patch has to be judged against, and a
        # surface repainted on every frame would replace that context by
        # default rather than on request.
        self._colour_forearm_by_depth: bool = False

        self.setWindowTitle(f"Postprocessing Stage Viewer | {recording_name}")

        self._load_stage_data(initial_stage)
        self._build_ui()
        self._init_actors()

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_advance)

        self._drag_timer = QTimer(self)
        self._drag_timer.timeout.connect(self._on_drag_timer_fired)
        self._drag_timer.setInterval(80)
        self._pending_drag_frame: Optional[int] = None
        self._slider_dragging = False
        self._stage_switch_generation: int = 0

    @property
    def _depth_colouring_active(self) -> bool:
        """Whether contact vertices are currently coloured by penetration depth.

        Both facts must hold, and they are different facts: the first is about
        this stage's data (a sidecar exists and decoded), the second about the
        user's view preference.  Collapsing them into one flag would make an
        absent field indistinguishable from an unchecked box.
        """
        return self._depth_series is not None and self._colour_contact_by_depth

    @property
    def _forearm_depth_available(self) -> bool:
        """Whether this stage *could* paint penetration depth on the forearm.

        Three independent facts, all required: a depth field, a ``vertex_id``
        column in it, and a forearm with vertices for the index to address.
        The middle one is why the control is absent rather than disabled before
        the projection stage -- there is no index there, and the only way to
        invent one is a nearest-vertex snap that picks different vertices than
        the projection stage did.
        """
        return (
            self._depth_series is not None
            and self._depth_series.has_vertex_ids
            and self._forearm_pv is not None
            and self._forearm_pv.n_points > 0
        )

    @property
    def _forearm_depth_colouring_active(self) -> bool:
        """Whether the forearm is currently painted by penetration depth."""
        return self._forearm_depth_available and self._colour_forearm_by_depth

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_stage_data(self, stage_idx: int) -> None:
        sp = self._stage_paths[stage_idx]

        # --- Depth field ---
        # Resolved before the CSV because it is a fact about the stage, not
        # about this stage's rows: which coordinate space it must declare, and
        # what to say when it is absent, are decided in the Qt-free policy leaf.
        # Nothing about parquet, schema versions, sign convention or path
        # derivation enters this widget — only a series and a message do.
        self._depth_field: StageDepthField = resolve_stage_depth_field(
            stage_idx, sp.depth_field_loader, sp.depth_field_path
        )
        self._depth_series = self._depth_field.series
        # The producer's global colour range, taken whole.  The one number this
        # widget must never compute: a range derived from a frame's own data
        # would recolour the same depth differently on every frame.
        self._contact_clim: Optional[Tuple[float, float]] = (
            None
            if self._depth_series is None
            else self._depth_series.clim_penetration_mm
        )

        # --- CSV ---
        if sp.csv_path is None or not sp.csv_path.exists():
            self._full_df = pd.DataFrame()
            self._kinect_df = pd.DataFrame()
            self._contact_pts_by_frame: List[Optional[np.ndarray]] = []
            self._frame_indices: Optional[np.ndarray] = None
            self._has_contact_source = False
            self._total_frames = 0
            self._forearm_pv = None
            self._contact_centroid = np.zeros(3, dtype=np.float64)
            return

        full_df = pd.read_csv(sp.csv_path)
        self._full_df = full_df
        self._kinect_df = full_df.dropna(subset=["time_kinect"]).reset_index(drop=True)
        self._total_frames = len(self._kinect_df)

        if "contact_points" in self._kinect_df.columns:
            self._contact_pts_by_frame = [
                _parse_contact_points_cell(c) for c in self._kinect_df["contact_points"]
            ]
        else:
            self._contact_pts_by_frame = []

        # --- Slider position -> Kinect frame ---
        # Built only when there is a depth field to join, and then it must
        # succeed: the sidecar is keyed by Kinect frame_index while the slider
        # walks row positions, and the CSV is upsampled to the nerve rate, so a
        # positional join would draw another frame's depths at every position.
        # See stage_depth_field.kinect_frame_indices for why this raises.
        self._frame_indices = (
            None
            if self._depth_series is None
            else kinect_frame_indices(self._kinect_df, sp.csv_path)
        )
        self._has_contact_source = (
            bool(self._contact_pts_by_frame) or self._depth_series is not None
        )

        # --- Forearm ---
        self._forearm_pv = self._load_forearm(sp.forearm)

        # --- Contact centroid ---
        self._contact_centroid = self._compute_contact_centroid()

    def _load_forearm(
        self, forearm: Optional[Union[Path, "o3d.geometry.PointCloud"]]
    ) -> Optional[pv.PolyData]:
        if forearm is None:
            return None

        if isinstance(forearm, Path):
            if not forearm.exists():
                return None
            o3d_pc = o3d.io.read_point_cloud(str(forearm))
        elif isinstance(forearm, o3d.geometry.PointCloud):
            o3d_pc = forearm
        else:
            raise ValueError(
                f"forearm must be None, Path, or o3d.geometry.PointCloud, got {type(forearm)}"
            )

        if not o3d_pc.has_points():
            return None

        pts = np.asarray(o3d_pc.points, dtype=np.float32)
        mesh = pv.PolyData(pts)
        if o3d_pc.has_colors():
            mesh["colors"] = (np.asarray(o3d_pc.colors) * 255).astype(np.uint8)
        else:
            mesh["colors"] = np.full((len(pts), 3), 160, dtype=np.uint8)
        return mesh

    def _compute_contact_centroid(self) -> np.ndarray:
        if self._contact_pts_by_frame:
            valid = [
                pts
                for pts in self._contact_pts_by_frame
                if pts is not None and len(pts) > 0
            ]
            if valid:
                return np.nanmean(np.vstack(valid), axis=0)
        return np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        self._outer_layout = QVBoxLayout(central)

        # --- Top bar: stage selector ---
        top_bar = QWidget()
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(4, 4, 4, 4)
        top_bar_layout.addWidget(QLabel("Stage:"))
        self._stage_combo = QComboBox()
        self._stage_combo.addItems([sp.stage_label for sp in self._stage_paths])
        self._stage_combo.setCurrentIndex(self._current_stage_idx)
        self._stage_combo.currentIndexChanged.connect(self._on_stage_changed)
        top_bar_layout.addWidget(self._stage_combo)
        top_bar_layout.addStretch()
        self._outer_layout.addWidget(top_bar)

        # --- 3D area + right panel ---
        mid_widget = QWidget()
        mid_layout = QHBoxLayout(mid_widget)
        mid_layout.setContentsMargins(0, 0, 0, 0)

        plotter_widget = QWidget()
        plotter_layout = QVBoxLayout(plotter_widget)
        plotter_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(plotter_widget)
        self.plotter.set_background("black")
        plotter_layout.addWidget(self.plotter.interactor)
        mid_layout.addWidget(plotter_widget, stretch=4)

        self._right_panel = QWidget()
        self._right_panel_layout = QVBoxLayout(self._right_panel)
        self._visibility: dict = {}
        self._point_sizes: dict = {"forearm": 5.0, "contact_points": 15.0}
        self._build_right_panel_controls()
        scroll = QScrollArea()
        scroll.setWidget(self._right_panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(220)
        mid_layout.addWidget(scroll, stretch=1)

        self._outer_layout.addWidget(mid_widget, stretch=4)

        # --- Frame controls ---
        self._outer_layout.addWidget(self._build_frame_controls())

        # --- Neural panel ---
        self._neural_panel: Optional[NeuralDataPanel] = None
        self._neural_scale: float = 1.0
        self._maybe_create_neural_panel()

    def _build_right_panel_controls(self) -> None:
        while self._right_panel_layout.count():
            item = self._right_panel_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._visibility = {}
        self._point_sizes = {"forearm": 5.0, "contact_points": 15.0}

        def _add_group(
            label: str,
            key: str,
            has_slider: bool,
            default_size: float,
            extra_widgets: Tuple[QWidget, ...] = (),
        ) -> None:
            self._visibility[key] = True
            box = QGroupBox(label)
            box_layout = QVBoxLayout(box)
            cb = QCheckBox("Visible")
            cb.setChecked(True)
            cb.stateChanged.connect(lambda state, k=key: self._on_visibility_changed(k, state))
            box_layout.addWidget(cb)
            if has_slider:
                row = QWidget()
                rl = QHBoxLayout(row)
                rl.setContentsMargins(0, 0, 0, 0)
                rl.addWidget(QLabel("Pt size"))
                sl = QSlider(Qt.Horizontal)
                sl.setMinimum(1)
                sl.setMaximum(20)
                sl.setValue(int(self._point_sizes.get(key, default_size)))
                sl.valueChanged.connect(lambda val, k=key: self._on_point_size_changed(k, val))
                rl.addWidget(sl)
                box_layout.addWidget(row)
            for extra in extra_widgets:
                box_layout.addWidget(extra)
            self._right_panel_layout.addWidget(box)

        # The forearm's depth layer is offered only where the sidecar carries a
        # ``vertex_id``, i.e. from the projection stage onward.  The control is
        # *absent* on the earlier stages rather than greyed out, because a
        # disabled box states "not for this stage yet" while its absence states
        # the truth -- there is no index, and no honest thing to show.
        _forearm_extras: Tuple[QWidget, ...] = ()
        if self._forearm_depth_available:
            forearm_depth_cb = QCheckBox("Colour by depth")
            # blockSignals for the same reason the contact box uses it: seeding
            # the widget must not be mistaken for the user clicking it.
            forearm_depth_cb.blockSignals(True)
            forearm_depth_cb.setChecked(self._colour_forearm_by_depth)
            forearm_depth_cb.blockSignals(False)
            low, high = self._depth_series.clim_penetration_mm
            forearm_depth_cb.setToolTip(
                "Paint each forearm vertex with the penetration depth of "
                "whatever touched it on this frame, joined by the sidecar's "
                "vertex_id. Untouched vertices stay flat grey -- that is "
                "'nothing touched here', which is not the same as 0.00 mm. "
                f"Same fixed colour scale as the contact points: {low:.2f} to "
                f"{high:.2f} mm."
            )
            forearm_depth_cb.stateChanged.connect(
                self._on_forearm_depth_colour_changed
            )
            _forearm_extras = (forearm_depth_cb,)

        _add_group(
            "Forearm",
            "forearm",
            has_slider=True,
            default_size=5.0,
            extra_widgets=_forearm_extras,
        )
        if self._has_contact_source:
            # "Colour by depth" sits beside the point-size slider so flat red
            # stays one click away for comparison.
            #
            # The checkbox is always created and enabled/disabled per stage,
            # never omitted: a control that is absent reads as "this viewer
            # cannot do that", while a greyed-out one with a tooltip states the
            # actual fact -- that *this* stage has no field, and which task
            # produces one.
            _has_field = self._depth_field.is_present
            depth_cb = QCheckBox("Colour by depth")
            depth_cb.setEnabled(_has_field)
            # blockSignals: seeding the widget must not be mistaken for the user
            # clicking it.  Without this, a fieldless stage would write ``False``
            # back over the persisted preference on every switch through it, and
            # the next stage that does have a field would open in flat red.
            depth_cb.blockSignals(True)
            depth_cb.setChecked(self._colour_contact_by_depth and _has_field)
            depth_cb.blockSignals(False)
            if _has_field:
                low, high = self._depth_series.clim_penetration_mm
                depth_cb.setToolTip(
                    "Colour contact vertices by penetration depth (inferno), on a "
                    f"colour scale fixed over the whole recording: {low:.2f} to "
                    f"{high:.2f} mm. Unchecked renders them in flat red."
                )
            else:
                # The leaf's message already names the producing DAG task; a
                # second wording here would be a second place to keep true.
                depth_cb.setToolTip(self._depth_field.message)
            depth_cb.stateChanged.connect(self._on_contact_depth_colour_changed)
            _add_group(
                "Contact Points",
                "contact_points",
                has_slider=True,
                default_size=15.0,
                extra_widgets=(depth_cb,),
            )
        self._right_panel_layout.addStretch()

    def _build_frame_controls(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.addWidget(QLabel("Frame:"))

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(self._total_frames - 1, 0))
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(self._total_frames > 0)
        self.frame_slider.valueChanged.connect(self._on_slider_change)
        self.frame_slider.sliderPressed.connect(self._on_slider_pressed)
        self.frame_slider.sliderReleased.connect(self._on_slider_released)
        layout.addWidget(self.frame_slider)

        label_text = f"1 / {self._total_frames}" if self._total_frames > 0 else "0 / 0"
        self.frame_label = QLabel(label_text)
        self.frame_label.setFixedWidth(100)
        layout.addWidget(self.frame_label)

        recenter_btn = QPushButton("Recenter")
        recenter_btn.clicked.connect(self._recenter_view)
        layout.addWidget(recenter_btn)

        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(self._toggle_play)
        layout.addWidget(self.play_button)
        return widget

    def _maybe_create_neural_panel(self) -> None:
        has_neural = (
            len(self._full_df) > 0 and "Nerve_freq" in self._full_df.columns
        )
        if has_neural:
            self._neural_panel = NeuralDataPanel(self._full_df, self._total_frames)
            self._neural_panel.frame_requested.connect(self._on_neural_frame_requested)
            self._outer_layout.addWidget(self._neural_panel)
            self._neural_scale = (
                len(self._full_df) / self._total_frames if self._total_frames > 0 else 1.0
            )
        else:
            self._neural_panel = None
            self._neural_scale = 1.0

    # ------------------------------------------------------------------
    # VTK actors
    # ------------------------------------------------------------------

    def _add_forearm_actor(self):
        """Add (or replace) the forearm actor in whichever colour mode is on.

        Two mutually exclusive modes, and the switch between them is the one
        place ``add_mesh`` is re-entered for this actor.  Direct-RGB and
        mapped-scalar colouring are different mapper configurations, not
        different arrays, so they cannot be toggled by swapping the active
        scalars the way the contact actor's on/off is.  Per-*frame* updates do
        not come through here: they overwrite the existing array in place, so
        playback never cycles ``remove_actor`` / ``add_mesh``.

        Returns:
            The forearm actor, already registered under the name ``"forearm"``
            so this call replaces any previous one.
        """
        if not self._forearm_depth_colouring_active:
            return self.plotter.add_mesh(
                self._forearm_pv,
                scalars="colors",
                rgb=True,
                name="forearm",
                render_points_as_spheres=True,
                point_size=self._point_sizes["forearm"],
            )

        # Seeded all-NaN: at this point no frame has been drawn, and NaN is
        # "untouched", which is the truthful state of every vertex until one is.
        self._forearm_pv[FOREARM_DEPTH_SCALAR_NAME] = np.full(
            self._forearm_pv.n_points, np.nan, dtype=np.float64
        )
        self._forearm_pv.set_active_scalars(FOREARM_DEPTH_SCALAR_NAME)
        return self.plotter.add_mesh(
            self._forearm_pv,
            scalars=FOREARM_DEPTH_SCALAR_NAME,
            cmap=COLORMAP,
            # The contact layer's range, taken whole and unmodified: the two
            # layers paint the same field and a second scale would make the
            # same depth two colours in one scene.
            clim=self._contact_clim,
            # Untouched vertices are NaN, and NaN needs its own flat colour or
            # the LUT clamps it to the bottom of the ramp -- which would render
            # "nobody touched this" identically to "touched at the shallowest
            # depth in the recording".  The grey is the same neutral this module
            # already paints a colourless PLY with.
            nan_color="#a0a0a0",
            nan_opacity=1.0,
            # One bar for both layers; the contact actor registers it.
            show_scalar_bar=False,
            name="forearm",
            render_points_as_spheres=True,
            point_size=self._point_sizes["forearm"],
        )

    def _init_actors(self) -> None:
        _seed = np.zeros((1, 3), dtype=np.float32)
        _seed_col = np.full((1, 3), 128, dtype=np.uint8)

        if self._forearm_pv is not None and self._visibility.get("forearm", True):
            self._actor_forearm = self._add_forearm_actor()
        else:
            _empty_forearm = pv.PolyData(_seed.copy())
            _empty_forearm["colors"] = _seed_col.copy()
            self._actor_forearm = self.plotter.add_mesh(
                _empty_forearm,
                scalars="colors",
                rgb=True,
                name="forearm",
                render_points_as_spheres=True,
                point_size=self._point_sizes["forearm"],
            )

        # --- Contact points -------------------------------------------------
        # Registered once per stage load.  The colour range is the producer's,
        # computed over the whole recording for this stage; nothing here derives
        # it.  The previous stage's scalar bar needs no explicit removal: the
        # `plotter.clear()` in `_on_stage_changed` destroys it before this runs,
        # measured on the installed PyVista 0.47.1.
        self._mesh_contact = _empty_contact_polydata(
            with_depth_scalars=self._contact_clim is not None
        )
        if self._contact_clim is None:
            self._actor_contact = self.plotter.add_mesh(
                self._mesh_contact,
                name="contact_points",
                color="red",
                render_points_as_spheres=True,
                point_size=self._point_sizes["contact_points"],
            )
        else:
            self._actor_contact = self.plotter.add_mesh(
                self._mesh_contact,
                name="contact_points",
                scalars=CONTACT_SCALAR_NAME,
                cmap=COLORMAP,
                # Explicit and global.  Without it the mapper reverts to
                # per-frame autoscale on PyVista 0.47.1, which makes the
                # animation lie about relative depth.
                clim=self._contact_clim,
                show_scalar_bar=True,
                scalar_bar_args={
                    "title": SCALAR_BAR_TITLE,
                    "vertical": True,
                    "n_labels": 6,
                    "fmt": "%.2f",
                    "title_font_size": 16,
                    "label_font_size": 13,
                    # Explicit white: the theme default is black, which is
                    # invisible against this viewer's black background — the bar
                    # renders but its title and ticks do not.
                    "color": "white",
                    "position_x": 0.85,
                    "position_y": 0.12,
                    "width": 0.05,
                    "height": 0.72,
                },
                render_points_as_spheres=True,
                point_size=self._point_sizes["contact_points"],
            )
            # Flat red is what the mapper falls back to when scalar visibility
            # is switched off, so it is set even in depth-colouring mode.
            self._actor_contact.GetProperty().SetColor(1.0, 0.0, 0.0)

        # The actor is always built in depth mode when a field exists; the user's
        # preference is applied afterwards, so a stage entered with the box
        # unchecked opens flat-red with its bar hidden rather than flashing
        # coloured for one frame.  Neither mode introduces a new actor, so
        # ``_on_point_size_changed``'s actor map needs no new entry.
        self._apply_contact_scalar_mode()

        if self._total_frames == 0:
            self.plotter.add_text(
                "No data available",
                position="upper_left",
                font_size=14,
                color="white",
                name="no_data_label",
            )
        else:
            self.plotter.add_text(
                self._recording_name,
                position="upper_left",
                font_size=10,
                color="white",
                name="recording_label",
            )

        self.plotter.add_axes(interactive=False, line_width=150, box=True)

        cx, cy, cz = self._contact_centroid.tolist()
        _cam_pos, _cam_focal, _cam_up = self._compute_camera_params()
        self.plotter.camera.focal_point = _cam_focal
        self.plotter.camera.position = _cam_pos
        self.plotter.camera.up = _cam_up
        self.plotter.camera_set = True

        _box_half = 400.0
        _bounds_proxy = pv.Box(
            bounds=(
                cx - _box_half, cx + _box_half,
                cy - _box_half, cy + _box_half,
                cz - _box_half, cz + _box_half,
            )
        )
        self._bounds_proxy_active = True
        self.plotter.add_mesh(_bounds_proxy, opacity=0.001, name="_bounds_proxy", pickable=False)

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------

    def _update_frame(self, frame_idx: int) -> None:
        self.current_index = frame_idx

        if self._bounds_proxy_active and (
            self._forearm_pv is not None
            # A depth field is real geometry too, and on a stage whose CSV
            # carries no contact_points blob it is the only geometry — without
            # this the invisible proxy would never be retired.
            or self._depth_series is not None
            or (
                self._contact_pts_by_frame
                and any(
                    p is not None
                    for p in self._contact_pts_by_frame[: min(frame_idx + 1, 10)]
                )
            )
        ):
            self._bounds_proxy_active = False
            try:
                self.plotter.remove_actor("_bounds_proxy")
            except Exception:
                pass

        if self._has_contact_source:
            # Source precedence: the depth field when this stage has one,
            # otherwise the CSV's pre-parsed contact_points.  Never a mix — the
            # two are different point sets.  The CSV blob is `%.1f` text, the
            # sidecar float32, so the same vertex differs between them by up to
            # 0.05 mm per axis and pairing the two would mean matching rows by
            # coordinate value, which the sidecar's design record forbids.
            _cpts: Optional[np.ndarray] = None
            _cdepths: Optional[np.ndarray] = None
            if self._visibility.get("contact_points", True):
                if self._depth_series is not None:
                    _pair = depth_frame_at_position(
                        self._depth_series, self._frame_indices, frame_idx
                    )
                    if _pair is not None:
                        _cpts, _cdepths = _pair
                elif frame_idx < len(self._contact_pts_by_frame):
                    _cpts = self._contact_pts_by_frame[frame_idx]

            if _cpts is not None and len(_cpts) > 0:
                self._mesh_contact.DeepCopy(
                    contact_polydata(_cpts.astype(np.float32), _cdepths)
                )
            else:
                self._mesh_contact.DeepCopy(
                    _empty_contact_polydata(
                        with_depth_scalars=self._contact_clim is not None
                    )
                )
            # Re-asserted after every dataset swap: without it the mapper
            # reverts to per-frame autoscale on PyVista 0.47.1 and the same
            # depth would take a different colour on a different frame.
            if self._contact_clim is not None:
                self._actor_contact.mapper.scalar_range = self._contact_clim

        self._update_forearm_depth_scalars(frame_idx)

        if hasattr(self, "_actor_forearm") and self._actor_forearm is not None:
            if self._visibility.get("forearm", True):
                self._actor_forearm.VisibilityOn()
            else:
                self._actor_forearm.VisibilityOff()

        self.plotter.renderer.ResetCameraClippingRange()
        self.plotter.render()

        if self._neural_panel is not None:
            self._neural_panel.update_cursor(int(frame_idx * self._neural_scale))

        if self._total_frames > 0:
            self.frame_label.setText(f"{frame_idx + 1} / {self._total_frames}")
        else:
            self.frame_label.setText("0 / 0")

    # ------------------------------------------------------------------
    # Stage switching
    # ------------------------------------------------------------------

    def _on_stage_changed(self, index: int) -> None:
        if index == self._current_stage_idx:
            return

        if self._play_timer.isActive():
            self._play_timer.stop()
            self.play_button.setText("▶ Play")

        saved_frame = self.current_index
        old_frame = self._stage_paths[self._current_stage_idx].coordinate_frame
        new_frame = self._stage_paths[index].coordinate_frame
        frame_changed = old_frame != new_frame

        self._current_stage_idx = index
        self._load_stage_data(index)

        self.plotter.clear()
        self.plotter.set_background("black")
        self._build_right_panel_controls()

        # Defer actor initialization and rendering to the next event loop tick
        # so the OpenGL context from clear() is fully released first.
        # Without this, wglMakeCurrent fails on Windows.
        self._stage_switch_generation += 1
        current_gen = self._stage_switch_generation

        def _deferred_stage_init():
            if self._stage_switch_generation != current_gen:
                return
            self._init_actors()

            if frame_changed:
                pos, focal, up = self._compute_camera_params()
                self.plotter.camera.position = pos
                self.plotter.camera.focal_point = focal
                self.plotter.camera.up = up
                self.plotter.camera_set = True

            # Recreate NeuralDataPanel
            old_panel = self._neural_panel
            if old_panel is not None:
                self._outer_layout.removeWidget(old_panel)
                old_panel.deleteLater()
                self._neural_panel = None
            self._maybe_create_neural_panel()

            # Update frame controls
            self.frame_slider.blockSignals(True)
            self.frame_slider.setMaximum(max(self._total_frames - 1, 0))
            self.frame_slider.setEnabled(self._total_frames > 0)
            restored_frame = min(saved_frame, max(self._total_frames - 1, 0))
            self.frame_slider.setValue(restored_frame)
            self.frame_slider.blockSignals(False)

            self.current_index = restored_frame
            self._update_frame(restored_frame)

        QTimer.singleShot(0, _deferred_stage_init)

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def _compute_camera_params(self):
        cx, cy, cz = self._contact_centroid.tolist()
        position = [cx, cy, cz - 400.0]
        up = [0.375, -0.904, -0.201]
        return position, [cx, cy, cz], up

    def _recenter_view(self) -> None:
        position, focal_point, up = self._compute_camera_params()
        self.plotter.camera.position = position
        self.plotter.camera.focal_point = focal_point
        self.plotter.camera.up = up
        self.plotter.render()

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:
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
        if self._total_frames > 0:
            self._update_frame(0)
        else:
            self.plotter.renderer.ResetCameraClippingRange()
            self.plotter.render()

    def closeEvent(self, event) -> None:
        self._play_timer.stop()
        self._drag_timer.stop()
        try:
            self.plotter.close()
        except Exception:
            pass
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Slider handlers
    # ------------------------------------------------------------------

    def _on_slider_pressed(self) -> None:
        self._slider_dragging = True
        self._pending_drag_frame = None
        self._drag_timer.start()

    def _on_slider_released(self) -> None:
        self._slider_dragging = False
        self._drag_timer.stop()
        self._pending_drag_frame = None
        self._update_frame(self.frame_slider.value())

    def _on_drag_timer_fired(self) -> None:
        if self._pending_drag_frame is not None:
            frame = self._pending_drag_frame
            self._pending_drag_frame = None
            self._update_frame(frame)

    def _on_slider_change(self, value: int) -> None:
        if self._slider_dragging:
            if self._total_frames > 0:
                self.frame_label.setText(f"{value + 1} / {self._total_frames}")
            self._pending_drag_frame = value
        else:
            self._update_frame(value)

    # ------------------------------------------------------------------
    # Visibility and point-size handlers
    # ------------------------------------------------------------------

    def _on_visibility_changed(self, key: str, state: int) -> None:
        self._visibility[key] = state == Qt.Checked
        self._update_frame(self.current_index)

    def _on_point_size_changed(self, key: str, value: int) -> None:
        self._point_sizes[key] = float(value)
        actor_map = {
            "forearm": getattr(self, "_actor_forearm", None),
            "contact_points": getattr(self, "_actor_contact", None),
        }
        actor = actor_map.get(key)
        if actor is not None:
            actor.GetProperty().SetPointSize(float(value))
            self.plotter.render()
        else:
            self._update_frame(self.current_index)

    def _apply_contact_scalar_mode(self) -> None:
        """Switch the contact actor between depth colouring and flat red.

        Deliberately *not* an ``add_mesh`` / ``remove_actor`` cycle: re-adding
        the mesh re-enters PyVista's scalar-bar range logic, which does not
        preserve a global ``clim``.  Toggling ``scalar_visibility`` leaves the
        actor, its mapper and its lookup table exactly where they are, so the
        colour scale is identical before and after the round trip.
        """
        actor = getattr(self, "_actor_contact", None)
        if actor is None or self._contact_clim is None:
            return

        show_scalars = self._depth_colouring_active
        actor.mapper.scalar_visibility = show_scalars
        # Re-asserted on every mode change for the same reason it is re-asserted
        # after every dataset swap: PyVista 0.47.1 otherwise reverts the mapper
        # to per-frame autoscale.
        actor.mapper.scalar_range = self._contact_clim

        self._apply_depth_scalar_bar_visibility()

    def _update_forearm_depth_scalars(self, frame_idx: int) -> None:
        """Repaint the forearm with this frame's per-vertex penetration depths.

        The array is overwritten in place under the name the mapper was bound
        to, so this is a value update and not an actor cycle -- the same
        discipline the contact mesh's ``DeepCopy`` follows, and for the same
        playback-cost reason.

        The join itself is not done here.  ``forearm_depth_scalars`` validates
        the sidecar's reference-PLY provenance against this forearm's vertex
        count *before* scattering anything, which is what stands between a
        re-deduplicated forearm and a silently mis-coloured surface.  It raises
        on a mismatch, and that exception is deliberately not caught: a wrong
        picture here is indistinguishable from a right one.
        """
        if not self._forearm_depth_colouring_active:
            return

        scalars = forearm_depth_scalars(
            self._depth_series,
            kinect_frame_at_position(self._frame_indices, frame_idx),
            self._forearm_pv.n_points,
        )
        self._forearm_pv[FOREARM_DEPTH_SCALAR_NAME] = scalars
        self._forearm_pv.set_active_scalars(FOREARM_DEPTH_SCALAR_NAME)
        # Re-asserted after every array swap, exactly as the contact actor's is:
        # PyVista 0.47.1 otherwise reverts the mapper to per-frame autoscale and
        # the same depth takes a different colour on a different frame.
        actor = getattr(self, "_actor_forearm", None)
        if actor is not None and self._contact_clim is not None:
            actor.mapper.scalar_range = self._contact_clim

    def _apply_depth_scalar_bar_visibility(self) -> None:
        """Show the depth colourbar while *either* layer is mapped to it.

        One bar serves the contact patch and the forearm: they paint the same
        field on the same fixed scale.  It must therefore be hidden only when
        neither layer is using it -- a bar with nothing mapped to it claims the
        flat-coloured points and the plain forearm mean something on that scale.
        """
        if SCALAR_BAR_TITLE not in self.plotter.scalar_bars:
            return
        visible = self._depth_colouring_active or self._forearm_depth_colouring_active
        self.plotter.scalar_bars[SCALAR_BAR_TITLE].SetVisibility(bool(visible))

    def _on_forearm_depth_colour_changed(self, state: int) -> None:
        """Handle the Forearm group's 'Colour by depth' checkbox."""
        self._colour_forearm_by_depth = state == Qt.Checked
        if self._forearm_pv is not None and self._visibility.get("forearm", True):
            # A colour *mode* change, so the actor is re-added -- see
            # ``_add_forearm_actor``.  ``_update_frame`` then refills the array
            # for the frame currently on screen.
            self._actor_forearm = self._add_forearm_actor()
        self._apply_depth_scalar_bar_visibility()
        self._update_frame(self.current_index)

    def _on_contact_depth_colour_changed(self, state: int) -> None:
        """Handle the 'Colour by depth' checkbox."""
        self._colour_contact_by_depth = state == Qt.Checked
        self._apply_contact_scalar_mode()
        self.plotter.render()

    def _on_neural_frame_requested(self, frame: int) -> None:
        if 0 <= frame < self._total_frames:
            self.frame_slider.setValue(frame)

    # ------------------------------------------------------------------
    # Play/pause
    # ------------------------------------------------------------------

    def _toggle_play(self) -> None:
        if self._play_timer.isActive():
            self._play_timer.stop()
            self.play_button.setText("▶ Play")
        else:
            self._play_timer.start(33)
            self.play_button.setText("⏸ Pause")

    def _play_advance(self) -> None:
        if self._total_frames == 0:
            return
        next_frame = self.current_index + 1
        if next_frame >= self._total_frames:
            next_frame = 0
        self.frame_slider.setValue(next_frame)
