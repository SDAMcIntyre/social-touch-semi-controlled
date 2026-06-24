"""Touch Playback Explorer GUI — dual 3D view for per-touch animation.

Presents two side-by-side PyVista 3D forearm views:
  - Left:  current-frame contact points as red spheres on the forearm point
           cloud (PLY vertex colours when available, grey otherwise)
  - Right: heatmap accumulating from touch start (jet colormap, NaN=grey);
           mode toggleable between spike density and mean IFF (Hz)

A toolbar provides session / trial / touch dropdowns, playback controls
(Play, Play All, Stop), a speed spinbox, a heatmap mode combo, and a frame
counter.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyvista as pv
import vtk
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from analysis.receptive_field_mapping.data.touch_playback_data import PlaybackData, TouchEvent

logger = logging.getLogger(__name__)


class TouchPlaybackExplorer(QMainWindow):
    """QMainWindow with dual 3D views for animating per-touch contact + spike data.

    Parameters
    ----------
    playback_data:
        Pre-loaded data for the initial session.
    sessions:
        Optional list of ``(label, PlaybackData)`` tuples for the session
        selector.  If ``None``, defaults to ``[("Session 1", playback_data)]``.
    title:
        Window title override.
    parent:
        Optional Qt parent widget.
    """

    def __init__(
        self,
        playback_data: PlaybackData,
        sessions: Optional[List[Tuple[str, PlaybackData]]] = None,
        title: str = "Touch Playback Explorer",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)

        self._data = playback_data
        self._sessions: List[Tuple[str, PlaybackData]] = (
            sessions if sessions is not None else [("Session 1", playback_data)]
        )
        self._initialized = False
        self._current_touch: Optional[TouchEvent] = None
        self._current_frame: int = 0
        self._syncing: bool = False  # camera link re-entrancy guard
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance_frame)
        self._play_queue: list[TouchEvent] = []  # for Play All
        self._forearm_cloud_left: Optional[pv.PolyData] = None
        self._forearm_cloud_right: Optional[pv.PolyData] = None

        # Running spike + IFF accumulators — reset in _load_touch().
        n_verts = len(self._data.session_data.forearm_vertices)
        self._spike_sum = np.zeros(n_verts, dtype=np.float64)
        self._iff_sum = np.zeros(n_verts, dtype=np.float64)
        self._contact_count = np.zeros(n_verts, dtype=np.float64)

        # Heatmap mode: "spike" or "iff" (default).
        self._heatmap_mode: str = "iff"
        # Stable IFF upper clim for the entire session (set in _deferred_start / _on_session_changed).
        self._session_max_iff: float = 1.0

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
        root.setSpacing(0)

        slider_row = QHBoxLayout()
        slider_row.setContentsMargins(4, 0, 4, 0)
        slider_row.addWidget(QLabel("Frame:"))
        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setRange(0, 0)
        self._frame_slider.setValue(0)
        self._frame_slider.setEnabled(False)
        slider_row.addWidget(self._frame_slider)
        self._slider_value_label = QLabel("0 / 0")
        self._slider_value_label.setFixedWidth(80)
        slider_row.addWidget(self._slider_value_label)
        root.addLayout(slider_row)
        self._frame_slider.valueChanged.connect(self._on_slider_changed)

        splitter = QSplitter(Qt.Horizontal)

        self._plotter_left = QtInteractor(self)
        splitter.addWidget(self._plotter_left.interactor)

        self._plotter_right = QtInteractor(self)
        splitter.addWidget(self._plotter_right.interactor)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter, 1)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Session combo
        toolbar.addWidget(QLabel("Session:"))
        self._session_combo = QComboBox()
        self._session_combo.setMinimumWidth(160)
        for label, _ in self._sessions:
            self._session_combo.addItem(label)
        self._session_combo.currentIndexChanged.connect(self._on_session_changed)
        toolbar.addWidget(self._session_combo)

        toolbar.addSeparator()

        # Block combo
        toolbar.addWidget(QLabel("Block:"))
        self._block_combo = QComboBox()
        self._block_combo.setMinimumWidth(80)
        self._block_combo.currentIndexChanged.connect(self._on_block_changed)
        toolbar.addWidget(self._block_combo)

        toolbar.addSeparator()

        # Trial combo
        toolbar.addWidget(QLabel("Trial:"))
        self._trial_combo = QComboBox()
        self._trial_combo.setMinimumWidth(100)
        self._trial_combo.currentIndexChanged.connect(self._on_trial_changed)
        toolbar.addWidget(self._trial_combo)

        toolbar.addSeparator()

        # Touch combo
        toolbar.addWidget(QLabel("Touch:"))
        self._touch_combo = QComboBox()
        self._touch_combo.setMinimumWidth(320)
        self._touch_combo.currentIndexChanged.connect(self._on_touch_changed)
        toolbar.addWidget(self._touch_combo)

        toolbar.addSeparator()

        # Heatmap mode combo
        toolbar.addWidget(QLabel("Mode:"))
        self._heatmap_mode_combo = QComboBox()
        self._heatmap_mode_combo.addItem("Spike density")
        self._heatmap_mode_combo.addItem("IFF (Hz)")
        self._heatmap_mode_combo.setCurrentIndex(1)
        self._heatmap_mode_combo.currentIndexChanged.connect(self._on_heatmap_mode_changed)
        toolbar.addWidget(self._heatmap_mode_combo)

        toolbar.addSeparator()

        # Playback buttons
        self._play_btn = QPushButton("▶ Play")
        self._play_btn.clicked.connect(self._play)
        toolbar.addWidget(self._play_btn)

        self._play_all_btn = QPushButton("▶▶ Play All")
        self._play_all_btn.clicked.connect(self._play_all)
        toolbar.addWidget(self._play_all_btn)

        self._stop_btn = QPushButton("■ Stop")
        self._stop_btn.clicked.connect(self._stop)
        toolbar.addWidget(self._stop_btn)

        self._export_btn = QPushButton("Export Video")
        self._export_btn.clicked.connect(self._on_export_video)
        toolbar.addWidget(self._export_btn)

        toolbar.addSeparator()

        # Speed spinbox
        toolbar.addWidget(QLabel("Speed:"))
        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(0.1, 300.0)
        self._speed_spin.setValue(33.0)
        self._speed_spin.setSingleStep(1.0)
        self._speed_spin.setDecimals(1)
        self._speed_spin.valueChanged.connect(self._on_speed_changed)
        toolbar.addWidget(self._speed_spin)

        toolbar.addSeparator()

        # Frame label
        self._frame_label = QLabel("Frame: 0 / 0")
        self._frame_label.setFixedWidth(100)
        toolbar.addWidget(self._frame_label)

    # ------------------------------------------------------------------
    # Combo cascade helpers
    # ------------------------------------------------------------------

    def _populate_block_combo(self, data: PlaybackData) -> None:
        """Repopulate the block combo from *data* and cascade to trial combo."""
        self._block_combo.blockSignals(True)
        self._block_combo.clear()
        for bid in data.block_order_ids:
            self._block_combo.addItem(f"Block {bid}")
        self._block_combo.blockSignals(False)

        if data.block_order_ids:
            self._populate_trial_combo(data, data.block_order_ids[0])

    def _populate_trial_combo(self, data: PlaybackData, block_id: str) -> None:
        """Repopulate the trial combo for *block_id* and cascade to touch combo."""
        self._trial_combo.blockSignals(True)
        self._trial_combo.clear()
        for tid in data.trial_ids_by_block.get(block_id, []):
            self._trial_combo.addItem(f"Trial {tid}")
        self._trial_combo.blockSignals(False)

        trial_ids = data.trial_ids_by_block.get(block_id, [])
        if trial_ids:
            self._populate_touch_combo(data, block_id, trial_ids[0])

    def _populate_touch_combo(self, data: PlaybackData, block_id: str, trial_id: int) -> None:
        """Repopulate the touch combo for *(block_id, trial_id)* and select index 0."""
        self._touch_combo.blockSignals(True)
        self._touch_combo.clear()
        for te in data.touches_by_block_trial.get((block_id, trial_id), []):
            self._touch_combo.addItem(
                f"Touch {te.single_touch_id} ({te.gesture_type})"
            )
        self._touch_combo.blockSignals(False)

        if self._touch_combo.count() > 0:
            self._touch_combo.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Current-selection helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_slider_changed(self, value: int) -> None:
        if not self._initialized or self._current_touch is None:
            return
        self._stop()
        n_verts = len(self._data.session_data.forearm_vertices)
        self._spike_sum = np.zeros(n_verts, dtype=np.float64)
        self._iff_sum = np.zeros(n_verts, dtype=np.float64)
        self._contact_count = np.zeros(n_verts, dtype=np.float64)
        self._reset_heatmap()
        # Replay accumulation for preceding frames so the heatmap
        # is persistent, matching Play-mode behaviour.
        touch = self._current_touch
        for fi in range(value):
            verts = touch.frame_vertex_indices[fi]
            np.add.at(self._spike_sum, verts, float(touch.frame_spikes[fi]))
            np.add.at(self._iff_sum, verts, float(touch.frame_iff[fi]))
            np.add.at(self._contact_count, verts, 1.0)
        self._render_frame(value)

    def _on_session_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._sessions):
            raise ValueError(
                f"TouchPlaybackExplorer: session index {index} out of range "
                f"(have {len(self._sessions)} sessions)"
            )
        _, new_data = self._sessions[index]
        self._data = new_data
        self._session_max_iff = self._compute_session_max_iff()
        self._populate_block_combo(new_data)

        if self._initialized:
            self._render_forearm()
            if new_data.block_order_ids:
                first_block = new_data.block_order_ids[0]
                first_trial_ids = new_data.trial_ids_by_block.get(first_block, [])
                if first_trial_ids:
                    touches = new_data.touches_by_block_trial.get(
                        (first_block, first_trial_ids[0]), []
                    )
                    if touches:
                        self._load_touch(touches[0])
                        self._render_frame(0)

    def _on_block_changed(self, index: int) -> None:
        if not self._initialized or index < 0:
            return
        bid = self._current_block_id()
        if bid is None:
            return
        self._populate_trial_combo(self._data, bid)
        trial_ids = self._data.trial_ids_by_block.get(bid, [])
        if trial_ids:
            touches = self._data.touches_by_block_trial.get((bid, trial_ids[0]), [])
            if touches:
                self._load_touch(touches[0])
                self._render_frame(0)

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

        touches = self._data.touches_by_block_trial.get((bid, trial_id), [])
        if touches:
            self._load_touch(touches[0])
            self._render_frame(0)

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
        touch = touches[index]
        self._load_touch(touch)
        self._render_frame(0)

    # ------------------------------------------------------------------
    # Touch loading and frame rendering
    # ------------------------------------------------------------------

    def _load_touch(self, touch: TouchEvent) -> None:
        """Set the current touch and reset all per-touch state."""
        self._current_touch = touch
        self._current_frame = 0
        n_frames = len(touch.frame_spikes)
        self._frame_label.setText(f"Frame: 0 / {n_frames}")

        self._frame_slider.blockSignals(True)
        self._frame_slider.setMaximum(max(0, n_frames - 1))
        self._frame_slider.setValue(0)
        self._frame_slider.setEnabled(n_frames > 0)
        self._slider_value_label.setText(f"0 / {n_frames}")
        self._frame_slider.blockSignals(False)

        # Reset running spike + IFF accumulators.
        n_verts = len(self._data.session_data.forearm_vertices)
        self._spike_sum = np.zeros(n_verts, dtype=np.float64)
        self._iff_sum = np.zeros(n_verts, dtype=np.float64)
        self._contact_count = np.zeros(n_verts, dtype=np.float64)

    def _render_forearm(self) -> None:
        """Render the forearm point cloud on both plotters.

        Left plotter: PLY vertex colours when available, grey otherwise.
        Right plotter: scalar cloud for the active heatmap mode (spike density
        or IFF), NaN everywhere initially so the forearm appears grey until
        data arrive.
        """
        self._plotter_left.clear()
        self._plotter_left.set_background("black")
        self._plotter_right.clear()
        self._plotter_right.set_background("black")

        vertices = self._data.session_data.forearm_vertices
        vertex_colors = self._data.session_data.forearm_vertex_colors

        # Left — PLY vertex colours when available, grey otherwise.
        cloud_left = pv.PolyData(vertices)
        self._forearm_cloud_left = cloud_left
        use_vertex_colors = (
            vertex_colors is not None
            and vertex_colors.shape == (len(vertices), 3)
        )
        if use_vertex_colors:
            cloud_left["rgb"] = vertex_colors
            self._plotter_left.add_mesh(
                cloud_left,
                scalars="rgb",
                rgb=True,
                point_size=3,
                name="forearm",
            )
        else:
            self._plotter_left.add_mesh(
                cloud_left,
                color=[0.3, 0.3, 0.3],
                point_size=3,
                name="forearm",
            )

        # Right — scalar cloud for the active heatmap mode.
        cloud_right = pv.PolyData(vertices)
        heatmap_scalars = np.full(len(vertices), np.nan, dtype=np.float64)
        cloud_right["heatmap"] = heatmap_scalars
        self._forearm_cloud_right = cloud_right
        clim = self._current_heatmap_clim()
        self._plotter_right.add_mesh(
            cloud_right,
            scalars="heatmap",
            cmap="inferno",
            clim=clim,
            nan_color=[0.3, 0.3, 0.3],
            show_scalar_bar=True,
            scalar_bar_args={"title": "", "n_labels": 5, "color": "white", "fmt": "%.3g"},
            render_points_as_spheres=False,
            point_size=3,
            name="forearm",
            copy_mesh=False,
        )

        self._plotter_left.view_xy()
        self._plotter_right.view_xy()
        self._plotter_left.render()
        self._plotter_right.render()

    def _reset_heatmap(self) -> None:
        """Reset the right plotter's heatmap to all-NaN without touching cameras."""
        if self._forearm_cloud_right is None:
            return
        n_verts = len(self._data.session_data.forearm_vertices)
        self._forearm_cloud_right["heatmap"] = np.full(n_verts, np.nan, dtype=np.float64)
        self._forearm_cloud_right.Modified()
        self._plotter_right.render()

    def _current_heatmap_clim(self) -> Tuple[float, float]:
        """Return the colour-limit pair for the active heatmap mode."""
        if self._heatmap_mode == "iff":
            return (0.0, self._session_max_iff)
        return (0.0, 1.0)

    def _compute_session_max_iff(self) -> float:
        """Return the maximum IFF value across ALL touches in the active session."""
        max_val = 0.0
        for touches in self._data.touches_by_block_trial.values():
            for te in touches:
                if len(te.frame_iff) > 0:
                    max_val = max(max_val, float(np.max(te.frame_iff)))
        return max_val if max_val > 0.0 else 1.0

    def _update_heatmap_scalars(self) -> None:
        """Write the active heatmap mode's mean values into the right-panel mesh."""
        if self._forearm_cloud_right is None:
            return
        has_contact = self._contact_count > 0
        if self._heatmap_mode == "iff":
            scalars = np.where(
                has_contact,
                self._iff_sum / np.where(has_contact, self._contact_count, 1.0),
                np.nan,
            )
        else:
            scalars = np.where(
                has_contact,
                self._spike_sum / np.where(has_contact, self._contact_count, 1.0),
                np.nan,
            )
        self._forearm_cloud_right["heatmap"] = scalars
        self._forearm_cloud_right.Modified()

    def _on_heatmap_mode_changed(self, index: int) -> None:
        """Switch the active heatmap mode and rebuild the display."""
        self._heatmap_mode = "iff" if index == 1 else "spike"
        # Re-add the right-panel mesh with the correct clim for the new mode.
        if self._initialized and self._forearm_cloud_right is not None:
            clim = self._current_heatmap_clim()
            self._plotter_right.add_mesh(
                self._forearm_cloud_right,
                scalars="heatmap",
                cmap="inferno",
                clim=clim,
                nan_color=[0.3, 0.3, 0.3],
                show_scalar_bar=True,
                scalar_bar_args={"title": "", "n_labels": 5, "color": "white", "fmt": "%.3g"},
                render_points_as_spheres=False,
                point_size=3,
                name="forearm",
                copy_mesh=False,
            )
            self._recompute_heatmap_from_scratch()

    def _recompute_heatmap_from_scratch(self) -> None:
        """Reset accumulators and replay accumulation up to the current frame."""
        if self._current_touch is None:
            return
        n_verts = len(self._data.session_data.forearm_vertices)
        self._spike_sum = np.zeros(n_verts, dtype=np.float64)
        self._iff_sum = np.zeros(n_verts, dtype=np.float64)
        self._contact_count = np.zeros(n_verts, dtype=np.float64)

        # Vectorised replay: accumulate all frames from 0 to current_frame (inclusive).
        touch = self._current_touch
        for fi in range(self._current_frame + 1):
            verts = touch.frame_vertex_indices[fi]
            np.add.at(self._spike_sum, verts, float(touch.frame_spikes[fi]))
            np.add.at(self._iff_sum, verts, float(touch.frame_iff[fi]))
            np.add.at(self._contact_count, verts, 1.0)

        self._update_heatmap_scalars()
        self._plotter_right.render()

    def _render_frame(self, frame_idx: int) -> None:
        """Update both 3D views to display *frame_idx* of the current touch.

        The frame index is clamped to the valid range.  The left view shows
        current-frame contact points; the right view accumulates spike density
        incrementally (O(K) per frame).
        """
        if self._current_touch is None:
            return

        n_frames = len(self._current_touch.frame_spikes)
        frame_idx = max(0, min(frame_idx, n_frames - 1))
        self._current_frame = frame_idx
        self._frame_label.setText(f"Frame: {frame_idx + 1} / {n_frames}")

        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(frame_idx)
        self._slider_value_label.setText(f"{frame_idx + 1} / {n_frames}")
        self._frame_slider.blockSignals(False)

        # ---- Left view: contact points ----
        pts = self._current_touch.frame_contact_pts[frame_idx]
        if len(pts) > 0:
            contact_cloud = pv.PolyData(np.asarray(pts, dtype=np.float64))
            self._plotter_left.add_mesh(
                contact_cloud,
                color="red",
                point_size=8,
                render_points_as_spheres=True,
                name="contacts",
            )
        else:
            # No valid contacts this frame — replace with an invisible dummy
            # so the named actor is cleared from the previous frame.
            empty = pv.PolyData(np.zeros((1, 3), dtype=np.float64))
            self._plotter_left.add_mesh(
                empty,
                color="red",
                point_size=1,
                name="contacts",
            )

        # ---- Right view: accumulate spike + IFF heatmap ----
        verts_this_frame = self._current_touch.frame_vertex_indices[frame_idx]
        spike_this_frame = float(self._current_touch.frame_spikes[frame_idx])
        iff_this_frame = float(self._current_touch.frame_iff[frame_idx])
        np.add.at(self._spike_sum, verts_this_frame, spike_this_frame)
        np.add.at(self._iff_sum, verts_this_frame, iff_this_frame)
        np.add.at(self._contact_count, verts_this_frame, 1.0)

        self._update_heatmap_scalars()

        self._plotter_left.render()
        self._plotter_right.render()

    # ------------------------------------------------------------------
    # Camera linking
    # ------------------------------------------------------------------

    def _setup_camera_link(self) -> None:
        """Link the cameras of both plotters via VTK ModifiedEvent observers.

        A ``_syncing`` flag prevents infinite re-entrancy when one camera
        change triggers the other (same pattern as the Qt blockSignals guard).
        """
        cam_left = self._plotter_left.camera
        cam_right = self._plotter_right.camera

        def _sync_left_to_right(obj, event):  # noqa: ANN001
            if self._syncing:
                return
            self._syncing = True
            cam_right.SetPosition(cam_left.GetPosition())
            cam_right.SetFocalPoint(cam_left.GetFocalPoint())
            cam_right.SetViewUp(cam_left.GetViewUp())
            self._plotter_right.render()
            self._syncing = False

        def _sync_right_to_left(obj, event):  # noqa: ANN001
            if self._syncing:
                return
            self._syncing = True
            cam_left.SetPosition(cam_right.GetPosition())
            cam_left.SetFocalPoint(cam_right.GetFocalPoint())
            cam_left.SetViewUp(cam_right.GetViewUp())
            self._plotter_left.render()
            self._syncing = False

        cam_left.AddObserver("ModifiedEvent", _sync_left_to_right)
        cam_right.AddObserver("ModifiedEvent", _sync_right_to_left)

    # ------------------------------------------------------------------
    # Deferred VTK initialisation
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
        """Finish VTK init after the window is visible and sized.

        PyVista / VTK render windows must be initialised after the Qt widget
        has been painted at least once, so this runs via a zero-delay timer
        from ``showEvent``.
        """
        for plotter in (self._plotter_left, self._plotter_right):
            try:
                plotter.interactor.Initialize()
            except Exception:
                pass
            sz = plotter.interactor.size()
            if sz.width() > 0 and sz.height() > 0:
                plotter.render_window.SetSize(sz.width(), sz.height())

        self._populate_block_combo(self._data)
        self._session_max_iff = self._compute_session_max_iff()
        self._render_forearm()

        if self._data.block_order_ids:
            first_block = self._data.block_order_ids[0]
            first_trial_ids = self._data.trial_ids_by_block.get(first_block, [])
            if first_trial_ids:
                touches = self._data.touches_by_block_trial.get(
                    (first_block, first_trial_ids[0]), []
                )
                if touches:
                    self._load_touch(touches[0])
                    self._render_frame(0)

        self._setup_camera_link()

    # ------------------------------------------------------------------
    # Window lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        self._plotter_left.close()
        self._plotter_right.close()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------

    def _on_speed_changed(self, _value: float) -> None:
        if self._timer.isActive():
            interval_ms = max(1, int(33 / self._speed_spin.value()))
            self._timer.setInterval(interval_ms)

    def _play(self) -> None:
        """Start playback of the current touch from frame 0."""
        self._timer.stop()
        self._play_queue.clear()
        if self._current_touch is None:
            return
        # Restart from frame 0: reset accumulators and heatmap (no camera reset)
        self._load_touch(self._current_touch)
        self._reset_heatmap()
        interval_ms = max(1, int(33 / self._speed_spin.value()))
        self._timer.start(interval_ms)

    def _play_all(self) -> None:
        """Queue all touches in the selected trial and play sequentially."""
        self._timer.stop()
        self._play_queue.clear()
        bid = self._current_block_id()
        tid = self._current_trial_id()
        if bid is None or tid is None:
            return
        touches = self._data.touches_by_block_trial.get((bid, tid), [])
        if not touches:
            return
        # Load first touch; queue the rest
        self._load_touch(touches[0])
        self._reset_heatmap()
        self._play_queue = list(touches[1:])
        interval_ms = max(1, int(33 / self._speed_spin.value()))
        self._timer.start(interval_ms)

    def _stop(self) -> None:
        """Stop playback and clear the play queue."""
        self._timer.stop()
        self._play_queue.clear()

    def _advance_frame(self) -> None:
        """Advance one frame; called by QTimer on each tick."""
        if self._current_touch is None:
            self._timer.stop()
            return
        n_frames = len(self._current_touch.frame_spikes)
        if self._current_frame >= n_frames:
            # Current touch finished — load next from queue or stop
            if self._play_queue:
                next_touch = self._play_queue.pop(0)
                self._load_touch(next_touch)
                self._reset_heatmap()
                # Update combos to reflect the new touch (for Play All)
                self._sync_combos_to_touch(next_touch)
            else:
                self._timer.stop()
            return
        self._render_frame(self._current_frame)
        # _render_frame sets self._current_frame = frame_idx; advance to next
        self._current_frame += 1

    def _sync_combos_to_touch(self, touch: TouchEvent) -> None:
        """Update block/trial/touch combos to reflect *touch* without triggering playback."""
        if touch.block_order_id not in self._data.block_order_ids:
            return
        block_idx = self._data.block_order_ids.index(touch.block_order_id)
        self._block_combo.blockSignals(True)
        self._block_combo.setCurrentIndex(block_idx)
        self._block_combo.blockSignals(False)

        trial_ids = self._data.trial_ids_by_block.get(touch.block_order_id, [])
        if touch.trial_id not in trial_ids:
            return
        trial_idx = trial_ids.index(touch.trial_id)
        self._trial_combo.blockSignals(True)
        self._trial_combo.setCurrentIndex(trial_idx)
        self._trial_combo.blockSignals(False)

        touches = self._data.touches_by_block_trial.get(
            (touch.block_order_id, touch.trial_id), []
        )
        touch_idx = next(
            (i for i, t in enumerate(touches) if t.single_touch_id == touch.single_touch_id),
            -1,
        )
        if touch_idx >= 0:
            self._touch_combo.blockSignals(True)
            self._touch_combo.setCurrentIndex(touch_idx)
            self._touch_combo.blockSignals(False)

    # ------------------------------------------------------------------
    # Video export
    # ------------------------------------------------------------------

    def _on_export_video(self) -> None:
        """Handle the Export Video button: check deps, open file dialog, render."""
        try:
            import imageio_ffmpeg  # noqa: F401
        except ImportError:
            QMessageBox.critical(
                self,
                "Missing Dependency",
                "The 'imageio-ffmpeg' library is required to export videos.\n\n"
                "Install it via:\n"
                "  pip install imageio-ffmpeg\n"
                "OR\n"
                "  conda install -c conda-forge imageio-ffmpeg",
            )
            return

        bid = self._current_block_id()
        tid = self._current_trial_id()
        if bid is None or tid is None:
            QMessageBox.warning(self, "No trial selected", "Select a block and trial first.")
            return
        touches = self._data.touches_by_block_trial.get((bid, tid), [])
        if not touches:
            QMessageBox.warning(self, "No touches", "The selected trial has no touches.")
            return

        session_label = self._session_combo.currentText().replace(" ", "_")
        proposed = f"{session_label}_block{bid}_trial{tid}_left.mp4"

        filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Video (left view)", proposed, "MP4 Files (*.mp4)",
        )
        if not filename:
            return
        if not filename.lower().endswith(".mp4"):
            filename += ".mp4"

        left_path = filename
        if left_path.endswith("_left.mp4"):
            right_path = left_path[:-len("_left.mp4")] + "_right.mp4"
        else:
            base = left_path[:-len(".mp4")]
            right_path = base + "_right.mp4"

        self._generate_videos(left_path, right_path, touches)

    def _generate_videos(
        self,
        left_path: str,
        right_path: str,
        touches: List[TouchEvent],
    ) -> None:
        """Render all *touches* to two off-screen 1920x1080 MP4 files at 120 FPS."""
        total_frames = sum(len(t.frame_spikes) for t in touches)
        if total_frames == 0:
            return

        vertices = self._data.session_data.forearm_vertices
        vertex_colors = self._data.session_data.forearm_vertex_colors
        n_verts = len(vertices)
        clim = self._current_heatmap_clim()
        use_vertex_colors = (
            vertex_colors is not None
            and vertex_colors.shape == (n_verts, 3)
        )

        progress = QProgressDialog("Rendering video...", "Cancel", 0, total_frames, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        pl_left = pv.Plotter(window_size=[1920, 1080], off_screen=True)
        pl_right = pv.Plotter(window_size=[1920, 1080], off_screen=True)
        pl_left.set_background("black")
        pl_right.set_background("black")

        for src, dst in [
            (self._plotter_left, pl_left),
            (self._plotter_right, pl_right),
        ]:
            dst.camera.position = src.camera.position
            dst.camera.focal_point = src.camera.focal_point
            dst.camera.up = src.camera.up
            dst.camera.view_angle = src.camera.view_angle
            dst.camera.clipping_range = src.camera.clipping_range

        try:
            pl_left.open_movie(left_path, framerate=120)
            pl_right.open_movie(right_path, framerate=120)
        except Exception as exc:
            QMessageBox.critical(
                self, "Export Error",
                f"Failed to initialise video writer:\n{exc}",
            )
            pl_left.close()
            pl_right.close()
            progress.close()
            return

        # Static left forearm mesh.
        cloud_left = pv.PolyData(vertices)
        if use_vertex_colors:
            cloud_left["rgb"] = vertex_colors
            pl_left.add_mesh(cloud_left, scalars="rgb", rgb=True, point_size=3, name="forearm")
        else:
            pl_left.add_mesh(cloud_left, color=[0.3, 0.3, 0.3], point_size=3, name="forearm")

        # Right heatmap mesh.
        cloud_right = pv.PolyData(vertices)
        heatmap = np.full(n_verts, np.nan, dtype=np.float64)
        cloud_right["heatmap"] = heatmap
        pl_right.add_mesh(
            cloud_right,
            scalars="heatmap",
            cmap="inferno",
            clim=clim,
            nan_color=[0.3, 0.3, 0.3],
            show_scalar_bar=True,
            scalar_bar_args={"title": "", "n_labels": 5, "color": "white", "fmt": "%.3g"},
            render_points_as_spheres=False,
            point_size=3,
            name="forearm",
            copy_mesh=False,
        )

        spike_sum = np.zeros(n_verts, dtype=np.float64)
        iff_sum = np.zeros(n_verts, dtype=np.float64)
        contact_count = np.zeros(n_verts, dtype=np.float64)
        frame_counter = 0
        cancelled = False

        try:
            for touch in touches:
                spike_sum[:] = 0.0
                iff_sum[:] = 0.0
                contact_count[:] = 0.0
                cloud_right["heatmap"] = np.full(n_verts, np.nan, dtype=np.float64)

                n_frames = len(touch.frame_spikes)
                for fi in range(n_frames):
                    if progress.wasCanceled():
                        cancelled = True
                        break

                    # Left: contact points.
                    pts = touch.frame_contact_pts[fi]
                    if len(pts) > 0:
                        contact_cloud = pv.PolyData(np.asarray(pts, dtype=np.float64))
                        pl_left.add_mesh(
                            contact_cloud, color="red", point_size=8,
                            render_points_as_spheres=True, name="contacts",
                        )
                    else:
                        empty = pv.PolyData(np.zeros((1, 3), dtype=np.float64))
                        pl_left.add_mesh(empty, color="red", point_size=1, name="contacts")

                    # Right: accumulate heatmap.
                    verts_fi = touch.frame_vertex_indices[fi]
                    np.add.at(spike_sum, verts_fi, float(touch.frame_spikes[fi]))
                    np.add.at(iff_sum, verts_fi, float(touch.frame_iff[fi]))
                    np.add.at(contact_count, verts_fi, 1.0)

                    has_contact = contact_count > 0
                    if self._heatmap_mode == "iff":
                        scalars = np.where(
                            has_contact,
                            iff_sum / np.where(has_contact, contact_count, 1.0),
                            np.nan,
                        )
                    else:
                        scalars = np.where(
                            has_contact,
                            spike_sum / np.where(has_contact, contact_count, 1.0),
                            np.nan,
                        )
                    cloud_right["heatmap"] = scalars
                    cloud_right.Modified()

                    pl_left.write_frame()
                    pl_right.write_frame()

                    frame_counter += 1
                    progress.setValue(frame_counter)
                    QApplication.processEvents()

                if cancelled:
                    break

        except Exception as exc:
            QMessageBox.critical(
                self, "Rendering Error",
                f"An error occurred during rendering:\n{exc}",
            )
        finally:
            pl_left.close()
            pl_right.close()
            progress.close()

        if not cancelled and os.path.exists(left_path) and os.path.exists(right_path):
            QMessageBox.information(
                self, "Export complete",
                f"Videos saved:\n  {left_path}\n  {right_path}",
            )
