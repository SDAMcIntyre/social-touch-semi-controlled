"""Touch Playback Explorer GUI — dual 3D view for per-touch animation.

Presents two side-by-side PyVista 3D forearm views:
  - Left:  current-frame contact points as red spheres on a grey point cloud
  - Right: spike heatmap accumulating from touch start (jet colormap, NaN=grey)

A toolbar provides session / trial / touch dropdowns, playback controls
(Play, Play All, Stop), a speed spinbox, and a frame counter.
"""

import logging
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
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QToolBar,
    QWidget,
)
from pyvistaqt import QtInteractor

from analysis.receptive_field_mapping.touch_playback_data import PlaybackData, TouchEvent

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

        # Running spike accumulators — reset in _load_touch().
        n_verts = len(self._data.session_data.forearm_vertices)
        self._spike_sum = np.zeros(n_verts, dtype=np.float64)
        self._contact_count = np.zeros(n_verts, dtype=np.float64)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_toolbar()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)

        self._plotter_left = QtInteractor(self)
        splitter.addWidget(self._plotter_left.interactor)

        self._plotter_right = QtInteractor(self)
        splitter.addWidget(self._plotter_right.interactor)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        root.addWidget(splitter)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Controls")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Session combo
        toolbar.addWidget(QLabel("Session:"))
        self._session_combo = QComboBox()
        for label, _ in self._sessions:
            self._session_combo.addItem(label)
        self._session_combo.currentIndexChanged.connect(self._on_session_changed)
        toolbar.addWidget(self._session_combo)

        toolbar.addSeparator()

        # Trial combo
        toolbar.addWidget(QLabel("Trial:"))
        self._trial_combo = QComboBox()
        self._trial_combo.currentIndexChanged.connect(self._on_trial_changed)
        toolbar.addWidget(self._trial_combo)

        toolbar.addSeparator()

        # Touch combo
        toolbar.addWidget(QLabel("Touch:"))
        self._touch_combo = QComboBox()
        self._touch_combo.currentIndexChanged.connect(self._on_touch_changed)
        toolbar.addWidget(self._touch_combo)

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

        toolbar.addSeparator()

        # Speed spinbox
        toolbar.addWidget(QLabel("Speed:"))
        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(0.1, 4.0)
        self._speed_spin.setValue(1.0)
        self._speed_spin.setSingleStep(0.25)
        self._speed_spin.setDecimals(2)
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

    def _populate_trial_combo(self, data: PlaybackData) -> None:
        """Repopulate the trial combo from *data* and cascade to touch combo."""
        self._trial_combo.blockSignals(True)
        self._trial_combo.clear()
        for tid in data.trial_ids:
            self._trial_combo.addItem(f"Trial {tid}")
        self._trial_combo.blockSignals(False)

        if data.trial_ids:
            self._populate_touch_combo(data, data.trial_ids[0])

    def _populate_touch_combo(self, data: PlaybackData, trial_id: int) -> None:
        """Repopulate the touch combo for *trial_id* and select index 0."""
        self._touch_combo.blockSignals(True)
        self._touch_combo.clear()
        for te in data.touches_by_trial.get(trial_id, []):
            self._touch_combo.addItem(
                f"Touch {te.single_touch_id} ({te.gesture_type})"
            )
        self._touch_combo.blockSignals(False)

        if self._touch_combo.count() > 0:
            self._touch_combo.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_session_changed(self, index: int) -> None:
        if index < 0 or index >= len(self._sessions):
            raise ValueError(
                f"TouchPlaybackExplorer: session index {index} out of range "
                f"(have {len(self._sessions)} sessions)"
            )
        _, new_data = self._sessions[index]
        self._data = new_data
        self._populate_trial_combo(new_data)

        if self._initialized:
            self._render_forearm()
            # Load first touch of first trial (if available).
            if new_data.trial_ids:
                first_trial = new_data.trial_ids[0]
                touches = new_data.touches_by_trial.get(first_trial, [])
                if touches:
                    self._load_touch(touches[0])
                    self._render_frame(0)

    def _on_trial_changed(self, index: int) -> None:
        if not self._initialized or index < 0:
            return
        trial_id = self._data.trial_ids[index]
        self._populate_touch_combo(self._data, trial_id)

        touches = self._data.touches_by_trial.get(trial_id, [])
        if touches:
            self._load_touch(touches[0])
            self._render_frame(0)

    def _on_touch_changed(self, index: int) -> None:
        if not self._initialized or index < 0:
            return
        trial_idx = self._trial_combo.currentIndex()
        if trial_idx < 0 or trial_idx >= len(self._data.trial_ids):
            return
        trial_id = self._data.trial_ids[trial_idx]
        touches = self._data.touches_by_trial.get(trial_id, [])
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

        # Reset running spike accumulators.
        n_verts = len(self._data.session_data.forearm_vertices)
        self._spike_sum = np.zeros(n_verts, dtype=np.float64)
        self._contact_count = np.zeros(n_verts, dtype=np.float64)

    def _render_forearm(self) -> None:
        """Render the grey forearm point cloud on both plotters.

        Left plotter shows a plain grey cloud.
        Right plotter shows the cloud with an initialised ``spike_density``
        scalar (NaN everywhere so the forearm appears grey until data arrive).
        """
        self._plotter_left.clear()
        self._plotter_left.set_background("black")
        self._plotter_right.clear()
        self._plotter_right.set_background("black")

        vertices = self._data.session_data.forearm_vertices

        # Left — plain grey cloud.
        cloud_left = pv.PolyData(vertices)
        self._forearm_cloud_left = cloud_left
        self._plotter_left.add_mesh(
            cloud_left,
            color=[0.3, 0.3, 0.3],
            point_size=3,
            name="forearm",
        )

        # Right — scalar cloud for spike heatmap.
        cloud_right = pv.PolyData(vertices)
        spike_density = np.full(len(vertices), np.nan, dtype=np.float64)
        cloud_right["spike_density"] = spike_density
        self._forearm_cloud_right = cloud_right
        self._plotter_right.add_mesh(
            cloud_right,
            scalars="spike_density",
            cmap="jet",
            clim=(0, 1),
            nan_color=[0.3, 0.3, 0.3],
            show_scalar_bar=True,
            render_points_as_spheres=False,
            point_size=3,
            name="forearm",
            copy_mesh=False,
        )

        self._plotter_left.view_xy()
        self._plotter_right.view_xy()
        self._plotter_left.render()
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

        # ---- Right view: accumulate spike heatmap ----
        verts_this_frame = self._current_touch.frame_vertex_indices[frame_idx]
        spike_this_frame = float(self._current_touch.frame_spikes[frame_idx])
        np.add.at(self._spike_sum, verts_this_frame, spike_this_frame)
        np.add.at(self._contact_count, verts_this_frame, 1.0)

        spike_density = np.where(
            self._contact_count > 0,
            self._spike_sum / self._contact_count,
            np.nan,
        )

        if self._forearm_cloud_right is not None:
            self._forearm_cloud_right["spike_density"] = spike_density
            self._forearm_cloud_right.Modified()

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

        self._populate_trial_combo(self._data)
        self._render_forearm()

        # Load first touch of the first trial if available.
        if self._data.trial_ids:
            first_trial = self._data.trial_ids[0]
            touches = self._data.touches_by_trial.get(first_trial, [])
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
        # Restart from frame 0: reset accumulators and re-render forearm
        self._load_touch(self._current_touch)
        self._render_forearm()
        interval_ms = max(1, int(33 / self._speed_spin.value()))
        self._timer.start(interval_ms)

    def _play_all(self) -> None:
        """Queue all touches in the selected trial and play sequentially."""
        self._timer.stop()
        self._play_queue.clear()
        trial_idx = self._trial_combo.currentIndex()
        if trial_idx < 0 or trial_idx >= len(self._data.trial_ids):
            return
        trial_id = self._data.trial_ids[trial_idx]
        touches = self._data.touches_by_trial.get(trial_id, [])
        if not touches:
            return
        # Load first touch; queue the rest
        self._load_touch(touches[0])
        self._render_forearm()
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
                self._render_forearm()
                # Update combos to reflect the new touch (for Play All)
                self._sync_combos_to_touch(next_touch)
            else:
                self._timer.stop()
            return
        self._render_frame(self._current_frame)
        # _render_frame sets self._current_frame = frame_idx; advance to next
        self._current_frame += 1

    def _sync_combos_to_touch(self, touch: TouchEvent) -> None:
        """Update trial/touch combos to reflect *touch* without triggering playback."""
        # Find trial index
        if touch.trial_id not in self._data.trial_ids:
            return
        trial_idx = self._data.trial_ids.index(touch.trial_id)
        self._trial_combo.blockSignals(True)
        self._trial_combo.setCurrentIndex(trial_idx)
        self._trial_combo.blockSignals(False)
        # Find touch index within trial
        touches = self._data.touches_by_trial.get(touch.trial_id, [])
        touch_idx = next(
            (i for i, t in enumerate(touches) if t.single_touch_id == touch.single_touch_id),
            -1,
        )
        if touch_idx >= 0:
            self._touch_combo.blockSignals(True)
            self._touch_combo.setCurrentIndex(touch_idx)
            self._touch_combo.blockSignals(False)
