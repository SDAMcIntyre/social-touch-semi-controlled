from pathlib import Path
from typing import Optional

import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor


def _load_vertex_data(npz_path: Path) -> dict:
    if not npz_path.exists():
        raise ValueError(f"_load_vertex_data: NPZ file not found: {npz_path}")

    npz = np.load(npz_path, allow_pickle=True)

    if "session_id" not in npz:
        raise ValueError(f"_load_vertex_data: missing key 'session_id' in {npz_path}")
    if "gesture_types" not in npz:
        raise ValueError(f"_load_vertex_data: missing key 'gesture_types' in {npz_path}")

    session_id = str(npz["session_id"].item())
    gesture_types = npz["gesture_types"].tolist()

    grids: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for gtype in gesture_types:
        u_key = f"grid_u_{gtype}"
        v_key = f"grid_v_{gtype}"
        z_key = f"grid_z_{gtype}"
        for key in (u_key, v_key, z_key):
            if key not in npz:
                raise ValueError(
                    f"_load_vertex_data: missing key '{key}' for gesture type '{gtype}' in {npz_path}"
                )
        grids[gtype] = (npz[u_key], npz[v_key], npz[z_key])

    return {
        "session_id": session_id,
        "gesture_types": gesture_types,
        "grids": grids,
    }


class RFSurfaceViewer(QMainWindow):
    def __init__(
        self,
        sessions: list[tuple[str, Path]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("RF Surface Viewer")

        self._sessions: list[tuple[str, dict]] = [
            (label, _load_vertex_data(npz_path)) for label, npz_path in sessions
        ]
        self._current_data: dict = self._sessions[0][1]
        self._initialized = False
        self._build_ui()

    def _build_ui(self) -> None:
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

        toolbar.addWidget(QLabel("Gesture Type:"))
        self._gesture_combo = QComboBox()
        self._gesture_combo.currentIndexChanged.connect(self._on_gesture_changed)
        toolbar.addWidget(self._gesture_combo)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._plotter = QtInteractor(self)
        layout.addWidget(self._plotter.interactor)

        self._repopulate_gesture_combo()

    def _build_surface(
        self,
        grid_u: np.ndarray,
        grid_v: np.ndarray,
        grid_z: np.ndarray,
    ) -> None:
        self._plotter.clear()
        self._plotter.set_background("black")

        grid = pv.StructuredGrid(grid_u, grid_v, grid_z)
        grid["mean_iff"] = grid_z.ravel(order="F")

        surface = grid.threshold(scalars="mean_iff")

        self._plotter.add_mesh(
            surface,
            scalars="mean_iff",
            cmap="jet",
            show_scalar_bar=True,
            scalar_bar_args={"title": "Mean IFF (Hz)", "color": "white"},
        )
        self._plotter.show_bounds(
            xlabel="U (mm)",
            ylabel="V (mm)",
            zlabel="Mean IFF (Hz)",
            color="white",
        )
        self._plotter.view_isometric()
        self._plotter.render()

    def _repopulate_gesture_combo(self) -> None:
        self._gesture_combo.blockSignals(True)
        self._gesture_combo.clear()
        for gtype in self._current_data["gesture_types"]:
            self._gesture_combo.addItem(gtype)
        self._gesture_combo.blockSignals(False)
        self._gesture_combo.setCurrentIndex(0)

    def _on_session_changed(self, index: int) -> None:
        if index < 0:
            return
        self._current_data = self._sessions[index][1]
        self._repopulate_gesture_combo()
        self._render_current()

    def _on_gesture_changed(self, index: int) -> None:
        if index < 0 or not self._initialized:
            return
        gtype = self._current_data["gesture_types"][index]
        grid_u, grid_v, grid_z = self._current_data["grids"][gtype]
        self._build_surface(grid_u, grid_v, grid_z)

    def _render_current(self) -> None:
        session_index = self._session_combo.currentIndex()
        gesture_index = self._gesture_combo.currentIndex()
        if session_index < 0 or gesture_index < 0:
            return
        gtype = self._current_data["gesture_types"][gesture_index]
        grid_u, grid_v, grid_z = self._current_data["grids"][gtype]
        self._build_surface(grid_u, grid_v, grid_z)

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
        self._render_current()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._plotter.close()
        super().closeEvent(event)
